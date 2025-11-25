"""GPU-accelerated Eikonal propagation using fim-python (Fast Iterative Method on GPU).

This module introduces the class `EikonalFIMGPU`, which offers an alternative to
`EikonalDjikstraTet` (CPU Dijkstra-like implementation) by delegating the solution of the
anisotropic Eikonal equation to the external library `fim-python` (package name `fimpy`).

Design goals (initial version):
1. Keep the public interface similar to existing propagation modules.
2. Optional dependency: it only activates if `fimpy` and a CUDA-capable GPU (with CuPy) are available.
3. Provide clear fallbacks / informative errors if prerequisites are missing.
4. Separate geometry-to-FIM conversion logic so we can later cache per-element fibre bases and metrics.

Limitations (initial MVP):
- Endocardial dense vs sparse heterogeneity is NOT yet mapped onto per-element metrics; we currently
  use a uniform anisotropy (fibre, sheet, normal speeds) for all tetrahedra. A TODO is left where
  that mapping should occur. This lets us benchmark raw speed improvements first.
- Scar / fibrosis / Purkinje intra-network delays beyond root node initial activation times are not
  explicitly modeled (mirrors current handling of root node seeds).
- Batch simulation (population) is done naively in a Python loop; future optimisation could reuse
  geometry + precomputed bases and rebuild only diagonal speed scalings.

License note:
Using fim-python (AGPLv3) as a runtime dependency is compatible with keeping this repository MIT,
as long as we do not copy substantial portions of its source code here. We only import and call it.
If in the future we inline / modify AGPL-covered code, the combined work would need to respect AGPL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from propagation_models import ElectricalPropagation  # Reuse base patterns

try:  # Optional dependency handling
    from fimpy.solver import create_fim_solver  # type: ignore
    import cupy as cp  # noqa: F401  (we only test availability explicitly later)
    _FIM_AVAILABLE = True
except Exception:  # Broad except to catch both ImportError and CUDA runtime issues cleanly
    _FIM_AVAILABLE = False


# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def _gram_schmidt_columns_3x3(M: np.ndarray) -> np.ndarray:
    """Orthonormalise a (3,3) matrix whose columns are intended to be (fibre, sheet, normal).

    We apply a simple Gram-Schmidt to ensure numerical robustness when averaging node-based
    directions into an element representative.
    """
    Q = np.zeros_like(M)
    eps = 1e-12
    v0 = M[:, 0]
    n0 = np.linalg.norm(v0)
    Q[:, 0] = v0 / (n0 + eps)
    v1 = M[:, 1] - np.dot(Q[:, 0], M[:, 1]) * Q[:, 0]
    n1 = np.linalg.norm(v1)
    if n1 < eps:  # fallback: pick a perpendicular vector
        v1 = np.array([Q[1, 0], -Q[0, 0], 0.0])
        n1 = np.linalg.norm(v1) + eps
    Q[:, 1] = v1 / n1
    v2 = M[:, 2] - np.dot(Q[:, 0], M[:, 2]) * Q[:, 0] - np.dot(Q[:, 1], M[:, 2]) * Q[:, 1]
    n2 = np.linalg.norm(v2)
    if n2 < eps:  # fallback: cross product
        v2 = np.cross(Q[:, 0], Q[:, 1])
        n2 = np.linalg.norm(v2) + eps
    Q[:, 2] = v2 / n2
    return Q


def _build_element_basis(geometry) -> np.ndarray:
    """Construct per-element orthonormal bases (fibre, sheet, normal) from node data.

    Strategy:
    - Retrieve per-node normalised fibre, sheet, normal via geometry helper methods if available.
    - Average those vectors over the tetra's 4 nodes.
    - Orthonormalise the resulting 3x3 matrix to mitigate drift.

    Returns
    -------
    bases : (num_elems, 3, 3) array, columns = (fibre, sheet, normal)
    """
    # Attempt to use provided API (names based on geometry_functions.py patterns)
    node_f = geometry.get_node_normalised_fibre_fibre()  # (N,3)
    node_s = geometry.get_node_normalised_fibre_sheet()
    node_n = geometry.get_node_normalised_fibre_normal()
    tets = geometry.get_tetra()  # (T,4)
    bases = np.zeros((tets.shape[0], 3, 3), dtype=np.float32)
    for i, tet in enumerate(tets):
        f_avg = node_f[tet, :].mean(axis=0)
        s_avg = node_s[tet, :].mean(axis=0)
        n_avg = node_n[tet, :].mean(axis=0)
        M = np.vstack([f_avg, s_avg, n_avg]).T  # shape (3,3) columns=approx directions
        bases[i] = _gram_schmidt_columns_3x3(M).astype(np.float32)
    return bases


def _speeds_to_metrics(bases: np.ndarray, v_f: float, v_s: float, v_n: float, inverse: bool = False) -> np.ndarray:
    """Convert anisotropic conduction speeds into per-element metric tensors.

    Eikonal PDE |∇T|_D = 1 is commonly written with a diffusion-like tensor D or metric M.
    Here we follow the FIM convention (fim-python) expecting a per-element symmetric matrix A
    s.t. sqrt( (x^T A x) ) encodes weighted lengths. If speeds = (v_f, v_s, v_n) along an
    orthonormal basis R, then the metric (inverse speed squared) is:

        M = R diag(1/v_f^2, 1/v_s^2, 1/v_n^2) R^T.

    Parameters
    ----------
    bases : (T,3,3) columns are orthonormal directions (f,s,n)
    v_f, v_s, v_n : floats > 0

    Returns
    -------
    metrics : (T,3,3) float32
    """
    if inverse:
        scal = np.array([1.0/(v_f**2), 1.0/(v_s**2), 1.0/(v_n**2)], dtype=np.float32)
    else:
        # Convenção B (teste): direto quadrado da velocidade
        scal = np.array([v_f**2, v_s**2, v_n**2], dtype=np.float32)
    metrics = np.einsum('tij,j,tkj->tik', bases, scal, bases).astype(np.float32)
    return metrics

# -------------------------------------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------------------------------------

@dataclass
class FIMGPUConfig:
    fibre_speed_name: str
    sheet_speed_name: str
    normal_speed_name: str
    purkinje_speed_name: str
    endo_dense_speed_name: str
    endo_sparse_speed_name: str
    nb_speed_parameters: int
    parameter_name_list_in_order: List[str]
    # If True, build metric as inverse(1/v^2); if False, use direct(v^2).
    metric_inverse: bool = False


class EikonalFIMGPU(ElectricalPropagation):
    """GPU Fast Iterative Method (fim-python) based propagation.

    Usage mirrors `EikonalDjikstraTet`, but compute core is offloaded to GPU if available.

    Parameters
    ----------
    geometry : object
        Must implement:
          - get_node_xyz(), get_tetra()
          - get_node_normalised_fibre_fibre/sheet/normal()
          - get_candidate_root_node_index(), get_candidate_root_node_time(purkinje_speed)
    config : FIMGPUConfig
        Names for parameter unpacking (keeps consistency with existing inference pipeline).
    module_name : str
        Key under which this module's parameter vector is stored.
    verbose : bool
        Print diagnostics.
    """

    def __init__(self, geometry, config: FIMGPUConfig, module_name: str, verbose: bool):
        super().__init__(geometry=geometry, module_name=module_name, verbose=verbose)
        self.config = config
        if verbose:
            if _FIM_AVAILABLE:
                print('[EikonalFIMGPU] fim-python + CuPy detected.')
            else:
                print('[EikonalFIMGPU] WARNING: fim-python GPU backend not available. Class will raise on use.')
        # Lazy initialisation caches
        self._element_bases: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self._tets: Optional[np.ndarray] = None
        # Endocardial classification per element: 0=normal, 1=dense endo, 2=sparse endo
        self._element_endo_type: Optional[np.ndarray] = None

    # ------------------------------------------------------------------------------------------
    # Parameter handling (mirrors CPU version structure)
    # ------------------------------------------------------------------------------------------
    def _repack(self, parameter_particle: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        cfg = self.config
        if len(parameter_particle) != len(cfg.parameter_name_list_in_order):
            # Graceful warning only
            if self.verbose:
                print('[EikonalFIMGPU] Parameter length mismatch: expected',
                      len(cfg.parameter_name_list_in_order), 'got', len(parameter_particle))
        speed_params = parameter_particle[:cfg.nb_speed_parameters]
        root_meta = parameter_particle[cfg.nb_speed_parameters:]
        p_dict = {}
        for i, name in enumerate(cfg.parameter_name_list_in_order[:len(speed_params)]):
            p_dict[name] = float(speed_params[i])
        return p_dict, root_meta

    # ------------------------------------------------------------------------------------------
    # Geometry conversion helpers
    # ------------------------------------------------------------------------------------------
    def _ensure_geom_cache(self):
        if self._points is None:
            pts = self.geometry.get_node_xyz().astype(np.float32)
            # Heuristic unit normalisation: geometry is typically in mm while speeds are in cm/ms.
            # Convert coordinates to cm so that FIM returns times in ms consistently with CPU model.
            # We detect mm-scale by checking the bounding-box extent; typical heart bbox > 50 mm.
            try:
                bbox_extent = np.ptp(pts, axis=0)
                max_extent = float(np.max(bbox_extent))
            except Exception:
                max_extent = 0.0
            if max_extent > 20.0:  # assume mm -> convert to cm
                if self.verbose:
                    print('[EikonalFIMGPU] Detected mm-scale geometry (bbox max ~', round(max_extent, 2), 'mm). Converting to cm.')
                pts = pts / 10.0
            else:
                if self.verbose:
                    print('[EikonalFIMGPU] Geometry appears already in cm (bbox max ~', round(max_extent, 2), ').')
            self._points = pts
        if self._tets is None:
            self._tets = self.geometry.get_tetra().astype(np.int32)
        if self._element_bases is None:
            if self.verbose:
                print('[EikonalFIMGPU] Building per-element fibre bases (one-time).')
            self._element_bases = _build_element_basis(self.geometry)
        if self._element_endo_type is None:
            # Build per-element endocardial classification using edge-level flags if available.
            try:
                edges = getattr(self.geometry, 'edge')  # (E,2) node indices
                is_dense = getattr(self.geometry, 'is_dense_endocardial')  # (E,)
                is_sparse = getattr(self.geometry, 'is_sparse_endocardial')  # (E,)
            except AttributeError:
                # Geometry does not expose endocardial flags; treat all as normal.
                self._element_endo_type = np.zeros((self._tets.shape[0],), dtype=np.uint8)
            else:
                # Map edge node pairs to endocardial status.
                dense_pairs = set()
                sparse_pairs = set()
                for ei in range(edges.shape[0]):
                    a, b = int(edges[ei, 0]), int(edges[ei, 1])
                    key = (a, b) if a < b else (b, a)
                    if is_dense[ei]:
                        dense_pairs.add(key)
                    elif is_sparse[ei]:
                        sparse_pairs.add(key)
                elem_type = np.zeros((self._tets.shape[0],), dtype=np.uint8)
                # For each tetra, inspect its 6 edges.
                for ti, tet in enumerate(self._tets):
                    n0, n1, n2, n3 = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
                    pairs = (
                        (n0, n1) if n0 < n1 else (n1, n0),
                        (n0, n2) if n0 < n2 else (n2, n0),
                        (n0, n3) if n0 < n3 else (n3, n0),
                        (n1, n2) if n1 < n2 else (n2, n1),
                        (n1, n3) if n1 < n3 else (n3, n1),
                        (n2, n3) if n2 < n3 else (n3, n2),
                    )
                    # Priority: dense > sparse
                    if any(p in dense_pairs for p in pairs):
                        elem_type[ti] = 1
                    elif any(p in sparse_pairs for p in pairs):
                        elem_type[ti] = 2
                self._element_endo_type = elem_type
                if self.verbose:
                    c_dense = int(np.sum(self._element_endo_type == 1))
                    c_sparse = int(np.sum(self._element_endo_type == 2))
                    print(f'[EikonalFIMGPU] Endocardial element classification built: dense={c_dense}, sparse={c_sparse}.')

    def _root_nodes_from_meta(self, root_meta_indexes: np.ndarray, purkinje_speed: float) -> Tuple[np.ndarray, np.ndarray]:
        # Root meta entries are expected to be 0/1 flags (after rounding) aligned with candidate list
        y = np.empty_like(root_meta_indexes)
        root_meta_indexes = np.round_(root_meta_indexes, 0, y)
        candidate_idx = self.get_candidate_root_node_index()
        candidate_times = self.get_candidate_root_node_time(purkinje_speed=purkinje_speed)
        active = candidate_idx[root_meta_indexes == 1]
        active_times = candidate_times[root_meta_indexes == 1]
        # Normalise start times to earliest = 0 (consistent with CPU code subtracting first time)
        if active_times.size:
            active_times = active_times - active_times.min()
        return active.astype(np.int32), active_times.astype(np.float32)

    # ------------------------------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------------------------------
    def simulate_propagation(self, parameter_particle_modules_dict):
        if not _FIM_AVAILABLE:
            raise RuntimeError('fim-python (GPU) not available. Install with: pip install fim-python[gpu]')
        parameter_vector = super().get_from_module_dict(parameter_particle_modules_dict)
        return self._simulate_single(parameter_vector)

    def simulate_propagation_population(self, parameter_population_modules_dict):
        if not _FIM_AVAILABLE:
            raise RuntimeError('fim-python (GPU) not available.')
        parameter_population = super().get_from_module_dict(parameter_population_modules_dict)
        # Unique to avoid redundant solves (like CPU version)
        unique_params, inv = np.unique(parameter_population, axis=0, return_inverse=True)
        lat_store = np.zeros((unique_params.shape[0], self.geometry.get_node_xyz().shape[0]), dtype=np.float32)
        for i, p in enumerate(unique_params):
            lat_store[i] = self._simulate_single(p)
        return lat_store[inv]

    def _simulate_single(self, parameter_particle: np.ndarray) -> np.ndarray:
        self._ensure_geom_cache()
        p_dict, root_meta = self._repack(parameter_particle)
        try:
            v_f = p_dict[self.config.fibre_speed_name]
            v_s = p_dict[self.config.sheet_speed_name]
            v_n = p_dict[self.config.normal_speed_name]
            purk_v = p_dict[self.config.purkinje_speed_name]
            endo_dense_v = p_dict[self.config.endo_dense_speed_name]
            endo_sparse_v = p_dict[self.config.endo_sparse_speed_name]
        except KeyError as e:
            raise KeyError(f'Missing speed parameter {e} in particle dict (names: {p_dict.keys()})')
        # Build metric from speeds according to selected convention.
        metrics = _speeds_to_metrics(self._element_bases, v_f, v_s, v_n, inverse=self.config.metric_inverse)

        root_nodes, root_times = self._root_nodes_from_meta(root_meta, purk_v)
        if root_nodes.size == 0:
            raise ValueError('No root nodes selected (all meta flags 0).')

        def solve(m):
            solver_local = create_fim_solver(
                self._points, self._tets, m.astype(np.float32), device='gpu', use_active_list=True
            )
            phi_local = solver_local.comp_fim(root_nodes, root_times)
            return np.asarray(phi_local.get() if hasattr(phi_local, 'get') else phi_local, dtype=np.float32)

        lat_float = solve(metrics)

        # Apply endocardial isotropic override then re-solve once (to include heterogeneity)
        if self._element_endo_type is not None:
            # Keep the same selected convention when overriding endocardial elements.
            metrics_final = metrics.copy()
            dense_mask = self._element_endo_type == 1
            sparse_mask = self._element_endo_type == 2
            if np.any(dense_mask):
                if self.config.metric_inverse:
                    metrics_final[dense_mask] = (np.eye(3, dtype=np.float32) / (endo_dense_v ** 2)).astype(np.float32)
                else:
                    metrics_final[dense_mask] = (np.eye(3, dtype=np.float32) * (endo_dense_v ** 2)).astype(np.float32)
            if np.any(sparse_mask):
                if self.config.metric_inverse:
                    metrics_final[sparse_mask] = (np.eye(3, dtype=np.float32) / (endo_sparse_v ** 2)).astype(np.float32)
                else:
                    metrics_final[sparse_mask] = (np.eye(3, dtype=np.float32) * (endo_sparse_v ** 2)).astype(np.float32)
            if np.any(dense_mask) or np.any(sparse_mask):
                lat_float = solve(metrics_final)
                if self.verbose:
                    print('[EikonalFIMGPU] Re-solved with endocardial isotropic metrics (' +
                          ('inverse' if self.config.metric_inverse else 'direct') + ').')

        # Normalise earliest activation to zero (like CPU subtracting earliest root time) before rounding
        if root_nodes.size:
            earliest_root_time = float(np.min(lat_float[root_nodes]))
            lat_float = lat_float - earliest_root_time

        if self.verbose:
            # Expected plausible max based on geometry span / effective speed (diagnostic only)
            seed_centroid = self._points[root_nodes[:1]]
            d_max = float(np.max(np.linalg.norm(self._points - seed_centroid, axis=1)))
            v_eff = float((v_f + v_s + v_n) / 3.0)
            T_est = d_max / max(v_eff, 1e-6)
            mode = 'inverse(1/speed^2)' if self.config.metric_inverse else 'direct(speed^2)'
            print(f'[EikonalFIMGPU] Using {mode}; max_raw={float(np.max(lat_float)):.2f} ms; T_est≈{T_est:.2f} ms')

        lat_int = np.round(lat_float).astype(np.int32) + 1
        return lat_int


# -------------------------------------------------------------------------------------------------
# Benchmark helper (optional script-like usage)
# -------------------------------------------------------------------------------------------------
def benchmark_compare(cpu_model, gpu_model, parameter_vector: np.ndarray) -> Dict[str, float]:
    """Run a simple wall-clock comparison between CPU Dijkstra and GPU FIM models.

    Parameters
    ----------
    cpu_model : instance of EikonalDjikstraTet (or similar exposing simulate_lat)
    gpu_model : instance of EikonalFIMGPU
    parameter_vector : np.ndarray

    Returns
    -------
    dict with timings (seconds) and basic difference metrics.
    """
    import time
    t0 = time.time()
    lat_cpu = cpu_model.simulate_lat(parameter_vector) if hasattr(cpu_model, 'simulate_lat') else cpu_model.simulate_propagation({cpu_model.module_name: parameter_vector})
    t_cpu = time.time() - t0

    t1 = time.time()
    lat_gpu = gpu_model._simulate_single(parameter_vector)
    t_gpu1 = time.time() - t1
    
    t2 = time.time()
    lat_gpu = gpu_model._simulate_single(parameter_vector)
    t_gpu2 = time.time() - t2

    # Ensure shapes comparable
    lat_cpu_arr = np.asarray(lat_cpu).astype(np.float32)
    lat_gpu_arr = np.asarray(lat_gpu).astype(np.float32)
    if lat_cpu_arr.shape != lat_gpu_arr.shape:
        raise ValueError('CPU/GPU LAT shape mismatch: ' + str((lat_cpu_arr.shape, lat_gpu_arr.shape)))

    diff = lat_cpu_arr - lat_gpu_arr
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    return {
        't_cpu_s': t_cpu,
        't_gpu1_s': t_gpu1,
        't_gpu2_s': t_gpu2,
        'speedup_x': t_cpu / t_gpu1 if t_gpu1 > 0 else np.inf,
        'speedup_x2': t_cpu / t_gpu2 if t_gpu2 > 0 else np.inf,
        'rmse': rmse,
        'max_abs_diff': max_abs,
    }


if __name__ == '__main__':  # Minimal CLI for ad-hoc manual testing
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU Eikonal LAT solvers (prototype).')
    parser.add_argument('--no-run', action='store_true', help='Only test import & environment readiness.')
    args = parser.parse_args()
    if not _FIM_AVAILABLE:
        print('fim-python GPU backend not available. Install with: pip install fim-python[gpu]')
    else:
        print('fim-python GPU backend detected.')
    if args.no_run:
        raise SystemExit(0)
    print('This entry point is a placeholder. Integrate with your existing pipeline to supply geometry & parameters.')
