"""Parallel LS-DYNA + EnKF inversion pipeline.

Default covariance inflation is RTPS.
Optional parameter rejuvenation can be enabled by config.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import matplotlib
import numpy as np

from lsdyna_io import (
    extract_z_disp_observation_array,
    modify_k_file_material_parameters,
    run_lsdyna_solver,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Tee:
    """Duplicate stdout to file and terminal."""

    def __init__(self, path: str, mode: str = "w", encoding: str = "utf-8") -> None:
        self._file = open(path, mode, encoding=encoding)
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def __enter__(self) -> "Tee":
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sys.stdout = self._stdout
        self._file.close()


@dataclass
class EnKFConfig:
    n_iter: int = 20
    n_ens: int = 100
    n_step: int = 3
    n_obs: int = 20
    n_pts: int = 54
    pred_idx: tuple[int, ...] = (3, 8, 12)
    max_workers: int = 12
    lsdyna_ncpu: int = 1
    lsdyna_memory: str = "256m"
    sigma_obs: float = 1e-3
    seed: int = 42
    timeout_sec: int = 3000
    inflation_method: str = "rtps"  # rtps | none
    rtps_alpha: float = 0.7
    enable_rejuvenation: bool = False
    rejuv_threshold: float = 1.5
    rejuv_scale_max: float = 3.0


def apply_covariance_inflation(
    x_f: np.ndarray,
    x_a: np.ndarray,
    parameter_cols: np.ndarray,
    method: str,
    rtps_alpha: float,
) -> np.ndarray:
    """Apply covariance inflation to selected parameter columns (default RTPS)."""
    if method == "none":
        return x_a
    if method != "rtps":
        raise ValueError(f"Unsupported inflation method: {method}")

    xa_bar = x_a.mean(axis=0)
    xf_bar = x_f.mean(axis=0)
    a_a = x_a - xa_bar
    a_f = x_f - xf_bar

    sigma_b = np.std(a_f[:, parameter_cols], axis=0, ddof=1)
    sigma_a = np.std(a_a[:, parameter_cols], axis=0, ddof=1)
    alpha = 1.0 + rtps_alpha * (sigma_b - sigma_a) / np.maximum(sigma_a, 1e-12)
    alpha = np.clip(alpha, 1.0, 1.2)

    a_a[:, parameter_cols] *= alpha
    return xa_bar + a_a


def apply_parameter_dispersion_recovery(
    x_a: np.ndarray,
    parameter_cols: np.ndarray,
    bounds: dict[int, tuple[float, float]],
    pred_idx: tuple[int, ...],
    misfit_j: float,
    threshold_j: float,
    scale_max: float,
) -> np.ndarray:
    """Optional multiplicative parameter rejuvenation step."""
    if misfit_j <= threshold_j:
        return x_a

    xbar = x_a.mean(axis=0)
    a = x_a - xbar
    sd = np.std(a[:, parameter_cols], axis=0, ddof=1)
    target_sd = np.maximum(sd * 1.2, 1e-12)
    scale = np.clip(target_sd / np.maximum(sd, 1e-12), 1.0, scale_max)
    a[:, parameter_cols] *= scale
    out = xbar + a

    for jj, idxp in enumerate(pred_idx):
        lo, hi = bounds[idxp]
        out[:, parameter_cols[jj]] = np.clip(out[:, parameter_cols[jj]], lo, hi)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel LS-DYNA + EnKF inversion")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--lsdyna-bat", required=True)
    parser.add_argument("--lsdyna-solver", required=True)
    args = parser.parse_args()

    cfg = EnKFConfig()
    root = os.path.abspath(args.project_root)

    with Tee(os.path.join(root, "terminal.txt")):
        t0 = time.time()
        rng = np.random.default_rng(cfg.seed)

        bounds = {3: (1e-8, 1e-1), 8: (1e-8, 5.0), 12: (1e-8, 10.0)}
        init_guess = {3: 0.0325, 8: 1.1845, 12: 3.85}
        cv_target = np.array([0.0617, 0.0784, 0.0823], dtype=float)

        n_par = len(cfg.pred_idx)
        param_cols = np.array([-n_par, -n_par + 1, -1], dtype=int)

        ens_mean = np.array([init_guess[i] for i in cfg.pred_idx], dtype=float)
        ens_params_full = np.empty((cfg.n_ens, n_par), dtype=float)
        for j, idx in enumerate(cfg.pred_idx):
            sigma0 = abs(ens_mean[j]) * cv_target[j]
            ens_params_full[:, j] = np.clip(ens_mean[j] + rng.normal(0.0, sigma0, cfg.n_ens), *bounds[idx])

        fig = plt.figure(figsize=(3 * n_par, 3))
        for k, idx in enumerate(cfg.pred_idx):
            ax = fig.add_subplot(1, n_par, k + 1)
            ax.hist(ens_params_full[:, k], bins=30, edgecolor="k", alpha=0.7)
            ax.set_title(["A", "B", "n", "C", "m", "D1", "D2", "D3", "D4", "D5", "CS", "S1", "G", "A0"][idx])
        fig.tight_layout()
        fig.savefig(os.path.join(root, "ensemble_histograms.png"), dpi=300)
        plt.close(fig)

        orig_k = os.path.join(root, "Run.k")
        for eid in range(1, cfg.n_ens + 1):
            ens_dir = os.path.join(root, f"Ensemble_{eid:02d}")
            os.makedirs(ens_dir, exist_ok=True)
            k_out = os.path.join(ens_dir, f"Run_ensemble_{eid:02d}.k")
            modify_k_file_material_parameters(orig_k, k_out, eid, cfg.pred_idx, [f"{ens_params_full[eid-1, j]:.3e}" for j in range(n_par)])

        obs_mean = np.loadtxt(os.path.join(root, "Observation", "z_displ_true_array.txt"), delimiter=",") - 1e-4
        assert obs_mean.size == cfg.n_step * cfg.n_obs

        h = np.zeros((cfg.n_step * cfg.n_obs, cfg.n_step * cfg.n_pts + n_par))
        for k in range(cfg.n_step):
            h[k * cfg.n_obs : (k + 1) * cfg.n_obs, k * cfg.n_pts : k * cfg.n_pts + cfg.n_obs] = np.eye(cfg.n_obs)

        for iter_idx in range(1, cfg.n_iter + 1):
            print(f"\n==== Iter {iter_idx} ====")

            def simulate_and_extract(eid: int):
                ens_dir = os.path.join(root, f"Ensemble_{eid:02d}")
                k_file = os.path.join(ens_dir, f"Run_ensemble_{eid:02d}.k")
                run_lsdyna_solver(
                    input_file=k_file,
                    cwd=ens_dir,
                    ncpu=cfg.lsdyna_ncpu,
                    memory=cfg.lsdyna_memory,
                    timeout_sec=cfg.timeout_sec,
                    bat_path=args.lsdyna_bat,
                    solver_path=args.lsdyna_solver,
                )
                z_out = os.path.join(ens_dir, f"z_disp_{eid:02d}_{iter_idx:02d}.txt")
                w_val = extract_z_disp_observation_array(
                    os.path.join(ens_dir, "nodout"),
                    z_out,
                    n_step=cfg.n_step,
                    n_extract=cfg.n_pts,
                )
                for fp in glob.glob(os.path.join(ens_dir, "d3*")):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
                return eid - 1, w_val

            ens_w = [None] * cfg.n_ens
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futs = [pool.submit(simulate_and_extract, eid) for eid in range(1, cfg.n_ens + 1)]
                for fut in as_completed(futs):
                    idx, w_val = fut.result()
                    ens_w[idx] = w_val

            ok_mask = [w is not None for w in ens_w]
            ens_w_ok = np.asarray([w for w in ens_w if w is not None], dtype=float)
            ens_params_ok = ens_params_full[ok_mask, :]
            x_f = np.hstack((ens_w_ok, ens_params_ok))

            x_bar = x_f.mean(axis=0)
            a = x_f - x_bar
            p_xx = (a.T @ a) / (x_f.shape[0] - 1)
            innovation = obs_mean.reshape(-1, 1) - (h @ x_f.T)
            r_yy = np.eye(cfg.n_step * cfg.n_obs) * (cfg.sigma_obs**2)
            k_gain = p_xx @ h.T @ np.linalg.inv(h @ p_xx @ h.T + r_yy)
            x_a = x_f + (k_gain @ innovation).T

            x_a = apply_covariance_inflation(
                x_f=x_f,
                x_a=x_a,
                parameter_cols=param_cols,
                method=cfg.inflation_method,
                rtps_alpha=cfg.rtps_alpha,
            )

            yhat = (h @ x_a.mean(axis=0).reshape(-1, 1)).reshape(-1)
            residual = obs_mean.reshape(-1) - yhat
            j_now = (residual @ residual) / (obs_mean.size * cfg.sigma_obs**2)

            if cfg.enable_rejuvenation:
                x_a = apply_parameter_dispersion_recovery(
                    x_a=x_a,
                    parameter_cols=param_cols,
                    bounds=bounds,
                    pred_idx=cfg.pred_idx,
                    misfit_j=j_now,
                    threshold_j=cfg.rejuv_threshold,
                    scale_max=cfg.rejuv_scale_max,
                )

            ens_params_full[ok_mask, :] = x_a[:, -n_par:]

            for eid in range(1, cfg.n_ens + 1):
                k_path = os.path.join(root, f"Ensemble_{eid:02d}", f"Run_ensemble_{eid:02d}.k")
                modify_k_file_material_parameters(k_path, k_path, eid, cfg.pred_idx, [f"{ens_params_full[eid-1, j]:.3e}" for j in range(n_par)])

        print(f"Total wall-time: {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
