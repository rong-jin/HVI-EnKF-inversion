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
    """Duplicate stdout to file and terminal, prefixing each printed line with a timestamp."""

    def __init__(self, path: str, mode: str = "w", encoding: str = "utf-8") -> None:
        self._file = open(path, mode, encoding=encoding)
        self._stdout = sys.stdout
        self._buffer = ""

    def write(self, msg: str) -> None:
        self._buffer += msg
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            stamped = f"[{timestamp}] {line}\n"
            self._stdout.write(stamped)
            self._file.write(stamped)

    def flush(self) -> None:
        if self._buffer:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            stamped = f"[{timestamp}] {self._buffer}"
            self._stdout.write(stamped)
            self._file.write(stamped)
            self._buffer = ""
        self._stdout.flush()
        self._file.flush()

    def __enter__(self) -> "Tee":
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
        sys.stdout = self._stdout
        self._file.close()

class EnKFConfig:
    n_iter: int = 3
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
    use_perturbed_obs: bool = True
    total_time: float = 1.2e-5
    obs_start_time: float = 1.0e-5
    obs_time_step: float = 1.0e-6


def get_case_settings(case_name: str, true_params: np.ndarray) -> dict[str, object]:
    """Return paper-aligned case settings."""
    case_key = case_name.lower()
    if case_key == "case1":
        factor, n_obs, rejuvenation = 0.75, 20, False
        description = "Case 1: under-biased initial guess, u0 = 0.75 * u_true, N_o = 20"
    elif case_key == "case2":
        factor, n_obs, rejuvenation = 1.25, 20, False
        description = "Case 2: over-biased initial guess, u0 = 1.25 * u_true, N_o = 20"
    elif case_key == "case3":
        factor, n_obs, rejuvenation = 0.75, 10, False
        description = "Case 3: limited observations, u0 = 0.75 * u_true, N_o = 10"
    elif case_key == "case4":
        factor, n_obs, rejuvenation = 2.5, 20, True
        description = "Case 4: strongly biased initial guess, u0 = 2.5 * u_true, N_o = 20"
    else:
        raise ValueError(f"Unsupported case: {case_name}")

    init_guess = {idx: factor * float(true_params[idx]) for idx in (3, 8, 12)}
    return {
        "case_name": case_key,
        "description": description,
        "init_guess": init_guess,
        "n_obs": n_obs,
        "enable_rejuvenation": rejuvenation,
    }


def apply_covariance_inflation(
    x_f: np.ndarray,
    x_a: np.ndarray,
    parameter_cols: np.ndarray,
    method: str,
    rtps_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply covariance inflation to selected parameter columns (default RTPS)."""
    if method == "none":
        alpha = np.ones(len(parameter_cols), dtype=float)
        return x_a, alpha
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
    return xa_bar + a_a, alpha


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
    parser.add_argument("--case", choices=["case1", "case2", "case3", "case4"], default="case1")
    parser.add_argument(
        "--lsdyna-bat",
        default=r"C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat",
    )
    parser.add_argument(
        "--lsdyna-solver",
        default=r"C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe",
    )
    parser.add_argument("--enable-rejuvenation", action="store_true", dest="enable_rejuvenation_override")
    parser.add_argument("--disable-rejuvenation", action="store_true", dest="disable_rejuvenation_override")
    args = parser.parse_args()

    cfg = EnKFConfig()
    root = os.path.abspath(args.project_root)

    with Tee(os.path.join(root, "terminal.txt")):
        obs_path = os.path.join(root, "Observation", "z_displ_true_array.txt")
        t0 = time.time()
        rng = np.random.default_rng(cfg.seed)

        true_pars = np.loadtxt(os.path.join(root, "MAT.txt"))[:14]
        case_settings = get_case_settings(args.case, true_pars)
        n_obs = int(case_settings["n_obs"])

        enable_rejuvenation = bool(case_settings["enable_rejuvenation"])
        if args.enable_rejuvenation_override:
            enable_rejuvenation = True
        if args.disable_rejuvenation_override:
            enable_rejuvenation = False

        print(case_settings["description"])
        print(f"Observation extraction: n_step={cfg.n_step}, n_obs={n_obs}, n_pts={cfg.n_pts}")
        print(
            "Observation times: "
            f"start={cfg.obs_start_time:.6e} s, dt={cfg.obs_time_step:.6e} s, total_time={cfg.total_time:.6e} s"
        )
        print(f"Parameter rejuvenation enabled: {enable_rejuvenation}")

        bounds = {3: (1e-8, 1e-1), 8: (1e-8, 5.0), 12: (1e-8, 10.0)}
        init_guess = case_settings["init_guess"]
        cv_target = np.array([0.0617, 0.0784, 0.0823], dtype=float)

        n_par = len(cfg.pred_idx)
        param_cols = np.array([-n_par, -n_par + 1, -1], dtype=int)

        ens_mean = np.array([init_guess[i] for i in cfg.pred_idx], dtype=float)
        ens_params_full = np.empty((cfg.n_ens, n_par), dtype=float)
        for j, idx in enumerate(cfg.pred_idx):
            sigma0 = abs(ens_mean[j]) * cv_target[j]
            ens_params_full[:, j] = np.clip(
                ens_mean[j] + rng.normal(0.0, sigma0, cfg.n_ens),
                *bounds[idx],
            )

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
            modify_k_file_material_parameters(
                orig_k,
                k_out,
                eid,
                cfg.pred_idx,
                [f"{ens_params_full[eid-1, j]:.3e}" for j in range(n_par)],
            )

        obs_mean = np.loadtxt(obs_path, delimiter=",") - 1e-4
        assert obs_mean.size == cfg.n_step * n_obs, (
            f"Observation length mismatch: expected {cfg.n_step * n_obs}, got {obs_mean.size}. "
            "Regenerate Observation/z_displ_true_array.txt for the selected case."
        )

        h = np.zeros((cfg.n_step * n_obs, cfg.n_step * cfg.n_pts + n_par))
        for k in range(cfg.n_step):
            h[k * n_obs : (k + 1) * n_obs, k * cfg.n_pts : k * cfg.n_pts + n_obs] = np.eye(n_obs)

        hist_dir = os.path.join(root, "history_value")
        os.makedirs(hist_dir, exist_ok=True)
        alpha_history: list[list[float]] = []

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
                    total_time=cfg.total_time,
                    start_time=cfg.obs_start_time,
                    step_time=cfg.obs_time_step,
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
            if not any(ok_mask):
                raise RuntimeError("All ensemble members failed this iteration.")

            ens_w_ok = np.asarray([w for w in ens_w if w is not None], dtype=float)
            ens_params_ok = ens_params_full[ok_mask, :]
            x_f = np.hstack((ens_w_ok, ens_params_ok))

            x_bar = x_f.mean(axis=0)
            a = x_f - x_bar
            p_xx = (a.T @ a) / (x_f.shape[0] - 1)

            x_safe = np.nan_to_num(x_f, nan=0.0)
            cmat = np.corrcoef(x_safe.T)
            corr_blk = cmat[: cfg.n_step * n_obs, -n_par:]
            max_corr = np.max(np.abs(corr_blk), axis=0)
            with open(os.path.join(hist_dir, "corr_history.csv"), "a", encoding="utf-8") as f:
                f.write(f"{iter_idx}," + ",".join(f"{c:.6f}" for c in max_corr) + "\n")

            m = cfg.n_step * n_obs
            y_mean = obs_mean.reshape(-1, 1)

            if cfg.use_perturbed_obs:
                eps = rng.normal(loc=0.0, scale=cfg.sigma_obs, size=(m, x_f.shape[0]))
                y_pert = y_mean + eps
                innovation = y_pert - (h @ x_f.T)
            else:
                innovation = y_mean - (h @ x_f.T)

            r_yy = np.eye(m) * (cfg.sigma_obs**2)
            k_gain = p_xx @ h.T @ np.linalg.inv(h @ p_xx @ h.T + r_yy)
            x_a = x_f + (k_gain @ innovation).T

            x_a, alpha_par = apply_covariance_inflation(
                x_f=x_f,
                x_a=x_a,
                parameter_cols=param_cols,
                method=cfg.inflation_method,
                rtps_alpha=cfg.rtps_alpha,
            )

            yhat = (h @ x_a.mean(axis=0).reshape(-1, 1)).reshape(-1)
            residual = obs_mean.reshape(-1) - yhat
            j_now = (residual @ residual) / (obs_mean.size * cfg.sigma_obs**2)

            if enable_rejuvenation:
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

            tag = f"{iter_idx:02d}"
            np.savetxt(os.path.join(hist_dir, f"X_f_{tag}.csv"), x_f, delimiter=",", fmt="%.6e")
            np.savetxt(os.path.join(hist_dir, f"Pxx_{tag}.csv"), p_xx, delimiter=",", fmt="%.6e")
            np.savetxt(os.path.join(hist_dir, f"K_{tag}.csv"), k_gain, delimiter=",", fmt="%.6e")
            np.savetxt(os.path.join(hist_dir, f"X_a_{tag}.csv"), x_a, delimiter=",", fmt="%.6e")
            np.savetxt(os.path.join(hist_dir, f"innovation_{tag}.csv"), innovation, delimiter=",", fmt="%.6e")
            np.savetxt(os.path.join(hist_dir, f"residual_{tag}.csv"), residual.reshape(1, -1), delimiter=",", fmt="%.6e")
            np.savetxt(os.path.join(hist_dir, f"misfit_{tag}.csv"), np.array([[j_now]]), delimiter=",", fmt="%.6e")

            alpha_history.append(alpha_par.tolist())
            np.savetxt(
                os.path.join(hist_dir, "alpha_history.csv"),
                np.asarray(alpha_history, dtype=float),
                delimiter=",",
                fmt="%.6f",
                header="C,D4,G",
                comments="",
            )

            for eid in range(1, cfg.n_ens + 1):
                if not ok_mask[eid - 1]:
                    continue
                k_path = os.path.join(root, f"Ensemble_{eid:02d}", f"Run_ensemble_{eid:02d}.k")
                modify_k_file_material_parameters(
                    k_path,
                    k_path,
                    eid,
                    cfg.pred_idx,
                    [f"{ens_params_full[eid-1, j]:.3e}" for j in range(n_par)],
                )

        print(f"Total wall-time: {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()

