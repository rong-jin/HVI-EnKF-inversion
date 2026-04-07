"""Parallel LS-DYNA + EnKF pipeline template.

This module is a cleaned integration version of the user-provided script with:
- structured configuration (dataclass)
- reusable helpers
- CLI entry point

It is intentionally solver-path configurable and safe for repository integration.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

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


def modify_k_file_mat_parameters_general(
    input_file: str,
    output_file: str,
    case_id: int,
    predicted_indices: Iterable[int],
    predicted_values: Iterable[str],
) -> None:
    labels = ["RA1", "RB1", "Rn1", "RC1", "Rm1", "RD11", "RD21", "RD31", "RD41", "RD51", "RCS", "RS1", "RG", "RA0"]
    with open(input_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    if len(lines) < 21:
        raise ValueError(f"{input_file} has insufficient lines.")

    for idx, val in zip(predicted_indices, predicted_values):
        lines[7 + idx] = f"{labels[idx]},{val}\n"

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"[E{case_id:02d}] k-file updated: {output_file}")


def run_lsdyna(input_file: str, cfg: EnKFConfig, cwd: str, bat_path: str, solver_path: str) -> None:
    k_name = Path(input_file).name
    tag = Path(cwd).name
    print(f"{tag} starts – solver launching ...")
    cmd = f'"{bat_path}" && "{solver_path}" i="{k_name}" ncpu={cfg.lsdyna_ncpu} memory={cfg.lsdyna_memory}'

    res = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=cfg.timeout_sec,
    )
    if res.returncode != 0:
        tail_src = (res.stderr or res.stdout or "").strip().splitlines()
        tail = "\n".join(tail_src[-20:] if tail_src else ["<no output>"])
        raise RuntimeError(f"LS-DYNA failed in {tag}, code={res.returncode}\n{tail}")
    print(f"LS-DYNA OK — {tag}")


def process_nodout_data(nodout_file_path: str, out_array_file: str, *, n_step: int, n_extract: int, node_num: int = 54, total_time: float = 1.2e-5, start_rt_value: float = 1e-5, rt_step: float = 0.1e-5) -> np.ndarray:
    # minimal extraction: keep compatibility with original fixed-width parser
    fields = ["nodal_point", "x_disp", "y_disp", "z_disp", "x_vel", "y_vel", "z_vel", "x_accl", "y_accl", "z_accl", "x_coor", "y_coor", "z_coor"]
    idx = fields.index("z_disp")
    start_idx = 10 + (idx - 1) * 12
    end_idx = start_idx + 12

    with open(nodout_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_line, line_offset, range_length = 68, 60, 54
    zvals: list[float] = []
    while start_line + range_length - 1 <= len(lines):
        for i in range(start_line - 1, start_line + range_length - 1):
            raw = lines[i][start_idx:end_idx].strip()
            if "e" not in raw.lower():
                for j in range(1, len(raw)):
                    if raw[j] in ["+", "-"]:
                        raw = f"{raw[:j]}e{raw[j:]}"
                        break
            zvals.append(float(raw))
        start_line += line_offset

    group_size = node_num
    num_groups = len(zvals) // group_size
    dt = total_time if num_groups < 2 else total_time / (num_groups - 1)
    start_gid = int(round(start_rt_value / dt))
    step_gid = int(round(rt_step / dt))

    picked: list[float] = []
    for g in range(n_step):
        gid = start_gid + g * step_gid
        s = gid * group_size
        picked.extend(zvals[s : s + n_extract])

    arr = np.asarray(picked, dtype=float)
    np.savetxt(out_array_file, arr.reshape(1, -1), delimiter=",", fmt="%.6e")
    return arr


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
            modify_k_file_mat_parameters_general(orig_k, k_out, eid, cfg.pred_idx, [f"{ens_params_full[eid-1, j]:.3e}" for j in range(n_par)])

        obs_mean = np.loadtxt(os.path.join(root, "Observation", "z_displ_true_array.txt"), delimiter=",") - 1e-4
        assert obs_mean.size == cfg.n_step * cfg.n_obs

        for iter_idx in range(1, cfg.n_iter + 1):
            print(f"\n==== Iter {iter_idx} ====")

            def simulate_and_extract(eid: int):
                ens_dir = os.path.join(root, f"Ensemble_{eid:02d}")
                k_file = os.path.join(ens_dir, f"Run_ensemble_{eid:02d}.k")
                run_lsdyna(k_file, cfg, ens_dir, args.lsdyna_bat, args.lsdyna_solver)
                z_out = os.path.join(ens_dir, f"z_disp_{eid:02d}_{iter_idx:02d}.txt")
                w_val = process_nodout_data(os.path.join(ens_dir, "nodout"), z_out, n_step=cfg.n_step, n_extract=cfg.n_pts)
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

            h = np.zeros((cfg.n_step * cfg.n_obs, cfg.n_step * cfg.n_pts + n_par))
            for k in range(cfg.n_step):
                h[k * cfg.n_obs : (k + 1) * cfg.n_obs, k * cfg.n_pts : k * cfg.n_pts + cfg.n_obs] = np.eye(cfg.n_obs)

            x_bar = x_f.mean(axis=0)
            a = x_f - x_bar
            p_xx = (a.T @ a) / (x_f.shape[0] - 1)
            innovation = obs_mean.reshape(-1, 1) - (h @ x_f.T)
            r_yy = np.eye(cfg.n_step * cfg.n_obs) * (cfg.sigma_obs**2)
            k_gain = p_xx @ h.T @ np.linalg.inv(h @ p_xx @ h.T + r_yy)
            x_a = x_f + (k_gain @ innovation).T
            ens_params_full[ok_mask, :] = x_a[:, -n_par:]

            for eid in range(1, cfg.n_ens + 1):
                k_path = os.path.join(root, f"Ensemble_{eid:02d}", f"Run_ensemble_{eid:02d}.k")
                modify_k_file_mat_parameters_general(k_path, k_path, eid, cfg.pred_idx, [f"{ens_params_full[eid-1, j]:.3e}" for j in range(n_par)])

        print(f"Total wall-time: {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
