"""Create Observation/z_displ_true_array.txt from a baseline k-file run."""

from __future__ import annotations

import argparse
import os
import shutil

from hvi_enkf_main import EnKFConfig
from lsdyna_io import extract_z_disp_observation_array, run_lsdyna_solver


def prepare_observation_from_baseline_k(
    root: str,
    cfg: EnKFConfig,
    lsdyna_bat: str,
    lsdyna_solver: str,
    baseline_k_name: str = "Run.k",
) -> str:
    obs_dir = os.path.join(root, "Observation")
    os.makedirs(obs_dir, exist_ok=True)

    baseline_dir = os.path.join(obs_dir, "baseline_run")
    os.makedirs(baseline_dir, exist_ok=True)

    src_k = os.path.join(root, baseline_k_name)
    dst_k = os.path.join(baseline_dir, baseline_k_name)
    shutil.copy2(src_k, dst_k)

    run_lsdyna_solver(
        input_file=dst_k,
        cwd=baseline_dir,
        ncpu=cfg.lsdyna_ncpu,
        memory=cfg.lsdyna_memory,
        timeout_sec=cfg.timeout_sec,
        bat_path=lsdyna_bat,
        solver_path=lsdyna_solver,
    )

    out_obs = os.path.join(obs_dir, "z_displ_true_array.txt")
    extract_z_disp_observation_array(
        nodout_file_path=os.path.join(baseline_dir, "nodout"),
        out_array_file=out_obs,
        n_step=cfg.n_step,
        n_extract=cfg.n_obs,
    )
    print(f"Observation array generated: {out_obs}")
    return out_obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EnKF observation file from baseline k-file")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-k", default="Run.k")
    parser.add_argument(
        "--lsdyna-bat",
        default=r"C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat",
    )
    parser.add_argument(
        "--lsdyna-solver",
        default=r"C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe",
    )
    args = parser.parse_args()

    cfg = EnKFConfig()
    root = os.path.abspath(args.project_root)
    prepare_observation_from_baseline_k(
        root=root,
        cfg=cfg,
        lsdyna_bat=args.lsdyna_bat,
        lsdyna_solver=args.lsdyna_solver,
        baseline_k_name=args.baseline_k,
    )


if __name__ == "__main__":
    main()
