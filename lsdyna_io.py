"""External callable utilities for LS-DYNA I/O and preprocessing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


def modify_k_file_material_parameters(
    input_file: str,
    output_file: str,
    case_id: int,
    predicted_indices: Iterable[int],
    predicted_values: Iterable[str],
) -> None:
    """Update selected material parameters in a LS-DYNA .k file by parameter name."""
    labels = ["RA1", "RB1", "Rn1", "RC1", "Rm1", "RD11", "RD21", "RD31", "RD41", "RD51", "RCS", "RS1", "RG", "RA0"]
    updates = {labels[idx]: val for idx, val in zip(predicted_indices, predicted_values)}

    with open(input_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    found = {k: False for k in updates}
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("$") or "," not in line:
            out_lines.append(line)
            continue

        key = stripped.split(",", 1)[0].strip()
        if key in updates:
            out_lines.append(f"{key},{updates[key]}\n")
            found[key] = True
        else:
            out_lines.append(line)

    missing = [k for k, v in found.items() if not v]
    if missing:
        raise ValueError(f"Missing parameters in k-file {input_file}: {missing}")

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)

    print(f"[E{case_id:02d}] k-file updated by-name: {output_file}")


def run_lsdyna_solver(
    input_file: str,
    cwd: str,
    ncpu: int,
    memory: str,
    timeout_sec: int,
    bat_path: str,
    solver_path: str,
) -> None:
    """Invoke LS-DYNA from the configured environment wrapper and solver binary."""
    k_name = Path(input_file).name
    tag = Path(cwd).name
    print(f"{tag} starts – solver launching ...")

    cmd = f'"{bat_path}" && "{solver_path}" i="{k_name}" ncpu={ncpu} memory={memory}'
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )

    if result.returncode != 0:
        tail_src = (result.stderr or result.stdout or "").strip().splitlines()
        tail = "\n".join(tail_src[-20:] if tail_src else ["<no output>"])
        raise RuntimeError(f"LS-DYNA failed in {tag}, code={result.returncode}\n{tail}")

    print(f"LS-DYNA OK — {tag}")


TIME_STEP_HEADER = "n o d a l   p r i n t   o u t   f o r   t i m e  s t e p"
FIELD_INDEX = {
    "x_disp": 0,
    "y_disp": 1,
    "z_disp": 2,
    "x_vel": 3,
    "y_vel": 4,
    "z_vel": 5,
    "x_accl": 6,
    "y_accl": 7,
    "z_accl": 8,
    "x_coor": 9,
    "y_coor": 10,
    "z_coor": 11,
}
SCI_PATTERN = re.compile(r"[+-]?\d+\.\d+E[+-]\d+", flags=re.IGNORECASE)


def _extract_data_lines_for_step(lines: list[str], step_header_idx: int) -> list[str]:
    i = step_header_idx + 1
    while i < len(lines) and "nodal point" not in lines[i].lower():
        i += 1
    i += 1

    block: list[str] = []
    while i < len(lines):
        line = lines[i]
        low = line.lower()
        if TIME_STEP_HEADER in low or low.strip().startswith("az31") or not line.strip():
            break
        if line.lstrip() and line.lstrip()[0].isdigit():
            block.append(line)
        i += 1
    return block


def _parse_nodout_row(line: str) -> tuple[int, list[float]] | None:
    stripped = line.strip()
    if not stripped:
        return None

    m = re.match(r"^(\d+)\s*(.*)$", stripped)
    if not m:
        return None

    node_id = int(m.group(1))
    rest = m.group(2)
    values = [float(x) for x in SCI_PATTERN.findall(rest)]
    if len(values) < 12:
        return None
    return node_id, values[:12]


def _resolve_time_selection(
    n_total_steps: int,
    *,
    total_time: float | None,
    start_time: float | None,
    step_time: float | None,
    start_step_index: int | None,
    step_interval: int | None,
) -> tuple[int, int]:
    """Resolve which nodout time-step blocks to read.

    If physical times are provided, mimic the legacy script logic:
    dt = total_time / (n_total_steps - 1), then round to nearest block index.
    """
    if start_step_index is not None or step_interval is not None:
        return int(start_step_index or 0), int(step_interval or 1)

    if total_time is None or start_time is None or step_time is None:
        return 0, 1

    if n_total_steps <= 1:
        dt = total_time
    else:
        dt = total_time / (n_total_steps - 1)

    start_idx = int(round(start_time / dt))
    step_idx = max(1, int(round(step_time / dt)))
    return start_idx, step_idx


def extract_z_disp_observation_array(
    nodout_file_path: str,
    out_array_file: str,
    *,
    n_step: int,
    n_extract: int,
    total_time: float | None = None,
    start_time: float | None = None,
    step_time: float | None = None,
    start_step_index: int | None = None,
    step_interval: int | None = None,
) -> np.ndarray:
    """Extract flattened z-disp vector from nodout using header-driven blocks."""
    return extract_field_observation_array(
        nodout_file_path=nodout_file_path,
        out_array_file=out_array_file,
        field="z_disp",
        n_step=n_step,
        n_extract=n_extract,
        total_time=total_time,
        start_time=start_time,
        step_time=step_time,
        start_step_index=start_step_index,
        step_interval=step_interval,
    )


def extract_field_observation_array(
    nodout_file_path: str,
    out_array_file: str,
    *,
    field: str,
    n_step: int,
    n_extract: int,
    total_time: float | None = None,
    start_time: float | None = None,
    step_time: float | None = None,
    start_step_index: int | None = None,
    step_interval: int | None = None,
) -> np.ndarray:
    """Extract flattened observation vector from nodout using header-driven blocks."""
    if field not in FIELD_INDEX:
        raise ValueError(f"Unsupported field: {field}")

    with open(nodout_file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    step_header_idxs = [i for i, line in enumerate(lines) if TIME_STEP_HEADER in line.lower()]
    if not step_header_idxs:
        raise ValueError("No nodal time-step blocks found in nodout.")

    start_idx, stride = _resolve_time_selection(
        len(step_header_idxs),
        total_time=total_time,
        start_time=start_time,
        step_time=step_time,
        start_step_index=start_step_index,
        step_interval=step_interval,
    )

    selected = []
    for k in range(n_step):
        sid = start_idx + k * stride
        if sid >= len(step_header_idxs):
            break
        selected.append(step_header_idxs[sid])

    if len(selected) != n_step:
        raise ValueError(
            f"Requested {n_step} time blocks but only found {len(selected)} after selection. "
            f"start_idx={start_idx}, stride={stride}, total_blocks={len(step_header_idxs)}"
        )

    values_flat: list[float] = []
    fidx = FIELD_INDEX[field]
    for hidx in selected:
        block_lines = _extract_data_lines_for_step(lines, hidx)
        block_vals: list[float] = []
        for row in block_lines:
            parsed = _parse_nodout_row(row)
            if parsed is None:
                continue
            _, vals = parsed
            block_vals.append(vals[fidx])
        values_flat.extend(block_vals[:n_extract])

    arr = np.asarray(values_flat, dtype=float)
    np.savetxt(out_array_file, arr.reshape(1, -1), delimiter=",", fmt="%.6e")
    return arr
