# Ensemble-based data assimilation for material model characterization in high-velocity impact
## Overview

This codebase corresponds to the implementation described in:

- https://doi.org/10.1016/j.ijimpeng.2026.105738

This repository contains an integration-ready **Parallel LS-DYNA + EnKF inversion** workflow:

- Main workflow: `hvi_enkf_main.py`
- External callable LS-DYNA utilities: `lsdyna_io.py`

Algorithm flow:
1. Run forecast for each ensemble member in parallel.
2. Build augmented state and run EnKF analysis.
3. Apply **covariance inflation** (default = RTPS).
4. Optionally apply **parameter rejuvenation** (disabled by default).
5. Update each ensemble `.k` file with posterior parameters.

---

## External callable functions

`lsdyna_io.py` provides reusable functions that can be imported from any script:

- `modify_k_file_material_parameters(...)`
- `run_lsdyna_solver(...)`
- `extract_z_disp_observation_array(...)`
- `extract_field_observation_array(...)`

This keeps file I/O and solver details outside the EnKF core pipeline.

---

## Robust nodout parsing (header-driven, not line-number-driven)

Nodout extraction is now based on detecting this block header:

`n o d a l   p r i n t   o u t   f o r   t i m e  s t e p`

Instead of using fixed `start_line` / `line_offset`, the parser:

1. Finds each time-step block via the header text.
2. Locates the `nodal point ...` column line for that block.
3. Parses all node rows under that block.
4. Extracts target field values (default `z_disp`).

### Why this is better

If the nodset size changes (more/fewer nodes), line-number offsets usually break.
Header-driven block parsing remains stable because it follows the block structure directly.

---

## Project layout

```text
EnKF-for-Material-Model-Characterization/
├─ hvi_enkf_main.py
├─ lsdyna_io.py
├─ Run.k
├─ MAT.txt
├─ Observation/
│  └─ z_displ_true_array.txt
└─ Ensemble_01 ... Ensemble_XX/   (created at runtime)
```

---

## Platform note

Current `LS-DYNA` invocation (`run_lsdyna_solver`) is configured for **Windows** execution, using:

- `--lsdyna-bat` (ANSYS/LS-Run environment batch file)
- `--lsdyna-solver` (Windows LS-DYNA executable path)

If you run on Linux/HPC, replace the command launcher in `lsdyna_io.py` accordingly.

---

## Run command

Default run (uses built-in Windows defaults for `project-root`, `lsdyna-bat`, and `lsdyna-solver`):

```bash
python hvi_enkf_main.py
```

Equivalent explicit run:

```bash
python hvi_enkf_main.py \
  --project-root . \
  --lsdyna-bat "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat" \
  --lsdyna-solver "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe"
```

---

## Inflation and rejuvenation settings

Default configuration in `EnKFConfig`:

- `inflation_method = "rtps"`
- `rtps_alpha = 0.7`
- `enable_rejuvenation = False`

You can enable rejuvenation by setting:

- `enable_rejuvenation = True`


---


---


---

## Observation generation and EnKF integration

Observation preparation is now executed by an external script, while `hvi_enkf_main.py` remains focused on EnKF inversion.

1. **Prepare observation from baseline k-file**

```bash
python prepare_observation.py --baseline-k Run.k
```

This generates:

- `Observation/z_displ_true_array.txt`

2. **Run EnKF inversion (main pipeline)**

```bash
python hvi_enkf_main.py
```


---

## k-file parameter update rule

`modify_k_file_material_parameters(...)` now updates parameter lines by **parameter name matching** (e.g., `RC1`, `RD41`, `RG`) rather than fixed line numbers.

So changes in header/comment line count will not break parameter replacement, as long as target parameter names still exist in the k-file.
