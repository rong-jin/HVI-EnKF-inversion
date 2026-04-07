# EnKF-for-Material-Model-Characterization

## Overview

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

## Run command

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
