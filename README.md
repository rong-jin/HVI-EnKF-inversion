# EnKF-for-Material-Model-Characterization

## Overview

This repository now contains an integration-ready **Parallel LS-DYNA + EnKF inversion** pipeline:

- Main workflow: `enkf_pipeline.py`
- External callable LS-DYNA helpers: `lsdyna_io.py`

The implementation follows your requested algorithmic flow:
1. Forecast each ensemble member in parallel.
2. Build augmented state and run EnKF analysis.
3. Apply **covariance inflation (default = RTPS)**.
4. Optionally apply **parameter rejuvenation** (disabled by default).
5. Extract updated parameters and write back into each `.k` file.

---

## Project layout

```text
EnKF-for-Material-Model-Characterization/
├─ enkf_pipeline.py
├─ lsdyna_io.py
├─ Run.k
├─ MAT.txt
├─ Observation/
│  └─ z_displ_true_array.txt
└─ Ensemble_01 ... Ensemble_XX/   (created at runtime)
```

---

## Run command

Use runtime arguments for LS-DYNA paths:

```bash
python enkf_pipeline.py \
  --project-root . \
  --lsdyna-bat "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat" \
  --lsdyna-solver "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe"
```

---

## Covariance inflation and rejuvenation strategy

### Default: RTPS covariance inflation

RTPS is enabled by default via:

- `EnKFConfig.inflation_method = "rtps"`
- `EnKFConfig.rtps_alpha = 0.7`

Implementation entry point:
- `apply_covariance_inflation(...)`

### Optional: parameter rejuvenation

Parameter rejuvenation is configurable and **off by default**:

- `EnKFConfig.enable_rejuvenation = False`

When enabled, it is applied after inflation using:
- `maybe_parameter_rejuvenation(...)`

