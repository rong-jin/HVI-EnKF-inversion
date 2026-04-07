# Ensemble-based Data Assimilation for Material Model Characterization in High-Velocity Impact

## Overview

This repository provides a research-oriented implementation of an **ensemble-based data assimilation (DA)** framework for **material model calibration in high-velocity impact (HVI)** simulations. The workflow couples a high-fidelity **LS-DYNA Smoothed Particle Hydrodynamics (SPH)** forward solver with an **Ensemble Kalman Filter (EnKF)** to infer selected constitutive, fracture, and equation-of-state (EOS) parameters from time-series back-face deflection data.

The implementation is based on the methodology reported in:

> **R. Jin, G. Wang, and X. Sun**  
> *Ensemble-based data assimilation for material model characterization in high-velocity impact*  
> **International Journal of Impact Engineering**, 215 (2026), 105738  
> DOI: https://doi.org/10.1016/j.ijimpeng.2026.105738

The framework is designed for **non-intrusive inverse calibration** of computationally expensive black-box simulators. In the present repository, LS-DYNA is treated as the forward model, while the EnKF iteratively updates an ensemble of material parameters using simulated observations extracted from `nodout`.

---

## Relation to the published paper

The published paper presents the broader EnKF–SPH methodology, including RTPS covariance inflation, a rejuvenation strategy for extreme prior bias, and numerical studies on synthetic HVI observations.

This repository implements the same overall inversion structure, but the **code should be understood as a practical research implementation rather than a line-by-line reproduction of every equation in the paper**. In particular:

- the default inversion targets the reduced three-parameter set $[C,\;D4,\;\gamma_0]$,
- the current code uses a **deterministic observation vector** in the analysis step,
- the optional `enable_rejuvenation` branch implements a **lightweight dispersion-recovery mechanism triggered by normalized misfit**, rather than the full multi-condition trigger logic described in the paper,
- the default repository setting is `sigma_obs = 1e-3`, which should be treated as a configurable code parameter.

These choices are intended to keep the workflow compact, stable, and easy to adapt.

---

## Scientific Background

Accurate simulation of HVI events requires reliable constitutive, fracture, and volumetric material models. In practice, these models often contain parameters that are difficult to determine uniquely from conventional experiments and are therefore tuned manually. This repository implements an alternative approach in which simulation outputs are assimilated against observation data using an ensemble-based inverse framework.

The current implementation follows the published study in which:

- the forward solver is an **SPH model in LS-DYNA**,
- the observable is **time-series back-face deflection**,
- the inverse method is the **Ensemble Kalman Filter**,
- **covariance inflation** is used to avoid filter overconfidence, and
- an optional **parameter-spread recovery mechanism** is available for difficult prior settings.

Because the EnKF uses only ensemble statistics and does not require tangent or adjoint operators, it is particularly suitable for expensive black-box HVI simulations.

---

## Scope of the Current Repository

The full material-modeling framework discussed in the paper includes:

- Johnson–Cook plasticity,
- Johnson–Cook fracture, and
- Mie–Grüneisen EOS.

For the reduced-dimensional calibration setting implemented here, the unknown parameter vector is:

$$
\mathbf{u} = [C,\; D4,\; \gamma_0]^T
$$

where:

- `C` is the Johnson–Cook strain-rate sensitivity coefficient,
- `D4` is a Johnson–Cook fracture parameter, and
- `RG` in the LS-DYNA keyword file corresponds to the Grüneisen parameter `γ0`.

All remaining material parameters are kept fixed.

The default observation setting is:

- `n_step = 3` observation times,
- `n_obs = 20` observed nodes per time,
- `n_pts = 54` extracted nodes per forward run,
- total observation dimension \(N_y = 3 \times 20 = 60\).

---

## Repository Structure

```text
.
├─ hvi_enkf_main.py
├─ lsdyna_io.py
├─ prepare_observation.py
├─ Run.k
├─ MAT.txt
├─ Observation/
│  └─ z_displ_true_array.txt
└─ Ensemble_01 ... Ensemble_XX/    # created automatically at runtime
```

### Main files

- **`hvi_enkf_main.py`**  
  Main EnKF inversion driver. Handles ensemble initialization, parallel forward simulation, state update, covariance inflation, optional rejuvenation, and iterative rewriting of posterior parameters to ensemble-specific `.k` files.

- **`lsdyna_io.py`**  
  External callable utilities for LS-DYNA I/O, including keyword-file parameter replacement, solver invocation, and robust `nodout` parsing.

- **`prepare_observation.py`**  
  Utility script that runs a baseline LS-DYNA simulation and generates the synthetic observation file used by the EnKF workflow.

- **`Run.k`**  
  Baseline LS-DYNA keyword file used as the forward-model template.

- **`MAT.txt`**  
  Baseline/reference material parameter vector.

---

## Methodology

### 1. Artificial-time EnKF formulation

This repository follows an **artificial-time EnKF inversion** strategy. Instead of assimilating data sequentially in physical time, each EnKF iteration uses the complete observation vector from a single HVI event.

The augmented state is written as

$$
\mathbf{x} =
\begin{bmatrix}
\mathbf{G}(\mathbf{u}) \\
\mathbf{u}
\end{bmatrix},
$$

where:

- $\mathbf{u}$ is the parameter vector,
- $\mathbf{G}(\mathbf{u})$ is the simulated observation vector,
- the observation operator extracts the observation portion of the state.

At every iteration, the workflow:

1. propagates each ensemble member through the forward solver,
2. constructs the augmented state,
3. estimates the forecast covariance from the ensemble,
4. computes the Kalman gain,
5. updates the state using the observation innovation,
6. applies inflation or rejuvenation if needed,
7. writes updated parameters back to LS-DYNA keyword files.

---

### 2. Parallel forward simulation

The most expensive step is the forecast stage, in which a full LS-DYNA simulation is run for each ensemble member. The current implementation parallelizes these forward solves across ensemble directories using Python's `ThreadPoolExecutor`.

Each ensemble member is stored in its own subdirectory:

```text
Ensemble_01/
Ensemble_02/
...
Ensemble_100/
```

and contains its own updated LS-DYNA keyword file.

---

### 3. Observation extraction

The observable used in the current workflow is the flattened array of **back-face `z_disp` values** extracted from `nodout`.

The observation file is stored as:

```text
Observation/z_displ_true_array.txt
```

and is written as a single comma-separated row vector.

The current setup assumes:

- 3 observation times,
- 20 retained observation points per time,
- total length = 60.

In the forward model, the code extracts `n_pts = 54` values per selected time step and then applies the observation operator to retain the first `n_obs = 20` values from each time block.

---

### 4. Covariance inflation

The default covariance-inflation method is **RTPS** (Relaxation-to-Prior-Spread). This is used to mitigate ensemble under-dispersion after the EnKF analysis step.

The current default settings are:

- `inflation_method = "rtps"`
- `rtps_alpha = 0.7`

Inflation is applied only to the selected parameter columns.

---

### 5. Optional parameter dispersion recovery

A parameter-dispersion recovery step is implemented and can be enabled when needed. This mechanism is intended for difficult cases where:

- the initial ensemble is strongly biased,
- the ensemble variance collapses too early,
- the predicted response remains inconsistent with the observation.

In the present codebase, rejuvenation is **disabled by default** and is activated only when the normalized misfit exceeds a user-defined threshold.

---

## Current Default Configuration

The main inversion script defines an `EnKFConfig` dataclass with the following defaults:

```python
n_iter = 20
n_ens = 100
n_step = 3
n_obs = 20
n_pts = 54
pred_idx = (3, 8, 12)
max_workers = 12
lsdyna_ncpu = 1
lsdyna_memory = "256m"
sigma_obs = 1e-3
seed = 42
timeout_sec = 3000
inflation_method = "rtps"
rtps_alpha = 0.7
enable_rejuvenation = False
rejuv_threshold = 1.5
rejuv_scale_max = 3.0
```

### Interpretation of `pred_idx`

The parameter indices correspond to entries in the material vector:

- `3`  → `C`
- `8`  → `D4`
- `12` → `RG` (`γ0`)

---

## LS-DYNA Utilities

The module `lsdyna_io.py` provides reusable helper functions:

- `modify_k_file_material_parameters(...)`
- `run_lsdyna_solver(...)`
- `extract_z_disp_observation_array(...)`
- `extract_field_observation_array(...)`

This modular split keeps LS-DYNA-specific I/O outside the EnKF driver and makes the workflow easier to maintain and extend.

---

## Robust Keyword-File Parameter Update

Material parameters in the LS-DYNA keyword file are updated by **parameter-name matching**, not by fixed line numbers.

For example, the updater searches for labels such as:

- `RC1`
- `RD41`
- `RG`

and replaces only the matched entries.

### Why this matters

A line-number-based updater is fragile and can fail when:

- comments are added,
- blank lines are inserted,
- the file header changes,
- formatting is modified.

By-name replacement is significantly more robust for long-term code maintenance.

---

## Robust `nodout` Parsing

Observation extraction is implemented using **header-driven parsing**, not fixed line offsets.

The parser detects blocks using the LS-DYNA nodal output header:

```text
n o d a l   p r i n t   o u t   f o r   t i m e  s t e p
```

For each selected time step, the parser:

1. finds the corresponding block header,
2. locates the `nodal point ...` line,
3. reads valid nodal rows,
4. parses the requested field,
5. assembles the flattened observation vector.

### Advantages of this approach

This strategy is much more robust when:

- the nodset size changes,
- the number of printed nodes changes,
- LS-DYNA output formatting shifts,
- additional lines appear between output blocks.

---

## Observation Preparation

Before running the EnKF inversion, first generate the synthetic observation from the baseline keyword file.

### Default command

```bash
python prepare_observation.py --baseline-k Run.k
```

### Equivalent explicit Windows command

```bash
python prepare_observation.py ^
  --project-root . ^
  --baseline-k Run.k ^
  --lsdyna-bat "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat" ^
  --lsdyna-solver "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe"
```

This generates:

```text
Observation/z_displ_true_array.txt
```

The script works by:

1. copying `Run.k` into `Observation/baseline_run/`,
2. launching LS-DYNA for the baseline simulation,
3. reading the resulting `nodout`,
4. extracting the observation vector,
5. writing the vector to the `Observation/` directory.

---

## Running the Inversion

### Default command

```bash
python hvi_enkf_main.py
```

### Equivalent explicit Windows command

```bash
python hvi_enkf_main.py ^
  --project-root . ^
  --lsdyna-bat "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat" ^
  --lsdyna-solver "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe"
```

---

## Platform Notes

The provided LS-DYNA launcher is currently configured for **Windows** execution, using:

- an LS-Run environment batch file (`--lsdyna-bat`),
- a Windows LS-DYNA solver executable (`--lsdyna-solver`).

If you run this repository on Linux or an HPC cluster, you will need to adapt the implementation of `run_lsdyna_solver(...)` in `lsdyna_io.py` to match your local execution environment, such as:

- shell-based launch,
- module-based launch,
- scheduler submission,
- containerized execution.

---

## Python Requirements

Recommended environment:

- Python 3.10 or newer
- NumPy
- Matplotlib

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

In addition, a valid LS-DYNA installation is required.

---

## Typical Outputs

A typical run generates:

- `Ensemble_01/ ... Ensemble_100/`
- ensemble-specific keyword files:
  - `Run_ensemble_01.k`, ...
- extracted displacement arrays:
  - `z_disp_XX_YY.txt`
- `ensemble_histograms.png`
- `terminal.txt`

These files are useful for diagnostics, reproducibility, and debugging.

---

## Reproducibility Recommendations

For reproducible use of this repository, it is strongly recommended to record:

- LS-DYNA version,
- operating system,
- solver executable path,
- ensemble size,
- number of EnKF iterations,
- observation noise level,
- inflation settings,
- rejuvenation settings,
- baseline `Run.k`,
- baseline `MAT.txt`,
- generated observation file.

Because LS-DYNA simulations are computationally expensive and can be environment-dependent, recording these settings is important for consistent reproduction of results.

---

## Limitations

This repository is a **research codebase**, not a general-purpose software package.

Current limitations include:

1. **Reduced-dimensional inversion**  
   The default implementation targets only three parameters (`C`, `D4`, `γ0`), not the full material-parameter set.

2. **Synthetic observation workflow**  
   The default pipeline uses synthetic observations generated from a baseline LS-DYNA run.

3. **Windows-oriented solver launcher**  
   The provided solver call is configured for Windows and should be adapted for Linux/HPC environments.

4. **Repository-level configuration**  
   Most settings are currently defined directly in Python rather than in an external configuration file.

5. **Single-observable emphasis**  
   The present implementation focuses on back-face deflection. Other fields can be extracted through `extract_field_observation_array(...)`, but a full multi-observable inversion interface is not yet packaged.

---

## Possible Extensions

Future extensions may include:

- assimilation of experimental 3D-DIC or DGS measurements,
- heteroskedastic observation-noise models,
- support for additional LS-DYNA output fields,
- higher-dimensional parameter calibration,
- native HPC launcher support,
- external YAML/JSON configuration files,
- automated diagnostics and postprocessing tools.

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{jin2026enkf_hvi,
  title   = {Ensemble-based data assimilation for material model characterization in high-velocity impact},
  author  = {Jin, Rong and Wang, Guangyao and Sun, Xingsheng},
  journal = {International Journal of Impact Engineering},
  volume  = {215},
  pages   = {105738},
  year    = {2026},
  doi     = {10.1016/j.ijimpeng.2026.105738}
}
```

See also `CITATION.cff` for GitHub-compatible citation metadata.

---

## Disclaimer

This repository is intended for research and academic use. Users are responsible for verifying:

- unit consistency,
- parameter bounds,
- solver settings,
- observation definitions,
- applicability of the workflow to their own HVI problems.

---

## Contact

For questions related to the methodology or the published study, please contact the corresponding author listed in the paper.
