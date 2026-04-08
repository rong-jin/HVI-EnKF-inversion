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

This repository implements the same overall inversion structure, but the code should be understood as a **practical research implementation** rather than a line-by-line transcription of every equation in the paper. The present version is aligned with the published HVI setup in the following ways:

- the reduced inversion targets the three-parameter set $[C,\;D4,\;\gamma_0]$,
- the forward model extracts back-face `z_disp` from `nodout`,
- the default observation times correspond to **10.0, 11.0, and 12.0 **$\mu$s,
- the code now supports **paper-style Case 1–4 switching** through a `--case` argument,
- the history outputs written by the older internal script are restored under `history_value/`.

---

## Scope of the current repository

The full material-modeling framework discussed in the paper includes:

- Johnson–Cook plasticity,
- Johnson–Cook fracture,
- Mie–Grüneisen EOS.

For the reduced-dimensional calibration setting implemented here, the unknown parameter vector is

$$
\mathbf{u} = [C,\; D4,\; \gamma_0]^T.
$$

Here:

- `C` is the Johnson–Cook strain-rate sensitivity coefficient,
- `D4` is a Johnson–Cook fracture parameter,
- `RG` in the LS-DYNA keyword file corresponds to the Grüneisen parameter $\gamma_0$.

All remaining material parameters are held fixed.

---

## Observation definition

### Selected node set

The numerical workflow distinguishes between:

- **extracted forward nodes**: `n_pts = 54`
- **assimilated observation nodes per time**: `n_obs = 20` by default

This means that each LS-DYNA forward run first extracts **54 back-face nodes** at each selected time step, and the EnKF observation operator then retains the first `n_obs` entries from each time block for data assimilation.

In the published paper, the standard observation configuration uses:

- `n_step = 3`
- `n_obs = 20`
- total observation dimension $N_y = 3 \times 20 = 60$

The reduced-data case uses:

- `n_step = 3`
- `n_obs = 10`
- total observation dimension $N_y = 30$

### Selected time steps

The current code extracts back-face displacement at three physical times:

- `10.0 μs`
- `11.0 μs`
- `12.0 μs`

In the scripts, this is represented by:

- `total_time = 1.2e-5`
- `obs_start_time = 1.0e-5`
- `obs_time_step = 1.0e-6`

The parser computes the corresponding `nodout` block indices automatically from the total number of detected output blocks, which reproduces the behavior of the older internal script while keeping the new header-driven parsing.

---

## Repository structure

```text
.
├─ hvi_enkf_main.py
├─ lsdyna_io.py
├─ prepare_observation.py
├─ Run.k
├─ MAT.txt
├─ Observation/
│  └─ z_displ_true_array.txt
├─ history_value/                 # generated at runtime
└─ Ensemble_01 ... Ensemble_XX/   # generated at runtime
```

### Main files

- **`hvi_enkf_main.py`**  
  Main EnKF inversion driver. Handles ensemble initialization, paper-case selection, parallel forward simulation, EnKF update, covariance inflation, optional rejuvenation, history output, and iterative rewriting of posterior parameters to ensemble-specific `.k` files.

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

The augmented state is

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

### 2. Parallel forward simulation

The most expensive step is the forecast stage, in which a full LS-DYNA simulation is run for each ensemble member. The current implementation parallelizes these forward solves across ensemble directories using Python's `ThreadPoolExecutor`.

### 3. Covariance inflation and optional rejuvenation

The default covariance-inflation method is **RTPS** (Relaxation-to-Prior-Spread), with:

- `inflation_method = "rtps"`
- `rtps_alpha = 0.7`

A parameter-dispersion recovery step is also implemented and may be enabled for strongly biased cases. In the paper-aligned case settings, rejuvenation is enabled by default for **Case 4**.

---

## Current default configuration

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
use_perturbed_obs = True
total_time = 1.2e-5
obs_start_time = 1.0e-5
obs_time_step = 1.0e-6
```

### Interpretation of `pred_idx`

The parameter indices correspond to entries in the material vector:

- `3`  → `C`
- `8`  → `D4`
- `12` → `RG` (`γ0`)

---

## Paper Case 1–4 settings

The repository now supports direct switching among the four published HVI cases through. The **default runtime case is now `case1`**, and `terminal.txt` is written with line-by-line timestamps:

```bash
python hvi_enkf_main.py --case case1
python hvi_enkf_main.py --case case2
python hvi_enkf_main.py --case case3
python hvi_enkf_main.py --case case1
```

The case definitions are:

- **Case 1**: under-biased initial guess  
  $$
  \mathbf{u}_0 = 0.75\,\mathbf{u}_{true}, \qquad N_o = 20
  $$

- **Case 2**: over-biased initial guess  
  $$
  \mathbf{u}_0 = 1.25\,\mathbf{u}_{true}, \qquad N_o = 20
  $$

- **Case 3**: limited observations  
  $$
  \mathbf{u}_0 = 0.75\,\mathbf{u}_{true}, \qquad N_o = 10
  $$

- **Case 4**: strongly biased initial guess  
  $$
  \mathbf{u}_0 = 2.5\,\mathbf{u}_{true}, \qquad N_o = 20
  $$

In the current implementation:

- `--case case3` changes the observation dimension to 10 nodes per selected time,
- `--case case4` enables rejuvenation by default,
- you may override rejuvenation behavior with:
  - `--enable-rejuvenation`
  - `--disable-rejuvenation`

---

## LS-DYNA utilities

The module `lsdyna_io.py` provides reusable helper functions:

- `modify_k_file_material_parameters(...)`
- `run_lsdyna_solver(...)`
- `extract_z_disp_observation_array(...)`
- `extract_field_observation_array(...)`

This modular split keeps LS-DYNA-specific I/O outside the EnKF driver and makes the workflow easier to maintain and extend.

---

## Robust keyword-file parameter update

Material parameters in the LS-DYNA keyword file are updated by **parameter-name matching**, not by fixed line numbers.

For example, the updater searches for labels such as:

- `RC1`
- `RD41`
- `RG`

and replaces only the matched entries.

This makes the update logic more robust to comments, blank lines, and minor formatting changes in `Run.k`.

---

## Robust `nodout` parsing

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

The current implementation computes the target time-step indices from the physical-time settings (`total_time`, `obs_start_time`, `obs_time_step`) so that the new parser remains consistent with the original script's extraction logic.

---

## Observation preparation

Before running the EnKF inversion, first generate the synthetic observation from the baseline keyword file.

### Standard Case 1 / 2 / 4 observation

```bash
python prepare_observation.py --baseline-k Run.k --case case1
```

or

```bash
python prepare_observation.py --baseline-k Run.k --case case4
```

### Reduced-data Case 3 observation

```bash
python prepare_observation.py --baseline-k Run.k --case case3
```

This generates:

```text
Observation/z_displ_true_array.txt
```

Important: if you switch between `case1/case2/case4` and `case3`, regenerate the observation file so that its length matches the expected observation dimension.

---

## Running the inversion

### Default command

```bash
python hvi_enkf_main.py --case case4
```

### Other cases

```bash
python hvi_enkf_main.py --case case1
python hvi_enkf_main.py --case case2
python hvi_enkf_main.py --case case3
```

### Example with explicit Windows paths

```bash
python hvi_enkf_main.py ^
  --project-root . ^
  --case case1 ^
  --lsdyna-bat "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsprepost412\LS-Run\lsdynamsvar.bat" ^
  --lsdyna-solver "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\lsdyna_dp.exe"
```

---

## History outputs

To remain consistent with the older internal script, the current implementation writes runtime diagnostics to:

```text
history_value/
```

Typical files include:

- `corr_history.csv`
- `alpha_history.csv`
- `X_f_XX.csv`
- `Pxx_XX.csv`
- `K_XX.csv`
- `X_a_XX.csv`
- `innovation_XX.csv`
- `residual_XX.csv`
- `misfit_XX.csv`

These files are useful for diagnosing filter behavior, parameter sensitivity, and convergence.

---

## Platform notes

The provided LS-DYNA launcher is currently configured for **Windows** execution, using:

- an LS-Run environment batch file (`--lsdyna-bat`),
- a Windows LS-DYNA solver executable (`--lsdyna-solver`).

If you run this repository on Linux or an HPC cluster, adapt the implementation of `run_lsdyna_solver(...)` in `lsdyna_io.py` to match your local execution environment.

---

## Python requirements

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
