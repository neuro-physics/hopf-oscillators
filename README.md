# Escape Times of Hopf Oscillators on the Connectome

This repository contains a Python program to simulate **stochastic Hopf oscillators coupled through an empirical brain connectome**, with heterogeneous local dynamics, delayed interactions derived from tract lengths, and additive noise.

The main output of the simulation is the **escape time** of each oscillator, defined as the first time the amplitude of the complex state exceeds a user-defined threshold.

---

## Overview

Each node of the network represents a brain region and follows noisy Hopf dynamics close to the bifurcation point. Nodes are coupled through weighted connections extracted from structural and functional connectivity data. Conduction delays are computed from tract lengths and a global conduction speed.

The code supports:
- Multiple simulation trials
- Heterogeneous oscillator parameters
- Delayed coupling
- Input connectomes in either **text (`.txt`)** or **MATLAB (`.mat`)** format
- Optional normalization of coupling strength
- Escape-time statistics across nodes and trials

---

## Usage

```bash
python escape_times.py [options]
```

To see all available options:

```bash
python escape_times.py -h
```

---

## Main Options

### Simulation control

| Option | Description |
|------|------------|
| `-ntrials INT` | Number of trials to repeat the simulation |
| `-tTrans FLOAT` | Transient time to discard before measurements (units: dt, **not implemented**) |
| `-tTotal FLOAT` | Total simulation time (units: dt) |
| `-dt FLOAT` | Integration time step |
| `-v_tract FLOAT` | Conduction speed along tracts |

### Oscillator parameters

| Option | Description |
|------|------------|
| `-omega FLOAT` | Mean natural frequency of oscillators |
| `-omega_range FLOAT` | Width of uniform distribution for natural frequencies |
| `-lmbda FLOAT` | Mean distance to Hopf bifurcation |
| `-lmbda_range FLOAT` | Width of uniform distribution for λ |
| `-alpha FLOAT` | Noise intensity |
| `-beta FLOAT` | Global coupling strength |

### Initial conditions and escape criterion

| Option | Description |
|------|------------|
| `-Z0 FLOAT` | Mean of the initial condition distribution |
| `-Z0_std FLOAT` | Width of uniform distribution for initial conditions |
| `-Z_amp_escape FLOAT` | Amplitude threshold used to define escape times |

---

## Input Data

Each input file (`txt` or `mat`) is supposed to have a subject code in its name. The code for controls is `ddd_d`, and the code for patients is `0ddd_d`.
The same code must be on the corresponding files for each matrix.
For example, for `txt` input, we can have `FL_301_1.txt`, `FN_301_1.txt`, `FMRI_301_1.txt` can be the names of the input files containing the fiber length (FL),
fiber number (FN) and rs-fMRI (FMRI) matrices for control subject identified by code `301_1`.

Similarly for MAT-file inputs, but then only one file with all matrices inside is needed. E.g., `mats_301_1.mat` can be the file containing all matrices
for MAT-file input. Each matrix must be identified by their corresponding parameters: `input_var_FL`, `input_var_FN`, `input_var_FMRI`.

The simulation can read connectome data in **two formats**:

### Text (`.txt`) input

```bash
-input_type txt
```

Required directories:
- `-input_dir_FL` : fiber length matrices
- `-input_dir_FN` : fiber number matrices
- `-input_dir_FMRI` : rs-fMRI correlation matrices

### MATLAB (`.mat`) input

```bash
-input_type mat
```

Required directory:
- `-input_dir_mat`

Required variable names inside `.mat` files:
- `-input_var_FL` : fiber length matrix
- `-input_var_FN` : fiber number matrix
- `-input_var_FMRI` : rs-fMRI correlation matrix

### Subject selection

| Option | Description |
|------|------------|
| `-input_code STR` | Code identifying the individual/subject to simulate |

---

## Output

| Option | Description |
|------|------------|
| `-outputFilePrefix PREFIX` | Prefix for output file names |
| `-savealltau` | Save escape times for all trials and all nodes (memory intensive) |
| `-writeOnRun` | Write output during the run (**not implemented**) |

⚠️ **Memory note**  
Using `-savealltau` can require significant memory  
(~150 MB for 60 subjects, 1000 trials, and 306 nodes).

---

## Coupling normalization

| Option | Description |
|------|------------|
| `-normalizecoupling` | If enabled, divides coupling strength β by the number of inputs to each node |

---

## Example

```bash
python escape_times.py \
    -ntrials 100 \
    -tTotal 200 \
    -dt 0.01 \
    -alpha 0.1 \
    -beta 0.2 \
    -omega 5.0 \
    -omega_range 0.5 \
    -lmbda 0.6 \
    -lmbda_range 0.06 \
    -Z_amp_escape 2.0 \
    -input_type mat \
    -input_code S001 \
    -input_dir_mat ./connectomes \
    -input_var_FL FL \
    -input_var_FN FN \
    -input_var_FMRI FC \
    -outputFilePrefix run01
```

---

## Notes

- The transient time option (`-tTrans`) and on-the-fly writing (`-writeOnRun`) are currently placeholders.
- Large simulations may require substantial memory, especially when saving all escape times.
- The first run may be slower if JIT compilation (e.g. via Numba) is used.
