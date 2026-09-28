# Escape-times simulation tutorial

This guide walks through `escape_times.py`: preparing connectome inputs, choosing
parameter combinations, running a small test, reading the MATLAB output, and
plotting escape times. Run commands from the repository root.

## 1. Install and inspect the command

Use an environment with the packages imported by the simulation and plotting
examples:

```bash
python -m pip install numpy scipy numba networkx matplotlib
```

Check that the script starts and review all options:

```bash
python escape_times.py -h
```

The integration uses Euler-Maruyama. Results can be sensitive to `-dt`; use a
smaller step and re-check stability when trajectories grow rapidly. Numba
compiles parts of the integration on first use, so the first test can take
longer.

## 2. Prepare connectome inputs

All three matrices must be square, have the same dimensions, and use the same
region ordering:

| Matrix | Meaning in the simulation |
|---|---|
| `FL` | Fiber/tract lengths. Used to calculate a delay for each connection. |
| `FN` | Fiber numbers / structural weights. |
| `FMRI` | Functional correlation weights. |

Coupling is formed element-wise as `beta * FN * FMRI`. The delay matrix is
`round((FL / v_tract) / dt)` in integration steps. Keep the length and velocity
units consistent. Ensure the matrices contain finite values and that connected
tracts have sensible nonnegative lengths.

Subject codes are extracted from filenames using patterns such as `301_1` or
`0301_1`. Select one with `-input_code`; omit it to process all matching codes.

### MATLAB input (`-input_type mat`)

Put one `.mat` file per subject in a single directory. The filename must contain
the subject code, for example `mats_301_1.mat`. By default, the file must
contain variables named `M_FL`, `M_FN`, and `M_FMRI`; override those names when
your file uses different MATLAB variable names.

```bash
python escape_times.py \
  -input_type mat \
  -input_dir_mat ./connectomes \
  -input_code 301_1 \
  -input_var_FL M_FL \
  -input_var_FN M_FN \
  -input_var_FMRI M_FMRI
```

### Text input (`-input_type txt`)

Store each matrix type in its own directory. Filenames for the same subject
must carry the same code, for example:

```text
connectomes/
  FL/FL_301_1.txt
  FN/FN_301_1.txt
  FMRI/FMRI_301_1.txt
```

Each file is a whitespace-delimited numeric matrix readable by `numpy.loadtxt`.
Run it with:

```bash
python escape_times.py \
  -input_type txt \
  -input_dir_FL ./connectomes/FL \
  -input_dir_FN ./connectomes/FN \
  -input_dir_FMRI ./connectomes/FMRI \
  -input_code 301_1
```

Use plain ASCII directory names in practice; `-input_dir_FMRI` is the CLI
option name. The script checks all three directories and only pairs subject
codes that are present in all three input sets.

## 3. Understand the parameter combinations

Values below are the defaults in `modules/io.py`. Ranges for oscillator
heterogeneity are half-widths: each node is sampled uniformly from
`[mean - range, mean + range]`.

| Parameter | Default | How to choose it |
|---|---:|---|
| `-ntrials` | `10` | Independent stochastic trials per subject. Increase for more stable means and first-escape probabilities. |
| `-tTotal` | `200` | Total simulated model time. The integrator uses about `tTotal / dt` steps. Increase it if many nodes do not cross the threshold. |
| `-dt` | `0.01` | Integration time step. Smaller values improve temporal resolution and may improve stability, but cost more runtime and memory. |
| `-tTrans` | `50` | Placeholder only: currently validated but not used to discard an initial transient. |
| `-v_tract` | `5` | Conduction speed. Together with `FL` and `dt`, sets delays. |
| `-Z0` | `0` | Mean initial complex state. A real scalar sets the real-part mean; a complex value can set both components. |
| `-Z0_std` | `0.1` | Standard deviation used for independent normal draws of the initial real and imaginary parts. Zero makes initial conditions deterministic. |
| `-Z_amp_escape` | `1` | Escape threshold. For each node/trial, escape is the first sample where `abs(Z) > threshold` (strictly greater). |
| `-alpha` | `0.1` | Noise amplitude. The random increment is scaled by `alpha * sqrt(dt / 2)`. Set zero for deterministic dynamics. |
| `-beta` | `0.001` | Global multiplier on the element-wise structural * functional coupling. Its sign matters, and coupling can also be signed through `FMRI`. |
| `-omega` | `5` | Mean natural angular frequency. |
| `-omega_range` | `0.05` | Half-width of the uniform node-to-node frequency variation. |
| `-lmbda` | `0.6` | Mean Hopf parameter; the local linear coefficient is `(lmbda - 1) + i * omega`. |
| `-lmbda_range` | `0.10` | Half-width of uniform node-to-node variation in `lmbda`. |
| `-normalizecoupling` | off | Divide each row of the coupling matrix by that row's number of nonzero inputs. This reduces dependence on the number of inputs per node. |
| `-savealltau` | off | Save every trial x node escape time in `tau_samp`; can use substantial memory. |
| `-testrun` | off | Run a single subject/trial and include its full trajectory `Z` in the output for debugging. Output filename gets a `_test` suffix. |
| `-writeOnRun` | off | Placeholder only; writing during a run is not implemented. |
| `-outputFilePrefix` | `osc` | Output path/name prefix. The simulation adds a parameterized suffix and `.mat`. |

The `-tTotal` and `-dt` values are durations and step size in model-time units,
not counts of steps. `-tTrans` does not currently alter the measured escape
times.

For interpretable comparisons, vary one setting at a time. For example, compare
`-normalizecoupling` on/off with all other settings fixed; compare multiple
`-beta` values while preserving the sign convention; or change only `-alpha`
to isolate noise effects. `-omega_range` and `-lmbda_range` control
heterogeneity, whereas `-ntrials` controls repeated stochastic samples.

## 4. Make a tiny reproducible test connectome

The following creates equivalent four-region inputs for both supported formats.
It requires SciPy, already used by the simulation:

```python
from pathlib import Path
import numpy as np
from scipy.io import savemat

root = Path("example_connectomes")
for folder in ("mat", "FL", "FN", "FMRI"):
    (root / folder).mkdir(parents=True, exist_ok=True)

FL = np.array([
    [0, 0.5, 0,   0],
    [0.5, 0, 0.7, 0],
    [0, 0.7, 0,   0.4],
    [0, 0,   0.4, 0],
], dtype=float)
FN = np.array([
    [0, 2, 0, 0],
    [2, 0, 3, 0],
    [0, 3, 0, 1],
    [0, 0, 1, 0],
], dtype=float)
FMRI = np.array([
    [0, 0.4, 0,    0],
    [0.4, 0, 0.25, 0],
    [0, 0.25, 0,   0.3],
    [0, 0, 0.3,    0],
], dtype=float)

savemat(root / "mat" / "mats_301_1.mat",
        {"M_FL": FL, "M_FN": FN, "M_FMRI": FMRI})
for matrix, folder in ((FL, "FL"), (FN, "FN"), (FMRI, "FMRI")):
    np.savetxt(root / folder / f"{folder}_301_1.txt", matrix)
```

Save that snippet as `make_example_data.py` and run `python make_example_data.py`.
Then execute a short deterministic smoke run against the `.mat` input:

```bash
python escape_times.py \
  -ntrials 1 -tTotal 2 -dt 0.01 -tTrans 0 \
  -alpha 0 -beta 0.05 -omega 5 -omega_range 0 \
  -lmbda 0.6 -lmbda_range 0 \
  -Z0 0.2 -Z0_std 0 -Z_amp_escape 0.1 \
  -v_tract 5 -normalizecoupling -savealltau -testrun \
  -input_type mat -input_dir_mat ./example_connectomes/mat \
  -input_code 301_1 \
  -outputFilePrefix ./example_output/mat_smoke
```

The positive initial amplitude is already above the `0.1` threshold, so each
node's measured escape time should be `0`. The short run is intended to check
input loading, integration, escape-time calculation, and output writing, not to
produce scientifically meaningful estimates.

Repeat the same test with text inputs by replacing the input options:

```bash
python escape_times.py \
  -ntrials 1 -tTotal 2 -dt 0.01 -tTrans 0 \
  -alpha 0 -beta 0.05 -omega 5 -omega_range 0 \
  -lmbda 0.6 -lmbda_range 0 \
  -Z0 0.2 -Z0_std 0 -Z_amp_escape 0.1 \
  -v_tract 5 -normalizecoupling -savealltau -testrun \
  -input_type txt \
  -input_dir_FL ./example_connectomes/FL \
  -input_dir_FN ./example_connectomes/FN \
  -input_dir_FMRI ./example_connectomes/FMRI \
  -input_code 301_1 \
  -outputFilePrefix ./example_output/txt_smoke
```

Both runs save a `.mat` file. `-testrun` forces one trial and one subject and
adds the `Z` trajectory, even if another `-ntrials` value was passed. Existing
output names are not overwritten; a numeric suffix is added instead.

## 5. Read the output

Load a result with the repository helper:

```python
import glob
import modules.io as io

result_path = sorted(glob.glob("example_output/mat_smoke*.mat"))[-1]
result = io.load_escape_times_file(result_path)
print(result["codes"])
print(result["simParam"])

code = "301_1"
subject = result[code]
print(subject.tau_mean)  # mean escape time per node
print(subject.tau_std)   # across-trial standard deviation per node
print(subject.Pk)        # fraction of trials where each node escaped first
print(subject.tau_samp)  # trial x node values; populated with -savealltau
print(subject.Z.shape)   # time x node trajectory; populated with -testrun
```

`tau_mean` and `tau_std` are computed across trials. A node that never crosses
the threshold in a trial has a `NaN` escape time; NaNs can therefore propagate
to the summary for that node. If no node escapes in a trial, the current
first-escape calculation cannot select a first node. Increase `tTotal`, adjust
the threshold, or investigate the model/input settings rather than interpreting
missing escapes as zero.

## 6. Plot escape times

### Node-by-node mean and variability

This compact plot uses `tau_mean` and `tau_std`. The x-axis is the input node
order; label it with atlas region names only when you have verified that their
ordering matches the matrices:

```python
import numpy as np
import matplotlib.pyplot as plt

tau_mean = np.asarray(subject.tau_mean).reshape(-1)
tau_std = np.asarray(subject.tau_std).reshape(-1)
nodes = np.arange(tau_mean.size)

fig, ax = plt.subplots(figsize=(12, 4))
ax.errorbar(nodes, tau_mean, yerr=tau_std, fmt=".", markersize=5, capsize=2)
ax.set(xlabel="Node (connectome order)", ylabel="Escape time",
       title=f"Mean escape time +/- SD ({code})")
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()
```

When `-savealltau` was used, visualize all trial/node values as a heatmap:

```python
tau_samp = np.asarray(subject.tau_samp)
fig, ax = plt.subplots(figsize=(12, 4))
image = ax.imshow(tau_samp, aspect="auto", interpolation="nearest",
                  origin="lower", cmap="viridis")
ax.set(xlabel="Node (connectome order)", ylabel="Trial",
       title=f"Escape times ({code})")
fig.colorbar(image, ax=ax, label="Escape time")
fig.tight_layout()
plt.show()
```

Unescaped nodes remain `NaN` and appear as blank values in the heatmap.

### Trajectory plot from `hopf_oscillators_test.ipynb`

The test notebook loads a `_test` result and plots the real part of each
oscillator trajectory with a vertical offset per node. This reproduces that
example for up to the first 10 model-time units:

```python
import numpy as np
import matplotlib.pyplot as plt

Z = np.asarray(subject.Z)
dt = float(np.asarray(result["simParam"].dt).squeeze())
time = np.arange(Z.shape[0]) * dt
offset = 1.5

fig, ax = plt.subplots(figsize=(15, 7))
for node in range(Z.shape[1]):
    ax.plot(time, Z[:, node].real + node * offset, lw=1, alpha=0.8)
ax.set(xlim=(0, min(10, time[-1])), xlabel="Time", ylabel="Real(Z), offset by node",
       title=f"Oscillator trajectories ({code})")
fig.tight_layout()
plt.show()
```

This offset display is schematic: escape is calculated from the complex
magnitude `abs(Z)`, not from the plotted real component.

### AAL surface visualization

The referenced `plot_AAL_surface.ipynb` is not present in this repository
checkout, so its exact atlas-to-surface mapping and figure settings cannot be
reproduced here. A surface plot requires an ordered mapping from each simulation
node to the corresponding AAL region and surface vertices; do not assume that
an atlas label order automatically matches the connectome matrix order. Once
the mapping from that notebook is available, plot `subject.tau_mean` as the
per-region statistic (or `subject.Pk` for first-escape probability), preserving
the notebook's region order and masking `NaN` values.

The node-wise error-bar, trial heatmap, and offset-trajectory examples above
are directly reproducible using the output structure and examples in
`hopf_oscillators_test.ipynb`. For an anatomically aligned AAL figure, add the
missing notebook (or its atlas/surface mapping) to the repository and apply its
mapping to the same `tau_mean`/`Pk` arrays.

## 7. Practical checks

- Start with `-testrun` and one subject; remove it for a full run.
- Use `-savealltau` only when trial-level plots or analyses need `tau_samp`.
- Keep `-input_code` explicit when debugging a dataset.
- Check matrix dimensions, subject-code matching, and region order before
  interpreting a result.
- `-tTrans` and `-writeOnRun` are currently placeholders.
- Repeated runs use fresh stochastic draws; set `-alpha 0` and zero parameter
  ranges for a deterministic smoke test.
