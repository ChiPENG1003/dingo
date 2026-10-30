# injection_combo — coverage heat maps for overlapping BBH signals

How much does a second, unmodelled BBH in the same 8 s segment damage the
posterior the network recovers for the target signal?

One **target signal (signal 1)** is fixed for the whole study. A **distraction
signal (signal 2)** is added on top of it with a controlled time separation and
one controlled parameter; everything else about signal 2 is redrawn from the
model's own prior. Each grid cell is `N` such injections, and the cell's value is
the fraction of them whose *true* signal-1 parameter fell inside the recovered
posterior's credible interval.

Nothing in `../inference.py` is modified. Everything here imports from it.

## The grid

* **y axis, always**: `|Δt_c|`, ten log-spaced separations from 0.05 s to 1.0 s.
  Positive means signal 2 merges **earlier** than signal 1 (measured, not
  assumed — `select_target.py` prints the check).
* **x axis, one of four sweeps**:

  | sweep | what the bin fixes | SNR₂ |
  |---|---|---|
  | `snr_ratio` | SNR₁/SNR₂, 0.2 → 5 | solved exactly per realisation |
  | `luminosity_distance` | d_L,2, 100 → 2000 Mpc | uncontrolled — scatters, deliberately |
  | `chirp_mass` | ℳ₂, 10 → 65 M⊙ (q₂ pinned to q₁) | pinned to SNR₁ |
  | `mass_ratio` | q₂, 0.2 → 1.0 (ℳ₂ redrawn conditionally) | pinned to SNR₁ |

Five figures come out of four sweeps: `Δd_L = d_L,2 − d_L,1` is an exact affine
relabelling of the distance axis, since signal 1 never moves.

## Why the axis ranges are not the obvious ones

The prior puts `Constraint(10, 80) M⊙` on **both** component masses, which makes
most of the (ℳ, q) rectangle empty:

* ℳ cannot exceed `80 · 2^(-1/5) = 69.64` — the requested 10 → 80 axis would have
  ~3 permanently empty columns. At ℳ = 65 only `q ≥ 0.872` survives, at ℳ = 10
  only `q ≥ 0.755`, so a redrawn q would sweep *with* the axis and confound it.
  Hence `q₂ ≡ q₁` on that axis, and a ceiling set by q₁ (`select_target.py`
  reports it per seed).
* `q = 0.125` is reachable only at the single point (m₁, m₂) = (80, 10) M⊙, and
  `q = 0.15` admits only ℳ ∈ [20.8, 24.9]. Hence the q axis starts at 0.2, with
  ℳ₂ drawn from the prior truncated to each q's admissible window.

## Controls that make the cells comparable

* **Signal 1 is byte-identical everywhere**, optionally moved to an exact network
  SNR (`--target-snr1`, default 25) by the same 1/d solve used for signal 2.
* **Common random numbers**: realisation *k* uses the same nuisance seed and the
  same noise seed in every cell of every sweep, and the flow is reseeded per run,
  so two cells differ only through the data.
* **`geocent_time` of signal 2 is tied to signal 1's.** The prior has it uniform
  on ±0.1 s; left free it would smear the separation by ±0.2 s and merge the
  0.05 s and 0.1 s rows into each other.
* **SNR is always measured on the pure, already-time-shifted signal.** The shift
  is non-circular and truncates inspiral pushed out of the 8 s segment, so
  measuring before it would let the realised SNR drift along the y axis.

## Workflow

```bash
# 0. env (see the dingo-run-env notes)
module load GCCcore/12.3.0 Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export MPLBACKEND=Agg

# 1. pick signal 1 and check the grid is reachable  (CPU, login node is fine)
python select_target.py --model 1M_7params_O2 --seeds 0 1 2 3 4 5 6 7
python select_target.py --model 1M_7params_O2 --choose 1

# 2. the sweeps (one array task per cell; safe to re-run to finish a partial grid)
sbatch --array=0-99 submit_grid.sh snr_ratio
sbatch --array=0-99 submit_grid.sh chirp_mass
sbatch --array=0-99 submit_grid.sh mass_ratio
sbatch --array=0-99 submit_grid.sh luminosity_distance
sbatch             submit_grid.sh reference    # single-signal control

# 3. coverage grids
python summarize_grid.py snr_ratio
python summarize_grid.py snr_ratio --bin-by luminosity_distance
python summarize_grid.py snr_ratio --bin-by delta_distance
python summarize_grid.py chirp_mass
python summarize_grid.py mass_ratio

# 4. figures
python plot_heatmap.py results/snr_ratio/coverage_snr_ratio.npz \
    --parameters chirp_mass mass_ratio luminosity_distance geocent_time
```

## Reading the figures

**Compare each cell with its panel's no-intruder baseline, not with 0.90.** A
coverage test only reaches its nominal level when the truth is redrawn from the
prior on every trial. This study deliberately fixes ONE target signal, so even a
perfect sampler would not give 0.90 — the quantiles of a single fixed truth are
not uniform. Measured on the control (100 noise realisations, no distraction
signal, same seeds):

| parameter | 90% coverage | 50% coverage | median quantile |
|---|---|---|---|
| `chirp_mass` | 0.77 | 0.33 | 0.82 |
| `luminosity_distance` | 0.77 | 0.08 | 0.88 |
| `dec` | 0.80 | 0.27 | 0.14 |
| `geocent_time` | 0.93 | 0.34 | 0.81 |
| `ra` | 0.91 | 0.41 | 0.80 |
| `mass_ratio` | 1.00 | 0.54 | 0.75 |
| `psi` | 1.00 | 1.00 | 0.29 |

So the quantity the heat maps measure is **degradation from that baseline**.
`summarize_grid.py` stores the baseline in the `.npz` and `plot_heatmap.py`
prints it under each panel title; the colour bar deliberately carries no
"nominal" marker, because the right comparison value differs per parameter.

`psi` and `mass_ratio` sit at 1.00 even at the 50% level — their posteriors are
essentially prior-wide, so they carry no information and make poor panels. The
informative parameters are `chirp_mass`, `luminosity_distance`, `dec` and
`geocent_time`.

Red (→0) means the posterior is too narrow or displaced; green (→1) means too
*wide*. With 100 injections per cell the binomial uncertainty is about ±0.03, so
0.87 and 0.91 are the same number. Hatched cells are incomplete, grey cells never
ran, and `--cmap cividis` redraws the figure without a red–green distinction.

**Run the `reference` control before any sweep.** Without it there is no
comparison value at all, and `summarize_grid.py` will say so loudly.

## Files

| file | role |
|---|---|
| `combo_common.py` | paths, mass-constraint algebra, SNR + the 1/d distance solve |
| `grid_axes.py` | the four x axes and the rule turning a cell into signal 2 |
| `select_target.py` | choose signal 1; measure the shift sign; check reachability |
| `run_grid.py` | the driver (one cell / one column / a whole sweep) |
| `summarize_grid.py` | per-run records → coverage grids, incl. `--bin-by` re-binning |
| `plot_heatmap.py` | the faceted heat map figure |
| `submit_grid.sh` | SLURM array wrapper |
| `results/<sweep>/` | `config.json`, `cells/x**_y**.hdf5`, `coverage_*.npz` |
