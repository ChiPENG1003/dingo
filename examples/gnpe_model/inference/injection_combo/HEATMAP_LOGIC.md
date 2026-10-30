# How one number on a heat map is produced

Read top to bottom; each stage names the code that does it. `verify_cell.py`
re-derives stages 1–4 from the raw stored samples, independently of the code
that wrote them.

```
theta1 (fixed, once per study)
   |
   +-- stage 0  pin SNR_1 by solving for d_L,1
   |
   v
for each cell (x = axis bin, y = time shift) and each realisation k:
   |
   +-- stage 1  theta2 = prior draw(seed k)  ->  axis applies its bin value
   |                                         ->  t_c tied to signal 1
   +-- stage 2  d_L,2 solved so SNR_2 hits its target      [controlled axes only]
   +-- stage 3  inject h1 + shift(h2) + noise  ->  sample  ->  truth quantile
   |
   v
stage 4  bin the runs, count coverage, colour the cell
```

## Stage 0 — the target signal, fixed for the entire study

`select_target.build_signal1` → `combo_common.distance_for_snr` (combo_common.py:194)

θ₁ is one prior draw (`--target-seed 1`), then moved to an exact network SNR by
solving for its distance. Result: ℳ₁ = 57.658, q₁ = 0.896, d_L,1 = 894.1 Mpc,
SNR₁ = 25.0000, t_c,1 = 0.007629.

The solve is exact, not iterative: dingo generates polarizations at a fiducial
100 Mpc (`prior.py:197`) and `ProjectOntoDetectors` multiplies by d_ref/d_new
before projecting (`detector_transforms.py:165`), so detector strain ∝ 1/d and
SNR(d) = k/d. One waveform evaluation fixes k.

**θ₁ never changes again** — not between cells, not between sweeps.

## Stage 1 — the distraction signal

`grid_axes.build_theta2` (grid_axes.py:226)

1. `inference._draw_reproducible_truth(ig, seed)` — a full prior draw (masses,
   spins, inclination, phase, sky, polarisation, distance).
2. `theta2["geocent_time"] = theta1["geocent_time"]` — **required**. The prior
   has t_c ~ U(−0.1, 0.1) s; left free it smears the separation by ±0.2 s and
   merges the 0.05 s and 0.1 s rows into each other.
3. `axis.apply(...)` — overwrites whatever the axis controls.

**Seeding is the load-bearing part.** `seed = nuisance_base + k + axis.replica_offset(x)`.

| sweep type | offset | effect |
|---|---|---|
| controlled (`snr_ratio`, `chirp_mass`, …) | 0 | realisation *k* is the **same** nuisance draw in every column — differences between columns come from the axis alone (common random numbers) |
| `population` | `x * 10000` | the ten columns are **independent** samples; there is no axis to hold fixed, and reusing the offset would make ten identical copies |

Noise seed follows the same rule (`run_grid.py:177`), and torch is reseeded per
run so the flow draws identical base samples in matching cells.

## Stage 2 — pinning SNR₂ (controlled axes only)

`combo_common.distance_for_snr`, via `projected_signal` (combo_common.py:168)

The signal is projected **and time-shifted first**, then its SNR is measured,
then the distance is rescaled. The order matters: `_linear_time_shift` is
non-circular and truncates inspiral pushed out of the 8 s segment (a ℳ=10
system carries ~4.7 s of band from 20 Hz), so measuring before the shift would
leave the realised SNR drifting along the y axis — the very axis being
controlled for.

- `snr_ratio` axis: target = SNR₁ / (bin centre)
- `chirp_mass`, `mass_ratio` axes: target = SNR₁ (equally loud intruder), so the
  mass axes carry no hidden loudness effect
- `population`, `luminosity_distance` axes: **no pinning** — SNR₂ is whatever
  the draw gives, and is recorded as a derived quantity

Verified exact: `verify_cell.py snr_ratio --cell 0 0` reports
|SNR₂ − target| < 5e-14.

## Stage 3 — inject, sample, score

`run_grid.run_cell` (run_grid.py:135) → `inference._inject_and_infer`

Data = h₁ + shift(h₂) + coloured Gaussian noise, using the fixed O2 ASD
(`asds[0]`). 1000 posterior samples, 30 GNPE iterations.

Scoring, `_truth_quantiles` (run_grid.py:96): for each inferred parameter, store
**the fraction of posterior samples below the true value**. Not a verdict — the
quantile. That is what lets the credible level be chosen later at plotting time
without re-running anything.

Stored per run (run_grid.py:112): the samples themselves (float32, gzip), every
θ₂ value, SNR₂ per detector, the SNR target, and the quantiles. The strain is
**not** stored — it is reproducible from the seeds and would cost ~24 GB.

## Stage 4 — from runs to a cell colour

`summarize_grid.coverage_grid` (summarize_grid.py:139)

Covered ⟺ the truth's quantile lies inside ((1−C)/2, (1+C)/2). Cell value =
n_covered / n_total, where **n_total is the number of runs that actually
finished** — never a hard-coded 100.

Two ways to define the columns:

- the sweep's own axis (controlled): cells *are* the bins, edges from `config.json`
- `--bin-by <quantity>` (derived): the runs are re-histogrammed by any recorded
  quantity — `derived_edges` (summarize_grid.py:113) gives `quantile` (equal
  count: column widths show where the population lives) or `log` (fixed
  log-spaced: every column physically distinct, unequal counts)

`reference_coverage` (summarize_grid.py:86) reads the single-signal control and
stores it in the `.npz`; `plot_heatmap` prints it under each panel title.

## The `mean` grid

`summarize_grid.py --mean` adds a pseudo-parameter `mean`: the arithmetic mean
of the per-parameter coverage grids. Every parameter shares the same `n_total`
in a cell, so this equals pooling `n_covered` — no weighting is hidden.

⚠️ **`psi` and `mass_ratio` are dead weight in the average.** Their posteriors
are essentially prior-wide: `psi` sits at coverage 1.000 with a *minimum of
0.990* across the whole grid — it does not respond to the intruder at all. Each
such parameter contributes a constant 1.0, putting a floor of (n_dead / n_total)
under every cell:

| averaged over | worst cell | baseline |
|---|---|---|
| all 7 | 0.216 | 0.88 |
| 5 (no `psi`, no `mass_ratio`) | 0.096 | 0.84 |

`--mean-over chirp_mass luminosity_distance geocent_time ra dec` gives the
undiluted version. The threshold structure is identical either way; only the
contrast changes.

## The comparison value is NOT 0.90

A coverage test reaches its nominal level only when the truth is redrawn from
the prior every trial. This study fixes ONE target, so even a perfect sampler
would not give 0.90 — the quantiles of a fixed truth are not uniform. Measured
on the control (100 runs, no distraction signal): ℳ 0.77, d_L 0.77, dec 0.80,
t_c 0.93, ra 0.91, and q / psi at 1.00 because their posteriors are essentially
prior-wide.

⚠️ **The control is only 100 runs → ±0.042.** The population sweep's safe region
sits at 0.82–0.88, ~1.7σ away. Either extend the control to 1000 runs (~1.8
GPU·h) or use the population sweep's high-ratio column as the baseline (~1000
runs, ±0.012, same noise and parameter distribution as the cells).

## Plotting

`plot_heatmap.py` — one figure per scored parameter; each panel is titled by its
own x axis, so title and axis always name the same thing.

- fixed 0–1 colour scale (`RdYlGn`), so every figure uses one ruler
- no "nominal" marker on the bar: the right comparison differs per parameter and
  is printed under each panel title
- derived axes get equal-width cells with the real edges on the ticks, because
  equal-count bins have wildly unequal ranges
- incomplete cells hatched, empty cells grey; the threshold is
  `combo_common.short_cell_threshold`, shared with the summarizer so the console
  and the figure cannot disagree

## Known caveats

1. **d₂ leaves the trained prior at extreme ratios.** Ratio 0.2 needs d₂ ≈ 49 Mpc
   (prior is U(100, 2000)). Physically fine — signal 2 is contaminating data the
   network never has to fit — but recorded, not hidden.
2. **Mass axes are constrained.** ℳ ≤ 69.64 (both components ≤ 80 M⊙), and the
   admissible q window collapses at both ends, which is why q₂ is pinned to q₁.
   q = 0.125 is a single point, so the q axis starts at 0.2.
3. **Shift direction was measured, not assumed** — `+0.5 s` moves the merger
   0.5000 s **earlier** (`verify_shift_direction`). inference.py's `--shift`
   help text says "later" and is wrong; nothing here relies on it.
4. **One target signal only.** Everything is conditional on this ℳ₁, q₁, sky
   position and SNR. Nothing here generalises to a population of targets.

## Auditing

```bash
python verify_cell.py snr_ratio  --cell 0 0 --recompute-snr
python verify_cell.py population --cell 0 0
```

Checks: t_c tie, SNR₂ vs its target, quantiles recomputed from the raw samples,
and the cell's coverage recounted. `--recompute-snr` regenerates the waveforms
and re-measures SNR₂ from scratch.
