# analysis/

Everything here reads the results under `inference_result/` and writes to
`inference_result/figures/` or `.../tables/`. Nothing here runs inference.

## Start here

```bash
python plot_posteriors.py --list        # what can be plotted, and what has a bilby baseline
python plot_posteriors.py GW170809      # then plot one
```

`--list` prints the exact target names, which models have a posterior for each,
and whether a bilby baseline exists. Everything else follows from it.

## The scripts

| script | use it when |
|---|---|
| **`plot_posteriors.py`** | **the default.** One or more Dingo posteriors for one event/injection, optionally overlaid on a bilby baseline. Corner + 1D marginals + a median/90% CI table. |
| `plot_single_corner.py` | **overlapping-signal work only.** Draws the second (distraction) signal's truth in its own style and labels curves by their time shift `Δt` — neither of which `plot_posteriors.py` knows about. Driven by `overlapping_corner_plot.sh`. |
| `plot_marginals.py` | 1D marginals across a *sweep* (the `series/` runs), one parameter per panel. Answers "what did the distraction do to the posterior", which a corner buries. |
| `stack_figures.py` | pastes finished PNGs into one panel. Reads no data on purpose — each row keeps its own axes and legend. |
| `coverage_stats.py` | credible-interval coverage over a population run. |
| `export_comparison.py` | dumps posteriors to CSV under `tables/`. |
| `batch_read.py`, `read_data.py` | bulk loading helpers. |

## plot_posteriors.py

Replaced `plot_real_event_corner.py`, `make_corner.py` and
`plot_corner_dingo_vs_bilby.py`. Those differed only in how they found their
input and which time reference they assumed; the plotting was the same code
three times over.

```bash
# compare two networks on one real event, against bilby
python plot_posteriors.py GW170809 --model chi_eff 1M_7params_O2

# an injection: the truth is read from the file and drawn as a black cross
python plot_posteriors.py seed0053_dprior --model ET \
    -p chirp_mass mass_ratio chi_eff luminosity_distance

# Dingo only
python plot_posteriors.py GW170809 --bilby none

# explicit files, bypassing discovery (e.g. inference.py's compare mode,
# which writes outside the <model>/<kind>/ layout)
python plot_posteriors.py --dingo a.hdf5 b.hdf5 --bilby run/bilby_result.json
```

Outputs `<target>_<models>_dingo_corner.png`,
`<target>_<models>_dingo_vs_bilby.png`, `<target>_<models>_dingo_vs_bilby_1d.png`,
where `<models>` is the model labels joined with `+`. The tag is what keeps a
two-model comparison from overwriting the single-model figure of the same
target — and one single-model run from overwriting another.

### Two things it handles that are easy to get wrong

**geocent_time.** Dingo stores it *relative* to a reference; bilby stores
absolute GPS. The reference is `Result.t_ref`, which is the trigger time for a
real event and the training `ref_time` for an injection. Reading
`event_metadata["time_event"]` directly — as `plot_real_event_corner.py` did —
returns `None` for an injection, leaving bilby's time unshifted: a ~1.1e9 s
error that produced a wrong figure with no exception.

**Mismatched parameter sets.** Models infer different parameters (`chi_eff` has
`chi_eff` but no `psi`; `1M_7params_O2` the reverse). The figure is reduced to
the intersection and says which parameters it dropped, so an overlay never
implies a comparison that isn't there.

## Naming trap

`plot_marginals.py` imports `LATEX` and `density_curve` from
**`bilby_baseline/plot_corner.py`** via a `sys.path` insert, deliberately, so
both sides use the same density estimate. Do not add a `plot_corner.py` to this
directory — it would shadow that import. (This is why the merged script is
`plot_posteriors.py`.)
