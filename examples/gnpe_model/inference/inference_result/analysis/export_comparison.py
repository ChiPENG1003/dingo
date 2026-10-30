#!/usr/bin/env python
import glob
import json
import os

import numpy as np
import pandas as pd
import h5py

from dingo.gw.result import Result
from dingo.core.utils.plotting import plot_corner_multi, get_latex_labels

# This file lives in inference_result/analysis/. Everything it reads and writes
# is a sibling of that directory, so anchor to inference_result rather than to
# the analysis folder itself.
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.dirname(HERE)
EVENTS_DIR = os.path.join(RESULT_ROOT, "events")        # observed strain (inputs)
TABLES_DIR = os.path.join(RESULT_ROOT, "tables")

# One Dingo injection and its matching bilby/dynesty baseline.
INJECTION = "seed0000_dprior"
DINGO_RESULT = os.path.join(
    RESULT_ROOT, "posteriors", "1M_7params_O2", "single",
    f"{INJECTION}.hdf5",
)
COMPARE_MODELS = [INJECTION]
BILBY_RUN_DIR = (
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/"
    f"bilby_injections/{INJECTION}"
)
BILBY_LABEL = "bilby"

TRUTH_JSON = os.path.join(
    RESULT_ROOT, "posteriors", "1M_7params_O2", "single",
    f"{INJECTION}.truth.json",
)
# Keep this injection's derived CSVs and figures separate from legacy outputs.
OUT_DIR = os.path.join(TABLES_DIR, INJECTION)
FIGURES_DIR = os.path.join(RESULT_ROOT, "figures", INJECTION)

# Internal / non-physical sample columns that must never enter the corner.
_NON_PHYSICAL = {
    "log_prob", "log_likelihood", "log_prior", "weights",
    "delta_log_prob_target", "H1_time_proxy", "L1_time_proxy",
}
# This 7-parameter model includes psi, but the requested Dingo-vs-bilby
# comparison is restricted to the six parameters below.
_PLOT_EXCLUDED = {"psi"}


def _compare_path(label):
    return DINGO_RESULT


def _find_bilby_result():
    for ext in (".hdf5", ".json"):
        path = os.path.join(BILBY_RUN_DIR, f"{BILBY_LABEL}_result{ext}")
        if os.path.exists(path):
            return path
    for ext in (".hdf5", ".json"):
        matches = sorted(glob.glob(os.path.join(BILBY_RUN_DIR, f"*_result{ext}")))
        if matches:
            return matches[0]
    return None


def load_truth():
    """Injected truth (relative geocent_time), matching the Dingo convention."""
    if os.path.exists(TRUTH_JSON):
        with open(TRUTH_JSON) as f:
            return json.load(f)["parameters"]
    # Fall back to the truth embedded in the reference HDF5.
    with h5py.File(_compare_path(COMPARE_MODELS[0]), "r") as h:
        if "injection_parameters" not in h:
            return None
        g = h["injection_parameters"]
        return {k: float(g[k][()]) for k in g.keys()}


def load_dingo_models():
    """Return [(label, DataFrame, weights_or_None, t_ref), ...] for each model."""
    out = []
    for label in COMPARE_MODELS:
        path = _compare_path(label)
        if not os.path.exists(path):
            print(f"  !! missing compare file for {label}: {path}; skipping.")
            continue
        result = Result(file_name=path)
        df = result.samples.copy()
        weights = df["weights"].to_numpy() if "weights" in df.columns else None
        out.append((label, df, weights, result.t_ref))
    return out


def load_bilby(t_ref):
    """bilby posterior DataFrame with geocent_time shifted to the Dingo convention."""
    path = _find_bilby_result()
    if path is None:
        print(f"bilby posterior  : not found in {BILBY_RUN_DIR}; skipping.")
        return None
    import bilby
    bres = bilby.result.read_in_result(filename=path)
    df = bres.posterior.copy()
    if "geocent_time" in df.columns:
        df["geocent_time"] = df["geocent_time"] - t_ref  # absolute GPS -> relative
    return df


def physical_params(df):
    return [c for c in df.columns
            if c not in _NON_PHYSICAL and not c.endswith("_time_proxy")]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    truth = load_truth()
    dingo = load_dingo_models()
    if not dingo:
        raise SystemExit("No Dingo compare posteriors found.")
    ref_label, ref_df, _, t_ref = dingo[0]
    ref_result = Result(file_name=_compare_path(ref_label))
    print(f"Reference model  : {ref_label}   t_ref = {t_ref}\n")

    bilby_df = load_bilby(t_ref)

    for label, df, _, _ in dingo:
        out = os.path.join(OUT_DIR, "dingo_posterior.csv")
        df[[c for c in df.columns
            if c not in {"H1_time_proxy", "L1_time_proxy"}]].to_csv(out, index=False)
        print(f"Dingo {label:>10}  : {len(df)} samples -> {out}")
    if bilby_df is not None:
        out = os.path.join(OUT_DIR, "bilby_posterior.csv")
        bilby_df[[c for c in bilby_df.columns
                  if c not in ("log_likelihood", "log_prior")]].to_csv(out, index=False)
        print(f"bilby posterior  : {len(bilby_df)} samples -> {out}")
    if truth is not None:
        pd.DataFrame([truth]).to_csv(os.path.join(OUT_DIR, "truth.csv"), index=False)
        print(f"Truth            : {len(truth)} params -> "
              f"{os.path.join(OUT_DIR, 'truth.csv')}")
    sample_frames = [df for _, df, _, _ in dingo]
    if bilby_df is not None:
        sample_frames.append(bilby_df)
    common = [p for p in physical_params(ref_df)
              if p not in _PLOT_EXCLUDED
              if all(p in df.columns for df in sample_frames)]
    print(f"\nOverlaying {len(sample_frames)} posteriors on {len(common)} params: "
          f"{common}")

    samples_list = [df[common] for _, df, _, _ in dingo]
    weights_list = [w for _, _, w, _ in dingo]
    labels = [lbl for lbl, _, _, _ in dingo]
    if bilby_df is not None:
        samples_list.append(bilby_df[common])
        weights_list.append(None)
        labels.append("bilby")

    kwargs = {"legend_font_size": 14}
    if truth is not None:
        kwargs["truths"] = [truth.get(p) for p in common]

    out = os.path.join(FIGURES_DIR, "dingo_vs_bilby_export_corner.png")
    plot_corner_multi(
        samples_list, weights=weights_list, labels=labels, filename=out,
        latex_labels_dict=get_latex_labels(ref_result.prior), **kwargs,
    )
    print(f"\nSaved overlaid corner ({len(samples_list)} posteriors) -> {out}")


if __name__ == "__main__":
    main()
