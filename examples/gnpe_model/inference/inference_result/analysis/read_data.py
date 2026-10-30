import os
import time
import argparse
from pathlib import Path

import coverage_stats as cs
from batch_read import (
    INJECTIONS_DIR,
    list_injection_files,
    load_injection_truth,
    load_injection_posterior,
)

# This file lives in inference_result/analysis/. Everything it reads and writes
# is a sibling of that directory, so anchor to inference_result rather than to
# the analysis folder itself.
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.dirname(HERE)
EVENTS_DIR = os.path.join(RESULT_ROOT, "events")        # observed strain (inputs)
LEGACY_DIR = os.path.join(RESULT_ROOT, "legacy")        # results from before the reorg
FIGURES_DIR = os.path.join(RESULT_ROOT, "figures")
TABLES_DIR = os.path.join(RESULT_ROOT, "tables")
Path(TABLES_DIR).mkdir(exist_ok=True)
Path(FIGURES_DIR).mkdir(exist_ok=True)
CACHE_FILE = os.path.join(TABLES_DIR, "dingo_analysis.csv")
SUMMARY_TXT = os.path.join(TABLES_DIR, "dingo_coverage_summary.txt")
SUMMARY_CSV = os.path.join(TABLES_DIR, "dingo_coverage_summary.csv")
RATIO_HIST_FILE = os.path.join(FIGURES_DIR, "ratio_histograms.png")


def _tag(label, n_samples):
    """Filename suffix identifying which population run a product came from.

    Without a label the names stay exactly as they were, so an existing cache
    from the original (LVK) population run is still found.
    """
    parts = []
    if label:
        parts.append(label)
    if n_samples is not None:
        parts.append(f"n{n_samples}")
    return ("_" + "_".join(parts)) if parts else ""


def _suffixed(path, tag):
    stem, ext = os.path.splitext(path)
    return f"{stem}{tag}{ext}"


def cache_file(n_samples, label=""):
    return _suffixed(CACHE_FILE, _tag(label, n_samples))


def run_coverage_analysis(injections_dir: str = INJECTIONS_DIR, n_samples=None):
    files = list_injection_files(injections_dir)
    records = []
    for i, path in enumerate(files):
        index = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])

        truth = load_injection_truth(path)
        posterior = load_injection_posterior(path)
        rows = cs.analyze_posterior(posterior, truth, n_samples=n_samples)
        records.extend(cs.rows_to_records(index, rows))
        print(f"  processing {i + 1}/{len(files)} injections", flush=True)
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  Already processed {i + 1}/{len(files)} injections", flush=True)
    return records


def get_analysis(rebuild: bool = False, n_samples=None,
                 injections_dir: str = INJECTIONS_DIR, label: str = ""):
    cf = cache_file(n_samples, label)
    if os.path.exists(cf) and not rebuild:
        print(f"Loading cached analysis from {cf}")
        try:
            return cs.load_records(cf)
        except ValueError as exc:
            print(f"{exc} Rebuilding it.")
    print(f"Reading injections from {injections_dir} ...")
    records = run_coverage_analysis(injections_dir=injections_dir, n_samples=n_samples)
    return cs.save_records(records, cf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-read all injection files instead of using the cache.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Subsample each posterior to this many samples before "
                         "reducing (match bilby's per-injection count). Default: all.")
    ap.add_argument("--injections-dir", default=INJECTIONS_DIR,
                    help="Directory of injection_*.hdf5 from a `multi` run. "
                         "Default: the original LVK population run.")
    ap.add_argument("--label", default="",
                    help="Name of this population run (e.g. ET). Appended to the "
                         "cache, summary and figure filenames so runs on different "
                         "networks do not overwrite each other.")
    ap.add_argument("-p", "--parameters", nargs="+", default=None,
                    help="Restrict the RATIO HISTOGRAM to these parameters. The "
                         "relative error is only meaningful where the truth is "
                         "positive and bounded away from zero, so the default set "
                         "of eight is rarely what you want to look at. The coverage "
                         "table is always computed over every parameter.")
    args = ap.parse_args()

    t0 = time.time()
    df = get_analysis(rebuild=args.rebuild, n_samples=args.n_samples,
                      injections_dir=args.injections_dir, label=args.label)

    coverage = cs.coverage_from_df(df)
    ratios = cs.ratios_from_df(df)
    n = cs.n_injections_from_df(df)

    tag = _tag(args.label, args.n_samples)
    title = "Dingo 90% mode-centred region coverage"
    if args.label:
        title = f"{title} [{args.label}]"

    summary = cs.summarize_coverage(coverage, n)
    cs.save_coverage_summary(summary, n,
                             _suffixed(SUMMARY_TXT, tag),
                             _suffixed(SUMMARY_CSV, tag),
                             title=title)

    if args.parameters:
        missing = [p for p in args.parameters if p not in ratios]
        if missing:
            raise SystemExit(
                f"No such parameter(s) in this run: {missing}. "
                f"Available: {sorted(ratios)}"
            )
        ratios = {p: ratios[p] for p in args.parameters}
        print(f"\nRatio histogram restricted to {args.parameters}")
    cs.plot_ratio_histograms(ratios, _suffixed(RATIO_HIST_FILE, tag))

    print(f"\nDone in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
