import argparse
import os.path
import pickle
from io import StringIO
from os.path import join
from os.path import isfile
import glob
import copy

import numpy as np
import requests
import yaml

from gwpy.table import EventTable
from dingo.gw.noise.asd_dataset import ASDDataset, HIGH_ASD_VALUE

"""
Catalogue against which to check that no event is present in the estimated PSDs
"""
CATALOGS = ["GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident"]

"""
Contains links for PSD segment lists with quality label BURST_CAT2 from the Gravitational Wave Open Science Center.
Some events are split up into multiple chunks such that there are multiple URLs for one observing run
"""
URL_DIRECTORY = {
    "O1_L1": [
        "https://www.gw-openscience.org/timeline/segments/O1/L1_BURST_CAT2/1126051217/1137254417/"
    ],
    "O1_H1": [
        "https://www.gw-openscience.org/timeline/segments/O1/H1_BURST_CAT2/1126051217/1137254417/"
    ],
    "O2_L1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/L1_BURST_CAT2/1164556817/1187733618/"
    ],
    "O2_H1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/H1_BURST_CAT2/1164556817/1187733618/"
    ],
    "O2_V1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/V1_BURST_CAT2/1164556817/1187733618/"
    ],
    "O3_L1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/L1_BURST_CAT2/1238166018/1253977218/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/L1_BURST_CAT2/1256655618/1269363618/",
    ],
    "O3_H1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/H1_BURST_CAT2/1238166018/1253977218/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/H1_BURST_CAT2/1256655618/1269363618/",
    ],
    "O3_V1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/V1_BURST_CAT2/1238166018/1253977218/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/V1_BURST_CAT2/1256655618/1269363618/",
    ],
    "ET_ET1": [
        "https://raw.githubusercontent.com/bilby-dev/bilby/main/bilby/gw/detector/noise_curves/ET_D_asd.txt"
    ],
    "ET_ET2": [
        "https://raw.githubusercontent.com/bilby-dev/bilby/main/bilby/gw/detector/noise_curves/ET_D_asd.txt"
    ],
    "ET_ET3": [
        "https://raw.githubusercontent.com/bilby-dev/bilby/main/bilby/gw/detector/noise_curves/ET_D_asd.txt"
    ],
}

"""
Observing runs in URL_DIRECTORY whose URLs are design sensitivity curves (two columns:
frequency [Hz], ASD [1/sqrt(Hz)]) rather than GWOSC segment lists. For these, no strain
data are downloaded: the ASD is deterministic, so the dataset contains exactly one ASD
per detector, with GPS time 0.
"""
DESIGN_SENSITIVITY_RUNS = {"ET"}


def psd_data_path(data_dir, run, detector):
    """
    Return the directory where the PSD data is to be stored
    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation
    Returns
    -------
    the path where the data is stored
    """
    return join(data_dir, "tmp", run, detector)


def get_event_gps_times():
    event_list = []
    for catalogue in CATALOGS:
        try:
            events = EventTable.fetch_open_data(catalogue)
            event_list += list(events["GPS"])
        except Exception as e:
            print(
                f"Warning: Could not fetch event catalog '{catalogue}' from GWOSC "
                f"({type(e).__name__}: {e}). Skipping catalog — segments containing "
                f"events from this catalog will not be excluded."
            )
    return event_list


def get_time_segments(settings):
    """
    Creates a dictionary storing time segments used for estimating PSDs
    Parameters
    ----------
    settings : dict
        Settings that determine the segments
    Returns
    -------
    Dictionary containing the time segments for each detector
    """

    time_segments = {}

    run = settings["observing_run"]
    time_psd = settings["time_psd"]
    time_gap = settings["time_gap"]
    num_psds_max = settings.get("num_psds_max")

    if run in DESIGN_SENSITIVITY_RUNS:
        # A design sensitivity curve is deterministic, so there is exactly one "PSD
        # segment" per detector and no associated GPS time. The pseudo-segment keeps
        # the return format (and hence all downstream code) unchanged.
        return {det: [(0, time_psd)] for det in settings["detectors"]}

    event_list = get_event_gps_times()

    for detector in settings["detectors"]:

        key = run + "_" + detector
        urls = URL_DIRECTORY[key]

        starts, stops, durations = [], [], []
        for url in urls:
            r = requests.get(url, allow_redirects=True)
            c = StringIO(r.content.decode("utf-8"))
            starts_seg, stops_seg, durations_seg = np.loadtxt(
                c, dtype="int", unpack=True
            )
            starts = np.hstack([starts, starts_seg])
            stops = np.hstack([stops, stops_seg])
            durations = np.hstack([durations, durations_seg])

        segs = np.array(list(zip(starts, stops, durations)), dtype=int)
        segs = segs[segs[:, 2] >= time_psd]

        valid_segments = []
        for idx in range(segs.shape[0]):
            seg = segs[idx, :]
            start_time = seg[0]
            end_time = start_time + time_psd

            while end_time <= seg[1]:
                # check that no event is present in the segments
                if not any(start_time <= event <= end_time for event in event_list):
                    valid_segments.append((start_time, end_time))
                start_time = end_time + time_gap
                end_time = start_time + time_psd

        if num_psds_max is not None and num_psds_max > 0:
            valid_segments = valid_segments[:num_psds_max]

        time_segments[detector] = valid_segments

    return time_segments


def load_design_curve_asd(url, domain, cache_dir=None):
    """
    Download a design sensitivity curve and interpolate it onto a dingo domain.

    The file behind `url` must contain two whitespace-separated columns, frequency [Hz]
    and ASD [1/sqrt(Hz)] -- the format of the ET-D curve referenced in URL_DIRECTORY.
    Interpolation is done in log-log space. Frequencies not covered by the file (in
    particular f = 0) are set to HIGH_ASD_VALUE, the same convention ASDDataset uses
    for frequencies below f_min.

    Parameters
    ----------
    url : str
        Location of the two-column ASD text file, as stored in URL_DIRECTORY.
    domain : UniformFrequencyDomain
        Domain onto which the curve is interpolated.
    cache_dir : str
        If provided, the downloaded file is cached here so that repeated dataset
        generation does not require network access.

    Returns
    -------
    asd : np.ndarray
        ASD evaluated on domain.sample_frequencies, i.e. on the full grid [0, f_max].
    """
    contents = None
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = join(cache_dir, os.path.basename(url.split("?")[0]))
        if isfile(cache_file):
            with open(cache_file, "r") as f:
                contents = f.read()
    if contents is None:
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        contents = r.content.decode("utf-8")
        if cache_file is not None:
            with open(cache_file, "w") as f:
                f.write(contents)

    f_curve, asd_curve = np.loadtxt(StringIO(contents), unpack=True)
    order = np.argsort(f_curve)
    f_curve, asd_curve = f_curve[order], asd_curve[order]

    f = np.asarray(domain.sample_frequencies, dtype=np.float64)
    asd = np.full(len(f), HIGH_ASD_VALUE, dtype=np.float64)
    covered = (f >= f_curve[0]) & (f <= f_curve[-1])
    asd[covered] = np.exp(
        np.interp(np.log(f[covered]), np.log(f_curve), np.log(asd_curve))
    )
    return asd


def merge_datasets(asd_dataset_list):
    """
    Merges a list of asd datasets into ont
    Parameters
    ----------
    asd_dataset_list: List of ASDDatasets to be merged

    Returns
    -------
    A single combined ASDDataset object
    """

    merged_dict = {"asds": {}, "gps_times": {}}

    for det, asd_list in asd_dataset_list.items():
        print(f"Merging {len(asd_list)} datasets into one for detector {det}.")
        merged_dict["asds"][det] = np.vstack(
            [asd_dataset.asds[det] for asd_dataset in asd_list]
        )
        merged_dict["gps_times"][det] = np.hstack(
            [asd_dataset.gps_times[det] for asd_dataset in asd_list]
        )

    # copy settings from last dataset
    merged_dict["settings"] = copy.deepcopy(asd_list[-1].settings)

    merged = ASDDataset(dictionary=merged_dict)
    return merged


def merge_datasets_cli():
    """
    Command-line function to combine a collection of datasets into one. Used for
    parallelized ASD dataset generation.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path where the PSD data is to be stored.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Path to a settings file that contains the settings for the ASD dataset generation",
    )
    parser.add_argument(
        "--time_segments_file",
        type=str,
        default=None,
        help="Path to a file containing the time segments for which PSDs should be estimated",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=-1,
        help="Number of ASD datasets that should be merged",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="File name of merged dataset",
    )
    args = parser.parse_args()

    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    time_segments = None
    if args.time_segments_file:
        with open(args.time_segments_file, "rb") as f:
            time_segments = pickle.load(f)

    detectors = settings["dataset_settings"]["detectors"]
    observing_run = settings["dataset_settings"]["observing_run"]
    asd_dataset_list = {det: [] for det in detectors}

    for det in detectors:
        file_dir = psd_data_path(args.data_dir, observing_run, det)
        if time_segments:
            filenames = [
                join(file_dir, f"asd_{seg[0]}.hdf5")
                for seg in time_segments[det]
                if isfile(join(file_dir, f"asd_{seg[0]}.hdf5"))
            ]
        else:  # if no time_segments are specified, use the first 'num_parts' ASD datasets
            filenames = sorted(glob.glob(join(file_dir, f"asd_*.hdf5")))
            num_parts = min(args.num_parts, len(filenames)) if args.num_parts > 0 else len(filenames)
            filenames = filenames[:num_parts]

        asd_dataset_list[det] = [ASDDataset(filename) for filename in filenames]

    merged_dataset = merge_datasets(asd_dataset_list)
    filename = args.out_name
    if filename is None:
        run = settings["dataset_settings"]["observing_run"]
        filename = join(args.data_dir, f"asds_{run}.hdf5")
    merged_dataset.to_file(filename)
