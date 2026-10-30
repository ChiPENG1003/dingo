import argparse
import os

import numpy as np
import pycbc.psd
from gwpy.timeseries import TimeSeries

from dingo.core.dataset import DingoDataset
from dingo.gw.domains import UniformFrequencyDomain, build_domain
from dingo.gw.gwutils import get_window

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(HERE)
REAL_EVENT_DIR = os.path.join(
    EXAMPLE_DIR, "inference", "inference_result", "events"
)

F_MIN = 20.0
F_MAX = 1024.0
DELTA_F = 0.125
DETECTORS = ("H1", "L1")

F_S = 4096.0
ROLL_OFF = 0.4
TIME_PSD = 1024.0
TIME_BUFFER = 2.0


class RealEventData(DingoDataset):
    dataset_type = "real_event_data"

    def __init__(self, file_name=None, dictionary=None):
        self.context = None
        self.event_metadata = None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["context", "event_metadata"],
        )


def resolve_gps(event=None, gps=None):
    if gps is not None:
        return float(gps)
    if event is None:
        raise ValueError("Provide either --event or --gps.")

    from gwosc.datasets import event_gps

    t = float(event_gps(event))
    print(f"{event}: GPS trigger time {t}")
    return t


def _fetch(det, t_start, t_end, f_s):
    strain = TimeSeries.fetch_open_data(
        det, t_start, t_end, sample_rate=int(f_s), cache=True
    )
    return strain.astype(np.float64)


def download_event_strain(det, time_event, T, time_buffer, f_s, window):
    t_end = time_event + time_buffer
    t_start = t_end - T
    print(f"[{det}] downloading event strain [{t_start}, {t_end})")
    strain = _fetch(det, t_start, t_end, f_s)

    if np.any(np.isnan(strain.value)):
        raise ValueError(
            f"NaN in event strain for {det} at GPS {time_event}. The detector was "
            f"probably not observing over the full {T} s segment."
        )
    if len(strain) != len(window):
        raise ValueError(
            f"[{det}] strain has {len(strain)} samples but the window has "
            f"{len(window)}. Is --f-s wrong for this event?"
        )

    strain = (strain.to_pycbc() * window).to_frequencyseries()
    strain = strain.cyclic_time_shift(time_buffer)
    return np.array(strain)


def estimate_asd(det, time_event, T, time_buffer, f_s, window, time_psd):
    num_segments = int(round(time_psd / T))
    t_end = time_event + time_buffer - T
    t_start = t_end - T * num_segments
    print(f"[{det}] downloading PSD strain [{t_start}, {t_end})")
    psd_strain = _fetch(det, t_start, t_end, f_s)

    if np.any(np.isnan(psd_strain.value)):
        dt = int(np.ceil(np.where(np.isnan(psd_strain.value))[0][-1] / f_s))
        print(f"[{det}] NaN in PSD strain, shifting segment back by {dt} s.")
        psd_strain = _fetch(det, t_start - dt, t_end - dt, f_s)
        if np.any(np.isnan(psd_strain.value)):
            raise ValueError(f"NaN in PSD strain for {det} even after shifting.")

    psd = pycbc.psd.estimate.welch(
        psd_strain.to_pycbc(),
        seg_len=len(window),
        seg_stride=len(window),
        window=window,
        avg_method="median",
    )
    return np.array(psd) ** 0.5


def build_real_event(
    time_event,
    detectors=DETECTORS,
    domain=None,
    f_s=F_S,
    roll_off=ROLL_OFF,
    time_psd=TIME_PSD,
    time_buffer=TIME_BUFFER,
    event_name=None,
):
    if domain is None:
        domain = build_domain(
            {
                "type": "UniformFrequencyDomain",
                "f_min": F_MIN,
                "f_max": F_MAX,
                "delta_f": DELTA_F,
            }
        )
    if not isinstance(domain, UniformFrequencyDomain):
        raise NotImplementedError(f"Unsupported domain type {type(domain)}.")

    T = 1.0 / domain.delta_f
    if f_s / 2.0 < domain.f_max:
        raise ValueError(f"f_s={f_s} cannot represent f_max={domain.f_max}.")

    window = get_window({"type": "tukey", "roll_off": roll_off, "T": T, "f_s": f_s})

    context = {"waveform": {}, "asds": {}}
    for det in detectors:
        strain = download_event_strain(det, time_event, T, time_buffer, f_s, window)
        asd = estimate_asd(det, time_event, T, time_buffer, f_s, window, time_psd)

        context["waveform"][det] = domain.update_data(strain)
        context["asds"][det] = domain.update_data(asd, low_value=1.0)

        whitened = (
            context["waveform"][det][domain.min_idx :]
            / context["asds"][det][domain.min_idx :]
            / domain.noise_std
        )
        print(
            f"[{det}] whitened |d| per bin: mean {np.abs(whitened).mean():.3f}, "
            f"max {np.abs(whitened).max():.1f}"
        )

    event_metadata = {
        "time_event": float(time_event),
        "time_buffer": float(time_buffer),
        "time_psd": float(time_psd),
        "T": float(T),
        "f_s": float(f_s),
        "roll_off": float(roll_off),
        "window_type": "tukey",
        "detectors": list(detectors),
        "minimum_frequency": float(domain.f_min),
        "maximum_frequency": float(domain.f_max),
        "real_event": 1,
    }
    if event_name is not None:
        event_metadata["event_name"] = event_name

    return context, event_metadata


def save_real_event(context, event_metadata, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dataset = RealEventData(
        dictionary={
            "context": context,
            "event_metadata": event_metadata,
            "settings": event_metadata,
        }
    )
    dataset.to_file(out_path)
    print(f"Saved real-event context -> {out_path}")
    return out_path


def domain_from_model(model_path):
    """Read the domain a checkpoint was trained with (slow: loads the .pt)."""
    import torch

    from dingo.gw.domains import build_domain_from_model_metadata

    metadata = torch.load(model_path, map_location="cpu", weights_only=False)[
        "metadata"
    ]
    domain = build_domain_from_model_metadata(metadata)
    detectors = tuple(metadata["train_settings"]["data"]["detectors"])
    return domain, detectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=str, default="GW190814",
                        help="GWOSC event name, e.g. GW150914")
    parser.add_argument("--gps", type=float, default=None,
                        help="GPS trigger time; overrides --event lookup")
    parser.add_argument("--detectors", type=str, nargs="+", default=list(DETECTORS))
    parser.add_argument("--out", type=str, default=None,
                        help="Output HDF5 (default: inference/inference_result/events/<event>.hdf5)")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint to read the domain/detectors from instead of "
                             "the module constants")
    parser.add_argument("--f_s", type=float, default=F_S)
    parser.add_argument("--roll_off", type=float, default=ROLL_OFF)
    parser.add_argument("--time_psd", type=float, default=TIME_PSD)
    parser.add_argument("--time_buffer", type=float, default=TIME_BUFFER)
    args = parser.parse_args()

    domain = None
    detectors = tuple(args.detectors)
    if args.model is not None:
        domain, model_detectors = domain_from_model(args.model)
        if set(detectors) != set(model_detectors):
            print(f"Using detectors {detectors} (model was trained on {model_detectors}).")

    time_event = resolve_gps(event=args.event, gps=args.gps)
    label = args.event if args.gps is None else f"gps_{time_event:.3f}"

    context, event_metadata = build_real_event(
        time_event,
        detectors=detectors,
        domain=domain,
        f_s=args.f_s,
        roll_off=args.roll_off,
        time_psd=args.time_psd,
        time_buffer=args.time_buffer,
        event_name=label,
    )

    out_path = args.out or os.path.join(REAL_EVENT_DIR, f"{label}.hdf5")
    save_real_event(context, event_metadata, out_path)


if __name__ == "__main__":
    main()
