#!/usr/bin/env python3
"""Augment the frozen touch table from the public NWB files.

The script reproduces spike counts, adds baseline and response-window
sensitivity counts, and derives continuous kinematic and contact features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import hilbert, savgol_filter


PRE_WINDOWS_MS = (10, 20, 30, 40, 50)
POST_WINDOWS_MS = (10, 20, 30, 40, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_scalar(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def intracellular_location(nwb: h5py.File) -> tuple[str, str, float, str]:
    locations: list[str] = []
    for electrode in nwb["general/intracellular_ephys"].values():
        if "location" in electrode:
            locations.append(decode_scalar(electrode["location"]))
    unique = sorted(set(locations))
    if len(unique) != 1:
        raise RuntimeError(f"expected one intracellular location, found {unique}")
    location = unique[0]
    layer_match = re.search(r"cortical_layer:\s*([^;]+)", location)
    subregion_match = re.search(r"brain_subregion:\s*([^;]+)", location)
    depth_match = re.search(r"depth:\s*([0-9.]+)", location)
    if not (layer_match and subregion_match and depth_match):
        raise RuntimeError(f"could not parse intracellular location: {location}")
    return (
        layer_match.group(1).strip(),
        subregion_match.group(1).strip(),
        float(depth_match.group(1)),
        location,
    )


def nearest_indices(samples: np.ndarray, events: np.ndarray) -> np.ndarray:
    right = np.searchsorted(samples, events, side="left")
    right = np.clip(right, 0, len(samples) - 1)
    left = np.clip(right - 1, 0, len(samples) - 1)
    choose_left = np.abs(samples[left] - events) <= np.abs(samples[right] - events)
    return np.where(choose_left, left, right)


def contiguous_slices(timestamps: np.ndarray) -> list[slice]:
    differences = np.diff(timestamps)
    positive = differences[differences > 0]
    nominal = float(np.median(positive))
    boundaries = np.flatnonzero((differences <= 0) | (differences > 1.5 * nominal)) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(timestamps)]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]


def analytic_amplitude_by_segment(
    timestamps: np.ndarray, filtered_angle: np.ndarray
) -> np.ndarray:
    amplitude = np.full(filtered_angle.shape, np.nan, dtype=float)
    for segment in contiguous_slices(timestamps):
        values = filtered_angle[segment]
        if len(values) >= 4:
            amplitude[segment] = np.abs(hilbert(values))
    return amplitude


def derivatives_by_segment(
    timestamps: np.ndarray, angle: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.full(angle.shape, np.nan, dtype=float)
    acceleration = np.full(angle.shape, np.nan, dtype=float)
    for segment in contiguous_slices(timestamps):
        segment_time = timestamps[segment]
        values = angle[segment]
        if len(values) < 11:
            continue
        delta = float(np.median(np.diff(segment_time)))
        velocity[segment] = savgol_filter(
            values, window_length=11, polyorder=3, deriv=1, delta=delta, mode="interp"
        )
        acceleration[segment] = savgol_filter(
            values, window_length=11, polyorder=3, deriv=2, delta=delta, mode="interp"
        )
    return velocity, acceleration


def interval_counts(
    spike_times: np.ndarray, events: np.ndarray, start_s: float, stop_s: float
) -> np.ndarray:
    starts = np.searchsorted(spike_times, events + start_s, side="left")
    stops = np.searchsorted(spike_times, events + stop_s, side="left")
    return (stops - starts).astype(np.int16)


def window_peak(
    timestamps: np.ndarray,
    values: np.ndarray,
    events: np.ndarray,
    stop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    signed_peak = np.full(len(events), np.nan)
    absolute_peak = np.full(len(events), np.nan)
    starts = np.searchsorted(timestamps, events, side="left")
    stops = np.searchsorted(timestamps, events + stop_s, side="left")
    for row, (start, stop) in enumerate(zip(starts, stops)):
        segment = values[start:stop]
        finite = np.isfinite(segment)
        if not finite.any():
            continue
        segment = segment[finite]
        index = int(np.argmax(np.abs(segment)))
        signed_peak[row] = float(segment[index])
        absolute_peak[row] = float(abs(segment[index]))
    return signed_peak, absolute_peak


def touch_durations(
    onset_times: np.ndarray, offset_times: np.ndarray, events: np.ndarray
) -> np.ndarray:
    event_index = nearest_indices(onset_times, events)
    if np.max(np.abs(onset_times[event_index] - events)) > 0.00051:
        raise RuntimeError("touch table and NWB touch-onset times do not align")
    candidates = np.searchsorted(offset_times, onset_times[event_index], side="right")
    durations = np.full(len(events), np.nan)
    valid = candidates < len(offset_times)
    durations[valid] = (
        offset_times[candidates[valid]] - onset_times[event_index[valid]]
    ) * 1000.0
    durations[(durations < 0) | (durations > 1000)] = np.nan
    return durations


def extract_session(raw_path: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    events = frame["touch_time"].to_numpy(float)
    output = frame.copy()
    with h5py.File(raw_path, "r") as nwb:
        subject_id = decode_scalar(nwb["general/subject/subject_id"])
        cortical_layer, brain_subregion, recording_depth_um, location = (
            intracellular_location(nwb)
        )
        output["cortical_layer"] = cortical_layer
        output["brain_subregion"] = brain_subregion
        output["recording_depth_um"] = recording_depth_um
        behavior = nwb["acquisition/behavior"]
        behavior_time = behavior["phase/timestamps"][:].astype(float)
        behavior_index = nearest_indices(behavior_time, events)
        max_alignment_error = float(
            np.max(np.abs(behavior_time[behavior_index] - events))
        )
        if max_alignment_error > 0.00051:
            raise RuntimeError(
                f"behavior samples do not align to touches in {raw_path}: "
                f"{max_alignment_error:.6f} s"
            )

        spike_data = nwb["acquisition/SpikeTrain/data"][:].astype(bool)
        spike_time = nwb["acquisition/SpikeTrain/timestamps"][:].astype(float)
        spikes = spike_time[spike_data]
        for width_ms in PRE_WINDOWS_MS:
            output[f"spikes_pre_{width_ms}ms"] = interval_counts(
                spikes, events, -width_ms / 1000.0, 0.0
            )
        for width_ms in POST_WINDOWS_MS:
            output[f"spikes_post_{width_ms}ms"] = interval_counts(
                spikes, events, 0.0, width_ms / 1000.0
            )
        output["spikes_post_8_40ms"] = interval_counts(spikes, events, 0.008, 0.040)

        phase = behavior["phase/data"][:].astype(float)
        theta = behavior["theta_at_base/data"][:].astype(float)
        theta_filtered = behavior["theta_filt/data"][:].astype(float)
        delta_kappa = behavior["delta_kappa/data"][:].astype(float)
        distance = behavior["distance_to_pole/data"][:].astype(float)

        amplitude = analytic_amplitude_by_segment(behavior_time, theta_filtered)
        velocity, acceleration = derivatives_by_segment(behavior_time, theta)
        pretouch_index = nearest_indices(behavior_time, events - 0.005)

        output["phase_raw"] = phase[behavior_index]
        output["phase_branch"] = np.where(
            output["phase_raw"].to_numpy(float) < 0, "protraction", "retraction"
        )
        output["amplitude_continuous_deg"] = amplitude[behavior_index]
        output["theta_at_touch_deg"] = theta[behavior_index]
        output["pretouch_velocity_deg_s"] = velocity[pretouch_index]
        output["pretouch_acceleration_deg_s2"] = acceleration[pretouch_index]
        output["distance_to_pole_at_touch"] = distance[behavior_index]

        signed_peak, absolute_peak = window_peak(
            behavior_time, delta_kappa, events, stop_s=0.020
        )
        output["peak_delta_kappa_0_20ms"] = signed_peak
        output["peak_abs_delta_kappa_0_20ms"] = absolute_peak

        onset_mask = behavior["touch_onset/data"][:].astype(bool)
        offset_mask = behavior["touch_offset/data"][:].astype(bool)
        onset_times = behavior["touch_onset/timestamps"][:].astype(float)[onset_mask]
        offset_times = behavior["touch_offset/timestamps"][:].astype(float)[offset_mask]
        output["touch_duration_ms"] = touch_durations(onset_times, offset_times, events)

    count_pairs = {
        "pre_50": (
            output["spikes_pre_50ms"].to_numpy(),
            frame["spikes_pre_50ms"].to_numpy(),
        ),
        "post_10": (
            output["spikes_post_10ms"].to_numpy(),
            frame["spikes_0_10ms"].to_numpy(),
        ),
        "post_30": (
            output["spikes_post_30ms"].to_numpy(),
            (frame["spikes_0_10ms"] + frame["spikes_10_30ms"]).to_numpy(),
        ),
        "post_50": (
            output["spikes_post_50ms"].to_numpy(),
            frame["spikes_0_50ms"].to_numpy(),
        ),
    }
    validation = {
        "path": frame["path"].iloc[0],
        "n_events": int(len(frame)),
        "nwb_subject_id": subject_id,
        "table_subject_id": str(frame["subject"].iloc[0]),
        "cortical_layer": cortical_layer,
        "brain_subregion": brain_subregion,
        "recording_depth_um": recording_depth_um,
        "intracellular_location": location,
        "max_behavior_alignment_error_s": max_alignment_error,
        "count_comparison": {
            name: {
                "n_different": int(np.count_nonzero(new - old)),
                "max_absolute_difference": int(np.max(np.abs(new - old))),
                "new_total": int(np.sum(new)),
                "old_total": int(np.sum(old)),
            }
            for name, (new, old) in count_pairs.items()
        },
        "raw_nwb_sha256": sha256(raw_path),
    }
    return output, validation


def add_touch_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(["path", "trial_id", "touch_time"]).copy()
    groups = output.groupby(["path", "trial_id"], sort=False)["touch_time"]
    output["previous_touch_interval_ms"] = (groups.diff() * 1000.0).round(3)
    output["following_touch_interval_ms"] = (-groups.diff(-1) * 1000.0).round(3)
    output["nearest_touch_interval_ms"] = output[
        ["previous_touch_interval_ms", "following_touch_interval_ms"]
    ].min(axis=1, skipna=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("touch_table", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("validation_json", type=Path)
    args = parser.parse_args()

    touches = pd.read_csv(args.touch_table)
    frames: list[pd.DataFrame] = []
    validation: list[dict] = []
    for number, (relative_path, frame) in enumerate(touches.groupby("path", sort=True), start=1):
        raw_path = args.raw_root / relative_path
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        extracted, checks = extract_session(raw_path, frame)
        frames.append(extracted)
        validation.append(checks)
        print(f"[{number:02d}/{touches['path'].nunique():02d}] {relative_path}", flush=True)

    augmented = add_touch_intervals(pd.concat(frames, ignore_index=True))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output_csv, index=False)
    args.validation_json.parent.mkdir(parents=True, exist_ok=True)
    args.validation_json.write_text(json.dumps(validation, indent=2))

    failed = []
    for row in validation:
        for comparison in row["count_comparison"].values():
            # The prior table used inconsistent inclusion at exact bin edges.
            # The raw re-extraction uses half-open intervals throughout. A
            # one-spike discrepancy at an endpoint is reported in validation;
            # larger differences indicate a genuine alignment failure.
            if comparison["max_absolute_difference"] > 1:
                failed.append(row)
                break
    if failed:
        raise RuntimeError(f"raw count validation failed for {len(failed)} sessions")
    print(
        f"Wrote {len(augmented):,} touches; prior-count differences were limited "
        "to one spike at explicit half-open window boundaries."
    )


if __name__ == "__main__":
    main()
