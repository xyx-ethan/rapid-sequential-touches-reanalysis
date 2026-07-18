#!/usr/bin/env python3
"""Stress-test label-level summaries after merging every possible label pair."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def bh_adjust(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    output = np.full(array.shape, np.nan)
    finite = np.isfinite(array)
    testable = array[finite]
    if not len(testable):
        return pd.Series(output, index=values.index)
    order = np.argsort(testable)
    ranked = testable[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    output[finite] = restored
    return pd.Series(output, index=values.index)


def summarize(values: pd.Series) -> dict[str, float | int]:
    values = values.dropna().astype(float)
    nonzero = values[values != 0]
    if len(values) >= 5 and len(nonzero):
        rank_p = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
        positives = int((nonzero > 0).sum())
        sign_p = float(binomtest(positives, len(nonzero), 0.5, alternative="two-sided").pvalue)
    else:
        rank_p = np.nan
        positives = 0
        sign_p = np.nan
    return {
        "n_groups": int(len(values)),
        "median": float(values.median()),
        "n_positive": positives,
        "n_nonzero": int(len(nonzero)),
        "p_wilcoxon": rank_p,
        "p_sign": sign_p,
    }


def merge_group_medians(frame: pd.DataFrame, value: str, first: str, second: str) -> pd.Series:
    working = frame[["subject", value]].dropna().copy()
    merged_name = f"merged:{first}+{second}"
    working["group"] = working["subject"].replace({first: merged_name, second: merged_name})
    return working.groupby("group")[value].median()


def phase_results(results: Path, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    data = pd.read_csv(results / "phase_session_metrics.csv")
    primary = data[
        (data["response_window_ms"] == 50)
        & (data["minimum_neighbor_ms"] == 50)
        & (data["phase_basis"] == "two_harmonics")
        & (data["regularization_strength"] == 1.0)
        & (data["endpoint"].isin(["count", "any_spike"]))
        & (data["base_model"].isin(["context", "continuous_kinematics"]))
    ]
    keys = ["endpoint", "base_model"]
    rows = []
    for first, second in pairs:
        pair_rows = []
        for key, group in primary.groupby(keys, sort=False):
            summary = summarize(merge_group_medians(group, "delta_predictive_score", first, second))
            pair_rows.append({"merge_a": first, "merge_b": second, "endpoint": key[0], "base_model": key[1], **summary})
        pair_frame = pd.DataFrame(pair_rows)
        pair_frame["q_wilcoxon"] = bh_adjust(pair_frame["p_wilcoxon"])
        pair_frame["q_sign"] = bh_adjust(pair_frame["p_sign"])
        rows.append(pair_frame)
    return pd.concat(rows, ignore_index=True)


def history_results(results: Path, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    data = pd.read_csv(results / "history_window_session_metrics.csv")
    keys = ["analysis", "response_window_ms", "baseline_window_ms", "minimum_previous_interval_ms"]
    rows = []
    for first, second in pairs:
        pair_rows = []
        for key, group in data.groupby(keys, dropna=False, sort=False):
            summary = summarize(merge_group_medians(group, "beta_log_previous_interval", first, second))
            pair_rows.append(
                {
                    "merge_a": first,
                    "merge_b": second,
                    "analysis": key[0],
                    "response_window_ms": key[1],
                    "baseline_window_ms": key[2],
                    "minimum_previous_interval_ms": key[3],
                    **summary,
                }
            )
        pair_frame = pd.DataFrame(pair_rows)
        pair_frame["q_wilcoxon"] = pair_frame.groupby("analysis", group_keys=False)["p_wilcoxon"].apply(bh_adjust)
        rows.append(pair_frame)
    return pd.concat(rows, ignore_index=True)


def range_summary(phase: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    selections = [
        ("Phase count, context", phase[(phase.endpoint == "count") & (phase.base_model == "context")]),
        ("Phase count, measured kinematics", phase[(phase.endpoint == "count") & (phase.base_model == "continuous_kinematics")]),
        (
            "Post-touch count, 50 ms",
            history[(history.analysis == "post_only") & (history.response_window_ms == 50) & (history.minimum_previous_interval_ms == 0)],
        ),
        (
            "Baseline-subtracted rate, dense",
            history[(history.analysis == "baseline_subtracted_rate") & (history.response_window_ms == 50) & (history.baseline_window_ms == 50) & (history.minimum_previous_interval_ms == 0)],
        ),
        (
            "Baseline-subtracted rate, 100 ms",
            history[(history.analysis == "baseline_subtracted_rate") & (history.response_window_ms == 50) & (history.baseline_window_ms == 50) & (history.minimum_previous_interval_ms == 100)],
        ),
    ]
    rows = []
    for label, frame in selections:
        row = {
            "analysis": label,
            "n_pairings": len(frame),
            "n_groups_min": int(frame.n_groups.min()),
            "n_groups_max": int(frame.n_groups.max()),
            "median_min": frame["median"].min(),
            "median_max": frame["median"].max(),
            "q_wilcoxon_min": frame["q_wilcoxon"].min(),
            "q_wilcoxon_max": frame["q_wilcoxon"].max(),
        }
        if "q_sign" in frame:
            row["q_sign_min"] = frame["q_sign"].min()
            row["q_sign_max"] = frame["q_sign"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    labels = sorted(pd.read_csv(results / "eligible_touch_events_with_features.csv")["subject"].dropna().unique())
    pairs = list(itertools.combinations(labels, 2))
    phase = phase_results(results, pairs)
    history = history_results(results, pairs)
    summary = range_summary(phase, history)
    phase.to_csv(results / "label_pair_merge_phase_sensitivity.csv", index=False)
    history.to_csv(results / "label_pair_merge_history_sensitivity.csv", index=False)
    summary.to_csv(results / "label_pair_merge_range_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
