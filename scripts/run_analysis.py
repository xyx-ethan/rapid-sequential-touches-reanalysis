#!/usr/bin/env python3
"""Run the manuscript analyses and sensitivity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image as PILImage
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WINDOWS_MS = (10, 20, 30, 40, 50)
MIN_INTERVALS_MS = (0, 20, 30, 50, 100, 250)
ICI_EDGES = (0, 20, 30, 50, 100, 250, np.inf)
ICI_LABELS = ("<20", "20-29", "30-49", "50-99", "100-249", "≥250")

BLUE = "#24557A"
LIGHT_BLUE = "#8FB3CF"
OCHRE = "#B07D24"
GRAY = "#6E7781"
LIGHT_GRAY = "#D9DEE3"
PAIR_GRAY = "#78858F"
BLACK = "#1B1D1F"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def bh_q(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    p = values[finite]
    if not len(p):
        return output
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    output[finite] = restored
    return output


def bootstrap_interval(
    values: np.ndarray, statistic=np.median, iterations: int = 5000, seed: int = 20260717
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations)
    for index in range(iterations):
        boot[index] = statistic(rng.choice(values, len(values), replace=True))
    return tuple(np.percentile(boot, [2.5, 97.5]))


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[
        frame["time_from_pole_in"].between(0, 2.0)
        & (frame["stim_present"].fillna(0).astype(int) == 0)
        & frame["phase_raw"].notna()
        & (frame["cortical_layer"].astype(str) == "4")
    ].copy()
    data["log_touch_number"] = np.log1p(data["touch_number"].astype(float))
    data["first_touch"] = (data["touch_number"] == 1).astype(float)
    data["log_previous_interval"] = np.log(
        data["previous_touch_interval_ms"].clip(lower=1)
    )
    data["log_amplitude"] = np.log1p(data["amplitude_continuous_deg"])
    data["signed_log_pretouch_velocity"] = np.sign(
        data["pretouch_velocity_deg_s"]
    ) * np.log1p(np.abs(data["pretouch_velocity_deg_s"]))
    data["signed_log_pretouch_acceleration"] = np.sign(
        data["pretouch_acceleration_deg_s2"]
    ) * np.log1p(np.abs(data["pretouch_acceleration_deg_s2"]))
    data["signed_log_curvature"] = np.sign(data["peak_delta_kappa_0_20ms"]) * np.log1p(
        np.abs(data["peak_delta_kappa_0_20ms"]) * 1000.0
    )
    data["log_abs_curvature"] = np.log1p(
        data["peak_abs_delta_kappa_0_20ms"] * 1000.0
    )
    data["log_touch_duration"] = np.log1p(data["touch_duration_ms"])
    data["session_time"] = data.groupby("path")["touch_time"].transform(
        lambda values: values - values.min()
    )
    data["is_go"] = (data["trial_type"].astype(str) == "Go").astype(float)

    phase = data["phase_raw"].to_numpy(float)
    retraction = (phase >= 0).astype(float)
    branch_progress = np.where(
        phase < 0, (phase + np.pi) / np.pi, phase / np.pi
    )
    data["phase_retraction"] = retraction
    data["phase_protraction_progress"] = (1 - retraction) * branch_progress
    data["phase_protraction_progress2"] = (1 - retraction) * branch_progress**2
    data["phase_retraction_progress"] = retraction * branch_progress
    data["phase_retraction_progress2"] = retraction * branch_progress**2
    data["phase_sin1"] = np.sin(phase)
    data["phase_cos1"] = np.cos(phase)
    data["phase_sin2"] = np.sin(2 * phase)
    data["phase_cos2"] = np.cos(2 * phase)
    for term in ("phase_sin1", "phase_cos1", "phase_sin2", "phase_cos2"):
        data[f"{term}_x_log_amplitude"] = data[term] * data["log_amplitude"]
    return data


CONTEXT_FEATURES = [
    "log_touch_number",
    "first_touch",
    "log_previous_interval",
    "pole_position",
    "time_from_pole_in",
    "session_time",
]

KINEMATIC_FEATURES = [
    "log_amplitude",
    "signed_log_pretouch_velocity",
    "signed_log_pretouch_acceleration",
    "distance_to_pole_at_touch",
    "theta_at_touch_deg",
]

CONTACT_DESCRIPTION_FEATURES = [
    "signed_log_curvature",
    "log_abs_curvature",
    "log_touch_duration",
]

ALL_KINEMATIC_AND_CONTACT_FEATURES = KINEMATIC_FEATURES + CONTACT_DESCRIPTION_FEATURES

PIECEWISE_PHASE = [
    "phase_retraction",
    "phase_protraction_progress",
    "phase_protraction_progress2",
    "phase_retraction_progress",
    "phase_retraction_progress2",
]

FOURIER_1 = ["phase_sin1", "phase_cos1"]
FOURIER_2 = ["phase_sin1", "phase_cos1", "phase_sin2", "phase_cos2"]
AMPLITUDE_PHASE_INTERACTIONS = [
    f"{term}_x_log_amplitude" for term in FOURIER_2
]


def oof_prediction(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    endpoint: str,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    groups = frame["trial_id"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        return None
    prediction = np.full(len(frame), np.nan)
    null_prediction = np.full(len(frame), np.nan)
    splits = min(5, len(unique_groups))
    for train, test in GroupKFold(n_splits=splits).split(frame, groups=groups):
        estimator = (
            PoissonRegressor(alpha=alpha, max_iter=1000)
            if endpoint == "count"
            else LogisticRegression(C=1.0 / alpha, max_iter=1000)
        )
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
        model.fit(frame.iloc[train][features], frame.iloc[train][target])
        if endpoint == "count":
            prediction[test] = np.clip(model.predict(frame.iloc[test][features]), 1e-9, None)
            null_prediction[test] = max(float(frame.iloc[train][target].mean()), 1e-9)
        else:
            prediction[test] = np.clip(
                model.predict_proba(frame.iloc[test][features])[:, 1], 1e-9, 1 - 1e-9
            )
            null_prediction[test] = np.clip(
                float(frame.iloc[train][target].mean()), 1e-9, 1 - 1e-9
            )
    return prediction, null_prediction


def predictive_score(
    target: np.ndarray,
    prediction: np.ndarray,
    null_prediction: np.ndarray,
    endpoint: str,
) -> float:
    if endpoint == "count":
        loss = mean_poisson_deviance(target, prediction)
        null_loss = mean_poisson_deviance(target, null_prediction)
    else:
        loss = log_loss(target, prediction, labels=[0, 1])
        null_loss = log_loss(target, null_prediction, labels=[0, 1])
    return float(1.0 - loss / null_loss) if null_loss > 0 else np.nan


def phase_session_metric(
    frame: pd.DataFrame,
    base_features: list[str],
    phase_features: list[str],
    target: str,
    endpoint: str,
    alpha: float = 1.0,
) -> dict | None:
    if (
        len(frame) < 80
        or (frame["phase_raw"] < 0).sum() < 15
        or (frame["phase_raw"] >= 0).sum() < 15
    ):
        return None
    target_values = frame[target].to_numpy(float)
    if np.nanstd(target_values) < 1e-9:
        return None
    base_output = oof_prediction(frame, base_features, target, endpoint, alpha)
    phase_output = oof_prediction(
        frame, base_features + phase_features, target, endpoint, alpha
    )
    if base_output is None or phase_output is None:
        return None
    base_prediction, base_null = base_output
    phase_prediction, phase_null = phase_output
    base_score = predictive_score(target_values, base_prediction, base_null, endpoint)
    phase_score = predictive_score(target_values, phase_prediction, phase_null, endpoint)
    return {
        "path": frame["path"].iloc[0],
        "subject": frame["subject"].iloc[0],
        "n_events": int(len(frame)),
        "n_protraction": int((frame["phase_raw"] < 0).sum()),
        "n_retraction": int((frame["phase_raw"] >= 0).sum()),
        "base_predictive_score": base_score,
        "phase_predictive_score": phase_score,
        "delta_predictive_score": float(phase_score - base_score),
    }


def summarize_session_metric(
    sessions: pd.DataFrame, family: dict[str, object]
) -> dict[str, object]:
    if sessions.empty or "subject" not in sessions:
        return {
            **family,
            "n_sessions": 0,
            "n_subjects": 0,
            "n_modeled_events": 0,
            "median_delta_predictive_score": np.nan,
            "mean_delta_predictive_score": np.nan,
            "n_positive": 0,
            "n_nonzero": 0,
            "p_two_sided": np.nan,
            "p_sign_two_sided": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    values = (
        sessions.groupby("subject")["delta_predictive_score"]
        .median()
        .dropna()
        .to_numpy(float)
    )
    if len(values) < 5:
        return {
            **family,
            "n_sessions": int(len(sessions)),
            "n_subjects": int(len(values)),
            "n_modeled_events": int(sessions["n_events"].sum()),
            "median_delta_predictive_score": np.nan,
            "mean_delta_predictive_score": np.nan,
            "n_positive": int(np.sum(values > 0)),
            "n_nonzero": int(np.sum(values != 0)),
            "p_two_sided": np.nan,
            "p_sign_two_sided": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    low, high = bootstrap_interval(values, np.median)
    nonzero = values[values != 0]
    n_positive = int(np.sum(nonzero > 0))
    sign_p = (
        float(stats.binomtest(n_positive, len(nonzero), 0.5).pvalue)
        if len(nonzero)
        else np.nan
    )
    return {
        **family,
        "n_sessions": int(len(sessions)),
        "n_subjects": int(len(values)),
        "n_modeled_events": int(sessions["n_events"].sum()),
        "median_delta_predictive_score": float(np.median(values)),
        "mean_delta_predictive_score": float(np.mean(values)),
        "n_positive": n_positive,
        "n_nonzero": int(len(nonzero)),
        "p_two_sided": float(stats.wilcoxon(values).pvalue),
        "p_sign_two_sided": sign_p,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def run_phase_family(
    data: pd.DataFrame,
    response_window_ms: int,
    minimum_neighbor_ms: int,
    endpoint: str,
    base_name: str,
    phase_name: str,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = data[
        (data["previous_touch_interval_ms"].fillna(np.inf) >= minimum_neighbor_ms)
        & (data["following_touch_interval_ms"].fillna(np.inf) >= minimum_neighbor_ms)
    ].copy()
    target = f"spikes_post_{response_window_ms}ms"
    if endpoint == "any_spike":
        target = f"any_spike_post_{response_window_ms}ms"
        selected[target] = (
            selected[f"spikes_post_{response_window_ms}ms"] > 0
        ).astype(float)
    if base_name == "continuous_kinematics":
        base_features = CONTEXT_FEATURES + KINEMATIC_FEATURES
    elif base_name == "kinematics_and_contact_descriptions":
        base_features = CONTEXT_FEATURES + ALL_KINEMATIC_AND_CONTACT_FEATURES
    else:
        base_features = CONTEXT_FEATURES
    phase_features = {
        "piecewise_branches": PIECEWISE_PHASE,
        "first_harmonic": FOURIER_1,
        "two_harmonics": FOURIER_2,
    }[phase_name]
    rows = []
    for _, session in selected.groupby("path", sort=True):
        metric = phase_session_metric(
            session, base_features, phase_features, target, endpoint, alpha=alpha
        )
        if metric is not None:
            rows.append(metric)
    sessions = pd.DataFrame(rows)
    family = {
        "response_window_ms": response_window_ms,
        "minimum_neighbor_ms": minimum_neighbor_ms,
        "endpoint": endpoint,
        "base_model": base_name,
        "phase_basis": phase_name,
        "regularization_strength": alpha,
        "n_selected_events": int(len(selected)),
    }
    return sessions, summarize_session_metric(sessions, family)


def run_amplitude_phase_interaction(
    data: pd.DataFrame, alpha: float = 1.0
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Test whether the continuous phase profile varies with whisking amplitude."""
    selected = data[
        (data["previous_touch_interval_ms"].fillna(np.inf) >= 50)
        & (data["following_touch_interval_ms"].fillna(np.inf) >= 50)
    ].copy()
    base_features = CONTEXT_FEATURES + KINEMATIC_FEATURES + FOURIER_2
    rows = []
    for _, session in selected.groupby("path", sort=True):
        metric = phase_session_metric(
            session,
            base_features,
            AMPLITUDE_PHASE_INTERACTIONS,
            "spikes_post_50ms",
            "count",
            alpha=alpha,
        )
        if metric is not None:
            rows.append(metric)
    sessions = pd.DataFrame(rows)
    family = {
        "analysis": "continuous_amplitude_by_phase_interaction",
        "response_window_ms": 50,
        "minimum_neighbor_ms": 50,
        "regularization_strength": alpha,
        "n_selected_events": int(len(selected)),
    }
    return sessions, summarize_session_metric(sessions, family)


def standardized_ols_coefficient(
    frame: pd.DataFrame, features: list[str], target: str, coefficient: str
) -> tuple[float, int]:
    matrix = frame[features].to_numpy(float)
    response = frame[target].to_numpy(float)
    valid = np.isfinite(response) & np.all(np.isfinite(matrix), axis=1)
    if valid.sum() < 80 or np.nanstd(response[valid]) < 1e-9:
        return np.nan, int(valid.sum())
    matrix = matrix[valid]
    response = response[valid]
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-9] = 1.0
    standardized = (matrix - means) / scales
    response = (response - response.mean()) / response.std()
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(standardized)), standardized]),
        response,
        rcond=None,
    )
    return float(coefficients[features.index(coefficient) + 1]), int(valid.sum())


HISTORY_CONTROLS = [
    "log_previous_interval",
    "log_touch_number",
    *KINEMATIC_FEATURES,
    "pole_position",
    "time_from_pole_in",
    "session_time",
    *FOURIER_2,
]


def history_coefficient_summary(
    selected: pd.DataFrame, target: str, metadata: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for _, session in selected.groupby("path", sort=True):
        coefficient, n_used = standardized_ols_coefficient(
            session, HISTORY_CONTROLS, target, "log_previous_interval"
        )
        if np.isfinite(coefficient):
            rows.append(
                {
                    "path": session["path"].iloc[0],
                    "subject": session["subject"].iloc[0],
                    "beta_log_previous_interval": coefficient,
                    "n_events": n_used,
                }
            )
    sessions = pd.DataFrame(rows)
    values = (
        sessions.groupby("subject")["beta_log_previous_interval"]
        .median()
        .dropna()
        .to_numpy(float)
    )
    low, high = bootstrap_interval(values, np.median)
    summary = {
        **metadata,
        "n_selected_events": int(len(selected)),
        "n_sessions": int(len(sessions)),
        "n_subjects": int(len(values)),
        "median_beta": float(np.median(values)) if len(values) else np.nan,
        "mean_beta": float(np.mean(values)) if len(values) else np.nan,
        "n_positive": int(np.sum(values > 0)),
        "p_two_sided": (
            float(stats.wilcoxon(values).pvalue) if len(values) >= 5 else np.nan
        ),
        "ci_low": float(low),
        "ci_high": float(high),
    }
    return sessions, summary


def run_history_windows(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    later = data[data["previous_touch_interval_ms"].notna()].copy()
    summaries = []
    session_frames = []
    for response_window in WINDOWS_MS:
        for minimum_previous in MIN_INTERVALS_MS:
            selected = later[
                (later["previous_touch_interval_ms"] >= minimum_previous)
                & (
                    later["following_touch_interval_ms"].fillna(np.inf)
                    >= response_window
                )
            ].copy()
            target = f"spikes_post_{response_window}ms"
            sessions, summary = history_coefficient_summary(
                selected,
                target,
                {
                    "analysis": "post_only",
                    "response_window_ms": response_window,
                    "baseline_window_ms": np.nan,
                    "minimum_previous_interval_ms": minimum_previous,
                },
            )
            sessions = sessions.assign(**summary)
            session_frames.append(sessions)
            summaries.append(summary)

    for baseline_window in WINDOWS_MS:
        for minimum_previous in MIN_INTERVALS_MS:
            selected = later[
                (later["previous_touch_interval_ms"] >= minimum_previous)
                & (later["following_touch_interval_ms"].fillna(np.inf) >= 50)
            ].copy()
            target = f"rate_difference_baseline_{baseline_window}ms"
            selected[target] = selected["spikes_post_50ms"] / 0.050 - selected[
                f"spikes_pre_{baseline_window}ms"
            ] / (baseline_window / 1000.0)
            sessions, summary = history_coefficient_summary(
                selected,
                target,
                {
                    "analysis": "baseline_subtracted_rate",
                    "response_window_ms": 50,
                    "baseline_window_ms": baseline_window,
                    "minimum_previous_interval_ms": minimum_previous,
                },
            )
            sessions = sessions.assign(**summary)
            session_frames.append(sessions)
            summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame["q_within_analysis"] = np.nan
    for _, indices in summary_frame.groupby("analysis").groups.items():
        summary_frame.loc[indices, "q_within_analysis"] = bh_q(
            summary_frame.loc[indices, "p_two_sided"].to_numpy(float)
        )
    return pd.concat(session_frames, ignore_index=True), summary_frame


def baseline_response_window_sensitivity(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    later = data[data["previous_touch_interval_ms"].notna()].copy()
    summaries = []
    session_frames = []
    for response_window in WINDOWS_MS:
        for baseline_window in WINDOWS_MS:
            for minimum_previous in (0, 100):
                selected = later[
                    (later["previous_touch_interval_ms"] >= minimum_previous)
                    & (
                        later["following_touch_interval_ms"].fillna(np.inf)
                        >= response_window
                    )
                ].copy()
                target = (
                    f"rate_difference_response_{response_window}ms_"
                    f"baseline_{baseline_window}ms"
                )
                selected[target] = selected[
                    f"spikes_post_{response_window}ms"
                ] / (response_window / 1000.0) - selected[
                    f"spikes_pre_{baseline_window}ms"
                ] / (baseline_window / 1000.0)
                sessions, summary = history_coefficient_summary(
                    selected,
                    target,
                    {
                        "analysis": "baseline_response_window_sensitivity",
                        "response_window_ms": response_window,
                        "baseline_window_ms": baseline_window,
                        "minimum_previous_interval_ms": minimum_previous,
                    },
                )
                sessions = sessions.assign(**summary)
                session_frames.append(sessions)
                summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame["q_sensitivity"] = bh_q(
        summary_frame["p_two_sided"].to_numpy(float)
    )
    return pd.concat(session_frames, ignore_index=True), summary_frame


def run_kinematic_coefficients(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = data[
        (data["previous_touch_interval_ms"].fillna(np.inf) >= 50)
        & (data["following_touch_interval_ms"].fillna(np.inf) >= 50)
    ].copy()
    features = CONTEXT_FEATURES + ALL_KINEMATIC_AND_CONTACT_FEATURES + FOURIER_2
    rows = []
    for _, session in selected.groupby("path", sort=True):
        for feature in ALL_KINEMATIC_AND_CONTACT_FEATURES:
            coefficient, n_used = standardized_ols_coefficient(
                session, features, "spikes_post_50ms", feature
            )
            if np.isfinite(coefficient):
                rows.append(
                    {
                        "path": session["path"].iloc[0],
                        "subject": session["subject"].iloc[0],
                        "feature": feature,
                        "standardized_beta": coefficient,
                        "n_events": n_used,
                    }
                )
    sessions = pd.DataFrame(rows)
    subject_values = (
        sessions.groupby(["subject", "feature"])["standardized_beta"]
        .median()
        .reset_index()
    )
    summaries = []
    for feature, group in subject_values.groupby("feature", sort=False):
        values = group["standardized_beta"].to_numpy(float)
        feature_sessions = sessions[sessions["feature"] == feature]
        low, high = bootstrap_interval(values, np.median)
        summaries.append(
            {
                "feature": feature,
                "n_subjects": int(len(values)),
                "n_sessions": int(feature_sessions["path"].nunique()),
                "n_complete_events": int(feature_sessions["n_events"].sum()),
                "median_beta": float(np.median(values)),
                "mean_beta": float(np.mean(values)),
                "n_positive": int(np.sum(values > 0)),
                "p_two_sided": float(stats.wilcoxon(values).pvalue),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["q"] = bh_q(summary["p_two_sided"].to_numpy(float))
    return sessions, subject_values, summary


def interval_bin_profile(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side, interval, other_interval in (
        ("previous", "previous_touch_interval_ms", "following_touch_interval_ms"),
        ("following", "following_touch_interval_ms", "previous_touch_interval_ms"),
    ):
        selected = data[
            data[interval].notna()
            & (data[other_interval].fillna(np.inf) >= 50)
        ].copy()
        selected["interval_bin"] = pd.cut(
            selected[interval],
            bins=ICI_EDGES,
            labels=ICI_LABELS,
            right=False,
        )
        event_counts = selected.groupby("interval_bin", observed=True).size()
        session_means = (
            selected.groupby(["path", "subject", "interval_bin"], observed=True)[
                "spikes_post_50ms"
            ]
            .mean()
            .reset_index()
        )
        subject_means = (
            session_means.groupby(["subject", "interval_bin"], observed=True)[
                "spikes_post_50ms"
            ]
            .median()
            .reset_index()
        )
        for interval_bin, group in subject_means.groupby("interval_bin", observed=True):
            values = group["spikes_post_50ms"].to_numpy(float)
            low, high = bootstrap_interval(values, np.mean)
            frames.append(
                {
                    "side": side,
                    "interval_bin": str(interval_bin),
                    "n_events": int(event_counts.loc[interval_bin]),
                    "n_subjects": int(len(values)),
                    "mean_spikes_per_touch": float(np.mean(values)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    return pd.DataFrame(frames)


def bilateral_isolation_profile(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for minimum_interval in MIN_INTERVALS_MS:
        selected = data[
            (data["previous_touch_interval_ms"].fillna(np.inf) >= minimum_interval)
            & (
                data["following_touch_interval_ms"].fillna(np.inf)
                >= minimum_interval
            )
        ].copy()
        session_means = (
            selected.groupby(["path", "subject"])["spikes_post_50ms"]
            .mean()
            .reset_index()
        )
        subject_means = session_means.groupby("subject")[
            "spikes_post_50ms"
        ].median()
        values = subject_means.to_numpy(float)
        low, high = bootstrap_interval(values, np.mean)
        rows.append(
            {
                "minimum_interval_to_both_neighbors_ms": minimum_interval,
                "n_events": int(len(selected)),
                "n_files": int(selected["path"].nunique()),
                "n_subjects": int(len(values)),
                "mean_spikes_per_touch": float(np.mean(values)),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def observed_neighbor_isolation_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Repeat bilateral isolation after requiring an observed neighbor on each side."""
    observed = data[
        data["previous_touch_interval_ms"].notna()
        & data["following_touch_interval_ms"].notna()
    ]
    rows = []
    for minimum_interval in MIN_INTERVALS_MS:
        selected = observed[
            (observed["previous_touch_interval_ms"] >= minimum_interval)
            & (observed["following_touch_interval_ms"] >= minimum_interval)
        ]
        session_means = (
            selected.groupby(["path", "subject"])["spikes_post_50ms"]
            .mean()
            .reset_index()
        )
        subject_means = session_means.groupby("subject")["spikes_post_50ms"].median()
        values = subject_means.to_numpy(float)
        low, high = bootstrap_interval(values, np.mean)
        rows.append(
            {
                "minimum_interval_to_both_observed_neighbors_ms": minimum_interval,
                "n_events": int(len(selected)),
                "n_files": int(selected["path"].nunique()),
                "n_subjects": int(len(values)),
                "mean_spikes_per_touch": float(np.mean(values)),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def phase_profiles(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data[
        (data["previous_touch_interval_ms"].fillna(np.inf) >= 50)
        & (data["following_touch_interval_ms"].fillna(np.inf) >= 50)
    ].copy()
    selected["phase_bin"] = pd.cut(
        selected["phase_raw"],
        bins=np.linspace(-np.pi, np.pi, 13),
        labels=False,
        include_lowest=True,
        right=False,
    )
    session_rows = []
    for _, session in selected.groupby("path", sort=True):
        if (
            len(session) < 80
            or session["trial_id"].nunique() < 4
            or (session["phase_raw"] < 0).sum() < 15
            or (session["phase_raw"] >= 0).sum() < 15
        ):
            continue
        target = "spikes_post_50ms"
        context_output = oof_prediction(session, CONTEXT_FEATURES, target, "count")
        kinematic_output = oof_prediction(
            session, CONTEXT_FEATURES + KINEMATIC_FEATURES, target, "count"
        )
        if context_output is None or kinematic_output is None:
            continue
        context_prediction, _ = context_output
        kinematic_prediction, _ = kinematic_output
        response_sd = float(session[target].std())
        if not np.isfinite(response_sd) or response_sd < 1e-9:
            continue
        working = session[["path", "subject", "phase_bin", target]].copy()
        working["context_residual"] = (
            session[target].to_numpy(float) - context_prediction
        ) / response_sd
        working["kinematic_residual"] = (
            session[target].to_numpy(float) - kinematic_prediction
        ) / response_sd
        for phase_bin, group in working.groupby("phase_bin", observed=True):
            session_rows.append(
                {
                    "path": group["path"].iloc[0],
                    "subject": group["subject"].iloc[0],
                    "phase_bin": int(phase_bin),
                    "n_events": int(len(group)),
                    "raw_spikes_per_touch": float(group[target].mean()),
                    "context_residual": float(group["context_residual"].mean()),
                    "kinematic_residual": float(group["kinematic_residual"].mean()),
                }
            )
    session_frame = pd.DataFrame(session_rows)
    subject_frame = (
        session_frame.groupby(["subject", "phase_bin"])[
            ["raw_spikes_per_touch", "context_residual", "kinematic_residual"]
        ]
        .median()
        .reset_index()
    )
    summaries = []
    for phase_bin, group in subject_frame.groupby("phase_bin"):
        row = {
            "phase_bin": int(phase_bin),
            "phase_center": float(-np.pi + (phase_bin + 0.5) * (2 * np.pi / 12)),
            "n_subjects": int(group["subject"].nunique()),
        }
        for measure in (
            "raw_spikes_per_touch",
            "context_residual",
            "kinematic_residual",
        ):
            values = group[measure].to_numpy(float)
            low, high = bootstrap_interval(values, np.mean)
            row[measure] = float(np.mean(values))
            row[f"{measure}_ci_low"] = float(low)
            row[f"{measure}_ci_high"] = float(high)
        summaries.append(row)
    return session_frame, pd.DataFrame(summaries).sort_values("phase_bin")


def phase_branch_contrasts(
    session_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize retraction-minus-protraction contrasts from held-out profiles."""
    rows = []
    measures = ("raw_spikes_per_touch", "context_residual", "kinematic_residual")
    for (path, subject), group in session_profile.groupby(["path", "subject"]):
        for measure in measures:
            branch_means = {}
            for branch, branch_group in (
                ("protraction", group[group["phase_bin"] < 6]),
                ("retraction", group[group["phase_bin"] >= 6]),
            ):
                if branch_group.empty:
                    branch_means[branch] = np.nan
                else:
                    branch_means[branch] = float(
                        np.average(branch_group[measure], weights=branch_group["n_events"])
                    )
            if np.isfinite(branch_means["protraction"]) and np.isfinite(
                branch_means["retraction"]
            ):
                rows.append(
                    {
                        "path": path,
                        "subject": subject,
                        "measure": measure,
                        "protraction_mean": branch_means["protraction"],
                        "retraction_mean": branch_means["retraction"],
                        "retraction_minus_protraction": (
                            branch_means["retraction"] - branch_means["protraction"]
                        ),
                    }
                )
    sessions = pd.DataFrame(rows)
    summaries = []
    for measure, group in sessions.groupby("measure", sort=False):
        subject_values = (
            group.groupby("subject")["retraction_minus_protraction"].median().dropna()
        )
        values = subject_values.to_numpy(float)
        low, high = bootstrap_interval(values, np.median)
        summaries.append(
            {
                "measure": measure,
                "n_files": int(group["path"].nunique()),
                "n_subjects": int(len(values)),
                "median_retraction_minus_protraction": float(np.median(values)),
                "n_positive": int(np.sum(values > 0)),
                "p_two_sided": float(stats.wilcoxon(values).pvalue),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return sessions, pd.DataFrame(summaries)


def phase_subregion_summary(
    data: pd.DataFrame, phase_sessions: pd.DataFrame
) -> pd.DataFrame:
    """Describe the primary adjusted phase gain by recorded barrel location."""
    primary = phase_sessions[
        (phase_sessions["response_window_ms"] == 50)
        & (phase_sessions["minimum_neighbor_ms"] == 50)
        & (phase_sessions["endpoint"] == "count")
        & (phase_sessions["base_model"] == "continuous_kinematics")
        & (phase_sessions["phase_basis"] == "two_harmonics")
        & (phase_sessions["regularization_strength"] == 1.0)
    ].copy()
    locations = data.groupby("path", as_index=False)["brain_subregion"].first()
    primary = primary.merge(locations, on="path", how="left")
    rows = []
    for location, group in primary.groupby("brain_subregion", dropna=False):
        subject_values = (
            group.groupby("subject")["delta_predictive_score"].median().dropna()
        )
        values = subject_values.to_numpy(float)
        low, high = bootstrap_interval(values, np.median)
        rows.append(
            {
                "brain_subregion": str(location),
                "n_files": int(group["path"].nunique()),
                "n_subjects": int(len(values)),
                "n_modeled_events": int(group["n_events"].sum()),
                "median_phase_gain": float(np.median(values)),
                "n_positive": int(np.sum(values > 0)),
                "p_two_sided": (
                    float(stats.wilcoxon(values).pvalue) if len(values) >= 5 else np.nan
                ),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def contact_duration_branch_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Describe contact duration using observed values within each phase branch."""
    selected = data[
        (data["previous_touch_interval_ms"].fillna(np.inf) >= 50)
        & (data["following_touch_interval_ms"].fillna(np.inf) >= 50)
    ].copy()
    selected["phase_branch_label"] = np.where(
        selected["phase_raw"] < 0, "protraction", "retraction"
    )
    rows = []
    for branch, group in selected.groupby("phase_branch_label"):
        observed = group["touch_duration_ms"].dropna().to_numpy(float)
        rows.append(
            {
                "phase_branch": branch,
                "n_events": int(len(group)),
                "n_observed_durations": int(len(observed)),
                "n_missing_durations": int(len(group) - len(observed)),
                "median_duration_ms": float(np.median(observed)),
                "percent_observed_at_least_50ms": float(np.mean(observed >= 50) * 100),
            }
        )
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=7.6, direction="out")


def panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(
        f"{letter}  {title}",
        loc="left",
        fontsize=8.8,
        weight="bold",
        pad=5,
    )


def save_publication_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        output.with_suffix(".tif"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    with PILImage.open(output.with_suffix(".tif")) as image:
        rgba = image.convert("RGBA")
        background = PILImage.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        background.convert("RGB").save(
            output.with_suffix(".tif"),
            format="TIFF",
            compression="tiff_lzw",
            dpi=(600, 600),
        )


def make_figure_1(
    isolation_profile: pd.DataFrame, history: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    thresholds = isolation_profile[
        "minimum_interval_to_both_observed_neighbors_ms"
    ].to_numpy(int)
    x = np.arange(len(thresholds))
    axes[0].errorbar(
        x,
        isolation_profile["mean_spikes_per_touch"],
        yerr=[
            isolation_profile["mean_spikes_per_touch"] - isolation_profile["ci_low"],
            isolation_profile["ci_high"] - isolation_profile["mean_spikes_per_touch"],
        ],
        fmt="o-",
        color=BLUE,
        markerfacecolor=BLUE,
        markersize=4.1,
        linewidth=1.0,
        elinewidth=1.0,
        capsize=2.0,
        capthick=0.8,
    )
    axes[0].set_xticks(x, [str(value) for value in thresholds])
    axes[0].set_xlabel("Minimum interval to each observed neighbor (ms)", fontsize=8.0)
    axes[0].set_ylabel("Spikes per touch (0-50 ms)", fontsize=8.0)
    panel_title(axes[0], "A", "Dense to isolated touches")
    style_axis(axes[0])

    baseline = history[
        (history["analysis"] == "baseline_subtracted_rate")
        & history["minimum_previous_interval_ms"].isin([0, 50, 100])
    ]
    for minimum, color, marker, linestyle, label in (
        (0, OCHRE, "o", "-", "0 ms"),
        (50, BLUE, "s", "--", "≥50 ms"),
        (100, GRAY, "^", ":", "≥100 ms"),
    ):
        part = baseline[baseline["minimum_previous_interval_ms"] == minimum].sort_values(
            "baseline_window_ms"
        )
        axes[1].errorbar(
            part["baseline_window_ms"],
            part["median_beta"],
            yerr=[
                part["median_beta"] - part["ci_low"],
                part["ci_high"] - part["median_beta"],
            ],
            color=color,
            marker=marker,
            markerfacecolor="white" if minimum else color,
            markersize=3.7,
            linewidth=1.1,
            linestyle=linestyle,
            elinewidth=0.7,
            capsize=1.6,
            capthick=0.7,
            label=label,
        )
        last = part.iloc[-1]
        axes[1].annotate(
            label,
            (last["baseline_window_ms"], last["median_beta"]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.8,
            color=BLACK,
        )
    axes[1].axhline(0, color=BLACK, linewidth=0.8)
    axes[1].set_xticks([10, 20, 30, 40, 50])
    axes[1].set_xlim(8, 63)
    axes[1].set_xlabel("Baseline window (ms)", fontsize=8.0)
    axes[1].set_ylabel("Standardized interval coefficient", fontsize=8.0)
    panel_title(axes[1], "B", "Baseline-subtracted rate")
    style_axis(axes[1])

    post = history[
        (history["analysis"] == "post_only")
        & (history["minimum_previous_interval_ms"] == 0)
    ].sort_values("response_window_ms")
    axes[2].errorbar(
        post["response_window_ms"],
        post["median_beta"],
        yerr=[
            post["median_beta"] - post["ci_low"],
            post["ci_high"] - post["median_beta"],
        ],
        fmt="o",
        color=BLUE,
        markerfacecolor=BLUE,
        markersize=4.0,
        elinewidth=1.0,
        capsize=2.0,
        capthick=0.8,
        linestyle="none",
    )
    axes[2].axhline(0, color=BLACK, linewidth=0.8)
    axes[2].set_xticks([10, 20, 30, 40, 50])
    axes[2].set_xlabel("Response window (ms)", fontsize=8.0)
    axes[2].set_ylabel("Standardized interval coefficient", fontsize=8.0)
    panel_title(axes[2], "C", "Post-touch count")
    style_axis(axes[2])
    fig.text(0.995, 0.004, "Fig. 1", ha="right", va="bottom", fontsize=6.5, color=BLACK)
    save_publication_figure(fig, output)
    plt.close(fig)


def make_figure_2(
    profile: pd.DataFrame,
    phase_summary: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.15, 2.75), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(0.85, 1.15, 1.0))
    polar = fig.add_subplot(grid[0, 0], projection="polar")
    adjusted = fig.add_subplot(grid[0, 1])
    windows = fig.add_subplot(grid[0, 2])

    theta = profile["phase_center"].to_numpy(float)
    radius = profile["raw_spikes_per_touch"].to_numpy(float)
    low = profile["raw_spikes_per_touch_ci_low"].to_numpy(float)
    high = profile["raw_spikes_per_touch_ci_high"].to_numpy(float)
    theta_closed = np.r_[theta, theta[0]]
    polar.plot(theta_closed, np.r_[radius, radius[0]], color=BLUE, linewidth=1.6)
    polar.fill_between(
        theta_closed,
        np.r_[low, low[0]],
        np.r_[high, high[0]],
        color=LIGHT_BLUE,
        alpha=0.50,
        linewidth=0,
    )
    polar.plot(theta_closed, np.r_[low, low[0]], color=BLUE, alpha=0.35, linewidth=0.5)
    polar.plot(theta_closed, np.r_[high, high[0]], color=BLUE, alpha=0.35, linewidth=0.5)
    polar.scatter(theta, radius, color=BLUE, s=17, zorder=3)
    polar.set_theta_zero_location("E")
    polar.set_theta_direction(1)
    polar.set_thetagrids(
        [0, 90, 180, 270],
        labels=["0", "π/2", "±π", "−π/2"],
        fontsize=7.4,
    )
    upper = max(0.7, float(np.nanmax(high)) * 1.05)
    polar.set_rlim(0, upper)
    polar.set_rgrids([0.2, 0.4, 0.6], angle=135, fontsize=6.8)
    polar.grid(color=LIGHT_GRAY, linewidth=0.5)
    polar.spines["polar"].set_linewidth(0.7)
    polar.text(
        0.5,
        -0.19,
        "Negative phase: protraction; nonnegative: retraction\n"
        "Events: 9,742 protraction; 2,554 retraction",
        transform=polar.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color=BLACK,
    )
    panel_title(polar, "A", "Raw spikes per touch (0-50 ms)")

    for measure, color, marker, linestyle, label in (
        ("context_residual", OCHRE, "o", "--", "Timing and task"),
        ("kinematic_residual", BLUE, "s", "-", "After measured movement"),
    ):
        adjusted.errorbar(
            theta,
            profile[measure],
            yerr=[
                profile[measure] - profile[f"{measure}_ci_low"],
                profile[f"{measure}_ci_high"] - profile[measure],
            ],
            color=color,
            marker=marker,
            markerfacecolor="white" if measure == "kinematic_residual" else color,
            markersize=3.5,
            linestyle=linestyle,
            linewidth=1.0,
            elinewidth=0.8,
            capsize=1.6,
            label=label,
        )
    adjusted.axhline(0, color=BLACK, linewidth=0.8)
    adjusted.axvline(0, color=LIGHT_GRAY, linewidth=0.6)
    adjusted.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["−π", "−π/2", "0", "π/2", "π"],
    )
    adjusted.set_xlabel("Whisking phase", fontsize=8.0)
    adjusted.set_ylabel("Prediction error in held-out trials", fontsize=8.0)
    adjusted.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        fontsize=7.0,
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.2,
        loc="upper left",
    )
    panel_title(adjusted, "B", "Residuals by whisking phase")
    style_axis(adjusted)

    selected = phase_summary[
        (phase_summary["minimum_neighbor_ms"] == 50)
        & (phase_summary["endpoint"] == "count")
        & (phase_summary["phase_basis"] == "two_harmonics")
        & (phase_summary["regularization_strength"] == 1.0)
    ]
    for model, color, marker, linestyle, label in (
        ("context", OCHRE, "o", "-", "Timing and task"),
        ("continuous_kinematics", BLUE, "s", "--", "After measured movement"),
    ):
        part = selected[selected["base_model"] == model].sort_values("response_window_ms")
        windows.errorbar(
            part["response_window_ms"],
            part["median_delta_predictive_score"],
            yerr=[
                part["median_delta_predictive_score"] - part["ci_low"],
                part["ci_high"] - part["median_delta_predictive_score"],
            ],
            color=color,
            marker=marker,
            markerfacecolor="white" if model == "continuous_kinematics" else color,
            markersize=3.8,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=1.8,
            linestyle=linestyle,
            label=label,
        )
    windows.axhline(0, color=BLACK, linewidth=0.8)
    windows.set_xticks([10, 20, 30, 40, 50])
    windows.set_xlabel("Response window (ms)", fontsize=8.0)
    windows.set_ylabel("Improvement in held-out prediction", fontsize=8.0)
    windows.legend(
        frameon=False,
        fontsize=7.0,
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.2,
        loc="upper left",
    )
    panel_title(windows, "C", "Prediction across response windows")
    style_axis(windows)
    fig.text(0.995, 0.004, "Fig. 2", ha="right", va="bottom", fontsize=6.5, color=BLACK)
    save_publication_figure(fig, output)
    plt.close(fig)


KINEMATIC_LABELS = {
    "log_amplitude": "Whisking amplitude",
    "signed_log_pretouch_velocity": "Pretouch angular velocity",
    "signed_log_pretouch_acceleration": "Pretouch angular acceleration",
    "signed_log_curvature": "Signed peak curvature change",
    "log_abs_curvature": "Absolute peak curvature change",
    "log_touch_duration": "Touch duration",
    "distance_to_pole_at_touch": "Distance to pole",
    "theta_at_touch_deg": "Base angle at touch",
}


def make_figure_3(
    kinematic_sessions: pd.DataFrame, kinematic_summary: pd.DataFrame, output: Path
) -> None:
    order = list(KINEMATIC_LABELS)
    summary = kinematic_summary.set_index("feature").loc[order]
    subjects = (
        kinematic_sessions.groupby(["subject", "feature"])["standardized_beta"]
        .median()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    y = np.arange(len(order))
    for index, feature in enumerate(order):
        values = subjects[subjects["feature"] == feature]["standardized_beta"]
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
        axis.scatter(
            values,
            np.full(len(values), index) + jitter,
            s=13,
            color=LIGHT_BLUE,
            edgecolor="none",
            alpha=0.75,
            zorder=1,
        )
    axis.errorbar(
        summary["median_beta"],
        y,
        xerr=[
            summary["median_beta"] - summary["ci_low"],
            summary["ci_high"] - summary["median_beta"],
        ],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
        markersize=5,
        linewidth=1.4,
        zorder=2,
    )
    axis.axvline(0, color=BLACK, linewidth=0.8)
    axis.set_yticks(y, [KINEMATIC_LABELS[feature] for feature in order], fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("Standardized multivariable coefficient", fontsize=9)
    axis.set_title(
        "Kinematic and contact measures with 50-ms neighbor exclusion",
        loc="left",
        fontsize=10,
        weight="bold",
    )
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", labelsize=8)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LIGHT_BLUE, markersize=5, label="Archive subject label"),
        Line2D([0], [0], marker="o", color=BLUE, markersize=5, label="Median and bootstrap 95% CI"),
    ]
    axis.legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tif"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    results = args.output_root / "results"
    figures = args.output_root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    data = prepare(pd.read_csv(args.input_csv))
    data.to_csv(results / "eligible_touch_events_with_features.csv", index=False)

    phase_sessions = []
    phase_summaries = []
    primary_configs = [
        (50, 50, endpoint, base, "two_harmonics", 1.0)
        for endpoint in ("count", "any_spike")
        for base in ("context", "continuous_kinematics")
    ]
    window_configs = [
        (window, 50, "count", base, "two_harmonics", 1.0)
        for window in WINDOWS_MS
        for base in ("context", "continuous_kinematics")
    ]
    interval_configs = [
        (50, interval, "count", base, "two_harmonics", 1.0)
        for interval in MIN_INTERVALS_MS
        for base in ("context", "continuous_kinematics")
    ]
    basis_configs = [
        (50, 50, "count", base, phase_basis, alpha)
        for base in ("context", "continuous_kinematics")
        for phase_basis in ("first_harmonic", "piecewise_branches")
        for alpha in (0.1, 1.0, 10.0)
    ]
    primary_penalty_configs = [
        (50, 50, "count", base, "two_harmonics", alpha)
        for base in ("context", "continuous_kinematics")
        for alpha in (0.01, 0.1, 10.0)
    ]
    descriptive_contact_config = [
        (50, 50, "count", "kinematics_and_contact_descriptions", "two_harmonics", 1.0)
    ]
    configs = list(
        dict.fromkeys(
            primary_configs
            + window_configs
            + interval_configs
            + basis_configs
            + primary_penalty_configs
            + descriptive_contact_config
        )
    )
    for config in configs:
        sessions, summary = run_phase_family(data, *config)
        sessions = sessions.assign(**summary)
        phase_sessions.append(sessions)
        phase_summaries.append(summary)
        print(
            "phase",
            config,
            summary["median_delta_predictive_score"],
            flush=True,
        )
    phase_session_frame = pd.concat(phase_sessions, ignore_index=True)
    phase_summary_frame = pd.DataFrame(phase_summaries)
    primary_mask = (
        (phase_summary_frame["response_window_ms"] == 50)
        & (phase_summary_frame["minimum_neighbor_ms"] == 50)
        & (phase_summary_frame["phase_basis"] == "two_harmonics")
        & (phase_summary_frame["regularization_strength"] == 1.0)
        & phase_summary_frame["base_model"].isin(
            ["context", "continuous_kinematics"]
        )
    )
    phase_summary_frame["q_primary"] = np.nan
    phase_summary_frame.loc[primary_mask, "q_primary"] = bh_q(
        phase_summary_frame.loc[primary_mask, "p_two_sided"].to_numpy(float)
    )
    phase_summary_frame["q_sign_primary"] = np.nan
    phase_summary_frame.loc[primary_mask, "q_sign_primary"] = bh_q(
        phase_summary_frame.loc[primary_mask, "p_sign_two_sided"].to_numpy(float)
    )
    phase_session_frame.to_csv(results / "phase_session_metrics.csv", index=False)
    phase_summary_frame.to_csv(results / "phase_subject_summary.csv", index=False)

    amplitude_phase_sessions, amplitude_phase_summary = run_amplitude_phase_interaction(
        data, alpha=1.0
    )
    amplitude_phase_sessions.assign(**amplitude_phase_summary).to_csv(
        results / "amplitude_phase_interaction_session_metrics.csv", index=False
    )
    pd.DataFrame([amplitude_phase_summary]).to_csv(
        results / "amplitude_phase_interaction_subject_summary.csv", index=False
    )

    observed_neighbor_data = data[
        data["previous_touch_interval_ms"].notna()
        & data["following_touch_interval_ms"].notna()
    ]
    observed_phase_sessions = []
    observed_phase_summaries = []
    for base in ("context", "continuous_kinematics"):
        sessions, summary = run_phase_family(
            observed_neighbor_data, 50, 50, "count", base, "two_harmonics", 1.0
        )
        sessions = sessions.assign(**summary)
        observed_phase_sessions.append(sessions)
        observed_phase_summaries.append(summary)
    pd.concat(observed_phase_sessions, ignore_index=True).to_csv(
        results / "observed_neighbor_phase_session_metrics.csv", index=False
    )
    pd.DataFrame(observed_phase_summaries).to_csv(
        results / "observed_neighbor_phase_subject_summary.csv", index=False
    )

    history_sessions, history_summary = run_history_windows(data)
    history_sessions.to_csv(results / "history_window_session_metrics.csv", index=False)
    history_summary.to_csv(results / "history_window_subject_summary.csv", index=False)

    baseline_sessions, baseline_summary = baseline_response_window_sensitivity(data)
    baseline_sessions.to_csv(
        results / "baseline_response_window_session_metrics.csv", index=False
    )
    baseline_summary.to_csv(
        results / "baseline_response_window_subject_summary.csv", index=False
    )

    kinematic_sessions, kinematic_subjects, kinematic_summary = run_kinematic_coefficients(data)
    kinematic_sessions.to_csv(results / "kinematic_session_coefficients.csv", index=False)
    kinematic_subjects.to_csv(results / "kinematic_subject_coefficients.csv", index=False)
    kinematic_summary.to_csv(results / "kinematic_subject_summary.csv", index=False)

    intervals = interval_bin_profile(data)
    intervals.to_csv(results / "intertouch_interval_bin_profile.csv", index=False)
    bilateral = bilateral_isolation_profile(data)
    bilateral.to_csv(results / "bilateral_isolation_profile.csv", index=False)
    observed_bilateral = observed_neighbor_isolation_profile(data)
    observed_bilateral.to_csv(
        results / "observed_neighbor_isolation_profile.csv", index=False
    )
    profile_sessions, profile_summary = phase_profiles(data)
    profile_sessions.to_csv(results / "phase_bin_session_profile.csv", index=False)
    profile_summary.to_csv(results / "phase_bin_subject_profile.csv", index=False)
    branch_sessions, branch_summary = phase_branch_contrasts(profile_sessions)
    branch_sessions.to_csv(results / "phase_branch_session_contrasts.csv", index=False)
    branch_summary.to_csv(results / "phase_branch_subject_summary.csv", index=False)
    subregion_summary = phase_subregion_summary(data, phase_session_frame)
    subregion_summary.to_csv(results / "phase_subregion_summary.csv", index=False)
    duration_summary = contact_duration_branch_summary(data)
    duration_summary.to_csv(results / "contact_duration_branch_summary.csv", index=False)

    make_figure_1(
        observed_bilateral,
        history_summary,
        figures / "Figure_1_intertouch_and_windows",
    )
    make_figure_2(
        profile_summary,
        phase_summary_frame,
        figures / "Figure_2_phase_and_continuous_adjustment",
    )
    make_figure_3(
        kinematic_sessions,
        kinematic_summary,
        figures / "Figure_3_continuous_kinematics",
    )

    main_history = history_summary[
        (
            (history_summary["analysis"] == "post_only")
            & (history_summary["response_window_ms"] == 50)
            & (history_summary["minimum_previous_interval_ms"] == 0)
        )
        | (
            (history_summary["analysis"] == "baseline_subtracted_rate")
            & (history_summary["baseline_window_ms"] == 50)
            & history_summary["minimum_previous_interval_ms"].isin([0, 100])
        )
    ]
    main_phase = phase_summary_frame[primary_mask]
    summary = {
        "eligible_events": int(len(data)),
        "sessions": int(data["path"].nunique()),
        "archive_subject_ids": int(data["subject"].nunique()),
        "cortical_layer": "4",
        "phase_protraction_events": int((data["phase_raw"] < 0).sum()),
        "phase_retraction_events": int((data["phase_raw"] >= 0).sum()),
        "main_history_results": main_history.to_dict(orient="records"),
        "main_phase_results": main_phase.to_dict(orient="records"),
        "amplitude_phase_interaction": amplitude_phase_summary,
        "kinematic_results": kinematic_summary.to_dict(orient="records"),
        "phase_branch_results": branch_summary.to_dict(orient="records"),
        "phase_subregion_results": subregion_summary.to_dict(orient="records"),
        "observed_neighbor_phase_results": observed_phase_summaries,
        "contact_duration_by_branch": duration_summary.to_dict(orient="records"),
        "notes": [
            "All spike windows are half-open intervals re-extracted from the 10 kHz NWB spike train.",
            "Phase models use grouped trial cross-validation and archive subject IDs as the final aggregation level.",
            "Behavioral outcome labels such as hit, miss, and false alarm are not used as model predictors.",
        ],
    }
    summary = json_safe(summary)
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    (results / "analysis_summary.json").write_text(serialized)
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
