import numpy as np
import pandas as pd

EPS = 1e-8

def robust_z(s, window=23):
    s = s.astype(float)
    med = s.rolling(window, min_periods=3).median()
    mad = (s-med).abs().rolling(window, min_periods=3).median()
    return ((s-med)/(1.4826*mad+EPS)).replace([np.inf,-np.inf], np.nan).fillna(0.0)

def build_shepherd_features(df, feature_columns, horizons=(3,7,23)):
    x = df[feature_columns].astype(float).copy()
    out = pd.DataFrame(index=df.index)

    for c in feature_columns:
        out[f"{c}__z"] = robust_z(x[c], max(horizons))
        out[f"{c}__dx"] = x[c].diff().fillna(0.0)

    def get(name):
        return x[name] if name in x.columns else pd.Series(0.0, index=x.index)

    out["error_pressure"] = (
        get("step_error")
        + get("tool_failure")
        + get("repeat_tool_call")
        + get("contradiction")
        + get("memory_mismatch")
        + get("unresolved_state")
    ) / 6.0

    out["recovery_deficit"] = (
        1.0 - get("recovery_success").clip(0,1)
        + get("retry_count").rank(pct=True)
    ) / 2.0

    out["progress_stall"] = (
        1.0 - get("task_progress").clip(0,1)
        + get("context_age").rank(pct=True)
    ) / 2.0

    out["trajectory_contradiction"] = (
        get("contradiction")
        + get("memory_mismatch")
        + get("unresolved_state")
    ) / 3.0

    # Recurrence proxy: repeated local error state over short windows.
    err_state = (
        (get("step_error") > 0).astype(int)
        + (get("tool_failure") > 0).astype(int)
        + (get("contradiction") > 0).astype(int)
        + (get("memory_mismatch") > 0).astype(int)
    )
    out["error_recurrence"] = err_state.rolling(7, min_periods=1).mean()

    # Drift = rising hazard components.
    out["hazard_velocity"] = (
        out["error_pressure"].diff().fillna(0.0)
        + out["recovery_deficit"].diff().fillna(0.0)
        + out["progress_stall"].diff().fillna(0.0)
    ) / 3.0

    out["hazard_acceleration"] = out["hazard_velocity"].diff().fillna(0.0)

    for h in horizons:
        out[f"error_pressure_mean_{h}"] = out["error_pressure"].rolling(h, min_periods=1).mean()
        out[f"recovery_deficit_mean_{h}"] = out["recovery_deficit"].rolling(h, min_periods=1).mean()
        out[f"progress_stall_mean_{h}"] = out["progress_stall"].rolling(h, min_periods=1).mean()
        out[f"contradiction_mean_{h}"] = out["trajectory_contradiction"].rolling(h, min_periods=1).mean()
        out[f"recurrence_mean_{h}"] = out["error_recurrence"].rolling(h, min_periods=1).mean()

    components = [
        out["error_pressure"],
        out["recovery_deficit"],
        out["progress_stall"],
        out["trajectory_contradiction"],
        out["error_recurrence"],
        out["hazard_velocity"].clip(lower=0),
        out["hazard_acceleration"].clip(lower=0),
    ]
    for h in horizons:
        components.extend([
            out[f"error_pressure_mean_{h}"],
            out[f"recovery_deficit_mean_{h}"],
            out[f"progress_stall_mean_{h}"],
            out[f"contradiction_mean_{h}"],
            out[f"recurrence_mean_{h}"],
        ])

    ranked = pd.concat(components, axis=1).rank(pct=True)
    out["shepherd_hazard_score"] = ranked.mean(axis=1).fillna(0.0)

    return out.replace([np.inf,-np.inf], np.nan).fillna(0.0)
