import pandas as pd
from features import build_shepherd_features
from shepherd import SHEPHERDExperiment

FEATURES = [
    "step_error",
    "tool_failure",
    "repeat_tool_call",
    "contradiction",
    "memory_mismatch",
    "recovery_success",
    "retry_count",
    "unresolved_state",
    "task_progress",
    "context_age",
]

def test_features():
    df = pd.read_csv("example_data.csv")
    sf = build_shepherd_features(df, FEATURES)
    assert len(sf) == len(df)
    assert "shepherd_hazard_score" in sf.columns
    assert sf["shepherd_hazard_score"].between(0,1).all()

def test_experiment_runs():
    df = pd.read_csv("example_data.csv")
    exp = SHEPHERDExperiment(
        time_column="step",
        target_column="future_failure",
        feature_columns=FEATURES,
    )
    out = exp.run(df)
    assert "results" in out
    assert "shepherd" in out["results"]
    assert "raw_state" in out["results"]
