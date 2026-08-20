import json
import pandas as pd
from shepherd import SHEPHERDExperiment

df = pd.read_csv("example_data.csv")

features = [
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

exp = SHEPHERDExperiment(
    time_column="step",
    target_column="future_failure",
    feature_columns=features,
    horizons=(3,7,23),
    random_state=23,
)

result = exp.run(df)
print(json.dumps(result, indent=2))
