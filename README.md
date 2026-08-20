# SHEPHERD

**Systemic Hazard Evaluation through Persistent History, Error, Recursion and Drift**

SHEPHERD is an EFMW-derived framework for long-horizon AI-agent reliability.

Its narrow question is:

> Can accumulated trajectory structure predict meaningful agent failure before a single output becomes obviously wrong?

## Intended uses

- tool-using agents
- long-running autonomous workflows
- customer-support agents
- coding agents
- research agents
- planning agents
- multi-step orchestration systems
- memory-enabled assistants
- agentic commerce systems

## Core signal path

```text
agent step
→ outcome / error state
→ memory state
→ tool-use state
→ contradiction
→ recurrence
→ recovery quality
→ drift
→ persistent hazard
→ SHEPHERD risk score
```

## Flat-stack repository

```text
README.md
LICENSE
CITATION.cff
pyproject.toml
requirements.txt
shepherd.py
features.py
baselines.py
metrics.py
freeze.py
hazard_schema.json
example_config.yaml
example_data.csv
example_run.py
test_shepherd.py
MANIFEST.sha256
```

## Scientific hypothesis

> Long-horizon agent failures may be preceded by persistent combinations of memory inconsistency, repeated tool errors, unresolved contradictions, and declining recovery quality.

SHEPHERD does not assume this is true.

## Example features

- step error rate
- tool-call failure rate
- repeated tool-call rate
- contradiction rate
- memory mismatch rate
- recovery success
- correction distance
- retry count
- unresolved state count
- task progress
- context age
- abstention rate

## Baselines

Included:

- raw-step logistic model
- error-rate-only baseline
- contradiction-only baseline
- recent-window persistence baseline
- base-rate predictor

## Primary endpoint

For a frozen future hazard window:

> Does SHEPHERD improve prospective failure prediction over a matched raw-feature model?

Primary metric:

- Brier score

Secondary metrics:

- log loss
- ROC-AUC
- average precision
- calibration
- performance by trajectory length / drift regime

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python example_run.py
```

## Tests

```bash
pytest -q
```

## Minimal use

```python
import pandas as pd
from shepherd import SHEPHERDExperiment

df = pd.read_csv("example_data.csv")

exp = SHEPHERDExperiment(
    time_column="step",
    target_column="future_failure",
    feature_columns=[
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
    ],
    horizons=(3, 7, 23),
)

print(exp.run(df))
```

## Freeze discipline

Freeze before prospective evaluation:

- event schema
- target definition
- trajectory boundary
- horizons
- recurrence rules
- recovery metric
- contradiction definition
- drift equations
- baselines
- metrics
- exclusions
- seeds
- compute budget

Then run:

```bash
python freeze.py
```

## Negative results

If raw recent-state features perform equally well, SHEPHERD has not demonstrated incremental trajectory value.

## Status

Research software. Experimental. Not validated for high-stakes autonomous decision-making.
