from dataclasses import dataclass
import pandas as pd

from features import build_shepherd_features
from baselines import LogisticModel, BaseRateModel, single_feature, persistence_feature
from metrics import binary_metrics, stratify

@dataclass
class SHEPHERDExperiment:
    time_column: str
    target_column: str
    feature_columns: list
    horizons: tuple = (3,7,23)
    train_fraction: float = 0.65
    random_state: int = 23

    def run(self, df):
        df = df.sort_values(self.time_column).reset_index(drop=True).copy()
        cut = max(20, int(len(df)*self.train_fraction))
        y = df[self.target_column].astype(int).to_numpy()

        sf = build_shepherd_features(df, self.feature_columns, self.horizons)
        raw = df[self.feature_columns].astype(float).ffill().fillna(0.0)
        full = pd.concat([raw, sf], axis=1).fillna(0.0)

        error_only = single_feature(df, "step_error")
        contradiction_only = single_feature(df, "contradiction")
        persistence = persistence_feature(
            df,
            [c for c in ["step_error","tool_failure","contradiction","memory_mismatch"] if c in df.columns],
            window=7,
        )

        models = {
            "shepherd": (LogisticModel(self.random_state), full),
            "raw_state": (LogisticModel(self.random_state), raw),
            "error_only": (LogisticModel(self.random_state), error_only),
            "contradiction_only": (LogisticModel(self.random_state), contradiction_only),
            "persistence": (LogisticModel(self.random_state), persistence),
            "base_rate": (BaseRateModel(), raw),
        }

        results = {}
        for name, (model, X) in models.items():
            model.fit(X.iloc[:cut], y[:cut])
            p = model.predict_proba(X.iloc[cut:])
            results[name] = {
                "overall": binary_metrics(y[cut:], p),
                "by_hazard": stratify(
                    y[cut:], p, sf["shepherd_hazard_score"].to_numpy()[cut:]
                )
            }

        return {
            "n_total": len(df),
            "train_n": cut,
            "test_n": len(df)-cut,
            "results": results,
            "primary_delta_brier_raw_minus_shepherd": (
                results["raw_state"]["overall"]["brier"]
                - results["shepherd"]["overall"]["brier"]
            ),
            "interpretation": (
                "Positive delta favors SHEPHERD over the matched raw-state model. "
                "A negative delta means trajectory engineering did not add Brier value."
            )
        }
