"""
scoring.py

Builds today's composite up/down score using ONLY metrics that beat the
naive baseline out-of-sample in backtest.py. Weight per metric = its
oos_accuracy edge over baseline (accuracy - baseline), normalized to sum to 1.
This means the weighting comes from validated performance, not from how
popular a metric is on Twitter/Reddit.

If zero metrics beat baseline, the honest output is "no edge detected" --
the scorer will say so rather than fabricate a confident number.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_weights(backtest_results: pd.DataFrame) -> pd.DataFrame:
    valid = backtest_results[backtest_results["beats_baseline"]].copy()
    if valid.empty:
        return valid
    valid["edge"] = valid["oos_accuracy"] - valid["naive_baseline"]
    valid["weight"] = valid["edge"] / valid["edge"].sum()
    return valid[["metric", "oos_accuracy", "naive_baseline", "edge", "weight"]]


def score_today(df: pd.DataFrame, weights: pd.DataFrame, target: str = "target_up") -> dict:
    """
    Fits one logistic model per validated metric on the FULL history (final
    production fit, not walk-forward -- walk-forward already told us these
    metrics generalize), applies each to today's row, and combines their
    up-probabilities using the validated weights.
    """
    if weights.empty:
        return {"composite_score": None, "direction": "no edge detected",
                "detail": "No metric beat the naive baseline out-of-sample. "
                          "Don't trade this signal; consider adding more/better metrics."}

    today = df.iloc[[-1]]
    contributions = []
    for _, row in weights.iterrows():
        metric = row["metric"]
        model = LogisticRegression()
        model.fit(df[[metric]].values, df[target].values)
        p_up = model.predict_proba(today[[metric]].values)[0][1]
        contributions.append({"metric": metric, "p_up": round(p_up, 3), "weight": round(row["weight"], 3)})

    composite = sum(c["p_up"] * c["weight"] for c in contributions)
    return {
        "composite_score": round(composite, 3),
        "direction": "up" if composite > 0.5 else "down",
        "confidence": round(abs(composite - 0.5) * 2, 3),  # 0 = coin flip, 1 = max conviction
        "contributions": contributions,
        "as_of": str(today.index[0].date()),
    }
