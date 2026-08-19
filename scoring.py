"""
scoring.py

Two things live here:

1. fit_weights() -- used by train.py (the one-time/occasional training run).
   Takes the walk-forward backtest results and decides which metrics earned
   a weight (only ones that beat the naive baseline out-of-sample).

2. score_with_saved_models() -- used by app.py (the live dashboard). Applies
   ALREADY-FITTED coefficients from the training run to today's live metric
   values. No model fitting happens here -- it's just arithmetic on numbers
   that were already computed once, which is why the dashboard is fast.

If zero metrics beat baseline, the honest output is "no edge detected" --
the scorer says so rather than fabricate a confident number.
"""

import math
import pandas as pd


def fit_weights(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """Used only during training (train.py). Weight per metric = its
    oos_accuracy edge over baseline, normalized to sum to 1."""
    valid = backtest_results[backtest_results["beats_baseline"]].copy()
    if valid.empty:
        return valid
    valid["edge"] = valid["oos_accuracy"] - valid["naive_baseline"]
    valid["weight"] = valid["edge"] / valid["edge"].sum()
    return valid[["metric", "oos_accuracy", "naive_baseline", "edge", "weight"]]


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def score_with_saved_models(latest_values: dict, production_models: dict) -> dict:
    """Fast path for the live dashboard: apply already-fitted coefficients
    (produced once by train.py) to today's metric values."""
    if not production_models:
        return {"composite_score": None, "direction": "no edge detected",
                "detail": "No metric beat the naive baseline out-of-sample in the last "
                          "training run. Don't trade this signal; consider retraining "
                          "later with more data, or adding more/better metrics."}

    contributions = []
    for metric, m in production_models.items():
        x = latest_values.get(metric)
        if x is None:
            continue
        p_up = _sigmoid(m["coef"] * x + m["intercept"])
        contributions.append({"metric": metric, "p_up": round(p_up, 3), "weight": round(m["weight"], 3)})

    if not contributions:
        return {"composite_score": None, "direction": "missing current data",
                "detail": "Today's value wasn't available for any validated metric."}

    total_weight = sum(c["weight"] for c in contributions)
    composite = sum(c["p_up"] * c["weight"] for c in contributions) / total_weight
    return {
        "composite_score": round(composite, 3),
        "direction": "up" if composite > 0.5 else "down",
        "confidence": round(abs(composite - 0.5) * 2, 3),
        "contributions": contributions,
    }
