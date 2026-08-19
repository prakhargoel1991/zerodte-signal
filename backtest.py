"""
backtest.py

Tests each candidate metric against the actual target (0DTE/1DTE up/down)
using WALK-FORWARD validation: for each test day, only data strictly before
it is used to fit anything. This matters a lot here -- with daily data you
have maybe 500 rows/2yrs, so a single in-sample correlation number is easy
to overfit and will look far more predictive than it is out of sample.

Two outputs per metric:
  - in_sample_corr: point-biserial correlation over the whole period (for
    reference / sanity-checking direction of the relationship only)
  - oos_accuracy: walk-forward out-of-sample accuracy of a simple logistic
    model using that metric ALONE to predict direction

Only metrics that beat a naive baseline (the historical up-rate, i.e. "always
guess up") out-of-sample should get weight in the composite score. A metric
with in-sample correlation but oos_accuracy <= baseline is very likely noise
-- flag it, don't discard it silently, but don't weight it either.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def walk_forward_single_metric(df: pd.DataFrame, metric: str, target: str = "target_up",
                                min_train: int = 60, step: int = 3) -> dict:
    """Fit metric -> target logistic regression walk-forward, one metric at a time.

    step: only refit/test every `step`-th day instead of every single day.
    This cuts runtime roughly `step`-fold. Testing every 3rd day instead of
    every day barely changes the accuracy estimate (the model itself doesn't
    change much day to day) but matters a lot for how long this takes on a
    free/shared server.
    """
    X = df[[metric]].values
    y = df[target].values
    n = len(df)
    preds = np.full(n, np.nan)

    for i in range(min_train, n, step):
        X_train, y_train = X[:i], y[:i]
        if len(np.unique(y_train)) < 2:
            continue  # can't fit if all one class so far
        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds[i] = model.predict(X[i].reshape(1, -1))[0]

    valid = ~np.isnan(preds)
    oos_accuracy = (preds[valid] == y[valid]).mean() if valid.sum() > 0 else np.nan
    baseline = max(y.mean(), 1 - y.mean())  # naive "always guess majority class"
    in_sample_corr = pd.Series(X.flatten()).corr(pd.Series(y))

    return {
        "metric": metric,
        "in_sample_corr": round(in_sample_corr, 3),
        "oos_accuracy": round(oos_accuracy, 3) if not np.isnan(oos_accuracy) else np.nan,
        "naive_baseline": round(baseline, 3),
        "beats_baseline": bool(oos_accuracy > baseline) if not np.isnan(oos_accuracy) else False,
        "n_test_days": int(valid.sum()),
    }


def run_all_metrics(df: pd.DataFrame, target: str = "target_up", min_train: int = 60, step: int = 3) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c != target]
    results = [walk_forward_single_metric(df, m, target, min_train, step) for m in metric_cols]
    out = pd.DataFrame(results).sort_values("oos_accuracy", ascending=False).reset_index(drop=True)
    return out


def walk_forward_combined(df: pd.DataFrame, target: str = "target_up", min_train: int = 60, step: int = 3) -> dict:
    """Same walk-forward discipline as the single-metric test, but fits ONE
    model using ALL metrics together at each step. This can catch signal
    that only shows up in combination (e.g. VIX elevated AND it's a Friday)
    even when no individual metric beats baseline alone.

    Still walk-forward: only data strictly before day i is used to predict
    day i, so this isn't just in-sample curve-fitting with more knobs.
    """
    metric_cols = [c for c in df.columns if c != target]
    X = df[metric_cols].values
    y = df[target].values
    n = len(df)
    preds = np.full(n, np.nan)

    for i in range(min_train, n, step):
        X_train, y_train = X[:i], y[:i]
        if len(np.unique(y_train)) < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds[i] = model.predict(X[i].reshape(1, -1))[0]

    valid = ~np.isnan(preds)
    oos_accuracy = (preds[valid] == y[valid]).mean() if valid.sum() > 0 else np.nan
    baseline = max(y.mean(), 1 - y.mean())

    return {
        "model": "combined (all metrics)",
        "metrics_used": metric_cols,
        "oos_accuracy": round(oos_accuracy, 3) if not np.isnan(oos_accuracy) else np.nan,
        "naive_baseline": round(baseline, 3),
        "beats_baseline": bool(oos_accuracy > baseline) if not np.isnan(oos_accuracy) else False,
        "n_test_days": int(valid.sum()),
    }
