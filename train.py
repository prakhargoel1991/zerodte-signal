"""
train.py

Run this ONCE (or occasionally -- weekly/monthly, whenever you want the
backtest refreshed) to do the expensive walk-forward validation. Saves
everything to model_store.json. The live dashboard (app.py) just reads that
file and applies it to today's numbers -- no backtest loop on every page
load, which is what was making the dashboard slow.

Run with:  python train.py

There's also a "Retrain now" button inside the dashboard that calls this
same logic on demand, so you don't strictly need to run this file by hand --
but running it once before first use means the dashboard has something to
show immediately instead of an empty state.
"""

import json
import datetime as dt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_sources import get_price_history, get_vix_family, get_putcall_ratio
from features import build_feature_matrix
from backtest import run_all_metrics, walk_forward_combined
from scoring import fit_weights

MODEL_STORE_PATH = "model_store.json"
TICKERS = ["SPY", "QQQ"]
HORIZONS = ["0DTE", "1DTE"]


def train_one(ticker: str, horizon: str, putcall_df=None) -> dict:
    """Run the full walk-forward validation for one ticker/horizon combo,
    then fit final production models (on the FULL history, one time) for
    whichever metrics actually validated."""
    price_df = get_price_history(ticker, period="5y")
    vix_df = get_vix_family(period="5y")
    df = build_feature_matrix(price_df, vix_df, horizon=horizon, putcall_df=putcall_df)

    backtest_results = run_all_metrics(df)
    combined = walk_forward_combined(df)
    weights = fit_weights(backtest_results)

    production_models = {}
    for _, row in weights.iterrows():
        metric = row["metric"]
        model = LogisticRegression()
        model.fit(df[[metric]].values, df["target_up"].values)
        production_models[metric] = {
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
            "weight": float(row["weight"]),
            "oos_accuracy": float(row["oos_accuracy"]),
            "naive_baseline": float(row["naive_baseline"]),
        }

    return {
        "trained_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_rows": int(len(df)),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "backtest_results": backtest_results.to_dict(orient="records"),
        "combined_model": combined,
        "production_models": production_models,
    }


def main():
    try:
        putcall_df = get_putcall_ratio()
        print("Put/call ratio source OK, including it in training.")
    except Exception as e:
        print(f"Put/call ratio source unavailable ({e}); training without it.")
        putcall_df = None

    store = {}
    for ticker in TICKERS:
        for horizon in HORIZONS:
            key = f"{ticker}_{horizon}"
            print(f"Training {key} ...")
            store[key] = train_one(ticker, horizon, putcall_df)
            n_validated = len(store[key]["production_models"])
            print(f"  -> {n_validated} metric(s) validated out-of-sample.")

    with open(MODEL_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2, allow_nan=True)
    print(f"\nSaved results to {MODEL_STORE_PATH}")


if __name__ == "__main__":
    main()
