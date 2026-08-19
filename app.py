"""
app.py

Run with:  streamlit run app.py

Lightweight live dashboard. Loads results from model_store.json (produced
by train.py or the "Retrain now" button below) and applies them to TODAY's
live metric values. Does NOT run the walk-forward backtest on every page
load -- that's the slow part, and it now only happens when you explicitly
retrain.
"""

import os
import json
import streamlit as st
import pandas as pd

from data_sources import get_price_history, get_vix_family, get_putcall_ratio
from features import build_current_features
from scoring import score_with_saved_models
from train import train_one, MODEL_STORE_PATH

st.set_page_config(page_title="0-1 DTE Signal Dashboard", layout="wide")
st.title("SPY / QQQ 0-1 DTE Composite Signal")
st.caption(
    "Not financial advice. This is a statistical composite of publicly "
    "discussed metrics, validated walk-forward against actual direction. "
    "Treat as one input, not a trade trigger."
)

col1, col2 = st.columns(2)
ticker = col1.selectbox("Underlying", ["SPY", "QQQ"])
horizon = col2.selectbox("Horizon", ["0DTE", "1DTE"])
key = f"{ticker}_{horizon}"

# ---- Load whatever's already been trained (fast, just reading a file) ----
if os.path.exists(MODEL_STORE_PATH):
    with open(MODEL_STORE_PATH) as f:
        store = json.load(f)
else:
    store = {}

entry = store.get(key)

if entry is None:
    st.warning(
        "No trained model yet for this combination. Click 'Retrain now' below "
        "to run the one-time backtest (~30-60s, only needed once per combo, or "
        "whenever you want it refreshed)."
    )
else:
    st.caption(
        f"Model last trained: {entry['trained_at']} | using {entry['n_rows']} days "
        f"({entry['date_range'][0]} to {entry['date_range'][1]})"
    )

if st.button("🔄 Retrain now (~30-60s)"):
    with st.spinner("Retraining -- this runs the full walk-forward backtest once..."):
        try:
            putcall_df = get_putcall_ratio()
        except Exception:
            putcall_df = None
        entry = train_one(ticker, horizon, putcall_df)
        store[key] = entry
        with open(MODEL_STORE_PATH, "w") as f:
            json.dump(store, f, indent=2)
    st.success("Retrained. Results below are now up to date.")

if entry is None:
    st.stop()

# ---- Show the (saved, not recomputed) backtest results ----
st.subheader("Walk-forward validation (per metric)")
st.dataframe(pd.DataFrame(entry["backtest_results"]))
st.caption(
    "oos_accuracy = out-of-sample directional accuracy using ONLY this metric. "
    "naive_baseline = accuracy from always guessing the majority class. "
    "A metric only earns weight below if oos_accuracy > naive_baseline. "
    "This table is from the last training run, not recalculated on page load."
)

st.subheader("Combined model (all metrics together)")
combined = entry["combined_model"]
c1, c2, c3 = st.columns(3)
c1.metric("OOS accuracy", combined.get("oos_accuracy"))
c2.metric("Naive baseline", combined.get("naive_baseline"))
c3.metric("Beats baseline?", "Yes" if combined.get("beats_baseline") else "No")

n_validated = len(entry["production_models"])
st.subheader("Validated metrics used in composite score")
if n_validated == 0:
    st.warning("No metric currently beats its naive baseline out-of-sample. "
               "The live score below will say 'no edge detected' -- this is "
               "the honest result from the last training run, not a bug.")
else:
    st.dataframe(pd.DataFrame(entry["production_models"]).T)

# ---- Fast path: pull only TODAY's values and apply the saved model ----
st.subheader(f"Today's live composite score ({ticker}, {horizon})")
with st.spinner("Pulling today's values..."):
    try:
        price_df = get_price_history(ticker, period="6mo")
        vix_df = get_vix_family(period="6mo")
        try:
            putcall_df = get_putcall_ratio()
        except Exception:
            putcall_df = None
        latest = build_current_features(price_df, vix_df, putcall_df)
    except RuntimeError as e:
        st.error(f"Couldn't pull today's data: {e}")
        st.stop()

st.write(f"As of {latest.name.date()}:")
st.dataframe(latest.to_frame("current value"))

result = score_with_saved_models(latest.to_dict(), entry["production_models"])
if result["composite_score"] is None:
    st.error(result["detail"])
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Direction", result["direction"].upper())
    c2.metric("Composite up-probability", result["composite_score"])
    c3.metric("Confidence (0-1)", result["confidence"])
    st.write("Per-metric contribution:")
    st.dataframe(pd.DataFrame(result["contributions"]))
