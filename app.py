"""
app.py

Run with:  streamlit run app.py

Live dashboard: pulls fresh data, shows current metric readings, the
validated backtest table, and today's composite 0DTE/1DTE score.
"""

import streamlit as st
import pandas as pd

from data_sources import get_price_history, get_vix_family, get_putcall_ratio
from features import build_feature_matrix
from backtest import run_all_metrics, walk_forward_combined
from scoring import fit_weights, score_today

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

with st.spinner("Pulling data..."):
    try:
        price_df = get_price_history(ticker, period="5y")
        vix_df = get_vix_family(period="5y")
    except RuntimeError as e:
        st.error(f"Data pull failed: {e}")
        st.stop()

    try:
        putcall_df = get_putcall_ratio()
    except Exception as e:
        putcall_df = None
        st.info(f"Put/call ratio source unavailable right now ({e}); continuing without it.")

    df = build_feature_matrix(price_df, vix_df, horizon=horizon, putcall_df=putcall_df)

if df.empty:
    st.error(
        "No rows survived after aligning price and VIX data (they may not "
        "share any overlapping dates). Try rebooting the app from the "
        "Streamlit 'Manage app' menu, or come back in a few minutes."
    )
    st.stop()

st.subheader("Latest metric readings")
st.dataframe(df.tail(5))

st.subheader("Walk-forward validation (per metric)")
backtest_results = run_all_metrics(df)
st.dataframe(backtest_results)
st.caption(
    "oos_accuracy = out-of-sample directional accuracy using ONLY this metric. "
    "naive_baseline = accuracy from always guessing the majority class. "
    "A metric only earns weight below if oos_accuracy > naive_baseline."
)

st.subheader("Combined model (all metrics together, walk-forward)")
combined = walk_forward_combined(df)
c1, c2, c3 = st.columns(3)
c1.metric("OOS accuracy", combined["oos_accuracy"])
c2.metric("Naive baseline", combined["naive_baseline"])
c3.metric("Beats baseline?", "Yes" if combined["beats_baseline"] else "No")
st.caption(
    "This fits all metrics jointly instead of one at a time -- catches signal "
    "that only appears in combination. Still walk-forward (no lookahead)."
)

weights = fit_weights(backtest_results)
st.subheader("Validated weights used in composite score")
if weights.empty:
    st.warning("No metric currently beats its naive baseline out-of-sample. "
               "The composite score below will say 'no edge detected' -- this "
               "is the honest result, not a bug.")
else:
    st.dataframe(weights)

st.subheader(f"Today's composite score ({ticker}, {horizon})")
result = score_today(df, weights)
if result["composite_score"] is None:
    st.error(result["detail"])
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Direction", result["direction"].upper())
    c2.metric("Composite up-probability", result["composite_score"])
    c3.metric("Confidence (0-1)", result["confidence"])
    st.write("Per-metric contribution:")
    st.dataframe(pd.DataFrame(result["contributions"]))
    st.caption(f"As of {result['as_of']}")
