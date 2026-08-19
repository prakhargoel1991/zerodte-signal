"""
app.py

Run with:  streamlit run app.py

Live dashboard: pulls fresh data, shows current metric readings, the
validated backtest table, and today's composite 0DTE/1DTE score.
"""

import streamlit as st
import pandas as pd

from data_sources import get_price_history, get_vix_family
from features import build_feature_matrix
from backtest import run_all_metrics
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
    price_df = get_price_history(ticker, period="2y")
    vix_df = get_vix_family(period="2y")
    df = build_feature_matrix(price_df, vix_df, horizon=horizon)

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
