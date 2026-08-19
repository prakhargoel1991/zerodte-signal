# SPY/QQQ 0-1 DTE Composite Signal

A dashboard that: pulls the metrics traders commonly cite for 0DTE/1DTE
options → validates each one walk-forward against actual direction (not just
in-sample correlation) → combines only the metrics that show real
out-of-sample edge into one weighted score.

## Why walk-forward, not just correlation

A metric that correlates with direction over the last 2 years might just be
fitting noise -- with ~500 trading days you can find "signals" that look
great in-sample and are worthless going forward. `backtest.py` retrains a
simple model at every step using only *past* data and scores it on the next
unseen day, which is a much more honest test. Metrics that don't beat a
naive "always guess the majority direction" baseline out-of-sample get
flagged and excluded from the composite score, not silently kept.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires normal internet access (this pulls live from Yahoo Finance / yfinance).
Won't work in a network-sandboxed environment.

## What's implemented now vs. what's a stub

**Implemented and wired into the dashboard:**
- VIX level, VIX9D/VIX term-structure ratio
- Realized-vol vs implied-vol (VIX) spread
- Prior day's range %
- RSI(14)
- Day of week

**Written but not yet wired into `build_feature_matrix()`** (in
`data_sources.py`, ready to plug in — the reason they're separate is each
needs a bit of care before trusting it):
- `get_putcall_ratio()` — pulls CBOE's total put/call ratio CSV. The exact
  CBOE URL/format changes periodically; verify `CBOE_PUTCALL_URL` still
  resolves before relying on it.
- `get_option_chain_snapshot()` + `call_put_oi_skew()` — a same-day near-the-money
  call/put open-interest skew, as a free-tier proxy for dealer gamma
  positioning. This is NOT real gamma exposure (that needs official greeks
  and market-maker net position data, which isn't free) — it's directional
  evidence, not the real thing. Label it as such if you show it to yourself
  in the dashboard.
- `get_futures_overnight_gap()` — overnight futures gap vs prior cash close.

To add any of these to the composite score: write a one-line feature
function in `features.py` (pattern is already established), add it to the
list in `build_feature_matrix()`, and it automatically gets backtested and
weighted next run — no other code changes needed.

## Honest limitations to keep in mind

1. **Daily granularity.** This scores direction for the session, not intraday
   timing. If you want intraday (e.g., "is the signal still valid at 11am"),
   the architecture supports it but the data pulls need to move to intraday
   bars, which yfinance only gives you ~60 days of at 1-minute resolution.
2. **Small sample.** ~500 daily observations for 2 years of history is not a
   lot for walk-forward testing several metrics — expect the "beats
   baseline" list to shift as more data accumulates or market regime
   changes. Re-run the backtest periodically, don't treat today's weights as
   permanent.
3. **This is a research/decision-support tool, not a trading signal
   guarantee.** Composite probability near 0.5 means no real edge that day,
   even if the dashboard shows a direction — check the `confidence` value,
   not just direction.

## Suggested next steps

1. Run it, look at which metrics currently beat baseline for SPY vs QQQ,
   0DTE vs 1DTE separately — they likely differ.
2. Wire in put/call ratio and the OI-skew proxy once you've verified the
   CBOE CSV pull still works.
3. If the free OI-based gamma proxy is too noisy, that's the natural place
   to pay for a real data feed (Tradier free tier has actual greeks; Polygon
   or ORATS if you want cleaner historical options data for backtesting).
