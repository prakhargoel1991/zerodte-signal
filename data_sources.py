"""
data_sources.py

All external data pulls live here, isolated from feature/scoring logic, so a
data source can be swapped (e.g. yfinance -> Tradier/Polygon for real greeks)
without touching anything downstream.

NOTE: This module requires normal internet access to yfinance / CBOE. It will
NOT run inside a network-sandboxed environment. Run it on your own machine or
a host with unrestricted egress.
"""

import io
import datetime as dt
import pandas as pd
import numpy as np
import requests
import yfinance as yf

CBOE_PUTCALL_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/totalpc.csv"
# CBOE occasionally changes this endpoint's exact path/format. If this 404s,
# search "CBOE total put call ratio historical csv" for the current URL and
# update this constant -- nothing else needs to change.


def get_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history for SPY/QQQ (or any ticker)."""
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_vix_family(period: str = "2y") -> pd.DataFrame:
    """VIX and VIX9D closes, plus the term-structure ratio VIX9D/VIX.

    Ratio < 1 = backwardation (near-term fear elevated, common pre-event)
    Ratio > 1 = contango (calm regime)
    """
    vix = yf.Ticker("^VIX").history(period=period)["Close"].rename("VIX")
    vix9d = yf.Ticker("^VIX9D").history(period=period)["Close"].rename("VIX9D")
    df = pd.concat([vix, vix9d], axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["VIX9D_VIX_ratio"] = df["VIX9D"] / df["VIX"]
    return df.dropna()


def get_putcall_ratio() -> pd.DataFrame:
    """CBOE total put/call ratio, daily. See CBOE_PUTCALL_URL note above."""
    resp = requests.get(CBOE_PUTCALL_URL, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # CBOE format varies; normalize expected columns defensively.
    date_col = next(c for c in df.columns if "date" in c.lower())
    ratio_col = next(c for c in df.columns if "ratio" in c.lower() or "p/c" in c.lower())
    df = df[[date_col, ratio_col]].rename(columns={date_col: "Date", ratio_col: "PutCallRatio"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def get_option_chain_snapshot(ticker: str) -> dict:
    """Current option chain across all available expiries for `ticker`.

    Used to build a same-day OI-based gamma/positioning proxy. This is a
    coarse substitute for real dealer gamma exposure (which needs official
    greeks + market-maker net positioning data you don't get for free) -- but
    call/put OI skew near-the-money is a reasonable directional proxy.
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options
    chains = {}
    for exp in expiries:
        chain = tk.option_chain(exp)
        chains[exp] = {"calls": chain.calls, "puts": chain.puts}
    return chains


def get_futures_overnight_gap(cash_ticker: str, futures_ticker: str) -> pd.DataFrame:
    """Approximate overnight gap using futures continuous contract vs prior
    cash close. ES=F / NQ=F track SPY/QQQ respectively.
    """
    cash = yf.Ticker(cash_ticker).history(period="2y")[["Close"]].rename(columns={"Close": "prev_close"})
    cash.index = pd.to_datetime(cash.index).tz_localize(None)
    fut = yf.Ticker(futures_ticker).history(period="2y")[["Open"]].rename(columns={"Open": "next_open_fut"})
    fut.index = pd.to_datetime(fut.index).tz_localize(None)
    df = cash.join(fut, how="inner")
    df["gap_pct"] = (df["next_open_fut"] - df["prev_close"]) / df["prev_close"] * 100
    return df
