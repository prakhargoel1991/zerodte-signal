"""
data_sources.py

All external data pulls live here, isolated from feature/scoring logic, so a
data source can be swapped without touching anything downstream.

IMPORTANT: Yahoo Finance (yfinance) actively rate-limits requests at an
authentication step ("crumb" cookie) shared by every method it offers --
this affects shared cloud hosts like Streamlit Community Cloud especially,
since many unrelated apps share the same outbound IP pool and collectively
trip Yahoo's limiter. When that happens, retrying via a different yfinance
method does NOT help, because the block happens before the actual data
request.

To work around this, price/VIX history is pulled from Stooq FIRST (a free
source that needs no authentication/cookies at all, so it isn't affected by
Yahoo's rate limiter), and only falls back to yfinance if Stooq doesn't have
what's needed. I have not been able to test the Stooq path myself (no
internet access to non-package-registry sites in my build environment) --
if Stooq's symbol format has changed, you'll see a clear error naming which
symbol failed, which is what to search for/report back.

NOTE: This module requires normal internet access. It will NOT run inside a
network-sandboxed environment.
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

_PERIOD_DAYS = {
    "1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653,
}


def _period_cutoff(period: str) -> pd.Timestamp:
    days = _PERIOD_DAYS.get(period, 731)
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=days)


def _stooq_daily(symbol: str) -> pd.DataFrame:
    """Pull daily OHLC history from Stooq. No API key, no cookies/auth --
    this is why it's immune to Yahoo's rate limiter. Raises if Stooq has no
    data for this symbol (format is typically 'spy.us', 'qqq.us', '^vix')."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise RuntimeError(f"Stooq returned no usable data for '{symbol}'")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def _fetch_history_yfinance(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """yfinance fallback path. Tries Ticker.history(), then yf.download() --
    both go through the same Yahoo auth step, so if Yahoo is rate-limiting
    this IP, expect BOTH to fail together. That's a real limitation, not a
    bug in this code.
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception:
        df = None
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for '{ticker}' (likely rate-limited -- "
            f"this is a known issue on shared cloud hosts, not a problem with your files). "
            f"The Stooq fallback should normally catch this; if you're seeing this error, "
            f"Stooq's format may have changed too -- worth reporting back."
        )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history for SPY/QQQ (or any ticker). Tries Stooq first."""
    try:
        df = _stooq_daily(f"{ticker.lower()}.us")
        df = df[df.index >= _period_cutoff(period)]
        if not df.empty:
            return df
    except Exception:
        pass  # fall through to yfinance
    return _fetch_history_yfinance(ticker, period, interval)


def get_vix_family(period: str = "2y") -> pd.DataFrame:
    """VIX and VIX9D closes, plus the term-structure ratio VIX9D/VIX.

    Ratio < 1 = backwardation (near-term fear elevated, common pre-event)
    Ratio > 1 = contango (calm regime)
    """
    try:
        vix = _stooq_daily("^vix")["Close"].rename("VIX")
        vix9d = _stooq_daily("^vix9d")["Close"].rename("VIX9D")
        df = pd.concat([vix, vix9d], axis=1).dropna()
        df = df[df.index >= _period_cutoff(period)]
        if not df.empty:
            df["VIX9D_VIX_ratio"] = df["VIX9D"] / df["VIX"]
            return df
    except Exception:
        pass  # fall through to yfinance

    vix = _fetch_history_yfinance("^VIX", period)["Close"].rename("VIX")
    vix9d = _fetch_history_yfinance("^VIX9D", period)["Close"].rename("VIX9D")
    df = pd.concat([vix, vix9d], axis=1)
    df["VIX9D_VIX_ratio"] = df["VIX9D"] / df["VIX"]
    return df.dropna()


def get_putcall_ratio() -> pd.DataFrame:
    """CBOE total put/call ratio, daily. See CBOE_PUTCALL_URL note above."""
    resp = requests.get(CBOE_PUTCALL_URL, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = next(c for c in df.columns if "date" in c.lower())
    ratio_col = next(c for c in df.columns if "ratio" in c.lower() or "p/c" in c.lower())
    df = df[[date_col, ratio_col]].rename(columns={date_col: "Date", ratio_col: "PutCallRatio"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def get_option_chain_snapshot(ticker: str) -> dict:
    """Current option chain across all available expiries for `ticker`.
    Stooq doesn't provide options chains, so this still depends on yfinance
    and will fail if Yahoo is currently rate-limiting this IP.
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
    cash close. ES=F / NQ=F track SPY/QQQ respectively. Futures aren't on
    Stooq under these symbols, so this one still uses yfinance directly.
    """
    cash = _fetch_history_yfinance(cash_ticker, "2y")[["Close"]].rename(columns={"Close": "prev_close"})
    fut = _fetch_history_yfinance(futures_ticker, "2y")[["Open"]].rename(columns={"Open": "next_open_fut"})
    df = cash.join(fut, how="inner")
    df["gap_pct"] = (df["next_open_fut"] - df["prev_close"]) / df["prev_close"] * 100
    return df
