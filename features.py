"""
features.py

Turns raw price/vol/positioning data into a single dataframe of per-day
metric values, aligned with the target we're trying to predict:
same-session (0DTE) or next-session (1DTE) direction of SPY/QQQ.

Add a new metric candidate by writing one function here that returns a
pd.Series indexed by date, then adding it to build_feature_matrix().
That's the only place new metrics need to be registered.
"""

import numpy as np
import pandas as pd


def realized_vol(price_df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Annualized realized vol of close-to-close log returns."""
    ret = np.log(price_df["Close"] / price_df["Close"].shift(1))
    return (ret.rolling(window).std() * np.sqrt(252) * 100).rename(f"RV_{window}d")


def rv_iv_spread(price_df: pd.DataFrame, vix: pd.Series, window: int = 10) -> pd.Series:
    """Realized vol minus VIX. Very negative = options 'expensive' relative
    to what's actually happening; some traders read this as mean-reversion
    pressure on IV (indirectly on direction too, via vol-selling flows)."""
    rv = realized_vol(price_df, window)
    return (rv - vix).rename("RV_IV_spread")


def prior_day_range_pct(price_df: pd.DataFrame) -> pd.Series:
    """Prior day's high-low range as % of close -- a crude 'how much did it
    move already' regime signal for 0DTE premium selling vs buying."""
    rng = (price_df["High"] - price_df["Low"]) / price_df["Close"] * 100
    return rng.shift(1).rename("prior_range_pct")

def rsi(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = price_df["Close"].diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / down
    return (100 - 100 / (1 + rs)).rename(f"RSI_{window}")


def day_of_week(price_df: pd.DataFrame) -> pd.Series:
    """0=Mon ... 4=Fri. 0DTE flow/behavior is known to differ Mon vs Fri (opex)."""
    return pd.Series(price_df.index.dayofweek, index=price_df.index, name="day_of_week")


def call_put_oi_skew(chain_calls: pd.DataFrame, chain_puts: pd.DataFrame, spot: float, window_pct: float = 0.02) -> float:
    """Near-the-money (+/- window_pct) call OI minus put OI, normalized.
    Positive = more call-side open interest nearby -> often read as resistance
    (dealers short calls hedge by selling into rallies) rather than bullish."""
    lo, hi = spot * (1 - window_pct), spot * (1 + window_pct)
    calls_ntm = chain_calls[(chain_calls["strike"] >= lo) & (chain_calls["strike"] <= hi)]["openInterest"].sum()
    puts_ntm = chain_puts[(chain_puts["strike"] >= lo) & (chain_puts["strike"] <= hi)]["openInterest"].sum()
    total = calls_ntm + puts_ntm
    if total == 0:
        return np.nan
    return (calls_ntm - puts_ntm) / total


def build_target(price_df: pd.DataFrame, horizon: str = "0DTE") -> pd.Series:
    """
    horizon="0DTE": same-session open->close return sign (you buy at/near open,
                    the option expires same day at close).
    horizon="1DTE": next session's open->close return sign (you buy today for
                    an option expiring tomorrow).
    Returns 1 for up, 0 for down, aligned to the date the SIGNAL is measured on.
    """
    if horizon == "0DTE":
        ret = (price_df["Close"] - price_df["Open"]) / price_df["Open"]
        target = (ret > 0).astype(int)
    elif horizon == "1DTE":
        next_ret = (price_df["Close"].shift(-1) - price_df["Open"].shift(-1)) / price_df["Open"].shift(-1)
        target = (next_ret > 0).astype(int)
    else:
        raise ValueError("horizon must be '0DTE' or '1DTE'")
    return target.rename("target_up")


def build_current_features(price_df: pd.DataFrame, vix_df: pd.DataFrame, putcall_df: pd.DataFrame = None) -> pd.Series:
    """Same metric set as build_feature_matrix, but for TODAY only -- no
    target column (today's outcome isn't known yet), so this is safe to
    call live without needing historical data for backtesting. Used by the
    dashboard for the fast, no-backtest "current score" path.
    """
    feats = pd.concat(
        [
            vix_df["VIX"],
            vix_df["VIX9D_VIX_ratio"],
            rv_iv_spread(price_df, vix_df["VIX"]),
            prior_day_range_pct(price_df),
            rsi(price_df),
            day_of_week(price_df),
        ],
        axis=1,
    )
    if putcall_df is not None and not putcall_df.empty:
        feats = feats.join(putcall_df["PutCallRatio"], how="left")
    feats = feats.dropna()
    return feats.iloc[-1]


def build_feature_matrix(price_df: pd.DataFrame, vix_df: pd.DataFrame, horizon: str = "0DTE",
                          putcall_df: pd.DataFrame = None) -> pd.DataFrame:
    """Assemble all registered metrics + target into one aligned dataframe.
    This is the single place that defines "what metrics are we testing".

    putcall_df is optional (pass None if the CBOE pull failed) so the whole
    app doesn't break if that one external source is unavailable.
    """
    feats = pd.concat(
        [
            vix_df["VIX"],
            vix_df["VIX9D_VIX_ratio"],
            rv_iv_spread(price_df, vix_df["VIX"]),
            prior_day_range_pct(price_df),
            rsi(price_df),
            day_of_week(price_df),
        ],
        axis=1,
    )
    if putcall_df is not None and not putcall_df.empty:
        feats = feats.join(putcall_df["PutCallRatio"], how="left")

    target = build_target(price_df, horizon=horizon)
    df = feats.join(target, how="inner").dropna()
    return df
