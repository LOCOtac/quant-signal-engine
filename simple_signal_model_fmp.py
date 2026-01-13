from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import os
import numpy as np
import pandas as pd
import requests

FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"


# =========================
# FMP DATA LOADER (stable)
# =========================
def fetch_prices_fmp(symbol: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Endpoint:
      https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=TSLA&apikey=...

    Response shape may be either:
      A) {"symbol": "TSLA", "historical": [ ... ]}
      B) [ ... ]  (list directly)

    This loader supports both.
    Returns: df indexed by date ascending with columns: Open, High, Low, Close, Volume
    """
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("Missing FMP_API_KEY. Set it: export FMP_API_KEY='YOUR_KEY'")

    url = f"{FMP_STABLE_BASE}/historical-price-eod/full"
    params = {"symbol": symbol.upper(), "apikey": api_key}

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FMP error {r.status_code}: {r.text[:800]}")

    data = r.json()

    # shape handling
    if isinstance(data, dict):
        hist = data.get("historical")
        if hist is None:
            # sometimes APIs return list under different keys; keep a fallback
            hist = data.get("data") or data.get("results")
    elif isinstance(data, list):
        hist = data
    else:
        raise ValueError(f"Unexpected JSON type from FMP: {type(data)}")

    if not hist:
        raise ValueError(f"No historical data returned for {symbol}. Response head: {str(data)[:200]}")

    df = pd.DataFrame(hist)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Got columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# =========================
# INDICATORS
# =========================
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI using EWMA smoothing.
    """
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.bfill()


# =========================
# STRUCTURE (HH/HL vs LH/LL)
# =========================
def structure_signal(df: pd.DataFrame, lookback: int = 40) -> int:
    """
    Pivot-based structure:
      HH & HL => +1
      LH & LL => -1
      else => 0
    """
    if len(df) < lookback + 5:
        return 0

    d = df.iloc[-lookback:].copy()
    highs = d["High"].to_numpy()
    lows = d["Low"].to_numpy()

    swing_high_idx = []
    swing_low_idx = []

    for i in range(2, len(d) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 2]:
            swing_high_idx.append(i)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 2]:
            swing_low_idx.append(i)

    if len(swing_high_idx) < 2 or len(swing_low_idx) < 2:
        return 0

    h1, h2 = swing_high_idx[-2], swing_high_idx[-1]
    l1, l2 = swing_low_idx[-2], swing_low_idx[-1]

    high_prev, high_last = float(highs[h1]), float(highs[h2])
    low_prev, low_last = float(lows[l1]), float(lows[l2])

    if high_last > high_prev and low_last > low_prev:
        return +1
    if high_last < high_prev and low_last < low_prev:
        return -1
    return 0


# =========================
# MODEL
# =========================
@dataclass
class SignalResult:
    symbol: str
    label: str
    total_score: int
    components: Dict[str, int]
    snapshot: Dict[str, float]


def trend_signal(df: pd.DataFrame) -> Tuple[int, float, float, float]:
    """
    Trend vote:
      Close > EMA20 > EMA50 => +1
      Close < EMA20 < EMA50 => -1
      else 0
    """
    close = df["Close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)

    c = float(close.iloc[-1])
    v20 = float(e20.iloc[-1])
    v50 = float(e50.iloc[-1])

    if c > v20 > v50:
        return +1, c, v20, v50
    if c < v20 < v50:
        return -1, c, v20, v50
    return 0, c, v20, v50


def momentum_signal(df: pd.DataFrame) -> Tuple[int, float]:
    """
    RSI vote:
      RSI > 60 => +1
      RSI < 40 => -1
      else 0
    """
    rv = float(rsi(df["Close"], 14).iloc[-1])
    if rv > 60:
        return +1, rv
    if rv < 40:
        return -1, rv
    return 0, rv


def long_term_filter(df: pd.DataFrame) -> Tuple[int, float]:
    """
    SMA200 filter (not part of score, just a helpful snapshot):
      Close > SMA200 => +1
      Close < SMA200 => -1
      insufficient history => 0, nan
    """
    close = df["Close"]
    s200 = sma(close, 200)

    if len(s200) == 0 or np.isnan(s200.iloc[-1]):
        return 0, float("nan")

    s200v = float(s200.iloc[-1])
    return (+1 if float(close.iloc[-1]) > s200v else -1), s200v


def label_from_score(score: int) -> str:
    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"


def run_simple_model(symbol: str, df: pd.DataFrame) -> SignalResult:
    t, c, e20, e50 = trend_signal(df)
    m, rsi_val = momentum_signal(df)
    s = structure_signal(df, lookback=40)

    total = int(t + m + s)
    label = label_from_score(total)

    sma200_state, sma200_val = long_term_filter(df)

    return SignalResult(
        symbol=symbol.upper(),
        label=label,
        total_score=total,
        components={"trend": int(t), "momentum_rsi": int(m), "structure": int(s)},
        snapshot={
            "close": float(c),
            "ema20": float(e20),
            "ema50": float(e50),
            "rsi14": float(rsi_val),
            "sma200": float(sma200_val),
            "sma200_filter": float(sma200_state),
        },
    )


# =========================
# OUTPUT
# =========================
def format_result(res: SignalResult) -> str:
    """
    Short summary output.
    """
    snap = res.snapshot
    sma200_val = snap["sma200"]
    sma200_str = "nan" if np.isnan(sma200_val) else f"{sma200_val:.4f}"

    return (
        f"\n{res.symbol} => {res.label} (score={res.total_score})\n"
        f"components: {res.components}\n"
        f"snapshot: close={snap['close']:.4f}, "
        f"ema20={snap['ema20']:.4f}, ema50={snap['ema50']:.4f}, "
        f"rsi14={snap['rsi14']:.2f}, sma200={sma200_str}, "
        f"sma200_filter={int(snap['sma200_filter'])}\n"
    )


def explain_scoring(res: SignalResult) -> str:
    """
    Detailed explanation of how the final score and label were produced.
    """
    c = res.components
    s = res.snapshot

    def vote_text(name: str, v: int) -> str:
        if v == 1:
            return f"{name}: +1 (Bullish)"
        if v == -1:
            return f"{name}: -1 (Bearish)"
        return f"{name}:  0 (Neutral)"

    trend_rule = (
        "Trend rule:\n"
        "  +1 if Close > EMA20 > EMA50\n"
        "  -1 if Close < EMA20 < EMA50\n"
        "   0 otherwise\n"
        f"  Current: Close={s['close']:.4f}, EMA20={s['ema20']:.4f}, EMA50={s['ema50']:.4f}"
    )

    rsi_rule = (
        "Momentum (RSI) rule:\n"
        "  +1 if RSI14 > 60\n"
        "  -1 if RSI14 < 40\n"
        "   0 otherwise\n"
        f"  Current: RSI14={s['rsi14']:.2f}"
    )

    structure_rule = (
        "Structure rule (HH/HL vs LH/LL):\n"
        "  +1 if last swing high is higher AND last swing low is higher (HH & HL)\n"
        "  -1 if last swing high is lower  AND last swing low is lower  (LH & LL)\n"
        "   0 otherwise / not enough pivots\n"
        "  Current: uses last ~40 bars to find pivots"
    )

    scoring = (
        "TOTAL SCORE = Trend + Momentum(RSI) + Structure\n"
        "Label mapping:\n"
        "  BULLISH if score >= +2\n"
        "  BEARISH if score <= -2\n"
        "  NEUTRAL otherwise\n"
    )

    sma200_str = "nan" if np.isnan(s["sma200"]) else f"{s['sma200']:.4f}"

    lines = [
        "\n=== SCORING EXPLANATION ===",
        vote_text("Trend", c["trend"]),
        vote_text("Momentum(RSI)", c["momentum_rsi"]),
        vote_text("Structure", c["structure"]),
        f"TOTAL: {res.total_score}  =>  {res.label}",
        "",
        trend_rule,
        "",
        rsi_rule,
        "",
        structure_rule,
        "",
        scoring,
        "SMA200 filter (not part of score):",
        "  +1 if Close > SMA200, -1 if Close < SMA200, 0 if SMA200 not available",
        f"  Current: SMA200={sma200_str}, filter={int(s['sma200_filter'])}",
    ]
    return "\n".join(lines)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple Bullish/Neutral/Bearish model using FMP stable EOD data"
    )
    parser.add_argument("symbol", type=str, help="Ticker symbol, e.g. TSLA")
    parser.add_argument("--short", action="store_true", help="Use ~1 year of data (~260 rows)")
    parser.add_argument("--explain", action="store_true", help="Print scoring explanation")
    args = parser.parse_args()

    df = fetch_prices_fmp(args.symbol)

    if args.short:
        df = df.iloc[-260:]

    res = run_simple_model(args.symbol, df)
    print(format_result(res))

    print("\n--- Component Meaning ---")
    print("trend: 1  = Bullish trend (Close > EMA20 > EMA50)")
    print("trend: 0  = Neutral trend (mixed EMA alignment)")
    print("trend: -1 = Bearish trend (Close < EMA20 < EMA50)")
    print("")
    print("momentum_rsi: 1  = Bullish momentum (RSI > 60)")
    print("momentum_rsi: 0  = Neutral momentum (RSI between 40 and 60)")
    print("momentum_rsi: -1 = Bearish momentum (RSI < 40)")
    print("")
    print("structure: 1  = Bullish structure (Higher High + Higher Low)")
    print("structure: 0  = Neutral structure (mixed / no clear HH-HL or LH-LL)")
    print("structure: -1 = Bearish structure (Lower High + Lower Low)")
    print("-------------------------\n")


    if args.explain:
        print(explain_scoring(res))
