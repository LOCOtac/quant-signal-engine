# app.py
# Railway-ready FastAPI backend + your full signal model in one file.
# Endpoints:
#   GET /health
#   GET /analyze?symbol=TSLA&short=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import os
import math
import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Quant Signal Engine", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "quant-signal-engine"}


@app.get("/analyze")
def analyze(
    symbol: str = Query(..., description="Stock ticker (e.g., TSLA, AMD)"),
    short: bool = Query(False, description="Use ~1 year of data (~260 rows)"),
):
    result = run_signal_analysis(symbol, short=short)

    # Return 400 on user input issues, 500 on other errors
    if result.get("final_label") == "error":
        msg = (result.get("error") or "").lower()
        if "missing symbol" in msg or "no historical data" in msg:
            return JSONResponse(status_code=400, content=result)
        return JSONResponse(status_code=500, content=result)

    return result


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

    Returns: df indexed by date ascending with columns: Open, High, Low, Close, Volume
    """
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("Missing FMP_API_KEY. Set it as an environment variable in Railway.")

    symbol = symbol.strip().upper()
    url = f"{FMP_STABLE_BASE}/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": api_key}

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FMP error {r.status_code}: {r.text[:800]}")

    data = r.json()

    # shape handling
    if isinstance(data, dict):
        hist = data.get("historical")
        if hist is None:
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
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
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
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 2]
        ):
            swing_high_idx.append(i)
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 2]
        ):
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
    SMA200 filter (not part of score, just snapshot):
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
# JSON-SAFE BACKEND WRAPPER
# =========================
def _nan_to_none(x: Any):
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return x
    except Exception:
        return None


def run_signal_analysis(symbol: str, short: bool = False) -> dict:
    """
    Returns a JSON-friendly dict:
      symbol
      final_label
      final_score
      components {trend, momentum_rsi, structure}
      values_used {close, ema20, ema50, rsi14, sma200, sma200_filter}
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {
            "symbol": symbol,
            "final_label": "error",
            "final_score": 0,
            "components": {"trend": None, "momentum_rsi": None, "structure": None},
            "values_used": {},
            "error": "Missing symbol",
        }

    try:
        df = fetch_prices_fmp(symbol, api_key=os.getenv("FMP_API_KEY"))
        if df is None or df.empty:
            return {
                "symbol": symbol,
                "final_label": "error",
                "final_score": 0,
                "components": {"trend": None, "momentum_rsi": None, "structure": None},
                "values_used": {},
                "error": "No price data returned",
            }

        if short:
            df = df.iloc[-260:]

        res = run_simple_model(symbol, df)

        values_used = {k: _nan_to_none(v) for k, v in res.snapshot.items()}
        components = {k: int(v) for k, v in res.components.items()}

        return {
            "symbol": res.symbol,
            "final_label": res.label,
            "final_score": int(res.total_score),
            "components": components,
            "values_used": values_used,
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "final_label": "error",
            "final_score": 0,
            "components": {"trend": None, "momentum_rsi": None, "structure": None},
            "values_used": {},
            "error": str(e),
        }
