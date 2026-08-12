"""K-line charting: historical OHLCV candlesticks + forecast overlay (M5).

The forecast is a *deterministic, labelled-as-speculative* path: the endpoint
is anchored to the M1 ``ExecutionAdvice`` (target price for buy ratings,
stop-loss for sell ratings) or a rating coefficient fallback, and the
intermediate candles are a reproducible random walk (fixed seed). The LLM
never generates OHLC arrays — that would be hallucination-prone and
unverifiable. Forecast candles are tagged ``forecast: True`` so the UI can
style them distinctly.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.interface import route_to_vendor

DEFAULT_FORECAST_DAYS = 10
_HISTORY_CANDLES = 120

_FALLBACK_ENDPOINT = {
    "Buy": 1.08,
    "Overweight": 1.05,
    "Hold": 1.0,
    "Underweight": 0.95,
    "Sell": 0.90,
}
_BUY_RATINGS = {"Buy", "Overweight"}
_SELL_RATINGS = {"Underweight", "Sell"}


# ---------------------------------------------------------------------------
# Markdown advice → typed levels
# ---------------------------------------------------------------------------


def advice_from_markdown(md: str) -> Optional[dict[str, float]]:
    """Extract stop_loss / target_price from a rendered ExecutionAdvice.

    Returns None for placeholders (N/A levels) so callers fall back to the
    rating-coefficient path.
    """
    if not md:
        return None
    stop = _level_from_line(md, "Stop Loss")
    target = _level_from_line(md, "Target Price")
    if stop is None or target is None or stop <= 0 or target <= 0:
        return None
    return {"stop_loss": stop, "target_price": target}


def _level_from_line(md: str, label: str) -> Optional[float]:
    for line in md.splitlines():
        if line.startswith(f"**{label}**"):
            m = re.search(r"[-+]?\d+\.?\d*", line.split("**", 2)[-1])
            if m:
                return float(m.group())
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_ohlcv_csv(text: str) -> Optional[pd.DataFrame]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        return None
    try:
        return pd.read_csv(pd.io.common.StringIO("\n".join(lines)))
    except Exception:  # noqa: BLE001
        return None


def load_ohlcv(ticker: str, trade_date: str, ohlcv_text: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load recent OHLCV via the vendor path (injectable for tests)."""
    if ohlcv_text is None:
        from datetime import datetime, timedelta

        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        start_dt = (end_dt - timedelta(days=250)).strftime("%Y-%m-%d")
        try:
            ohlcv_text = route_to_vendor("get_stock_data", ticker, start_dt, trade_date)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(ohlcv_text, str) or ohlcv_text.startswith("K线数据获取失败"):
        return None
    df = _parse_ohlcv_csv(ohlcv_text)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df.sort_values("Date").tail(_HISTORY_CANDLES).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecast synthesis
# ---------------------------------------------------------------------------


def _forecast_endpoint(last_close: float, rating: str, advice: Optional[dict]) -> float:
    """Endpoint price for the forecast path."""
    if advice:
        if rating in _BUY_RATINGS:
            return advice["target_price"]
        if rating in _SELL_RATINGS:
            return advice["stop_loss"]
    return last_close * _FALLBACK_ENDPOINT.get(rating, 1.0)


def synthesize_forecast(
    hist: pd.DataFrame,
    rating: str,
    advice: Optional[dict] = None,
    days: int = DEFAULT_FORECAST_DAYS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build ``days`` speculative candles from the last close to the endpoint.

    Each candle: open = previous close, close = path point, high/low = path
    point ± noise, all tagged ``forecast: True``. Fixed seed → reproducible.
    """
    last_close = float(hist.iloc[-1]["Close"])
    end = _forecast_endpoint(last_close, rating, advice)
    rng = np.random.default_rng(seed)
    path = np.linspace(last_close, end, days + 1) + rng.normal(0, last_close * 0.004, days + 1)

    rows: list[dict[str, Any]] = []
    prev_close = last_close
    for i in range(days):
        o, c = prev_close, float(path[i + 1])
        hi = max(o, c) + abs(float(rng.normal(0, last_close * 0.006)))
        lo = min(o, c) - abs(float(rng.normal(0, last_close * 0.006)))
        rows.append(
            {
                "open": round(o, 2), "high": round(hi, 2),
                "low": round(lo, 2), "close": round(c, 2),
                "forecast": True,
            }
        )
        prev_close = c
    return rows


def build_chart_data(
    ticker: str,
    trade_date: str,
    advice_md: Optional[str] = None,
    rating: str = "Hold",
    forecast_days: int = DEFAULT_FORECAST_DAYS,
    ohlcv_text: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build the full chart payload ``{ticker, history, forecast, rating}``.

    Returns None when OHLCV cannot be loaded (UI shows a caption instead).
    """
    hist = load_ohlcv(ticker, trade_date, ohlcv_text)
    if hist is None:
        return None
    rating = parse_rating(rating, default="Hold")
    advice = advice_from_markdown(advice_md)
    history = [
        {
            "index": i,
            "open": float(r["Open"]), "high": float(r["High"]),
            "low": float(r["Low"]), "close": float(r["Close"]),
        }
        for i, r in hist.iterrows()
    ]
    forecast = synthesize_forecast(hist, rating, advice, days=forecast_days)
    return {
        "ticker": ticker,
        "history": history,
        "forecast": forecast,
        "rating": rating,
    }
