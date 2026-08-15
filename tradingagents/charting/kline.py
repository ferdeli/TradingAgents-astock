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

import hashlib
import json
import os
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
# Disk cache: {data_cache_dir}/kline/{ticker}_{trade_date}_{advice_hash}.json
# ---------------------------------------------------------------------------


def _kline_cache_dir() -> str:
    """Cache directory for chart payloads (data_cache_dir/kline)."""
    from tradingagents.dataflows.config import get_config

    base = get_config().get("data_cache_dir") or os.path.expanduser("~/.tradingagents/cache")
    path = os.path.join(base, "kline")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_key(ticker: str, trade_date: str, advice_md: Optional[str]) -> str:
    """Stable per-analysis cache key: ticker + date + advice signature.

    The advice hash keeps two analyses of the same ticker/date with different
    execution levels from sharing a forecast. History reports reuse the exact
    advice they were generated with, so they always hit the cache.
    """
    sig = hashlib.md5((advice_md or "").encode("utf-8")).hexdigest()[:8]
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", str(ticker))
    return f"{safe_ticker}_{trade_date}_{sig}.json"


def _load_disk_cache(ticker: str, trade_date: str, advice_md: Optional[str]) -> Optional[dict]:
    path = os.path.join(_kline_cache_dir(), _cache_key(ticker, trade_date, advice_md))
    if not os.path.exists(path):
        return None
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (OSError, ValueError):
        return None


def _save_disk_cache(ticker: str, trade_date: str, advice_md: Optional[str], chart: dict) -> None:
    path = os.path.join(_kline_cache_dir(), _cache_key(ticker, trade_date, advice_md))
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(chart, fh, ensure_ascii=False)
    except OSError:
        pass  # cache write must never break charting


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
    use_disk_cache: bool = True,
    offline: bool = False,
) -> Optional[dict[str, Any]]:
    """Build the full chart payload ``{ticker, history, forecast, rating}``.

    Returns None when OHLCV cannot be loaded (UI shows a caption instead).

    With ``use_disk_cache`` (default) the payload is cached per
    ticker+date+advice on disk, so rendering a report — including reopening a
    historical one — hits the cache instead of re-fetching OHLCV over the
    network. ``ohlcv_text`` injection bypasses the cache entirely (tests).

    ``offline=True`` (history browsing): the disk cache is the ONLY data
    source — a cache miss returns None immediately instead of fetching OHLCV
    over the network, so browsing a historical report never blocks the page
    on mootdx/sina requests.
    """
    if use_disk_cache and ohlcv_text is None:
        cached = _load_disk_cache(ticker, trade_date, advice_md)
        if cached is not None:
            return cached
    if offline:
        return None  # cache miss while browsing history: degrade fast, no network

    hist = load_ohlcv(ticker, trade_date, ohlcv_text)
    if hist is None:
        return None
    rating = parse_rating(rating, default="Hold")
    advice = advice_from_markdown(advice_md)
    history = [
        {
            "index": i,
            "date": str(r["Date"]),
            "open": float(r["Open"]), "high": float(r["High"]),
            "low": float(r["Low"]), "close": float(r["Close"]),
        }
        for i, r in hist.iterrows()
    ]
    forecast = synthesize_forecast(hist, rating, advice, days=forecast_days)
    chart = {
        "ticker": ticker,
        "history": history,
        "forecast": forecast,
        "rating": rating,
    }
    if use_disk_cache and ohlcv_text is None:
        _save_disk_cache(ticker, trade_date, advice_md, chart)
    return chart
