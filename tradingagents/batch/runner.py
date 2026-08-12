"""Batch analysis: candidate pool → per-ticker full analysis → collected results.

Pure orchestration — no report writing here (the CLI layer persists reports
and renders the summary). Each ticker gets its own ``TradingAgentsGraph``
instance so states never bleed into each other; a single failing ticker is
isolated and never aborts the batch (v1 runs serially — the East Money
throttle is module-global, so parallelism would just queue up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.graph.trading_graph import TradingAgentsGraph

ALL_ANALYSTS = ["market", "social", "news", "fundamentals", "policy", "hot_money", "lockup"]
QUICK_ANALYSTS = ["market", "social", "fundamentals", "news"]  # 快速模式砍 3 个特化

# Rating priority for sorting the summary (Buy first). Unknown → last.
_RATING_RANK = {"Buy": 0, "Overweight": 1, "Hold": 2, "Underweight": 3, "Sell": 4}


@dataclass
class BatchResult:
    code: str
    signal: str = ""
    execution_advice: str = ""
    thesis: str = ""
    status: str = "ok"                      # ok | failed
    error: str = ""
    state: dict = field(default_factory=dict)


def estimate_calls(n: int, quick: bool, debate_rounds: int = 1, risk_rounds: int = 1) -> int:
    """Rough LLM-call count for a batch, printed before execution (cost gate)."""
    analysts = 4 if quick else 7
    per_stock = (
        analysts                    # 分析师
        + 2 * debate_rounds         # Bull/Bear 辩论
        + 1                         # Quality Gate
        + 1                         # Research Manager
        + 1                         # Trader
        + 3 * risk_rounds           # 三方风险辩论
        + 1                         # Portfolio Manager
        + 1                         # Execution Advisor (M1)
    )
    return n * per_stock


def _one_line(state: dict, max_len: int = 120) -> str:
    """First non-empty sentence of the PM decision, as a summary row."""
    decision = state.get("final_trade_decision", "") or ""
    for line in decision.splitlines():
        line = line.strip()
        if line.startswith("**Executive Summary**"):
            text = line.split("**", 2)[-1].strip()
            return text[:max_len] + ("…" if len(text) > max_len else "")
    return decision.strip()[:max_len]


def run_batch(
    pool: list[str],
    trade_date: str,
    config: Optional[dict] = None,
    quick: bool = False,
    limit: int = 5,
) -> list[BatchResult]:
    """Analyse up to ``limit`` tickers serially and return one result each.

    Every ticker goes through ``safe_ticker_component`` (the path-safety
    boundary); failures are captured per ticker, never raised.
    """
    results: list[BatchResult] = []
    analysts = QUICK_ANALYSTS if quick else ALL_ANALYSTS
    for raw in pool[:limit]:
        try:
            code = safe_ticker_component(str(raw))
        except Exception as exc:  # noqa: BLE001 — invalid input → isolated failure
            results.append(BatchResult(str(raw), status="failed", error=f"ticker 校验失败: {exc}"))
            continue
        try:
            graph = TradingAgentsGraph(selected_analysts=analysts, config=config)
            final_state, signal = graph.propagate(code, trade_date)
            results.append(
                BatchResult(
                    code,
                    signal=signal,
                    execution_advice=final_state.get("execution_advice", ""),
                    thesis=_one_line(final_state),
                    state=final_state,
                )
            )
        except Exception as exc:  # noqa: BLE001 — isolate, do not abort the batch
            results.append(BatchResult(code, status="failed", error=str(exc)))
    return results


def sort_results(results: list[BatchResult]) -> list[BatchResult]:
    """Buy-first order, failed items last."""
    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status != "ok"]
    ok.sort(key=lambda r: _RATING_RANK.get(r.signal, 9))
    return ok + bad
