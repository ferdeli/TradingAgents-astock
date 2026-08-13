"""Execution Advisor: translates the final decision into research-reference execution levels.

Runs *after* the Portfolio Manager. The PM's mandate deliberately excludes
executable levels (see ``schemas.TraderProposal`` / ``_NO_LEVELS_RULE``);
this node adds the counterpart: entry zone, stop-loss, target price, and a
position size computed deterministically from the stop distance.

Behaviour by rating:
- Buy / Overweight → ask the LLM for price levels, validate them against the
  market snapshot (stop < close < target), and derive the position size.
- Hold / Underweight / Sell → return a placeholder ("no buy levels"), so no
  LLM call is wasted and no levels are fabricated for a non-buy rating.

Everything produced here is explicitly research-reference material for study
and education, not an investment-advisory signal.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import pandas as pd

from tradingagents.agents.schemas import (
    ExecutionAdvice,
    render_execution_advice,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.position_sizing import validate_advice
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.interface import route_to_vendor

logger = logging.getLogger(__name__)

_BUY_RATINGS = {"Buy", "Overweight"}
_HOLD_RATINGS = {"Hold"}

_REFERENCE_DISCLAIMER = (
    "\nThese are research-reference levels for study and education only, NOT "
    "an investment-advisory signal. Keep the entry zone consistent with the "
    "latest close, stop-loss below it, and target price above it."
)

_LOOKBACK_DAYS = 60  # recent OHLCV window used for price/ATR/range snapshot


def _parse_ohlcv_csv(text: str) -> Optional[pd.DataFrame]:
    """Parse the CSV returned by the a_stock vendor (skips ``#`` header lines)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:  # header + at least one data row
        return None
    try:
        return pd.read_csv(pd.io.common.StringIO("\n".join(lines)))
    except Exception as exc:  # noqa: BLE001 — parsing failure -> no snapshot
        logger.warning("Execution Advisor: could not parse OHLCV CSV (%s)", exc)
        return None


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over the trailing ``period`` rows."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.tail(period).mean())


def _fetch_price_snapshot(ticker: str, trade_date: str) -> Optional[dict[str, float]]:
    """Return ``{price, atr, high20, low20}`` from recent OHLCV, or ``None``.

    Uses the same vendor path as the Market Analyst tool
    (``route_to_vendor("get_stock_data", ...)``) so the levels the advisor
    sees match what the analysts saw.
    """
    from datetime import datetime, timedelta

    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = (end_dt - timedelta(days=_LOOKBACK_DAYS * 2)).strftime("%Y-%m-%d")
    try:
        raw = route_to_vendor("get_stock_data", ticker, start_dt, trade_date)
    except Exception as exc:  # noqa: BLE001 — any vendor failure -> no snapshot
        logger.warning("Execution Advisor: price snapshot failed for %s (%s)", ticker, exc)
        return None
    if not isinstance(raw, str) or raw.startswith("K线数据获取失败"):
        return None

    df = _parse_ohlcv_csv(raw)
    if df is None or df.empty or "Close" not in df.columns:
        return None

    df = df.sort_values("Date").tail(_LOOKBACK_DAYS).reset_index(drop=True)
    price = float(df.iloc[-1]["Close"])
    high20 = float(df["High"].tail(20).max())
    low20 = float(df["Low"].tail(20).min())
    return {"price": price, "atr": _atr(df), "high20": high20, "low20": low20}


def _placeholder(rating: str) -> str:
    """Markdown placeholder for non-buy ratings: no actionable levels."""
    if rating not in _HOLD_RATINGS:
        msg = "当前评级为卖出/减持，建议离场或回避，不给出买入执行建议。"
    else:
        msg = "当前评级为持有/观望，维持现状，不给出买入执行建议。"
    return f"**Position Size**: 0%\n**Rationale**: {msg}"


def _data_unavailable(reason: str) -> str:
    """Placeholder for a Buy-level advice that cannot be produced safely.

    Distinct from :func:`_placeholder` (which speaks for non-buy ratings):
    this says "no data / could not validate", never a contradictory rating
    message.
    """
    return f"**Position Size**: 0%\n**Rationale**: {reason}"


# ---------------------------------------------------------------------------
# Holding advice (M2): a standalone "what to do with my position" block.
# ---------------------------------------------------------------------------

_ACTION_CN = {
    "hold": "持有",
    "add": "加仓",
    "reduce": "减仓",
    "exit": "清仓/离场",
}


def parse_position_advice(decision_md: str) -> tuple[Optional[str], Optional[float]]:
    """Extract (position_action, target_position_pct) from the PM decision markdown.

    Returns ``(None, None)`` when the PM produced no holding guidance.
    """
    action: Optional[str] = None
    target: Optional[float] = None
    for line in (decision_md or "").splitlines():
        if line.startswith("**Position Action**"):
            action = line.split("**", 2)[-1].strip().lstrip(":").strip().lower()
        elif line.startswith("**Target Position**"):
            m = re.search(r"(\d+\.?\d*)", line.split("**", 2)[-1])
            if m:
                target = float(m.group(1))
    return action, target


def build_holding_advice(
    holdings: list, snapshot: Optional[dict], decision_md: str
) -> str:
    """Render a standalone holding-action block for immediate display.

    Combines the user's position (cost/quantity/PnL) with the PM's
    position_action / target_position_pct. ``holdings`` is already filtered to
    the analysed ticker by ``Propagator.create_initial_state``.
    """
    if not holdings:
        return ""
    h = holdings[0]
    quantity = h.get("quantity", 0)
    cost = h.get("cost_price")

    lines = ["**当前持仓（研究参考）**", ""]
    if cost:
        lines.append(f"- 持仓成本：{cost}，数量：{quantity}")
        price = snapshot["price"] if snapshot else None
        if price:
            pnl = (price - float(cost)) / float(cost) * 100
            lines.append(f"- 现价：{price:.2f}，浮动盈亏：{pnl:+.1f}%")
    else:
        lines.append(f"- 数量：{quantity}（成本未知）")

    action, target = parse_position_advice(decision_md)
    if action:
        cn = _ACTION_CN.get(action, action)
        target_txt = f"，目标仓位 {target}%" if target is not None else ""
        lines.append(f"- **建议操作：{cn}**{target_txt}")
    else:
        lines.append("- 建议操作：维持现状（本次决策未给出明确加减仓信号）")
    lines.append("- ⚠️ 研究参考，非自动调仓指令")
    return "\n".join(lines)


def create_execution_advisor(llm):
    """Create the Execution Advisor graph node.

    ``llm`` is the same quick-thinking model used by the Trader; structured
    output is bound when the provider supports it, with free-text fallback
    otherwise (identical to the Portfolio Manager pattern).

    Besides ``execution_advice``, the node also emits ``holding_advice`` — a
    standalone position-action block (M2) built from the user's holding and
    the PM's decision, so the CLI/Web can surface it immediately after the
    analysis finishes.
    """
    structured_llm = bind_structured(llm, ExecutionAdvice, "Execution Advisor")

    def execution_advisor_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        rating = parse_rating(state.get("final_trade_decision", ""))
        instrument_context = build_instrument_context(ticker)

        holdings = state.get("holdings", []) or []
        # Snapshot is needed for holding PnL (holdings present) and for the
        # buy path's price levels; fetch once, reuse both.
        need_snapshot = bool(holdings) or rating in _BUY_RATINGS
        snapshot = _fetch_price_snapshot(ticker, trade_date) if need_snapshot else None

        # Non-buy ratings: no LLM call, deterministic placeholder.
        if rating not in _BUY_RATINGS:
            advice = _placeholder(rating)
        else:
            price = snapshot["price"] if snapshot else None
            if price is None:
                advice = _data_unavailable(
                    "无法获取有效行情快照（mootdx/新浪均不可用），暂不给出执行建议。"
                )
            else:
                if snapshot["atr"] <= 0:
                    atr_line = "n/a"
                else:
                    atr_line = f"{snapshot['atr']:.2f}"

                prompt = f"""As the Execution Advisor, translate the final decision into concrete research-reference execution levels.

{instrument_context}

---
**Market Snapshot (reference):**
- Latest close: {price:.2f}
- 14-day ATR: {atr_line}
- 20-day high / low: {snapshot['high20']:.2f} / {snapshot['low20']:.2f}
- Final rating: {rating}

**Context:**
- Portfolio Manager final decision:
{state.get('final_trade_decision', '')}

- Trader's transaction proposal:
{state.get('trader_investment_plan', '')}

---
Guidance:
- entry_zone: a narrow band around the latest close (e.g. "{price * 0.98:.2f} - {price * 1.02:.2f}"), aligned with the 20-day low / ATR.
- stop_loss: strictly below the latest close, near the 20-day low or close minus ~1.5x ATR.
- target_price: strictly above the latest close, near the 20-day high or close plus ~2x ATR.
- rationale: one short sentence (<= 60 chars) tying the levels to the snapshot.
- Do NOT propose position_size_pct; it is computed by a deterministic rule.
{_REFERENCE_DISCLAIMER}{get_language_instruction()}"""

                # Validate + derive position size BEFORE rendering; free-text
                # fallback (no structured output) passes the raw model text
                # through unchanged, mirroring the PM's degradation behaviour.
                def render_validated(advice: ExecutionAdvice) -> str:
                    return render_execution_advice(validate_advice(advice, price))

                rendered = invoke_structured_or_freetext(
                    structured_llm,
                    llm,
                    prompt,
                    render_validated,
                    "Execution Advisor",
                )
                # The free-text fallback path (provider without structured
                # output, or a failed structured call) returns raw
                # response.content that bypassed validate_advice and the
                # deterministic position-size overwrite. Never publish
                # unvalidated levels: detect the structured-render header and
                # replace anything else with a data-unavailable note.
                if not rendered.strip().startswith("**Entry Zone**"):
                    rendered = _data_unavailable(
                        "执行建议生成已降级为自由文本，未经价位校验，不发布具体价位/仓位。"
                    )
                advice = rendered

        result = {"execution_advice": advice}
        if holdings:
            result["holding_advice"] = build_holding_advice(
                holdings,
                snapshot,
                state.get("final_trade_decision", ""),
            )
        return result

    return execution_advisor_node
