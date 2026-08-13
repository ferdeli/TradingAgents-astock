"""Unit tests for the Execution Advisor node.

Covers the price-snapshot derivation (mock vendor CSV), the non-buy
placeholder path (no LLM call), the buy path where the LLM proposes
price levels that are validated and sized deterministically, the P1
free-text-fallback safety gate, and the P2b state-log serialization.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers import execution_advisor as ea
from tradingagents.agents.schemas import ExecutionAdvice

# Same shape the a_stock vendor returns: # comment headers + CSV body.
OHLCV_CSV = """# Stock data for 600519 (A-stock)
# Total records: 3
Date,Open,High,Low,Close,Volume
2026-08-01,10.0,10.5,9.8,10.2,10000
2026-08-02,10.2,10.6,9.9,10.4,12000
2026-08-03,10.4,10.8,9.9,10.0,11000
"""


@pytest.mark.unit
class TestPriceSnapshot:
    def test_parse_and_derive(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        snap = ea._fetch_price_snapshot("600519", "2026-08-03")
        assert snap is not None
        assert snap["price"] == 10.0          # last close
        assert snap["high20"] == 10.8
        assert snap["low20"] == 9.8
        # TR: row1 (no prev close) = high-low = 0.7; row2 = 0.7; row3 = 0.9
        # ATR(14) over 3 rows = mean(0.7, 0.7, 0.9) = 2.3/3
        assert snap["atr"] == pytest.approx(2.3 / 3)

    def test_vendor_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            ea, "route_to_vendor",
            lambda *a, **k: "K线数据获取失败：mootdx和新浪备用源均不可用",
        )
        assert ea._fetch_price_snapshot("600519", "2026-08-03") is None

    def test_exception_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(ea, "route_to_vendor", boom)
        assert ea._fetch_price_snapshot("600519", "2026-08-03") is None


def _state(rating_text="**Rating**: Buy\n\n**Executive Summary**: s"):
    return {
        "company_of_interest": "600519",
        "trade_date": "2026-08-03",
        "final_trade_decision": rating_text,
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
    }


@pytest.mark.unit
class TestExecutionAdvisorNode:
    def test_sell_rating_placeholder_and_no_llm(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        llm = MagicMock()
        node = ea.create_execution_advisor(llm)
        out = node(_state("**Rating**: Sell"))
        assert "Position Size**: 0%" in out["execution_advice"]
        assert "离场" in out["execution_advice"]
        llm.invoke.assert_not_called()

    def test_hold_rating_placeholder(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        node = ea.create_execution_advisor(MagicMock())
        out = node(_state("**Rating**: Hold"))
        assert "Position Size**: 0%" in out["execution_advice"]

    def test_buy_rating_structured_levels_and_size(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = ExecutionAdvice(
            entry_zone="9.8 - 10.2", stop_loss=9.0, target_price=11.5,
            rationale="near support",
        )
        llm.with_structured_output.return_value = structured
        node = ea.create_execution_advisor(llm)

        out = node(_state())
        text = out["execution_advice"]
        assert "**Entry Zone**: 9.8 - 10.2" in text
        assert "**Stop Loss**: 9.0" in text
        assert "**Target Price**: 11.5" in text
        # 1.5% budget / 10% stop distance = 15% — deterministic, not from LLM
        assert "**Position Size**: 15.0%" in text

    def test_buy_rating_invalid_levels_rejected(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = ExecutionAdvice(
            entry_zone="9.8 - 10.2", stop_loss=10.5, target_price=9.0,
            rationale="bad levels",
        )
        llm.with_structured_output.return_value = structured
        node = ea.create_execution_advisor(llm)

        out = node(_state())
        assert "N/A" in out["execution_advice"]
        assert "Position Size**: 0.0%" in out["execution_advice"]

    def test_buy_rating_without_price_snapshot_placeholder(self, monkeypatch):
        monkeypatch.setattr(
            ea, "route_to_vendor",
            lambda *a, **k: "K线数据获取失败：mootdx和新浪备用源均不可用",
        )
        llm = MagicMock()
        node = ea.create_execution_advisor(llm)
        out = node(_state())
        assert "Position Size**: 0%" in out["execution_advice"]
        assert "离场" not in out["execution_advice"]          # 不与 Buy 评级矛盾
        assert "行情快照" in out["execution_advice"]           # 数据不可用语义
        llm.invoke.assert_not_called()

    def test_buy_freetext_fallback_not_published(self, monkeypatch):
        # P1: provider without structured output → plain_llm.invoke returns raw
        # text that bypassed validate_advice; must NOT be published as advice.
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        llm = MagicMock()
        # with_structured_output unsupported → bind_structured returns None
        llm.with_structured_output.side_effect = NotImplementedError
        llm.invoke.return_value = MagicMock(
            content="entry 9.5, stop 12.0, target 8.0, position 88%"  # unvalidated junk
        )
        node = ea.create_execution_advisor(llm)
        out = node(_state())
        text = out["execution_advice"]
        assert "Position Size**: 0%" in text
        assert "降级" in text                                    # explicit degradation note
        assert "9.5" not in text and "88%" not in text           # no unvalidated levels

    def test_buy_freetext_fallback_after_structured_failure(self, monkeypatch):
        # P1: structured call raises → invoke_structured_or_freetext falls back
        # to plain text; the unvalidated text must be replaced too.
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.side_effect = RuntimeError("malformed json")
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content="随便写的价位 100 元")
        node = ea.create_execution_advisor(llm)
        out = node(_state())
        assert "降级" in out["execution_advice"]
        assert "100" not in out["execution_advice"]


@pytest.mark.unit
class TestStateLogSerialization:
    """P2b: execution_advice must survive into the saved history JSON."""

    def _minimal_state(self):
        return {
            "company_of_interest": "600519",
            "trade_date": "2026-08-06",
            "market_report": "m", "sentiment_report": "s", "news_report": "n",
            "fundamentals_report": "f", "policy_report": "p",
            "hot_money_report": "h", "lockup_report": "l",
            "investment_debate_state": {
                "bull_history": "b", "bear_history": "r", "history": "h",
                "current_response": "c", "judge_decision": "j",
            },
            "trader_investment_plan": "trader",
            "risk_debate_state": {
                "aggressive_history": "a", "conservative_history": "c",
                "neutral_history": "n", "history": "h", "judge_decision": "j",
            },
            "investment_plan": "plan",
            "final_trade_decision": "**Rating**: Buy\n\n**Executive Summary**: ok",
            "execution_advice": "**Position Size**: 15.0%\n**Rationale**: test",
        }

    def test_log_state_contains_execution_advice(self, tmp_path):
        import json

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = object.__new__(TradingAgentsGraph)
        graph.ticker = "600519"
        graph.config = {"results_dir": str(tmp_path)}
        graph.log_states_dict = {}

        graph._log_state("2026-08-06", self._minimal_state())

        log_file = next((tmp_path / "600519" / "TradingAgentsStrategy_logs").glob("full_states_log_*.json"))
        payload = json.loads(log_file.read_text(encoding="utf-8"))
        assert payload["execution_advice"] == (
            "**Position Size**: 15.0%\n**Rationale**: test"
        )
        assert payload["final_trade_decision"].startswith("**Rating**: Buy")

    def test_missing_execution_advice_degrades_to_empty(self, tmp_path):
        import json

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        state = self._minimal_state()
        state.pop("execution_advice")

        graph = object.__new__(TradingAgentsGraph)
        graph.ticker = "600519"
        graph.config = {"results_dir": str(tmp_path)}
        graph.log_states_dict = {}

        graph._log_state("2026-08-06", state)
        log_file = next((tmp_path / "600519" / "TradingAgentsStrategy_logs").glob("full_states_log_*.json"))
        payload = json.loads(log_file.read_text(encoding="utf-8"))
        assert payload["execution_advice"] == ""


@pytest.mark.unit
class TestHoldingAdvice:
    """M2 enhancement: standalone position-action block."""

    DECISION_WITH_ACTION = (
        "**Rating**: Buy\n\n**Executive Summary**: ok\n"
        "**Position Action**: add\n**Target Position**: 15.0%"
    )

    def test_parse_position_advice(self):
        action, target = ea.parse_position_advice(self.DECISION_WITH_ACTION)
        assert action == "add" and target == 15.0

    def test_parse_no_action(self):
        assert ea.parse_position_advice("**Rating**: Hold") == (None, None)

    def test_build_holding_advice(self):
        md = ea.build_holding_advice(
            [{"code": "600519", "quantity": 100, "cost_price": 1400.0}],
            {"price": 1540.0},
            self.DECISION_WITH_ACTION,
        )
        assert "持仓成本：1400.0" in md
        assert "浮动盈亏：+10.0%" in md
        assert "**建议操作：加仓**" in md
        assert "目标仓位 15.0%" in md

    def test_build_holding_advice_no_action(self):
        md = ea.build_holding_advice(
            [{"code": "600519", "quantity": 100, "cost_price": 1400.0}],
            None,
            "**Rating**: Hold",
        )
        assert "建议操作：维持现状" in md
        assert "盈亏" not in md                      # no snapshot → no PnL

    def test_build_holding_advice_empty_holdings(self):
        assert ea.build_holding_advice([], None, "") == ""


@pytest.mark.unit
class TestNodeHoldingAdviceEmission:
    def _state_with_holding(self, rating_text="**Rating**: Buy"):
        s = _state(rating_text)
        s["holdings"] = [{"code": "600519", "quantity": 100, "cost_price": 1400.0}]
        return s

    def test_holding_advice_emitted_for_buy(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = ExecutionAdvice(
            entry_zone="9.8 - 10.2", stop_loss=9.0, target_price=11.5, rationale="ok"
        )
        llm.with_structured_output.return_value = structured
        node = ea.create_execution_advisor(llm)
        out = node(self._state_with_holding())
        assert "holding_advice" in out
        assert "持仓成本：1400.0" in out["holding_advice"]

    def test_holding_advice_emitted_for_hold(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        node = ea.create_execution_advisor(MagicMock())
        out = node(self._state_with_holding("**Rating**: Hold"))
        assert "holding_advice" in out
        assert "维持现状" in out["holding_advice"] or "建议操作" in out["holding_advice"]

    def test_no_holdings_no_holding_advice_key(self, monkeypatch):
        monkeypatch.setattr(ea, "route_to_vendor", lambda *a, **k: OHLCV_CSV)
        node = ea.create_execution_advisor(MagicMock())
        out = node(_state("**Rating**: Sell"))
        assert "holding_advice" not in out
