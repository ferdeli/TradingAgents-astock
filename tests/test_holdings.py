"""Unit tests for M2 holding-management support (simplified structure).

The holding is now a single ``{"quantity", "cost_price"}`` dict for the
analysed ticker (no code/name, not an array); the legacy list shape is still
accepted via ``_normalize_holdings``. Covers: normalisation, the PM prompt
block (incl. PnL), the backward compatibility of the rendered decision, and
the initial-state injection.
"""

import pytest

from tradingagents.agents.managers.portfolio_manager import _holdings_block
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    PositionAction,
    render_pm_decision,
)
from tradingagents.graph.propagation import Propagator, _normalize_holdings


@pytest.mark.unit
class TestNormalizeHoldings:
    def test_new_simple_dict(self):
        assert _normalize_holdings({"quantity": 100, "cost_price": 1400.0}) == {
            "quantity": 100, "cost_price": 1400.0,
        }

    def test_legacy_list_first_entry_wins(self):
        out = _normalize_holdings([
            {"code": "600519", "name": "贵州茅台", "quantity": 100, "cost_price": 1400.0},
            {"code": "000001", "name": "平安银行", "quantity": 500, "cost_price": 10.0},
        ])
        assert out == {"quantity": 100, "cost_price": 1400.0}

    def test_empty_and_absent_are_none(self):
        assert _normalize_holdings(None) is None
        assert _normalize_holdings([]) is None
        assert _normalize_holdings({}) is None

    def test_code_and_name_are_ignored(self):
        # 新结构：code/name 冗余，不再参与匹配
        out = _normalize_holdings({"quantity": 200, "cost_price": 12.5})
        assert "code" not in out and "name" not in out


@pytest.mark.unit
class TestHoldingsBlock:
    def test_none_returns_empty(self):
        assert _holdings_block(None, None) == ""

    def test_with_cost_and_price_shows_pnl(self):
        block = _holdings_block(
            {"quantity": 100, "cost_price": 1400.0}, price=1540.0
        )
        assert "Cost price: 1400.0" in block
        assert "PnL +10.0%" in block
        assert "position_action" in block

    def test_cost_without_price_degrades(self):
        block = _holdings_block({"quantity": 100, "cost_price": 1400.0}, price=None)
        assert "Cost price: 1400.0" in block
        assert "PnL" not in block


@pytest.mark.unit
class TestRenderedDecisionBackwardCompat:
    def _decision(self, position_action=None, target=None):
        return PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="summary",
            investment_thesis="thesis",
            position_action=position_action,
            target_position_pct=target,
        )

    def test_no_holding_output_shape_unchanged(self):
        text = render_pm_decision(self._decision())
        assert "**Rating**: Buy" in text
        assert "**Position Action**" not in text
        assert "**Target Position**" not in text

    def test_with_position_action_renders(self):
        text = render_pm_decision(
            self._decision(position_action=PositionAction.ADD, target=15.0)
        )
        assert "**Position Action**: add" in text
        assert "**Target Position**: 15.0%" in text

    def test_rating_still_parses_with_position_lines(self):
        text = render_pm_decision(
            self._decision(position_action=PositionAction.HOLD, target=20.0)
        )
        from tradingagents.agents.utils.rating import parse_rating
        assert parse_rating(text) == "Buy"


@pytest.mark.unit
class TestInitialStateInjection:
    def test_simple_dict_injected(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"holdings": {"quantity": 100, "cost_price": 1400.0}},
        )
        state = Propagator().create_initial_state("600519", "2026-08-06")
        assert state["holdings"] == {"quantity": 100, "cost_price": 1400.0}

    def test_legacy_list_normalized(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"holdings": [{"code": "600519", "name": "贵州茅台",
                                   "quantity": 100, "cost_price": 1400.0}]},
        )
        state = Propagator().create_initial_state("600519", "2026-08-06")
        assert state["holdings"] == {"quantity": 100, "cost_price": 1400.0}

    def test_no_config_holdings_is_none(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config", lambda: {"holdings": None}
        )
        state = Propagator().create_initial_state("600519", "2026-08-06")
        assert state["holdings"] is None
