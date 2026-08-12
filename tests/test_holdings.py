"""Unit tests for M2 holding-management support.

Covers: holding matching, the PM prompt block (incl. PnL), the backward
compatibility of the rendered decision (no holding → byte-compatible), and
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
from tradingagents.graph.propagation import Propagator, _holding_matches


@pytest.mark.unit
class TestHoldingMatch:
    def test_match_by_code(self):
        assert _holding_matches({"code": "600519", "name": "贵州茅台"}, "600519")
        assert _holding_matches({"code": "600519"}, "sh600519")
        assert _holding_matches({"code": "600519.SH"}, "600519")

    def test_match_by_name(self):
        assert _holding_matches({"code": "600519", "name": "贵州茅台"}, "贵州茅台")

    def test_no_match(self):
        assert not _holding_matches({"code": "000001", "name": "平安银行"}, "600519")
        assert not _holding_matches({"code": "600519"}, "000001")

    def test_fuzzy_names_do_not_match(self):
        # Partial name matches must not leak into the prompt.
        assert not _holding_matches({"code": "600519", "name": "贵州茅台"}, "贵州")


@pytest.mark.unit
class TestHoldingsBlock:
    def test_empty_returns_empty(self):
        assert _holdings_block([], None) == ""

    def test_with_cost_and_price_shows_pnl(self):
        block = _holdings_block(
            [{"code": "600519", "name": "贵州茅台", "quantity": 100, "cost_price": 1400.0}],
            price=1540.0,
        )
        assert "Cost price: 1400.0" in block
        assert "PnL +10.0%" in block
        assert "position_action" in block

    def test_cost_without_price_degrades(self):
        block = _holdings_block(
            [{"code": "600519", "name": "贵州茅台", "quantity": 100, "cost_price": 1400.0}],
            price=None,
        )
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
    def test_matching_holding_injected(self, monkeypatch):
        holdings = [
            {"code": "600519", "name": "贵州茅台", "quantity": 100, "cost_price": 1400.0},
            {"code": "000001", "name": "平安银行", "quantity": 500, "cost_price": 10.0},
        ]
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"holdings": holdings},
        )
        state = Propagator().create_initial_state("600519", "2026-08-06")
        assert len(state["holdings"]) == 1
        assert state["holdings"][0]["code"] == "600519"

    def test_no_config_holdings_is_empty(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config", lambda: {"holdings": []}
        )
        state = Propagator().create_initial_state("600519", "2026-08-06")
        assert state["holdings"] == []
