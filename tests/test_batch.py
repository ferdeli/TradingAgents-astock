"""Unit tests for the M4 batch runner and summary report (all offline).

``TradingAgentsGraph`` is swapped for a fake so no LLM/data calls happen;
the same fake exercises per-ticker isolation, the limit, quick mode, and
the summary rendering.
"""

import pytest

from tradingagents.batch import runner
from tradingagents.batch.report import render_summary


class FakeGraph:
    instances: list = []
    failing: set = set()

    def __init__(self, selected_analysts=None, config=None):
        self.selected_analysts = selected_analysts
        FakeGraph.instances.append(self)

    def propagate(self, code, trade_date):
        if code in FakeGraph.failing:
            raise RuntimeError(f"boom {code}")
        state = {
            "final_trade_decision": (
                "**Rating**: Buy\n\n**Executive Summary**: 基本面扎实，估值合理"
            ),
            "execution_advice": "**Position Size**: 15.0%\n**Entry Zone**: 9.8 - 10.2",
        }
        return (state, "Buy")


@pytest.fixture(autouse=True)
def _fake_graph(monkeypatch):
    FakeGraph.instances = []
    FakeGraph.failing = set()
    monkeypatch.setattr(runner, "TradingAgentsGraph", FakeGraph)
    yield


@pytest.mark.unit
class TestEstimateCalls:
    def test_full_mode_formula(self):
        # 7分析师 + 2辩论 + 1质量门 + 1RM + 1Trader + 3风险 + 1PM + 1执行建议
        assert runner.estimate_calls(5, quick=False) == 5 * 17

    def test_quick_mode_formula(self):
        assert runner.estimate_calls(5, quick=True) == 5 * 14    # 4分析师版本

    def test_zero_pool(self):
        assert runner.estimate_calls(0, quick=False) == 0


@pytest.mark.unit
class TestRunBatch:
    def test_all_succeed(self):
        results = runner.run_batch(["600519", "000001"], "2026-08-06", limit=5)
        assert [r.status for r in results] == ["ok", "ok"]
        assert [r.code for r in results] == ["600519", "000001"]
        assert results[0].signal == "Buy"

    def test_limit_caps_pool(self):
        results = runner.run_batch(["1", "2", "3"], "2026-08-06", limit=2)
        assert len(results) == 2
        assert len(FakeGraph.instances) == 2

    def test_failure_isolation(self):
        FakeGraph.failing = {"000001"}
        results = runner.run_batch(["600519", "000001", "300750"], "2026-08-06", limit=5)
        assert [r.status for r in results] == ["ok", "failed", "ok"]
        assert results[1].error == "boom 000001"

    def test_quick_mode_analysts(self):
        runner.run_batch(["600519"], "2026-08-06", quick=True, limit=5)
        assert FakeGraph.instances[0].selected_analysts == runner.QUICK_ANALYSTS

    def test_full_mode_analysts(self):
        runner.run_batch(["600519"], "2026-08-06", quick=False, limit=5)
        assert FakeGraph.instances[0].selected_analysts == runner.ALL_ANALYSTS


@pytest.mark.unit
class TestSummary:
    def test_sorted_and_failed_listed(self):
        results = [
            runner.BatchResult("300750", signal="Hold", thesis="t"),
            runner.BatchResult("600519", signal="Buy", thesis="t"),
            runner.BatchResult("000001", status="failed", error="boom"),
        ]
        md = render_summary(results)
        assert md.index("600519") < md.index("300750")           # Buy before Hold
        assert "失败清单" in md and "boom" in md
        assert "| 600519 | Buy |" in md

    def test_execution_advice_extracted(self):
        r = runner.BatchResult(
            "600519", signal="Buy",
            execution_advice="**Position Size**: 15.0%\n**Entry Zone**: 9.8 - 10.2",
        )
        md = render_summary([r])
        assert "| 600519 | Buy | 15.0% |" in md
