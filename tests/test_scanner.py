"""Unit tests for the M3 screener (mock the East Money clist endpoint).

The snapshot is injected so tests stay fully offline; the same fixture
exercises parsing, AND-condition filtering, ST exclusion and the limit.
"""

import pytest

from tradingagents.scanner.scanner import (
    ScreenCriteria,
    _is_st,
    parse_row,
    run_screen,
)

# Raw clist rows in the shape the API returns (fltt=2 numbers, '-' PE for loss-makers).
ROWS = [
    {"f12": "600519", "f14": "贵州茅台", "f2": 1500.0, "f3": 1.2, "f20": 1.9e12, "f9": 28.0, "f100": "酿酒行业"},
    {"f12": "000001", "f14": "平安银行", "f2": 11.0, "f3": -0.5, "f20": 2.1e11, "f9": 5.0, "f100": "银行"},
    {"f12": "300750", "f14": "宁德时代", "f2": 190.0, "f3": 3.5, "f20": 8.4e11, "f9": 22.0, "f100": "电池"},
    {"f12": "600123", "f14": "*ST泛海", "f2": 0.8, "f3": 5.0, "f20": 4.0e9, "f9": "-", "f100": "房地产"},
    {"f12": "688981", "f14": "中芯国际", "f2": 55.0, "f3": -2.0, "f20": 4.4e11, "f9": "-", "f100": "半导体"},
]


@pytest.mark.unit
class TestParseRow:
    def test_normal_row(self):
        p = parse_row(ROWS[0])
        assert p["code"] == "600519" and p["name"] == "贵州茅台"
        assert p["price"] == 1500.0 and p["chg_pct"] == 1.2
        assert p["mktcap"] == 1.9e12 and p["pe"] == 28.0

    def test_loss_maker_pe_is_none(self):
        p = parse_row(ROWS[3])
        assert p["pe"] is None

    def test_st_detection(self):
        assert _is_st("*ST泛海") and _is_st("ST泛海") and _is_st("退市股")
        assert not _is_st("贵州茅台")


@pytest.mark.unit
class TestRunScreen:
    def test_no_criteria_returns_snapshot_limited(self):
        pool = run_screen(ScreenCriteria(limit=2), snapshot=lambda limit: ROWS)
        assert len(pool) == 2

    def test_pe_range_and(self):
        pool = run_screen(
            ScreenCriteria(pe_min=20, pe_max=30, limit=10), snapshot=lambda limit: ROWS
        )
        assert [p["code"] for p in pool] == ["600519", "300750"]

    def test_mktcap_min(self):
        pool = run_screen(
            ScreenCriteria(mktcap_min=1000, limit=10), snapshot=lambda limit: ROWS
        )  # 1000 亿元 = 1e11 元
        # 600519(19000亿) 000001(2100亿) 300750(8400亿) 688981(4400亿) 均超线
        assert [p["code"] for p in pool] == ["600519", "000001", "300750", "688981"]

    def test_industry_filter(self):
        pool = run_screen(
            ScreenCriteria(industries=["半导体"], limit=10),
            snapshot=lambda limit: ROWS,
        )
        assert [p["code"] for p in pool] == ["688981"]

    def test_st_excluded_by_default(self):
        pool = run_screen(ScreenCriteria(limit=10), snapshot=lambda limit: ROWS)
        assert "*ST泛海" not in [p["name"] for p in pool]

    def test_include_st_keeps_them(self):
        pool = run_screen(
            ScreenCriteria(exclude_st=False, limit=10), snapshot=lambda limit: ROWS
        )
        assert "*ST泛海" in [p["name"] for p in pool]

    def test_chg_min(self):
        pool = run_screen(
            ScreenCriteria(chg_min=3.0, limit=10), snapshot=lambda limit: ROWS
        )
        # 300750(+3.5) 达标；600123(+5.0) 是 *ST，默认被排除
        assert [p["code"] for p in pool] == ["300750"]
