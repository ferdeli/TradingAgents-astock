"""Unit tests for history grouping / labels (calendar & task-list support)."""

import pytest

from web.history import count_tasks_on, group_history, history_label

ENTRIES = [
    {"ticker": "600519", "date": "2026-08-19", "path": "/a/600519", "batch_id": "b1",
     "title": "", "name": "贵州茅台"},
    {"ticker": "000001", "date": "2026-08-19", "path": "/b/000001", "batch_id": "b1",
     "title": "", "name": "平安银行"},
    {"ticker": "300750", "date": "2026-08-18", "path": "/c/300750", "batch_id": "",
     "title": "", "name": "宁德时代"},
]


@pytest.mark.unit
class TestGroupHistory:
    def test_batch_collapses_into_one_item(self):
        items = group_history(ENTRIES)
        batches = [i for i in items if i["kind"] == "batch"]
        singles = [i for i in items if i["kind"] == "single"]
        assert len(batches) == 1
        assert batches[0]["batch_id"] == "b1"
        assert len(batches[0]["entries"]) == 2
        assert len(singles) == 1 and singles[0]["ticker"] == "300750"

    def test_sorted_newest_first(self):
        items = group_history(ENTRIES)
        assert items[0]["date"] == "2026-08-19"


@pytest.mark.unit
class TestHistoryLabel:
    def test_single_unnamed_with_name(self):
        item = group_history([ENTRIES[2]])[0]
        assert history_label(item) == "宁德时代（300750）· 2026-08-18"

    def test_single_unnamed_without_name(self):
        item = group_history([{**ENTRIES[2], "name": ""}])[0]
        assert history_label(item) == "300750 · 2026-08-18"

    def test_single_with_custom_title(self):
        item = group_history([{**ENTRIES[2], "title": "复盘"}])[0]
        assert history_label(item) == "复盘 · 2026-08-18"

    def test_batch_unnamed_shows_count(self):
        items = group_history(ENTRIES)
        batch = next(i for i in items if i["kind"] == "batch")
        assert history_label(batch) == "批量分析 · 2 个标的 · 2026-08-19"

    def test_batch_with_title(self):
        items = group_history([{**e, "title": "大盘组合"} for e in ENTRIES])
        batch = next(i for i in items if i["kind"] == "batch")
        assert history_label(batch) == "大盘组合 · 2026-08-19"


@pytest.mark.unit
class TestCountTasksOn:
    def test_counts_ticker_per_day(self, monkeypatch):
        monkeypatch.setattr(
            "web.history.get_history", lambda: ENTRIES,
        )
        assert count_tasks_on("2026-08-19") == 2
        assert count_tasks_on("2026-08-18") == 1
        assert count_tasks_on("2026-08-01") == 0
