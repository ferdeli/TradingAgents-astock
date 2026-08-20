"""Unit tests for the M5 K-line charting (fully offline).

OHLCV is injected as CSV text; plotly figures are constructed but never
rendered, so no browser/DOM is involved.
"""

import pytest

from tradingagents.charting import kline

CSV = """# Stock data for 600519 (A-stock)
Date,Open,High,Low,Close,Volume
2026-07-01,10.0,10.5,9.8,10.2,10000
2026-07-02,10.2,10.6,9.9,10.4,12000
2026-07-03,10.4,10.8,9.9,10.0,11000
2026-07-04,10.0,10.3,9.7,10.1,13000
2026-07-05,10.1,10.7,10.0,10.5,14000
"""

ADVICE_MD = (
    "**Entry Zone**: 9.8 - 10.2\n"
    "**Stop Loss**: 9.0\n"
    "**Target Price**: 11.5\n"
    "**Position Size**: 15.0%"
)
PLACEHOLDER_MD = "**Position Size**: 0%\n**Rationale**: 当前评级为持有/观望"


@pytest.mark.unit
class TestAdviceParsing:
    def test_full_advice(self):
        a = kline.advice_from_markdown(ADVICE_MD)
        assert a == {"stop_loss": 9.0, "target_price": 11.5}

    def test_placeholder_returns_none(self):
        assert kline.advice_from_markdown(PLACEHOLDER_MD) is None

    def test_empty_returns_none(self):
        assert kline.advice_from_markdown("") is None


@pytest.mark.unit
class TestSynthesizeForecast:
    def _hist(self):
        return kline._parse_ohlcv_csv(CSV)

    def test_length_and_forecast_flag(self):
        fut = kline.synthesize_forecast(self._hist(), "Buy", days=10)
        assert len(fut) == 10
        assert all(c["forecast"] for c in fut)

    def test_endpoint_from_advice_target(self):
        fut = kline.synthesize_forecast(self._hist(), "Buy", {"stop_loss": 9.0, "target_price": 11.5}, days=5)
        assert fut[-1]["close"] == pytest.approx(11.5, abs=0.5)   # 终点锚定 target（±噪声）

    def test_endpoint_from_advice_stop_for_sell(self):
        fut = kline.synthesize_forecast(self._hist(), "Sell", {"stop_loss": 8.5, "target_price": 12.0}, days=5)
        assert fut[-1]["close"] == pytest.approx(8.5, abs=0.5)

    def test_fallback_coefficient_without_advice(self):
        fut = kline.synthesize_forecast(self._hist(), "Buy", None, days=5)
        last = self._hist().iloc[-1]["Close"]
        assert fut[-1]["close"] == pytest.approx(last * 1.08, abs=0.5)

    def test_ohlc_validity(self):
        for c in kline.synthesize_forecast(self._hist(), "Hold", days=10):
            assert c["high"] >= max(c["open"], c["close"])
            assert c["low"] <= min(c["open"], c["close"])

    def test_reproducible_seed(self):
        a = kline.synthesize_forecast(self._hist(), "Buy", days=10)
        b = kline.synthesize_forecast(self._hist(), "Buy", days=10)
        assert a == b


@pytest.mark.unit
class TestBuildChartData:
    def test_structure(self):
        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", ohlcv_text=CSV
        )
        assert chart is not None
        assert chart["ticker"] == "600519"
        assert chart["rating"] == "Buy"
        assert len(chart["history"]) == 5
        assert len(chart["forecast"]) == 10
        assert "forecast" not in chart["history"][0]

    def test_no_data_returns_none(self):
        assert kline.build_chart_data(
            "600519", "2026-07-05", ohlcv_text="K线数据获取失败：网络不可用"
        ) is None

    def test_rating_fallback_parses(self):
        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD,
            rating="**Rating**: Sell\n\n**Executive Summary**: x", ohlcv_text=CSV,
        )
        assert chart["rating"] == "Sell"


@pytest.mark.unit
class TestDiskCache:
    def _chart(self, monkeypatch, tmp_path, calls, advice_md=ADVICE_MD):
        """First build fetches (route_to_vendor), later calls hit the disk cache."""
        cache_dir = tmp_path / "kline"
        monkeypatch.setattr(kline, "_kline_cache_dir", lambda: str(cache_dir))

        def fake_vendor(*a, **k):
            calls.append(1)
            return CSV

        monkeypatch.setattr(kline, "route_to_vendor", fake_vendor)
        return kline.build_chart_data(
            "600519", "2026-07-05", advice_md=advice_md, rating="Buy"
        )

    def test_second_call_hits_cache_no_network(self, monkeypatch, tmp_path):
        calls: list = []
        first = self._chart(monkeypatch, tmp_path, calls)
        second = self._chart(monkeypatch, tmp_path, calls)
        assert calls == [1]                      # only the first call fetched
        assert first == second                   # byte-identical payload
        assert len(second["history"]) == 5

    def test_different_advice_gets_different_cache(self, monkeypatch, tmp_path):
        calls: list = []
        self._chart(monkeypatch, tmp_path, calls, advice_md=ADVICE_MD)
        self._chart(monkeypatch, tmp_path, calls, advice_md=PLACEHOLDER_MD)
        assert calls == [1, 1]                   # different advice → re-fetch

    def test_ohlcv_text_injection_bypasses_cache(self, monkeypatch, tmp_path):
        # Injected text never calls the vendor and never writes a cache file.
        monkeypatch.setattr(kline, "_kline_cache_dir", lambda: str(tmp_path / "kline"))
        monkeypatch.setattr(kline, "route_to_vendor", lambda *a, **k: pytest.fail("vendor called"))
        kline.build_chart_data("600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy",
                               ohlcv_text=CSV)
        assert not (tmp_path / "kline").exists() or not list((tmp_path / "kline").iterdir())


@pytest.mark.unit
class TestFigure:
    def test_two_traces_with_forecast(self):
        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", ohlcv_text=CSV
        )
        import plotly.graph_objects as go
        from web.components.kline_viewer import _build_figure

        fig = _build_figure(chart)
        assert len(fig.data) == 2
        assert fig.data[1].name == "预测（示意）"

    def test_single_trace_without_forecast(self):
        import plotly.graph_objects as go
        from web.components.kline_viewer import _build_figure

        fig = _build_figure({"history": [{"index": 0, "open": 1, "high": 2, "low": 0.5, "close": 1.5}], "forecast": [], "rating": "Hold"})
        assert len(fig.data) == 1


@pytest.mark.unit
class TestHistoryDateAndOffline:
    """fix: K-line hover date + history browsing must never fetch the network."""

    def test_history_records_carry_date(self):
        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", ohlcv_text=CSV
        )
        assert chart["history"][0]["date"] == "2026-07-01"
        assert chart["history"][-1]["date"] == "2026-07-05"

    def test_offline_cache_miss_returns_none_without_network(self, monkeypatch):
        monkeypatch.setattr(kline, "_kline_cache_dir", lambda: "/nonexistent/kline-cache")
        # route_to_vendor must NOT be called in offline mode
        monkeypatch.setattr(
            kline, "route_to_vendor", lambda *a, **k: pytest.fail("offline mode must not fetch")
        )
        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", offline=True
        )
        assert chart is None

    def test_offline_cache_hit_returns_cached(self, monkeypatch, tmp_path):
        cache_dir = tmp_path / "kline"
        monkeypatch.setattr(kline, "_kline_cache_dir", lambda: str(cache_dir))
        # First build online (populates cache), then offline hit.
        monkeypatch.setattr(kline, "route_to_vendor", lambda *a, **k: CSV)
        first = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy"
        )
        assert first is not None
        # Offline: same cache dir, route_to_vendor must not be called.
        monkeypatch.setattr(
            kline, "route_to_vendor", lambda *a, **k: pytest.fail("offline must not fetch")
        )
        second = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", offline=True
        )
        assert second == first
        assert second["history"][0]["date"] == "2026-07-01"

    def test_figure_hover_date_customdata(self):
        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", ohlcv_text=CSV
        )
        from web.components.kline_viewer import _build_figure

        fig = _build_figure(chart)
        hist_trace = fig.data[0]
        assert [list(c) for c in hist_trace.customdata] == [[d] for d in ("2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05")]
        assert "%{customdata[0]}" in hist_trace.hovertemplate
        # Legacy cache without date → falls back to index label.
        legacy = {"history": [{"index": 0, "open": 1, "high": 2, "low": 0.5, "close": 1.5}],
                  "forecast": [], "rating": "Hold"}
        fig2 = _build_figure(legacy)
        assert [list(c) for c in fig2.data[0].customdata] == [["#0"]]


@pytest.mark.unit
class TestLegacyCacheUpgrade:
    """Legacy kline caches (no per-candle date) must be regenerated with dates."""

    LEGACY = {
        "ticker": "600519",
        "history": [{"index": 0, "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2}],
        "forecast": [],
        "rating": "Buy",
    }

    def test_legacy_cache_miss_and_rebuild_from_csv(self, monkeypatch, tmp_path):
        cache_dir = tmp_path / "kline"
        csv_dir = tmp_path / "csv"
        monkeypatch.setattr(kline, "_kline_cache_dir", lambda: str(cache_dir))
        csv_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"data_cache_dir": str(csv_dir)},
        )
        # Disk cache files are plain CSV (no '#' header lines — that shape is
        # only for the get_stock_data *text* payload).
        plain_csv = "\n".join(
            ln for ln in CSV.splitlines() if not ln.lstrip().startswith("#")
        )
        (csv_dir / "600519-astock-daily.csv").write_text(plain_csv, encoding="utf-8")

        # Seed a legacy (date-less) kline cache so the key already exists.
        kline._save_disk_cache("600519", "2026-07-05", ADVICE_MD, self.LEGACY)

        chart = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy",
            offline=True,  # history browsing: no network
        )
        assert chart is not None
        assert chart["history"][0]["date"] == "2026-07-01"   # rebuilt WITH dates
        assert chart["history"][-1]["date"] == "2026-07-05"

        # The regenerated payload is now cached (with dates) → next hit.
        chart2 = kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", offline=True
        )
        assert chart2 == chart

    def test_offline_without_any_local_data_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(kline, "_kline_cache_dir", lambda: str(tmp_path / "kline"))
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"data_cache_dir": str(tmp_path / "csv")},
        )
        assert kline.build_chart_data(
            "600519", "2026-07-05", advice_md=ADVICE_MD, rating="Buy", offline=True
        ) is None
