"""Unit tests for offline OHLCV cache fallback.

When mootdx and sina both fail, get_stock_data / _load_ohlcv_astock must fall
back to the on-disk CSV cache ({code}-astock-daily.csv) regardless of mtime,
so K-lines still render offline. A missing cache keeps the failure message.
"""

import pytest

from tradingagents.dataflows import a_stock

CACHE_CSV = """Date,Open,High,Low,Close,Volume
2026-07-01,10.0,10.5,9.8,10.2,10000
2026-07-02,10.2,10.6,9.9,10.4,12000
2026-07-03,10.4,10.8,9.9,10.0,11000
"""


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.config.get_config",
        lambda: {"data_cache_dir": str(tmp_path)},
    )
    (tmp_path / "600519-astock-daily.csv").write_text(CACHE_CSV, encoding="utf-8")
    return tmp_path


@pytest.mark.unit
class TestGetStockDataOfflineFallback:
    def test_offline_cache_used_when_all_sources_fail(self, cache_dir, monkeypatch):
        def boom_mootdx(*a, **k):
            raise ConnectionError("mootdx unreachable")

        def boom_sina(*a, **k):
            raise ConnectionError("sina unreachable")

        monkeypatch.setattr(a_stock, "_mootdx_call", boom_mootdx)
        monkeypatch.setattr(a_stock, "_sina_kline_fallback", boom_sina)

        out = a_stock.get_stock_data("600519", "2026-07-01", "2026-07-03")
        assert "offline cache (fallback)" in out
        assert "2026-07-01" in out and "10.2" in out       # cached rows present
        assert "K线数据获取失败" not in out

    def test_no_cache_keeps_failure_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"data_cache_dir": str(tmp_path)},  # empty cache dir
        )
        monkeypatch.setattr(a_stock, "_mootdx_call",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        monkeypatch.setattr(a_stock, "_sina_kline_fallback",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))

        out = a_stock.get_stock_data("600519", "2026-07-01", "2026-07-03")
        assert out.startswith("K线数据获取失败")

    def test_date_range_applied_to_cache(self, cache_dir, monkeypatch):
        monkeypatch.setattr(a_stock, "_mootdx_call",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        monkeypatch.setattr(a_stock, "_sina_kline_fallback",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))

        out = a_stock.get_stock_data("600519", "2026-07-02", "2026-07-03")
        assert "2026-07-01" not in out                       # filtered out
        assert "2026-07-02" in out and "2026-07-03" in out


@pytest.mark.unit
class TestLoadOhlcvAstockOfflineFallback:
    def test_load_ohlcv_falls_back_to_stale_cache(self, cache_dir, monkeypatch):
        monkeypatch.setattr(a_stock, "_mootdx_call",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        monkeypatch.setattr(a_stock, "_sina_kline_fallback",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))

        df = a_stock._load_ohlcv_astock("600519", "2026-07-03")
        assert not df.empty
        assert "Close" in df.columns
        assert df["Close"].iloc[-1] == pytest.approx(10.0)

    def test_load_ohlcv_no_cache_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"data_cache_dir": str(tmp_path)},
        )
        monkeypatch.setattr(a_stock, "_mootdx_call",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        monkeypatch.setattr(a_stock, "_sina_kline_fallback",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))

        with pytest.raises(ValueError, match="No OHLCV data"):
            a_stock._load_ohlcv_astock("600519", "2026-07-03")
