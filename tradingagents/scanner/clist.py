"""East Money clist wrapper: market-wide snapshots (gainers, mktcap, ...).

The ``clist/get`` endpoint is a generic market list API (the same one the
industry comparison uses). We parameterise ``fs`` / ``sort`` / ``fields`` to
build rankings the scanner filters on. ALL requests go through
``a_stock._em_get`` — the throttled entry point — never a raw ``requests``
call (CLAUDE.md rule).
"""

from __future__ import annotations

from typing import Any, Optional

from tradingagents.dataflows.a_stock import _em_get

CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"

# 全 A：深主板 + 创业板 + 沪主板 + 科创板
ALL_A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"

# clist 字段: f12=code f14=name f2=现价 f3=涨跌幅 f20=总市值(元) f9=PE(动,亏损为'-') f100=行业
DEFAULT_FIELDS = "f12,f14,f2,f3,f20,f9,f100"


def get_market_snapshot(
    fs: str = ALL_A_FS,
    sort: str = "3:desc",          # 3=涨跌幅, 20=总市值
    fields: str = DEFAULT_FIELDS,
    pz: int = 100,
) -> list[dict[str, Any]]:
    """Fetch up to ``pz`` market rows sorted by ``sort``.

    Returns the raw ``data.diff`` list; values are kept as returned by the
    API (numbers or '-' for PE of loss-makers). Callers parse with
    :func:`tradingagents.scanner.scanner.parse_row`.
    """
    params = {
        "pn": "1",
        "pz": str(pz),
        "po": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "fid": sort.split(":")[0],
        "fs": fs,
        "fields": fields,
    }
    resp = _em_get(CLIST_URL, params=params, timeout=15)
    data = resp.json().get("data") or {}
    return data.get("diff", []) or []


def top_gainers(limit: int = 100) -> list[dict[str, Any]]:
    """Top gainers across all A-shares (sorted by change % desc)."""
    return get_market_snapshot(sort="3:desc", pz=limit)


def top_mktcap(limit: int = 100) -> list[dict[str, Any]]:
    """Largest A-shares by total market cap."""
    return get_market_snapshot(sort="20:desc", pz=limit)
