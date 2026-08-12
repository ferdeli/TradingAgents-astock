"""Screener: filter criteria → candidate pool of A-share tickers.

Pure data layer — zero LLM calls. Consumes clist snapshots and applies AND
conditions. Output rows are consumed by the batch analyser (M4) or exported
to CSV via ``tradingagents scan``.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from tradingagents.scanner.clist import top_gainers


class ScreenCriteria(BaseModel):
    """Filter conditions; all specified conditions must hold (AND)."""

    industries: list[str] = Field(default_factory=list, description="所属行业（clist f100），精确匹配")
    pe_min: Optional[float] = None
    pe_max: Optional[float] = None
    mktcap_min: Optional[float] = None  # 总市值下限，亿元
    chg_min: Optional[float] = None     # 涨跌幅下限，%
    exclude_st: bool = True
    limit: int = Field(default=20, ge=1, le=200)


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw clist row into typed values; '-' (loss-makers) -> None."""
    def num(key: str) -> Optional[float]:
        v = row.get(key)
        if v is None or v == "-" or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    raw_ind = row.get("f100")
    return {
        "code": str(row.get("f12", "")),
        "name": str(row.get("f14", "")),
        "price": num("f2"),
        "chg_pct": num("f3"),
        "mktcap": num("f20"),          # 元
        "pe": num("f9"),
        "industry": "" if raw_ind in (None, "-", "") else str(raw_ind),
    }


def _is_st(name: str) -> bool:
    return "ST" in name.upper() or "退" in name


def run_screen(criteria: ScreenCriteria, snapshot=None) -> list[dict[str, Any]]:
    """Apply the criteria to a market snapshot and return the matched pool.

    ``snapshot`` is injectable for tests (defaults to ``top_gainers``); it
    must return a list of raw clist rows.
    """
    rows = (snapshot or top_gainers)(limit=max(500, criteria.limit))
    matched: list[dict[str, Any]] = []

    industries = {s.strip() for s in criteria.industries if s.strip()}
    for raw in rows:
        p = parse_row(raw)
        if criteria.exclude_st and _is_st(p["name"]):
            continue
        if industries and p["industry"] not in industries:
            continue
        if criteria.pe_min is not None and (p["pe"] is None or p["pe"] < criteria.pe_min):
            continue
        if criteria.pe_max is not None and (p["pe"] is None or p["pe"] > criteria.pe_max):
            continue
        if criteria.mktcap_min is not None:
            if p["mktcap"] is None or p["mktcap"] / 1e8 < criteria.mktcap_min:
                continue
        if criteria.chg_min is not None and (p["chg_pct"] is None or p["chg_pct"] < criteria.chg_min):
            continue
        matched.append(p)
        if len(matched) >= criteria.limit:
            break
    return matched
