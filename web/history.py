"""Manage completed and incomplete analysis history."""

from __future__ import annotations

import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG


logger = logging.getLogger(__name__)

_INCOMPLETE_TASKS_FILE = Path.home() / ".tradingagents" / "incomplete_tasks.json"
_INCOMPLETE_TASKS_LOCK = threading.Lock()


def _results_dir() -> Path:
    return Path.home() / ".tradingagents" / "logs"


def get_history() -> list[dict[str, str]]:
    """Scan saved analysis logs and return a sorted list (newest first).

    Each entry: {"ticker", "date", "path", "batch_id", "title", "name"} —
    ``batch_id`` groups the tickers of one multi-ticker analysis; ``title`` /
    ``name`` drive the history label.
    """
    root = _results_dir()
    if not root.exists():
        return []

    entries: list[dict[str, str]] = []
    for log_file in root.rglob("full_states_log_*.json"):
        match = re.search(r"full_states_log_(\d{4}-\d{2}-\d{2})\.json$", log_file.name)
        if not match:
            continue
        date = match.group(1)
        ticker = log_file.parent.parent.name
        meta = _log_meta(log_file)
        entries.append({
            "ticker": ticker,
            "date": date,
            "path": str(log_file),
            **meta,
        })

    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


def _log_meta(path: Path) -> dict[str, str]:
    """Extract batch_id / title / display name from a saved state log."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"batch_id": "", "title": "", "name": ""}
    name = ""
    try:
        from web.stock_display import _extract_stock_name_from_state

        name = _extract_stock_name_from_state(
            str(data.get("company_of_interest", "")), data
        ) or ""
    except Exception:  # noqa: BLE001 — name is cosmetic
        name = ""
    return {
        "batch_id": str(data.get("batch_id", "") or ""),
        "title": str(data.get("title", "") or ""),
        "name": name,
    }


def get_batch(batch_id: str) -> list[dict[str, str]]:
    """All history entries belonging to one multi-ticker batch."""
    return [e for e in get_history() if e.get("batch_id") == batch_id]


def group_history(entries: list[dict]) -> list[dict]:
    """Group history entries into display items: multi-ticker batches collapse
    into one item, single-ticker entries stay as-is.

    Returns items with: kind (batch|single), key, date, title, plus batch
    fields (batch_id/entries) or single fields (path/ticker/name).
    """
    batches: dict[str, list] = {}
    singles: list = []
    for e in entries:
        if e.get("batch_id"):
            batches.setdefault(e["batch_id"], []).append(e)
        else:
            singles.append(e)

    display: list[dict] = []
    for bid, batch_entries in batches.items():
        display.append({
            "kind": "batch", "key": bid, "batch_id": bid, "entries": batch_entries,
            "date": max(e["date"] for e in batch_entries),
            "title": next((e["title"] for e in batch_entries if e.get("title")), ""),
        })
    for e in singles:
        display.append({
            "kind": "single",
            "key": f"{e['ticker']}_{e['date']}_{abs(hash(e['path']))}",
            "path": e["path"], "ticker": e["ticker"], "date": e["date"],
            "title": e.get("title", ""), "name": e.get("name", ""),
        })
    display.sort(key=lambda x: x["date"], reverse=True)
    return display


def history_label(item: dict) -> str:
    """Human label for a history item (single or batch), per the display rules:

    - unnamed single ticker: 「名称（代码）· 日期」 (name from the log, code only when unknown)
    - unnamed batch:        「批量分析 · N 个标的 · 日期」
    - custom title wins over both.
    """
    if item["kind"] == "batch":
        n = len(item["entries"])
        return (item["title"] or f"批量分析 · {n} 个标的") + f" · {item['date']}"
    if item["title"]:
        return f"{item['title']} · {item['date']}"
    if item["name"]:
        return f"{item['name']}（{item['ticker']}）· {item['date']}"
    return f"{item['ticker']} · {item['date']}"


def count_tasks_on(day: str) -> int:
    """Number of analysed tickers on a given day (batch members count as tasks)."""
    return sum(
        1 for e in get_history() if e["date"] == day
    )


def set_log_title(path: str, title: str) -> None:
    """Persist a user-edited title into a saved state log.

    Only touches the ``title`` field — the analysis content is never changed,
    so editing a title never alters the historical record itself.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = title
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except (OSError, json.JSONDecodeError):
        pass  # title editing is cosmetic; failure must not break the page


def static_batch_from_item(item: dict) -> dict:
    """Reconstruct a read-only batch dict from a history item (single or batch)
    so the shared grid-panel view can render historical results."""
    entries = item["entries"] if item["kind"] == "batch" else [item]
    tickers: list[str] = []
    results: dict = {}
    titles: dict = {}
    for e in entries:
        state = load_analysis(e["path"])
        tickers.append(e["ticker"])
        titles[e["ticker"]] = e.get("title") or e["ticker"]
        results[e["ticker"]] = {
            "final_state": state,
            "signal": extract_signal(state),
            "trade_date": e["date"],
            "error": None,
        }
    return {
        "tickers": tickers,
        "index": len(tickers),
        "done": True,
        "interrupted": False,
        "titles": titles,
        "results": results,
        "trade_date": entries[0]["date"] if entries else "",
    }


def _completed_key(ticker: str, trade_date: str) -> tuple[str, str]:
    return ticker.upper(), trade_date


def _completed_keys() -> set[tuple[str, str]]:
    return {
        _completed_key(entry["ticker"], entry["date"])
        for entry in get_history()
    }


def _load_incomplete_index() -> list[dict[str, Any]]:
    if not _INCOMPLETE_TASKS_FILE.exists():
        return []

    try:
        with open(_INCOMPLETE_TASKS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        trade_date = str(item.get("trade_date", "")).strip()
        if not ticker or not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            continue
        item["ticker"] = ticker
        item["trade_date"] = trade_date
        entries.append(item)
    return entries


def _save_incomplete_index(entries: list[dict[str, Any]]) -> None:
    """原子写 incomplete_tasks.json，兼容 Windows 文件占用。

    目标文件可能被其他进程短暂占用（如多实例 Web UI、杀毒软件扫描），
    此时 ``tmp.replace`` 在 Windows 上会抛 ``PermissionError``（#77）。
    先重试几次等待锁释放，仍失败则降级为直接覆写——读取端
    （``_load_incomplete_index``）已容错损坏 JSON，索引写不进去不致命。
    """
    parent = _INCOMPLETE_TASKS_FILE.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, ensure_ascii=False, indent=2)

    for attempt in range(3):
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=parent,
                prefix=f"{_INCOMPLETE_TASKS_FILE.stem}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                f.write(payload)
                tmp = Path(f.name)
            tmp.replace(_INCOMPLETE_TASKS_FILE)
            return
        except PermissionError:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            if attempt < 2:
                # 锁通常是瞬时的，短暂等待后重试
                time.sleep(0.15 * (attempt + 1))
        except OSError:
            raise

    # 重试耗尽仍被占用：直接覆写（非原子但可接受）。
    try:
        _INCOMPLETE_TASKS_FILE.write_text(payload, encoding="utf-8")
    except OSError as e:
        # 索引写不进去不致命——读取端容错、下次写入会自动重建，所以不往上抛。
        # 但**不能一声不吭**：完全静默的话，用户永远不会知道它一直在失败，
        # 「未完成任务」列表长期不更新时也无从排查。
        logger.warning(
            "写入未完成任务索引失败（已重试并降级为直接覆写）：%s。"
            "不影响本次分析，但侧边栏的未完成任务列表可能不是最新的。", e
        )


def _checkpoint_step(ticker: str, trade_date: str) -> int | None:
    try:
        from tradingagents.graph.checkpointer import checkpoint_step

        return checkpoint_step(DEFAULT_CONFIG["data_cache_dir"], ticker, trade_date)
    except Exception:
        return None


def record_incomplete_task(
    ticker: str,
    trade_date: str,
    *,
    status: str,
    error: str | None = None,
    completed_stages: list[str] | None = None,
) -> None:
    """Upsert a resumable task entry."""
    ticker = ticker.strip().upper()
    trade_date = trade_date.strip()
    if not ticker or not trade_date:
        return

    with _INCOMPLETE_TASKS_LOCK:
        entries = [
            entry
            for entry in _load_incomplete_index()
            if _completed_key(entry["ticker"], entry["trade_date"])
            != _completed_key(ticker, trade_date)
        ]
        now = time.time()
        entries.append(
            {
                "ticker": ticker,
                "trade_date": trade_date,
                "status": status,
                "error": error or "",
                "completed_stages": completed_stages or [],
                "updated_at": now,
            }
        )
        entries.sort(key=lambda e: float(e.get("updated_at", 0)), reverse=True)
        _save_incomplete_index(entries)


def clear_incomplete_task(ticker: str, trade_date: str) -> None:
    """Remove an incomplete task once it completes successfully."""
    ticker = ticker.strip().upper()
    trade_date = trade_date.strip()
    with _INCOMPLETE_TASKS_LOCK:
        entries = [
            entry
            for entry in _load_incomplete_index()
            if _completed_key(entry["ticker"], entry["trade_date"])
            != _completed_key(ticker, trade_date)
        ]
        _save_incomplete_index(entries)


def get_incomplete_history() -> list[dict[str, Any]]:
    """Return unfinished tasks that can be resumed from their checkpoint."""
    completed = _completed_keys()
    active_entries: list[dict[str, Any]] = []

    with _INCOMPLETE_TASKS_LOCK:
        entries = _load_incomplete_index()
        for entry in entries:
            key = _completed_key(entry["ticker"], entry["trade_date"])
            if key in completed:
                continue

            step = _checkpoint_step(entry["ticker"], entry["trade_date"])
            entry["checkpoint_step"] = step
            active_entries.append(entry)

        active_entries.sort(key=lambda e: float(e.get("updated_at", 0)), reverse=True)
        if len(active_entries) != len(entries):
            _save_incomplete_index(active_entries)
    return active_entries


def load_analysis(path: str) -> dict[str, Any]:
    """Load a saved analysis JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_signal(state: dict[str, Any]) -> str:
    """Extract the 5-tier rating from a final state dict for history reload.

    Delegates to the shared ``parse_rating`` heuristic so the history-reload
    display matches the live signal (``TradingAgentsGraph.process_signal``) and
    understands Chinese free-text decisions — not just English keywords. The
    old English-only ``BUY/SELL/HOLD`` scan silently returned Hold/N/A for
    every Chinese-output run (issues #78 / #80). ``final_trade_decision`` is
    checked first so the reload matches the authoritative live signal.
    """
    import re

    from tradingagents.agents.utils.rating import parse_rating

    _UNKNOWN = ""
    for field in (
        "final_trade_decision",
        "trader_investment_decision",
        "investment_plan",
    ):
        text = state.get(field, "")
        if not text:
            continue
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        rating = parse_rating(cleaned, default=_UNKNOWN)
        if rating:
            return rating
    return "N/A"
