"""TradingAgents A股分析 — Streamlit Web UI."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# override=True：让 .env 的值优先于进程里可能残留的空/旧环境变量（#66）。
# 注意：load_dotenv 仅在进程启动时执行一次，启动后修改 .env 仍需重启 Web 服务才生效。
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

from web.components.progress_panel import render_progress  # noqa: E402
from web.components.report_viewer import render_report  # noqa: E402
from web.components.sidebar import render_sidebar  # noqa: E402
from web.history import clear_incomplete_task, extract_signal, load_analysis  # noqa: E402
from web.progress import ProgressTracker  # noqa: E402
from web.runner import run_analysis_in_thread  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradingAgents-Astock A股分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    /* Hide Streamlit chrome for clean video recording.
       IMPORTANT: do NOT `display:none` the whole header OR the whole toolbar.
       In Streamlit >= 1.36 the "expand sidebar" button lives *inside* the
       toolbar (header > stToolbar > stExpandSidebarButton), so hiding either
       one makes a collapsed sidebar impossible to reopen (issue #36). Instead
       keep the header/toolbar in the DOM, make the header transparent, and
       hide only the individual chrome widgets we don't want on camera. */
    #MainMenu,
    footer,
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stToolbarActions"],
    div[data-testid="stAppDeployButton"],
    span[data-testid="stMainMenu"] { display: none !important; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Keep the sidebar collapse / expand controls always visible & clickable.
       Selector list spans multiple Streamlit versions. */
    button[data-testid="stExpandSidebarButton"],
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stApp {
        background: #0a0a0a;
    }
    section[data-testid="stSidebar"] {
        background: #0f0f0f;
        border-right: 1px solid #1a1a1a;
    }
    .stMetric label { color: #888 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] {
        color: #ff5a1f !important;
        font-weight: 700 !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff5a1f, #ff8c42) !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #ff5a1f, #ff8c42) !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 15px rgba(255,90,31,0.3) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #e04d15, #ff5a1f) !important;
        box-shadow: 0 6px 20px rgba(255,90,31,0.4) !important;
        transform: translateY(-1px) !important;
    }
    /* Secondary buttons (history items) */
    button[kind="secondary"] {
        background: #161616 !important;
        border: 1px solid #2a2a2a !important;
        color: #ccc !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background: #1e1e1e !important;
        border-color: #ff5a1f !important;
        color: #ff5a1f !important;
    }
    .stExpander {
        border: 1px solid #222 !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ff5a1f !important;
        border-bottom-color: #ff5a1f !important;
    }
    div[data-testid="stDownloadButton"] button {
        background: #1a1a2e !important;
        border: 1px solid #ff5a1f !important;
        color: #ff5a1f !important;
    }
    /* Text input styling */
    input[data-testid="stTextInputRootElement"] input,
    .stTextInput input {
        background: #161616 !important;
        border-color: #2a2a2a !important;
        color: #f5f1eb !important;
    }
    .stTextInput input:focus {
        border-color: #ff5a1f !important;
        box-shadow: 0 0 0 1px #ff5a1f !important;
    }
    /* Date input styling */
    .stDateInput input {
        background: #161616 !important;
        border-color: #2a2a2a !important;
        color: #f5f1eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Build config ─────────────────────────────────────────────────────────────

def _build_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = st.session_state.get("llm_provider", "deepseek")
    config["deep_think_llm"] = st.session_state.get("deep_think_llm", "deepseek-v4-pro")
    config["quick_think_llm"] = st.session_state.get("quick_think_llm", "deepseek-v4-flash")
    # Optional third-party / proxy endpoint. Sidebar input wins, else .env BACKEND_URL.
    backend_url = (st.session_state.get("llm_base_url") or os.getenv("BACKEND_URL") or "").strip()
    config["backend_url"] = backend_url or None
    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }
    # Analysis window (#16): start-date input in the sidebar → look-back days.
    config["market_lookback_days"] = st.session_state.get("market_lookback_days")
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["checkpoint_enabled"] = True
    config["output_language"] = "Chinese"
    # Optional: route nodes through a personal Claude Pro/Max subscription (Agent
    # SDK). Scope: "deep" = Research/Portfolio only; "all" = + the 7 analysts.
    # Leaving the fallback keys None makes the graph fall back to the
    # sidebar-selected llm_provider + models on quota/failure.
    scope = st.session_state.get("subscription_scope", "off")
    # 侧栏那个输入框只配**深度节点**的模型。不要把它同时赋给 quick——
    # quick 节点有 7 个分析师 + 多空/交易员/风险辩手，把深度节点的 opus 复制过去
    # 会让订阅额度烧得极快，也与 README / 侧栏提示所说的「quick 默认 sonnet」矛盾。
    # quick 的模型交给 DEFAULT_CONFIG（默认 sonnet），需要时在 config 层单独覆盖。
    sub_model = st.session_state.get("agent_sdk_model")
    if scope in ("deep", "all"):
        config["deep_think_provider_override"] = "claude_agent_sdk"
        if sub_model:
            config["agent_sdk_model"] = sub_model
    if scope == "all":
        config["quick_think_provider_override"] = "claude_agent_sdk"
    # Holdings are per-ticker now (each sidebar row carries its own cost/qty);
    # the batch runner injects the matching holding into config per ticker.
    return config


def _start_batch_current() -> None:
    """Start (or restart) the tracker for the current batch index."""
    from tradingagents.graph.checkpointer import clear_checkpoint

    batch = st.session_state["batch"]
    ticker = batch["tickers"][batch["index"]]
    clear_incomplete_task(ticker, batch["trade_date"])
    clear_checkpoint(DEFAULT_CONFIG["data_cache_dir"], ticker, batch["trade_date"])
    # Per-ticker holdings: this ticker's row (均价>0) → config.holdings.
    config = dict(batch["config"])
    config["holdings"] = (batch.get("holdings_map") or {}).get(ticker)
    tracker = ProgressTracker(
        ticker=ticker,
        trade_date=batch["trade_date"],
    )
    st.session_state["tracker"] = tracker
    run_analysis_in_thread(
        ticker=ticker,
        trade_date=batch["trade_date"],
        config=config,
        tracker=tracker,
    )


def _advance_batch(*, failed: bool = False) -> None:
    """Record the current ticker's outcome and start the next one, or finish."""
    batch = st.session_state["batch"]
    tracker = st.session_state.get("tracker")
    result = {
        "final_state": None if failed else tracker.final_state,
        "signal": "ERROR" if failed else tracker.signal,
        "trade_date": tracker.trade_date,
        "error": tracker.error if failed else None,
    }
    batch["results"][tracker.ticker] = result
    st.session_state["tracker"] = None
    batch["index"] += 1
    if batch["index"] < len(batch["tickers"]) and not batch.get("interrupted"):
        _start_batch_current()
    else:
        batch["done"] = True


def _render_batch_results() -> None:
    """Render per-ticker tabs for the completed results of a batch."""
    batch = st.session_state.get("batch")
    if not batch or not batch.get("results"):
        return
    done_tickers = [t for t in batch["tickers"] if t in batch["results"]]
    if not done_tickers:
        return
    if not batch.get("done"):
        idx = batch["index"]
        status = (
            f"，正在分析 {batch['tickers'][idx]}"
            if idx < len(batch["tickers"]) else ""
        )
        st.info(f"⏳ 批量进度：已完成 {len(batch['results'])}/{len(batch['tickers'])}{status}")
    elif batch.get("interrupted"):
        st.warning("已停止：仅展示已完成标的的分析结果。")
    tabs = st.tabs([f"📈 {t}" for t in done_tickers])
    for tab, t in zip(tabs, done_tickers):
        with tab:
            r = batch["results"][t]
            if r.get("error"):
                st.error(f"分析失败: {r['error']}")
            else:
                render_report(r["final_state"], t, r["trade_date"], r["signal"], offline=True)


def _render_batch_progress(tracker) -> None:
    """Per-ticker tabs WHILE the batch runs: the current ticker shows its live
    progress panel, finished tickers show their results, pending ones wait.

    Keeps every ticker's output on its own tab so a multi-ticker run stays
    readable. Results inside the progress view render offline (cache-only) so
    a rerun never blocks on network fetches; the full report is available
    from the final results tabs once the batch finishes.
    """
    batch = st.session_state["batch"]
    labels: list[str] = []
    for i, t in enumerate(batch["tickers"]):
        if t in batch["results"]:
            labels.append(f"✅ {t}")
        elif i == batch["index"]:
            labels.append(f"⏳ {t}")
        else:
            labels.append(f"⏸ {t}")
    tabs = st.tabs(labels)
    for i, (tab, t) in enumerate(zip(tabs, batch["tickers"])):
        with tab:
            if t in batch["results"]:
                r = batch["results"][t]
                if r.get("error"):
                    st.error(f"分析失败: {r['error']}")
                else:
                    render_report(r["final_state"], t, r["trade_date"], r["signal"], offline=True)
            elif i == batch["index"]:
                render_progress(tracker)
            else:
                st.caption("⏸ 等待分析…")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    render_sidebar()


# ── Handle "Start Analysis" trigger (multi-ticker batch) ─────────────────────

start_req = st.session_state.pop("start_analysis", None)
if start_req:
    tickers = start_req.get("tickers") or [start_req.get("ticker", "")]
    trade_date = start_req["trade_date"]
    st.session_state["batch"] = {
        "tickers": [t for t in tickers if t],
        "index": 0,
        "trade_date": trade_date,
        "config": _build_config(),
        "holdings_map": start_req.get("holdings_map", {}),  # ticker -> {quantity, cost_price}
        "results": {},          # ticker -> {final_state, signal, trade_date, error?}
        "done": False,
        "interrupted": False,
    }
    st.session_state["viewing_history"] = None
    _start_batch_current()


# ── Main area state machine ─────────────────────────────────────────────────

tracker: ProgressTracker | None = st.session_state.get("tracker")
viewing_history: str | None = st.session_state.get("viewing_history")
batch = st.session_state.get("batch")

# State 1: Viewing a historical analysis
if viewing_history:
    try:
        state = load_analysis(viewing_history)
        signal = extract_signal(state)
        ticker = Path(viewing_history).parent.parent.name
        trade_date = Path(viewing_history).stem.replace("full_states_log_", "")
        # offline=True: history browsing must never block on network fetches
        # (K-line shows from cache only; name from the saved state).
        render_report(state, ticker, trade_date, signal, offline=True)
    except Exception as exc:
        st.error(f"加载失败: {exc}")

# Batch mode: stopped current ticker → interrupt the whole batch
elif batch and tracker and tracker.stop_requested:
    batch["interrupted"] = True
    batch["done"] = True
    st.session_state["tracker"] = None
    st.rerun()

# Batch mode: analysis running → per-ticker tabs (current shows live progress)
elif batch and tracker and tracker.is_running:
    _render_batch_progress(tracker)
    time.sleep(2)
    st.rerun()

# Batch mode: current ticker finished → advance to next / done
elif batch and tracker and tracker.is_complete:
    _advance_batch()
    st.rerun()

# Batch mode: current ticker errored → skip it and continue the batch
elif batch and tracker and tracker.error:
    _advance_batch(failed=True)
    st.rerun()

# Batch mode: done (or interrupted) → render all results in tabs
elif batch and batch.get("done"):
    _render_batch_results()

# State 2: Analysis running (single ticker, no batch)
elif tracker and tracker.is_running:
    render_progress(tracker)
    time.sleep(2)
    st.rerun()

# State 3: Analysis complete (single ticker)
elif tracker and tracker.is_complete:
    render_report(
        tracker.final_state,
        tracker.ticker,
        tracker.trade_date,
        tracker.signal,
        elapsed=tracker.elapsed,
    )

# State 4: Analysis errored (single ticker)
elif tracker and tracker.error:
    st.error(f"分析失败: {tracker.error}")
    st.caption("已完成阶段会保存在本地断点中；修复模型额度或配置后，可以继续未完成的部分。")
    if st.button("继续未完成任务", type="primary"):
        st.session_state["start_analysis"] = {
            "ticker": tracker.ticker,
            "trade_date": tracker.trade_date,
        }
        st.session_state["viewing_history"] = None
        st.rerun()

# State 0: Idle — welcome screen
else:
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <div style="
                font-size: 2.5rem;
                font-weight: 900;
                margin-bottom: 0.5rem;
            ">
                <span style="color: #ff5a1f;">Trading</span><span style="color: #f5f1eb;">Agents</span><span style="color: #f5f1eb;">-</span><span style="color: #ff5a1f;">Astock</span>
            </div>
            <div style="color: #888; font-size: 1.1rem; max-width: 500px; line-height: 1.6;">
                A股多Agent投研分析系统<br>
                7位AI分析师 → 质量门控 → 多空辩论 → 风控评估 → 最终决策
            </div>
            <div style="
                margin-top: 2rem;
                padding: 1rem 2rem;
                border: 1px solid #222;
                border-radius: 12px;
                color: #666;
                font-size: 0.9rem;
            ">
                ← 在左侧输入股票代码，开始分析
            </div>
            <div style="
                margin-top: 2.5rem;
                padding: 0.8rem 1.5rem;
                color: #555;
                font-size: 0.75rem;
                max-width: 500px;
                line-height: 1.6;
                border-top: 1px solid #1a1a1a;
            ">
                ⚠️ 本项目仅供学习研究与技术演示，不构成任何投资建议。<br>
                投资决策请咨询持牌专业机构。作者不对使用本工具产生的任何损失承担责任。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
