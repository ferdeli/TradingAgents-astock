"""Sidebar: stock input, LLM config, and history calendar."""

from __future__ import annotations

import os
from datetime import date

import streamlit as st

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.checkpointer import clear_checkpoint
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
from web.history import (
    clear_incomplete_task,
    get_history,
    get_incomplete_history,
    record_incomplete_task,
)

# Provider display names in recommended order
_PROVIDERS: list[tuple[str, str]] = [
    ("MiniMax（推荐·国内直连）", "minimax"),
    ("DeepSeek", "deepseek"),
    ("通义千问 Qwen", "qwen"),
    ("智谱 GLM", "glm"),
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("Google Gemini", "google"),
    ("xAI Grok", "xai"),
    ("OpenRouter（聚合·填 vendor/model 形式 ID）", "openrouter"),
    ("OpenAI 兼容（自定义 base_url·9Router/AI Router/自建代理）", "openai_compatible"),
    ("Ollama（本地）", "ollama"),
]

_PROVIDER_DISPLAY = [name for name, _ in _PROVIDERS]
_PROVIDER_KEYS = [key for _, key in _PROVIDERS]


def _resolve_user_input(raw: str) -> tuple[str, str | None]:
    """Resolve raw user input to (ticker_code, error_msg).

    Accepts 6-digit codes or Chinese stock names (e.g. '宝光股份').
    Returns (code, None) on success or ("", error_msg) on failure.
    """
    from tradingagents.dataflows.a_stock import resolve_ticker

    try:
        code = resolve_ticker(raw)
        return code, None
    except ValueError as e:
        return "", str(e)


def _clear_analysis_artifacts(ticker: str, trade_date: str) -> None:
    clear_incomplete_task(ticker, trade_date)
    clear_checkpoint(DEFAULT_CONFIG["data_cache_dir"], ticker, trade_date)


def _render_analysis_controls(raw_ticker: str, trade_date_value: date) -> None:
    tracker = st.session_state.get("tracker")
    is_running = tracker is not None and tracker.is_running
    trade_date = trade_date_value.strftime("%Y-%m-%d")

    pause_col, resume_col, stop_col = st.columns(3)

    pause_disabled = not is_running or tracker.is_paused or tracker.stop_requested
    if pause_col.button(
        "暂停",
        key="sidebar_pause_analysis",
        use_container_width=True,
        disabled=pause_disabled,
    ):
        if tracker.pause():
            record_incomplete_task(
                tracker.ticker,
                tracker.trade_date,
                status="paused",
                completed_stages=tracker.completed_stages,
            )
        st.rerun()

    resume_disabled = not is_running or not tracker.is_paused or tracker.stop_requested
    if resume_col.button(
        "恢复",
        key="sidebar_resume_analysis",
        use_container_width=True,
        disabled=resume_disabled,
    ):
        if tracker.resume():
            record_incomplete_task(
                tracker.ticker,
                tracker.trade_date,
                status="running",
                completed_stages=tracker.completed_stages,
            )
        st.rerun()

    can_stop = tracker is not None or bool(raw_ticker.strip())
    if stop_col.button(
        "停止",
        key="sidebar_stop_analysis",
        use_container_width=True,
        disabled=not can_stop,
    ):
        target_ticker = tracker.ticker if tracker is not None and tracker.ticker else ""
        target_date = (
            tracker.trade_date
            if tracker is not None and tracker.trade_date
            else trade_date
        )

        if not target_ticker:
            target_ticker, err = _resolve_user_input(raw_ticker)
            if err:
                st.error(f"❌ {err}")
                return

        if tracker is not None and tracker.is_running:
            tracker.request_stop()
            clear_incomplete_task(target_ticker, target_date)
        else:
            if tracker is not None:
                tracker.mark_stopped()
                st.session_state["tracker"] = None
            _clear_analysis_artifacts(target_ticker, target_date)

        st.session_state["viewing_history"] = None
        st.success("已清空当前进度；下一次开始分析会从头生成。")
        st.rerun()

    if tracker is not None and tracker.stop_requested:
        st.caption("正在停止并清空，收尾完成后可重新开始。")


def _render_llm_config() -> None:
    """Render LLM provider and model selection controls."""

    provider_idx = st.selectbox(
        "LLM 供应商",
        range(len(_PROVIDERS)),
        format_func=lambda i: _PROVIDER_DISPLAY[i],
        index=_PROVIDER_KEYS.index("deepseek"),   # 默认 DeepSeek
        key="llm_provider_idx",
        help="选择你配置了 API Key 的供应商",
    )
    provider_key = _PROVIDER_KEYS[provider_idx]
    st.session_state["llm_provider"] = provider_key

    if provider_key in MODEL_OPTIONS:
        quick_options = MODEL_OPTIONS[provider_key]["quick"]
        deep_options = MODEL_OPTIONS[provider_key]["deep"]

        quick_labels = [label for label, _ in quick_options]
        quick_values = [value for _, value in quick_options]
        deep_labels = [label for label, _ in deep_options]
        deep_values = [value for _, value in deep_options]

        quick_idx = st.selectbox(
            "快速思考模型",
            range(len(quick_options)),
            format_func=lambda i: quick_labels[i],
            key="quick_model_idx",
            help="用于常规分析任务，速度优先",
        )
        st.session_state["quick_think_llm"] = quick_values[quick_idx]

        deep_idx = st.selectbox(
            "深度思考模型",
            range(len(deep_options)),
            format_func=lambda i: deep_labels[i],
            key="deep_model_idx",
            help="用于辩论/决策等需要深度推理的任务",
        )
        st.session_state["deep_think_llm"] = deep_values[deep_idx]
    else:
        custom_quick = st.text_input("快速思考模型 ID", key="custom_quick_model")
        custom_deep = st.text_input("深度思考模型 ID", key="custom_deep_model")
        st.session_state["quick_think_llm"] = custom_quick
        st.session_state["deep_think_llm"] = custom_deep

    base_url_required = provider_key == "openai_compatible"
    st.text_input(
        "API Base URL（第三方/代理" + ("·必填" if base_url_required else "，可选") + "）",
        key="llm_base_url",
        placeholder="例: https://your-relay.example/v1",
        help=(
            "通过第三方中转/代理访问模型时填写网关地址；留空则用所选供应商的官方地址。"
            "API Key 仍从 .env 读取，每个供应商用各自的环境变量——"
            "OpenAI=OPENAI_API_KEY、DeepSeek=DEEPSEEK_API_KEY、"
            "通义=DASHSCOPE_API_KEY、智谱=ZHIPU_API_KEY、MiniMax=MINIMAX_API_KEY、"
            "Claude=ANTHROPIC_API_KEY、OpenRouter=OPENROUTER_API_KEY、xAI=XAI_API_KEY、"
            "OpenAI 兼容（自定义）=OPENAI_COMPATIBLE_API_KEY（也接受 OPENAI_API_KEY）。"
            "也可在 .env 里设 BACKEND_URL 代替此处。"
        ),
    )
    if base_url_required:
        st.caption(
            "已选「OpenAI 兼容（自定义）」：**Base URL 必填**（你的网关，走标准 Chat "
            "Completions），模型 ID 手动填写，Key 在 .env 设 `OPENAI_COMPATIBLE_API_KEY`。"
        )

    # ── 个人 Claude 订阅额度（可选，仅个人自用）────────────────────────
    _scope_labels = [
        "关闭（走上面选的供应商）",
        "仅深度节点（Research/Portfolio）",
        "所有节点（含 7 个工具分析师）",
    ]
    _scope_values = ["off", "deep", "all"]
    scope_idx = st.selectbox(
        "个人 Claude 订阅覆盖 (Agent SDK)",
        range(len(_scope_labels)),
        format_func=lambda i: _scope_labels[i],
        key="subscription_scope_idx",
        help=(
            "让部分/全部节点经 Claude Agent SDK 走你个人 Pro/Max 订阅额度，"
            "而非按 token 计费。「所有节点」含 7 个工具分析师（其工具调用已桥接到订阅）。"
            "需装 [agentsdk] 依赖，且本机 claude 已登录（或设 CLAUDE_CODE_OAUTH_TOKEN）。"
        ),
    )
    scope = _scope_values[scope_idx]
    st.session_state["subscription_scope"] = scope
    if scope != "off":
        # 用别名而非写死版本号：claude CLI 的 opus/sonnet 恒指向最新模型。
        st.session_state.setdefault("agent_sdk_model", "opus")
        st.text_input(
            "订阅使用的 Claude 模型",
            key="agent_sdk_model",
            help=(
                "填别名 opus / sonnet（恒指向最新模型，推荐）或完整模型 id。"
                "撞额度/失败时自动降级到上面选的供应商 + 对应模型。"
            ),
        )
        if scope == "all":
            st.caption(
                "⚠️ 「所有节点」会把 7 个分析师 + 多空/交易员/风险辩手全部压到订阅上，"
                "订阅是按额度限流的，跑几轮就可能撞上限。可在 config 里把 "
                "`agent_sdk_quick_model` 设为 `sonnet` 降低消耗（默认已是）。"
            )
        if os.getenv("ANTHROPIC_API_KEY"):
            st.info(
                "检测到 ANTHROPIC_API_KEY。它**不会**泄进 Agent SDK 子进程"
                "（已在子进程环境显式置空），所以订阅额度照常生效；"
                "父进程保留它，是为了让 `anthropic` 仍能作为撞额度后的降级 provider。"
                "如果你并不打算保留付费降级，可在 .env 里清掉它。"
            )


def render_sidebar() -> None:
    """Render the sidebar with input controls and history."""

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1.5rem;">
            <span style="font-size:2rem; font-weight:800; color:#ff5a1f;">Trading</span><span style="font-size:2rem; font-weight:800; color:#f5f1eb;">Agents</span><span style="font-size:2rem; font-weight:800; color:#f5f1eb;">-</span><span style="font-size:2rem; font-weight:800; color:#ff5a1f;">Astock</span>
            <div style="font-size:0.85rem; color:#888; margin-top:0.2rem;">
                A股多Agent投研系统
            </div>
            <div style="font-size:0.7rem; color:#555; margin-top:0.3rem;">
                by <a href="https://github.com/simonlin1212" style="color:#ff5a1f; text-decoration:none;">simonlin1212</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 新建分析")

    # ── 多标的：动态列表录入（每行标的联动该行持仓均价/总量）──
    st.markdown("**分析标的**")
    ticker_count = st.session_state.get("ticker_count", 1)

    def _remove_ticker_row(idx: int) -> None:
        n = st.session_state.get("ticker_count", 1)
        keys = ("ticker_row_", "holding_cost_", "holding_qty_")
        vals = {k: [st.session_state.get(f"{k}{j}", "") for j in range(n)] for k in keys}
        for k in keys:
            vals[k].pop(idx)
            for j in range(n - 1):
                st.session_state[f"{k}{j}"] = vals[k][j]
            st.session_state[f"{k}{n - 1}"] = ""   # clear the tail slot
        st.session_state["ticker_count"] = max(1, n - 1)

    # 任务标题：属于整个分析任务（一个任务可含多个标的），不是单标的
    st.text_input(
        "任务标题（可选）",
        placeholder="本次分析任务的标题，可含一个或多个标的",
        key="batch_title",
        help="标题属于整个分析任务；留空则显示「批量分析 · N 个标的」或标的代码。",
    )

    h1, h2, h3 = st.columns([4, 2, 2])
    with h1:
        st.caption("标的")
    with h2:
        st.caption("持仓均价")
    with h3:
        st.caption("持仓总量")

    ticker_inputs: list[str] = []
    for i in range(ticker_count):
        # 每个任务一个分组容器：标的/持仓行对齐
        with st.container(border=True):
            c_t, c_p, c_q, c_x = st.columns([3, 1, 1, 1])
            with c_t:
                st.text_input(
                    f"标的 {i+1}",
                    placeholder="代码或名称",
                    key=f"ticker_row_{i}",
                    label_visibility="collapsed",
                    help="输入6位A股代码或中文股票全称；每行可填该标的的持仓均价/总量，分析时联动。",
                )
            with c_p:
                st.number_input(
                    f"均价{i}", min_value=0.0, step=0.01, value=0.0, format="%.2f",
                    key=f"holding_cost_{i}", label_visibility="collapsed",
                    help="该标的持仓均价（留 0 = 无持仓，不生成持仓操作建议）",
                )
            with c_q:
                st.number_input(
                    f"总量{i}", min_value=0, step=100, value=0,
                    key=f"holding_qty_{i}", label_visibility="collapsed",
                    help="该标的持仓总量（股）",
                )
        if val := st.session_state.get(f"ticker_row_{i}", ""):
            if val.strip():
                ticker_inputs.append(val.strip())
    if st.button("➕ 添加标的", key="add_ticker_row", use_container_width=True):
        st.session_state["ticker_count"] = ticker_count + 1

    # 键盘快捷：焦点在标的输入框时按 + / = 添加一行，并自动聚焦新行输入框。
    # st.iframe（HTML string → srcdoc iframe，允许 JS 且与主页面同源）注入
    # keydown 监听、程序化点击添加按钮，并在 rerun 完成后聚焦新行输入框。
    # 注：st.components.v1.html 已弃用（1.61 起），故改用 st.iframe。
    st.iframe(
        """
        <script>
        (function () {
          var p = window.parent.document;
          // iframe 每次 rerun 重建，script 会重新执行；用 parent 上的标记防重复注册
          if (p.__taAddRowHook) return;
          p.__taAddRowHook = true;
          p.addEventListener('keydown', function (e) {
            if ((e.key === '+' || e.key === '=') && e.target
                && e.target.tagName === 'INPUT'
                && e.target.placeholder === '代码或名称') {
              e.preventDefault();
              var btn = Array.prototype.slice.call(p.querySelectorAll('button'))
                  .find(function (b) { return b.textContent.indexOf('添加标的') >= 0; });
              if (btn) {
                btn.click();
                setTimeout(function () {
                  var inputs = Array.prototype.slice.call(p.querySelectorAll('input'))
                      .filter(function (i) { return i.placeholder === '代码或名称'; });
                  if (inputs.length) {
                    var last = inputs[inputs.length - 1];
                    last.focus();
                    last.select();
                  }
                }, 400);
              }
            }
          });
        })();
        </script>
        """,
        height="content",
    )

    tickers = ticker_inputs

    trade_date = st.date_input(
        "分析日期",
        value=date.today(),
        key="input_date",
    )

    start_date = st.date_input(
        "数据起始日期",
        value=trade_date.replace(day=1),   # 默认本月第一天
        key="input_start_date",
        help="技术分析回溯到该日期（默认本月第一天）。分析区间 = 起始日期 → 分析日期，"
             "用于「按月」或自定义时段分析；留默认即分析当月至今。",
    )
    # 分析窗口天数 → market_lookback_days（下限 5 天，保证指标有意义）
    st.session_state["market_lookback_days"] = max((trade_date - start_date).days, 5)
    if start_date >= trade_date:
        st.caption("⚠️ 起始日期应早于分析日期，已按最小窗口（5 天）处理。")

    with st.expander("⚙️ 模型配置", expanded=False):
        _render_llm_config()

    tracker = st.session_state.get("tracker")
    is_busy = tracker is not None and tracker.is_running
    is_stopping = is_busy and tracker.stop_requested

    if st.button(
        "开始分析" if not is_busy else "停止中..." if is_stopping else "分析进行中...",
        use_container_width=True,
        disabled=is_busy or not tickers,
        type="primary",
    ):
        resolved: list[str] = []
        errors: list[str] = []
        holdings_map: dict[str, dict] = {}
        for i in range(ticker_count):
            raw = st.session_state.get(f"ticker_row_{i}", "")
            if not raw or not raw.strip():
                continue
            code, err = _resolve_user_input(raw.strip())
            if err:
                errors.append(f"{raw}: {err}")
                continue
            resolved.append(code)
            # 每行标的联动该行持仓：均价 > 0 才视为有持仓
            cost = float(st.session_state.get(f"holding_cost_{i}", 0.0) or 0.0)
            if cost > 0:
                qty = int(st.session_state.get(f"holding_qty_{i}", 0) or 0)
                holdings_map[code] = {"quantity": qty, "cost_price": cost}
        if errors:
            st.error("❌ " + "；".join(errors))
        elif resolved:
            # 任务标题属于整个分析任务（一个任务可含多个标的）
            batch_title = str(st.session_state.get("batch_title", "") or "").strip()
            st.session_state["start_analysis"] = {
                "tickers": resolved,
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "holdings_map": holdings_map,
                "title": batch_title,
                "fresh": True,
            }
            st.session_state["viewing_history"] = None

    _render_analysis_controls(tickers[0] if tickers else "", trade_date)

    st.markdown("---")
    st.markdown("#### 未完成任务")

    incomplete = get_incomplete_history()
    if not incomplete:
        st.caption("暂无未完成任务")
    else:
        for entry in incomplete[:10]:
            t, d = entry["ticker"], entry["trade_date"]
            status_label = {
                "error": "出错",
                "paused": "已暂停",
                "running": "进行中",
            }.get(entry.get("status"), "可继续")
            step = entry.get("checkpoint_step")
            step_label = f" · step {step}" if step is not None else ""
            label = f"{t}  ·  {d}  ·  {status_label}{step_label}"
            if st.button(
                label,
                key=f"resume_{t}_{d}",
                use_container_width=True,
                disabled=is_busy,
            ):
                st.session_state["start_analysis"] = {
                    "ticker": t,
                    "trade_date": d,
                }
                st.session_state["viewing_history"] = None

    st.markdown("---")
    st.markdown("#### 历史记录")
    if st.button("📅 打开历史日历", key="open_calendar", use_container_width=True):
        st.session_state["show_calendar"] = True
        st.session_state.pop("history_board", None)
        st.session_state.pop("active_task", None)
        st.session_state["viewing_history"] = None
        st.session_state["viewing_batch"] = None
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ 仅供学习研究，不构成投资建议")
