"""History calendar view: a full-width month grid + the selected day's tasks.

Rendered in the main area (not the narrow sidebar) so the button grid has
room to breathe. All interaction uses native ``st.button`` widgets — every
click reruns immediately, so month switching and day picking are reliable
(HTML-component callbacks do not trigger reruns).
"""

from __future__ import annotations

import calendar as _cal
from datetime import date, timedelta

import streamlit as st

from web.history import (
    count_tasks_on,
    get_history,
    group_history,
    history_label,
    set_log_title,
    static_batch_from_item,
)


def _month_grid() -> None:
    """Full-width month grid: ◀/▶ navigation + weekday header + day buttons."""
    today = date.today()
    y, m = st.session_state.get("cal_ym") or (today.year, today.month)

    nav = st.columns([1, 3, 1])
    with nav[0]:
        if st.button("◀ 上月", key="cal_prev", use_container_width=True):
            m2 = m - 1
            st.session_state["cal_ym"] = (y + (m2 - 1) // 12, (m2 - 1) % 12 + 1)
            st.rerun()
    with nav[1]:
        st.markdown(
            f"<div style='text-align:center; font-size:1.1rem; font-weight:700;"
            f"padding-top:4px;'>{y}年{m}月</div>",
            unsafe_allow_html=True,
        )
    with nav[2]:
        if st.button("下月 ▶", key="cal_next", use_container_width=True):
            m2 = m + 1
            st.session_state["cal_ym"] = (y + (m2 - 1) // 12, (m2 - 1) % 12 + 1)
            st.rerun()

    counts: dict[str, int] = {}
    first = date(y, m, 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    d = first
    while d < nxt:
        cnt = count_tasks_on(d.strftime("%Y-%m-%d"))
        if cnt:
            counts[d.strftime("%Y-%m-%d")] = cnt
        d += timedelta(days=1)

    selected = st.session_state.get("cal_date") or ""

    heads = st.columns(7)
    for col, h in zip(heads, ["一", "二", "三", "四", "五", "六", "日"]):
        with col:
            st.caption(h, unsafe_allow_html=False)
    for week in _cal.Calendar(firstweekday=0).monthdayscalendar(y, m):
        cols = st.columns(7)
        for col, day in zip(cols, week):
            with col:
                if not day:
                    st.write("")
                    continue
                key = f"{y}-{m:02d}-{day:02d}"
                label = f"{day}" + (f"  ·{counts[key]}" if key in counts else "")
                if st.button(
                    label, key=f"calday_{key}", use_container_width=True,
                    type="primary" if key == selected else "secondary",
                ):
                    st.session_state["cal_date"] = key
                    st.session_state["history_board"] = None
                    st.session_state.pop("active_task", None)
                    st.rerun()
    st.caption("选中日 = 橙色按钮 · 日期后 ·N = 当日任务数")


def _day_tasks() -> None:
    """Task list of the selected day (classic list style + title editing)."""
    cal_date = st.session_state.get("cal_date")
    if not cal_date:
        st.caption("← 点击左侧日历中的日期查看当日任务")
        return
    st.subheader(f"📅 {cal_date} 的任务")
    items = group_history([e for e in get_history() if e["date"] == cal_date])
    if not items:
        st.caption("当日无分析记录")
        return
    for it in items:
        c1, c2 = st.columns([6, 1])
        with c1:
            if st.button(
                history_label(it), key=f"daytask_{it['key']}", use_container_width=True,
            ):
                st.session_state["history_board"] = static_batch_from_item(it)
                st.session_state.pop("active_task", None)
                st.rerun()
        with c2:
            if st.button("✎", key=f"daytask_edit_{it['key']}", use_container_width=True):
                st.session_state["editing_hist"] = it["key"]
                st.rerun()
        if st.session_state.get("editing_hist") == it["key"]:
            new_title = st.text_input(
                "新标题", value=it.get("title", ""), key=f"daytask_ti_{it['key']}"
            )
            if st.button("保存", key=f"daytask_save_{it['key']}"):
                if it["kind"] == "batch":
                    for e in it["entries"]:
                        set_log_title(e["path"], new_title)
                else:
                    set_log_title(it["path"], new_title)
                st.session_state["editing_hist"] = None
                st.rerun()


def render_calendar_view() -> None:
    """Left column: month grid. Right column: selected day's task list."""
    col_cal, col_tasks = st.columns([3, 2])
    with col_cal:
        _month_grid()
    with col_tasks:
        _day_tasks()
