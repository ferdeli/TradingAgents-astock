"""Streamlit K-line viewer: historical candlesticks + forecast overlay (M5).

``_build_figure`` is a pure plotly construction (testable offline);
``render_kline`` wires it into Streamlit.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def _build_figure(chart_data: dict[str, Any]) -> go.Figure:
    """Build the candlestick figure. History solid; forecast dashed/translucent.

    A-share colour convention: **red = up, green = down** (opposite of the
    US convention plotly defaults to).
    """
    hist = chart_data.get("history", [])
    forecast = chart_data.get("forecast", [])
    rating = chart_data.get("rating", "")

    fig = go.Figure()
    # Date shown on hover via customdata (x stays an integer index, so history
    # and forecast axes never mix types). Legacy disk caches without a "date"
    # field fall back to the candle index.
    hist_custom = [[h.get("date") or f"#{h['index']}"] for h in hist]
    fig.add_trace(
        go.Candlestick(
            x=[h["index"] for h in hist],
            open=[h["open"] for h in hist],
            high=[h["high"] for h in hist],
            low=[h["low"] for h in hist],
            close=[h["close"] for h in hist],
            name="历史",
            increasing_line_color="#ef4444",   # 红涨
            decreasing_line_color="#22c55e",   # 绿跌
            customdata=hist_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "开 %{open:.2f} 高 %{high:.2f} 低 %{low:.2f} 收 %{close:.2f}"
                "<extra></extra>"
            ),
        )
    )
    if forecast:
        x = list(range(len(hist), len(hist) + len(forecast)))
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=[f["open"] for f in forecast],
                high=[f["high"] for f in forecast],
                low=[f["low"] for f in forecast],
                close=[f["close"] for f in forecast],
                name="预测（示意）",
                increasing_line_color="rgba(239,68,68,0.65)",
                decreasing_line_color="rgba(34,197,94,0.65)",
                increasing_fillcolor="rgba(239,68,68,0.25)",
                decreasing_fillcolor="rgba(34,197,94,0.25)",
                hovertemplate=(
                    "<b>预测（示意）</b><br>"
                    "开 %{open:.2f} 高 %{high:.2f} 低 %{low:.2f} 收 %{close:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"K线走势（评级: {rating}）",
        xaxis_rangeslider_visible=False,
        height=420,
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    return fig


def render_kline(st, chart_data: dict[str, Any]) -> None:
    """Render the chart into the passed ``streamlit`` module."""
    fig = _build_figure(chart_data)
    st.plotly_chart(fig, use_container_width=True)
    if chart_data.get("forecast"):
        st.caption("⚠️ 虚线区为基于评级与目标/止损位生成的示意性预测，仅供研究参考，非真实行情。")


def render_kline_thumbnail(st, chart_data: dict[str, Any], height: int = 110) -> None:
    """Render a small, static K-line cover for a task panel cell."""
    fig = _build_figure(chart_data)
    fig.update_layout(
        height=height,
        showlegend=False,
        margin=dict(l=2, r=2, t=6, b=2),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode=False,
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False, "staticPlot": True},
    )
