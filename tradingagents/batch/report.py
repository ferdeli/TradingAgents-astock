"""Batch summary report rendering (markdown)."""

from __future__ import annotations

from tradingagents.batch.runner import BatchResult, sort_results


def render_summary(results: list[BatchResult]) -> str:
    """Render the batch summary as a markdown table + failure list."""
    ordered = sort_results(results)
    lines = [
        "# 批量分析汇总",
        "",
        f"共 {len(results)} 个标的，成功 {sum(1 for r in results if r.status == 'ok')}，"
        f"失败 {sum(1 for r in results if r.status != 'ok')}。",
        "",
        "| 标的 | 评级 | 执行建议 | 一页理由 | 状态 |",
        "|---|---|---|---|---|",
    ]
    for r in ordered:
        advice = ""
        if r.execution_advice:
            for ln in r.execution_advice.splitlines():
                if ln.startswith("**Position Size**") or ln.startswith("**Entry Zone**"):
                    advice = ln.split(":", 1)[-1].strip()
                    break
        status = "✅" if r.status == "ok" else f"❌ {r.error}"
        thesis = (r.thesis or "").replace("|", "\\|")
        lines.append(f"| {r.code} | {r.signal or '-'} | {advice or '-'} | {thesis} | {status} |")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        lines += ["", "## 失败清单", ""]
        for r in failed:
            lines.append(f"- **{r.code}**: {r.error}")
    return "\n".join(lines)
