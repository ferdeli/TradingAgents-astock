# TradingAgents/graph/propagation.py

import re
from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


def _digits(value: object) -> str:
    """Extract the numeric part of a code, e.g. 'sh600519' / '600519' -> '600519'."""
    return re.sub(r"\D", "", str(value))


def _holding_matches(holding: Dict[str, Any], company_name: str) -> bool:
    """True when a holding entry corresponds to the analysed instrument.

    Matches on the 6-digit code (normalised, prefix/suffix tolerant) or on an
    exact Chinese/display name. No fuzzy/partial matching, so unrelated
    names never leak into the prompt.
    """
    code = _digits(holding.get("code", ""))
    name = str(holding.get("name", "")).strip()
    target_code = _digits(company_name)
    target_name = str(company_name).strip()
    if code and len(code) >= 6 and code == target_code:
        return True
    return bool(name) and name == target_name


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str, past_context: str = ""
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        ``holdings`` is injected from the current config, filtered to entries
        that match ``company_name`` (6-digit code or exact name), so the
        Portfolio Manager can produce holding-management guidance.
        """
        from tradingagents.dataflows.config import get_config

        all_holdings = get_config().get("holdings", []) or []
        matching = [
            h for h in all_holdings if _holding_matches(h, company_name)
        ]
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "holdings": matching,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "policy_report": "",
            "hot_money_report": "",
            "lockup_report": "",
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
