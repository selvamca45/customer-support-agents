"""
agents/log_agent.py - Agent 4 of 4: the Log Agent.

Job: record the ticket and decide whether a human must take over.
It calls ONE tool, log_ticket(extra_concern), which:
  1. collects everything the other agents produced (stored in a TicketContext),
  2. computes the confidence and runs the grounding check,
  3. applies the escalation rules (agents/escalation.py),
  4. appends one row to logs/tickets.csv (+ Excel copy),
  5. returns "LOGGED ..." so the team knows the work is finished.

Why does the tool read a TicketContext instead of taking every field as an argument?
If the LLM had to copy the whole reply and analysis into the tool call, it could
shorten or change them by mistake. Reading them directly from what the agents
produced guarantees the log contains exactly what really happened.
The LLM's own judgement is still used: it can raise an extra_concern that the fixed
rules might miss (for example "customer says this is the third time").
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from agents.escalation import decide_escalation
from agents.kb_search import parse_kb_output
from agents.query_agent import QueryAnalysis
from agents.resolution_agent import ResolutionResult, compute_confidence, grounding_check
from agents.ticket_logger import append_ticket, next_ticket_id

# Sent instead of the draft if the draft failed the grounding check
SAFE_REPLY = (
    "Thank you for contacting BluePeak Store. We want to be sure you get a fully "
    "accurate answer, so a support specialist is reviewing your request and will "
    "reply to you shortly.\n\nWarm regards,\nBluePeak Support Team"
)


@dataclass
class TicketContext:
    """Everything known about one ticket while the agents work on it."""
    complaint: str
    customer_name: str = ""
    order_id: str = ""
    started_at: float = field(default_factory=time.perf_counter)
    analysis: QueryAnalysis | None = None       # from query_agent
    kb_text: str = ""                           # from knowledge_agent
    resolution: ResolutionResult | None = None  # from resolution_agent
    result: dict | None = None                  # the logged row, after log_ticket
    steps: list = field(default_factory=list)   # timeline of agent steps (for the UI)
    stop_reason: str = ""                       # why the team stopped
    error: str = ""                             # set if something went wrong


def build_ticket_row(ctx: TicketContext, extra_concern: str = "") -> dict:
    """Combine all agent outputs into one log row with the escalation decision."""
    a, r = ctx.analysis, ctx.resolution
    kb = parse_kb_output(ctx.kb_text)

    if r is not None:
        check = grounding_check(r, ctx.kb_text, kb["sources"], ctx.complaint, ctx.order_id)
        confidence = compute_confidence(r.coverage, kb["best_score"], check["grounded"])
        draft, coverage = r.reply, r.coverage
    else:  # the Resolution Agent did not produce a result
        check = {"grounded": False, "ungrounded_numbers": [], "ungrounded_paths": [], "invalid_sources": []}
        confidence, draft, coverage = 0.0, "", "none"

    status, reasons = decide_escalation(
        urgency=a.urgency if a else "High",
        sentiment=a.sentiment if a else "Neutral",
        risk_flags=list(a.risk_flags) if a else [],
        kb_no_match=kb["no_match"],
        confidence=confidence,
        grounded=check["grounded"],
        extra_concern=extra_concern,
    )

    if check["grounded"]:
        grounding = "OK"
    else:
        problems = check["ungrounded_numbers"] + check["ungrounded_paths"] + check["invalid_sources"]
        grounding = "FAILED: " + (", ".join(problems) or "no reply")

    return {
        "ticket_id": next_ticket_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": ctx.customer_name,
        "order_id": ctx.order_id,
        "complaint": ctx.complaint,
        "category": a.category if a else "",
        "urgency": a.urgency if a else "",
        "sentiment": a.sentiment if a else "",
        "language": a.language if a else "",
        "risk_flags": ", ".join(a.risk_flags) if a and a.risk_flags else "none",
        "kb_sources": ", ".join(kb["sources"]) or "none",
        "kb_best_score": f"{kb['best_score']:.2f}",
        "coverage": coverage,
        "confidence": f"{confidence:.2f}",
        "grounding": grounding,
        # Never send a reply that failed the grounding check
        "response": draft if check["grounded"] and draft else SAFE_REPLY,
        "status": status,
        "escalation_reason": "; ".join(reasons),
        "processing_seconds": f"{time.perf_counter() - ctx.started_at:.1f}",
    }


def make_log_tool(ctx: TicketContext):
    """Create the log_ticket tool, connected to this ticket's context."""

    async def log_ticket(extra_concern: str = "") -> str:
        """Save the current support ticket to the log file and decide the escalation
        status using the company rules. extra_concern: a short reason why a human
        should review this case that the standard rules might miss, or an empty
        string if there is none."""
        # In a team, the other agents' outputs are stored in ctx while the team runs.
        # Wait briefly (max 5 s) in case the last one has not been stored yet.
        for _ in range(50):
            if ctx.resolution is not None:
                break
            await asyncio.sleep(0.1)
        if ctx.result is not None:  # already logged: never log the same ticket twice
            return f"LOGGED {ctx.result['ticket_id']} | status={ctx.result['status']} | already logged"
        row = build_ticket_row(ctx, extra_concern)
        append_ticket(row)
        ctx.result = row
        reason = row["escalation_reason"] or "none"
        return f"LOGGED {row['ticket_id']} | status={row['status']} | reason={reason}"

    return log_ticket


SYSTEM_MESSAGE = """You are the Log Agent for BluePeak Store's support desk.
Your job is to record the ticket by calling the log_ticket tool exactly once.

The tool automatically applies the company escalation rules (high urgency, angry customer,
no matching document, legal threats, fraud, payment disputes, low confidence, failed checks).

Before calling it, review the conversation. Set extra_concern ONLY if a human should review
the case for a reason those rules could miss, for example: the customer says this is a
repeated complaint, mentions a health problem, a vulnerable person, or a threat to post on
social media. Keep it under 12 words. Do NOT repeat anything the rules already cover
(urgency, anger, lawyers, fraud, payments, hacking). Otherwise pass an empty string.

Do not write anything else and do not reply to the customer."""


def create_log_agent(model_client: ChatCompletionClient, ctx: TicketContext) -> AssistantAgent:
    """Build the Log Agent for one ticket."""
    return AssistantAgent(
        name="log_agent",
        description="Records every ticket to the log file and decides whether to escalate it to a human.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        tools=[make_log_tool(ctx)],
        reflect_on_tool_use=False,
    )
