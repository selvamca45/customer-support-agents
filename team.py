"""
team.py - connects the four agents into ONE AutoGen team (Task 4).

    Query Agent -> Knowledge Agent -> Resolution Agent -> Log Agent

- RoundRobinGroupChat makes the agents speak in exactly that order, and every
  agent sees the full conversation so far (that is how work is passed along).
- The team stops when the Log Agent's tool answers "LOGGED ..."
  (TextMentionTermination), with a MaxMessageTermination safety net in case
  something goes wrong.
- run_stream() gives us each agent's output the moment it is produced, so the
  Streamlit app can show the agents working live.
- Safety net: whatever happens (an API error, an agent forgetting its tool),
  every ticket is still logged, and such tickets are escalated to a human.

Try it from the terminal:
    python team.py "I forgot my password and can't log in."
"""
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Callable

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import (
    StructuredMessage,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient

from config import get_model_client
from agents.kb_search import parse_kb_output
from agents.knowledge_agent import create_knowledge_agent
from agents.log_agent import TicketContext, build_ticket_row, create_log_agent
from agents.query_agent import QueryAnalysis, create_query_agent
from agents.resolution_agent import ResolutionResult, create_resolution_agent
from agents.ticket_logger import append_ticket

MAX_MESSAGES = 10  # task + 4 agents = 5 messages normally; 10 is a safe upper limit


@dataclass
class Step:
    """One visible step of the workflow (shown in the terminal and in the UI)."""
    agent: str
    title: str
    detail: str
    seconds: float  # time since the ticket started


def build_team(model_client: ChatCompletionClient, ctx: TicketContext) -> RoundRobinGroupChat:
    """Create the four agents and put them in a fixed-order team."""
    termination = TextMentionTermination("LOGGED") | MaxMessageTermination(MAX_MESSAGES)
    return RoundRobinGroupChat(
        participants=[
            create_query_agent(model_client),
            create_knowledge_agent(model_client),
            create_resolution_agent(model_client),
            create_log_agent(model_client, ctx),
        ],
        termination_condition=termination,
    )


def build_task(complaint: str, customer_name: str = "", order_id: str = "") -> str:
    """The first message of the conversation: the new ticket."""
    return (
        "New support ticket.\n"
        f"Customer name: {customer_name or 'not given'}\n"
        f"Order ID: {order_id or 'not given'}\n"
        f"Complaint: {complaint}"
    )


def to_step(item, ctx: TicketContext) -> Step | None:
    """Turn a message from the team into a readable step (or None to skip it)."""
    source = getattr(item, "source", "")
    seconds = time.perf_counter() - ctx.started_at

    if isinstance(item, StructuredMessage) and isinstance(item.content, QueryAnalysis):
        a = item.content
        detail = (f"Category: {a.category} | Urgency: {a.urgency} | Sentiment: {a.sentiment} | "
                  f"Language: {a.language}\nSummary: {a.summary}\n"
                  f"Risk flags: {', '.join(a.risk_flags) or 'none'}")
        return Step(source, "Analysed the complaint", detail, seconds)

    if isinstance(item, ToolCallRequestEvent) and item.content:
        call = item.content[0]
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            args = {"raw": call.arguments}
        shown = ", ".join(f'{k}="{v}"' for k, v in args.items())
        return Step(source, f"Called tool {call.name}", f"{call.name}({shown})", seconds)

    if isinstance(item, ToolCallSummaryMessage) and source == "knowledge_agent":
        info = parse_kb_output(item.content)
        title = ("No matching document found" if info["no_match"]
                 else f"Found passages in {', '.join(info['sources'])}")
        return Step(source, title, item.content, seconds)

    if isinstance(item, StructuredMessage) and isinstance(item.content, ResolutionResult):
        r = item.content
        detail = f"Coverage: {r.coverage} | Sources used: {', '.join(r.sources_used) or 'none'}\n\n{r.reply}"
        return Step(source, "Wrote the customer reply", detail, seconds)

    if isinstance(item, ToolCallSummaryMessage) and source == "log_agent":
        return Step(source, "Logged the ticket", item.content, seconds)

    return None


async def run_ticket(
    complaint: str,
    customer_name: str = "",
    order_id: str = "",
    model_client: ChatCompletionClient | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> TicketContext:
    """
    Run the whole team on one complaint. Returns the TicketContext, whose
    .result is the logged row (status, reply, reasons...) and .steps the timeline.
    on_step(step) is called live for each step (used by the Streamlit app).
    """
    own_client = model_client is None
    model_client = model_client or get_model_client()
    ctx = TicketContext(complaint, customer_name, order_id)
    team = build_team(model_client, ctx)

    try:
        async for item in team.run_stream(task=build_task(complaint, customer_name, order_id)):
            if isinstance(item, TaskResult):
                ctx.stop_reason = item.stop_reason or ""
                continue
            # Store each agent's output so the Log Agent's tool can read it
            if isinstance(item, StructuredMessage) and isinstance(item.content, QueryAnalysis):
                ctx.analysis = item.content
            elif isinstance(item, ToolCallSummaryMessage) and item.source == "knowledge_agent":
                ctx.kb_text = item.content
            elif isinstance(item, StructuredMessage) and isinstance(item.content, ResolutionResult):
                ctx.resolution = item.content

            step = to_step(item, ctx)
            if step:
                ctx.steps.append(step)
                if on_step:
                    on_step(step)
    except Exception as exc:  # API down, no credit, network problem...
        ctx.error = f"{type(exc).__name__}: {exc}"

    if ctx.result is None:
        # Safety net: the team did not finish normally, but the ticket must still be logged
        concern = (f"System error ({ctx.error[:60]})" if ctx.error
                   else "Log agent did not complete the ticket")
        row = build_ticket_row(ctx, extra_concern=concern)
        append_ticket(row)
        ctx.result = row
        step = Step("system", "Logged the ticket (safety net)",
                    f"LOGGED {row['ticket_id']} | status={row['status']} | reason={row['escalation_reason']}",
                    time.perf_counter() - ctx.started_at)
        ctx.steps.append(step)
        if on_step:
            on_step(step)

    if own_client:
        await model_client.close()
    return ctx


def print_step(step: Step) -> None:
    print(f"\n[{step.seconds:5.1f}s] {step.agent} - {step.title}")
    for line in step.detail.splitlines()[:12]:
        print(f"         {line}")


async def _main() -> None:
    complaint = " ".join(sys.argv[1:]) or "Where is my order? It was supposed to arrive yesterday."
    print(f"Complaint: {complaint}")
    ctx = await run_ticket(complaint, on_step=print_step)
    r = ctx.result
    print(f"\n{'=' * 70}\n{r['ticket_id']}  STATUS: {r['status'].upper()}  "
          f"(confidence {r['confidence']}, {r['processing_seconds']}s)")
    if r["escalation_reason"]:
        print(f"Escalation reason: {r['escalation_reason']}")
    print(f"Stop reason: {ctx.stop_reason or ctx.error}")


if __name__ == "__main__":
    asyncio.run(_main())
