"""
tests/try_log_agent.py - Step 7 test.

Part A: checks the escalation rules on their own (no AI, no cost, instant).
Part B: runs all four agents on every test complaint, logs each ticket,
        and compares Resolved / Escalated with the expected result.

Run from the project folder:  python tests/try_log_agent.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_model_client  # noqa: E402
from agents.escalation import decide_escalation  # noqa: E402
from agents.knowledge_agent import create_knowledge_agent  # noqa: E402
from agents.log_agent import TicketContext, create_log_agent  # noqa: E402
from agents.query_agent import create_query_agent  # noqa: E402
from agents.resolution_agent import create_resolution_agent  # noqa: E402
from agents.ticket_logger import CSV_PATH, XLSX_PATH, read_tickets  # noqa: E402
from tests.test_cases import TEST_CASES  # noqa: E402


def part_a_rules() -> None:
    print("PART A - escalation rules (no AI)")
    calm = dict(urgency="Low", sentiment="Neutral", risk_flags=[], kb_no_match=False,
                confidence=0.9, grounded=True)
    checks = [
        ("calm, answered case", calm, "Resolved"),
        ("high urgency", {**calm, "urgency": "High"}, "Escalated"),
        ("angry customer", {**calm, "sentiment": "Angry"}, "Escalated"),
        ("legal threat", {**calm, "risk_flags": ["legal_threat"]}, "Escalated"),
        ("no document found", {**calm, "kb_no_match": True}, "Escalated"),
        ("low confidence", {**calm, "confidence": 0.3}, "Escalated"),
        ("failed grounding", {**calm, "grounded": False}, "Escalated"),
        ("log agent concern", {**calm, "extra_concern": "third complaint"}, "Escalated"),
    ]
    ok = 0
    for name, kwargs, expected in checks:
        status, reasons = decide_escalation(**kwargs)
        ok += status == expected
        print(f"   [{'PASS' if status == expected else 'FAIL'}] {name:20} -> {status:9} {reasons}")
    print(f"   {ok}/{len(checks)} rule checks passed.\n")


async def run_ticket(case: dict, model_client) -> dict:
    """Run the four agents in order on one complaint (Step 8 turns this into a team)."""
    ctx = TicketContext(case["complaint"], case["customer_name"], case["order_id"])

    ctx.analysis = (await create_query_agent(model_client).run(
        task=case["complaint"])).messages[-1].content

    ctx.kb_text = (await create_knowledge_agent(model_client).run(
        task=f"Analysis from query_agent:\n{ctx.analysis.model_dump_json()}")).messages[-1].content

    ctx.resolution = (await create_resolution_agent(model_client).run(task=(
        f"Customer complaint: {ctx.complaint}\n"
        f"Customer name: {ctx.customer_name or 'not given'}\n"
        f"Order ID: {ctx.order_id or 'not given'}\n\n"
        f"query_agent analysis:\n{ctx.analysis.model_dump_json()}\n\n"
        f"knowledge_agent passages:\n{ctx.kb_text}"
    ))).messages[-1].content

    log_result = await create_log_agent(model_client, ctx).run(task=(
        f"Customer complaint: {ctx.complaint}\n\n"
        f"query_agent analysis:\n{ctx.analysis.model_dump_json()}\n\n"
        f"resolution_agent reply:\n{ctx.resolution.reply}\n\n"
        "Log this ticket now."
    ))
    print(f"   log_agent said: {log_result.messages[-1].content}")
    return ctx.result


async def part_b_full_run() -> None:
    print("PART B - all four agents on every test complaint")
    model_client = get_model_client()
    passed = 0
    for case in TEST_CASES:
        print(f"\n#{case['id']} {case['complaint'][:70]}")
        row = await run_ticket(case, model_client)
        if row is None:
            print("   [CHECK] the Log Agent did not call log_ticket")
            continue
        ok = row["status"] == case["expected_status"]
        passed += ok
        print(f"   [{'PASS' if ok else 'CHECK'}] {row['ticket_id']} | {row['category']} | "
              f"{row['urgency']} | {row['sentiment']} | confidence={row['confidence']} | "
              f"grounding={row['grounding']}")
        print(f"   status={row['status']} (expected {case['expected_status']}) | "
              f"reason: {row['escalation_reason'] or '-'} | {row['processing_seconds']}s")
    print(f"\nResult: {passed}/{len(TEST_CASES)} tickets got the expected status.")
    print(f"Log files: {CSV_PATH}\n           {XLSX_PATH}")
    df = read_tickets()
    print(f"\nLast {min(len(df), len(TEST_CASES))} rows of the log:")
    print(df[["ticket_id", "category", "urgency", "sentiment", "kb_sources",
              "confidence", "status", "escalation_reason"]].tail(len(TEST_CASES)).to_string(index=False))
    await model_client.close()


if __name__ == "__main__":
    part_a_rules()
    asyncio.run(part_b_full_run())
