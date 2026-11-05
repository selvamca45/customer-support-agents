"""
tests/try_team.py - run the full AutoGen team on every test complaint.

This is the end-to-end test: one team, four agents in order, termination on
LOGGED, one log row per ticket, and the expected Resolved / Escalated status.

Run from the project folder:  python tests/try_team.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_model_client  # noqa: E402
from team import run_ticket  # noqa: E402
from tests.test_cases import TEST_CASES  # noqa: E402


async def main() -> None:
    model_client = get_model_client()
    rows, passed = [], 0

    for case in TEST_CASES:
        ctx = await run_ticket(case["complaint"], case["customer_name"], case["order_id"],
                               model_client=model_client)
        r = ctx.result
        ok = r["status"] == case["expected_status"]
        passed += ok
        order = " -> ".join(s.agent for s in ctx.steps)
        print(f"\n#{case['id']} [{'PASS' if ok else 'CHECK'}] {case['complaint'][:65]}")
        print(f"   flow: {order}")
        print(f"   stop: {ctx.stop_reason or ctx.error}")
        print(f"   {r['ticket_id']} | {r['category']} | {r['urgency']} | {r['sentiment']} | "
              f"sources: {r['kb_sources']} | confidence {r['confidence']} | grounding {r['grounding']}")
        print(f"   status: {r['status']} (expected {case['expected_status']})"
              f"{' | reason: ' + r['escalation_reason'] if r['escalation_reason'] else ''}")
        rows.append(float(r["processing_seconds"]))

    print(f"\nResult: {passed}/{len(TEST_CASES)} tickets got the expected status.")
    print(f"Average time per ticket: {sum(rows) / len(rows):.1f}s")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
