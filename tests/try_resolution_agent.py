"""
tests/try_resolution_agent.py - Query -> Knowledge -> Resolution on all test cases.

Shows the customer reply for every complaint, its confidence, the sources used,
and the result of the Python grounding check.

Run from the project folder:  python tests/try_resolution_agent.py
"""
import asyncio
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_model_client  # noqa: E402
from agents.kb_search import parse_kb_output  # noqa: E402
from agents.knowledge_agent import create_knowledge_agent  # noqa: E402
from agents.query_agent import create_query_agent  # noqa: E402
from agents.resolution_agent import (  # noqa: E402
    ResolutionResult,
    create_resolution_agent,
    grounding_check,
)
from tests.test_cases import TEST_CASES  # noqa: E402


async def main() -> None:
    model_client = get_model_client()
    passed = 0

    for case in TEST_CASES:
        # Agent 1: understand the complaint
        analysis = (await create_query_agent(model_client).run(
            task=case["complaint"])).messages[-1].content

        # Agent 2: find the policy passages
        kb_text = (await create_knowledge_agent(model_client).run(
            task=f"Analysis from query_agent:\n{analysis.model_dump_json()}"
        )).messages[-1].content
        kb_info = parse_kb_output(kb_text)

        # Agent 3: write the reply from those passages only
        task = (
            f"Customer complaint: {case['complaint']}\n"
            f"Customer name: {case['customer_name'] or 'not given'}\n"
            f"Order ID: {case['order_id'] or 'not given'}\n\n"
            f"query_agent analysis:\n{analysis.model_dump_json()}\n\n"
            f"knowledge_agent passages:\n{kb_text}"
        )
        result: ResolutionResult = (await create_resolution_agent(model_client).run(
            task=task)).messages[-1].content

        check = grounding_check(result, kb_text, kb_info["sources"],
                                case["complaint"], case["order_id"])

        # What we expect: honest "don't know" for NO_MATCH, grounded answer otherwise
        if case.get("expected_source") is None:
            ok = not result.answered_from_kb and result.confidence < 0.6
        else:
            ok = check["grounded"] and bool(result.sources_used)
        passed += ok

        print(f"\n{'=' * 78}")
        print(f"#{case['id']} [{'PASS' if ok else 'CHECK'}] {case['complaint']}")
        print(f"   category={analysis.category} | urgency={analysis.urgency} | "
              f"sentiment={analysis.sentiment} | language={analysis.language}")
        print(f"   kb sources: {', '.join(kb_info['sources']) or 'NO_MATCH'}")
        print(f"   confidence={result.confidence:.2f} | answered_from_kb={result.answered_from_kb} | "
              f"sources_used={result.sources_used}")
        print(f"   grounding: {'OK' if check['grounded'] else 'PROBLEM'} "
              f"(ungrounded numbers: {check['ungrounded_numbers'] or 'none'}, "
              f"invalid sources: {check['invalid_sources'] or 'none'})")
        if result.gaps:
            print(f"   gaps: {result.gaps}")
        print("   --- reply ---")
        for para in result.reply.split("\n"):
            print(textwrap.fill(para, width=76, initial_indent="   ", subsequent_indent="   ")
                  if para.strip() else "")

    print(f"\n{'=' * 78}\nResult: {passed}/{len(TEST_CASES)} test cases passed.")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
