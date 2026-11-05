"""
tests/try_query_agent.py - run the Query Agent on all test complaints.

Run from the project folder:  python tests/try_query_agent.py
"""
import asyncio
import sys
from pathlib import Path

# Let Python find config.py and the agents/ folder in the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_model_client  # noqa: E402
from agents.query_agent import QueryAnalysis, create_query_agent  # noqa: E402
from tests.test_cases import TEST_CASES  # noqa: E402


def check(case: dict, a: QueryAnalysis) -> list[str]:
    """Compare the agent's analysis with what we expect. Returns a list of problems."""
    problems = []
    if a.category not in case["expected_category"]:
        problems.append(f"category {a.category}, expected {sorted(case['expected_category'])}")
    if "expected_urgency" in case and a.urgency != case["expected_urgency"]:
        problems.append(f"urgency {a.urgency}, expected {case['expected_urgency']}")
    if "expected_sentiment" in case and a.sentiment != case["expected_sentiment"]:
        problems.append(f"sentiment {a.sentiment}, expected {case['expected_sentiment']}")
    if "expected_language" in case and case["expected_language"].lower() not in a.language.lower():
        problems.append(f"language {a.language}, expected {case['expected_language']}")
    return problems


async def main() -> None:
    model_client = get_model_client(temperature=0)
    passed = 0

    for case in TEST_CASES:
        # A fresh agent per complaint, so one case cannot influence the next
        agent = create_query_agent(model_client)
        result = await agent.run(task=case["complaint"])
        analysis: QueryAnalysis = result.messages[-1].content

        problems = check(case, analysis)
        status = "PASS" if not problems else "CHECK"
        passed += not problems

        print(f"\n#{case['id']} [{status}] {case['complaint'][:70]}")
        print(f"   category={analysis.category} | urgency={analysis.urgency} | "
              f"sentiment={analysis.sentiment} | language={analysis.language}")
        print(f"   risk_flags={analysis.risk_flags}")
        print(f"   summary: {analysis.summary}")
        print(f"   search_query: {analysis.search_query}")
        for p in problems:
            print(f"   -> {p}")

    print(f"\nResult: {passed}/{len(TEST_CASES)} test cases passed.")
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
