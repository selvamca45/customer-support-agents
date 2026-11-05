"""
tests/try_knowledge_agent.py - test Query Agent -> Knowledge Agent on all cases.

For every test complaint it shows: the search the Knowledge Agent made,
which files it found, the best score, and whether that is the expected file.

Run from the project folder:  python tests/try_knowledge_agent.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autogen_agentchat.messages import ToolCallRequestEvent  # noqa: E402

from config import get_model_client  # noqa: E402
from agents.kb_search import get_kb, parse_kb_output  # noqa: E402
from agents.knowledge_agent import create_knowledge_agent  # noqa: E402
from agents.query_agent import create_query_agent  # noqa: E402
from tests.test_cases import TEST_CASES  # noqa: E402


async def main() -> None:
    kb = get_kb()
    files = len({c.source for c in kb.chunks})
    print(f"Knowledge base: {len(kb.chunks)} passages from {files} files")
    print(f"Search mode: {kb.mode} | minimum score: {kb.min_score:.2f}\n")

    model_client = get_model_client(temperature=0)
    passed, calibration = 0, []

    for case in TEST_CASES:
        query_agent = create_query_agent(model_client)
        analysis = (await query_agent.run(task=case["complaint"])).messages[-1].content

        knowledge_agent = create_knowledge_agent(model_client)
        result = await knowledge_agent.run(
            task=f"Analysis from query_agent:\n{analysis.model_dump_json()}"
        )

        tool_query = "(the agent did not call the tool)"
        for msg in result.messages:
            if isinstance(msg, ToolCallRequestEvent) and msg.content:
                tool_query = json.loads(msg.content[0].arguments).get("query", "")
        info = parse_kb_output(result.messages[-1].content)

        expected = case.get("expected_source")
        if expected is None:
            ok = info["no_match"]
            expected_text = "NO_MATCH"
        else:
            ok = expected in info["sources"]
            expected_text = expected
        passed += ok
        calibration.append((case["id"], expected is None, info["best_score"]))

        found = "NO_MATCH" if info["no_match"] else ", ".join(info["sources"])
        print(f"#{case['id']} [{'PASS' if ok else 'CHECK'}] {case['complaint'][:60]}")
        print(f"   search: {tool_query}")
        print(f"   found:  {found} (best score {info['best_score']:.2f}) | expected: {expected_text}\n")

    print(f"Result: {passed}/{len(TEST_CASES)} test cases passed.\n")

    # Calibration help: the threshold must sit between these two numbers
    should_match = [s for _, is_none, s in calibration if not is_none]
    should_not = [s for _, is_none, s in calibration if is_none]
    if should_match and should_not:
        print("Threshold check:")
        print(f"   lowest score among cases that SHOULD match:     {min(should_match):.2f}")
        print(f"   highest score among cases that should NOT match: {max(should_not):.2f}")
        print(f"   current minimum score:                          {kb.min_score:.2f}")

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
