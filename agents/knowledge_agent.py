"""
agents/knowledge_agent.py - Agent 2 of 4: the Knowledge Base Agent.

Job: find the company policy passages that answer the customer's problem.
It uses ONE tool, search_kb(query), and returns the passages exactly as found,
with the source file name of each one. It never writes the reply itself.

Design choice: reflect_on_tool_use=False
The agent's final message is the tool's raw output (the real policy text), not a
summary written by the LLM. That way no policy can be changed or invented on its
way to the Resolution Agent: the passages are copied word for word.
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from agents.kb_search import format_results, get_kb


def search_kb(query: str) -> str:
    """Search BluePeak Store's policy documents (shipping, refunds and returns,
    account and login, product FAQ and warranty, billing and payments, contact and
    escalation). Returns the most relevant passages with their source file names,
    or NO_MATCH if no document covers the request."""
    results, best = get_kb().search(query)
    return format_results(results, best)


SYSTEM_MESSAGE = """You are the Knowledge Base Agent for BluePeak Store.
Your ONLY job is to find the company policy passages relevant to the customer's issue.

1. Read the analysis written by query_agent (JSON with category, summary and search_query).
2. Call the search_kb tool exactly once. Use the search_query from the analysis as the query.
   You may add the category name if it makes the query clearer.
3. Do not answer the customer, do not summarise, and do not add any information yourself.
   The tool's output is handed to the Resolution Agent exactly as it is."""


def create_knowledge_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the Knowledge Base Agent."""
    return AssistantAgent(
        name="knowledge_agent",
        description="Searches the company documents and returns matching passages with their source file names.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        tools=[search_kb],
        reflect_on_tool_use=False,
    )
