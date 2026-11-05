"""
agents/resolution_agent.py - Agent 3 of 4: the Resolution Agent.

Job: write a clear, polite, customer-ready reply using ONLY the policy passages
the Knowledge Agent found. If nothing relevant was found, it says so honestly.

Output (structured, validated): the reply, a confidence score (0-1), the source
files it used, and whether the knowledge base actually answered the question.

Extra safety (grounding_check): after the LLM writes the reply, plain Python
checks that every number in the reply (days, fees, hours...) really appears in
the retrieved passages, and that every cited source was really retrieved.
This catches invented policies even if the LLM makes a mistake.
"""
import re

from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient


class ResolutionResult(BaseModel):
    """The structured result of the Resolution Agent."""

    reply: str = Field(
        description="The complete customer-ready reply, in the customer's language."
    )
    confidence: float = Field(
        description="0.0 to 1.0: how fully the retrieved passages answer the customer."
    )
    sources_used: list[str] = Field(
        description="File names of the passages the reply is based on, e.g. ['refund_policy.md']. Empty if none."
    )
    answered_from_kb: bool = Field(
        description="True if the passages answered the customer's question, False otherwise."
    )
    gaps: str = Field(
        description="What the customer asked that the passages did NOT cover. Empty string if nothing."
    )


SYSTEM_MESSAGE = """You are the Resolution Agent for BluePeak Store, an online store in India.
You write the final reply to the customer.

You receive: the customer's complaint, the query_agent's analysis (category, urgency,
sentiment, risk_flags, language) and the knowledge_agent's policy passages.

GROUNDING RULES (most important):
- Use ONLY facts that appear in the knowledge_agent's passages. Never invent timelines,
  fees, amounts, links, phone numbers, email addresses or policies.
- Every number you write (days, hours, rupees, percentages) must come from the passages.
- If the passages say NO_MATCH or do not answer the question, say honestly that you do not
  have that information, and that the request has been passed to our support team who will
  get back to them. Do not guess. Set answered_from_kb to false and confidence below 0.3.

STYLE:
- Greet the customer by name if a name is given, and mention the order ID if given.
- Match the emotion: if the customer is Angry or Frustrated, start with a sincere,
  specific apology. Never be defensive.
- Then give the answer. Use numbered steps when the customer has to do something.
- End with what happens next.
- Keep it under 150 words. Plain text, no markdown headings, no file names in the reply.
- Write the reply in the customer's language (the "language" field of the analysis).

RISK CASES: if risk_flags contains legal_threat, fraud, payment_dispute or security,
do not promise any outcome or refund. Explain that a specialist team will handle the case,
using only what the passages say about it.

CONFIDENCE:
- 0.9-1.0: the passages fully and directly answer the question.
- 0.6-0.8: the passages answer most of it.
- 0.3-0.5: the passages only partly help.
- below 0.3: the passages do not answer it (or NO_MATCH)."""


def create_resolution_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the Resolution Agent."""
    return AssistantAgent(
        name="resolution_agent",
        description="Writes a polite, customer-ready reply grounded only in the retrieved policy passages.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        output_content_type=ResolutionResult,
    )


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text or "")}


def grounding_check(result: ResolutionResult, kb_text: str, kb_sources: list[str],
                    complaint: str = "", order_id: str = "") -> dict:
    """
    Verify the reply in plain Python (no AI):
    - every number in the reply appears in the passages, complaint or order ID
    - every cited source file was really returned by the Knowledge Agent
    """
    allowed = _numbers(kb_text) | _numbers(complaint) | _numbers(order_id)
    ungrounded = sorted(n for n in _numbers(result.reply) if n not in allowed)
    invalid_sources = sorted(s for s in result.sources_used if s not in kb_sources)
    return {
        "grounded": not ungrounded and not invalid_sources,
        "ungrounded_numbers": ungrounded,
        "invalid_sources": invalid_sources,
    }
