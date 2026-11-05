"""
agents/resolution_agent.py - Agent 3 of 4: the Resolution Agent.

Job: write a clear, polite, customer-ready reply using ONLY the policy passages
the Knowledge Agent found. If nothing relevant was found, it says so honestly.

Output (structured, validated): the reply, a coverage label (full / partial /
minimal / none), the source files it used, and whether the knowledge base
actually answered the question.

Confidence: LLMs are bad at picking exact numbers like 0.87, but good at choosing
a label. So the LLM picks the coverage label, and compute_confidence() turns it
into a 0-1 confidence in plain Python, together with the search score and the
grounding check. Same inputs always give the same confidence.

Extra safety (grounding_check): after the LLM writes the reply, plain Python
checks that every number (days, fees, hours...) and every menu path
(e.g. "My Account > My Orders") in the reply really appears in the retrieved
passages, and that every cited source was really retrieved.
This catches invented policies even if the LLM makes a mistake.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from agents.kb_search import MIN_SCORE


class ResolutionResult(BaseModel):
    """The structured result of the Resolution Agent."""

    reply: str = Field(
        description="The complete customer-ready reply, in the customer's language."
    )
    coverage: Literal["full", "partial", "minimal", "none"] = Field(
        description="How completely the passages cover what the customer asked."
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
- Never invent steps, buttons, menu paths or website pages. Write menu paths exactly as they
  appear in the passages (for example "My Account > My Orders"). If the passages do not
  explain how to do something, do not make up the steps.
- Include the key limits from the passages that apply to this customer (time windows,
  conditions, timelines), so the customer knows exactly what to expect.
- If the passages say NO_MATCH or do not answer the question, say honestly that you do not
  have that information, and that the request has been passed to our support team who will
  get back to them. Do not guess. Set answered_from_kb to false and coverage to "none".

STYLE:
- The customer is already talking to BluePeak support. NEVER write "contact support",
  "contact our support team" or "reach out to us", even if a passage says so. We already
  have their message and order ID, so tell them what WE will do next instead.
- Greet the customer by name if a name is given, and mention the order ID if given.
- Apologise sincerely and specifically ONLY if the sentiment is Frustrated or Angry.
  If the sentiment is Neutral or Positive, thank them instead of apologising.
- Then give the answer. Use numbered steps when the customer has to do something.
- Close with one natural sentence about the next step (do not write a label like
  "What happens next:"), then sign off with "Warm regards," and "BluePeak Support Team"
  on separate lines.
- Keep it under 150 words. Plain text, no markdown headings, no file names in the reply.
- Write the reply in the customer's language (the "language" field of the analysis).

RISK CASES: if risk_flags contains legal_threat, fraud, payment_dispute or security,
do not promise any outcome or refund. Explain that a specialist team will handle the case,
using only what the passages say about it.

COVERAGE = how completely the PASSAGES cover what the customer asked
(not how happy the customer will be; escalation is decided separately):
- "full": the passages contain everything needed (the exact steps or the exact policy).
  Use this whenever "gaps" is empty.
- "partial": most of it is covered, one detail is missing (describe it in "gaps").
- "minimal": only a small part of the question is covered.
- "none": NO_MATCH, or the passages are about something else."""


def create_resolution_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the Resolution Agent."""
    return AssistantAgent(
        name="resolution_agent",
        description="Writes a polite, customer-ready reply grounded only in the retrieved policy passages.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        output_content_type=ResolutionResult,
    )


# ---------------- confidence (plain Python, explainable) ----------------
COVERAGE_SCORE = {"full": 0.95, "partial": 0.70, "minimal": 0.40, "none": 0.10}
STRONG_MATCH = 0.60   # a search score at or above this counts as a very strong match


def compute_confidence(coverage: str, kb_best_score: float, grounded: bool) -> float:
    """
    confidence = 70% coverage label + 30% search strength, halved if grounding failed.
    Search strength is 0 at the minimum match score and 1 at a very strong match.
    """
    strength = (kb_best_score - MIN_SCORE) / (STRONG_MATCH - MIN_SCORE)
    strength = min(max(strength, 0.0), 1.0)
    confidence = 0.7 * COVERAGE_SCORE.get(coverage, 0.1) + 0.3 * strength
    if not grounded:
        confidence *= 0.5
    return round(confidence, 2)


# ---------------- grounding check ----------------
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)   # "1. ", "2) " at line start
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# A UI path such as "My Account > My Orders > Track Order" (Title Case words joined by ">")
_TITLE = r"[A-Z][\w-]*(?: [A-Z][\w-]*)*"
_UI_PATH = re.compile(rf"{_TITLE}(?:\s*>\s*{_TITLE})+")


def _numbers(text: str) -> set[str]:
    text = _LIST_MARKER.sub(" ", text or "")          # list numbering is not a fact
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def _norm(text: str) -> str:
    return re.sub(r"\s*>\s*", " > ", re.sub(r"\s+", " ", text or "")).lower()


def grounding_check(result: ResolutionResult, kb_text: str, kb_sources: list[str],
                    complaint: str = "", order_id: str = "") -> dict:
    """
    Verify the reply in plain Python (no AI):
    - every number in the reply appears in the passages, complaint or order ID
    - every menu path (e.g. "My Account > My Orders") appears in the passages
    - every cited source file was really returned by the Knowledge Agent
    """
    allowed = _numbers(kb_text) | _numbers(complaint) | _numbers(order_id)
    ungrounded = sorted(n for n in _numbers(result.reply) if n not in allowed)
    kb_norm = _norm(kb_text)
    bad_paths = sorted({p for p in _UI_PATH.findall(result.reply) if _norm(p) not in kb_norm})
    invalid_sources = sorted(s for s in result.sources_used if s not in kb_sources)
    return {
        "grounded": not ungrounded and not bad_paths and not invalid_sources,
        "ungrounded_numbers": ungrounded,
        "ungrounded_paths": bad_paths,
        "invalid_sources": invalid_sources,
    }
