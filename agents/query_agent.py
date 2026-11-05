"""
agents/query_agent.py - Agent 1 of 4: the Customer Query Agent.

Job: read the raw complaint and UNDERSTAND it. It never answers the customer.
Output: a strict, validated JSON object (QueryAnalysis) with category, urgency,
sentiment, a one-line summary, risk flags, language and a search query.

Why structured output?
The next agents (and our escalation rules) need reliable fields, not free text.
`output_content_type=QueryAnalysis` makes OpenAI return JSON that matches this
Pydantic model exactly, so the result is always valid and easy to use in Python.
"""
from typing import Literal

from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient


class QueryAnalysis(BaseModel):
    """The structured result of analysing one customer complaint."""

    category: Literal["Shipping", "Refund", "Account", "Product", "Billing", "Other"] = Field(
        description="The main topic of the complaint."
    )
    urgency: Literal["Low", "Medium", "High"] = Field(
        description="How quickly the case needs attention, using the urgency rules."
    )
    sentiment: Literal["Positive", "Neutral", "Frustrated", "Angry"] = Field(
        description="How the customer feels."
    )
    summary: str = Field(
        description="One sentence (max 25 words) describing the problem, in English, third person."
    )
    risk_flags: list[Literal["legal_threat", "fraud", "payment_dispute", "security", "safety"]] = Field(
        description="Serious risks mentioned in the complaint. Empty list if none."
    )
    language: str = Field(
        description="The language the customer wrote in, e.g. English, Tamil, Hindi."
    )
    search_query: str = Field(
        description="3-8 English keywords to search the company policy documents."
    )


SYSTEM_MESSAGE = """You are the Customer Query Agent for BluePeak Store, an online store in India.
Your ONLY job is to analyse the customer's complaint. You never reply to the customer.

CATEGORY (pick one):
- Shipping: delivery, tracking, late or missing orders, address changes.
- Refund: returns, exchanges, refund status.
- Account: login, password, OTP, locked or hacked accounts, profile changes.
- Product: broken, damaged or defective items, warranty, how a product works, sizes.
- Billing: payments, charges, double charges, invoices, coupons, EMI.
- Other: anything not covered above.
If a complaint covers two topics, choose the one the customer most needs solved.

URGENCY:
- High: money charged wrongly, fraud, hacked account, legal threats, safety hazards,
  a very angry customer, or a problem ongoing for more than 7 days.
- Medium: late delivery, damaged or defective item, cannot log in, refund not received.
- Low: general questions, how-to questions, information requests.

SENTIMENT:
- Angry: threats, insults, ALL CAPS, many exclamation marks, demands like "fix this now".
- Frustrated: clearly unhappy or disappointed, but calm.
- Neutral: factual question or request.
- Positive: friendly or thankful.

RISK FLAGS (include every one that applies, otherwise an empty list):
- legal_threat: mentions a lawyer, court, consumer forum, police or legal action.
- fraud: unauthorised orders or charges, scams, identity theft.
- payment_dispute: chargeback, contacting the bank, disputed or multiple charges.
- security: hacked account, someone else accessing the account.
- safety: injury, fire, smoke, electric shock.

SEARCH QUERY: always write 3-8 English keywords that would find the right policy,
even if the customer wrote in another language. Example: "late delivery order tracking delay".

Be accurate and consistent. Do not invent facts that are not in the complaint."""


def create_query_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the Customer Query Agent."""
    return AssistantAgent(
        name="query_agent",
        description="Understands the complaint: category, urgency, sentiment, risks and a one-line summary.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        output_content_type=QueryAnalysis,
    )
