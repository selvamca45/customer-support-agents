"""
agents/escalation.py - the escalation rules, in plain Python.

Why plain Python and not the LLM?
Escalation is a business rule. It must be predictable, testable and explainable:
the same ticket must always get the same decision, with a clear written reason.
The Log Agent (an LLM) can ADD a concern on top, but it can never remove a rule.

Rules from the assignment (Task 5), plus one of our own (grounding check):
  1. urgency is High
  2. the customer is very angry
  3. the Knowledge Agent found no matching document
  4. legal threats, fraud or payment disputes (plus security and safety risks)
  5. the Resolution Agent's confidence is low
  6. the reply failed the grounding check (possible invented facts)
  7. the Log Agent raised an extra concern
  8. the customer's photo does not match the complaint (checked by AI)
"""

LOW_CONFIDENCE = 0.5

RISK_REASONS = {
    "legal_threat": "Legal threat",
    "fraud": "Possible fraud",
    "payment_dispute": "Payment dispute",
    "security": "Account security risk",
    "safety": "Safety hazard",
}


def decide_escalation(
    urgency: str,
    sentiment: str,
    risk_flags: list[str],
    kb_no_match: bool,
    confidence: float,
    grounded: bool,
    extra_concern: str = "",
    photo_mismatch: bool = False,
) -> tuple[str, list[str]]:
    """Return ("Resolved" or "Escalated", list of reasons)."""
    reasons: list[str] = []
    if urgency == "High":
        reasons.append("High urgency")
    if sentiment == "Angry":
        reasons.append("Very angry customer")
    for flag in risk_flags:
        if flag in RISK_REASONS:
            reasons.append(RISK_REASONS[flag])
    if kb_no_match:
        reasons.append("No matching document in knowledge base")
    if confidence < LOW_CONFIDENCE:
        reasons.append(f"Low confidence ({confidence:.2f})")
    if not grounded:
        reasons.append("Reply failed grounding check")
    if extra_concern and extra_concern.strip():
        reasons.append(f"Log agent: {extra_concern.strip()}")
    if photo_mismatch:
        reasons.append("Photo does not match the complaint")
    return ("Escalated" if reasons else "Resolved"), reasons
