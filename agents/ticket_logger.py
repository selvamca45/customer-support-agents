"""
agents/ticket_logger.py - saves every ticket as one row in logs/tickets.csv
and keeps an Excel copy (logs/tickets.xlsx) in sync.

Escalated tickets start with review_status "Pending review". A support person
can approve (and edit) the reply in the app's Human Review tab, which updates
the same row: human-in-the-loop.
"""
import csv

import pandas as pd

from config import LOG_DIR

CSV_PATH = LOG_DIR / "tickets.csv"
XLSX_PATH = LOG_DIR / "tickets.xlsx"

COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint",
    "category", "urgency", "sentiment", "language", "risk_flags",
    "kb_sources", "kb_best_score", "coverage", "confidence", "grounding",
    "response", "status", "escalation_reason", "processing_seconds",
    # human-in-the-loop
    "review_status", "reviewed_reply", "reviewer_note", "reviewed_at",
]


def next_ticket_id() -> str:
    """TKT-0001, TKT-0002, ... based on the highest ID already in the log."""
    if not CSV_PATH.exists():
        return "TKT-0001"
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        numbers = [
            int(row["ticket_id"].split("-")[1])
            for row in csv.DictReader(f)
            if row.get("ticket_id", "").startswith("TKT-")
            and row["ticket_id"].split("-")[1].isdigit()
        ]
    return f"TKT-{(max(numbers) if numbers else 0) + 1:04d}"


def read_tickets() -> pd.DataFrame:
    """The whole log as a table, always with all COLUMNS (empty if no tickets yet)."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df.reindex(columns=COLUMNS, fill_value="")
    # Tickets logged before the review step existed: give them a review status too
    missing = df["review_status"] == ""
    df.loc[missing, "review_status"] = df.loc[missing, "status"].map(
        lambda s: "Pending review" if s == "Escalated" else "Not needed")
    return df


def _write_all(df: pd.DataFrame) -> None:
    """Rewrite the CSV (utf-8-sig so Excel shows Tamil/Hindi correctly) and the Excel copy."""
    LOG_DIR.mkdir(exist_ok=True)
    df = df.reindex(columns=COLUMNS, fill_value="")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(XLSX_PATH, index=False)
    except Exception as exc:  # e.g. the Excel file is open in Excel right now
        print(f"[ticket log] Excel copy not updated ({exc}). The CSV log is complete.")


def append_ticket(row: dict) -> None:
    """Add one ticket to the log."""
    defaults = {
        "review_status": "Pending review" if row.get("status") == "Escalated" else "Not needed",
        "reviewed_reply": "", "reviewer_note": "", "reviewed_at": "",
    }
    df = read_tickets()
    df = pd.concat([df, pd.DataFrame([{**defaults, **row}])], ignore_index=True)
    _write_all(df)


def update_ticket(ticket_id: str, updates: dict) -> None:
    """Change fields of one existing ticket (used by the human review step)."""
    df = read_tickets()
    mask = df["ticket_id"] == ticket_id
    for key, value in updates.items():
        if key in COLUMNS:
            df.loc[mask, key] = value
    _write_all(df)
