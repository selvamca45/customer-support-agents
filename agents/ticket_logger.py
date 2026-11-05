"""
agents/ticket_logger.py - saves every ticket as one row in logs/tickets.csv
and keeps an Excel copy (logs/tickets.xlsx) in sync.
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


def append_ticket(row: dict) -> None:
    """Add one ticket to the CSV log, then refresh the Excel copy."""
    LOG_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        # utf-8-sig adds a marker so Excel shows Tamil/Hindi text correctly
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore").writerow(row)
    try:
        read_tickets().to_excel(XLSX_PATH, index=False)
    except Exception as exc:  # e.g. the Excel file is open in Excel right now
        print(f"[ticket log] Excel copy not updated ({exc}). The CSV log is complete.")


def read_tickets() -> pd.DataFrame:
    """The whole log as a table (empty table if there are no tickets yet)."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
