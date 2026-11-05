"""
app.py - the Streamlit user interface for the BluePeak AI Support Desk.

Run:  streamlit run app.py

Tabs:
  New Ticket    - complaint form, live agent timeline, final reply with badge
  Human Review  - support staff approve or edit escalated replies (human-in-the-loop)
  Ticket Log    - the full log as a table, with filters and CSV / Excel download
  Dashboard     - tickets per category, resolved vs escalated, urgency, top reasons
  About         - why the app exists and how it was built (ui/about.py)
"""
import asyncio
import html
import io
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from config import MODEL_NAME
from agents.kb_search import get_kb
from agents.ticket_logger import read_tickets, update_ticket
from team import Step, run_ticket
from tests.test_cases import TEST_CASES
from ui.about import render_about
from ui.theme import CSS, hero_html

st.set_page_config(page_title="BluePeak AI Support Desk", page_icon="🎧", layout="wide")

AGENTS = {
    "query_agent": ("🔎", "Query Agent"),
    "knowledge_agent": ("📚", "Knowledge Agent"),
    "resolution_agent": ("✍️", "Resolution Agent"),
    "log_agent": ("🗂️", "Log Agent"),
    "system": ("🛟", "Safety net"),
}
URGENCY_SCORE = {"Low": 1, "Medium": 2, "High": 3}

st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def badge(status: str) -> str:
    if status == "Resolved":
        return '<span class="badge badge-resolved">✅ Resolved</span>'
    return '<span class="badge badge-escalated">🚨 Escalated</span>'


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_kb():
    return get_kb()


def render_step(step: Step) -> None:
    """Draw one agent step as a bordered card (used live and in the saved timeline)."""
    icon, name = AGENTS.get(step.agent, ("⚙️", step.agent))
    with st.container(border=True):
        st.markdown(
            f'<span class="step-head">{icon} {name} — {html.escape(step.title)}</span> '
            f'<span class="muted">· {step.seconds:.1f}s</span>',
            unsafe_allow_html=True,
        )
        detail = step.detail
        if step.agent == "knowledge_agent" and len(detail) > 900:
            detail = detail[:900] + "\n... (full passages are shown below the reply)"
        st.text(detail)


def render_result(last: dict) -> None:
    """The final outcome: badge, key facts, escalation reasons and the customer reply."""
    row = last["row"]
    escalated = row["status"] == "Escalated"
    st.markdown(f"### Ticket {row['ticket_id']} {badge(row['status'])}", unsafe_allow_html=True)

    c = st.columns(5)
    c[0].metric("Category", row["category"] or "-")
    c[1].metric("Urgency", row["urgency"] or "-")
    c[2].metric("Sentiment", row["sentiment"] or "-")
    c[3].metric("Language", row["language"] or "-")
    c[4].metric("Confidence", row["confidence"])

    if escalated:
        reasons = "\n".join(f"- {r}" for r in row["escalation_reason"].split("; ") if r)
        st.error(f"**Escalated to human support.** Reasons:\n{reasons}\n\n"
                 "The draft below waits for approval in the **Human Review** tab.")
    else:
        st.success("**Resolved automatically.** The reply is grounded in company policy.")

    st.markdown("**Reply to the customer**")
    reply = html.escape(row["response"]).replace("\n", "<br>")
    st.markdown(f'<div class="reply-box {"escalated" if escalated else ""}">{reply}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p class="muted">📄 Sources: {html.escape(row["kb_sources"])} · '
        f'🛡️ Grounding check: {html.escape(row["grounding"])} · '
        f'Coverage: {row["coverage"]} · ⏱️ {row["processing_seconds"]}s · Model: {MODEL_NAME}</p>',
        unsafe_allow_html=True,
    )
    if last.get("kb_text"):
        with st.expander("📚 Policy passages the reply is based on"):
            st.text(last["kb_text"])


# Chart colours: one blue for single-series charts; status green/red only for
# Resolved/Escalated, always paired with an icon + label (never colour alone).
SERIES_BLUE = "#2a78d6"
STATUS_COLORS = {"✅ Resolved": "#0ca30c", "🚨 Escalated": "#d03b3b"}
INK = "#3d3d3a"
GRID = "#e1e0d9"


def bar_chart(series: pd.Series, label: str, order: list[str] | None = None,
              colors: dict[str, str] | None = None) -> alt.Chart:
    """Horizontal bar chart of counts: labelled bars, value at the bar end, tooltips."""
    data = (series[series != ""].value_counts()
            .rename_axis(label).reset_index(name="tickets"))
    if order:
        order = [o for o in order if o in set(data[label])]
    y = alt.Y(f"{label}:N", sort=order or "-x", title=None,
              axis=alt.Axis(labelLimit=280, ticks=False, domain=False, labelColor=INK,
                            labelFontSize=13))
    x = alt.X("tickets:Q", title=None,
              axis=alt.Axis(tickMinStep=1, format="d", gridColor=GRID, domain=False,
                            ticks=False, labelColor="#8a8983"))
    color = (alt.Color(f"{label}:N", legend=None,
                       scale=alt.Scale(domain=list(colors), range=list(colors.values())))
             if colors else alt.value(SERIES_BLUE))
    base = alt.Chart(data).encode(y=y, x=x, tooltip=[alt.Tooltip(f"{label}:N"),
                                                     alt.Tooltip("tickets:Q", title="tickets")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.62}).encode(color=color)
    values = base.mark_text(align="left", dx=6, color=INK, fontSize=13,
                            fontWeight="bold").encode(text="tickets:Q")
    return ((bars + values)
            .properties(width="container", height=max(110, 46 * len(data)))
            .configure_view(strokeWidth=0))


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### BluePeak Support Desk")
    st.caption("AI customer support automation")
    st.markdown(
        "**Quick start**\n"
        "1. Open **New ticket** and pick a sample complaint.\n"
        "2. Press **Answer this complaint** and watch the agents work.\n"
        "3. Approve escalated replies in **Human review**.\n"
        "4. See every ticket in **Ticket log** and trends in **Dashboard**."
    )
    st.divider()
    st.markdown("**System status**")
    try:
        kb = load_kb()
        st.markdown(f"Knowledge base: {len(kb.chunks)} passages from "
                    f"{len({c.source for c in kb.chunks})} policy documents")
        st.markdown(f"Search: {kb.mode}")
    except Exception as exc:
        st.error(f"The knowledge base could not load: {exc}")
    st.markdown(f"Model: `{MODEL_NAME}`")
    st.divider()
    st.caption("Built with AutoGen, OpenAI and Streamlit for the Social Eagle "
               "AI Engineering Program.")

# ----------------------------------------------------------------- header
log_df = read_tickets()
if len(log_df):
    n_resolved = int((log_df["status"] == "Resolved").sum())
    avg = pd.to_numeric(log_df["processing_seconds"], errors="coerce").mean()
    st.markdown(hero_html(len(log_df), f"{n_resolved / len(log_df):.0%}",
                          f"{avg:.1f}s" if pd.notna(avg) else "-"), unsafe_allow_html=True)
else:
    st.markdown(hero_html(0, "-", "-"), unsafe_allow_html=True)

tab_ticket, tab_review, tab_log, tab_dash, tab_about = st.tabs(
    ["🎫 New ticket", "👩‍💼 Human review", "📋 Ticket log", "📊 Dashboard",
     "💡 About & how it works"]
)

# ----------------------------------------------------------------- new ticket
with tab_ticket:
    samples = {"✍️ Write my own complaint": None}
    samples.update({f"Test #{c['id']}: {c['complaint'][:60]}": c for c in TEST_CASES})

    def fill_sample() -> None:
        case = samples[st.session_state.sample]
        st.session_state.complaint = case["complaint"] if case else ""
        st.session_state.customer_name = case["customer_name"] if case else ""
        st.session_state.order_id = case["order_id"] if case else ""

    form_col, guide_col = st.columns([3, 2], gap="large")
    with form_col:
        st.selectbox("Start from a sample complaint, or write your own", list(samples),
                     key="sample", on_change=fill_sample)
        with st.form("ticket_form"):
            c1, c2 = st.columns(2)
            customer_name = c1.text_input("Customer name (optional)", key="customer_name",
                                          placeholder="Priya")
            order_id = c2.text_input("Order ID (optional)", key="order_id", placeholder="ORD-4512")
            complaint = st.text_area("Customer complaint", key="complaint", height=130,
                                     placeholder="For example: My order is 10 days late "
                                                 "and nobody replies to my emails.")
            submitted = st.form_submit_button("Answer this complaint", type="primary")
    with guide_col:
        st.markdown(
            """<div class="panel guide"><h4>How to use this desk</h4>
            <ol>
              <li>Pick a sample complaint, or type one in any language.</li>
              <li>Add the customer's name and order ID if you have them, so the reply is personal.</li>
              <li>Press <b>Answer this complaint</b>. Each agent's work appears as it finishes,
                  usually within 10 seconds.</li>
              <li>Read the reply and its badge below.</li>
            </ol></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="panel" style="margin-top:12px"><h4>What the result means</h4>
            <ul>
              <li><b style="color:#0ca30c">✅ Resolved</b>: the reply is based on store policy
                  and can be sent.</li>
              <li><b style="color:#d03b3b">🚨 Escalated</b>: a person must check it first.
                  The reasons are listed, and the reply waits in <b>Human review</b>.</li>
            </ul></div>""",
            unsafe_allow_html=True,
        )

    just_ran = False
    if submitted:
        if not complaint.strip():
            st.warning("Write the customer's complaint first, or pick a sample above.")
        else:
            box = st.status("The agents are working on this complaint...", expanded=True)

            def on_step(step: Step) -> None:
                with box:
                    render_step(step)

            ctx = asyncio.run(run_ticket(complaint.strip(), customer_name.strip(),
                                         order_id.strip(), on_step=on_step))
            row = ctx.result
            box.update(
                label=f"Finished: {len(ctx.steps)} agent steps in {row['processing_seconds']}s",
                state="error" if ctx.error else "complete",
                expanded=False,
            )
            st.session_state.last = {"row": row, "steps": ctx.steps, "kb_text": ctx.kb_text}
            just_ran = True

    if "last" not in st.session_state:
        st.markdown('<div class="empty">The answer to your complaint will appear here, with '
                    'each agent\'s work shown step by step.</div>', unsafe_allow_html=True)
    if "last" in st.session_state:
        if not just_ran:
            with st.expander("🧭 Agent timeline of the last ticket"):
                for step in st.session_state.last["steps"]:
                    render_step(step)
        render_result(st.session_state.last)

# ----------------------------------------------------------------- human review
with tab_review:
    df = read_tickets()
    pending = df[(df["status"] == "Escalated") & (df["review_status"] == "Pending review")]
    done = df[df["review_status"].isin(["Approved", "Returned"])]
    st.subheader(f"Escalated tickets waiting for a human: {len(pending)}")
    st.caption("These tickets were escalated by the rules. Read why, edit the AI's draft reply if "
               "needed, then approve it. Every decision is saved in the ticket log.")

    if pending.empty:
        st.info("No tickets are waiting for review. Escalated tickets will appear here.")
    for _, r in pending.iloc[::-1].iterrows():
        tid = r["ticket_id"]
        with st.expander(f"🚨 {tid} · {r['category']} · {r['customer_name'] or 'Customer'} · "
                         f"{r['escalation_reason'][:70]}"):
            st.markdown(f"**Complaint:** {r['complaint']}")
            st.markdown(f"**Why escalated:** {r['escalation_reason']}")
            st.markdown(f"**Sources:** {r['kb_sources']} · **Confidence:** {r['confidence']} · "
                        f"**Grounding:** {r['grounding']}")
            edited = st.text_area("Reply to send (edit if needed)", value=r["response"],
                                  key=f"edit_{tid}", height=220)
            note = st.text_input("Reviewer note (optional)", key=f"note_{tid}")
            b1, b2 = st.columns(2)
            if b1.button("✅ Approve and send", key=f"ok_{tid}", type="primary"):
                update_ticket(tid, {"review_status": "Approved", "reviewed_reply": edited,
                                    "reviewer_note": note,
                                    "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                st.rerun()
            if b2.button("↩️ Needs more work", key=f"back_{tid}"):
                update_ticket(tid, {"review_status": "Returned", "reviewer_note": note,
                                    "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                st.rerun()

    if not done.empty:
        st.divider()
        st.markdown(f"**Reviewed tickets: {len(done)}**")
        st.dataframe(done[["ticket_id", "category", "review_status", "reviewer_note", "reviewed_at"]]
                     .iloc[::-1], hide_index=True)

# ----------------------------------------------------------------- ticket log
with tab_log:
    df = read_tickets()
    if df.empty:
        st.info("No tickets yet. Answer a complaint in the New ticket tab and it will be logged here.")
    else:
        f1, f2, f3 = st.columns([1, 1, 2])
        status_filter = f1.multiselect("Status", sorted(df["status"].unique()))
        category_filter = f2.multiselect("Category", sorted(df["category"].unique()))
        search = f3.text_input("Search complaints and replies")
        view = df
        if status_filter:
            view = view[view["status"].isin(status_filter)]
        if category_filter:
            view = view[view["category"].isin(category_filter)]
        if search:
            text = view["complaint"] + " " + view["response"]
            view = view[text.str.contains(search, case=False, regex=False)]

        st.caption(f"Showing {len(view)} of {len(df)} tickets, newest first. Use the filters to "
                   "narrow the list, and download the view as CSV or Excel.")
        main_cols = ["ticket_id", "timestamp", "customer_name", "order_id", "complaint",
                     "category", "urgency", "sentiment", "kb_sources", "response",
                     "status", "escalation_reason", "confidence", "review_status"]
        st.dataframe(view[main_cols].iloc[::-1], hide_index=True, height=420)

        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇️ Download CSV", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name="tickets.csv", mime="text/csv")
        buffer = io.BytesIO()
        view.to_excel(buffer, index=False)
        d2.download_button("⬇️ Download Excel", buffer.getvalue(), file_name="tickets.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with st.expander("🔍 Full details of one ticket"):
            tid = st.selectbox("Ticket", view["ticket_id"].iloc[::-1])
            st.json(view[view["ticket_id"] == tid].iloc[0].to_dict())

# ----------------------------------------------------------------- dashboard
with tab_dash:
    df = read_tickets()
    if df.empty:
        st.info("The dashboard fills up as complaints are answered.")
    else:
        total = len(df)
        resolved = int((df["status"] == "Resolved").sum())
        escalated = total - resolved
        confidence = pd.to_numeric(df["confidence"], errors="coerce").mean()
        seconds = pd.to_numeric(df["processing_seconds"], errors="coerce").mean()
        urgency = df["urgency"].map(URGENCY_SCORE).dropna().mean()
        urgency_label = ("Low" if urgency < 1.5 else "Medium" if urgency < 2.5 else "High") \
            if pd.notna(urgency) else "-"

        st.caption("A live summary of the ticket log: what customers complain about, how much "
                   "the AI resolves on its own, and why tickets need a person.")
        m = st.columns(6)
        m[0].metric("Total tickets", total)
        m[1].metric("Resolved by AI", resolved, f"{resolved / total:.0%} automation")
        m[2].metric("Escalated", escalated)
        m[3].metric("Average urgency", urgency_label,
                    help=(f"Low = 1, Medium = 2, High = 3. Average score: {urgency:.1f}"
                          if pd.notna(urgency) else None))
        m[4].metric("Average confidence", f"{confidence:.2f}" if pd.notna(confidence) else "-")
        m[5].metric("Average handling time", f"{seconds:.1f}s" if pd.notna(seconds) else "-")

        status_labels = df["status"].map({"Resolved": "✅ Resolved", "Escalated": "🚨 Escalated"})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tickets per category**")
            st.altair_chart(bar_chart(df["category"], "category"))
        with c2:
            st.markdown("**Resolved vs escalated**")
            st.altair_chart(bar_chart(status_labels.fillna(""), "status",
                                      order=list(STATUS_COLORS), colors=STATUS_COLORS))

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Urgency**")
            st.altair_chart(bar_chart(df["urgency"], "urgency", order=["High", "Medium", "Low"]))
        with c4:
            st.markdown("**Customer sentiment**")
            st.altair_chart(bar_chart(df["sentiment"], "sentiment",
                                      order=["Angry", "Frustrated", "Neutral", "Positive"]))

        reasons = (df["escalation_reason"].str.split("; ").explode().dropna()
                   .map(lambda r: re.sub(r"\s*\(.*\)\s*$", "", r).strip()))
        reasons = reasons.map(lambda r: "Log agent concern" if r.startswith("Log agent") else r)
        if not reasons[reasons != ""].empty:
            st.markdown("**Why tickets were escalated**")
            st.altair_chart(bar_chart(reasons, "reason"))

# ----------------------------------------------------------------- about
with tab_about:
    render_about()
