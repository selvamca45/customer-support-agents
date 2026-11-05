"""
app.py - the Streamlit user interface for the BluePeak AI Support Desk.

Run:  streamlit run app.py

Two views, switched at the top of the page:

  Customer portal - chat with the support agents: type or record a voice message in any
                    language, attach a photo, watch the progress, and track "My tickets"
                    (including replies approved by a specialist).
  Support team    - New ticket (live agent timeline), Case management (human-in-the-loop),
                    Ticket log (filters, CSV / Excel), Dashboard (charts, AI cost),
                    Knowledge base (documents, upload, rebuild the ChromaDB index, test search),
                    About & how it works.
"""
import asyncio
import html
import io
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import KB_DIR, LOG_DIR, MODEL_NAME, USD_TO_INR
from agents.kb_search import get_kb, rebuild_kb
from agents.media import check_photo, transcribe_audio
from agents.ticket_logger import read_tickets, update_ticket
from team import Step, run_ticket
from tests.test_cases import TEST_CASES
from ui.about import render_about
from ui.theme import CSS, customer_hero_html, footer_html, hero_html, nav_html

st.set_page_config(page_title="BluePeak Support Desk", page_icon="🏔️", layout="wide",
                   initial_sidebar_state="collapsed")

AGENTS = {
    "query_agent": ("🔎", "Query Agent"),
    "knowledge_agent": ("📚", "Knowledge Agent"),
    "resolution_agent": ("✍️", "Resolution Agent"),
    "log_agent": ("🗂️", "Log Agent"),
    "system": ("🛟", "Safety net"),
}
# Progress shown to the customer while the agents work
STAGES = {
    "query_agent": (25, "Understanding your message"),
    "knowledge_agent": (50, "Searching BluePeak Store's policies"),
    "resolution_agent": (75, "Writing your answer"),
    "log_agent": (100, "Recording your ticket"),
}
URGENCY_SCORE = {"Low": 1, "Medium": 2, "High": 3}
CASE_STATUSES = ["Open", "In progress", "Closed", "Rejected"]
EVIDENCE_DIR = LOG_DIR / "evidence"

st.markdown(CSS, unsafe_allow_html=True)


# ================================================================= helpers
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def usage_line(row: dict) -> str:
    if not row.get("cost_usd"):
        return ""
    cost = float(row["cost_usd"])
    tokens = int(row["prompt_tokens"] or 0) + int(row["completion_tokens"] or 0)
    return (f"🧮 AI usage: {row['llm_calls']} model calls, {tokens:,} tokens · "
            f"estimated cost ${cost:.5f} (about ₹{cost * USD_TO_INR:.3f})")


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
                 "The draft below waits for approval in the **Case management** tab.")
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
    if usage_line(row):
        st.markdown(f'<p class="muted">{usage_line(row)}</p>', unsafe_allow_html=True)
    if last.get("kb_text"):
        with st.expander("📚 Policy passages the reply is based on"):
            st.text(last["kb_text"])


def customer_status(r) -> str:
    """How a ticket's state is described to the customer."""
    if r["review_status"] == "Approved":
        return "✅ Answered by a specialist"
    if r["case_status"] == "In progress":
        return "🔧 A specialist is working on it"
    if r["case_status"] == "Rejected":
        return "📁 Closed"
    if r["status"] == "Resolved":
        return "✅ Answered"
    return "🕐 Waiting for a specialist"


def save_evidence(ticket_id: str, photo) -> str:
    """Store the customer's photo next to the log and return its path."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(photo.name).suffix.lower() or ".jpg"
    path = EVIDENCE_DIR / f"{ticket_id}{suffix}"
    path.write_bytes(photo.getvalue())
    return str(path.relative_to(LOG_DIR.parent))


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


# ================================================================= header
try:
    kb = load_kb()
    kb_summary = f"{len(kb.chunks)} policy passages in {getattr(kb, 'store', 'memory')}"
except Exception as exc:  # e.g. no API key or no internet
    kb_summary = "Knowledge base not loaded"
    st.error(f"The knowledge base could not load: {exc}")
st.markdown(nav_html(MODEL_NAME, kb_summary), unsafe_allow_html=True)

view = st.radio("View", ["🙋 Customer portal", "🎧 Support team"], horizontal=True,
                key="view", label_visibility="collapsed")


# ================================================================= customer portal
def render_customer_portal() -> None:
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("composer_v", 0)
    v = st.session_state.composer_v

    st.markdown(customer_hero_html(st.session_state.get("cust_name", "")), unsafe_allow_html=True)
    chat_col, side_col = st.columns([3, 2], gap="large")

    # ---------------- right: who you are + my tickets
    with side_col:
        st.markdown("#### Your details")
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name", key="cust_name", placeholder="Priya")
        order_id = c2.text_input("Order ID (optional)", key="cust_order", placeholder="ORD-4512")

        st.markdown("#### My tickets")
        if not name.strip():
            st.caption("Enter your name to see your tickets and the replies from our team.")
        else:
            df = read_tickets()
            mine = df[df["customer_name"].str.strip().str.lower() == name.strip().lower()]
            if mine.empty:
                st.caption("No tickets yet. Your messages will appear here as tickets.")
            for _, r in mine.iloc[::-1].head(8).iterrows():
                with st.expander(f"{customer_status(r)} · {r['ticket_id']} · {r['category']}"):
                    st.markdown(f'<div class="tk-meta">Sent {r["timestamp"]}</div>',
                                unsafe_allow_html=True)
                    st.markdown(f"**You wrote:** {r['complaint']}")
                    if r["review_status"] == "Approved":
                        st.markdown("**Reply from our specialist:**")
                        st.markdown(r["reviewed_reply"] or r["response"])
                    elif r["status"] == "Resolved":
                        st.markdown("**Our reply:**")
                        st.markdown(r["response"])
                    else:
                        st.markdown("A specialist is reviewing your case and will reply here.")
                    if r["message_to_customer"]:
                        st.info(f"Message from our team: {r['message_to_customer']}")

    # ---------------- left: conversation + composer
    with chat_col:
        st.markdown("#### Chat with BluePeak support")
        with st.chat_message("assistant", avatar="🏔️"):
            st.markdown("Hello! I can help with deliveries, returns and refunds, your account, "
                        "products and payments. Tell me what happened, in any language.")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🏔️"):
                st.markdown(msg["text"])
                if msg.get("photo"):
                    st.image(msg["photo"], width=220)
                if msg.get("meta"):
                    st.caption(msg["meta"])

        progress_slot = st.empty()

        def on_voice() -> None:
            audio = st.session_state.get(f"voice_{v}")
            if audio is None:
                return
            try:
                text = transcribe_audio(audio.getvalue())
            except Exception as exc:
                st.session_state.voice_error = str(exc)
                return
            key = f"msg_{v}"
            previous = st.session_state.get(key, "").strip()
            st.session_state[key] = f"{previous} {text}".strip() if previous else text
            st.session_state.voice_note = True
            st.session_state.pop("voice_error", None)

        with st.container(border=True):
            st.text_area("Your message", key=f"msg_{v}", height=110,
                         placeholder="For example: My laptop arrived with a cracked screen. "
                                     "I need a replacement.")
            if st.session_state.get("voice_note"):
                st.caption("🎙️ Transcribed from your voice message. Check the text and edit it "
                           "if needed before sending.")
            if st.session_state.get("voice_error"):
                st.warning(f"The voice message could not be transcribed: "
                           f"{st.session_state.voice_error}")
            a1, a2 = st.columns(2)
            a1.audio_input("Or record a voice message", key=f"voice_{v}", on_change=on_voice)
            photo = a2.file_uploader("Add a photo (optional)", type=["png", "jpg", "jpeg", "webp"],
                                     key=f"photo_{v}")
            send = st.button("Send to support", type="primary", key=f"send_{v}")

    if not send:
        return
    text = st.session_state.get(f"msg_{v}", "").strip()
    if not name.strip():
        st.warning("Please add your name under Your details, so we can find your tickets.")
        return
    if not text:
        st.warning("Type or record your message first.")
        return

    photo_bytes = photo.getvalue() if photo else None
    evidence = None
    if photo:
        progress_slot.progress(10, text="Checking your photo…")
        evidence = check_photo(photo_bytes, photo.type or "image/jpeg", text)

    def on_step(step: Step) -> None:
        stage = STAGES.get(step.agent)
        if stage:
            pct, label = stage
            progress_slot.progress(pct, text=f"Support agents working ({pct // 25}/4): {label}…")

    progress_slot.progress(5, text="Support agents working (0/4): starting…")
    ctx = asyncio.run(run_ticket(text, name.strip(), order_id.strip(), on_step=on_step,
                                 evidence=evidence, channel="Customer portal"))
    row = ctx.result
    if photo:
        path = save_evidence(row["ticket_id"], photo)
        update_ticket(row["ticket_id"], {"evidence_file": path})
    progress_slot.empty()

    if row["status"] == "Resolved":
        answer = row["response"]
        meta = (f"Ticket {row['ticket_id']} · answered from {row['kb_sources']} "
                f"in {row['processing_seconds']}s")
    else:
        first = name.strip().split()[0]
        answer = (f"Thank you, {first}. Your case **{row['ticket_id']}** needs a specialist's "
                  "attention, so a member of our support team will review it and reply to you. "
                  "You can follow it under **My tickets**.")
        meta = f"Ticket {row['ticket_id']} · passed to a specialist"
    st.session_state.chat.append({"role": "user", "text": text, "photo": photo_bytes})
    st.session_state.chat.append({"role": "assistant", "text": answer, "meta": meta})
    st.session_state.composer_v += 1          # gives the composer fresh, empty fields
    st.session_state.pop("voice_note", None)
    st.rerun()


# ================================================================= support team
def render_new_ticket() -> None:
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
                  The reasons are listed, and the case waits in <b>Case management</b>.</li>
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
    else:
        if not just_ran:
            with st.expander("🧭 Agent timeline of the last ticket"):
                for step in st.session_state.last["steps"]:
                    render_step(step)
        render_result(st.session_state.last)


def render_case_management() -> None:
    df = read_tickets()
    open_cases = df[df["case_status"].isin(["Open", "In progress"])]
    closed = df[(df["status"] == "Escalated") & df["case_status"].isin(["Closed", "Rejected"])]

    m = st.columns(3)
    m[0].metric("Open cases", int((df["case_status"] == "Open").sum()))
    m[1].metric("In progress", int((df["case_status"] == "In progress").sum()))
    m[2].metric("Escalated cases closed", len(closed))
    st.caption("Escalated tickets wait here for a person. Read why the rules escalated each one, "
               "check the customer's photo, edit the AI's draft reply, and approve it. The customer "
               "sees the approved reply and your message in the portal's My tickets.")

    if open_cases.empty:
        st.info("No open cases. Escalated tickets will appear here.")
    for _, r in open_cases.iloc[::-1].iterrows():
        tid = r["ticket_id"]
        icon = "🔧" if r["case_status"] == "In progress" else "🚨"
        with st.expander(f"{icon} {tid} · {r['category']} · {r['customer_name'] or 'Customer'} · "
                         f"{r['case_status']} · {r['escalation_reason'][:60]}"):
            info_col, photo_col = st.columns([3, 1])
            with info_col:
                st.markdown(f"**Complaint:** {r['complaint']}")
                st.markdown(f"**Why escalated:** {r['escalation_reason']}")
                st.markdown(f"**Sources:** {r['kb_sources']} · **Confidence:** {r['confidence']} · "
                            f"**Grounding:** {r['grounding']} · **Channel:** {r['channel']}")
            with photo_col:
                photo_path = LOG_DIR.parent / r["evidence_file"] if r["evidence_file"] else None
                if photo_path and photo_path.exists():
                    st.image(str(photo_path), caption="Customer's photo")
                    st.caption(f"AI photo check: {r['evidence_check']}")
            edited = st.text_area("Reply to send (edit if needed)",
                                  value=r["reviewed_reply"] or r["response"],
                                  key=f"edit_{tid}", height=220)
            c1, c2 = st.columns([1, 2])
            new_status = c1.selectbox("Case status", CASE_STATUSES,
                                      index=CASE_STATUSES.index(r["case_status"]),
                                      key=f"status_{tid}")
            message = c2.text_input("Message to the customer (shown in My tickets)",
                                    value=r["message_to_customer"], key=f"msg_cm_{tid}",
                                    placeholder="A replacement is arranged for 7 November.")
            note = st.text_input("Internal note (not shown to the customer)",
                                 value=r["reviewer_note"], key=f"note_{tid}")
            b1, b2 = st.columns(2)
            if b1.button("✅ Approve reply and close case", key=f"ok_{tid}", type="primary"):
                update_ticket(tid, {"review_status": "Approved", "reviewed_reply": edited,
                                    "case_status": "Closed", "message_to_customer": message,
                                    "reviewer_note": note, "reviewed_at": now()})
                st.rerun()
            if b2.button("💾 Save update", key=f"save_{tid}"):
                update_ticket(tid, {"case_status": new_status, "message_to_customer": message,
                                    "reviewer_note": note, "reviewed_at": now()})
                st.rerun()

    if not closed.empty:
        st.divider()
        st.markdown(f"**Closed escalated cases: {len(closed)}**")
        st.dataframe(closed[["ticket_id", "category", "case_status", "review_status",
                             "message_to_customer", "reviewer_note", "reviewed_at"]].iloc[::-1],
                     hide_index=True)


def render_ticket_log() -> None:
    df = read_tickets()
    if df.empty:
        st.info("No tickets yet. Answer a complaint and it will be logged here.")
        return
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    status_filter = f1.multiselect("Status", sorted(df["status"].unique()))
    category_filter = f2.multiselect("Category", sorted(df["category"].unique()))
    channel_filter = f3.multiselect("Channel", sorted(df["channel"].unique()))
    search = f4.text_input("Search complaints and replies")
    view_df = df
    if status_filter:
        view_df = view_df[view_df["status"].isin(status_filter)]
    if category_filter:
        view_df = view_df[view_df["category"].isin(category_filter)]
    if channel_filter:
        view_df = view_df[view_df["channel"].isin(channel_filter)]
    if search:
        text = view_df["complaint"] + " " + view_df["response"]
        view_df = view_df[text.str.contains(search, case=False, regex=False)]

    st.caption(f"Showing {len(view_df)} of {len(df)} tickets, newest first. Use the filters to "
               "narrow the list, and download the view as CSV or Excel.")
    main_cols = ["ticket_id", "timestamp", "channel", "customer_name", "order_id", "complaint",
                 "category", "urgency", "sentiment", "kb_sources", "response", "status",
                 "escalation_reason", "confidence", "case_status", "cost_usd"]
    st.dataframe(view_df[main_cols].iloc[::-1], hide_index=True, height=420)

    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Download CSV", view_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="tickets.csv", mime="text/csv")
    buffer = io.BytesIO()
    view_df.to_excel(buffer, index=False)
    d2.download_button("⬇️ Download Excel", buffer.getvalue(), file_name="tickets.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("🔍 Full details of one ticket"):
        tid = st.selectbox("Ticket", view_df["ticket_id"].iloc[::-1])
        st.json(view_df[view_df["ticket_id"] == tid].iloc[0].to_dict())


def render_dashboard() -> None:
    df = read_tickets()
    if df.empty:
        st.info("The dashboard fills up as complaints are answered.")
        return
    total = len(df)
    resolved = int((df["status"] == "Resolved").sum())
    escalated = total - resolved
    confidence = pd.to_numeric(df["confidence"], errors="coerce").mean()
    seconds = pd.to_numeric(df["processing_seconds"], errors="coerce").mean()
    urgency = df["urgency"].map(URGENCY_SCORE).dropna().mean()
    urgency_label = ("Low" if urgency < 1.5 else "Medium" if urgency < 2.5 else "High") \
        if pd.notna(urgency) else "-"

    st.caption("A live summary of the ticket log: what customers complain about, how much "
               "the AI resolves on its own, why tickets need a person, and what the AI costs.")
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

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Case status**")
        st.altair_chart(bar_chart(df["case_status"], "case status", order=CASE_STATUSES))
    with c6:
        st.markdown("**Channel**")
        st.altair_chart(bar_chart(df["channel"], "channel"))

    reasons = (df["escalation_reason"].str.split("; ").explode().dropna()
               .map(lambda r: re.sub(r"\s*\(.*\)\s*$", "", r).strip()))
    reasons = reasons.map(lambda r: "Log agent concern" if r.startswith("Log agent") else r)
    if not reasons[reasons != ""].empty:
        st.markdown("**Why tickets were escalated**")
        st.altair_chart(bar_chart(reasons, "reason"))

    costed = df[df["cost_usd"] != ""]
    if not costed.empty:
        cost = pd.to_numeric(costed["cost_usd"], errors="coerce")
        tokens = (pd.to_numeric(costed["prompt_tokens"], errors="coerce")
                  + pd.to_numeric(costed["completion_tokens"], errors="coerce"))
        calls = pd.to_numeric(costed["llm_calls"], errors="coerce")
        st.markdown("**AI usage and cost**")
        st.caption(f"Based on the {len(costed)} tickets logged with usage data. Prices: "
                   "gpt-4o-mini list prices, set in config.py / .env.")
        u = st.columns(4)
        u[0].metric("Average tokens per ticket", f"{tokens.mean():,.0f}")
        u[1].metric("Model calls per ticket", f"{calls.mean():.1f}")
        u[2].metric("Average cost per ticket", f"₹{cost.mean() * USD_TO_INR:.3f}",
                    help=f"${cost.mean():.5f} per ticket")
        u[3].metric("Cost per 1,000 tickets", f"₹{cost.mean() * USD_TO_INR * 1000:,.0f}",
                    help=f"${cost.mean() * 1000:.2f} per 1,000 tickets")


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^a-z0-9_]+", "_", Path(name).stem.lower()).strip("_") or "document"
    return f"{stem}.md"


def render_knowledge_base() -> None:
    kb = load_kb()
    docs = sorted({c.source for c in kb.chunks})
    st.caption(f"{len(kb.chunks)} passages from {len(docs)} policy documents, stored in "
               f"{kb.store}. Every reply is built only from these documents.")

    rows = []
    for doc in docs:
        doc_chunks = [c for c in kb.chunks if c.source == doc]
        rows.append({"document": doc, "title": doc_chunks[0].title, "passages": len(doc_chunks)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    with st.expander("📖 Read a document"):
        choice = st.selectbox("Document", docs, key="kb_doc")
        st.markdown((KB_DIR / choice).read_text(encoding="utf-8"))

    st.markdown("**Add or update policy documents**")
    st.caption("Upload Markdown or text files. Use '## ' headings to split a document into "
               "sections; each section becomes one searchable passage. A file with the same "
               "name replaces the old one.")
    st.session_state.setdefault("kb_upload_v", 0)
    uploads = st.file_uploader("Policy documents (.md or .txt)", type=["md", "txt"],
                               accept_multiple_files=True,
                               key=f"kb_up_{st.session_state.kb_upload_v}")
    b1, b2 = st.columns(2)
    if b1.button("💾 Save documents and rebuild the index", type="primary", disabled=not uploads):
        for up in uploads:
            text = up.getvalue().decode("utf-8", errors="replace")
            if not any(line.startswith("## ") for line in text.splitlines()):
                title = Path(up.name).stem.replace("_", " ").title()
                text = f"# {title}\n\n## {title}\n{text}"
            (KB_DIR / _safe_filename(up.name)).write_text(text, encoding="utf-8")
        rebuild_kb()
        load_kb.clear()
        st.session_state.kb_upload_v += 1
        st.success(f"Saved {len(uploads)} document(s) and rebuilt the index.")
        st.rerun()
    if b2.button("🔄 Rebuild the index from the folder"):
        rebuild_kb()
        load_kb.clear()
        st.success("The index was rebuilt.")
        st.rerun()

    st.markdown("**Test the search**")
    st.caption("See exactly which passages the Knowledge Agent would find, and their scores.")
    query = st.text_input("Search query", placeholder="refund for shoes", key="kb_query")
    if query:
        results, best = get_kb().search(query)
        if not results:
            st.warning(f"NO_MATCH: the best score was {best:.2f}, below the minimum "
                       f"{kb.min_score:.2f}. The agents would escalate this question.")
        for chunk, score in results:
            with st.container(border=True):
                st.markdown(f"**{chunk.source} › {chunk.section}**  ·  score {score:.2f}")
                st.text(chunk.text)


def render_support_team() -> None:
    log_df = read_tickets()
    if len(log_df):
        n_resolved = int((log_df["status"] == "Resolved").sum())
        avg = pd.to_numeric(log_df["processing_seconds"], errors="coerce").mean()
        st.markdown(hero_html(len(log_df), f"{n_resolved / len(log_df):.0%}",
                              f"{avg:.1f}s" if pd.notna(avg) else "-"), unsafe_allow_html=True)
    else:
        st.markdown(hero_html(0, "-", "-"), unsafe_allow_html=True)

    tabs = st.tabs(["🎫 New ticket", "🗂️ Case management", "📋 Ticket log", "📊 Dashboard",
                    "📚 Knowledge base", "💡 About & how it works"])
    with tabs[0]:
        render_new_ticket()
    with tabs[1]:
        render_case_management()
    with tabs[2]:
        render_ticket_log()
    with tabs[3]:
        render_dashboard()
    with tabs[4]:
        render_knowledge_base()
    with tabs[5]:
        render_about()


# ================================================================= page
if view.startswith("🙋"):
    render_customer_portal()
else:
    render_support_team()

st.markdown(footer_html(), unsafe_allow_html=True)
