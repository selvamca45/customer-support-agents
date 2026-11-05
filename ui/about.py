"""
ui/about.py - the "About & how it works" tab: why the app exists, how a ticket
flows through the system, and how it was built.
"""
import streamlit as st

WORKFLOW_DOT = r"""
digraph G {
  rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        color="#b8c6d8", fillcolor="#ffffff", fontcolor="#12355b", margin="0.14,0.08"];
  edge [color="#6b7a90", fontname="Helvetica", fontsize=9, fontcolor="#5b6b80", arrowsize=0.7];

  customer [label="Customer complaint\n(Streamlit form)", fillcolor="#eef4fb"];

  subgraph cluster_team {
    label="AutoGen RoundRobinGroupChat (stops when the Log Agent says LOGGED)";
    fontname="Helvetica"; fontsize=10; fontcolor="#2f6fb2";
    style="rounded,dashed"; color="#2f6fb2";
    query [label="1  Query Agent\ncategory, urgency,\nsentiment, risk flags"];
    knowledge [label="2  Knowledge Agent\ncalls search_kb()"];
    resolution [label="3  Resolution Agent\nreply + coverage"];
    log [label="4  Log Agent\ncalls log_ticket()"];
  }

  kb [label="Knowledge base\n6 policy documents\n34 passages", shape=cylinder, fillcolor="#f1f5f9"];
  check [label="Grounding check\nnumbers, menu paths,\nsources", fillcolor="#fff7e6"];
  rules [label="Escalation rules\n(plain Python)", fillcolor="#fff7e6"];
  store [label="logs/tickets.csv\nlogs/tickets.xlsx", shape=folder, fillcolor="#f1f5f9"];
  resolved [label="Resolved\nreply sent", fillcolor="#e3f6e3", color="#0ca30c"];
  human [label="Escalated\nCase management tab", fillcolor="#fbe7e7", color="#d03b3b"];

  customer -> query -> knowledge -> resolution -> log;
  knowledge -> kb [dir=both, label="hybrid search"];
  resolution -> check [style=dashed];
  check -> log [style=dashed];
  log -> rules -> store;
  rules -> resolved; rules -> human;
}
"""


def render_about() -> None:
    st.subheader("Why this application exists")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """<div class="panel"><h4>The problem</h4>
            <p>A growing online store receives hundreds of complaints a day: late deliveries,
            refunds, broken products, payment problems and login issues. The support team
            answers the same questions again and again, while urgent cases such as fraud
            or angry customers get lost in the queue.</p></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="panel"><h4>What this desk does about it</h4>
            <p>Routine questions are answered instantly from the store's own policies, in the
            customer's language. Risky or unclear cases are sent to a person with a written
            reason, and every ticket is logged so the team can see trends and improve the
            policies.</p></div>""",
            unsafe_allow_html=True,
        )

    st.subheader("How a ticket flows through the system")
    st.graphviz_chart(WORKFLOW_DOT)
    st.markdown(
        """
1. **Query Agent** reads the complaint and returns strict JSON (a Pydantic model):
   category, urgency, sentiment, risk flags, language and English search keywords.
2. **Knowledge Agent** calls the `search_kb` tool. The six policy documents are split into
   34 passages and searched by meaning (OpenAI embeddings) plus exact keywords. Below a
   calibrated minimum score the tool returns `NO_MATCH` instead of guessing.
3. **Resolution Agent** writes the reply from those passages only, and labels how fully
   they cover the question. Python then checks every number and menu path in the reply
   against the passages.
4. **Log Agent** calls the `log_ticket` tool, which applies the escalation rules, writes one
   row to the CSV and Excel log, and answers `LOGGED`, which ends the team's conversation.
5. **Resolved** replies are ready to send. **Escalated** tickets wait in the
   Case management tab, where a person edits and approves the reply, sets the case status
   and writes a message the customer sees under My tickets in the customer portal.
"""
    )

    st.subheader("Built to be trustworthy")
    st.markdown(
        """
| Safeguard | What it prevents |
|---|---|
| Structured outputs (Pydantic) | Agents passing vague or malformed text to each other |
| Calibrated `NO_MATCH` threshold (0.33) | Answering questions the policies do not cover |
| Grounding check on numbers, menu paths and sources | Invented timelines, fees or website steps reaching a customer |
| Escalation rules written in Python | Inconsistent decisions; the same ticket always gets the same result |
| Log Agent concerns must quote the customer | The AI escalating for something the customer never said |
| Safety net in the team | A lost ticket if the AI service fails; it is still logged and escalated |
| Case management (human review) | Risky replies going out without a person's approval |
| Photo check by AI | A photo that does not match the complaint going unnoticed; such cases are escalated |
"""
    )

    st.subheader("When a ticket goes to a person")
    st.markdown(
        """
A ticket is escalated if **any** of these is true, and every rule that fires is written
into the log as the reason:

- urgency is High
- the customer is very angry
- no policy document matches the request
- a legal threat, possible fraud or a payment dispute
- an account security or safety risk
- confidence below 0.5
- the reply failed the grounding check
- the Log Agent quotes a concern such as a repeated complaint or an injury
"""
    )

    st.subheader("Two views: customers and the support team")
    st.markdown(
        """
| Customer portal | Support team |
|---|---|
| Chat in any language, by typing or recording a voice message (OpenAI speech to text, with the text shown for checking before sending) | **New ticket**: the live timeline of all four agents |
| Attach a photo; the AI describes it and checks it matches the complaint | **Case management**: approve or edit escalated replies, set the case status, message the customer |
| A progress bar shows which agent is working (1/4 to 4/4) | **Ticket log** with filters and CSV / Excel downloads |
| **My tickets**: status of every ticket and replies approved by a specialist | **Dashboard** with charts and the AI cost per ticket |
| | **Knowledge base**: read and upload policy documents, rebuild the ChromaDB index, test the search |
"""
    )

    st.subheader("Technology")
    st.markdown(
        """
| Layer | Tool | Used for |
|---|---|---|
| Agents | AutoGen AgentChat 0.7 | `AssistantAgent`, `RoundRobinGroupChat`, `TextMentionTermination`, `MaxMessageTermination`, tools, structured output |
| Language model | OpenAI `gpt-4o-mini` | Understanding complaints and writing replies |
| Vector store | ChromaDB (persistent, `chroma_db/`) + OpenAI `text-embedding-3-small` | Storing the 34 passage vectors and the nearest-neighbour search |
| Search scoring | Hybrid: 80 % meaning + 20 % exact keywords | Finding the right policy passage; `NO_MATCH` below 0.33 |
| Voice and photos | OpenAI speech to text, `gpt-4o-mini` vision | Voice messages in any language, photo checks |
| Cost tracking | Token counts from every agent call | Cost per ticket and per 1,000 tickets on the Dashboard |
| Data checks | Pydantic | Validating every agent output |
| Logging | pandas, openpyxl | `tickets.csv` and `tickets.xlsx`, one row per ticket |
| Interface | Streamlit, Altair | This app, live agent timeline, dashboard |
"""
    )

    st.subheader("Tested before release")
    st.markdown(
        """
Nine test complaints (six from the assignment, three of our own: a hacked account, a
complaint in Tamil and a calm address change) run through the full team with
`python tests/try_team.py`. All nine get the expected Resolved or Escalated status.
The `NO_MATCH` threshold was measured, not guessed: the weakest correct match scored
0.41 and the corporate gift-card question scored 0.25, so the threshold sits at the midpoint.
"""
    )

    with st.expander("Project structure"):
        st.code(
            """customer-support-agents/
├── app.py                  Streamlit interface (this app)
├── team.py                 the four agents as one AutoGen team
├── config.py               settings, API key loading, model client
├── agents/
│   ├── query_agent.py      Agent 1 + QueryAnalysis model
│   ├── knowledge_agent.py  Agent 2 + search_kb tool
│   ├── kb_search.py        chunking, embeddings, hybrid search
│   ├── resolution_agent.py Agent 3 + grounding check + confidence
│   ├── log_agent.py        Agent 4 + log_ticket tool
│   ├── escalation.py       escalation rules
│   └── ticket_logger.py    CSV / Excel log
├── ui/                     theme and this About page
├── knowledge_base/         6 BluePeak Store policy documents
├── tests/                  test cases and test scripts
└── logs/                   tickets.csv, tickets.xlsx (created automatically)""",
            language=None,
        )
