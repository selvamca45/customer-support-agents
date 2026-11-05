"""
ui/theme.py - the look of the app: colours, fonts and the header ("hero").

Design idea: the app is a support desk, so the header shows a support TICKET STUB
(with a perforated edge) carrying the four-agent route every complaint takes.
Colours: ink #12355b, glacier blue #2f6fb2, snow #f6f8fb; green and red are
reserved for Resolved / Escalated only.
"""
import html

INK = "#12355b"
BLUE = "#2f6fb2"
SNOW = "#f6f8fb"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

/* Font only on text elements (not on icons) */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th,
.hero, .panel, .reply-box, .stTabs button p {{
  font-family: 'Manrope', system-ui, sans-serif;
}}

/* Hide Streamlit's developer chrome for a product look */
[data-testid="stAppDeployButton"], #MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1240px; }}

/* ---------- hero ---------- */
.hero {{
  background: {INK}; color: #fff; border-radius: 18px;
  padding: 30px 34px; display: flex; gap: 36px; align-items: stretch;
  flex-wrap: wrap; margin-bottom: 14px;
}}
.hero-main {{ flex: 1 1 420px; min-width: 300px; }}
.hero-name {{ font-size: 0.95rem; font-weight: 600; color: #b9cde6; margin: 0 0 6px; }}
.hero-title {{ font-size: 2.15rem; line-height: 1.15; font-weight: 800;
               letter-spacing: -0.02em; margin: 0 0 12px; color: #fff; }}
.hero-lead {{ font-size: 1.04rem; line-height: 1.6; color: #dbe6f3;
              max-width: 34em; margin: 0; }}
.hero-stats {{ display: flex; gap: 34px; margin-top: 22px; flex-wrap: wrap; }}
.hero-stats .num {{ font-size: 1.7rem; font-weight: 800; color: #fff; line-height: 1.1; }}
.hero-stats .lbl {{ font-size: 0.85rem; color: #b9cde6; }}

/* the ticket stub */
.stub {{
  flex: 0 1 330px; min-width: 270px; color: {INK};
  /* two punched notches on the perforated edge make it read as a ticket */
  background:
    radial-gradient(circle at 0 22px, {INK} 9px, transparent 10px),
    radial-gradient(circle at 0 calc(100% - 22px), {INK} 9px, transparent 10px),
    {SNOW};
  border-radius: 14px; padding: 18px 20px 18px 30px; position: relative;
}}
.stub::before {{
  content: ""; position: absolute; left: 12px; top: 14px; bottom: 14px;
  border-left: 2px dashed #b8c6d8;
}}
.stub-head {{ display: flex; justify-content: space-between; font-weight: 700;
              font-size: 0.9rem; margin-bottom: 10px; }}
.stub-head span:last-child {{ color: #6b7a90; font-weight: 600; }}
.stub ol {{ margin: 0; padding-left: 20px; }}
.stub li {{ font-size: 0.92rem; line-height: 1.45; margin-bottom: 7px; }}
.stub li b {{ color: {INK}; }}
.stub li span {{ color: #5b6b80; }}
.stub-foot {{ border-top: 1px solid #dbe3ee; margin-top: 10px; padding-top: 9px;
              font-size: 0.85rem; color: #5b6b80; }}

/* ---------- panels and results ---------- */
.panel {{ background: #fff; border: 1px solid #dfe6ef; border-radius: 12px;
          padding: 18px 20px; }}
.panel h4 {{ margin: 0 0 8px; color: {INK}; font-size: 1.02rem; }}
.panel p, .panel li {{ color: #44546a; font-size: 0.93rem; line-height: 1.55; }}
.panel ol, .panel ul {{ margin: 4px 0 0 18px; padding: 0; }}
.guide {{ background: #eef4fb; border-color: #cfdcec; }}

.badge {{ display: inline-block; padding: 4px 14px; border-radius: 999px; font-weight: 700;
          font-size: 0.95rem; color: #fff; vertical-align: middle; margin-left: 8px; }}
.badge-resolved {{ background: {GOOD}; }}
.badge-escalated {{ background: {CRITICAL}; }}
.reply-box {{ border-left: 6px solid {GOOD}; background: rgba(12,163,12,0.06);
              padding: 16px 20px; border-radius: 8px; line-height: 1.65; color: #1f2a37; }}
.reply-box.escalated {{ border-left-color: {CRITICAL}; background: rgba(208,59,59,0.06); }}
.step-head {{ font-weight: 700; color: {INK}; }}
.muted {{ color: #6b7a90; font-size: 0.86rem; }}
.empty {{ border: 2px dashed #cfdcec; border-radius: 12px; padding: 26px; text-align: center;
          color: #5b6b80; }}

@media (max-width: 700px) {{
  .hero {{ padding: 22px; }} .hero-title {{ font-size: 1.6rem; }}
}}
</style>
"""


def hero_html(total: int, automation: str, avg_seconds: str) -> str:
    """The header band: purpose on the left, the ticket stub on the right."""
    return f"""
<div class="hero">
  <div class="hero-main">
    <p class="hero-name">BluePeak Store customer support</p>
    <p class="hero-title">Every complaint answered in seconds, every hard case sent to a person.</p>
    <p class="hero-lead">Four AI agents read each complaint, look up the answer in the store's own
    policies, write a reply in the customer's language and hand risky cases to the support team.
    Every ticket is recorded so the team can learn from it.</p>
    <div class="hero-stats">
      <div><div class="num">{html.escape(str(total))}</div><div class="lbl">tickets handled</div></div>
      <div><div class="num">{html.escape(automation)}</div><div class="lbl">resolved without a person</div></div>
      <div><div class="num">{html.escape(avg_seconds)}</div><div class="lbl">average time per ticket</div></div>
    </div>
  </div>
  <div class="stub">
    <div class="stub-head"><span>Route of a ticket</span><span>4 agents</span></div>
    <ol>
      <li><b>Query Agent</b><br><span>understands the problem, urgency and mood</span></li>
      <li><b>Knowledge Agent</b><br><span>finds the matching store policy</span></li>
      <li><b>Resolution Agent</b><br><span>writes a reply using only that policy</span></li>
      <li><b>Log Agent</b><br><span>records the ticket and decides on escalation</span></li>
    </ol>
    <div class="stub-foot">Resolved, or escalated to a person with the reason.</div>
  </div>
</div>
"""
