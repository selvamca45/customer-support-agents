"""
ui/theme.py - the look of the app: colours, fonts, top navigation bar, hero and footer.

Design idea: the app is a support desk, so the hero shows a support TICKET STUB
(perforated edge, punched notches) carrying the four-agent route every complaint takes.
Colours: ink #12355b, glacier blue #2f6fb2, snow #f6f8fb; green and red are reserved
for Resolved / Escalated only.

Fonts: the system font stack (no Google Fonts download, which company networks can block).
Note: Streamlit styles <p> tags with its own CSS, so the header/hero/footer use <div>s
and a ".bp" prefix with !important on key type settings, so Streamlit cannot override them.
"""
import html

INK = "#12355b"
BLUE = "#2f6fb2"
SNOW = "#f6f8fb"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
GITHUB_URL = "https://github.com/selvamca45/customer-support-agents"
TEXT = "#0f1c2e"     # body text: 16:1 contrast on white
MUTED = "#46566c"    # secondary text: 7:1 contrast on white
FONT = ('-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Inter, Roboto, '
        '"Helvetica Neue", Arial, sans-serif, '
        '"Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"')   # emoji fonts last

LOGO_SVG = f"""<svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">
<rect width="36" height="36" rx="10" fill="{INK}"/>
<path d="M6 27 L15 13 L19.5 19 L23 14.5 L30 27 Z" fill="#7fb0e6"/>
<path d="M15 13 L17.8 17.4 L15.2 16.4 L12.6 17 Z" fill="#ffffff"/>
</svg>"""

CSS = f"""
<style>
/* System fonts: always installed, very legible, no download needed
   (San Francisco on Mac/iPhone, Segoe UI on Windows, Inter/Roboto elsewhere) */
.stApp, .stApp p, .stApp li, .stApp td, .stApp th, .stApp label, .stApp input,
.stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-baseweb="select"] div, .stApp [data-testid="stMetricValue"] div,
.reply-box, .panel, .badge, .empty, .stApp [data-testid="stTable"] *,
.stApp [data-testid="stMetricLabel"] p, .bp, .bp * {{
  font-family: {FONT} !important;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}}
.stApp {{ color: {TEXT}; }}

/* ---------- readable Streamlit widgets ---------- */
[data-testid="stWidgetLabel"] p {{ color: {TEXT} !important; font-weight: 600 !important;
                                   font-size: 0.97rem !important; }}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{
  background: #ffffff !important; border: 1.5px solid #b3c2d6 !important; border-radius: 10px !important;
}}
/* the inner boxes carry their own grey-blue background: make them white too */
[data-baseweb="input"] > div, [data-baseweb="base-input"], [data-baseweb="textarea"] > div,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div > div {{ background: #ffffff !important; }}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {{ border-color: {BLUE} !important;
  box-shadow: 0 0 0 3px rgba(47,111,178,0.18) !important; }}
.stApp input, .stApp textarea, [data-baseweb="select"] div {{ color: {TEXT} !important;
  font-size: 1rem !important; }}
.stApp input::placeholder, .stApp textarea::placeholder {{ color: #66768c !important; opacity: 1; }}
[data-testid="stForm"] {{ background: #ffffff; border: 1px solid #d5deea !important;
                          border-radius: 14px; padding: 20px 22px 16px; }}
.stApp [data-testid="stFormSubmitButton"] button, .stApp button[kind="primary"] {{
  background: {INK} !important; border: none !important; color: #fff !important;
  font-weight: 700 !important; font-size: 1rem !important; padding: 0.55rem 1.3rem !important;
  border-radius: 10px !important; }}
.stApp [data-testid="stFormSubmitButton"] button:hover, .stApp button[kind="primary"]:hover {{
  background: {BLUE} !important; }}
.stApp [data-testid="stFormSubmitButton"] button p, .stApp button[kind="primary"] p {{ color: #fff !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important;
  font-size: 0.93rem !important; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-weight: 600 !important; }}
[data-testid="stMetricValue"] div {{ color: {INK} !important; font-weight: 800 !important; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size: 1rem; line-height: 1.6; }}
.stApp h2, .stApp h3 {{ color: {INK} !important; font-weight: 800 !important; letter-spacing: -0.01em; }}
[data-testid="stExpander"] summary p {{ color: {TEXT} !important; font-weight: 600 !important; }}

/* A website, not a dev tool: no Streamlit header, toolbar, sidebar or footer */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stAppDeployButton"],
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{
  display: none !important;
}}
.block-container {{ padding-top: 0 !important; padding-bottom: 2rem; max-width: 1240px; }}

/* ---------- top navigation bar ---------- */
.bp-nav {{
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 16px 2px; margin-bottom: 18px; border-bottom: 1px solid #dfe6ef; flex-wrap: wrap;
}}
.bp-brand {{ display: flex; align-items: center; gap: 12px; text-decoration: none; }}
.bp-wordmark {{ line-height: 1.15; }}
.bp-wordmark .name {{ font-size: 1.15rem !important; font-weight: 800 !important; color: {INK};
                       letter-spacing: -0.01em; }}
.bp-wordmark .sub {{ font-size: 0.86rem !important; color: {MUTED}; font-weight: 500; }}
.bp-navright {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.bp-pill {{ display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px;
            border-radius: 999px; font-size: 0.85rem !important; font-weight: 600;
            background: #fff; border: 1px solid #dfe6ef; color: {TEXT}; }}
.bp-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {GOOD};
           box-shadow: 0 0 0 3px rgba(12,163,12,0.15); }}
a.bp-btn {{ padding: 7px 14px; border-radius: 9px; background: {INK}; color: #fff !important;
            font-size: 0.86rem !important; font-weight: 700; text-decoration: none !important; }}
a.bp-btn:hover {{ background: {BLUE}; }}
a.bp-btn:focus-visible {{ outline: 3px solid #7fb0e6; outline-offset: 2px; }}

/* ---------- hero ---------- */
.bp-hero {{
  background: {INK}; color: #fff; border-radius: 20px; padding: 40px 42px;
  display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(280px, 1fr);
  gap: 40px; align-items: center; margin-bottom: 22px;
}}
.bp-hero .kicker {{ font-size: 1rem !important; font-weight: 600; color: #bcd6f3; margin-bottom: 12px; }}
.bp-hero .headline {{ font-size: 2.6rem !important; line-height: 1.1 !important; font-weight: 800 !important;
                      letter-spacing: -0.025em; color: #fff; margin-bottom: 16px; }}
.bp-hero .lead {{ font-size: 1.12rem !important; line-height: 1.6 !important; color: #eaf1fa;
                  max-width: 36em; }}
.bp-stats {{ display: flex; gap: 40px; margin-top: 28px; flex-wrap: wrap; }}
.bp-stats .num {{ font-size: 2rem !important; font-weight: 800; line-height: 1.05; color: #fff;
                  white-space: nowrap; }}
.bp-stats .lbl {{ font-size: 0.92rem !important; color: #c7dcf3; margin-top: 4px; white-space: nowrap; }}

.bp-stub {{
  color: {INK}; border-radius: 16px; padding: 22px 24px 20px 36px; position: relative;
  background:
    radial-gradient(circle at 0 26px, {INK} 10px, transparent 11px),
    radial-gradient(circle at 0 calc(100% - 26px), {INK} 10px, transparent 11px),
    {SNOW};
}}
.bp-stub::before {{ content: ""; position: absolute; left: 15px; top: 16px; bottom: 16px;
                    border-left: 2px dashed #b8c6d8; }}
.bp-stub .head {{ display: flex; justify-content: space-between; font-size: 0.92rem !important;
                  font-weight: 700; margin-bottom: 12px; }}
.bp-stub .head span:last-child {{ color: {MUTED}; font-weight: 600; }}
.bp-route {{ display: grid; gap: 10px; }}
.bp-route .row {{ display: grid; grid-template-columns: 26px 1fr; gap: 8px; align-items: start; }}
.bp-route .n {{ width: 22px; height: 22px; border-radius: 50%; background: {INK}; color: #fff;
                font-size: 0.75rem !important; font-weight: 700; display: grid; place-items: center;
                margin-top: 1px; }}
.bp-route .t {{ font-size: 1rem !important; font-weight: 700; color: {INK}; line-height: 1.3; }}
.bp-route .d {{ font-size: 0.92rem !important; color: {MUTED}; line-height: 1.4; }}
.bp-stub .foot {{ border-top: 1px solid #dbe3ee; margin-top: 14px; padding-top: 10px;
                  font-size: 0.92rem !important; color: {MUTED}; }}
.bp-stub .foot b {{ color: {GOOD}; }} .bp-stub .foot i {{ color: {CRITICAL}; font-style: normal; font-weight: 700; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #dfe6ef; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.02rem !important; font-weight: 700 !important; color: {MUTED}; }}
.stTabs [aria-selected="true"] p {{ color: {INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {INK} !important; height: 3px !important; }}

/* ---------- panels and results ---------- */
.panel {{ background: #fff; border: 1px solid #d5deea; border-radius: 12px; padding: 18px 20px; }}
.panel h4 {{ margin: 0 0 8px; color: {INK}; font-size: 1.08rem; font-weight: 800; }}
.panel p, .panel li {{ color: {TEXT}; font-size: 0.98rem; line-height: 1.6; }}
.panel ol, .panel ul {{ margin: 4px 0 0 18px; padding: 0; }}
.guide {{ background: #eef4fb; border-color: #cfdcec; }}

.badge {{ display: inline-block; padding: 4px 14px; border-radius: 999px; font-weight: 700;
          font-size: 0.95rem; color: #fff; vertical-align: middle; margin-left: 8px; }}
.badge-resolved {{ background: {GOOD}; }}
.badge-escalated {{ background: {CRITICAL}; }}
.reply-box {{ border-left: 6px solid {GOOD}; background: rgba(12,163,12,0.06);
              padding: 16px 20px; border-radius: 8px; line-height: 1.7; color: {TEXT}; font-size: 1.02rem; }}
.reply-box.escalated {{ border-left-color: {CRITICAL}; background: rgba(208,59,59,0.06); }}
.step-head {{ font-weight: 700; color: {INK}; }}
.muted {{ color: {MUTED}; font-size: 0.92rem; }}
.empty {{ border: 2px dashed #cfdcec; border-radius: 12px; padding: 26px; text-align: center;
          color: {MUTED}; font-size: 1rem; }}

/* ---------- footer ---------- */
.bp-footer {{ margin-top: 40px; padding: 22px 2px 8px; border-top: 1px solid #dfe6ef;
              display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
              color: {MUTED}; font-size: 0.92rem !important; }}
.bp-footer a {{ color: {BLUE} !important; font-weight: 600; text-decoration: none; }}
.bp-footer b {{ color: {INK}; font-weight: 700; }}

@media (max-width: 860px) {{
  .bp-hero {{ grid-template-columns: 1fr; padding: 26px; gap: 24px; }}
  .bp-hero .headline {{ font-size: 1.85rem !important; }}
  .bp-stats {{ gap: 24px; }}
}}
</style>
"""


def nav_html(model: str, kb_summary: str) -> str:
    """Top navigation bar: logo + name on the left, live status on the right."""
    return f"""<div class="bp bp-nav">
<div class="bp-brand">{LOGO_SVG}<div class="bp-wordmark"><div class="name">BluePeak Support Desk</div><div class="sub">AI customer support automation</div></div></div>
<div class="bp-navright">
<span class="bp-pill"><span class="bp-dot"></span>4 agents ready</span>
<span class="bp-pill">{html.escape(kb_summary)}</span>
<span class="bp-pill">Model: {html.escape(model)}</span>
<a class="bp-btn" href="{GITHUB_URL}" target="_blank" rel="noopener">View code on GitHub</a>
</div>
</div>"""


def hero_html(total: int, automation: str, avg_seconds: str) -> str:
    """Purpose on the left, the ticket stub with the agent route on the right."""
    steps = [
        ("Query Agent", "understands the problem, urgency and mood"),
        ("Knowledge Agent", "finds the matching store policy"),
        ("Resolution Agent", "writes a reply using only that policy"),
        ("Log Agent", "records the ticket and decides on escalation"),
    ]
    rows = "".join(
        f'<div class="row"><div class="n">{i}</div><div><div class="t">{t}</div>'
        f'<div class="d">{d}</div></div></div>'
        for i, (t, d) in enumerate(steps, 1)
    )
    return f"""<div class="bp bp-hero">
<div>
<div class="kicker">For BluePeak Store's customer support team</div>
<div class="headline">Every complaint answered in seconds. Every hard case sent to a person.</div>
<div class="lead">Four AI agents read each complaint, look up the answer in the store's own policies, reply in the customer's language and hand risky cases to the support team, with every ticket recorded.</div>
<div class="bp-stats">
<div><div class="num">{html.escape(str(total))}</div><div class="lbl">tickets handled</div></div>
<div><div class="num">{html.escape(automation)}</div><div class="lbl">resolved without a person</div></div>
<div><div class="num">{html.escape(avg_seconds)}</div><div class="lbl">average time per ticket</div></div>
</div>
</div>
<div class="bp-stub">
<div class="head"><span>Route of a ticket</span><span>4 agents</span></div>
<div class="bp-route">{rows}</div>
<div class="foot">Ends as <b>Resolved</b>, or <i>Escalated</i> to a person with the reason.</div>
</div>
</div>"""


def footer_html() -> str:
    return f"""<div class="bp bp-footer">
<div>Developed by <b>Selvakumar</b>, Social Eagle AI Engineering Program. Built with AutoGen, OpenAI and Streamlit.</div>
<div><a href="{GITHUB_URL}" target="_blank" rel="noopener">Source code on GitHub</a></div>
</div>"""
