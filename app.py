
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Migrant Health Risk-Tier Segmentation", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Plotly is not installed. In the black window run:  python -m pip install plotly   then refresh this page.")
    st.stop()

from sklearn.decomposition import PCA

from common import FEATURES, RISK_SIGNS, score_workers
from generate_data import HOUSING, OCC, generate

# ======================================================================= STYLE
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root {
 --bg:#080d12; --panel:#111a21; --panel2:#0d151c; --text:#e8f0ef;
 --muted:#81918f; --line:rgba(143,178,171,.14); --mint:#55d6b0;
 --gold:#d7bd82; --champ:#e9d8ad; --ivory:#f0f3ed;
}
html, body, .stApp, [class*="css"] { font-family:'DM Sans',sans-serif !important; }
.stApp, [data-testid="stAppViewContainer"], .main {
 background:
  radial-gradient(900px 520px at 92% -8%, rgba(64,190,157,.10), transparent 62%),
  radial-gradient(800px 520px at -5% 0%, rgba(64,111,130,.13), transparent 62%),
  var(--bg) !important;
 color:var(--text);
}
#MainMenu, footer, header[data-testid="stHeader"], .stDeployButton { visibility:hidden; height:0; }
.block-container { padding-top:1.1rem; padding-bottom:3rem; max-width:1600px; }
[data-testid="stAppViewContainer"] > .main .block-container { animation:pageIn .7s ease both; }
@keyframes pageIn { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }
@keyframes fadeUp { from{opacity:0; transform:translateY(16px)} to{opacity:1; transform:none} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(85,214,176,.5)} 70%{box-shadow:0 0 0 8px rgba(85,214,176,0)} 100%{box-shadow:0 0 0 0 rgba(85,214,176,0)} }

/* Navigation */
.stTabs [data-baseweb="tab-list"] {
 gap:8px; border-bottom:1px solid var(--line); padding:0 0 10px; background:transparent;
}
.stTabs [data-baseweb="tab"] {
 height:42px; padding:0 17px; border-radius:9px; background:rgba(255,255,255,.025);
 color:var(--muted); font-weight:600; font-size:12px; letter-spacing:.07em;
 text-transform:uppercase; border:1px solid transparent; transition:all .25s;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--text); background:rgba(85,214,176,.06); }
.stTabs [aria-selected="true"] {
 color:#8be6ca !important; background:rgba(85,214,176,.10) !important;
 border:1px solid rgba(85,214,176,.22) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background:transparent !important; }
.stTabs [data-baseweb="tab-border"] { background:transparent; }

/* Controls */
.stButton > button, .stDownloadButton > button {
 width:100%; background:linear-gradient(135deg,rgba(85,214,176,.13),rgba(85,214,176,.04));
 color:#a4efd9; border:1px solid rgba(85,214,176,.28); border-radius:9px;
 padding:.58rem 1rem; font-weight:600; font-size:13px; transition:all .25s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
 background:#55d6b0; color:#07110f; border-color:#55d6b0;
 box-shadow:0 8px 26px rgba(85,214,176,.18); transform:translateY(-2px);
}
[data-baseweb="select"] > div, [data-testid="stFileUploader"] section {
 background:var(--panel2); border-color:var(--line); border-radius:10px;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#55d6b0; }
.stCheckbox label, .stRadio label { color:var(--text) !important; }

/* Cards */
.stat, .glass, div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
 background:linear-gradient(145deg,rgba(20,32,39,.96),rgba(13,21,28,.96));
 border:1px solid var(--line); border-radius:16px;
 box-shadow:0 16px 38px rgba(0,0,0,.24);
 animation:fadeUp .65s ease both;
 transition:transform .28s,border-color .28s,box-shadow .28s;
}
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] { padding:10px; }
.stat, .glass { padding:18px 22px; }
.stat:hover, .glass:hover, div[data-testid="stPlotlyChart"]:hover {
 transform:translateY(-3px); border-color:rgba(85,214,176,.28);
 box-shadow:0 20px 44px rgba(0,0,0,.32);
}
.stat .l { color:var(--muted); font-size:10.5px; letter-spacing:.15em; text-transform:uppercase; font-weight:700; }
.stat .v { font-family:'Playfair Display',serif; font-size:35px; font-weight:600; margin:8px 0 2px; color:#f1f5ef; }
.stat .s { color:var(--muted); font-size:12px; }

/* Section headers */
.sec { position:relative; display:flex; align-items:baseline; gap:14px; margin:38px 0 16px; padding-bottom:12px;
 border-bottom:1px solid var(--line); animation:fadeUp .65s ease both; }
.sec::after { content:""; position:absolute; left:0; bottom:-1px; height:2px; width:58px; background:linear-gradient(90deg,var(--mint),transparent); }
.sec h3 { margin:0; font-family:'Playfair Display',serif; font-size:25px; font-weight:600; color:#edf2eb; }
.sec small { color:var(--muted); font-size:12px; }

/* Risk tier result */
.tier-badge { font-family:'Playfair Display',serif; font-size:68px; font-weight:700; line-height:1.05; color:var(--c);
 text-shadow:0 0 34px color-mix(in srgb,var(--c) 35%,transparent); animation:fadeUp .65s ease both; margin:6px 0; }
.chip { display:inline-block; margin:6px 8px 0 0; padding:7px 14px; border-radius:999px; font-size:12px;
 color:#dce9e4; background:rgba(85,214,176,.07); border:1px solid rgba(85,214,176,.18); animation:fadeUp .6s both; }
.eyebrow { color:var(--muted); font-size:10px; letter-spacing:.18em; text-transform:uppercase; font-weight:700; }
.live { display:inline-flex; align-items:center; gap:9px; padding:6px 13px; border-radius:999px; font-weight:700;
 font-size:11px; letter-spacing:.13em; color:#8be6ca; background:rgba(85,214,176,.08); border:1px solid rgba(85,214,176,.25); }
.live i { width:7px; height:7px; border-radius:50%; background:#55d6b0; animation:pulse 1.7s infinite; }

/* Process steps and tech chips */
.step { display:flex; gap:18px; align-items:flex-start; background:linear-gradient(145deg,#142129,#101920);
 border:1px solid var(--line); border-radius:14px; padding:19px 22px; margin-bottom:13px;
 animation:fadeUp .7s both; transition:transform .25s,border-color .25s; }
.step:hover { transform:translateX(4px); border-color:rgba(85,214,176,.28); }
.step .n { flex:0 0 44px; height:44px; border-radius:13px; display:flex; align-items:center; justify-content:center;
 font-family:'Playfair Display',serif; font-weight:700; font-size:19px; color:#8be6ca;
 background:rgba(85,214,176,.08); border:1px solid rgba(85,214,176,.28); }
.step h4 { margin:0 0 5px; color:#edf2eb; font-family:'Playfair Display',serif; font-size:17px; font-weight:600; }
.step p { margin:0; color:#9eafaa; font-size:13px; line-height:1.7; }
.tech span { display:inline-block; margin:6px 8px 0 0; padding:7px 15px; border-radius:8px; font-weight:600;
 font-size:12px; background:rgba(85,214,176,.05); border:1px solid rgba(85,214,176,.17); color:#a4efd9; }
.foot { text-align:center; color:var(--muted); font-size:11.5px; margin:42px 0 10px; }
[data-testid="stMetric"] { background:rgba(255,255,255,.025); border:1px solid var(--line); border-radius:12px; padding:12px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
hr { border-color:var(--line); }

/* Sidebar navigation styled like a dedicated product rail */
[data-testid="stSidebar"] {background:#0d1419 !important;border-right:1px solid rgba(150,180,174,.12);min-width:238px !important}
[data-testid="stSidebar"] > div:first-child {padding:1.25rem .85rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:#dce8e3}
.side-brand{display:flex;align-items:center;gap:10px;padding:9px 7px 28px;border-bottom:1px solid rgba(255,255,255,.07)}
.brand-mark{width:34px;height:34px;border-radius:10px;background:linear-gradient(145deg,#55d6b0,#247e70);display:grid;place-items:center;color:#071411;font-size:20px;font-weight:800}
.side-brand b{font-size:10px;letter-spacing:.12em;color:#eaf2ed}.side-brand b span{color:#55d6b0}
.side-brand small{display:block;font-size:8px;letter-spacing:.14em;color:#72847f;margin-top:4px}
.side-label{font-size:9px;letter-spacing:.16em;color:#71817d;font-weight:700;margin:25px 8px 9px}
.side-item{font-size:12px;color:#91a09b;padding:10px 11px;border-radius:8px;margin:3px 0;transition:all .2s}
.side-item.active{color:#83e6c9;background:rgba(85,214,176,.12);border-left:2px solid #55d6b0}
.side-item:hover{color:#e8f2ed;background:rgba(255,255,255,.045)}
.side-profile{display:flex;align-items:center;gap:9px;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:10px;margin-top:38px}
.side-profile>span{width:28px;height:28px;display:grid;place-items:center;border-radius:50%;background:#2c8f83;color:white;font-size:10px;font-weight:700}
.side-profile b{font-size:10px;color:#dce8e3}.side-profile small{display:block;color:#71817d;font-size:9px;margin-top:3px}
.app-header{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:4px 0 15px;padding:4px 2px 15px;border-bottom:1px solid rgba(150,180,174,.12);animation:fadeUp .65s ease both}
.app-header .eyebrow{font-size:9px;letter-spacing:.17em;color:#7d918b;font-weight:700}
.app-header h1{font:600 27px 'Playfair Display',serif;color:#edf3ee;margin:8px 0 3px}
.app-header h1 em{font-style:normal;color:#79dfc0}
.app-header p{font-size:12px;color:#879792;margin:0}
.header-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px}
.header-right small{font-size:9px;letter-spacing:.12em;color:#72827e}
.stTabs [data-baseweb="tab-list"]{gap:5px;border-bottom:1px solid var(--line);padding:0 0 8px}
.stTabs [data-baseweb="tab"]{height:38px;padding:0 13px;border-radius:7px;font-size:10px;letter-spacing:.04em;text-transform:none}

</style>
"""

def html(s):
    """Render HTML (strip indentation so markdown does not treat it as a code block)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


html(CSS)


# =================================================================== LOAD DATA
@st.cache_resource
def load_bundle():
    return joblib.load("models/bundle.joblib")


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("Model not found. Run `python generate_data.py` and then `python train.py` first.")
    st.stop()

K = bundle["k"]
NAMES = bundle["tier_names"]
PAL = {3: ["#55d6b0", "#e9b96e", "#ed777d"], 4: ["#55d6b0", "#e9b96e", "#e58e65", "#d75e70"],
       5: ["#79b8d2", "#55d6b0", "#e9b96e", "#e58e65", "#d75e70"]}
COLORS = PAL.get(K, px.colors.sequential.YlOrRd[:K])
CMAP = dict(zip(NAMES, COLORS))

scored = load_csv("outputs/workers_scored.csv")
inc = load_csv("outputs/incidence_by_tier.csv").set_index("risk_tier").reindex(NAMES)
prof = load_csv("outputs/tier_profile.csv").set_index("risk_tier").reindex(NAMES)
base = load_csv("outputs/baseline_comparison.csv")
kscan = load_csv("outputs/k_selection.csv")
with open("outputs/validation.json") as f:
    val = json.load(f)
counts = scored.risk_tier.value_counts().reindex(NAMES)


def pchart(fig):
    """plotly chart at full width; works on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.plotly_chart(fig, width="stretch")


def dframe(df, **kw):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)
    except Exception:
        st.dataframe(df, width="stretch", hide_index=True, **kw)


def style_fig(fig, h=380, title=None):
    kw = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
              font=dict(family="Inter, sans-serif", color="#b8c4d9", size=12), margin=dict(l=20, r=20, t=58 if title else 20, b=20),
              legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor="#080d12", font_size=12, bordercolor="#55d6b0"))
    if title:
        kw["title"] = dict(text=title, font=dict(size=18, color="#edf2eb", family="Playfair Display, serif"), x=0.03)
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)")
    return fig


def section(title, sub=""):
    html(f"<div class='sec'><h3>{title}</h3><small>{sub}</small></div>")


def stat(label, value, sub=""):
    return f"<div class='stat'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>"



# =============================================================== APP SHELL
# A compact clinical-analytics header replaces the oversized landing hero.
st.sidebar.markdown("""
<div class="side-brand">
  <div class="brand-mark">✚</div>
  <div><b>HEALTH<span> INTELLIGENCE</span></b><small>RISK ANALYTICS PLATFORM</small></div>
</div>
<div class="side-label">OVERVIEW</div>
<div class="side-item active">▦ &nbsp; Dashboard</div>
<div class="side-item">♙ &nbsp; Worker Records</div>
<div class="side-item">♡ &nbsp; Risk Tiers</div>
<div class="side-label">ANALYTICS</div>
<div class="side-item">⌁ &nbsp; Health Indicators</div>
<div class="side-item">◉ &nbsp; Model Insights</div>
<div class="side-item">⌖ &nbsp; Priority Outreach</div>
<div class="side-label">OPERATIONS</div>
<div class="side-item">◷ &nbsp; Live Monitor</div>
<div class="side-item">⚙ &nbsp; Settings</div>
<div class="side-profile"><span>DR</span><div><b>Risk Analytics</b><small>Model workspace</small></div></div>
""", unsafe_allow_html=True)

header_html = """
<div class="app-header">
  <div>
    <div class="eyebrow">HEALTH INTELLIGENCE / WORKER RISK MONITORING</div>
    <h1><em>Migrant Worker Health Risk-Tier Segmentation Using K-Means Clustering</em></h1>
    <p>migrant-worker health screening and risk tiers.</p>
  </div>
  <div class="header-right"><span class="live"><i></i> MODEL MONITORING</span><small>UNSUPERVISED LEARNING · K-MEANS</small></div>
</div>
"""
html(header_html)

tier_data = [dict(name=n, color=c, pct=round(100 * counts[n] / len(scored), 1), value=int(counts[n]))
             for n, c in zip(NAMES, COLORS)]
kpi_data = [
    dict(label="Workers analysed", value=int(len(scored)), pct=100, color="#55d6b0", sub="screening records"),
    dict(label="Low risk", value=int(counts.iloc[0]), pct=round(100 * counts.iloc[0] / len(scored), 1), color=COLORS[0], sub=f"{100 * counts.iloc[0] / len(scored):.1f}% of workers"),
    dict(label="Medium risk", value=int(counts.iloc[min(1, len(counts)-1)]), pct=round(100 * counts.iloc[min(1, len(counts)-1)] / len(scored), 1), color=COLORS[min(1, len(COLORS)-1)], sub=f"{100 * counts.iloc[min(1, len(counts)-1)] / len(scored):.1f}% of workers"),
    dict(label="High risk", value=int(counts.iloc[-1]), pct=round(100 * counts.iloc[-1] / len(scored), 1), color=COLORS[-1], sub=f"{100 * counts.iloc[-1] / len(scored):.1f}% of workers"),
]
KPI = """
<html><head><style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
*{box-sizing:border-box}html,body{margin:0;background:transparent;overflow:hidden;font-family:'DM Sans',sans-serif}
#r{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;padding:5px 1px 9px}
.k{position:relative;min-height:118px;background:linear-gradient(145deg,#151e24,#10171d);border:1px solid rgba(150,180,174,.13);border-top:2px solid var(--c);border-radius:13px;padding:15px 17px;box-shadow:0 12px 28px rgba(0,0,0,.22);opacity:0;animation:rise .7s ease forwards;transition:transform .25s,border-color .25s,box-shadow .25s}
.k:hover{transform:translateY(-4px);border-color:var(--c);box-shadow:0 18px 34px rgba(0,0,0,.32)}
@keyframes rise{from{opacity:0;transform:translateY(15px)}to{opacity:1;transform:translateY(0)}}
.l{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#899a96;font-weight:700}
.v{font:600 31px 'Playfair Display',serif;color:#eef3ef;margin:8px 0 1px}
.s{font-size:11px;color:#82928e}.bar{height:3px;background:rgba(255,255,255,.08);margin-top:10px;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--c);animation:fill 1.2s ease .4s forwards}
@keyframes fill{to{width:var(--w)}}
@media(max-width:700px){#r{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><div id="r"></div><script>
const D=__DATA__,root=document.getElementById('r');
D.forEach((d,i)=>{const e=document.createElement('div');e.className='k';e.style.animationDelay=(i*.1)+'s';e.style.setProperty('--c',d.color);e.style.setProperty('--w',Math.max(3,d.pct)+'%');
e.innerHTML='<div class="l">'+d.label+'</div><div class="v">0</div><div class="s">'+d.sub+'</div><div class="bar"><i></i></div>';root.appendChild(e);
const v=e.querySelector('.v'),t=performance.now()+i*120;function tick(now){let q=Math.min(1,(now-t)/1050);q=1-Math.pow(1-q,3);v.textContent=Math.round(d.value*q).toLocaleString('en-US');if(q<1)requestAnimationFrame(tick)}requestAnimationFrame(tick)});
</script></body></html>
""".replace("__DATA__", json.dumps(kpi_data))
components.html(KPI, height=142)

hi_i = inc["any_disease"].iloc[-1]
lo_i = inc["any_disease"].iloc[0]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Score a Worker", "Batch Upload", "Live Monitor", "How It Works"])

# ==================================================================== DASHBOARD
with tab1:
    section("Risk overview", "distribution of screened workers and disease patterns")
    ratio = hi_i / lo_i
    c = st.columns(4)
    c[0].markdown(stat("High vs Low incidence", f"×{ratio:.1f}", f"{hi_i:.1f}% vs {lo_i:.1f}% any disease"), unsafe_allow_html=True)
    c[1].markdown(stat("Chi-square", f"{val['chi2']['stat']:,.0f}", "tier × disease, p < 0.001"), unsafe_allow_html=True)
    c[2].markdown(stat("Spearman ρ", f"{val['spearman']['rho']:.3f}", "risk rises with tier"), unsafe_allow_html=True)
    c[3].markdown(stat("Stability (ARI)", f"{val['stability_ari']['seeds_mean']:.3f}", "same tiers across random seeds"), unsafe_allow_html=True)

    section("Risk-tier distribution", "how workers are grouped across the model")
    a, b = st.columns([1, 1.4])
    fig = go.Figure(go.Pie(labels=NAMES, values=counts.values, hole=.68, sort=False,
                           marker=dict(colors=COLORS, line=dict(color="#111a21", width=4)), textinfo="percent",
                           textfont=dict(size=14), pull=[0] * (K - 1) + [.05]))
    fig.add_annotation(text=f"<b>{len(scored):,}</b><br><span style='font-size:12px;color:#81918f'>workers</span>", showarrow=False,
                       font=dict(size=26, color="#edf2eb", family="Playfair Display, serif"))
    with a:
        pchart(style_fig(fig, 400, "Tier distribution"))
    labels = {"tb": "TB", "respiratory_illness": "Respiratory", "anaemia": "Anaemia", "hypertension": "Hypertension",
              "diabetes": "Diabetes", "any_disease": "Any disease"}
    fig = go.Figure()
    for n, col in zip(NAMES, COLORS):
        fig.add_bar(name=n, x=[labels[c_] for c_ in inc.columns], y=inc.loc[n].values, marker_color=col,
                    hovertemplate="%{x}: %{y:.1f}%<extra>" + n + "</extra>")
    fig.update_layout(barmode="group", bargap=.25, yaxis_title="Incidence (%)")
    with b:
        pchart(style_fig(fig, 400, "Disease incidence rises with risk tier"))


    section("Health indicators & priority outreach", "risk signals and workers requiring earlier follow-up")
    a, b = st.columns([1.05, 1.25], gap="medium")
    trend_fields = [("systolic_bp", "Systolic BP"), ("glucose", "Glucose"), ("spo2", "SpO₂")]
    trend = prof.copy()
    fig = go.Figure()
    for field, label in trend_fields:
        if field in trend.columns:
            vals = trend[field].astype(float).tolist()
            fig.add_trace(go.Scatter(x=NAMES, y=vals, mode="lines+markers", name=label,
                                     line=dict(width=2.5), marker=dict(size=7)))
    fig.update_layout(hovermode="x unified", margin=dict(l=15,r=15,t=38,b=10), height=320,
                      legend=dict(orientation="h",y=1.12,x=0), yaxis_title="Average value")
    with a:
        pchart(style_fig(fig, 320, "Health indicator profile across risk tiers"))
    top_preview = scored[scored.tier_rank == K - 1].sort_values("risk_index", ascending=False)
    cols_preview = [c for c in ["worker_id","risk_tier","risk_index","top_risk_drivers"] if c in top_preview.columns]
    with b:
        st.markdown("<div class='eyebrow' style='margin:5px 0 10px'>PRIORITY WORKER QUEUE</div>", unsafe_allow_html=True)
        dframe(top_preview[cols_preview].head(6),
               column_config={"risk_index": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=100, format="%.0f")})

    section("Worker risk landscape", "explore how the records separate into groups")
    a, b = st.columns([1.5, 1])

    @st.cache_data
    def pca_points():
        s = scored.sample(min(2500, len(scored)), random_state=1)
        xt = load_bundle()["pre"].transform(s[FEATURES])
        p = PCA(3, random_state=42).fit_transform(xt)
        return pd.DataFrame({"PC1": p[:, 0], "PC2": p[:, 1], "PC3": p[:, 2], "Tier": s.risk_tier.values})

    pts = pca_points()
    fig = px.scatter_3d(pts, x="PC1", y="PC2", z="PC3", color="Tier", color_discrete_map=CMAP,
                        category_orders={"Tier": NAMES}, opacity=.85)
    fig.update_traces(marker=dict(size=2.6))
    ax = dict(showbackground=False, gridcolor="rgba(255,255,255,.08)", zeroline=False, showticklabels=False, title="")
    fig.update_layout(scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.8, y=1.2, z=.8))),
                      updatemenus=[dict(type="buttons", showactive=False, x=.02, y=.02, bgcolor="rgba(212,175,55,.85)",
                                        font=dict(color="#080d12"), buttons=[
                          dict(label="Rotate", method="animate", args=[None, dict(frame=dict(duration=70, redraw=True),
                                                                                    transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                          dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                                     mode="immediate")])])])
    fig.frames = [go.Frame(layout=go.Layout(scene=dict(camera=dict(eye=dict(x=2.1 * np.cos(t), y=2.1 * np.sin(t), z=.8)))))
                  for t in np.linspace(0, 2 * np.pi, 100)]
    with a:
        pchart(style_fig(fig, 470, "Workers projected to 3 principal components"))

    feats = [("age", "Age"), ("systolic_bp", "Systolic BP"), ("glucose", "Glucose"), ("hemoglobin", "Low haemoglobin"),
             ("spo2", "Low SpO2"), ("missed_followups", "Missed follow-ups"), ("days_since_last_visit", "Gap since last visit"),
             ("dust_exposure", "Dust exposure")]
    fig = go.Figure()
    for n, col in zip(NAMES, COLORS):
        r = []
        for f_, _ in feats:
            v = prof[f_]
            nrm = (v[n] - v.min()) / (v.max() - v.min() + 1e-9)
            r.append(nrm if RISK_SIGNS.get(f_, 1) > 0 else 1 - nrm)
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=[l for _, l in feats] + [feats[0][1]], fill="toself", name=n,
                                      line=dict(color=col, width=2), opacity=.5))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 1]),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,.12)")))
    with b:
        pchart(style_fig(fig, 470, "Risk fingerprint of each tier"))

    section("Disease heat-map and method comparison")
    a, b = st.columns(2)
    z = inc.T.rename(index=labels)
    fig = go.Figure(go.Heatmap(z=z.values, x=NAMES, y=list(z.index), colorscale=[[0, "#111a21"], [.55, "#55d6b0"], [1, "#ed777d"]],
                               text=np.round(z.values, 1), texttemplate="%{text}%", showscale=False, xgap=3, ygap=3))
    with a:
        pchart(style_fig(fig, 380, "Incidence (%) by disease and tier"))
    gap = base.incidence_gap_pct.fillna(0)
    fig = go.Figure(go.Bar(x=base.method, y=gap, marker_color=["#55d6b0" if "ours" in m else "#33434a" for m in base.method],
                           text=[f"{g:.1f}" if g > 0 else "1 cluster" for g in gap], textposition="outside"))
    fig.update_layout(yaxis_title="High-minus-Low incidence gap (pp)", yaxis_range=[0, max(gap) * 1.25])
    with b:
        pchart(style_fig(fig, 380, "k-Means separates disease risk best"))

    section("How k was chosen", "elbow method + three validity indices; dashed line = chosen k")
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Inertia (elbow)", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"))
    for i, (col_, colr) in enumerate([("inertia", "#8ba9b0"), ("silhouette", "#55d6b0"), ("davies_bouldin", "#e9b96e"),
                                      ("calinski_harabasz", "#ed777d")], 1):
        fig.add_trace(go.Scatter(x=kscan.k, y=kscan[col_], mode="lines+markers", line=dict(color=colr, width=3),
                                 marker=dict(size=8), showlegend=False), row=1, col=i)
        fig.add_vline(x=K, line_dash="dash", line_color="rgba(212,175,55,.7)", row=1, col=i)
    pchart(style_fig(fig, 300))

    section("Tier profiles", "average values in original units")
    dframe(prof.reset_index().round(2))

    section("Priority outreach list", "highest-risk workers first")
    top = scored[scored.tier_rank == K - 1].sort_values("risk_index", ascending=False)
    dframe(top[["worker_id", "risk_tier", "risk_index", "top_risk_drivers", "occupation", "housing", "work_state"]].head(300),
           column_config={"risk_index": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=100, format="%.0f"),
                          "worker_id": "Worker", "risk_tier": "Tier", "top_risk_drivers": "Main risk drivers",
                          "occupation": "Occupation", "housing": "Housing", "work_state": "Works in"})
    st.download_button("Download full high-risk list (CSV)", top.to_csv(index=False), "outreach_list.csv")

# =============================================================== SCORE A WORKER
PRESETS = {
    "low": dict(w_age=29, w_bmi=23.0, w_sbp=114, w_dbp=72, w_hb=13.8, w_glu=90, w_spo2=98, w_yrs=2.0, w_occ="Domestic",
                w_house="Rented room", w_dust=False, w_chem=False, w_vis=4, w_miss=0, w_since=45),
    "mid": dict(w_age=37, w_bmi=25.0, w_sbp=128, w_dbp=80, w_hb=12.4, w_glu=106, w_spo2=97, w_yrs=5.0, w_occ="Factory",
                w_house="Shared dormitory", w_dust=True, w_chem=False, w_vis=2, w_miss=1, w_since=150),
    "high": dict(w_age=47, w_bmi=27.5, w_sbp=152, w_dbp=96, w_hb=10.2, w_glu=148, w_spo2=93, w_yrs=9.0, w_occ="Mining",
                 w_house="Informal settlement", w_dust=True, w_chem=True, w_vis=1, w_miss=4, w_since=380)}
for k_, v_ in PRESETS["mid"].items():
    st.session_state.setdefault(k_, v_)


def set_preset(name):
    st.session_state.update(PRESETS[name])


with tab2:
    section("Score one worker", "move the sliders, the tier updates instantly")
    p = st.columns(3)
    p[0].button("Load healthy example", on_click=set_preset, args=("low",))
    p[1].button("Load medium example", on_click=set_preset, args=("mid",))
    p[2].button("Load high-risk example", on_click=set_preset, args=("high",))
    left, right = st.columns([1.15, 1])
    with left:
        c1, c2 = st.columns(2)
        age = c1.slider("Age", 18, 70, key="w_age")
        bmi = c2.slider("BMI", 14.0, 42.0, step=.5, key="w_bmi")
        sbp = c1.slider("Systolic BP (mmHg)", 90, 200, key="w_sbp")
        dbp = c2.slider("Diastolic BP (mmHg)", 55, 120, key="w_dbp")
        hb = c1.slider("Haemoglobin (g/dL)", 6.0, 17.0, step=.1, key="w_hb")
        glu = c2.slider("Glucose (mg/dL)", 60, 300, key="w_glu")
        spo2 = c1.slider("SpO2 (%)", 85, 100, key="w_spo2")
        yrs = c2.slider("Years in current job", 0.0, 35.0, step=.5, key="w_yrs")
        occ = c1.selectbox("Occupation", OCC, key="w_occ")
        house = c2.selectbox("Housing", HOUSING, key="w_house")
        vis = c1.slider("Clinic visits (12 months)", 0, 15, key="w_vis")
        miss = c2.slider("Missed follow-ups", 0, 10, key="w_miss")
        since = c1.slider("Days since last visit", 0, 900, key="w_since")
        dust = c2.checkbox("Dust exposure", key="w_dust")
        chem = c2.checkbox("Chemical exposure", key="w_chem")

    row = pd.DataFrame([dict(age=age, bmi=bmi, systolic_bp=sbp, diastolic_bp=dbp, hemoglobin=hb, glucose=glu, spo2=spo2,
                             years_in_job=yrs, dust_exposure=int(dust), chemical_exposure=int(chem), clinic_visits_12m=vis,
                             missed_followups=miss, days_since_last_visit=since, occupation=occ, housing=house)])
    r = score_workers(row, bundle).iloc[0]
    colr = CMAP[r.risk_tier]
    with right:
        html(f"<div class='glass' style='text-align:center'><div class='eyebrow'>Predicted risk tier</div>"
             f"<div class='tier-badge' style='--c:{colr}'>{r.risk_tier}</div></div>")
        g = go.Figure(go.Indicator(mode="gauge+number", value=float(r.risk_index),
                                   number=dict(font=dict(size=46, color="#edf2eb", family="Playfair Display, serif")),
                                   gauge=dict(axis=dict(range=[0, 100], tickcolor="#718581"), bar=dict(color=colr, thickness=.3),
                                              bgcolor="rgba(255,255,255,.05)", borderwidth=0)))
        pchart(style_fig(g, 250, "Risk index (0 to 100)"))
        chips = "".join(f"<span class='chip' style='animation-delay:{i * .12}s'>{d}</span>"
                        for i, d in enumerate(str(r.top_risk_drivers).split(", ")))
        html(f"<div class='glass'><div class='eyebrow'>Main risk drivers</div>{chips}</div>")
    avg = prof.loc[r.risk_tier]
    m = st.columns(5)
    m[0].metric("Systolic BP", f"{sbp}", f"{sbp - avg.systolic_bp:+.0f} vs tier avg", delta_color="inverse")
    m[1].metric("Glucose", f"{glu}", f"{glu - avg.glucose:+.0f} vs tier avg", delta_color="inverse")
    m[2].metric("Haemoglobin", f"{hb}", f"{hb - avg.hemoglobin:+.1f} vs tier avg")
    m[3].metric("SpO2", f"{spo2}%", f"{spo2 - avg.spo2:+.1f} vs tier avg")
    m[4].metric("Missed follow-ups", f"{miss}", f"{miss - avg.missed_followups:+.1f} vs tier avg", delta_color="inverse")

# ==================================================================== BATCH UPLOAD
with tab3:
    section("Batch scoring", "upload a CSV, get a tier for every worker")
    st.caption("Required columns: " + ", ".join(FEATURES))
    up = st.file_uploader("Drop your CSV file here", type="csv")
    st.download_button("Download a sample input file",
                       generate(60, seed=7, with_outcomes=False)[["worker_id"] + FEATURES].to_csv(index=False), "sample_input.csv")
    if up is not None:
        d = pd.read_csv(up)
        miss_cols = [c_ for c_ in FEATURES if c_ not in d.columns]
        if miss_cols:
            st.error(f"Missing columns: {miss_cols}")
        else:
            res = score_workers(d, bundle)
            st.success(f"Scored {len(res):,} workers")
            a, b = st.columns(2)
            cn = res.risk_tier.value_counts().reindex(NAMES).fillna(0)
            with a:
                pchart(style_fig(go.Figure(go.Pie(labels=NAMES, values=cn.values, hole=.62, sort=False,
                                                  marker=dict(colors=COLORS, line=dict(color="#111a21", width=3)))), 340, "Tier mix"))
            with b:
                pchart(style_fig(px.histogram(res, x="risk_index", color="risk_tier", color_discrete_map=CMAP, nbins=30,
                                              category_orders={"risk_tier": NAMES}), 340, "Risk index distribution"))
            dframe(res.sort_values("risk_index", ascending=False),
                   column_config={"risk_index": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=100, format="%.0f")})
            st.download_button("Download scored results", res.to_csv(index=False), "scored_workers.csv")

# ==================================================================== LIVE MONITOR
if "live" not in st.session_state:
    st.session_state.live, st.session_state.seed = pd.DataFrame(), 1000


def render_live(auto_on):
    c1, c2 = st.columns(2)
    add = c1.button("Add 10 screenings")
    if c2.button("Reset feed"):
        st.session_state.live = pd.DataFrame()
    if add or auto_on:
        new = generate(10, seed=st.session_state.seed, with_outcomes=False)
        new["worker_id"] = [f"LIVE{st.session_state.seed}-{i}" for i in range(10)]
        st.session_state.seed += 1
        st.session_state.live = pd.concat([score_workers(new, bundle), st.session_state.live]).head(400)
    live = st.session_state.live
    if len(live):
        cols = st.columns(K)
        for cc, n in zip(cols, NAMES):
            cc.markdown(stat(f"{n} risk", int((live.risk_tier == n).sum()), "in this feed"), unsafe_allow_html=True)
        chrono = live.iloc[::-1].reset_index(drop=True)
        fig = px.scatter(chrono, x=chrono.index, y="risk_index", color="risk_tier", color_discrete_map=CMAP,
                         category_orders={"risk_tier": NAMES}, hover_data=["worker_id", "top_risk_drivers"])
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="#111a21")))
        fig.update_layout(xaxis_title="Arrival order", yaxis_title="Risk index")
        pchart(style_fig(fig, 340, "Risk index of incoming workers"))
        section("Newest high-risk alerts")
        al = live[live.tier_rank == K - 1][["worker_id", "risk_index", "top_risk_drivers", "occupation", "housing"]].head(12)
        dframe(al, column_config={"risk_index": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=100, format="%.0f")})
    else:
        html("<div class='glass' style='text-align:center;color:#81918f'>Press <b>Add 10 screenings</b> or switch on auto-stream to start the feed.</div>")


with tab4:
    section("Live screening feed", "new records are standardized and tiered the moment they arrive")
    html("<span class='live'><i></i> LIVE</span>")
    auto = st.toggle("Auto-stream every 2 seconds (turn off when you finish)")
    if hasattr(st, "fragment"):
        st.fragment(run_every=2 if auto else None)(render_live)(auto)
    else:
        render_live(auto)
        if auto:
            time.sleep(2)
            st.rerun()

# ================================================================== HOW IT WORKS
KDEMO = """
<html><head><style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
html,body{margin:0;background:transparent;overflow:hidden;font-family:'Inter',sans-serif}
.w{border-radius:14px;background:#111a21;border:1px solid rgba(255,255,255,.08);box-shadow:0 12px 32px rgba(0,0,0,.32);overflow:hidden}
canvas{display:block;width:100%;height:340px}
.bar{display:flex;gap:12px;align-items:center;padding:12px 18px;border-top:1px solid rgba(255,255,255,.08);flex-wrap:wrap}
button{font-family:'Inter',sans-serif;font-size:12.5px;letter-spacing:.06em;font-weight:600;color:#a4efd9;background:transparent;border:1px solid #55d6b0;
 border-radius:8px;padding:9px 20px;cursor:pointer;transition:.25s}
button:hover{background:#55d6b0;color:#080d12;box-shadow:0 8px 22px rgba(212,175,55,.28)}
#s{color:#9eafaa;font-size:13px;flex:1;min-width:240px}#s b{color:#a4efd9}
</style></head><body><div class="w"><canvas id="c"></canvas><div class="bar">
<button id="st">Step</button><button id="au">Auto-run</button><button id="rs">Reset</button><div id="s"></div></div></div><script>
const cv=document.getElementById('c'),x=cv.getContext('2d'),S=document.getElementById('s');let W,H;
function size(){W=cv.width=cv.offsetWidth;H=cv.height=cv.offsetHeight}size();
const col=['#55d6b0','#e9b96e','#ed777d'];let seed=11;
function rnd(){seed=(seed*16807)%2147483647;return(seed-1)/2147483646}
function gs(){let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(6.283*v)}
let pts,cent,phase,iter,conv,moving,timer;
function init(){seed=11;pts=[];[[.22,.68],[.52,.28],[.80,.66]].forEach(c=>{for(let i=0;i<90;i++)pts.push({x:(c[0]+gs()*.075)*W,y:(c[1]+gs()*.13)*H,g:-1})});
 cent=[];for(let i=0;i<3;i++){const p=pts[Math.floor(rnd()*pts.length)];cent.push({x:p.x,y:p.y,tx:p.x,ty:p.y,tr:[]})}
 phase=0;iter=0;conv=false;moving=false;clearInterval(timer);timer=null;
 S.innerHTML='3 random centres placed. Press <b>Step</b>: every worker joins the nearest centre.'}
function assign(){let ch=0;pts.forEach(p=>{let b=0,bd=1e18;cent.forEach((c,i)=>{const d=(p.x-c.x)**2+(p.y-c.y)**2;if(d<bd){bd=d;b=i}});if(p.g!==b)ch++;p.g=b});return ch}
function targets(){cent.forEach((c,i)=>{const m=pts.filter(p=>p.g===i);if(m.length){c.tx=m.reduce((s,p)=>s+p.x,0)/m.length;c.ty=m.reduce((s,p)=>s+p.y,0)/m.length}})}
function step(){if(conv||moving)return;
 if(phase===0){const ch=assign();iter++;targets();phase=1;
  if(ch===0){conv=true;clearInterval(timer);S.innerHTML='<b>Converged</b> after '+iter+' rounds. Nobody changed group, so these are the final tiers.';return}
  S.innerHTML='Round '+iter+': <b>'+ch+'</b> workers joined a new centre. Now each centre moves to the middle of its group.'}
 else{moving=true;phase=0;S.innerHTML='Round '+iter+': centres moving to the average position of their members...'}}
function draw(){x.clearRect(0,0,W,H);x.lineWidth=1;
 if(moving){let done=true;cent.forEach(c=>{c.tr.push([c.x,c.y]);if(c.tr.length>40)c.tr.shift();c.x+=(c.tx-c.x)*.14;c.y+=(c.ty-c.y)*.14;
  if(Math.abs(c.tx-c.x)>.4||Math.abs(c.ty-c.y)>.4)done=false});if(done){moving=false;cent.forEach(c=>{c.x=c.tx;c.y=c.ty});
  if(!conv)S.innerHTML='Round '+iter+' done. Press <b>Step</b> again: workers re-check which centre is nearest.'}}
 pts.forEach(p=>{if(p.g>=0){x.globalAlpha=.14;x.strokeStyle=col[p.g];x.beginPath();x.moveTo(p.x,p.y);x.lineTo(cent[p.g].x,cent[p.g].y);x.stroke()}});
 pts.forEach(p=>{x.globalAlpha=.92;x.fillStyle=p.g<0?'#718581':col[p.g];x.beginPath();x.arc(p.x,p.y,3.3,0,6.283);x.fill()});
 cent.forEach((c,i)=>{x.globalAlpha=.45;x.strokeStyle='#55d6b0';x.beginPath();c.tr.forEach((q,j)=>j?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.stroke();
  x.globalAlpha=1;x.lineWidth=2.4;x.strokeStyle='#55d6b0';x.fillStyle='rgba(16,26,46,.85)';x.beginPath();x.arc(c.x,c.y,12,0,6.283);x.fill();x.stroke();
  x.beginPath();x.moveTo(c.x-6,c.y);x.lineTo(c.x+6,c.y);x.moveTo(c.x,c.y-6);x.lineTo(c.x,c.y+6);x.stroke();x.lineWidth=1});
 requestAnimationFrame(draw)}
document.getElementById('st').onclick=step;document.getElementById('rs').onclick=init;
document.getElementById('au').onclick=()=>{if(timer)return;timer=setInterval(step,950)};
init();draw();
</script></body></html>
"""

with tab5:
    section("Try k-Means yourself", "press Step and watch the three centres find the groups")
    components.html(KDEMO, height=420)
    section("From raw records to outreach list", "the five steps of this project")
    steps = [
        ("1", "Collect and clean the records", "Vitals (BP, glucose, haemoglobin, SpO2), exposure history (dust, chemicals, housing, occupation) and visit frequency. Missing values are filled and text columns are encoded."),
        ("2", "Standardize every feature", "Each number is converted to a z-score so that no feature wins just because its numbers are bigger. Without this, glucose (about 100) would drown out SpO2 (about 97)."),
        ("3", "Group with k-Means", "The algorithm drops k centre points, assigns every worker to the nearest centre, moves the centres to the middle of their groups, and repeats until nothing changes. We use k-means++ starts and 20 restarts."),
        ("4", "Turn clusters into risk tiers", "Clusters are ranked by a composite risk score (high BP, high glucose, low haemoglobin, missed follow-ups and so on) and named Low, Medium and High."),
        ("5", "Validate and act", "If the tiers are meaningful, disease should rise from Low to High. It does: 19.6%, 38.9%, 72.5%. The High tier becomes the priority outreach list."),
    ]
    for n_, t_, d_ in steps:
        html(f"<div class='step' style='animation-delay:{int(n_) * .1}s'><div class='n'>{n_}</div><div><h4>{t_}</h4><p>{d_}</p></div></div>")
    section("Technology stack")
    html("<div class='tech'><span>Python</span><span>pandas</span><span>NumPy</span><span>scikit-learn</span><span>SciPy</span>"
         "<span>Plotly</span><span>Streamlit</span><span>joblib</span></div>")
    section("SIH26006 objectives covered")
    html("<div class='glass'>✓ &nbsp;<b>Standardize</b> migrant-worker health-record features (vitals, exposure history, visit frequency)<br><br>"
         "✓ &nbsp;<b>Apply k-Means</b> to segment workers into risk-tier clusters<br><br>"
         "✓ &nbsp;<b>Validate tiers</b> against reported disease-incidence patterns</div>")
    html("<div class='foot'>Prototype built on a simulated dataset because individual migrant-worker records are not public. "
         "The same pipeline runs on real records with the same columns.</div>")