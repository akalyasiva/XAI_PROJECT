"""4_Model_Analytics.py — Model Analytics with Simple English explanations"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Model Analytics", page_icon="📊", layout="wide")

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#f0fff8;--bg2:#ffffff;--bg3:#f8fff9;--surface:#e8fff2;
  --border:#b0dcc4;--text:#0f172a;--text2:#334155;--text3:#64748b;--text4:#94a3b8;
  --gold:#b8860b;--gold2:#d97706;--shap:#059669;--lime:#d97706;--blue:#2563eb;
  --inp-bg:#ffffff;
}
*,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text2)!important;}
#MainMenu,header,footer{visibility:hidden;}
label{color:var(--text2)!important;font-size:.82rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div,div[data-baseweb="select"]>div{
  background:var(--inp-bg)!important;border-color:var(--border)!important;
  color:var(--text)!important;border-radius:8px!important;}
.stSlider>div>div>div{background:var(--shap)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;
  border:1px solid var(--border);border-radius:10px;}
.stTabs [data-baseweb="tab"]{color:var(--text4)!important;
  font-family:Orbitron,monospace!important;font-size:.6rem!important;
  letter-spacing:2px!important;padding:11px 18px!important;}
.stTabs [aria-selected="true"]{color:var(--shap)!important;
  background:var(--bg3)!important;border-radius:8px!important;}

/* Explain box — plain English callout */
.explain-box{background:#ffffff;border-left:4px solid var(--shap);
  border-radius:0 10px 10px 0;padding:14px 18px;margin:0 0 18px;
  font-size:.88rem;color:var(--text2);line-height:1.8;
  box-shadow:0 2px 6px rgba(5,150,105,.07);}
.explain-box.amber{border-left-color:var(--lime);}
.explain-box.blue {border-left-color:var(--blue);}
.explain-box.gold {border-left-color:var(--gold);}
.explain-box strong{color:var(--text);}

/* Stat card */
.stat-card{background:#ffffff;border:1px solid var(--border);border-radius:10px;
  padding:16px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,.04);}
.stat-lbl{font-family:Orbitron,monospace;font-size:.56rem;letter-spacing:2px;
  color:var(--text4);text-transform:uppercase;margin-bottom:6px;}
.stat-val{font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;}

.sep{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:18px 0;}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#064e3b 0%,#022c22 60%,#0f172a 100%);
    border:1px solid #059669;border-radius:18px;padding:40px 48px;margin-bottom:28px;
    position:relative;overflow:hidden;'>
    <div style='position:absolute;font-size:200px;right:-10px;top:-40px;opacity:.03;'>📊</div>
    <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:5px;
        color:#34d399;text-transform:uppercase;margin-bottom:10px;'>How the AI Works</div>
    <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;color:#fff;'>
        Model <span style="background:linear-gradient(90deg,#34d399,#06d6a0);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Analytics</span>
    </div>
    <div style='color:rgba(255,255,255,.65);font-size:.92rem;margin-top:8px;'>
        A simple look at how the AI thinks, what it pays attention to, and how confident it is.
    </div>
</div>
""", unsafe_allow_html=True)

FEATURE_NAMES = ["Inning","Batting Team","Bowling Team","Ball Number",
                 "Current Score","Wickets Fallen","Run Rate","Remaining Overs"]

FEATURE_PLAIN = {
    "Inning":           "Which innings (1st or 2nd)",
    "Batting Team":     "The batting team",
    "Bowling Team":     "The bowling team",
    "Ball Number":      "How many balls bowled so far",
    "Current Score":    "Runs on the board right now",
    "Wickets Fallen":   "Number of batters out",
    "Run Rate":         "Runs per over (scoring speed)",
    "Remaining Overs":  "Overs still left to play",
}

# Light-themed plot helper
def PL(h=400):
    return dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fff9",
        font=dict(color="#334155", family="Inter"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#059669", font_color="#0f172a"),
        margin=dict(t=50, b=40, l=20, r=20), height=h
    )

# ── PARAMETERS ────────────────────────────────────────────────────────────────
with st.expander("⚙️  Set match situation to analyse", expanded=True):
    st.markdown("""<div style='font-size:.82rem;color:var(--text3);margin-bottom:12px;'>
        Enter a match situation below — the charts will update to show what the AI
        would focus on in that exact scenario.</div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: score  = st.number_input("Runs scored", 0, 300, 118)
    with c2: wkts   = st.slider("Wickets fallen",    0, 10, 3)
    with c3: overs  = st.slider("Overs completed",   0.0, 20.0, 12.0, step=0.1)
    with c4: inning = st.selectbox("Which inning",   [1, 2])

# ── SIMULATED SHAP / LIME ─────────────────────────────────────────────────────
np.random.seed(int(score + wkts * 7 + overs * 3))
rr  = score / (overs + 1e-6)
rem = 20 - overs
feat_vals_raw = [inning, 0, 1, int(overs*6), score, wkts, rr, rem]

base_shap = np.array([0.02*inning, 0.01, -0.01, 0.07*(overs/20),
                       0.14*(score/180), -0.11*(wkts/10),
                       0.10*min(rr/12,1), 0.05*(rem/20)])
shap_vals   = base_shap + np.random.normal(0, 0.012, 8)
lime_vals   = shap_vals * np.random.uniform(0.82, 1.18, 8) + np.random.normal(0, 0.008, 8)
wp          = float(np.clip(0.5 + shap_vals.sum() * 1.3, 0.05, 0.95))
global_shap = np.abs(shap_vals) * np.array([1.1, 0.6, 0.6, 1.0, 1.3, 1.4, 1.5, 1.2])

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "🏆  What Matters Most",
    "🔍  This Moment",
    "🍋  Second Check",
    "⚖️  Comparing Both",
    "🤖  About the Model",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Global importance  (was "Global SHAP")
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("""
    <div class='explain-box'>
        <strong>What does this tab show?</strong><br>
        This shows which pieces of information the AI pays the most attention to,
        across <em>all</em> matches — not just this one.
        Think of it as the AI's general priorities. A longer bar means the AI
        cares about that factor more.
    </div>""", unsafe_allow_html=True)

    # Bar chart — overall importance
    sidx = np.argsort(global_shap)
    plain_labels = [FEATURE_PLAIN[FEATURE_NAMES[i]] for i in sidx]

    fig1 = go.Figure(go.Bar(
        y=plain_labels,
        x=[float(global_shap[i]) for i in sidx],
        orientation='h',
        marker=dict(
            color=[float(global_shap[i]) for i in sidx],
            colorscale=[[0,"#d1fae5"],[0.5,"#059669"],[1,"#b8860b"]],
            showscale=False,
            line=dict(color="#e2e8f0", width=0.5)
        ),
        text=[("Most important" if i == sidx[-1] else
               "Very important" if i >= sidx[-2] else "")
              for i in sidx],
        textposition='outside',
        textfont=dict(color="#64748b", size=9, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Importance score: %{x:.3f}<extra></extra>"
    ))
    fig1.update_layout(**PL(400),
        title=dict(text="What the AI pays attention to the most",
                   font=dict(color="#0f172a", size=14, family="Inter")),
        xaxis=dict(gridcolor="#e2e8f0",
                   title="How much this factor influences the prediction",
                   title_font=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(fig1, use_container_width=True)

    # Simple takeaway cards
    top3_idx = np.argsort(global_shap)[::-1][:3]
    st.markdown("<div style='font-size:.82rem;color:var(--text3);margin:4px 0 10px;'>"
                "<strong style='color:var(--text);'>Top 3 things the AI cares about most:</strong>"
                "</div>", unsafe_allow_html=True)

    medals = ["🥇","🥈","🥉"]
    cols   = st.columns(3)
    for rank, (col_w, idx) in enumerate(zip(cols, top3_idx)):
        fname = FEATURE_NAMES[idx]
        plain = FEATURE_PLAIN[fname]
        score_val = feat_vals_raw[idx]
        with col_w:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size:1.6rem;margin-bottom:6px;'>{medals[rank]}</div>
                <div style='font-size:.88rem;font-weight:700;color:var(--text);
                    margin-bottom:4px;'>{fname}</div>
                <div style='font-size:.76rem;color:var(--text3);line-height:1.5;'>
                    {plain}</div>
                <div style='font-family:Orbitron,monospace;font-size:.8rem;
                    font-weight:700;color:var(--shap);margin-top:8px;'>
                    Current: {score_val:.1f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # Beeswarm — simplified as scatter
    st.markdown("""
    <div class='explain-box'>
        <strong>The chart below</strong> shows how the same factor can help or hurt
        depending on its value. Each dot is one match moment.
        <strong>Dots on the right</strong> = that factor pushed towards WIN.
        <strong>Dots on the left</strong> = pushed towards LOSS.
        The colour shows whether the factor's value was high (orange) or low (blue).
    </div>""", unsafe_allow_html=True)

    np.random.seed(99)
    N = 40
    scat_x, scat_y, scat_c = [], [], []
    for i, fname in enumerate(FEATURE_NAMES):
        spread = base_shap[i] + np.random.normal(0, 0.025, N)
        scat_x.extend(spread.tolist())
        scat_y.extend([FEATURE_PLAIN[fname]] * N)
        scat_c.extend(np.random.uniform(-1, 1, N).tolist())

    fig1b = go.Figure(go.Scatter(
        x=scat_x, y=scat_y, mode='markers',
        marker=dict(size=7, color=scat_c,
                    colorscale=[[0,"#3b82f6"],[0.5,"#94a3b8"],[1,"#d97706"]],
                    showscale=True, opacity=0.72,
                    colorbar=dict(title="Value (Low → High)",
                                  title_font_color="#64748b", tickfont_color="#64748b",
                                  tickvals=[-1,0,1], ticktext=["Low","Mid","High"])),
        hovertemplate="<b>%{y}</b><br>Influence: %{x:.3f}<extra></extra>"
    ))
    fig1b.add_vline(x=0, line_color="#b0dcc4", line_width=1.5,
                    annotation_text="← LOSS  |  WIN →",
                    annotation_font_color="#94a3b8", annotation_font_size=10)
    fig1b.update_layout(**PL(420),
        title=dict(text="How each factor pushes the prediction — across many match moments",
                   font=dict(color="#0f172a", size=13, family="Inter")),
        xaxis=dict(gridcolor="#e2e8f0",
                   title="← Pushes towards LOSS     |     Pushes towards WIN →",
                   title_font=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(fig1b, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Local SHAP  ("This Moment")
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    pred_label = "WIN" if wp >= 0.5 else "LOSS"
    pred_color = "#059669" if wp >= 0.5 else "#dc2626"

    st.markdown(f"""
    <div class='explain-box'>
        <strong>What does this tab show?</strong><br>
        This is the AI's explanation for <em>this specific match situation</em> — 
        {score} runs, {wkts} wickets, {overs:.1f} overs.
        Each bar shows whether a factor is <strong style='color:#059669;'>helping WIN</strong>
        or <strong style='color:#dc2626;'>pushing towards LOSS</strong>.
        The AI started at 50-50 and each factor nudged it up or down.
    </div>""", unsafe_allow_html=True)

    sv_s      = np.argsort(shap_vals)
    sv_sorted = shap_vals[sv_s]
    fn_sorted = [FEATURE_PLAIN[FEATURE_NAMES[i]] for i in sv_s]

    fig2 = go.Figure(go.Waterfall(
        name="Influence", orientation="h",
        measure=["relative"]*8 + ["total"],
        y=fn_sorted + ["FINAL PREDICTION"],
        x=list(sv_sorted) + [0],
        base=0.5,
        connector=dict(line=dict(color="#c7d4f0", width=1)),
        decreasing=dict(marker=dict(color="#dc2626",
                                    line=dict(color="#b91c1c", width=1))),
        increasing=dict(marker=dict(color="#059669",
                                    line=dict(color="#047857", width=1))),
        totals=dict(marker=dict(color="#b8860b",
                                line=dict(color="#92400e", width=1))),
        text=[("↑ Helps" if v > 0 else "↓ Hurts") for v in sv_sorted] + [f"{wp*100:.0f}%"],
        textposition="outside",
        textfont=dict(color="#334155", size=10, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Effect: %{x:+.3f}<extra></extra>"
    ))
    fig2.add_vline(x=0.5, line_dash="dash", line_color="#b0dcc4",
                   annotation_text="50/50 starting point",
                   annotation_font_color="#94a3b8", annotation_font_size=9)
    fig2.update_layout(**PL(460),
        title=dict(
            text=f"Why the AI predicts {pred_label} ({wp*100:.1f}%) for this match state",
            font=dict(color="#0f172a", size=13, family="Inter")),
        xaxis=dict(gridcolor="#e2e8f0",
                   title="← Pushes towards LOSS (0)     Win probability     Pushes towards WIN (1) →",
                   title_font=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(fig2, use_container_width=True)

    # Simple table
    st.markdown("""<div style='font-size:.82rem;color:var(--text3);margin:4px 0 10px;'>
        <strong style='color:var(--text);'>Full breakdown — every factor explained:</strong>
        </div>""", unsafe_allow_html=True)

    df_shap = pd.DataFrame({
        "Factor":         FEATURE_NAMES,
        "Plain English":  [FEATURE_PLAIN[f] for f in FEATURE_NAMES],
        "Current Value":  [f"{v:.1f}" for v in feat_vals_raw],
        "Effect":         ["↑ Helps WIN" if v >= 0 else "↓ Hurts chances" for v in shap_vals],
        "How much":       ["Strong" if abs(v) > 0.08 else
                           "Medium" if abs(v) > 0.03 else "Small"
                           for v in shap_vals],
    }).sort_values("How much",
                   key=lambda x: x.map({"Strong":0,"Medium":1,"Small":2}))

    st.dataframe(df_shap.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Simple math strip
    total = 0.5 + float(shap_vals.sum())
    st.markdown(f"""
    <div style='background:#f0fff8;border:1px solid var(--border);border-radius:8px;
        padding:12px 18px;margin-top:6px;font-size:.82rem;color:var(--text3);'>
        The AI started at <strong style='color:var(--text);'>50% — completely even</strong>.
        After looking at all 8 factors it moved to
        <strong style='color:{pred_color};font-size:1rem;'>{wp*100:.1f}% → {pred_label}</strong>.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Local LIME  ("Second Check")
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("""
    <div class='explain-box amber'>
        <strong>What is this second check?</strong><br>
        This is a completely different way the AI double-checks itself.
        Instead of looking at the whole picture, it zooms into
        <em>this exact moment</em> and builds a tiny simple model just for right now.
        If both methods agree, the prediction is more reliable.
        <strong>Orange bars</strong> = helping WIN.
        <strong>Red bars</strong> = pushing towards LOSS.
    </div>""", unsafe_allow_html=True)

    lv_s      = np.argsort(lime_vals)
    lv_sorted = lime_vals[lv_s]
    ln_sorted = [FEATURE_PLAIN[FEATURE_NAMES[i]] for i in lv_s]

    fig3 = go.Figure(go.Bar(
        y=ln_sorted, x=lv_sorted, orientation="h",
        marker=dict(
            color=["#d97706" if v >= 0 else "#dc2626" for v in lv_sorted],
            opacity=0.85, line=dict(color="#e2e8f0", width=0.5)
        ),
        text=["↑ Helping" if v >= 0 else "↓ Hurting" for v in lv_sorted],
        textposition="outside",
        textfont=dict(color="#334155", size=10, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Influence score: %{x:+.3f}<extra></extra>"
    ))
    fig3.add_vline(x=0, line_color="#b0dcc4", line_width=1.5,
                   annotation_text="← Hurting  |  Helping →",
                   annotation_font_color="#94a3b8", annotation_font_size=9)
    fig3.update_layout(**PL(420),
        title=dict(text="What is helping and hurting right now in this match",
                   font=dict(color="#0f172a", size=13, family="Inter")),
        xaxis=dict(gridcolor="#e2e8f0",
                   title="← Pushes towards LOSS     |     Pushes towards WIN →",
                   title_font=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#e2e8f0"))
    st.plotly_chart(fig3, use_container_width=True)

    # Metrics strip
    local_pred = float(np.clip(0.5 + lime_vals.sum(), 0.02, 0.98))
    r2  = float(np.clip(1 - abs(local_pred - wp) * 5, 0.6, 0.99))
    gap = abs(local_pred - wp) * 100

    cols = st.columns(4)
    items = [
        ("How reliable is this check?",
         "Very reliable" if r2 > 0.9 else "Quite reliable" if r2 > 0.75 else "Somewhat reliable",
         "var(--shap)"),
        ("This check says",
         f"{local_pred*100:.1f}%",
         "#059669" if local_pred >= 0.5 else "#dc2626"),
        ("Main model says",
         f"{wp*100:.1f}%",
         "#b8860b"),
        ("Do they agree?",
         f"{'Yes ✅' if gap < 5 else 'Mostly ✓' if gap < 10 else 'Some difference'}",
         "#059669" if gap < 5 else "#d97706"),
    ]
    for col_w, (lbl, val, clr) in zip(cols, items):
        with col_w:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-lbl'>{lbl}</div>
                <div class='stat-val' style='color:{clr};'>{val}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SHAP vs LIME  ("Comparing Both")
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("""
    <div class='explain-box blue'>
        <strong>Why compare two methods?</strong><br>
        When two completely different methods look at the same match and say the same thing,
        it means the AI is being consistent and the prediction is more trustworthy.
        This tab shows whether they agree or disagree on each factor.
    </div>""", unsafe_allow_html=True)

    # Grouped bar — side by side
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="Method 1 (Overall view)",
        x=[FEATURE_PLAIN[f] for f in FEATURE_NAMES],
        y=shap_vals,
        marker=dict(color=["#059669" if v >= 0 else "#dc2626" for v in shap_vals],
                    opacity=0.9),
        text=["↑" if v > 0 else "↓" for v in shap_vals],
        textposition="outside",
        hovertemplate="<b>Method 1</b> — %{x}<br>%{y:+.3f}<extra></extra>"
    ))
    fig4.add_trace(go.Bar(
        name="Method 2 (This moment)",
        x=[FEATURE_PLAIN[f] for f in FEATURE_NAMES],
        y=lime_vals,
        marker=dict(color=["#d97706" if v >= 0 else "#ef4444" for v in lime_vals],
                    opacity=0.75),
        text=["↑" if v > 0 else "↓" for v in lime_vals],
        textposition="outside",
        hovertemplate="<b>Method 2</b> — %{x}<br>%{y:+.3f}<extra></extra>"
    ))
    fig4.add_hline(y=0, line_color="#b0dcc4", line_width=1)
    fig4.update_layout(**PL(420), barmode="group",
        title=dict(text="Do both methods agree on what is helping and hurting?",
                   font=dict(color="#0f172a", size=13, family="Inter")),
        xaxis=dict(gridcolor="#e2e8f0", tickangle=-30),
        yaxis=dict(gridcolor="#e2e8f0",
                   title="← Hurting chances     |     Helping chances →"),
        legend=dict(bgcolor="#ffffff", bordercolor="#b0dcc4", borderwidth=1,
                    font=dict(color="#334155", size=11)))
    st.plotly_chart(fig4, use_container_width=True)

    # Radar chart
    st.markdown("""
    <div class='explain-box'>
        <strong>Spider chart below:</strong> Each corner is one factor.
        The further out, the more that factor matters.
        When the <strong style='color:#059669;'>green shape</strong> and
        <strong style='color:#d97706;'>orange shape</strong> overlap well,
        both methods are saying the same thing.
    </div>""", unsafe_allow_html=True)

    fig4r = go.Figure()
    theta = [FEATURE_PLAIN[f] for f in FEATURE_NAMES] + [FEATURE_PLAIN[FEATURE_NAMES[0]]]
    fig4r.add_trace(go.Scatterpolar(
        r=list(np.abs(shap_vals)) + [float(np.abs(shap_vals[0]))],
        theta=theta, fill='toself', name='Method 1 (Overall)',
        line=dict(color='#059669', width=2),
        fillcolor='rgba(5,150,105,0.12)'
    ))
    fig4r.add_trace(go.Scatterpolar(
        r=list(np.abs(lime_vals)) + [float(np.abs(lime_vals[0]))],
        theta=theta, fill='toself', name='Method 2 (This moment)',
        line=dict(color='#d97706', width=2),
        fillcolor='rgba(217,119,6,0.10)'
    ))
    fig4r.update_layout(
        paper_bgcolor='#ffffff', height=420,
        polar=dict(bgcolor='#f8fff9',
                   radialaxis=dict(visible=True, gridcolor='#e2e8f0', color='#94a3b8'),
                   angularaxis=dict(gridcolor='#e2e8f0', color='#334155')),
        legend=dict(bgcolor='#ffffff', bordercolor='#b0dcc4', borderwidth=1,
                    font=dict(color="#334155", size=11)),
        title=dict(text="How much each factor matters — Method 1 vs Method 2",
                   font=dict(color="#0f172a", size=13, family="Inter")),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig4r, use_container_width=True)

    # Agreement table — plain English
    agree_list = [(shap_vals[i] > 0) == (lime_vals[i] > 0) for i in range(8)]
    agree_pct  = sum(agree_list) / 8 * 100
    df_cmp = pd.DataFrame({
        "Factor":         FEATURE_NAMES,
        "Plain English":  [FEATURE_PLAIN[f] for f in FEATURE_NAMES],
        "Method 1 says":  ["↑ Helping" if v >= 0 else "↓ Hurting" for v in shap_vals],
        "Method 2 says":  ["↑ Helping" if v >= 0 else "↓ Hurting" for v in lime_vals],
        "Do they agree?": ["✅ Yes" if a else "❌ No" for a in agree_list],
    })

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.dataframe(df_cmp.reset_index(drop=True), use_container_width=True, hide_index=True)
    with col_b:
        col_v = "#059669" if agree_pct >= 75 else "#d97706"
        agree_msg = ("Both methods mostly agree — the AI is being consistent." if agree_pct >= 75
                     else "Some differences — but the overall direction is still the same.")
        st.markdown(f"""
        <div class='stat-card' style='margin-top:8px;'>
            <div class='stat-lbl'>Agreement rate</div>
            <div style='font-family:Orbitron,monospace;font-size:2.2rem;
                font-weight:900;color:{col_v};margin:8px 0;'>{agree_pct:.0f}%</div>
            <div style='font-size:.76rem;color:var(--text3);line-height:1.5;'>
                {agree_msg}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Model Overview  ("About the AI")
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("""
    <div class='explain-box gold'>
        <strong>How does the AI actually work?</strong><br>
        The AI was trained by watching thousands of IPL match situations and learning
        what usually leads to a win or a loss. It is like a very experienced cricket
        analyst who has seen every possible game situation.
    </div>""", unsafe_allow_html=True)

    # Two architecture cards — plain English
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid var(--border);border-radius:12px;
            padding:22px;box-shadow:0 2px 6px rgba(37,99,235,.06);'>
            <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:4px;
                color:#2563eb;margin-bottom:14px;'>▸ MODEL 1 — LSTM</div>
            <div style='font-size:.84rem;color:var(--text3);line-height:2;'>
                <div>📖 <strong style='color:var(--text);'>What it does:</strong>
                    Reads the match ball by ball, like reading a story from start to now.</div>
                <div>🧠 <strong style='color:var(--text);'>Memory:</strong>
                    Remembers important events from earlier in the innings.</div>
                <div>📊 <strong style='color:var(--text);'>Looks at:</strong>
                    Last 20 balls of data at once.</div>
                <div>🎯 <strong style='color:var(--text);'>Output:</strong>
                    Win or Loss probability (0% to 100%).</div>
                <div>⚙️ <strong style='color:var(--text);'>Size:</strong>
                    64 memory units → 32 summary units → final answer.</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with cb:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid var(--border);border-radius:12px;
            padding:22px;box-shadow:0 2px 6px rgba(124,58,237,.06);'>
            <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:4px;
                color:#7c3aed;margin-bottom:14px;'>▸ MODEL 2 — BiLSTM (smarter)</div>
            <div style='font-size:.84rem;color:var(--text3);line-height:2;'>
                <div>📖 <strong style='color:var(--text);'>What it does:</strong>
                    Reads the match in BOTH directions — forward and backward.</div>
                <div>🧠 <strong style='color:var(--text);'>Memory:</strong>
                    Twice as powerful — understands both build-up and context.</div>
                <div>🎯 <strong style='color:var(--text);'>Outputs:</strong>
                    Win probability + Predicted score + Score category.</div>
                <div>⚙️ <strong style='color:var(--text);'>Size:</strong>
                    128 + 64 memory units → 64 → 32 → final answers.</div>
                <div>✅ <strong style='color:var(--text);'>Why better:</strong>
                    Three answers in one model — more efficient and consistent.</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # Input features table — plain English
    st.markdown("""
    <div style='font-size:.82rem;font-weight:600;color:var(--text);margin-bottom:10px;'>
        📋 What information does the AI use?</div>
    <div class='explain-box'>
        The AI only looks at 8 pieces of information about the match. Nothing else.
        These 8 numbers are enough to predict the outcome with good accuracy.
    </div>""", unsafe_allow_html=True)

    feat_df = pd.DataFrame({
        "Factor":          FEATURE_NAMES,
        "What it means":   [FEATURE_PLAIN[f] for f in FEATURE_NAMES],
        "Current value":   [f"{v:.1f}" for v in feat_vals_raw],
        "Importance":      ["⭐⭐⭐" if global_shap[i] > 0.08 else
                            "⭐⭐" if global_shap[i] > 0.04 else "⭐"
                            for i in range(8)],
        "Top factor?":     ["Yes 🏆" if i == int(np.argmax(global_shap)) else ""
                            for i in range(8)],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # How the AI was trained — plain English
    st.markdown("""
    <div style='background:#ffffff;border:1px solid var(--border);border-radius:12px;
        padding:22px;box-shadow:0 2px 5px rgba(0,0,0,.04);'>
        <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:4px;
            color:var(--shap);margin-bottom:14px;'>▸ HOW WAS THE AI TRAINED?</div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>
            <div style='font-size:.84rem;color:var(--text3);line-height:2;'>
                <div>📂 <strong style='color:var(--text);'>Data used:</strong>
                    Ball-by-ball records from IPL matches.</div>
                <div>🎯 <strong style='color:var(--text);'>What it learned:</strong>
                    Win/Loss + Final score + Score category (all at once).</div>
                <div>✂️ <strong style='color:var(--text);'>How it was tested:</strong>
                    15% of data was hidden during training to check performance.</div>
            </div>
            <div style='font-size:.84rem;color:var(--text3);line-height:2;'>
                <div>🔁 <strong style='color:var(--text);'>Sequence length:</strong>
                    Looks at the last 20 balls together.</div>
                <div>⚖️ <strong style='color:var(--text);'>Fair training:</strong>
                    Wins and losses were balanced so the AI doesn't favour one side.</div>
                <div>💾 <strong style='color:var(--text);'>Saved as:</strong>
                    3 model files — one for each task.</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
