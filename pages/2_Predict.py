"""2_Predict.py — Win/Loss + Score + Score Bucket Prediction"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Predict", page_icon="⚡", layout="wide")

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#f0f4ff;--bg2:#ffffff;--bg3:#f8faff;--surface:#eef2ff;
  --border:#c7d4f0;--text:#0f172a;--text2:#334155;--text3:#64748b;--text4:#94a3b8;
  --gold:#b8860b;--gold2:#d97706;--win:#059669;--loss:#dc2626;--blue:#2563eb;--purple:#7c3aed;
  --inp-bg:#ffffff;--hdr-border:#3b82f6;
}
*,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text2)!important;}
#MainMenu,header,footer{visibility:hidden;}
label{color:var(--text2)!important;font-size:.82rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div,div[data-baseweb="select"]>div{
  background:var(--inp-bg)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
.stSlider>div>div>div{background:var(--gold)!important;}
.stButton>button{
  background:linear-gradient(90deg,var(--gold),var(--gold2))!important;
  color:#ffffff!important;border:none!important;border-radius:10px!important;
  font-family:Orbitron,monospace!important;font-size:.9rem!important;
  font-weight:900!important;letter-spacing:3px!important;
  padding:16px 0!important;width:100%!important;
  box-shadow:0 4px 14px rgba(184,134,11,.25)!important;}
.stButton>button:hover{box-shadow:0 6px 20px rgba(184,134,11,.4)!important;}
.page-hdr{background:linear-gradient(135deg,#1d4ed8 0%,#1e3a8a 55%,#0f172a 100%);
  border:1px solid var(--hdr-border);border-radius:18px;padding:40px 48px;margin-bottom:28px;position:relative;overflow:hidden;}
.page-hdr::before{content:'⚡';position:absolute;font-size:220px;right:-20px;top:-50px;opacity:.03;}
.inp-head{font-family:Orbitron,monospace;font-size:.6rem;letter-spacing:4px;color:var(--gold);text-transform:uppercase;margin-bottom:18px;}
.verdict{text-align:center;padding:22px 0 14px;}
.v-label{font-family:Orbitron,monospace;font-size:.62rem;letter-spacing:4px;color:var(--text4);margin-bottom:10px;}
.v-badge{display:inline-block;padding:10px 36px;border-radius:40px;
  font-family:Orbitron,monospace;font-size:2rem;font-weight:900;letter-spacing:4px;}
.v-win{background:linear-gradient(90deg,#059669,#10b981);color:#fff;}
.v-loss{background:linear-gradient(90deg,#dc2626,#ef4444);color:#fff;}
.metric-box{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center;}
.metric-lbl{font-size:.6rem;letter-spacing:2px;color:var(--text4);font-family:Orbitron,monospace;text-transform:uppercase;}
.metric-val{font-family:Orbitron,monospace;font-size:1rem;font-weight:700;margin-top:5px;}
.sep{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:16px 0;}
.section-lbl{font-family:Orbitron,monospace;font-size:.6rem;letter-spacing:4px;
  color:var(--gold);text-transform:uppercase;margin:14px 0 10px;}
.score-card{background:#ffffff;border:2px solid var(--border);border-radius:14px;
  padding:20px 24px;text-align:center;position:relative;overflow:hidden;}
.score-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.score-card.low::before    {background:linear-gradient(90deg,#94a3b8,#64748b);}
.score-card.below::before  {background:linear-gradient(90deg,#f59e0b,#d97706);}
.score-card.par::before    {background:linear-gradient(90deg,#3b82f6,#2563eb);}
.score-card.good::before   {background:linear-gradient(90deg,#10b981,#059669);}
.score-card.excel::before  {background:linear-gradient(90deg,#b8860b,#FFD700);}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hdr">
    <div style='font-family:Orbitron,monospace;font-size:.6rem;letter-spacing:5px;
        color:#FFD700;text-transform:uppercase;margin-bottom:10px;'>Real-Time Prediction Engine</div>
    <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;color:#fff;'>
        Win · Score · <span style="background:linear-gradient(90deg,#FFD700,#FF8C00);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Bucket Predictor</span>
    </div>
    <div style='color:rgba(255,255,255,.7);font-size:.9rem;margin-top:8px;'>
        Multi-output BiLSTM — Win/Loss probability · Projected score · Score category breakdown
    </div>
</div>
""", unsafe_allow_html=True)

# ── SCORE BUCKET CONFIG ───────────────────────────────────────────────────────
BUCKET_NAMES  = ['Low (<100)', 'Below Par (100-139)', 'Par (140-169)',
                 'Good (170-199)', 'Excellent (200+)']
BUCKET_COLORS = ['#94a3b8', '#f59e0b', '#3b82f6', '#10b981', '#b8860b']
BUCKET_CSS    = ['low', 'below', 'par', 'good', 'excel']
BUCKET_EMOJIS = ['📉', '📊', '🎯', '✅', '🔥']

# ── LOAD ENGINE ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    try:
        from utils.prediction_engine import preprocess_input, predict_match, get_teams
        return preprocess_input, predict_match, get_teams(), True
    except Exception as e:
        return None, None, [
            "Chennai Super Kings","Mumbai Indians","Royal Challengers Bangalore",
            "Kolkata Knight Riders","Sunrisers Hyderabad","Delhi Capitals",
            "Rajasthan Royals","Punjab Kings","Lucknow Super Giants","Gujarat Titans",
            "Deccan Chargers","Kochi Tuskers Kerala","Pune Warriors",
            "Rising Pune Supergiant","Gujarat Lions"
        ], False

preprocess_fn, predict_fn, TEAMS, MODEL_OK = load_engine()
if not MODEL_OK:
    st.info("🔧 Demo mode — model files not found. Showing simulated predictions.")

# ── LAYOUT ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="inp-head">▸ Match State Inputs</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: inning = st.selectbox("🏏 Inning", [1, 2])
    with c2: overs  = st.number_input("🕐 Overs Completed", 0.0, 20.0, 10.0, step=0.1, format="%.1f")

    batting_team = st.selectbox("🟡 Batting Team", TEAMS)
    bowling_team = st.selectbox("🔵 Bowling Team", [t for t in TEAMS if t != batting_team])

    c3, c4 = st.columns(2)
    with c3: current_score = st.number_input("📊 Current Score", 0, 300, 95)
    with c4: wickets       = st.slider("💀 Wickets Fallen", 0, 10, 2)

    predict_btn = st.button("⚡  PREDICT NOW")

# ── DEMO SIMULATION ───────────────────────────────────────────────────────────
def simulate_demo(current_score, wickets, overs, inning, run_rate, rem_overs):
    np.random.seed(int(current_score + wickets * 10 + overs * 3))
    wp = float(np.clip(
        0.5 + (run_rate - 8) * 0.04 - wickets * 0.035 + (inning - 1.5) * -0.05
        + np.random.normal(0, .03), .05, .95))

    proj = int(np.clip(current_score + run_rate * rem_overs * (1 - wickets * 0.018), 50, 280))

    raw_probs = np.array([
        max(0.01, 0.15 - proj * 0.001),
        max(0.01, 0.20 - abs(proj - 120) * 0.003),
        max(0.01, 0.30 - abs(proj - 155) * 0.004),
        max(0.01, 0.25 - abs(proj - 180) * 0.004),
        max(0.01, 0.10 + (proj - 190) * 0.003),
    ]).clip(0.01)
    raw_probs = raw_probs / raw_probs.sum()
    bucket_idx = int(np.argmax(raw_probs))

    return {
        "prediction":           "WIN" if wp >= .5 else "LOSS",
        "win_probability":      wp,
        "loss_probability":     1 - wp,
        "lstm_probability":     float(np.clip(wp + np.random.normal(0, .025), .05, .95)),
        "bilstm_probability":   float(np.clip(wp + np.random.normal(0, .025), .05, .95)),
        "predicted_score":      float(proj),
        "score_bucket":         bucket_idx,
        "score_bucket_name":    BUCKET_NAMES[bucket_idx],
        "bucket_probabilities": {
            BUCKET_NAMES[i]: float(raw_probs[i]) for i in range(5)
        },
    }

# ── OUTPUT PANEL ──────────────────────────────────────────────────────────────
with right:
    if predict_btn:
        ball_number = int(overs * 6)
        run_rate    = current_score / (overs + 1e-6)
        rem_overs   = 20.0 - overs

        if MODEL_OK:
            with st.spinner("🤖 Running Multi-Output BiLSTM..."):
                try:
                    X_flat = preprocess_fn(inning, batting_team, bowling_team,
                                           ball_number, current_score, wickets,
                                           run_rate, rem_overs)
                    result = predict_fn(X_flat)
                except Exception as e:
                    st.warning(f"Model error, switching to demo: {e}")
                    result = simulate_demo(current_score, wickets, overs,
                                          inning, run_rate, rem_overs)
        else:
            result = simulate_demo(current_score, wickets, overs,
                                   inning, run_rate, rem_overs)

        # ── Unpack ────────────────────────────────────────────────────────────
        pred       = result["prediction"]
        wp         = result["win_probability"]
        lp         = result["loss_probability"]
        lstm_p     = result.get("lstm_probability", wp)
        bi_p       = result.get("bilstm_probability", wp)
        proj_score = result.get("predicted_score",
                                current_score + run_rate * rem_overs)
        bkt_idx    = result.get("score_bucket", 2)
        bkt_name   = result.get("score_bucket_name", BUCKET_NAMES[bkt_idx])
        bkt_probs  = result.get("bucket_probabilities", {})

        badge_cls = "v-win" if pred == "WIN" else "v-loss"
        pct_color = "var(--win)" if pred == "WIN" else "var(--loss)"

        # ════════════════════════════════════════════════════════════════
        #  SECTION 1 — Win / Loss verdict
        # ════════════════════════════════════════════════════════════════
        st.markdown(f"""
        <div class="verdict">
            <div class="v-label">AI PREDICTION</div>
            <div class="v-badge {badge_cls}">{pred}</div>
            <div style="font-family:Orbitron,monospace;font-size:2.8rem;font-weight:900;
                color:{pct_color};margin-top:10px;">{wp*100:.1f}%</div>
            <div style="color:var(--text4);font-size:.78rem;margin-top:2px;">win probability</div>
        </div>""", unsafe_allow_html=True)

        # Split bar
        st.markdown(f"""
        <div>
            <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
                <span style='font-family:Orbitron,monospace;font-size:.68rem;
                    font-weight:700;color:var(--win);'>WIN {wp*100:.1f}%</span>
                <span style='font-family:Orbitron,monospace;font-size:.68rem;
                    font-weight:700;color:var(--loss);'>LOSS {lp*100:.1f}%</span>
            </div>
            <div style='background:var(--border);border-radius:50px;height:16px;overflow:hidden;'>
                <div style='width:{wp*100:.1f}%;background:linear-gradient(90deg,
                    var(--win),#10b981);height:100%;border-radius:50px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  SECTION 2 — Projected Score + Score Bucket cards
        # ════════════════════════════════════════════════════════════════
        sa, sb = st.columns(2)

        with sa:
            st.markdown(f"""
            <div style='background:#ffffff;border:2px solid var(--border);
                border-radius:14px;padding:20px;text-align:center;
                position:relative;overflow:hidden;
                box-shadow:0 2px 8px rgba(37,99,235,0.08);'>
                <div style='position:absolute;top:0;left:0;right:0;height:3px;
                    background:linear-gradient(90deg,#2563eb,#06b6d4);'></div>
                <div class="metric-lbl" style="margin-bottom:8px;">🎯 Projected Final Score</div>
                <div style='font-family:Orbitron,monospace;font-size:2.6rem;
                    font-weight:900;color:#2563eb;line-height:1;'>{int(proj_score)}</div>
                <div style='font-size:.72rem;color:var(--text4);margin-top:4px;'>runs</div>
            </div>""", unsafe_allow_html=True)

        with sb:
            bkt_color = BUCKET_COLORS[bkt_idx]
            bkt_css   = BUCKET_CSS[bkt_idx]
            bkt_emoji = BUCKET_EMOJIS[bkt_idx]
            st.markdown(f"""
            <div class="score-card {bkt_css}"
                style='box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div class="metric-lbl" style="margin-bottom:8px;">
                    {bkt_emoji} Score Category</div>
                <div style='font-family:Orbitron,monospace;font-size:1rem;
                    font-weight:900;color:{bkt_color};line-height:1.4;'>
                    {bkt_name}</div>
                <div style='font-size:.72rem;color:var(--text4);margin-top:4px;'>
                    predicted bucket</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  SECTION 3 — Score Bucket Probability Breakdown
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-lbl">▸ Score Bucket Probabilities</div>',
                    unsafe_allow_html=True)

        left_bkt, right_bkt = st.columns([1.1, 1])

        with left_bkt:
            if bkt_probs:
                prob_vals = list(bkt_probs.values())
                max_p     = max(prob_vals) if max(prob_vals) > 0 else 1.0
                top_i     = int(np.argmax(prob_vals))

                for i, (name, prob) in enumerate(bkt_probs.items()):
                    pct    = prob * 100
                    bar_w  = prob / max_p * 100
                    bc     = BUCKET_COLORS[i]
                    is_top = (i == top_i)
                    bold   = "font-weight:700;" if is_top else ""
                    bg     = "background:rgba(0,0,0,0.025);" if is_top else ""
                    border = f"border:1px solid {bc};" if is_top else \
                             "border:1px solid transparent;"

                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:138px 1fr 50px;
                        align-items:center;gap:8px;margin:4px 0;padding:4px 6px;
                        border-radius:8px;{bg}{border}'>
                        <div style='font-size:.75rem;color:var(--text);{bold}
                            white-space:nowrap;overflow:hidden;
                            text-overflow:ellipsis;' title='{name}'>
                            {BUCKET_EMOJIS[i]} {name}
                        </div>
                        <div style='background:var(--border);border-radius:50px;
                            height:11px;overflow:hidden;'>
                            <div style='width:{bar_w:.1f}%;background:{bc};
                                height:100%;border-radius:50px;opacity:.9;'></div>
                        </div>
                        <div style='font-family:Orbitron,monospace;font-size:.7rem;
                            font-weight:700;color:{bc};'>{pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

        with right_bkt:
            # Donut chart
            if bkt_probs:
                fig_donut = go.Figure(go.Pie(
                    labels=list(bkt_probs.keys()),
                    values=list(bkt_probs.values()),
                    hole=0.65,
                    marker=dict(colors=BUCKET_COLORS,
                                line=dict(color='#ffffff', width=2)),
                    textfont=dict(family='Orbitron', size=9),
                    hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                    sort=False,
                ))
                fig_donut.add_annotation(
                    text=f"<b>{int(proj_score)}</b><br>runs",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(family='Orbitron', size=16, color='#2563eb'),
                    align='center'
                )
                fig_donut.update_layout(
                    paper_bgcolor='#ffffff',
                    margin=dict(t=4, b=4, l=4, r=4),
                    height=210, showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  SECTION 4 — Win probability gauge
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-lbl">▸ Win Probability Gauge</div>',
                    unsafe_allow_html=True)

        gauge_color = "#059669" if pred == "WIN" else "#dc2626"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=wp * 100,
            number={"suffix":"%","font":{"color":gauge_color,"size":32,
                                          "family":"Orbitron"}},
            title={"text":"Win Probability",
                   "font":{"color":"#64748b","family":"Orbitron","size":11}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#c7d4f0",
                         "tickfont":{"color":"#94a3b8","size":9}},
                "bar":{"color":gauge_color,"thickness":0.22},
                "bgcolor":"#f8faff","bordercolor":"#c7d4f0",
                "steps":[{"range":[0,30],"color":"#fee2e2"},
                          {"range":[30,50],"color":"#fef9c3"},
                          {"range":[50,70],"color":"#d1fae5"},
                          {"range":[70,100],"color":"#bbf7d0"}],
                "threshold":{"line":{"color":"#b8860b","width":3},
                             "thickness":0.75,"value":50}
            }
        ))
        fig_g.update_layout(paper_bgcolor="#ffffff",
                            margin=dict(t=30, b=10, l=20, r=20), height=200)
        st.plotly_chart(fig_g, use_container_width=True)

        # ════════════════════════════════════════════════════════════════
        #  SECTION 5 — Model breakdown + summary metrics
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-lbl">▸ Model Breakdown</div>',
                    unsafe_allow_html=True)

        for model_lbl, prob, col in [("LSTM",   float(lstm_p), "#2563eb"),
                                      ("BiLSTM", float(bi_p),  "#7c3aed")]:
            p = float(np.clip(prob, 0, 1))
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:68px 1fr 52px;
                align-items:center;gap:10px;margin:5px 0;'>
                <span style='font-family:Orbitron,monospace;font-size:.63rem;
                    color:var(--text4);'>{model_lbl}</span>
                <div style='background:var(--border);border-radius:50px;
                    height:10px;overflow:hidden;'>
                    <div style='width:{p*100:.1f}%;background:{col};
                        height:100%;border-radius:50px;'></div>
                </div>
                <span style='font-family:Orbitron,monospace;font-size:.68rem;
                    font-weight:700;color:{col};'>{p*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        conf = ("High"     if abs(wp - 0.5) > 0.25 else
                "Moderate" if abs(wp - 0.5) > 0.10 else "Low")

        cols_m = st.columns(4)
        for col_w, lbl, val, clr in zip(
            cols_m,
            ["Confidence",  "Run Rate",        "Rem Overs",      "Score Bucket"],
            [conf,          f"{run_rate:.2f}",  f"{rem_overs:.1f}", BUCKET_EMOJIS[bkt_idx]],
            ["var(--gold)", "var(--blue)",      "var(--purple)",  BUCKET_COLORS[bkt_idx]]
        ):
            with col_w:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-lbl">{lbl}</div>
                    <div class="metric-val" style="color:{clr};">{val}</div>
                </div>""", unsafe_allow_html=True)

        # Match state summary strip
        st.markdown(f"""
        <div style='background:var(--bg3);border:1px solid var(--border);
            border-radius:10px;padding:12px 18px;margin-top:14px;
            font-size:.78rem;color:var(--text3);line-height:1.9;'>
            <strong style='color:var(--text2);'>{batting_team}</strong> vs
            <strong style='color:var(--text2);'>{bowling_team}</strong> ·
            <strong style='color:var(--gold);'>{current_score}/{wickets}</strong>
            in <strong style='color:var(--gold);'>{overs:.1f}</strong> overs ·
            RR <strong style='color:var(--blue);'>{run_rate:.2f}</strong> ·
            {rem_overs:.1f} overs left ·
            Projected <strong style='color:#2563eb;'>~{int(proj_score)} runs</strong>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='display:flex;flex-direction:column;align-items:center;
            justify-content:center;height:520px;gap:14px;'>
            <div style='font-size:72px;opacity:.08;'>⚡</div>
            <div style='font-family:Orbitron,monospace;font-size:.85rem;
                color:#c7d4f0;letter-spacing:4px;'>AWAITING INPUT</div>
            <div style='font-size:.76rem;color:var(--text4);'>
                Fill in match state and hit Predict
            </div>
            <div style='display:flex;gap:10px;margin-top:8px;flex-wrap:wrap;
                justify-content:center;'>
                <span style='background:#eef2ff;border:1px solid #c7d4f0;
                    border-radius:20px;padding:4px 14px;font-size:.72rem;
                    color:#64748b;'>🏆 Win / Loss</span>
                <span style='background:#eef2ff;border:1px solid #c7d4f0;
                    border-radius:20px;padding:4px 14px;font-size:.72rem;
                    color:#64748b;'>🎯 Score Prediction</span>
                <span style='background:#eef2ff;border:1px solid #c7d4f0;
                    border-radius:20px;padding:4px 14px;font-size:.72rem;
                    color:#64748b;'>📊 Score Bucket %</span>
            </div>
        </div>""", unsafe_allow_html=True)
