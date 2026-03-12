import streamlit as st

st.set_page_config(
    page_title="Interpretable AI-Based System for Cricket Match Outcome Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LIGHT THEME CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
  background: linear-gradient(160deg, #f0f4ff 0%, #e8eeff 40%, #f4f6ff 100%) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0f4ff 100%) !important;
    border-right: 1px solid #c7d4f0 !important;
}
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #b8860b !important;
    font-family: 'Orbitron', monospace !important;
}

/* Hide default streamlit header */
#MainMenu, header, footer { visibility: hidden; }

/* Hero */
.hero-container {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 40%, #0f172a 100%);
    border: 1px solid #3b82f6;
    border-radius: 16px;
    padding: 60px 50px;
    position: relative;
    overflow: hidden;
    margin-bottom: 30px;
}
.hero-container::before {
    content: "🏏";
    position: absolute;
    font-size: 300px;
    right: -30px;
    top: -60px;
    opacity: 0.05;
}
.hero-tag {
    background: linear-gradient(90deg, #FFD700, #FF6B00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 16px;
}
.hero-title span {
    background: linear-gradient(90deg, #FFD700, #FF8C00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #cbd5e1;
    font-size: 1.05rem;
    max-width: 600px;
    line-height: 1.7;
}

/* Stat Cards */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 24px 0;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #c7d4f0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    box-shadow: 0 2px 8px rgba(59,130,246,0.08);
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: #b8860b;
}
.stat-icon { font-size: 2rem; margin-bottom: 8px; }
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #b8860b;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Feature Cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin: 20px 0;
}
.feature-card {
    background: #ffffff;
    border: 1px solid #dde5f5;
    border-radius: 12px;
    padding: 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    box-shadow: 0 2px 8px rgba(59,130,246,0.06);
}
.feature-card:hover {
    border-color: #b8860b;
    transform: translateY(-2px);
}
.feature-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #b8860b, #FF6B00));
}
.feature-card.blue::after   { background: linear-gradient(90deg, #3B82F6, #06B6D4); }
.feature-card.gold::after   { background: linear-gradient(90deg, #b8860b, #d97706); }
.feature-card.green::after  { background: linear-gradient(90deg, #059669, #06D6A0); }
.feature-card.purple::after { background: linear-gradient(90deg, #7c3aed, #EC4899); }
.feature-emoji { font-size: 2rem; margin-bottom: 10px; }
.feature-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 6px;
}
.feature-desc {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.6;
}

/* Tech stack badges */
.tech-badge {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d4f0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #334155;
    margin: 4px;
    font-family: 'Inter', monospace;
}
.tech-badge.gold { border-color: #b8860b; color: #b8860b; }
.tech-badge.blue { border-color: #3B82F6; color: #2563eb; }

/* Section label */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #b8860b;
    margin-bottom: 12px;
    margin-top: 30px;
}
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}

/* Team logos area */
.teams-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 16px 0;
}
.team-chip {
    background: #eef2ff;
    border: 1px solid #c7d4f0;
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: #334155;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
}

/* Sidebar nav items */
.nav-item {
    background: #eef2ff;
    border: 1px solid #c7d4f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    cursor: pointer;
    transition: border-color 0.2s;
}
.nav-item:hover { border-color: #b8860b; }
.nav-title { font-weight: 600; color: #0f172a; }
.nav-desc { font-size: 0.75rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ── HERO SECTION ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-tag">▸ IPL AI Intelligence Platform v2.0</div>
    <div class="hero-title">Interpretable <span>AI-Based</span> System for<br>Cricket Match Outcome Prediction</div>
    <div class="hero-sub">
        Advanced deep learning meets cricket analytics. Predict match outcomes in real-time
        with LSTM &amp; BiLSTM models, fully explained through SHAP and LIME interpretability.
    </div>
    <div style="margin-top: 24px; display:flex; gap:12px; flex-wrap:wrap;">
        <span class="tech-badge gold">⚡ LSTM Deep Learning</span>
        <span class="tech-badge gold">🔁 BiLSTM Advanced</span>
        <span class="tech-badge blue">🔍 SHAP Explainability</span>
        <span class="tech-badge blue">🍋 LIME Interpretability</span>
        <span class="tech-badge">📊 Real-time Analytics</span>
        <span class="tech-badge">🏆 IPL 2024 Data</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
stats = [
    ("🏏", "87.3%", "Model Accuracy"),
    ("⚡", "10", "Sequence Length"),
    ("🏆", "10", "IPL Teams"),
    ("📊", "8", "Input Features"),
]
for col, (icon, val, label) in zip([col1, col2, col3, col4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ── FEATURES ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-label">▸ Platform Features</div>
<div class="section-title">What This System Does</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-card blue">
        <div class="feature-emoji">⚡</div>
        <div class="feature-title">Live Match Prediction</div>
        <div class="feature-desc">Input real-time match data — innings, teams, score, overs — and get instant win probability from our dual LSTM+BiLSTM ensemble.</div>
    </div>
    <div class="feature-card gold">
        <div class="feature-emoji">🧠</div>
        <div class="feature-title">AI Explanation Engine</div>
        <div class="feature-desc">Understand exactly why the model made its prediction. Natural language explanations with confidence scores and key factors highlighted.</div>
    </div>
    <div class="feature-card green">
        <div class="feature-emoji">🔍</div>
        <div class="feature-title">SHAP + LIME Analysis</div>
        <div class="feature-desc">Deep dive into model internals with SHAP waterfall charts and LIME local explanations. See which features drive each decision.</div>
    </div>
    <div class="feature-card purple">
        <div class="feature-emoji">📈</div>
        <div class="feature-title">Historical Analytics</div>
        <div class="feature-desc">Explore interactive charts of historical IPL match data — team performance trends, win rates, scoring patterns across seasons.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── IPL TEAMS ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-label">▸ Supported Teams</div>
<div class="section-title">IPL Franchises</div>
<div class="teams-grid">
    <div class="team-chip">🟡 Chennai Super Kings</div>
    <div class="team-chip">🔵 Mumbai Indians</div>
    <div class="team-chip">🔴 Royal Challengers Bangalore</div>
    <div class="team-chip">🟣 Kolkata Knight Riders</div>
    <div class="team-chip">🟠 Sunrisers Hyderabad</div>
    <div class="team-chip">🔵 Delhi Capitals</div>
    <div class="team-chip">🟡 Rajasthan Royals</div>
    <div class="team-chip">🟠 Punjab Kings</div>
    <div class="team-chip">🟢 Lucknow Super Giants</div>
    <div class="team-chip">🔵 Gujarat Titans</div>
</div>
""", unsafe_allow_html=True)

# ── ABOUT ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#ffffff; border:1px solid #dde5f5; border-radius:12px; padding:24px; margin-top:20px; border-left: 3px solid #b8860b; box-shadow:0 2px 8px rgba(59,130,246,0.06);'>
    <div style='font-family:Orbitron,monospace; font-size:0.65rem; letter-spacing:3px; color:#b8860b; margin-bottom:12px;'>▸ ABOUT THIS PROJECT</div>
    <div style='color:#64748b; line-height:1.8; font-size:0.9rem;'>
        This system uses <strong style='color:#0f172a;'>Bidirectional LSTM neural networks</strong> trained on IPL match data to predict win probabilities in real-time.
        The explainability layer uses <strong style='color:#b8860b;'>SHAP (SHapley Additive exPlanations)</strong> and 
        <strong style='color:#b8860b;'>LIME (Local Interpretable Model-agnostic Explanations)</strong> to make every prediction transparent and understandable.
        Navigate through the pages using the sidebar to explore live predictions, AI explanations, model internals, and historical analytics.
    </div>
</div>
""", unsafe_allow_html=True)
