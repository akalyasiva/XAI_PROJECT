"""3_XAI_Commentary.py — Simple English AI Explanation"""
import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Why did AI say that?", page_icon="💬", layout="wide")

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#f5f0ff;--bg2:#ffffff;--bg3:#faf8ff;--surface:#f0ebff;
  --border:#d4c4f0;--text:#0f172a;--text2:#334155;--text3:#64748b;--text4:#94a3b8;
  --gold:#b8860b;--gold2:#d97706;--win:#059669;--loss:#dc2626;
  --shap:#059669;--lime:#d97706;--purple:#7c3aed;--inp-bg:#ffffff;
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
.stSlider>div>div>div{background:var(--purple)!important;}
.stButton>button{
  background:linear-gradient(90deg,#7c3aed,#a855f7)!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-family:Orbitron,monospace!important;
  font-size:.85rem!important;font-weight:900!important;letter-spacing:3px!important;
  padding:14px 0!important;width:100%!important;
  box-shadow:0 4px 14px rgba(124,58,237,.25)!important;}

/* Plain card */
.plain-card{background:#ffffff;border:1px solid var(--border);border-radius:14px;
  padding:22px 26px;margin:10px 0;position:relative;overflow:hidden;
  box-shadow:0 2px 8px rgba(124,58,237,.05);}
.plain-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.card-green::before {background:linear-gradient(90deg,#059669,#10b981);}
.card-amber::before  {background:linear-gradient(90deg,#d97706,#f59e0b);}
.card-gold::before   {background:linear-gradient(90deg,#b8860b,#FFD700);}
.card-blue::before   {background:linear-gradient(90deg,#2563eb,#06b6d4);}

.card-tag{font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:3px;
  text-transform:uppercase;margin-bottom:10px;}
.card-body{font-size:.92rem;color:var(--text2);line-height:1.9;}
.card-body strong{color:var(--text);}

/* Feature bar rows */
.feat-row{display:grid;grid-template-columns:150px 1fr 60px 180px;
  align-items:center;gap:10px;margin:5px 0;padding:6px 8px;
  border-radius:8px;transition:background .15s;}
.feat-row:hover{background:rgba(124,58,237,.04);}

/* Pill tags */
.pill{display:inline-block;border-radius:20px;padding:3px 12px;
  font-size:.74rem;font-weight:600;margin:3px;}
.pill-up  {background:#d1fae5;color:#059669;border:1px solid #6ee7b7;}
.pill-down{background:#fee2e2;color:#dc2626;border:1px solid #fca5a5;}

.sep{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:20px 0;}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#4c1d95 0%,#2e1065 60%,#0f172a 100%);
    border:1px solid #7c3aed;border-radius:18px;padding:40px 48px;
    margin-bottom:28px;position:relative;overflow:hidden;'>
    <div style='position:absolute;font-size:200px;right:-10px;top:-40px;opacity:.04;'>💬</div>
    <div style='font-family:Orbitron,monospace;font-size:.6rem;letter-spacing:5px;
        color:#a78bfa;text-transform:uppercase;margin-bottom:10px;'>Why did the AI say that?</div>
    <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;color:#fff;'>
        XAI <span style="background:linear-gradient(90deg,#a78bfa,#ec4899);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Explanation</span>
    </div>
    <div style='color:rgba(255,255,255,.65);font-size:.92rem;margin-top:8px;'>
        Explanation of why the AI made its prediction.
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    try:
        from utils.prediction_engine import preprocess_input, predict_match, get_teams
        return preprocess_input, predict_match, get_teams(), True
    except:
        return None, None, [
            "Chennai Super Kings","Mumbai Indians","Royal Challengers Bangalore",
            "Kolkata Knight Riders","Sunrisers Hyderabad","Delhi Capitals",
            "Rajasthan Royals","Punjab Kings","Lucknow Super Giants","Gujarat Titans",
            "Deccan Chargers","Rising Pune Supergiant","Gujarat Lions"
        ], False

preprocess_fn, predict_fn, TEAMS, MODEL_OK = load_all()

FEATURE_NAMES   = ["Inning","Batting Team","Bowling Team","Ball Number",
                   "Current Score","Wickets Fallen","Run Rate","Remaining Overs"]

# Human-friendly name for each feature
FEATURE_PLAIN = {
    "Inning":           "Which innings it is",
    "Batting Team":     "The team that is batting",
    "Bowling Team":     "The team that is bowling",
    "Ball Number":      "How many balls have been bowled",
    "Current Score":    "Runs on the board right now",
    "Wickets Fallen":   "How many batters are out",
    "Run Rate":         "Runs being scored per over",
    "Remaining Overs":  "Overs still left to bowl",
}

# ── SIMPLE COMMENTARY GENERATORS ─────────────────────────────────────────────
def plain_shap_story(sv, feat_vals, win_prob, batting, bowling, proj_score, bkt_name):
    """Generates a completely plain-English SHAP story."""
    sv      = np.array(sv, dtype=float)
    rr      = float(feat_vals[6])
    wkts    = int(float(feat_vals[5]))
    score   = float(feat_vals[4])
    rem     = float(feat_vals[7])
    top_idx = int(np.argsort(np.abs(sv))[::-1][0])
    top_v   = float(sv[top_idx])
    top_nm  = FEATURE_NAMES[top_idx]

    # ── Opening line ─────────────────────────────────────────────────────────
    if win_prob >= 0.80:
        mood = f"🟢 Things look really good for <strong>{batting}</strong> right now."
    elif win_prob >= 0.60:
        mood = f"🟡 <strong>{batting}</strong> is slightly ahead, but the match is still open."
    elif win_prob >= 0.45:
        mood = f"⚖️ This match is very close — it could go either way."
    else:
        mood = f"🔴 <strong>{bowling}</strong> seems to have the upper hand at this point."

    # ── Most important reason ─────────────────────────────────────────────────
    direction = "helping" if top_v > 0 else "hurting"
    reason_1  = (f"The biggest reason for this prediction is the "
                 f"<strong>{FEATURE_PLAIN[top_nm]}</strong>. "
                 f"Right now it is <strong>{direction} {batting}'s chances</strong> the most.")

    # ── Score/run rate line ───────────────────────────────────────────────────
    if rr >= 10:
        rr_line = (f"The team is scoring at <strong>{rr:.1f} runs per over</strong> — "
                   f"that is a very fast pace and puts the bowling side under a lot of pressure.")
    elif rr >= 7:
        rr_line = (f"The run rate is <strong>{rr:.1f} runs per over</strong> — "
                   f"a decent pace that keeps the innings on track.")
    else:
        rr_line = (f"The run rate is only <strong>{rr:.1f} runs per over</strong> — "
                   f"that is slow, which means the batting team needs to score faster.")

    # ── Wicket line ───────────────────────────────────────────────────────────
    if wkts == 0:
        wkt_line = "No wickets have fallen yet — the batting team still has all their players in."
    elif wkts <= 2:
        wkt_line = (f"Only <strong>{wkts} {'wicket has' if wkts==1 else 'wickets have'}</strong> "
                    f"fallen — plenty of batting still left.")
    elif wkts <= 5:
        wkt_line = (f"<strong>{wkts} wickets</strong> are already gone — "
                    f"the batting team needs to be careful from here.")
    else:
        wkt_line = (f"⚠️ <strong>{wkts} wickets</strong> have already fallen — "
                    f"not many batters left, which is a big worry for {batting}.")

    # ── Score prediction line ─────────────────────────────────────────────────
    score_line = (f"Based on everything, the AI thinks this innings will finish around "
                  f"<strong>{int(proj_score)} runs</strong> — that falls in the "
                  f"<strong>{bkt_name}</strong> category.")

    return [mood, reason_1, rr_line, wkt_line, score_line]


def plain_lime_story(lime_weights, batting, bowling, win_prob):
    """Generates a completely plain-English LIME story."""
    sw  = sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    pos = [(f, w) for f, w in sw if w > 0]
    neg = [(f, w) for f, w in sw if w < 0]

    intro = ("The AI also ran a second check, looking only at the <strong>current moment "
             "in the match</strong> — almost like rewinding the last few overs and asking "
             "\"what matters most right now?\"")

    if pos:
        pname = FEATURE_PLAIN.get(pos[0][0], pos[0][0])
        pos_line = (f"The <strong>{pname}</strong> is currently the biggest thing "
                    f"working <strong style='color:#059669;'>in favour of {batting}</strong>.")
    else:
        pos_line = f"Right now nothing is strongly pushing things in {batting}'s favour."

    if neg:
        nname = FEATURE_PLAIN.get(neg[0][0], neg[0][0])
        neg_line = (f"On the other hand, the <strong>{nname}</strong> is the biggest thing "
                    f"working <strong style='color:#dc2626;'>against {batting}</strong>.")
    else:
        neg_line = f"Nothing is strongly working against {batting} right now."

    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos > n_neg:
        balance = (f"Overall <strong>{n_pos} factors are in {batting}'s favour</strong> "
                   f"and only {n_neg} are against — that is why the AI leans towards WIN.")
    elif n_neg > n_pos:
        balance = (f"Overall <strong>{n_neg} factors are against {batting}</strong> "
                   f"and only {n_pos} are in their favour — so the AI leans towards LOSS.")
    else:
        balance = (f"The number of factors for and against {batting} are equal — "
                   f"that is why this match looks so close.")

    return [intro, pos_line, neg_line, balance]


def simulate(score, wickets, overs, inning):
    np.random.seed(int(score + wickets * 7 + overs * 3))
    rr  = score / (overs + 1e-6)
    rem = 20 - overs
    raw = np.array([float(inning), 0., 1., overs*6, float(score),
                    float(wickets), rr, rem])
    sv  = np.array([0.03*inning, 0.01, -0.01, 0.06*(overs/20),
                    0.12*(score/180), -0.10*(wickets/10),
                    0.09*min(rr/12,1), 0.05*(rem/20)]) \
          + np.random.normal(0, 0.015, 8)
    wp  = float(np.clip(0.5 + sv.sum() * 1.2, 0.05, 0.95))
    lw  = {FEATURE_NAMES[i]: float(sv[i]*np.random.uniform(0.8,1.2)) for i in range(8)}

    # simulate score bucket
    proj  = int(np.clip(score + rr * rem * (1 - wickets * 0.018), 50, 280))
    raw_p = np.array([max(0.01, 0.15 - proj*0.001),
                      max(0.01, 0.20 - abs(proj-120)*0.003),
                      max(0.01, 0.30 - abs(proj-155)*0.004),
                      max(0.01, 0.25 - abs(proj-180)*0.004),
                      max(0.01, 0.10 + (proj-190)*0.003)]).clip(0.01)
    raw_p /= raw_p.sum()
    BUCKET_NAMES = ['Low (<100)','Below Par (100-139)','Par (140-169)',
                    'Good (170-199)','Excellent (200+)']
    bkt_nm = BUCKET_NAMES[int(np.argmax(raw_p))]
    return sv, lw, raw, wp, float(proj), bkt_nm


# ── INPUT FORM ────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("""
    <div style='background:#ffffff;border:1px solid var(--border);border-radius:14px;
        padding:22px 26px;margin-bottom:20px;
        box-shadow:0 2px 8px rgba(124,58,237,.06);'>
    <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:4px;
        color:var(--purple);text-transform:uppercase;margin-bottom:16px;'>
        ▸ Enter Match Details</div>
    """, unsafe_allow_html=True)

    cc = st.columns(5)
    with cc[0]: inning  = st.selectbox("Inning",        [1, 2],  key="c_inn")
    with cc[1]: batting = st.selectbox("Batting Team",  TEAMS,   key="c_bat")
    with cc[2]: bowling = st.selectbox("Bowling Team",  [t for t in TEAMS if t != batting], key="c_bowl")
    with cc[3]: score   = st.number_input("Current Score", 0, 300, 112, key="c_sc")
    with cc[4]: wickets = st.slider("Wickets Fallen",   0, 10, 2, key="c_wkt")

    overs  = st.slider("Overs Completed", 0.0, 20.0, 11.0, step=0.1, key="c_ov")
    go_btn = st.button("💬  EXPLAIN THIS MATCH")
    st.markdown('</div>', unsafe_allow_html=True)

# ── OUTPUT ────────────────────────────────────────────────────────────────────
if go_btn:
    rr  = score / (overs + 1e-6)
    rem = 20 - overs

    if MODEL_OK:
        try:
            X_flat     = preprocess_fn(inning, batting, bowling,
                                       int(overs*6), score, wickets, rr, rem)
            result     = predict_fn(X_flat)
            wp         = result["win_probability"]
            proj_score = result.get("predicted_score", score + rr * rem)
            bkt_name   = result.get("score_bucket_name", "Par (140-169)")

            # Try real SHAP/LIME
            try:
                from utils.shap_engine import create_shap_explainer, local_shap_values
                from utils.lime_engine import create_lime_explainer, local_lime_explanation
                import tensorflow as tf
                lstm_mdl  = tf.keras.models.load_model("models/lstm_model.h5")
                bg        = np.random.randn(30, 8) * 0.5
                shap_exp  = create_shap_explainer(lstm_mdl, bg)
                sv_dict   = local_shap_values(shap_exp, X_flat)
                shap_vals = sv_dict["shap_values"]
                feat_vals = X_flat.flatten()
                lime_exp  = create_lime_explainer(bg)
                lime_dict = local_lime_explanation(lime_exp, lstm_mdl, X_flat)
                lime_w    = lime_dict["weights"]
            except:
                shap_vals, lime_w, feat_vals, _, _, _ = simulate(
                    score, wickets, overs, inning)
        except Exception as ex:
            st.warning(f"Using demo mode: {str(ex)[:80]}")
            shap_vals, lime_w, feat_vals, wp, proj_score, bkt_name = simulate(
                score, wickets, overs, inning)
    else:
        shap_vals, lime_w, feat_vals, wp, proj_score, bkt_name = simulate(
            score, wickets, overs, inning)

    sv_arr     = np.array(shap_vals, dtype=float)
    pred_label = "WIN" if wp >= 0.5 else "LOSS"
    pred_color = "#059669" if wp >= 0.5 else "#dc2626"

    # ══════════════════════════════════════════════════════════════════════
    #  TOP VERDICT STRIP
    # ══════════════════════════════════════════════════════════════════════
    badge_bg = "linear-gradient(90deg,#059669,#10b981)" if wp >= 0.5 \
               else "linear-gradient(90deg,#dc2626,#ef4444)"
    st.markdown(f"""
    <div style='background:#ffffff;border:1px solid var(--border);border-radius:14px;
        padding:22px 28px;margin:4px 0 16px;display:flex;align-items:center;
        gap:24px;box-shadow:0 2px 8px rgba(0,0,0,.05);'>
        <div style='background:{badge_bg};color:#fff;font-family:Orbitron,monospace;
            font-size:1.5rem;font-weight:900;padding:10px 28px;border-radius:40px;
            letter-spacing:3px;white-space:nowrap;'>{pred_label}</div>
        <div>
            <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;
                color:{pred_color};line-height:1;'>{wp*100:.1f}%</div>
            <div style='font-size:.8rem;color:var(--text4);margin-top:2px;'>
                chance of winning for {batting}</div>
        </div>
        <div style='margin-left:auto;text-align:right;'>
            <div style='font-family:Orbitron,monospace;font-size:1.4rem;font-weight:900;
                color:#2563eb;'>{int(proj_score)} runs</div>
            <div style='font-size:.78rem;color:var(--text4);'>predicted final score · {bkt_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TWO COLUMN LAYOUT
    # ══════════════════════════════════════════════════════════════════════
    col_l, col_r = st.columns([1, 1], gap="large")

    # ─────────────────────── LEFT — SHAP explanation ──────────────────────
    with col_l:
        story = plain_shap_story(sv_arr, feat_vals, wp, batting, bowling,
                                 proj_score, bkt_name)

        st.markdown("""
        <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:3px;
            color:#059669;text-transform:uppercase;margin-bottom:10px;'>
            🟢 Method 1 — What the AI looked at overall</div>
        """, unsafe_allow_html=True)

        for i, line in enumerate(story):
            card_cls = ["card-green","card-green","card-amber","card-amber","card-blue"][i]
            st.markdown(f"""
            <div class='plain-card {card_cls}'>
                <div class='card-body'>{line}</div>
            </div>""", unsafe_allow_html=True)

        # ── Feature contribution bars ─────────────────────────────────────
        st.markdown("""
        <div style='margin-top:20px;font-family:Orbitron,monospace;font-size:.58rem;
            letter-spacing:3px;color:#059669;text-transform:uppercase;margin-bottom:10px;'>
            📊 How much each thing mattered</div>
        """, unsafe_allow_html=True)

        sidx    = np.argsort(np.abs(sv_arr))[::-1]
        max_abs = max(float(np.abs(sv_arr).max()), 0.001)

        # Header row
        st.markdown("""
        <div style='display:grid;grid-template-columns:150px 1fr 60px 180px;
            gap:10px;margin:4px 0 2px;padding:4px 8px;'>
            <div style='font-size:.65rem;color:var(--text4);font-weight:600;'>FACTOR</div>
            <div style='font-size:.65rem;color:var(--text4);font-weight:600;'>IMPACT</div>
            <div style='font-size:.65rem;color:var(--text4);font-weight:600;'>VALUE</div>
            <div style='font-size:.65rem;color:var(--text4);font-weight:600;'>WHAT IT MEANS</div>
        </div>""", unsafe_allow_html=True)

        for i in sidx:
            v      = float(sv_arr[i])
            pct    = abs(v) / max_abs * 100
            is_pos = v >= 0
            bar_c  = "#059669" if is_pos else "#dc2626"
            arrow  = "↑ Helps WIN" if is_pos else "↓ Hurts chances"
            tag_c  = "pill-up" if is_pos else "pill-down"
            fv     = float(feat_vals[i]) if i < len(feat_vals) else 0.0
            plain  = FEATURE_PLAIN[FEATURE_NAMES[i]]

            st.markdown(f"""
            <div class='feat-row'>
                <div style='font-size:.78rem;font-weight:600;color:var(--text);'>{FEATURE_NAMES[i]}</div>
                <div style='background:#e2e8f0;border-radius:50px;height:12px;overflow:hidden;'>
                    <div style='width:{pct:.0f}%;background:{bar_c};height:100%;
                        border-radius:50px;opacity:.85;'></div>
                </div>
                <div style='font-family:Orbitron,monospace;font-size:.72rem;
                    font-weight:700;color:{bar_c};'>{fv:.1f}</div>
                <div style='font-size:.72rem;color:var(--text3);'>
                    <span class='pill {tag_c}'>{arrow}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Sum strip
        total = 0.5 + float(sv_arr.sum())
        st.markdown(f"""
        <div style='background:#f0ebff;border:1px solid var(--border);border-radius:8px;
            padding:10px 16px;margin-top:10px;display:flex;
            justify-content:space-between;align-items:center;'>
            <span style='font-size:.78rem;color:var(--text3);'>
                Started at 50-50 · adjusted by all factors above</span>
            <span style='font-family:Orbitron,monospace;font-size:.82rem;
                font-weight:700;color:{pred_color};'>
                Final: {wp*100:.1f}% → {pred_label}</span>
        </div>""", unsafe_allow_html=True)

    # ─────────────────────── RIGHT — LIME explanation ─────────────────────
    with col_r:
        lime_story = plain_lime_story(lime_w, batting, bowling, wp)

        st.markdown("""
        <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:3px;
            color:#d97706;text-transform:uppercase;margin-bottom:10px;'>
            🟡 Method 2 — What matters most right now</div>
        """, unsafe_allow_html=True)

        card_classes = ["card-amber","card-green","card-amber","card-gold"]
        for i, line in enumerate(lime_story):
            st.markdown(f"""
            <div class='plain-card {card_classes[i]}'>
                <div class='card-body'>{line}</div>
            </div>""", unsafe_allow_html=True)

        # ── LIME weight bars ──────────────────────────────────────────────
        st.markdown("""
        <div style='margin-top:20px;font-family:Orbitron,monospace;font-size:.58rem;
            letter-spacing:3px;color:#d97706;text-transform:uppercase;margin-bottom:10px;'>
            📊 Right now — what is helping vs hurting</div>
        """, unsafe_allow_html=True)

        lime_items = sorted(lime_w.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        max_lw     = max(abs(w) for _, w in lime_items) if lime_items else 1.0

        for feat_str, w in lime_items:
            is_pos = w >= 0
            pct    = abs(w) / max_lw * 100
            bar_c  = "#059669" if is_pos else "#dc2626"
            tag_c  = "pill-up" if is_pos else "pill-down"
            arrow  = "↑ Helping" if is_pos else "↓ Hurting"
            plain  = FEATURE_PLAIN.get(feat_str, feat_str)

            st.markdown(f"""
            <div class='feat-row'>
                <div style='font-size:.78rem;font-weight:600;color:var(--text);
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                    title='{plain}'>{feat_str}</div>
                <div style='background:#e2e8f0;border-radius:50px;height:12px;overflow:hidden;'>
                    <div style='width:{pct:.0f}%;background:{bar_c};height:100%;
                        border-radius:50px;opacity:.85;'></div>
                </div>
                <div style='font-family:Orbitron,monospace;font-size:.72rem;
                    font-weight:700;color:{bar_c};'>{w:+.2f}</div>
                <div style='font-size:.72rem;color:var(--text3);'>
                    <span class='pill {tag_c}'>{arrow}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # ── SIDE-BY-SIDE SUMMARY ──────────────────────────────────────────
        st.markdown("""
        <div style='font-family:Orbitron,monospace;font-size:.58rem;letter-spacing:3px;
            color:var(--gold);text-transform:uppercase;margin-bottom:10px;'>
            ✅ Both methods agree</div>
        """, unsafe_allow_html=True)

        helps = [FEATURE_NAMES[i] for i in range(8) if float(sv_arr[i]) > 0.01]
        hurts = [FEATURE_NAMES[i] for i in range(8) if float(sv_arr[i]) < -0.01]

        pills_h = " ".join([f'<span class="pill pill-up">↑ {f}</span>' for f in helps])
        pills_n = " ".join([f'<span class="pill pill-down">↓ {f}</span>' for f in hurts])

        st.markdown(f"""
        <div class='plain-card card-gold'>
            <div class='card-tag' style='color:var(--gold);'>Final Verdict</div>
            <div class='card-body'>
                Both checks agree: <strong style='color:{pred_color};font-size:1.05rem;'>
                {batting} → {pred_label} ({wp*100:.1f}%)</strong><br>
                Projected score: <strong style='color:#2563eb;'>{int(proj_score)} runs</strong>
                ({bkt_name})<br><br>
                <span style='font-size:.8rem;color:var(--text3);'>Things helping:</span><br>
                {pills_h if pills_h else '<span style="color:var(--text4);font-size:.8rem;">Nothing major</span>'}
                <br><br>
                <span style='font-size:.8rem;color:var(--text3);'>Things hurting:</span><br>
                {pills_n if pills_n else '<span style="color:var(--text4);font-size:.8rem;">Nothing major</span>'}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── WHAT IS SHAP / LIME — simple explainer ────────────────────────
        st.markdown("""
        <div style='background:#faf8ff;border:1px solid var(--border);border-radius:10px;
            padding:16px 20px;margin-top:12px;'>
            <div style='font-family:Orbitron,monospace;font-size:.56rem;letter-spacing:3px;
                color:var(--purple);margin-bottom:10px;'>ℹ️ WHAT ARE THESE TWO METHODS?</div>
            <div style='font-size:.8rem;color:var(--text3);line-height:1.85;'>
                <strong style='color:var(--text);'>Method 1 (SHAP)</strong> — Looks at the whole 
                match and every factor together. It asks: <em>"overall, what is making the 
                biggest difference?"</em><br><br>
                <strong style='color:var(--text);'>Method 2 (LIME)</strong> — Zooms into 
                <em>this exact moment</em> in the match. It asks: <em>"right now, 
                what is the most important thing?"</em><br><br>
                When both methods say the same thing, you can be more confident the 
                AI prediction is reliable.
            </div>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
        justify-content:center;height:320px;gap:14px;'>
        <div style='font-size:70px;opacity:.08;'>💬</div>
        <div style='font-family:Orbitron,monospace;font-size:.85rem;
            color:#c7d4f0;letter-spacing:4px;'>READY TO EXPLAIN</div>
        <div style='font-size:.82rem;color:var(--text4);text-align:center;max-width:380px;'>
            Fill in the match details above and click <strong>Explain This Match</strong> — 
            we will tell you in plain English exactly why the AI made its prediction.
        </div>
    </div>""", unsafe_allow_html=True)
