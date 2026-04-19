"""
Phase 5 — Deployment with Streamlit
Smartphone Addiction Risk Predictor

Run locally:
    pip install streamlit pandas numpy matplotlib shap scikit-learn plotly
    streamlit run streamlit_app.py

Deploy free on Streamlit Cloud:
    1. Push project folder to GitHub
    2. Go to share.streamlit.io → New app
    3. Select repo, set main file to streamlit_app.py
    4. Click Deploy — live in ~60 seconds
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "📱 Smartphone Addiction Risk Predictor",
    page_icon   = "📱",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8FAFC; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A3A5C;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: #378ADD;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        padding: 10px 20px;
    }

    /* Recommendation cards */
    .rec-card {
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .rec-title { font-weight: 600; font-size: 15px; margin-bottom: 4px; }
    .rec-detail { font-size: 13px; color: #555; line-height: 1.5; }

    /* Risk badges */
    .badge-low    { background:#E1F5EE; color:#0F6E56; border-radius:20px;
                    padding:6px 18px; font-weight:700; font-size:16px; }
    .badge-medium { background:#FAEEDA; color:#854F0B; border-radius:20px;
                    padding:6px 18px; font-weight:700; font-size:16px; }
    .badge-high   { background:#FAECE7; color:#993C1D; border-radius:20px;
                    padding:6px 18px; font-weight:700; font-size:16px; }

    /* Hide Streamlit branding */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_BIN = os.path.join(BASE, "models", "model_binary_best.pkl")
MODEL_MLC = os.path.join(BASE, "models", "model_multiclass_best.pkl")
RISK_CSV  = os.path.join(BASE, "data",   "risk_profiles.csv")
SHAP_CSV  = os.path.join(BASE, "data",   "shap_mean_importance.csv")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
FEATURES = [
    "age","daily_screen_time_hours","social_media_hours","gaming_hours",
    "work_study_hours","sleep_hours","notifications_per_day","app_opens_per_day",
    "weekend_screen_time","stress_level_enc","academic_impact_enc",
    "gender_Male","gender_Other","social_pct_of_screen","gaming_pct_of_screen",
    "productive_ratio","passive_ratio","weekend_weekday_delta","screen_sleep_ratio",
    "notifications_per_screen_hour","apps_per_screen_hour",
    "addiction_pressure_score","nonproductive_hours",
]

MLC_LABELS = {0:"None (not addicted)", 1:"Mild", 2:"Moderate", 3:"Severe"}
MLC_EMOJI  = {0:"✅", 1:"🔵", 2:"🟡", 3:"🔴"}

C_BLUE   = "#378ADD"; C_CORAL  = "#D85A30"
C_TEAL   = "#1D9E75"; C_AMBER  = "#EF9F27"
C_GRAY   = "#888780"; C_PURPLE = "#7F77DD"
C_NAVY   = "#1A3A5C"

# ── LOAD ASSETS (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    with open(MODEL_BIN, "rb") as f: m_bin = pickle.load(f)
    with open(MODEL_MLC, "rb") as f: m_mlc = pickle.load(f)
    exp = shap.TreeExplainer(m_bin)
    return m_bin, m_mlc, exp

@st.cache_data(show_spinner=False)
def load_data():
    rp = pd.read_csv(RISK_CSV)
    si = pd.read_csv(SHAP_CSV)
    return rp, si

model_bin, model_mlc, explainer = load_models()
risk_profiles, shap_imp         = load_data()

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def engineer(raw: dict) -> pd.DataFrame:
    eps = 1e-6
    dst = raw["daily_screen_time_hours"]
    r   = raw.copy()
    r["social_pct_of_screen"]          = min(r["social_media_hours"]  / (dst+eps), 1)
    r["gaming_pct_of_screen"]          = min(r["gaming_hours"]        / (dst+eps), 1)
    r["productive_ratio"]              = min(r["work_study_hours"]    / (dst+eps), 1)
    r["passive_ratio"]                 = min((r["social_media_hours"]+r["gaming_hours"]) / (dst+eps), 1)
    r["weekend_weekday_delta"]         = r["weekend_screen_time"] - dst
    r["screen_sleep_ratio"]            = dst / (r["sleep_hours"]+eps)
    r["notifications_per_screen_hour"] = r["notifications_per_day"] / (dst+eps)
    r["apps_per_screen_hour"]          = r["app_opens_per_day"]    / (dst+eps)
    r["addiction_pressure_score"]      = min(
        0.45*(dst/12) + 0.35*(r["weekend_screen_time"]/15) + 0.20*(r["social_media_hours"]/6), 1)
    r["nonproductive_hours"]           = max(dst - r["work_study_hours"], 0)
    return pd.DataFrame([r])[FEATURES]

# ── PLOTLY CHARTS ─────────────────────────────────────────────────────────────
def make_gauge(prob: float) -> go.Figure:
    tier  = "Low" if prob<0.33 else "Medium" if prob<0.66 else "High"
    color = C_TEAL if prob<0.33 else C_AMBER if prob<0.66 else C_CORAL
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(prob*100, 1),
        number= {"suffix":"%", "font":{"size":48, "color":color, "family":"Arial"}},
        gauge = {
            "axis"     : {"range":[0,100], "tickwidth":1, "tickcolor":"#ccc",
                          "tickvals":[0,33,66,100]},
            "bar"      : {"color":color, "thickness":0.28},
            "bgcolor"  : "white",
            "borderwidth": 0,
            "steps"    : [
                {"range":[0,33],  "color":"#E1F5EE"},
                {"range":[33,66], "color":"#FAEEDA"},
                {"range":[66,100],"color":"#FAECE7"},
            ],
            "threshold": {"line":{"color":color,"width":4},
                          "thickness":0.75,"value":prob*100},
        },
        title={"text":f"<b>{tier} Risk</b>","font":{"size":20,"color":C_NAVY,"family":"Arial"}},
    ))
    fig.update_layout(
        height=300, margin=dict(t=80,b=20,l=30,r=30),
        paper_bgcolor="white", font={"family":"Arial"},
    )
    return fig


def make_waterfall(X: pd.DataFrame) -> go.Figure:
    sv_raw = explainer.shap_values(X)
    if isinstance(sv_raw, list):    sv = sv_raw[1][0]
    elif sv_raw.ndim == 3:          sv = sv_raw[0,:,1]
    else:                           sv = sv_raw[0]

    sidx   = np.argsort(np.abs(sv))[-12:]
    labels = [FEATURES[i].replace("_"," ") for i in sidx]
    vals   = sv[sidx]
    colors = [C_CORAL if v>0 else C_BLUE for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#555", line_width=1)
    fig.update_layout(
        title=dict(text="<b>What drove this prediction?</b> (SHAP values)",
                   font=dict(size=15, color=C_NAVY)),
        xaxis_title="SHAP value  (🔴 red = raises risk  ·  🔵 blue = lowers risk)",
        height=420, margin=dict(t=60,b=40,l=10,r=20),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font={"family":"Arial","size":12},
        xaxis=dict(gridcolor="#EBEBEB"),
        yaxis=dict(gridcolor="#EBEBEB"),
    )
    return fig


def make_comparison(raw: dict) -> go.Figure:
    cats   = ["Screen time<br>(hrs)","Social media<br>(hrs)","Sleep<br>(hrs)",
              "Weekend screen<br>(hrs)","Pressure<br>score"]
    eps    = 1e-6; dst = raw["daily_screen_time_hours"]
    user_v = [
        raw["daily_screen_time_hours"], raw["social_media_hours"],
        raw["sleep_hours"],             raw["weekend_screen_time"],
        min(0.45*(dst/12)+0.35*(raw["weekend_screen_time"]/15)+0.20*(raw["social_media_hours"]/6),1),
    ]
    feats  = ["avg_screen_time","avg_social_media","avg_sleep","avg_weekend_screen","avg_pressure_score"]
    low_v  = risk_profiles.loc[risk_profiles.risk_tier=="Low",  feats].values[0].tolist()
    high_v = risk_profiles.loc[risk_profiles.risk_tier=="High", feats].values[0].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="🟢 Low-risk avg",  x=cats, y=low_v,  marker_color=C_TEAL,  opacity=0.82))
    fig.add_trace(go.Bar(name="🟣 You",           x=cats, y=user_v, marker_color=C_PURPLE, opacity=0.92))
    fig.add_trace(go.Bar(name="🔴 High-risk avg", x=cats, y=high_v, marker_color=C_CORAL,  opacity=0.82))
    fig.update_layout(
        title=dict(text="<b>Your usage vs population averages</b>",
                   font=dict(size=15, color=C_NAVY)),
        barmode="group", height=380,
        margin=dict(t=60,b=40,l=40,r=20),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font={"family":"Arial","size":12},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(gridcolor="#EBEBEB"),
        xaxis=dict(gridcolor="#EBEBEB"),
    )
    return fig


def make_global_shap() -> go.Figure:
    top10  = shap_imp.head(10).sort_values("mean_shap", ascending=True)
    q70    = top10["mean_shap"].quantile(0.70)
    q40    = top10["mean_shap"].quantile(0.40)
    colors = [C_CORAL if v>q70 else C_AMBER if v>q40 else C_GRAY for v in top10["mean_shap"]]
    fig    = go.Figure(go.Bar(
        x=top10["mean_shap"],
        y=top10["feature"].str.replace("_"," "),
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Top 10 predictors of addiction risk</b>",
                   font=dict(size=15, color=C_NAVY)),
        xaxis_title="Mean |SHAP value|",
        height=400, margin=dict(t=60,b=40,l=10,r=20),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font={"family":"Arial","size":12},
        xaxis=dict(gridcolor="#EBEBEB"),
        yaxis=dict(gridcolor="#EBEBEB"),
    )
    return fig


def make_whatif_gauges(prob_cur: float, prob_sim: float) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type":"indicator"},{"type":"indicator"}]],
                        subplot_titles=["Current risk","Simulated risk"])
    for col, prob in [(1, prob_cur), (2, prob_sim)]:
        color = C_TEAL if prob<0.33 else C_AMBER if prob<0.66 else C_CORAL
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(prob*100,1),
            number={"suffix":"%","font":{"size":36,"color":color}},
            gauge={
                "axis":{"range":[0,100],"tickvals":[0,33,66,100]},
                "bar":{"color":color,"thickness":0.3},
                "steps":[{"range":[0,33],"color":"#E1F5EE"},
                          {"range":[33,66],"color":"#FAEEDA"},
                          {"range":[66,100],"color":"#FAECE7"}],
            },
        ), row=1, col=col)
    fig.update_layout(
        height=280, margin=dict(t=60,b=10,l=30,r=30),
        paper_bgcolor="white", font={"family":"Arial"},
    )
    return fig


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
def get_recs(raw: dict, X: pd.DataFrame) -> list:
    sv_raw = explainer.shap_values(X)
    if isinstance(sv_raw, list):    sv = sv_raw[1][0]
    elif sv_raw.ndim == 3:          sv = sv_raw[0,:,1]
    else:                           sv = sv_raw[0]

    recs = []
    for fi in np.argsort(sv)[::-1][:8]:
        feat   = FEATURES[fi]
        val    = float(X.iloc[0, fi])
        shap_v = float(sv[fi])
        if shap_v <= 0: break

        if feat=="daily_screen_time_hours" and val>6:
            recs.append(("📵","Reduce daily screen time",
                f"You average **{val:.1f} hrs/day**. Low-risk avg is 5.2 hrs. "
                "Try cutting 30 min per week until you reach under 6 hrs."))
        elif feat=="social_media_hours" and val>2.5:
            recs.append(("📲","Limit social media",
                f"**{val:.1f} hrs/day** on social media is above the safe threshold. "
                "Set an app timer to cap it at 1.5 hrs."))
        elif feat=="weekend_screen_time" and val>8:
            recs.append(("🏕️","Plan screen-free weekends",
                f"**{val:.1f} hrs/day** on weekends. Replace one screen block "
                "with outdoor or social activity."))
        elif feat=="addiction_pressure_score" and val>0.6:
            recs.append(("🧘","Lower your composite risk score",
                f"Your pressure score is **{val:.2f}** (high-risk avg: 0.68). "
                "Reduce screen time and social media together."))
        elif feat=="screen_sleep_ratio" and val>1.2:
            recs.append(("😴","Improve sleep balance",
                f"Screen/sleep ratio is **{val:.2f}**. "
                "Aim for ≥7 hrs sleep and no screens 30 min before bed."))
        elif feat=="nonproductive_hours" and val>4:
            recs.append(("🎯","Replace passive use with purpose",
                f"**{val:.1f} hrs/day** of non-productive screen time. "
                "Shift some to learning or creative tasks."))
        elif feat=="notifications_per_screen_hour" and val>18:
            recs.append(("🔕","Reduce notifications",
                f"**{val:.0f} notifications/screen-hour** keeps you tethered. "
                "Turn off non-essential app badges."))

    if not recs:
        recs.append(("✅","Your usage looks healthy!",
                     "No major risk factors detected. Keep up your current habits."))
    return recs[:4]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px'>
        <div style='font-size:2.5rem'>📱</div>
        <div style='font-size:1.1rem; font-weight:700; color:white'>Usage Profile</div>
        <div style='font-size:0.8rem; color:#9EC5E8'>Fill in your details below</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Demographics**")
    age    = st.slider("Age", 18, 35, 24)
    gender = st.radio("Gender", ["Female","Male","Other"],
                      horizontal=True, index=1)

    st.markdown("---")
    st.markdown("**Screen Time**")
    daily_screen   = st.slider("Daily screen time (hrs)",   1.0, 14.0, 7.0, 0.5)
    social_media   = st.slider("Social media (hrs)",        0.5,  6.0, 2.5, 0.5)
    gaming         = st.slider("Gaming (hrs)",              0.0,  4.0, 1.5, 0.5)
    work_study     = st.slider("Work / study (hrs)",        0.5,  6.0, 3.0, 0.5)
    weekend_screen = st.slider("Weekend screen time (hrs)", 3.0, 15.0, 8.5, 0.5)

    st.markdown("---")
    st.markdown("**Health & Habits**")
    sleep    = st.slider("Sleep hours per night",   4.5, 9.0, 7.0, 0.5)
    stress   = st.radio("Stress level", ["Low","Medium","High"],
                         horizontal=True, index=1)
    academic = st.radio("Affects work/study?", ["No","Yes"],
                         horizontal=True)

    st.markdown("---")
    st.markdown("**Phone Behaviour**")
    notifs    = st.slider("Notifications per day", 20,  250, 120, 5)
    app_opens = st.slider("App opens per day",     15,  180,  90, 5)

    st.markdown("---")
    predict_btn = st.button("🔍  Predict My Risk",
                             use_container_width=True, type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Page header
st.markdown(f"""
<div style='padding: 8px 0 16px'>
    <h1 style='color:{C_NAVY}; margin:0; font-size:1.9rem'>
        📱 Smartphone Addiction Risk Predictor
    </h1>
    <p style='color:#666; margin:4px 0 0; font-size:0.95rem'>
        Adjust your usage profile in the sidebar and click
        <strong>Predict My Risk</strong> to get your personalised risk score,
        SHAP explanations, and recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Predict my risk",
    "🔬  What-if simulator",
    "📖  Model & methods",
    "📋  Severity levels",
])

# ════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════
with tab1:
    if not predict_btn:
        st.info("👈  Set your usage profile in the sidebar and click **Predict My Risk** to get started.")

        # Show model stats while waiting
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training users",  "7,500")
        c2.metric("Test ROC-AUC",    "0.989")
        c3.metric("Test F1 score",   "0.952")
        c4.metric("Test accuracy",   "93.2%")

        st.markdown("---")
        with st.expander("📊 View global feature importance"):
            st.plotly_chart(make_whatif_gauges(prob_cur, prob_sim), use_container_width=True, key="whatif")

    else:
        # ── RUN PREDICTION ─────────────────────────────────────────────────
        raw = dict(
            age=age, daily_screen_time_hours=daily_screen,
            social_media_hours=social_media, gaming_hours=gaming,
            work_study_hours=work_study, sleep_hours=sleep,
            notifications_per_day=notifs, app_opens_per_day=app_opens,
            weekend_screen_time=weekend_screen,
            stress_level_enc={"Low":0,"Medium":1,"High":2}[stress],
            academic_impact_enc=1 if academic=="Yes" else 0,
            gender_Male=1 if gender=="Male"  else 0,
            gender_Other=1 if gender=="Other" else 0,
        )
        X    = engineer(raw)
        prob = float(model_bin.predict_proba(X)[0,1])
        mlc  = int(model_mlc.predict(X)[0])
        tier = "Low" if prob<0.33 else "Medium" if prob<0.66 else "High"
        sev  = MLC_LABELS[mlc]
        t_class = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high"}[tier]
        t_emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}[tier]

        # ── ROW 1: Gauge + stats + recommendations ────────────────────────
        col_gauge, col_right = st.columns([1, 2], gap="large")

        with col_gauge:
            st.plotly_chart(make_waterfall(X), use_container_width=True, key="waterfall")
            st.markdown(f"""
            <div style='text-align:center; margin-top:-10px'>
                <span class='{t_class}'>{t_emoji} {tier} Risk</span>
            </div>
            <div style='text-align:center; margin-top:10px;
                        font-size:14px; color:#555'>
                Severity: <strong>{MLC_EMOJI[mlc]} {sev}</strong>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Quick metrics
            delta_s  = daily_screen  - 5.18
            delta_sm = social_media  - 2.33
            delta_sl = sleep         - 6.74
            delta_ws = weekend_screen- 6.92

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk probability",   f"{prob*100:.1f}%",
                      delta=f"{(prob-0.33)*100:+.1f}% vs threshold",
                      delta_color="inverse")
            m2.metric("Daily screen",       f"{daily_screen:.1f} hrs",
                      delta=f"{delta_s:+.1f} vs low-risk avg",
                      delta_color="inverse")
            m3.metric("Social media",       f"{social_media:.1f} hrs",
                      delta=f"{delta_sm:+.1f} vs low-risk avg",
                      delta_color="inverse")
            m4.metric("Sleep",              f"{sleep:.1f} hrs",
                      delta=f"{delta_sl:+.1f} vs avg")

            # Alert
            if tier == "High":
                st.error("🚨 Your usage pattern matches **high-risk users** in the training data. "
                         "Review the recommendations below.")
            elif tier == "Medium":
                st.warning("⚠️ Your usage is **borderline**. Small reductions in screen time "
                           "could meaningfully lower your risk.")
            else:
                st.success("✅ Your usage pattern is associated with **low addiction risk**. "
                           "Keep it up!")

            # Recommendations
            st.markdown("#### 🎯 Personalised recommendations")
            recs = get_recs(raw, X)
            r1, r2 = st.columns(2)
            cols_rec = [r1, r2, r1, r2]
            for i, (icon, title, detail) in enumerate(recs):
                with cols_rec[i]:
                    st.markdown(f"""
                    <div class='rec-card'>
                        <div style='font-size:1.5rem'>{icon}</div>
                        <div class='rec-title'>{title}</div>
                        <div class='rec-detail'>{detail}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ── ROW 2: SHAP + Comparison ──────────────────────────────────────
        col_shap, col_comp = st.columns(2, gap="medium")
        with col_shap:
            st.plotly_chart(make_gauge(prob), use_container_width=True, key="gauge")
        with col_comp:
            st.plotly_chart(make_comparison(raw), use_container_width=True, key="comparison")

        # ── ROW 3: Full feature table (expander) ──────────────────────────
        with st.expander("🔧 Full feature values (debug)"):
            feat_df = X.T.reset_index()
            feat_df.columns = ["Feature","Value"]
            sv_raw  = explainer.shap_values(X)
            sv      = (sv_raw[1][0] if isinstance(sv_raw,list)
                       else sv_raw[0,:,1] if sv_raw.ndim==3
                       else sv_raw[0])
            feat_df["SHAP"] = sv
            feat_df = feat_df.sort_values("SHAP", key=abs, ascending=False)
            st.dataframe(
                feat_df.style.format({"Value":"{:.3f}","SHAP":"{:+.4f}"}),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════
# TAB 2 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    ### 🔬 What if I changed my habits?
    Adjust these sliders to instantly see how changing a single behaviour
    would affect your predicted risk.
    """)

    wa, wb = st.columns(2)
    with wa:
        sim_screen = st.slider("Simulated — daily screen time (hrs)",
                               1.0, 14.0, float(daily_screen), 0.5, key="s1")
        sim_social = st.slider("Simulated — social media (hrs)",
                               0.5, 6.0, float(social_media), 0.5, key="s2")
    with wb:
        sim_sleep  = st.slider("Simulated — sleep hours",
                               4.5, 9.0, float(sleep), 0.5, key="s3")
        sim_wknd   = st.slider("Simulated — weekend screen time (hrs)",
                               3.0, 15.0, float(weekend_screen), 0.5, key="s4")

    # Always compute both probabilities safely
    raw_sim = dict(
        age=age, daily_screen_time_hours=sim_screen,
        social_media_hours=sim_social, gaming_hours=gaming,
        work_study_hours=work_study, sleep_hours=sim_sleep,
        notifications_per_day=notifs, app_opens_per_day=app_opens,
        weekend_screen_time=sim_wknd,
        stress_level_enc={"Low":0,"Medium":1,"High":2}[stress],
        academic_impact_enc=1 if academic=="Yes" else 0,
        gender_Male=1 if gender=="Male" else 0,
        gender_Other=1 if gender=="Other" else 0,
    )
    raw_cur = dict(
        age=age, daily_screen_time_hours=daily_screen,
        social_media_hours=social_media, gaming_hours=gaming,
        work_study_hours=work_study, sleep_hours=sleep,
        notifications_per_day=notifs, app_opens_per_day=app_opens,
        weekend_screen_time=weekend_screen,
        stress_level_enc={"Low":0,"Medium":1,"High":2}[stress],
        academic_impact_enc=1 if academic=="Yes" else 0,
        gender_Male=1 if gender=="Male" else 0,
        gender_Other=1 if gender=="Other" else 0,
    )

    X_sim    = engineer(raw_sim)
    X_cur    = engineer(raw_cur)
    prob_sim = float(model_bin.predict_proba(X_sim)[0, 1])
    prob_cur = float(model_bin.predict_proba(X_cur)[0, 1])
    delta    = (prob_sim - prob_cur) * 100

    st.plotly_chart(make_whatif_gauges(prob_cur, prob_sim),
                    use_container_width=True, key="whatif")

    tier_sim = "Low" if prob_sim<0.33 else "Medium" if prob_sim<0.66 else "High"
    arrow    = "▲" if delta>0.5 else "▼" if delta<-0.5 else "●"
    if delta < -0.5:
        st.success(f"**{arrow} {delta:+.1f} percentage points** — simulated changes reduce your risk "
                   f"({prob_cur*100:.1f}% → {prob_sim*100:.1f}%, {tier_sim} risk)")
    elif delta > 0.5:
        st.error(f"**{arrow} {delta:+.1f} percentage points** — simulated changes increase your risk "
                 f"({prob_cur*100:.1f}% → {prob_sim*100:.1f}%, {tier_sim} risk)")
    else:
        st.info(f"**● Minimal change** — risk stays at ~{prob_sim*100:.1f}% ({tier_sim} risk)")


# ════════════════════════════════════════════════════════
# TAB 3 — MODEL & METHODS
# ════════════════════════════════════════════════════════
with tab3:
    col_info, col_shap = st.columns([1,1], gap="large")

    with col_info:
        st.markdown("### How this model works")
        st.markdown("""
        This tool uses a **Random Forest classifier** trained on 7,500
        smartphone users aged 18–35. It predicts the probability that your
        usage pattern is associated with smartphone addiction.

        | Metric | Value |
        |---|---|
        | Algorithm | Random Forest (400 trees) |
        | Test ROC-AUC | **0.989** |
        | Test F1 score | **0.952** |
        | Test Accuracy | **93.2%** |
        | Training samples | 7,500 users |
        | Features used | 23 (13 raw + 10 engineered) |

        ### Risk tiers
        | Tier | Probability | True addiction rate |
        |---|---|---|
        | 🟢 Low | < 33% | 0.9% |
        | 🟡 Medium | 33–66% | 42.2% |
        | 🔴 High | > 66% | 99.8% |

        ### Key finding
        > Social media hours and daily screen time are the two strongest predictors.
        > Age, gender, stress, and gaming add virtually no predictive signal.

        ---
        *For educational purposes only. Does not constitute medical advice.*
        """)

    with col_shap:
        st.markdown("### Global feature importance")
        st.plotly_chart(make_whatif_gauges(prob_cur, prob_sim), use_container_width=True, key="whatif")
        st.caption("Mean |SHAP| measures each feature's average contribution "
                   "to the model's predictions across all 7,500 users. "
                   "Red = strong predictor · Amber = moderate · Grey = weak.")


# ════════════════════════════════════════════════════════
# TAB 4 — SEVERITY LEVELS
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Severity levels explained")

    c1, c2, c3, c4 = st.columns(4)
    for col, emoji, level, desc, avg in [
        (c1,"✅","None (not addicted)",
         "Usage is healthy and balanced.","Avg screen: ~5.2 hrs/day"),
        (c2,"🔵","Mild",
         "Early signs — elevated but manageable.","Avg screen: ~6–7 hrs/day"),
        (c3,"🟡","Moderate",
         "Clear dependency — interferes with daily life.","Avg screen: ~8 hrs/day"),
        (c4,"🔴","Severe",
         "Strong addiction — significant impairment.","Avg screen: ~9–12 hrs/day"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:white; border-radius:12px; padding:16px;
                        border:1px solid #E2E8F0; text-align:center;
                        box-shadow:0 2px 4px rgba(0,0,0,0.06)'>
                <div style='font-size:2rem'>{emoji}</div>
                <div style='font-weight:700; font-size:14px; margin:8px 0 4px'>{level}</div>
                <div style='font-size:12px; color:#555; margin-bottom:8px'>{desc}</div>
                <div style='font-size:11px; color:#888'>{avg}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Multi-class model performance")
    perf_df = pd.DataFrame({
        "Class":    ["None","Mild","Moderate","Severe","Weighted avg"],
        "Precision":["0.51","0.50","0.52","0.54","0.52"],
        "Recall":   ["0.42","0.55","0.54","0.52","0.52"],
        "F1":       ["0.46","0.52","0.53","0.53","0.52"],
        "Support":  ["129", "200", "434", "362", "1,125"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.warning("""
    **Important finding:** Even an 80% reduction in any single feature
    (e.g. cutting social media from 3.7 hrs to 0.7 hrs) is **not enough**
    to de-risk confirmed high-risk users. Effective intervention requires
    simultaneous changes across multiple behaviours — total screen time,
    social media, and weekend usage at minimum.
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:0.8rem; padding-bottom:16px'>
    Built with Streamlit · Random Forest (400 trees) · ROC-AUC 0.989 on held-out test set<br>
    <strong>For educational purposes only — does not constitute medical advice.</strong>
</div>
""", unsafe_allow_html=True)
