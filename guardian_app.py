# ============================================================
# PHASE 5 - Guardian Demo UI
# NIBSS Nigerian Fraud Detection Dataset
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title="Guardian — Nigerian Payment Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg: #0A0E1A; --panel: #131829; --panel-2: #1B2238;
    --border: #29314A; --text: #F4F6FB; --blue: #5C7FFF;
    --blue-glow: rgba(92,127,255,0.35); --muted: #8B93B0;
    --red: #FF5C57; --red-bg: rgba(255,92,87,0.12);
    --green: #34D399; --green-bg: rgba(52,211,153,0.12);
    --amber: #FFB454;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 17px; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1240px; }
.topbar {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;
    background: linear-gradient(120deg, #131829 0%, #1B2238 100%);
    border: 1px solid var(--border); border-radius: 20px; padding: 24px 30px; margin-bottom: 30px;
}
.brand { display: flex; align-items: center; gap: 16px; }
.brand-mark { width: 50px; height: 50px; border-radius: 14px; background: var(--blue);
              display: flex; align-items: center; justify-content: center;
              font-size: 24px; box-shadow: 0 0 24px var(--blue-glow); }
.brand-name { font-family:'Space Grotesk',sans-serif; font-size: 25px; font-weight: 700; color: var(--text); }
.brand-sub  { font-size: 14px; color: var(--muted); font-weight: 500; }
.status-chip { background: var(--green-bg); color: var(--green); padding: 10px 20px;
               border-radius: 24px; font-size: 14px; font-weight: 700;
               display: flex; align-items: center; gap: 10px; border: 1px solid rgba(52,211,153,0.3); }
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--green);
              animation: pulse 2s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(52,211,153,0.5)} 70%{box-shadow:0 0 0 8px rgba(52,211,153,0)} 100%{box-shadow:0 0 0 0 rgba(52,211,153,0)} }
.eyebrow { font-size: 14px; font-weight: 800; color: var(--blue); letter-spacing: 0.07em;
           text-transform: uppercase; margin-bottom: 9px; }
.title-lg { font-family:'Space Grotesk',sans-serif; font-size: 30px; font-weight: 700;
            margin-bottom: 9px; color: var(--text); }
.desc { color: var(--muted); font-size: 16px; margin-bottom: 26px; line-height: 1.6; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 28px 30px; }
.metric-card { background: var(--panel); border: 1.5px solid var(--border); border-radius: 16px;
               padding: 20px 22px; transition: transform 0.15s ease, border-color 0.15s ease; }
.metric-card:hover { transform: translateY(-2px); border-color: var(--blue); }
.metric-label { font-size: 13px; color: var(--muted); font-weight: 700; text-transform: uppercase; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size: 30px; font-weight: 700;
                margin-top: 6px; color: var(--text); }
.verdict { border-radius: 16px; padding: 20px 26px; display: flex; align-items: center; gap: 16px; margin-bottom: 22px; }
.verdict-fraud { background: var(--red-bg); border: 2px solid rgba(255,92,87,0.3); }
.verdict-clear { background: var(--green-bg); border: 2px solid rgba(52,211,153,0.3); }
.verdict-icon { font-size: 28px; }
.verdict-title { font-family:'Space Grotesk',sans-serif; font-weight: 700; font-size: 19px; }
.verdict-fraud .verdict-title { color: var(--red); }
.verdict-clear .verdict-title { color: var(--green); }
.stButton button { background: var(--blue); color: #0A0E1A; border: none; border-radius: 14px;
                   font-weight: 800; padding: 16px 0; font-size: 17px;
                   font-family:'Space Grotesk',sans-serif; box-shadow: 0 0 30px var(--blue-glow);
                   transition: all 0.15s ease; }
.stButton button:hover { filter: brightness(1.1); transform: translateY(-1px); }
div[data-testid="stNumberInput"] input { background: var(--panel-2) !important;
    border: 2px solid var(--border) !important; color: var(--text) !important;
    border-radius: 10px; font-size: 17px !important; font-weight: 700; }
div[data-testid="stNumberInput"] label p { font-size: 15px !important; font-weight: 700 !important; color: var(--text) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 32px; border-bottom: 2px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--muted); font-weight: 700; font-size: 16px; padding-bottom: 16px; font-family:'Space Grotesk',sans-serif; }
.stTabs [aria-selected="true"] { color: var(--blue) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--blue) !important; }
.stRadio label { background: var(--panel-2) !important; border: 2px solid var(--border) !important;
                 border-radius: 12px; padding: 12px 16px !important; margin-bottom: 6px; transition: all 0.15s ease; }
.stRadio label:hover { border-color: var(--blue) !important; }
.stRadio label p { font-size: 15px !important; font-weight: 600 !important; color: var(--text) !important; }
# .stSelectbox div { background: var(--panel-2) !important; border: 2px solid var(--border) !important;
#                    color: var(--text) !important; border-radius: 10px; font-size: 16px; }
/* Clean Selectbox Styling */
div[data-testid="stSelectbox"] {
    margin-bottom: 12px;
}

div[data-testid="stSelectbox"] label {
    color: var(--text) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--panel-2) !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    min-height: 50px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: var(--text) !important;
    font-size: 16px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"]:hover {
    border-color: var(--blue) !important;
}
.feature-row { display: flex; justify-content: space-between; align-items: center;
               padding: 14px 0; border-bottom: 1px solid var(--border); font-size: 15px; }
.feature-row:last-child { border-bottom: none; }
.feature-name { color: var(--text); font-weight: 700; min-width: 160px; font-family:'Space Grotesk',sans-serif; }
.feature-bar-bg { flex: 1; height: 8px; background: var(--panel-2); border-radius: 4px;
                   margin: 0 18px; overflow: hidden; }
.feature-bar-fill { height: 100%; background: linear-gradient(90deg, var(--blue), #8AA3FF);
                     border-radius: 4px; }
.feature-val { font-weight: 800; color: var(--text); min-width: 60px; text-align: right; font-family:'Space Grotesk',sans-serif; }
.empty-state { text-align: center; padding: 80px 24px; color: var(--muted); font-size: 16px; }
.empty-icon { font-size: 46px; margin-bottom: 16px; display: inline-block; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & DATA ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data/nibss_fraud_dataset.csv", low_memory=False)
    return df

model = load_model()
df    = load_data()

CHANNELS   = sorted(df["channel"].dropna().unique().tolist())
MERCHANTS  = sorted(df["merchant_category"].dropna().unique().tolist())
BANKS      = sorted(df["bank"].dropna().unique().tolist())
LOCATIONS  = sorted(df["location"].dropna().unique().tolist())
AGE_GROUPS = sorted(df["age_group"].dropna().unique().tolist())
DAYS       = sorted(df["day_of_week"].dropna().unique().tolist())

# ── TOP BAR ───────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="brand">
        <div class="brand-mark">🛡️</div>
        <div>
            <div class="brand-name">Guardian</div>
            <div class="brand-sub">NIBSS-Calibrated Nigerian Payment Fraud Detection</div>
        </div>
    </div>
    <div class="status-chip"><span class="status-dot"></span> Model Online · XGBoost · 1M Nigerian Transactions</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Screen a transaction", "Portfolio view", "How it works"])

# ── TAB 1: PREDICT ────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([0.95, 1.25], gap="large")

    with col_input:
        st.markdown('<div class="eyebrow">Transaction</div>', unsafe_allow_html=True)
        st.markdown('<div class="title-lg">Screen a transaction</div>', unsafe_allow_html=True)
        st.markdown('<div class="desc">Enter transaction details or load a sample from the NIBSS Nigerian dataset.</div>', unsafe_allow_html=True)

        test_type = st.radio(" ",
            ["Sample — normal transaction", "Sample — known fraud", "Enter manually"],
            label_visibility="collapsed")

        if test_type == "Sample — normal transaction":
            sample = df[df["is_fraud"] == 0].sample(1).iloc[0]
            st.success("✅ Loaded a normal Nigerian transaction")
        elif test_type == "Sample — known fraud":
            sample = df[df["is_fraud"] == 1].sample(1).iloc[0]
            st.warning("⚠️ Loaded a known fraud transaction")
        else:
            sample = df.iloc[0]

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Transaction Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            amount   = st.number_input("Amount (₦)", value=float(sample["amount"]), format="%.2f")
            channel  = st.selectbox("Channel", CHANNELS, index=CHANNELS.index(sample["channel"]) if sample["channel"] in CHANNELS else 0)
            bank     = st.selectbox("Bank", BANKS, index=BANKS.index(sample["bank"]) if sample["bank"] in BANKS else 0)
            location = st.selectbox("Location", LOCATIONS, index=LOCATIONS.index(sample["location"]) if sample["location"] in LOCATIONS else 0)
        with c2:
            merchant = st.selectbox("Merchant Category", MERCHANTS, index=MERCHANTS.index(sample["merchant_category"]) if sample["merchant_category"] in MERCHANTS else 0)
            age_grp  = st.selectbox("Age Group", AGE_GROUPS, index=AGE_GROUPS.index(sample["age_group"]) if sample["age_group"] in AGE_GROUPS else 0)
            hour     = st.number_input("Hour (0-23)", value=int(sample["hour"]), min_value=0, max_value=23)
            is_wknd  = st.selectbox("Weekend?", [0, 1], index=int(sample["is_weekend"]))

        # Live signal preview
        risk_hint = float(sample.get("composite_risk", 0.5))
        if risk_hint > 0.6:
            lbl, lc, lb = "Transaction signals look risky", "#FF5C57", "rgba(255,92,87,0.12)"
        elif risk_hint > 0.3:
            lbl, lc, lb = "Transaction signals look borderline", "#FFB454", "rgba(255,180,84,0.12)"
        else:
            lbl, lc, lb = "Transaction signals look normal", "#34D399", "rgba(52,211,153,0.12)"
        st.markdown(f"""
        <div style="background:{lb}; border-radius:12px; padding:13px 18px; margin-top:8px;
                    display:flex; align-items:center; gap:10px; font-size:15px; font-weight:700;
                    color:{lc}; border:1px solid {lc}33;">
            <span style="width:9px;height:9px;border-radius:50%;background:{lc};display:inline-block;"></span>
            {lbl} — press Screen for the full risk score
        </div>""", unsafe_allow_html=True)

        # Build feature vector
        le = LabelEncoder()
        scaler = StandardScaler()
        feature_cols = [c for c in pd.read_csv("data/processed/X_train.csv", nrows=1).columns]

        input_row = sample.copy()
        input_row["amount"]           = amount
        input_row["channel"]          = channel
        input_row["bank"]             = bank
        input_row["location"]         = location
        input_row["merchant_category"]= merchant
        input_row["age_group"]        = age_grp
        input_row["hour"]             = hour
        input_row["is_weekend"]       = is_wknd

        drop_cols = ["transaction_id", "customer_id", "timestamp", "fraud_technique", "is_fraud"]
        input_df  = pd.DataFrame([input_row]).drop(columns=[c for c in drop_cols if c in input_row.index], errors="ignore")

        cat_cols = ["channel", "merchant_category", "bank", "location", "age_group", "day_of_week"]
        for col in cat_cols:
            if col in input_df.columns:
                input_df[col] = le.fit_transform(input_df[col].astype(str))

        num_cols = [c for c in input_df.columns if input_df[c].dtype in [np.float64, np.int64]]
        input_df[num_cols] = scaler.fit_transform(input_df[num_cols])

        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        run = st.button("⚡ Screen this transaction", use_container_width=True)

    with col_result:
        st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="title-lg">Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown('<div class="desc">Live fraud score from the tuned XGBoost model trained on 1 million Nigerian transactions.</div>', unsafe_allow_html=True)

        if run:
            prob = float(model.predict_proba(input_df)[0][1])
            pred = int(model.predict(input_df)[0])
            pct  = prob * 100

            if pred == 1:
                st.markdown("""<div class="verdict verdict-fraud">
                    <div class="verdict-icon">⛔</div>
                    <div class="verdict-title">Flagged as fraud — block and review</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="verdict verdict-clear">
                    <div class="verdict-icon">✓</div>
                    <div class="verdict-title">Looks legitimate — no action needed</div>
                </div>""", unsafe_allow_html=True)

            needle_angle = -90 + (pct / 100) * 180
            gauge_color  = "#FF5C57" if pct >= 60 else ("#FFB454" if pct >= 30 else "#34D399")

            gauge_html = f"""
            <div style="background:#131829; border:1.5px solid #29314A; border-radius:20px;
                        padding:28px; text-align:center; font-family:'Space Grotesk',sans-serif;">
              <svg width="270" height="165" viewBox="0 0 260 160">
                <path d="M 30 140 A 100 100 0 0 1 230 140" fill="none" stroke="#1B2238" stroke-width="18" stroke-linecap="round"/>
                <path d="M 30 140 A 100 100 0 0 1 230 140" fill="none" stroke="{gauge_color}" stroke-width="18"
                      stroke-linecap="round" style="filter:drop-shadow(0 0 8px {gauge_color}88)">
                  <animate attributeName="stroke-dasharray" from="0 314" to="{(pct/100)*314} 314"
                           dur="0.9s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>
                </path>
                <g>
                  <animateTransform attributeName="transform" type="rotate"
                    from="-90 130 140" to="{needle_angle} 130 140"
                    dur="0.9s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>
                  <line x1="130" y1="140" x2="130" y2="54" stroke="#F4F6FB" stroke-width="4" stroke-linecap="round"/>
                </g>
                <circle cx="130" cy="140" r="9" fill="#F4F6FB"/>
                <text x="130" y="118" text-anchor="middle" font-size="36" font-weight="700" fill="#F4F6FB">{pct:.1f}%</text>
                <text x="130" y="153" text-anchor="middle" font-size="12" fill="#8B93B0" font-weight="700" letter-spacing="0.5">FRAUD PROBABILITY</text>
              </svg>
            </div>"""
            components.html(gauge_html, height=220)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Confidence</div>
                <div class="metric-value">{max(prob,1-prob)*100:.1f}%</div></div>""", unsafe_allow_html=True)
            with m2:
                tier = "High" if pct >= 60 else ("Medium" if pct >= 30 else "Low")
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Risk Tier</div>
                <div class="metric-value">{tier}</div></div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Key signals</div>', unsafe_allow_html=True)
            top_feats = [
                ("Amount (₦)",  amount),
                ("Composite Risk", float(sample.get("composite_risk", 0))),
                ("Velocity Score", float(sample.get("velocity_score", 0))),
                ("Merchant Risk",  float(sample.get("merchant_risk_score", 0))),
            ]
            rows = ""
            for name, val in top_feats:
                width = min(abs(val) / max(abs(amount), 1) * 100 if name == "Amount (₦)" else abs(val) * 100, 100)
                rows += f"""<div class="feature-row"><span class="feature-name">{name}</span>
                <div class="feature-bar-bg"><div class="feature-bar-fill" style="width:{width}%"></div></div>
                <span class="feature-val">{val:.2f}</span></div>"""
            st.markdown(f'<div class="panel" style="padding:20px 26px;">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div class="panel empty-state">
                <div class="empty-icon">🛡️</div><br>
                Load a transaction and press <b>Screen this transaction</b><br>
                to get an instant fraud risk score.
            </div>""", unsafe_allow_html=True)

# ── TAB 2: PORTFOLIO ──────────────────────────────────────────
with tab2:
    st.markdown('<div class="eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-lg">Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc">1,000,000 NIBSS-calibrated Nigerian banking transactions (2023–2026).</div>', unsafe_allow_html=True)

    total  = len(df)
    fraud  = int(df["is_fraud"].sum())
    normal = total - fraud

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4],
        ["Total Transactions", "Normal", "Fraud Cases", "Fraud Rate"],
        [f"{total:,}", f"{normal:,}", f"{fraud:,}", f"{fraud/total*100:.2f}%"]):
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Sample Records</div>', unsafe_allow_html=True)
    preview = df[["transaction_id", "amount", "channel", "merchant_category",
                  "location", "bank", "age_group", "is_fraud"]].head(10).copy()
    preview["is_fraud"] = preview["is_fraud"].map({0: "Normal", 1: "Fraud"})
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: ABOUT ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="eyebrow">Final Year Project — ESUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-lg">How Guardian Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel" style="line-height:1.9; color:#C7CCDE; font-size:16px;">
    <b style="color:#F4F6FB;">Student:</b> Onwugbolu Francis Chigozie · Matric No: 2022030200406<br>
    <b style="color:#F4F6FB;">Institution:</b> Enugu State University of Science and Technology (ESUT)<br>
    <b style="color:#F4F6FB;">Supervisor:</b> Sir David Mba<br><br>
    <b style="color:#F4F6FB;">Dataset:</b> NIBSS-calibrated Nigerian banking dataset — 1,000,000 real transactions
    from Nigerian payment infrastructure (2023–2026), covering channels such as mobile banking,
    POS, internet banking, and USSD across multiple banks and locations.<br><br>
    <b style="color:#F4F6FB;">Model:</b> XGBoost (Extreme Gradient Boosting), tuned via grid search.
    Trained on SMOTE-balanced data to handle the class imbalance inherent in fraud data.<br><br>
    <b style="color:#F4F6FB;">Key features:</b> Transaction amount (₦), channel, merchant category, bank,
    location, age group, time of day, velocity score, merchant risk score, and composite risk index.<br><br>
    <b style="color:#F4F6FB;">Why Nigeria?</b> Over ₦12.2 billion was lost to fraudulent transactions in Nigeria
    in 2023 alone. This system directly addresses that problem using local, relevant transaction data.
    </div>
    """, unsafe_allow_html=True)
