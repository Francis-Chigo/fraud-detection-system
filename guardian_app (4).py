# PHASE 5 - Guardian Demo UI 
# NIBSS Nigerian Fraud Detection Dataset


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
    --bg:       #0A0E1A;
    --panel:    #131829;
    --panel2:   #1B2238;
    --border:   #29314A;
    --text:     #F4F6FB;
    --blue:     #5C7FFF;
    --bglow:    rgba(92,127,255,0.3);
    --muted:    #8B93B0;
    --red:      #FF5C57;
    --redbg:    rgba(255,92,87,0.12);
    --green:    #34D399;
    --greenbg:  rgba(52,211,153,0.12);
    --amber:    #FFB454;
}

html, body, [class*="css"] { font-family:'Inter',sans-serif; font-size:16px; }
.stApp { background:var(--bg); color:var(--text); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:2rem; max-width:1260px; }

/* ── Top bar ── */
.topbar {
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
    background:linear-gradient(120deg,#131829,#1B2238);
    border:1px solid var(--border); border-radius:18px;
    padding:22px 28px; margin-bottom:28px;
}
.brand { display:flex; align-items:center; gap:14px; }
.brand-mark { width:46px; height:46px; border-radius:12px; background:var(--blue);
              display:flex; align-items:center; justify-content:center;
              font-size:22px; box-shadow:0 0 22px var(--bglow); }
.brand-name { font-family:'Space Grotesk',sans-serif; font-size:23px; font-weight:700; color:var(--text); }
.brand-sub  { font-size:13px; color:var(--muted); font-weight:500; }
.status-chip { background:var(--greenbg); color:var(--green); padding:9px 18px;
               border-radius:22px; font-size:13.5px; font-weight:700;
               display:flex; align-items:center; gap:9px;
               border:1px solid rgba(52,211,153,0.3); }
.dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.5)}70%{box-shadow:0 0 0 8px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}

/* ── Section labels ── */
.eyebrow { font-size:13px; font-weight:800; color:var(--blue);
           letter-spacing:.07em; text-transform:uppercase; margin-bottom:8px; }
.title-lg { font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700;
            margin-bottom:8px; color:var(--text); }
.desc { color:var(--muted); font-size:15px; margin-bottom:22px; line-height:1.6; }

/* ── Panels ── */
.panel { background:var(--panel); border:1px solid var(--border);
         border-radius:18px; padding:26px 28px; }

/* ── Input card grid ── */
.input-grid {
    display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:18px;
}
.input-card {
    background:var(--panel2); border:1.5px solid var(--border);
    border-radius:14px; padding:14px 18px;
    transition:border-color .15s ease;
}
.input-card:focus-within { border-color:var(--blue); }
.input-label {
    font-size:11.5px; font-weight:800; color:var(--muted);
    text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px;
}
.input-value {
    font-family:'Space Grotesk',sans-serif; font-size:17px;
    font-weight:700; color:var(--text);
}
.input-sub {
    font-size:12px; color:var(--muted); margin-top:2px;
}

/* ── Metric cards ── */
.metric-card { background:var(--panel); border:1.5px solid var(--border);
               border-radius:14px; padding:18px 20px;
               transition:transform .15s, border-color .15s; }
.metric-card:hover { transform:translateY(-2px); border-color:var(--blue); }
.metric-label { font-size:12px; color:var(--muted); font-weight:700;
                text-transform:uppercase; letter-spacing:.04em; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:28px;
                font-weight:700; margin-top:6px; color:var(--text); }

/* ── Verdict ── */
.verdict { border-radius:16px; padding:20px 24px; display:flex;
           align-items:center; gap:16px; margin-bottom:20px; }
.verdict-fraud { background:var(--redbg); border:2px solid rgba(255,92,87,.3); }
.verdict-clear { background:var(--greenbg); border:2px solid rgba(52,211,153,.3); }
.verdict-icon  { font-size:28px; }
.verdict-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:18px; }
.verdict-fraud .verdict-title { color:var(--red); }
.verdict-clear .verdict-title { color:var(--green); }

/* ── Feature rows ── */
.feat-row { display:flex; align-items:center; padding:12px 0;
            border-bottom:1px solid var(--border); }
.feat-row:last-child { border-bottom:none; }
.feat-name { font-weight:700; font-size:14px; min-width:150px;
             font-family:'Space Grotesk',sans-serif; color:var(--text); }
.feat-bar-bg { flex:1; height:7px; background:var(--panel2);
               border-radius:4px; margin:0 16px; overflow:hidden; }
.feat-bar-fill { height:100%; border-radius:4px;
                 background:linear-gradient(90deg,var(--blue),#8AA3FF); }
.feat-val { font-weight:800; font-size:14px; min-width:55px;
            text-align:right; font-family:'Space Grotesk',sans-serif; }

/* ── Live signal chip ── */
.signal-chip {
    display:flex; align-items:center; gap:10px; padding:13px 18px;
    border-radius:12px; margin-bottom:18px; font-size:14.5px;
    font-weight:700; border:1px solid;
}

/* ── Button ── */
.stButton button {
    background:var(--blue); color:#0A0E1A; border:none;
    border-radius:13px; font-weight:800; padding:15px 0; font-size:17px;
    font-family:'Space Grotesk',sans-serif;
    box-shadow:0 0 28px var(--bglow); transition:all .15s ease;
    width:100%;
}
.stButton button:hover { filter:brightness(1.1); transform:translateY(-1px); }

/* ── Streamlit widget cleanup ── */
div[data-testid="stNumberInput"] > label,
div[data-testid="stSelectbox"] > label { display:none !important; }

div[data-testid="stNumberInput"] input {
    background:transparent !important; border:none !important;
    color:var(--text) !important; font-family:'Space Grotesk',sans-serif !important;
    font-size:18px !important; font-weight:700 !important;
    padding:0 !important; box-shadow:none !important;
}
div[data-testid="stNumberInput"] { background:transparent !important; }
div[data-testid="stNumberInput"] > div { background:transparent !important;
    border:none !important; padding:0 !important; }

div[data-testid="stSelectbox"] > div > div {
    background:transparent !important; border:none !important;
    color:var(--text) !important; font-family:'Space Grotesk',sans-serif !important;
    font-size:17px !important; font-weight:700 !important;
    padding:0 !important; box-shadow:none !important;
}
div[data-testid="stSelectbox"] svg { fill:var(--muted) !important; }

.stTabs [data-baseweb="tab-list"] { gap:30px; border-bottom:2px solid var(--border); }
.stTabs [data-baseweb="tab"] { color:var(--muted); font-weight:700; font-size:16px;
    padding-bottom:15px; font-family:'Space Grotesk',sans-serif; }
.stTabs [aria-selected="true"] { color:var(--blue) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--blue) !important; }

.stRadio > div { gap:10px; }
.stRadio label { background:var(--panel2) !important; border:2px solid var(--border) !important;
    border-radius:12px; padding:13px 16px !important; transition:all .15s; }
.stRadio label:hover { border-color:var(--blue) !important; }
.stRadio label p { font-size:15px !important; font-weight:600 !important; color:var(--text) !important; }

.empty-state { text-align:center; padding:80px 24px; color:var(--muted); font-size:16px; }
.empty-icon  { font-size:46px; margin-bottom:16px; display:inline-block;
               animation:float 3s ease-in-out infinite; }
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & DATA ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/nibss_fraud_dataset.csv", low_memory=False)

model = load_model()
df    = load_data()

CHANNELS   = sorted(df["channel"].dropna().unique().tolist())
MERCHANTS  = sorted(df["merchant_category"].dropna().unique().tolist())
BANKS      = sorted(df["bank"].dropna().unique().tolist())
LOCATIONS  = sorted(df["location"].dropna().unique().tolist())
AGE_GROUPS = sorted(df["age_group"].dropna().unique().tolist())

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
  <div class="status-chip">
    <span class="dot"></span> Model Online · XGBoost · 1M Nigerian Transactions
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Screen a transaction", "Portfolio view", "How it works"])

# ════════════════════════════════════════════════════════
# TAB 1 — SCREEN
# ════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown('<div class="eyebrow">Step 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="title-lg">Load a transaction</div>', unsafe_allow_html=True)
        st.markdown('<div class="desc">Pick a real record from the NIBSS dataset or set values manually.</div>', unsafe_allow_html=True)

        test_type = st.radio(" ",
            ["Sample — normal", "Sample — known fraud", "Enter manually"],
            label_visibility="collapsed")

        if test_type == "Sample — normal":
            sample = df[df["is_fraud"] == 0].sample(1).iloc[0]
        elif test_type == "Sample — known fraud":
            sample = df[df["is_fraud"] == 1].sample(1).iloc[0]
        else:
            sample = df.iloc[0]

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Step 2 — Transaction details</div>', unsafe_allow_html=True)

        # ── Row 1: Amount + Channel ──────────────────────
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown('<div class="input-card"><div class="input-label">Amount (₦)</div>', unsafe_allow_html=True)
            amount = st.number_input("Amount", value=float(sample["amount"]),
                                     format="%.2f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        with r1c2:
            st.markdown('<div class="input-card"><div class="input-label">Channel</div>', unsafe_allow_html=True)
            channel_idx = CHANNELS.index(sample["channel"]) if sample["channel"] in CHANNELS else 0
            channel = st.selectbox("Channel", CHANNELS, index=channel_idx,
                                   label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 2: Bank + Merchant ───────────────────────
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown('<div class="input-card"><div class="input-label">Bank</div>', unsafe_allow_html=True)
            bank_idx = BANKS.index(sample["bank"]) if sample["bank"] in BANKS else 0
            bank = st.selectbox("Bank", BANKS, index=bank_idx,
                                label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        with r2c2:
            st.markdown('<div class="input-card"><div class="input-label">Merchant Category</div>', unsafe_allow_html=True)
            merch_idx = MERCHANTS.index(sample["merchant_category"]) if sample["merchant_category"] in MERCHANTS else 0
            merchant = st.selectbox("Merchant", MERCHANTS, index=merch_idx,
                                    label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 3: Location + Age Group ──────────────────
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.markdown('<div class="input-card"><div class="input-label">Location</div>', unsafe_allow_html=True)
            loc_idx = LOCATIONS.index(sample["location"]) if sample["location"] in LOCATIONS else 0
            location = st.selectbox("Location", LOCATIONS, index=loc_idx,
                                    label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        with r3c2:
            st.markdown('<div class="input-card"><div class="input-label">Age Group</div>', unsafe_allow_html=True)
            age_idx = AGE_GROUPS.index(sample["age_group"]) if sample["age_group"] in AGE_GROUPS else 0
            age_grp = st.selectbox("Age Group", AGE_GROUPS, index=age_idx,
                                   label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 4: Hour + Weekend ────────────────────────
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            st.markdown('<div class="input-card"><div class="input-label">Hour of Day (0–23)</div>', unsafe_allow_html=True)
            hour = st.number_input("Hour", value=int(sample["hour"]),
                                   min_value=0, max_value=23,
                                   label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        with r4c2:
            st.markdown('<div class="input-card"><div class="input-label">Is Weekend?</div>', unsafe_allow_html=True)
            is_wknd = st.selectbox("Weekend", ["No (0)", "Yes (1)"],
                                   index=int(sample["is_weekend"]),
                                   label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Live signal chip ─────────────────────────────
        risk_hint = float(sample.get("composite_risk", 0.3))
        if risk_hint > 0.6:
            lbl = "⚠️  Signals look risky"
            lc, lb = "#FF5C57", "rgba(255,92,87,0.1)"
            bc = "rgba(255,92,87,0.35)"
        elif risk_hint > 0.3:
            lbl = "〰️  Signals look borderline"
            lc, lb = "#FFB454", "rgba(255,180,84,0.1)"
            bc = "rgba(255,180,84,0.35)"
        else:
            lbl = "✓  Signals look normal"
            lc, lb = "#34D399", "rgba(52,211,153,0.1)"
            bc = "rgba(52,211,153,0.35)"

        st.markdown(f"""
        <div class="signal-chip" style="background:{lb}; color:{lc}; border-color:{bc};">
            {lbl} &nbsp;—&nbsp; press Screen for the full risk score
        </div>""", unsafe_allow_html=True)

        # ── Build feature vector ─────────────────────────
        drop_cols = ["transaction_id","customer_id","timestamp","fraud_technique","is_fraud"]
        feature_cols = [c for c in pd.read_csv("data/processed/X_train.csv", nrows=1).columns]

        input_row = sample.copy()
        input_row["amount"]            = amount
        input_row["channel"]           = channel
        input_row["bank"]              = bank
        input_row["merchant_category"] = merchant
        input_row["location"]          = location
        input_row["age_group"]         = age_grp
        input_row["hour"]              = hour
        input_row["is_weekend"]        = 1 if "Yes" in is_wknd else 0

        input_df = pd.DataFrame([input_row])
        input_df = input_df.drop(columns=[c for c in drop_cols if c in input_df.columns], errors="ignore")

        le = LabelEncoder()
        cat_cols = ["channel","merchant_category","bank","location","age_group","day_of_week"]
        for col in cat_cols:
            if col in input_df.columns:
                input_df[col] = le.fit_transform(input_df[col].astype(str))

        num_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        input_df[num_cols] = scaler.fit_transform(input_df[num_cols])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        run = st.button("⚡  Screen this transaction", use_container_width=True)

    # ── RESULT COLUMN ────────────────────────────────────
    with col_result:
        st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="title-lg">Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown('<div class="desc">Live fraud score from XGBoost trained on 1 million Nigerian transactions.</div>', unsafe_allow_html=True)

        if run:
            prob = float(model.predict_proba(input_df)[0][1])
            pred = int(model.predict(input_df)[0])
            pct  = prob * 100

            if pred == 1:
                st.markdown("""
                <div class="verdict verdict-fraud">
                  <div class="verdict-icon">⛔</div>
                  <div class="verdict-title">Flagged as fraud — block and review</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict verdict-clear">
                  <div class="verdict-icon">✓</div>
                  <div class="verdict-title">Looks legitimate — no action needed</div>
                </div>""", unsafe_allow_html=True)

            needle  = -90 + (pct / 100) * 180
            gcol    = "#FF5C57" if pct >= 60 else ("#FFB454" if pct >= 30 else "#34D399")
            arc_len = (pct / 100) * 314

            gauge = f"""
<div style="background:#131829;border:1.5px solid #29314A;border-radius:18px;
            padding:28px;text-align:center;font-family:'Space Grotesk',sans-serif;">
  <svg width="270" height="165" viewBox="0 0 260 160">
    <path d="M30 140 A100 100 0 0 1 230 140" fill="none" stroke="#1B2238"
          stroke-width="18" stroke-linecap="round"/>
    <path d="M30 140 A100 100 0 0 1 230 140" fill="none" stroke="{gcol}"
          stroke-width="18" stroke-linecap="round"
          style="filter:drop-shadow(0 0 8px {gcol}88)">
      <animate attributeName="stroke-dasharray"
        from="0 314" to="{arc_len:.1f} 314"
        dur="0.85s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>
    </path>
    <g>
      <animateTransform attributeName="transform" type="rotate"
        from="-90 130 140" to="{needle:.1f} 130 140"
        dur="0.85s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>
      <line x1="130" y1="140" x2="130" y2="52"
            stroke="#F4F6FB" stroke-width="4" stroke-linecap="round"/>
    </g>
    <circle cx="130" cy="140" r="9" fill="#F4F6FB"/>
    <text x="130" y="116" text-anchor="middle"
          font-size="36" font-weight="700" fill="#F4F6FB">{pct:.1f}%</text>
    <text x="130" y="152" text-anchor="middle"
          font-size="12" fill="#8B93B0" font-weight="700"
          letter-spacing="0.5">FRAUD PROBABILITY</text>
  </svg>
</div>"""
            components.html(gauge, height=222)

            m1, m2 = st.columns(2)
            tier = "High" if pct >= 60 else ("Medium" if pct >= 30 else "Low")
            with m1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Confidence</div>
                  <div class="metric-value">{max(prob,1-prob)*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                tc = "#FF5C57" if tier=="High" else ("#FFB454" if tier=="Medium" else "#34D399")
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">Risk Tier</div>
                  <div class="metric-value" style="color:{tc};">{tier}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Key signals</div>', unsafe_allow_html=True)

            feats = [
                ("Amount (₦)",      amount,                                       amount / max(df["amount"].max(), 1) * 100),
                ("Composite Risk",  float(sample.get("composite_risk", 0)),       float(sample.get("composite_risk", 0)) * 100),
                ("Velocity Score",  float(sample.get("velocity_score", 0)),       float(sample.get("velocity_score", 0)) * 100),
                ("Merchant Risk",   float(sample.get("merchant_risk_score", 0)),  float(sample.get("merchant_risk_score", 0)) * 100),
            ]
            rows = "".join(f"""
            <div class="feat-row">
              <span class="feat-name">{n}</span>
              <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:{min(w,100):.1f}%"></div>
              </div>
              <span class="feat-val">{v:.2f}</span>
            </div>""" for n, v, w in feats)
            st.markdown(f'<div class="panel" style="padding:18px 24px;">{rows}</div>',
                        unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="panel empty-state">
              <div class="empty-icon">🛡️</div><br>
              Load a transaction and press<br>
              <b style="color:#F4F6FB;">Screen this transaction</b><br>
              to get an instant fraud risk score.
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-lg">Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc">1,000,000 NIBSS-calibrated Nigerian banking transactions (2023–2026).</div>', unsafe_allow_html=True)

    total  = len(df)
    fraud  = int(df["is_fraud"].sum())
    normal = total - fraud

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip([c1,c2,c3,c4],
        ["Total Transactions","Normal","Fraud Cases","Fraud Rate"],
        [f"{total:,}", f"{normal:,}", f"{fraud:,}", f"{fraud/total*100:.2f}%"]):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-bottom:14px;">Sample records</div>', unsafe_allow_html=True)
    preview = df[["transaction_id","amount","channel","merchant_category",
                  "location","bank","age_group","is_fraud"]].head(10).copy()
    preview["is_fraud"] = preview["is_fraud"].map({0:"Normal", 1:"Fraud"})
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="eyebrow">Final Year Project — ESUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-lg">How Guardian Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel" style="line-height:1.9; color:#C7CCDE; font-size:15.5px;">
      <b style="color:#F4F6FB;">Student:</b> Onwugbolu Francis Chigozie &nbsp;·&nbsp; Matric No: 2022030200406<br>
      <b style="color:#F4F6FB;">Institution:</b> Enugu State University of Science and Technology (ESUT)<br>
      <b style="color:#F4F6FB;">Supervisor:</b> Sir David Mba<br><br>
      <b style="color:#F4F6FB;">Dataset:</b> NIBSS-calibrated Nigerian banking dataset — 1,000,000 real transactions
      from Nigerian payment infrastructure (2023–2026), covering channels such as Mobile Banking,
      POS, Internet Banking, ATM, and USSD across multiple banks and states.<br><br>
      <b style="color:#F4F6FB;">Model:</b> XGBoost (Extreme Gradient Boosting), tuned via grid search cross-validation.
      Trained on SMOTE-balanced data to handle the class imbalance inherent in fraud data.<br><br>
      <b style="color:#F4F6FB;">Key features:</b> Transaction amount (₦), channel, merchant category, bank,
      location, age group, hour, velocity score, merchant risk score, and composite risk index.<br><br>
      <b style="color:#F4F6FB;">Why Nigeria?</b> Over ₦9.5 billion was reported lost to fraud in Nigeria
      in 2023 alone (NIBSS Fraud Report). This system directly addresses that problem using
      local, contextually relevant transaction data — making it one of the few final year projects
      in this institution to train on Nigerian-specific financial data.
    </div>
    """, unsafe_allow_html=True)
