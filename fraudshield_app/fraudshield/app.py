# ============================================================
# 🛡️  FRAUDSHIELD — Credit Card Fraud Detection Platform

#  Dataset : https://www.kaggle.com/datasets/kartik2112/fraud-detection
#  Dataset: https://drive.google.com/uc?export=download&id=1fCH4sZApDjE6s9KVtH14R82ReCk6JaS2
#  Models  : Support Vector Machine (LinearSVC + Platt Calibration)
#             Random Forest (200 trees, class_weight='balanced')
#             XGBoost — Bonus model
#  Imbalance: SMOTE oversampling on training split only
#  Evaluation: Holdout (80/20 stratified split)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings, os, time
warnings.filterwarnings("ignore")

# ── ML Imports ──────────────────────────────────────────────
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.svm               import LinearSVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix, roc_auc_score,
                                       roc_curve)
from imblearn.over_sampling    import SMOTE
import xgboost as xgb


# ============================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="FraudShield | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM — Professional Finance Theme
# Navy  ·  White  ·  Gold
# ============================================================
NAVY   = "#0A2342"
NAVY2  = "#1E3A5F"
NAVY3  = "#2C4F7C"
GOLD   = "#C9A84C"
GOLD2  = "#F0C870"
WHITE  = "#FFFFFF"
LIGHT  = "#F2F5F9"
RED    = "#D64045"
GREEN  = "#2DC653"
GRAY   = "#8D99AE"
LGRAY  = "#E8ECF2"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
.stApp {{ background-color:{LIGHT}; font-family:'DM Sans',sans-serif; }}
.main .block-container {{ padding-top:24px; padding-bottom:48px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,{NAVY} 0%,{NAVY2} 60%,{NAVY3} 100%);
}}
[data-testid="stSidebar"] * {{ color:{WHITE} !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    background:rgba(255,255,255,0.07); border-radius:8px;
    padding:10px 14px !important; transition:all .2s; width:100%; cursor:pointer;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background:rgba(201,168,76,.25) !important;
}}

/* ── Hero ── */
.hero-banner {{
    background:linear-gradient(135deg,{NAVY} 0%,{NAVY2} 55%,{GOLD} 140%);
    padding:42px 38px; border-radius:20px; margin-bottom:32px;
    position:relative; overflow:hidden;
}}
.hero-banner::before {{
    content:''; position:absolute; top:-70px; right:-70px;
    width:240px; height:240px; background:rgba(201,168,76,.15); border-radius:50%;
}}
.hero-banner::after {{
    content:''; position:absolute; bottom:-50px; right:100px;
    width:160px; height:160px; background:rgba(255,255,255,.05); border-radius:50%;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background:{WHITE}; border-radius:14px; padding:22px 20px;
    border-left:5px solid {GOLD};
    box-shadow:0 2px 16px rgba(10,35,66,.07);
    transition:transform .2s;
}}
.kpi-card:hover {{ transform:translateY(-2px); }}
.kpi-value {{
    font-family:'Playfair Display',serif; font-size:2rem; font-weight:700;
    color:{NAVY}; line-height:1.1; margin:0;
}}
.kpi-label {{
    color:{GRAY}; font-size:.78rem; margin:6px 0 0;
    text-transform:uppercase; letter-spacing:.08em; font-weight:500;
}}

/* ── Model Card ── */
.model-card {{
    background:{WHITE}; border-radius:12px; padding:20px;
    box-shadow:0 2px 12px rgba(10,35,66,.06);
    border-top:4px solid {GOLD}; height:100%;
}}

/* ── Info Boxes ── */
.info-box {{
    background:#EEF4FF; border-left:4px solid {NAVY};
    padding:14px 18px; border-radius:0 8px 8px 0;
    margin:14px 0; font-size:.88rem; color:{NAVY}; line-height:1.6;
}}
.warn-box {{
    background:#FFF8E7; border-left:4px solid {GOLD};
    padding:14px 18px; border-radius:0 8px 8px 0;
    margin:14px 0; font-size:.88rem; color:#5C4200; line-height:1.6;
}}

/* ── Page Typography ── */
.page-title {{
    font-family:'Playfair Display',serif; font-size:2rem;
    font-weight:800; color:{NAVY}; margin:0 0 4px;
}}
.page-subtitle {{ color:{GRAY}; font-size:.95rem; margin:0 0 24px; }}

/* ── Buttons ── */
.stButton > button {{
    background:linear-gradient(90deg,{NAVY},{NAVY2}) !important;
    color:{WHITE} !important; border:none !important;
    border-radius:8px !important; padding:12px 32px !important;
    font-weight:600 !important; letter-spacing:.04em !important;
    box-shadow:0 4px 14px rgba(10,35,66,.3) !important; transition:all .2s !important;
}}
.stButton > button:hover {{
    background:linear-gradient(90deg,{GOLD},#A07830) !important;
    box-shadow:0 4px 18px rgba(201,168,76,.4) !important;
    transform:translateY(-1px) !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{WHITE}; border-radius:10px; padding:4px; gap:2px;
    box-shadow:0 1px 6px rgba(10,35,66,.06);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:7px !important; padding:9px 20px !important;
    font-weight:500 !important; color:{GRAY} !important;
}}
.stTabs [aria-selected="true"] {{
    background:{NAVY} !important; color:{WHITE} !important;
}}

/* ── Misc ── */
h1,h2,h3 {{ font-family:'Playfair Display',serif; color:{NAVY}; }}
h4,h5,h6 {{ font-family:'DM Sans',sans-serif; color:{NAVY}; font-weight:600; }}
hr {{ border-color:{LGRAY}; }}
[data-testid="stDataFrame"] {{ border-radius:10px; overflow:hidden; }}
.stProgress > div > div {{ background:linear-gradient(90deg,{NAVY},{GOLD}) !important; }}
.status-ok {{
    background:rgba(45,198,83,.15); border:1px solid rgba(45,198,83,.5);
    border-radius:8px; padding:10px 14px; font-size:.82rem; margin-top:12px;
}}
.status-warn {{
    background:rgba(201,168,76,.15); border:1px solid rgba(201,168,76,.5);
    border-radius:8px; padding:10px 14px; font-size:.82rem; margin-top:12px;
}}

/* ── Data Upload ── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px dashed rgba(255,255,255,0.4) !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
    color: white !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: rgba(255,255,255,0.06) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{
    color: rgba(255,255,255,0.85) !important;
}}


section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
    background: {GOLD} !important;
    color: {NAVY} !important;
    border: none !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {{
    color: rgba(255,255,255,0.85) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {{
    fill: rgba(255,255,255,0.85) !important;
    stroke: rgba(255,255,255,0.85) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p {{
    color: rgba(255,255,255,0.7) !important;
}}




</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
MODEL_COLORS = {"SVM": NAVY, "Random Forest": GOLD, "XGBoost": RED}
MODEL_ICONS  = {"SVM": "", "Random Forest": "", "XGBoost": ""}
LAYOUT_BASE  = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=NAVY),
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine formula — great-circle distance (km)
    between cardholder home and merchant location.

    Fraud signal: Unusually large distance implies the cardholder
    is unlikely to be the person conducting the transaction.
    O(n) computation using NumPy — no slow row-by-row .apply().
    """
    R   = 6371.0
    la1 = np.radians(lat1.values); lo1 = np.radians(lon1.values)
    la2 = np.radians(lat2.values); lo2 = np.radians(lon2.values)
    dlat = la2 - la1;  dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def haversine_scalar(lat1, lon1, lat2, lon2):
    """Scalar version for single-transaction prediction input."""
    R   = 6371.0
    la1,lo1,la2,lo2 = map(np.radians,[lat1,lon1,lat2,lon2])
    dlat = la2-la1; dlon = lo2-lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))


def tidy(fig, h=320):
    """Apply consistent, transparent Plotly layout."""
    fig.update_layout(height=h, **LAYOUT_BASE,
                      margin=dict(t=40,b=40,l=40,r=20))
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY)
    return fig


def kpi_html(value, label, border=GOLD):
    return f"""
    <div class="kpi-card" style="border-left-color:{border};">
      <p class="kpi-value">{value}</p>
      <p class="kpi-label">{label}</p>
    </div>"""


# ============================================================
# DATA LOADING  —  cached so it only runs once per session
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(file_path=None, uploaded=None):
    """
    Load the raw fraud dataset.
    Priority: uploaded file → local 'fraudTest.csv'.
    Index column (if present) is dropped.
    Result is cached to avoid repeated disk I/O.
    """
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return None
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


# ============================================================
# FEATURE ENGINEERING  —  cached per unique dataset
# ============================================================
@st.cache_data(show_spinner=False)
def engineer_features(df):
    """
    Feature Engineering Pipeline — 10 predictive features

    Raw columns dropped (no generalizable signal):
      cc_num, merchant, first, last, street, trans_num, unix_time, zip, job

    Derived features:
      log_amt      : log(1+amt)   — normalises right-skewed amount distribution
      hour         : 0–23         — fraud peaks late-night/early-morning
      day_of_week  : 0=Mon…6=Sun  — temporal pattern
      month        : 1–12         — seasonality
      is_weekend   : 0/1          — weekend flag (elevated fraud base rate)
      age          : years        — cardholder age from dob vs transaction date
      city_pop     : direct       — socioeconomic/urban proxy
      geo_distance : km           — Haversine cardholder↔merchant distance
      gender_enc   : F=0, M=1     — binary gender encoding
      category_enc : 0–13         — label-encoded merchant category

    Encoding: LabelEncoder (appropriate for tree-based & SVM pipelines).
    Scaling: StandardScaler fitted on TRAINING data only (see train page).

    Returns:
      df_proc      : processed DataFrame (features + target)
      FEATURES     : ordered list of feature column names
      cat_map      : dict {category_str → int_code}
      gen_map      : dict {gender_str  → int_code}
    """
    df  = df.copy()
    REF = pd.Timestamp("2020-06-21")      # Reference date (first transaction)

    # ── 1. Temporal features ────────────────────────────────
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"]       = dt.dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # ── 2. Cardholder age ───────────────────────────────────
    dob = pd.to_datetime(df["dob"])
    df["age"] = ((REF - dob).dt.days // 365).astype(int)

    # ── 3. Geographic distance (vectorised — no .apply) ─────
    df["geo_distance"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # ── 4. Log-transform amount ─────────────────────────────
    df["log_amt"] = np.log1p(df["amt"])

    # ── 5. Categorical encoding ─────────────────────────────
    le_cat = LabelEncoder()
    le_gen = LabelEncoder()
    df["category_enc"] = le_cat.fit_transform(df["category"])
    df["gender_enc"]   = le_gen.fit_transform(df["gender"])

    cat_map = dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_)))
    gen_map = dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_)))

    FEATURES = [
        "log_amt", "hour", "day_of_week", "month", "is_weekend",
        "age", "city_pop", "geo_distance", "gender_enc", "category_enc",
    ]
    TARGET = "is_fraud"

    return df[FEATURES + [TARGET]].copy(), FEATURES, cat_map, gen_map


# ============================================================
# MODEL TRAINING
# ============================================================
def train_all_models(X_tr, y_tr, X_te, y_te, seed=42):
    """
    Train and evaluate three classifiers on SMOTE-balanced data.

    Model 1 — SVM (LinearSVC + CalibratedClassifierCV)
      LinearSVC is O(n) vs kernel SVM's O(n²) — essential at financial scale.
      CalibratedClassifierCV wraps it with 3-fold Platt Scaling to produce
      calibrated probability estimates required by the risk gauge.
      C=1.0 provides standard L2 regularisation.

    Model 2 — Random Forest (200 trees)
      Ensemble of decorrelated trees trained on bootstrap samples.
      class_weight='balanced' provides secondary imbalance correction on top
      of SMOTE. max_depth=12 balances expressivity vs. overfitting.
      Provides native mean-decrease-in-impurity feature importances.

    Model 3 — XGBoost (Gradient Boosted Trees)  [Bonus model]
      Sequentially corrects residual errors of prior trees.
      Built-in L1/L2 regularisation (reg_alpha, reg_lambda).
      subsample=0.8 and colsample_bytree=0.8 add stochastic regularisation.
      Consistently achieves state-of-the-art on tabular fraud datasets.

    All three are evaluated on the UNMODIFIED test set (original imbalance)
    to simulate real-world deployment conditions accurately.

    Returns: models dict, results dict with full metric suite.
    """
    models  = {}
    results = {}

    # ── SVM ─────────────────────────────────────────────────
    svm_base = LinearSVC(C=1.0, max_iter=3000, random_state=seed)
    svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")
    svm.fit(X_tr, y_tr)
    models["SVM"] = svm

    # ── Random Forest ────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced",
        random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    models["Random Forest"] = rf

    # ── XGBoost ──────────────────────────────────────────────
    # scale_pos_weight = negatives/positives (before SMOTE, i.e., on original split)
    neg_n = int((y_tr == 0).sum()); pos_n = int((y_tr == 1).sum())
    spw   = neg_n / max(pos_n, 1)
    xgb_m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr)
    models["XGBoost"] = xgb_m

    # ── Evaluate all models on held-out test set ─────────────
    for name, model in models.items():
        yp        = model.predict(X_te)
        yprb      = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, yprb)
        results[name] = {
            "accuracy" : round(accuracy_score(y_te, yp) * 100, 2),
            "precision": round(precision_score(y_te, yp, zero_division=0) * 100, 2),
            "recall"   : round(recall_score(y_te, yp, zero_division=0) * 100, 2),
            "f1"       : round(f1_score(y_te, yp, zero_division=0) * 100, 2),
            "auc"      : round(roc_auc_score(y_te, yprb) * 100, 2),
            "cm"       : confusion_matrix(y_te, yp),
            "fpr"      : fpr, "tpr": tpr,
            "y_pred"   : yp,  "y_proba": yprb,
        }

    return models, results


# ============================================================
# PAGE 1 — HOME / DASHBOARD
# ============================================================
def page_home(df):
    # ── Hero Banner ──────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner">
      <div style="position:relative;z-index:1;">
        <div style="font-size:3.2rem;line-height:1;margin-bottom:12px;">🛡️</div>
        <h1 style="font-family:'Playfair Display',serif;color:white;font-size:2.7rem;
                   font-weight:800;margin:0 0 10px;letter-spacing:-.02em;">
          FraudShield
        </h1>
        <p style="color:rgba(255,255,255,.78);font-size:1.05rem;margin:0 0 8px;">
          Credit Card Fraud Detection Intelligence Platform
        </p>
        <p style="color:{GOLD2};font-size:.8rem;margin:0;letter-spacing:.1em;font-weight:600;">
          INTELLIGENT ANOMALY DETECTION POWERED BY MACHINE LEARNING &nbsp
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)



    # ── KPI Strip ────────────────────────────────────────────
    total   = len(df)
    fraud_n = int(df["is_fraud"].sum())
    legit_n = total - fraud_n
    rate    = fraud_n / total * 100
    f_amt   = df[df["is_fraud"]==1]["amt"].sum()
    avg_f   = df[df["is_fraud"]==1]["amt"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi_html(f"{total:,}",   "Total Transactions"), unsafe_allow_html=True)
    c2.markdown(kpi_html(f"{fraud_n:,}", "Fraudulent Transactions", RED),   unsafe_allow_html=True)
    c3.markdown(kpi_html(f"{legit_n:,}", "Legitimate Transactions", GREEN), unsafe_allow_html=True)
    c4.markdown(kpi_html(f"{rate:.2f}%", "Fraud Rate",    RED),  unsafe_allow_html=True)
    c5.markdown(kpi_html(f"${f_amt:,.0f}", "Total Fraud Amount $", NAVY), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Category bar ──────────────────────────
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("#### Transaction Class Split")
        fig = go.Figure(go.Pie(
            labels=["Legitimate","Fraudulent"],
            values=[legit_n, fraud_n],
            hole=0.62,
            marker_colors=[GREEN, RED],
            textinfo="label+percent",
            pull=[0, 0.06],
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{rate:.2f}%</b><br><span style='font-size:11px'>Fraud</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=NAVY),
        )
        fig.update_layout(
            showlegend=True, height=300, **LAYOUT_BASE,
            margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(orientation="h", y=-0.14),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Fraud Rate by Merchant Category")
        cat = df.groupby("category")["is_fraud"].agg(["sum","count"]).reset_index()
        cat["rate"] = cat["sum"] / cat["count"] * 100
        cat = cat.sort_values("rate", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=cat["rate"], y=cat["category"],
            orientation="h",
            marker=dict(color=cat["rate"],
                        colorscale=[[0,GOLD],[0.5,GOLD],[1,RED]],
                        showscale=False),
            text=cat["rate"].round(2).astype(str)+"%",
            textposition="outside",
        ))
        tidy(fig2, 310)
        fig2.update_layout(xaxis_title="Fraud Rate (%)", yaxis_title="",
                           margin=dict(t=10,b=10,l=10,r=80))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Hour trend + Amount dist ──────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Fraud Count by Hour of Day")
        tmp = df.copy()
        tmp["hour"] = pd.to_datetime(tmp["trans_date_trans_time"]).dt.hour
        hrly = tmp[tmp["is_fraud"]==1].groupby("hour").size().reset_index(name="cnt")
        fig3 = go.Figure(go.Scatter(
            x=hrly["hour"], y=hrly["cnt"],
            mode="lines+markers",
            line=dict(color=RED, width=2.5),
            marker=dict(size=7, color=RED, line=dict(color=WHITE,width=1.5)),
            fill="tozeroy", fillcolor="rgba(214,64,69,.12)",
        ))
        tidy(fig3, 270)
        fig3.update_layout(xaxis_title="Hour (0–23)", yaxis_title="# Fraud Transactions",
                           xaxis=dict(tickmode="linear", dtick=3, gridcolor=LGRAY),
                           margin=dict(t=10,b=45,l=55,r=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Amount Distribution - Fraud vs Legitimate")
        l_samp = df[df["is_fraud"]==0]["amt"].clip(upper=600).sample(8000, random_state=42)
        f_samp = df[df["is_fraud"]==1]["amt"].clip(upper=600)
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=l_samp, nbinsx=60, name="Legitimate",
                                    marker_color=GREEN, opacity=0.55))
        fig4.add_trace(go.Histogram(x=f_samp, nbinsx=60, name="Fraud",
                                    marker_color=RED, opacity=0.85))
        fig4.update_layout(barmode="overlay", height=270, **LAYOUT_BASE,
                           xaxis_title="Amount (USD, clipped $600)", yaxis_title="Count",
                           legend=dict(orientation="h", y=1.12),
                           margin=dict(t=30,b=45,l=55,r=20))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Insight callout ───────────────────────────────────────
    st.markdown(f"""
    <div class="warn-box">
    <b>  The Accuracy Trap:</b>  Only <b>{rate:.2f}%</b> of {total:,} transactions are
    fraudulent. A naïve classifier that labels <i>everything</i> as "Legitimate" scores
    <b>{100-rate:.2f}% accuracy</b>,  yet catches <b>zero fraud</b> and fails completely.
    This is why FraudShield applies <b>SMOTE oversampling</b> and benchmarks models on
    <b>Recall</b>, <b>F1-Score</b>, and <b>AUC-ROC</b>, not raw accuracy.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — DATA EXPLORER
# ============================================================
def page_data_explorer(df):
    st.markdown('<p class="page-title"> Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Examine the raw transaction dataset before any transformation.</p>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs([" Raw Data"," Statistics"," Distributions"," Correlation"])

    # ── TAB 1: Raw Data ──────────────────────────────────────
    with tab1:
        fraud_n = int(df["is_fraud"].sum())
        st.markdown(f"""
        <div class="info-box">
        <b>{len(df):,}</b> transactions &nbsp;·&nbsp; <b>{df.shape[1]}</b> columns &nbsp;·&nbsp;
        Fraud: <b>{fraud_n:,}</b> ({df['is_fraud'].mean()*100:.2f}%)
        &nbsp;·&nbsp; Missing values: <b>None </b>
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns([3,1])
        with c1: n = st.slider("Rows to preview:", 5, 200, 25)
        with c2: fraud_only = st.checkbox("Fraud only")
        disp = df[df["is_fraud"]==1] if fraud_only else df
        st.dataframe(disp.head(n), use_container_width=True)

        st.markdown("#### Feature Glossary")
        gl = pd.DataFrame({
            "Feature":["trans_date_trans_time","cc_num","merchant","category","amt",
                       "first / last","gender","city / state / zip",
                       "lat / long","city_pop","job","dob",
                       "merch_lat / merch_long","is_fraud"],
            "Description":[
                "Transaction timestamp → hour, day-of-week, month extracted",
                "Credit card number — unique identifier, dropped in modelling",
                "Merchant name — too sparse (>600 unique) for direct encoding",
                "Merchant category (14 classes) → label-encoded",
                "Transaction amount (USD, right-skewed) → log-transformed",
                "Cardholder name — PII with no predictive signal",
                "Cardholder gender (M/F) → binary-encoded",
                "Address fields — city_pop retained; others dropped",
                "Cardholder GPS coordinates → Haversine distance to merchant",
                "City population — socioeconomic / urban-density proxy",
                "Job title — too sparse for reliable encoding in this setup",
                "Date of birth → engineered to cardholder age in years",
                "Merchant GPS → paired with cardholder coords for distance",
                " TARGET — 1 = Fraud, 0 = Legitimate",
            ],
            "Used in Model?":["→ hour/dow/month"," Dropped"," Dropped"," Encoded",
                              " log_amt"," Dropped"," Encoded","→ city_pop",
                              " geo_distance"," Direct"," Dropped"," Age",
                              " geo_distance"," Target"],
        })
        st.dataframe(gl, use_container_width=True, hide_index=True)

    # ── TAB 2: Statistics ─────────────────────────────────────
    with tab2:
        st.markdown("#### Numerical Summary Statistics")
        num_c = ["amt","city_pop","lat","long","merch_lat","merch_long","is_fraud"]
        st.dataframe(df[num_c].describe().T.round(3), use_container_width=True)

        st.markdown("#### Categorical Summary")
        rows = []
        for c in ["category","gender","state","merchant"]:
            vc = df[c].value_counts()
            rows.append({"Column":c, "Unique Values":df[c].nunique(),
                         "Most Common":vc.index[0], "Frequency":int(vc.iloc[0]),
                         "Least Common":vc.index[-1]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Fraud by gender bar
        gen = df.groupby("gender")["is_fraud"].agg(["sum","count"]).reset_index()
        gen["rate"] = gen["sum"]/gen["count"]*100
        fig = px.bar(gen, x="gender", y="rate",
                     color="gender", color_discrete_map={"M":NAVY,"F":GOLD},
                     title="Fraud Rate by Gender (%)",
                     labels={"rate":"Fraud Rate (%)","gender":"Gender"})
        tidy(fig, 300)
        fig.update_layout(showlegend=False, margin=dict(t=40,b=30,l=60,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Distributions ──────────────────────────────────
    with tab3:
        feat = st.selectbox("Feature:", ["amt (Transaction Amount)","city_pop (City Population)"])
        col  = "amt" if "amt" in feat else "city_pop"

        c1,c2 = st.columns(2)
        with c1:
            samp = df.sample(min(50_000, len(df)), random_state=42)
            fig = px.histogram(samp, x=col, color="is_fraud",
                               color_discrete_map={0:GREEN,1:RED},
                               barmode="overlay", nbins=60,
                               title=f"Distribution of {feat}",
                               labels={"is_fraud":"Is Fraud"})
            tidy(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Top 12 States by Fraud Volume")
            sf = (df[df["is_fraud"]==1]
                  .groupby("state").size()
                  .reset_index(name="count")
                  .sort_values("count", ascending=False)
                  .head(12))
            fig2 = px.bar(sf, x="state", y="count", color="count",
                          color_continuous_scale=[[0,GOLD],[1,RED]])
            tidy(fig2, 300)
            fig2.update_layout(coloraxis_showscale=False, yaxis_title="Fraud Count",
                               margin=dict(t=40,b=30,l=55,r=20))
            st.plotly_chart(fig2, use_container_width=True)

        # Fraud by day of week
        st.markdown("#### Fraud Volume by Day of Week")
        tmp2 = df.copy()
        tmp2["dow"] = pd.to_datetime(tmp2["trans_date_trans_time"]).dt.dayofweek
        dow_f = tmp2[tmp2["is_fraud"]==1].groupby("dow").size().reset_index(name="cnt")
        dow_f["day"] = dow_f["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",
                                          4:"Fri",5:"Sat",6:"Sun"})
        fig3 = go.Figure(go.Bar(x=dow_f["day"], y=dow_f["cnt"],
                                marker_color=[RED if d>=5 else NAVY for d in dow_f["dow"]]))
        tidy(fig3, 280)
        fig3.update_layout(yaxis_title="# Fraud Transactions",
                           margin=dict(t=10,b=30,l=55,r=20))
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 4: Correlation ────────────────────────────────────
    with tab4:
        st.markdown("#### Pearson Correlation Matrix — Numerical Features")
        nc = df[["amt","city_pop","lat","long","merch_lat","merch_long",
                 "unix_time","is_fraud"]]
        corr = nc.corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,WHITE],[0.5,GOLD2],[1,NAVY]],
            text=corr.values, texttemplate="%{text:.2f}",
            showscale=True, zmin=-1, zmax=1,
        ))
        tidy(fig, 480)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
          Raw correlations with <code>is_fraud</code> are weak because fraud signals are
        largely <b>non-linear</b> and <b>cross-feature</b> in nature. This motivates our
        feature engineering stage — derived features (<code>geo_distance</code>,
        <code>log_amt</code>, <code>hour</code>, <code>age</code>) expose
        patterns invisible in this raw-feature matrix.
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 3 — PREPROCESSING
# ============================================================
def page_preprocessing(df):
    st.markdown('<p class="page-title">Preprocessing Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Feature engineering · Encoding · Standardisation · SMOTE class balancing.</p>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(
        [" Feature Engineering"," Encoding & Scaling"," SMOTE"," Processed Dataset"])

    # ── TAB 1: Feature Engineering ───────────────────────────
    with tab1:
        st.markdown(f"""
        <div class="info-box">
        Raw identifier columns (names, card numbers, addresses) carry zero
        generalizable predictive signal. We engineer <b>10 structured features</b>
        from the raw data — each grounded in fraud-detection literature.
        </div>""", unsafe_allow_html=True)

        steps = [
            ("","Temporal Features",
             "hour · day_of_week · month · is_weekend",
             "Fraud spikes 00:00–03:00 and on weekends. Extracting time components lets "
             "models capture these temporal rhythms that flat attributes would miss.","HIGH",RED),
            ("","Cardholder Age",
             "age = (txn_date − dob) ÷ 365",
             "Derived from date-of-birth. Age correlates with fraud susceptibility; "
             "certain demographic cohorts are disproportionately targeted.","MEDIUM",GOLD),
            ("","Geographic Distance",
             "geo_distance — Haversine (km)",
             "Distance between cardholder home and merchant. Distances >100 km strongly "
             "indicate stolen-card usage. Computed vectorially (no slow .apply()).","HIGH",RED),
            ("","Log Amount",
             "log_amt = log(1 + amt)",
             "Raw amounts are right-skewed. Log-transform normalises the distribution, "
             "compresses large outliers, and prevents high values from dominating models.","HIGH",RED),
            ("","Category Encoding",
             "category_enc — LabelEncoder (14 classes → 0–13)",
             "Fraud rates vary >10× across categories. LabelEncoding is appropriate here "
             "because tree splits don't assume ordinality.","MEDIUM",GOLD),
            ("","Gender Encoding",
             "gender_enc — F=0, M=1",
             "Binary encoding of cardholder gender. Modest individual contribution; "
             "participates in interaction effects within ensemble models.","LOW",GRAY),
        ]
        for ic,nm,feat,rationale,impact,color in steps:
            st.markdown(f"""
            <div class="model-card" style="margin-bottom:12px;border-top-color:{color};">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <span style="font-size:1.6rem;">{ic}</span>
                <span style="background:{color};color:white;padding:2px 12px;
                             border-radius:12px;font-size:.74rem;font-weight:700;">
                  {impact} SIGNAL
                </span>
              </div>
              <b style="color:{NAVY};font-size:1rem;">{nm}</b>
              &nbsp;→&nbsp;<code style="color:{GOLD};font-size:.85rem;">{feat}</code><br>
              <p style="color:{GRAY};font-size:.86rem;margin:8px 0 0;line-height:1.55;">
                {rationale}
              </p>
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: Encoding & Scaling ─────────────────────────────
    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="model-card">
              <h4>Label Encoding</h4>
              <p style="color:{GRAY};font-size:.87rem;line-height:1.6;">
              Applied to <code>category</code> (14 classes) and <code>gender</code> (2 classes).
              Each unique string value is mapped to an integer code. Appropriate here because
              tree-based models (RF, XGBoost) treat these as split thresholds and do <i>not</i>
              assume ordinality. SVM benefits from the z-score normalisation applied afterward.
              </p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="model-card">
              <h4>StandardScaler (Z-score Normalisation)</h4>
              <p style="color:{GRAY};font-size:.87rem;line-height:1.6;">
              Transforms each feature to mean=0, std=1. <b>Critical for SVM</b> — the
              maximal-margin hyperplane is scale-sensitive; large-range features dominate.
              The scaler is <code>fit_transform</code>-ed on <code>X_train</code> only,
              then <code>transform</code>-ed on <code>X_test</code>.
              Fitting on test data would constitute <b>data leakage</b>.
              </p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:18px;">
         <b>Leakage Prevention:</b> The StandardScaler is fitted exclusively on the
        training split. Test-set statistics (mean, std) never contaminate the training
        process — a common but critical mistake in student ML pipelines.
        </div>""", unsafe_allow_html=True)

    # ── TAB 3: SMOTE ──────────────────────────────────────────
    with tab3:
        st.markdown("### SMOTE — Synthetic Minority Over-sampling Technique")
        st.markdown(f"""
        <div class="warn-box">
        <b>  Why accuracy alone deceives:</b> With 99.61% legitimate transactions,
        any model that always predicts "Legitimate" achieves 99.61% accuracy while
        catching <b>zero fraud</b>. SMOTE corrects the training distribution without
        discarding majority-class data — unlike random under-sampling.
        </div>""", unsafe_allow_html=True)

        total_n = len(df)
        fraud_n = int(df["is_fraud"].sum())
        legit_n = total_n - fraud_n
        # Approximate training split (80%)
        tr_leg = int(legit_n * 0.8); tr_fr = int(fraud_n * 0.8)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Before SMOTE — Training Split**")
            fig = go.Figure(go.Bar(
                x=["Legitimate","Fraudulent"], y=[tr_leg, tr_fr],
                marker_color=[GREEN, RED],
                text=[f"{tr_leg:,}", f"{tr_fr:,}"],
                textposition="outside", width=[0.5,0.5],
            ))
            tidy(fig, 310)
            fig.update_layout(
                title=dict(text=f"Ratio ≈ {tr_leg//max(tr_fr,1)}:1  (severe imbalance)",
                           font=dict(size=13,color=NAVY)),
                yaxis=dict(gridcolor=LGRAY), margin=dict(t=50,b=20,l=45,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**After SMOTE — Balanced Training Set**")
            fig2 = go.Figure(go.Bar(
                x=["Legitimate","Fraud (synth + real)"],
                y=[tr_leg, tr_leg],
                marker_color=[GREEN, GOLD],
                text=[f"{tr_leg:,}", f"{tr_leg:,}"],
                textposition="outside", width=[0.5,0.5],
            ))
            tidy(fig2, 310)
            fig2.update_layout(
                title=dict(text="1:1 ratio — balanced training data",
                           font=dict(size=13,color=NAVY)),
                yaxis=dict(gridcolor=LGRAY), margin=dict(t=50,b=20,l=45,r=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
         <b>How SMOTE works:</b> For each minority (fraud) sample, SMOTE selects its
        <i>k</i> nearest fraud neighbours in feature space and generates new synthetic samples
        by linear interpolation along those connecting lines. This is fundamentally different
        from simple duplication — synthetic points explore new regions of the fraud manifold,
        giving the classifier a richer, more robust decision boundary.<br><br>
         <b>SMOTE is applied to the training set ONLY.</b>
        The test set retains the original imbalanced distribution to simulate
        real-world deployment conditions accurately.
        </div>""", unsafe_allow_html=True)

    # ── TAB 4: Processed Dataset ──────────────────────────────
    with tab4:
        with st.spinner("Running feature engineering…"):
            processed, feature_cols, cat_map, gen_map = engineer_features(df)

        st.markdown(f"**Processed matrix: {len(feature_cols)} features × {len(processed):,} rows**")
        st.dataframe(processed.head(40), use_container_width=True)

        st.markdown("#### Engineered Feature Correlation with Target (is_fraud)")
        corr_t = (processed.corr()["is_fraud"]
                  .drop("is_fraud")
                  .sort_values(key=abs, ascending=False))
        fig = go.Figure(go.Bar(
            x=corr_t.index, y=corr_t.values,
            marker_color=[RED if v<0 else GOLD for v in corr_t.values],
            text=[f"{v:.4f}" for v in corr_t.values], textposition="outside",
        ))
        tidy(fig, 310)
        fig.update_layout(yaxis_title="Pearson ρ  with  is_fraud",
                          margin=dict(t=10,b=50,l=65,r=20))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4 — MODEL TRAINING
# ============================================================
def page_training(df):
    st.markdown('<p class="page-title"> Model Training</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Configure, train, and inspect SVM · Random Forest · XGBoost on SMOTE-balanced data.</p>', unsafe_allow_html=True)

    # ── Config panel ─────────────────────────────────────────
    st.markdown("###   Training Configuration")
    c1,c2,c3 = st.columns(3)
    with c1:
        sample_n = st.slider("Training sample size:", 10_000, 80_000, 40_000, 5_000,
                             help="Stratified subset preserving the fraud/non-fraud ratio.")
    with c2:
        test_pct = st.slider("Test split %:", 15, 35, 20, 5) / 100
    with c3:
        seed = int(st.number_input("Random seed:", value=42, step=1))

    st.markdown(f"""
    <div class="info-box">
     <b>Why stratified sampling?</b> The full dataset has 555,719 rows. Kernel SVMs
    are O(n²) — computationally infeasible at this scale. We use a
    <b>stratified sample of {sample_n:,} rows</b> preserving the 0.39% fraud ratio.
    <b>LinearSVC</b> (O(n) complexity) + Platt calibration is used — the theoretically
    correct approach for large-scale fraud detection.
    </div>""", unsafe_allow_html=True)

    if st.button("  Train All Three Models"):
        # Feature engineering
        with st.spinner("Engineering features…"):
            processed, feature_cols, cat_map, gen_map = engineer_features(df)

        # Stratified sample
        _, samp = train_test_split(
            processed,
            test_size=min(sample_n / len(processed), 0.95),
            stratify=processed["is_fraud"],
            random_state=seed,
        )
        X = samp[feature_cols].values
        y = samp["is_fraud"].values

        # Holdout split (stratified)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_pct, stratify=y, random_state=seed
        )

        prog = st.progress(0)

        # Scale (fit on train only — no leakage)
        prog.progress(5, "Standardising features (Z-score, fit on train)…")
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # SMOTE (training set only)
        prog.progress(15, "Applying SMOTE to training split…")
        smote = SMOTE(random_state=seed, k_neighbors=5)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_sc, y_tr)
        bf = int(y_tr.sum()); af = int(y_tr_res.sum())

        prog.progress(28, f"SMOTE: fraud samples {bf:,} → {af:,}  |  Training SVM…")
        time.sleep(0.15)

        # Train
        models, results = train_all_models(X_tr_res, y_tr_res, X_te_sc, y_te, seed)

        prog.progress(90, "Computing evaluation metrics…")
        time.sleep(0.15)
        prog.progress(100, "  All models trained!")

        # Persist to session state
        st.session_state.update({
            "models":       models,  "results":      results,
            "scaler":       scaler,  "cat_map":      cat_map,
            "gen_map":      gen_map, "feature_cols": feature_cols,
            "X_te":         X_te_sc, "y_te":         y_te,
            "smote_before": bf,      "smote_after":  af,
            "train_n":      len(X_tr_res),
            "test_n":       len(X_te_sc),
        })

        st.success(
            f"  Done!  Train: {len(X_tr_res):,} (post-SMOTE) · Test: {len(X_te_sc):,}"
        )

    # ── Post-training panels ──────────────────────────────────
    if "models" in st.session_state:
        st.markdown("---")
        st.markdown("###   Model Specifications")

        specs = [
            ("","Support Vector Machine",NAVY,
             "LinearSVC + CalibratedClassifierCV",
             "C=1.0 · max_iter=3000 · Platt scaling (cv=3)",
             ("Constructs the hyperplane that maximises the margin between fraud and legitimate "
              "classes in high-dimensional feature space. LinearSVC scales linearly with sample "
              "count — essential at financial-data scale. CalibratedClassifierCV wraps it with "
              "3-fold Platt Scaling to produce calibrated probability estimates required by the "
              "risk gauge and ROC curve.")),
            ("","Random Forest",GOLD,
             "RandomForestClassifier (200 trees)",
             "max_depth=12 · class_weight='balanced' · n_jobs=-1",
             ("Ensemble of 200 decorrelated decision trees trained on bootstrap samples. "
              "Variance is reduced by averaging. class_weight='balanced' provides a secondary "
              "imbalance correction layer on top of SMOTE. max_depth=12 balances expressivity "
              "vs overfitting. Provides interpretable mean-decrease-in-impurity feature importances.")),
            ("","XGBoost ",RED,
             "XGBClassifier (gradient-boosted trees)",
             "n_estimators=200 · max_depth=6 · lr=0.1 · subsample=0.8",
             ("Sequentially builds trees correcting the residual errors of the prior ensemble. "
              "Built-in L1/L2 regularisation. subsample=0.8 and colsample_bytree=0.8 add "
              "stochastic regularisation. Consistently achieves state-of-the-art performance "
              "on tabular fraud datasets and Kaggle leaderboards.")),
        ]
        cols = st.columns(3)
        for col,(ic,nm,cl,mt,pm,desc) in zip(cols,specs):
            col.markdown(f"""
            <div class="model-card" style="border-top-color:{cl};">
              <div style="font-size:2rem;">{ic}</div>
              <h4 style="color:{cl};margin:4px 0 2px;">{nm}</h4>
              <code style="font-size:.77rem;color:{GRAY};">{mt}</code><br>
              <code style="font-size:.77rem;color:{GOLD};">{pm}</code>
              <hr style="border-color:{LGRAY};margin:10px 0;">
              <p style="color:{GRAY};font-size:.82rem;line-height:1.55;margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

        # Feature Importances (RF & XGBoost side by side)
        st.markdown("---")
        st.markdown("###   Feature Importance")
        fc = st.session_state["feature_cols"]
        ca,cb = st.columns(2)

        with ca:
            fi_rf = pd.Series(
                st.session_state["models"]["Random Forest"].feature_importances_,
                index=fc).sort_values()
            fig = go.Figure(go.Bar(x=fi_rf.values, y=fi_rf.index, orientation="h",
                                   marker_color=GOLD,
                                   text=[f"{v:.3f}" for v in fi_rf.values],
                                   textposition="outside"))
            tidy(fig, 320)
            fig.update_layout(title="Random Forest — Mean Decrease in Impurity",
                              xaxis_title="Feature Importance",
                              margin=dict(t=50,b=20,l=10,r=65))
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            fi_xgb = pd.Series(
                st.session_state["models"]["XGBoost"].feature_importances_,
                index=fc).sort_values()
            fig2 = go.Figure(go.Bar(x=fi_xgb.values, y=fi_xgb.index, orientation="h",
                                    marker_color=RED,
                                    text=[f"{v:.3f}" for v in fi_xgb.values],
                                    textposition="outside"))
            tidy(fig2, 320)
            fig2.update_layout(title="XGBoost — F-score Importance",
                               xaxis_title="Feature Importance",
                               margin=dict(t=50,b=20,l=10,r=65))
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown(f"""
        <div class="info-box">  Set your parameters and click
        <b>Train All Three Models</b> to begin. Typical runtime: 30–90 s
        depending on sample size.
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 5 — MODEL EVALUATION
# ============================================================
def page_evaluation():
    st.markdown('<p class="page-title"> Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Holdout validation · Classification metrics · Confusion matrices · ROC curves</p>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.warning("  No trained models found. Please go to **Model Training** first.")
        return

    results = st.session_state["results"]
    mc = MODEL_COLORS; mi = MODEL_ICONS

    # ── Performance Metric Cards ──────────────────────────────
    st.markdown("###  Performance Metrics — All Models")
    cols = st.columns(len(results))
    best_f1 = max(results, key=lambda k: results[k]["f1"])

    for col,(name,r) in zip(cols, results.items()):
        crown = "SVM" if name == best_f1 else ""
        col.markdown(f"""
        <div class="model-card" style="border-top-color:{mc[name]};text-align:center;">
          <div style="font-size:2rem;">{mi[name]}</div>
          <h4 style="color:{mc[name]};margin:4px 0 10px;">{name}{crown}</h4>
          <hr style="border-color:{LGRAY};">
          <table style="width:100%;font-size:.9rem;border-collapse:collapse;">
            <tr>
              <td style="color:{GRAY};text-align:left;padding:5px 0;">Accuracy</td>
              <td style="color:{NAVY};font-weight:700;text-align:right;">{r['accuracy']}%</td>
            </tr>
            <tr>
              <td style="color:{GRAY};text-align:left;padding:5px 0;">Precision</td>
              <td style="color:{NAVY};font-weight:700;text-align:right;">{r['precision']}%</td>
            </tr>
            <tr style="background:#FFF8E7;">
              <td style="color:{GRAY};text-align:left;padding:5px 4px;">Recall </td>
              <td style="color:{RED};font-weight:700;text-align:right;">{r['recall']}%</td>
            </tr>
            <tr style="background:#FFF8E7;">
              <td style="color:{GRAY};text-align:left;padding:5px 4px;">F1-Score </td>
              <td style="color:{GOLD};font-weight:700;text-align:right;">{r['f1']}%</td>
            </tr>
            <tr>
              <td style="color:{GRAY};text-align:left;padding:5px 0;">AUC-ROC</td>
              <td style="color:{NAVY};font-weight:700;text-align:right;">{r['auc']}%</td>
            </tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top:18px;">
     <b>Primary metrics for fraud detection:</b>
    <b>Recall</b> (minimise missed fraud = direct financial loss) and
    <b>F1-Score</b> (harmonic mean of Precision and Recall; balances both error types).
    Accuracy is reported for completeness but is <i>not</i> the decision metric
    given the severe class imbalance.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Radar ─────────────────────────────────────────────────
    st.markdown("###   Comparative Performance Radar")
    cats = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
    fig = go.Figure()
    for name,r in results.items():
        vals = [r["accuracy"],r["precision"],r["recall"],r["f1"],r["auc"]]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", name=f"{mi[name]} {name}",
            line=dict(color=mc[name], width=2.5),
            fillcolor=mc[name], opacity=0.12,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[50,100],
                                   tickfont=dict(size=10))),
        showlegend=True, height=430, **LAYOUT_BASE,
        margin=dict(t=30,b=30,l=30,r=30),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Confusion Matrices ─────────────────────────────────────
    st.markdown("###   Confusion Matrices")
    cm_cols = st.columns(len(results))
    for col,(name,r) in zip(cm_cols, results.items()):
        tn,fp,fn,tp = r["cm"].ravel()
        fig = go.Figure(go.Heatmap(
            z=[[tn,fp],[fn,tp]],
            x=["Pred: Legit","Pred: Fraud"],
            y=["Actual: Legit","Actual: Fraud"],
            colorscale=[[0,WHITE],[0.5,GOLD2],[1,mc[name]]],
            text=[[f"TN\n{tn:,}",f"FP\n{fp:,}"],[f"FN\n{fn:,}",f"TP\n{tp:,}"]],
            texttemplate="%{text}", textfont=dict(size=12),
            showscale=False,
        ))
        fig.update_layout(
            title=dict(text=f"{mi[name]}  {name}", font=dict(size=14,color=mc[name])),
            height=280, **LAYOUT_BASE, margin=dict(t=44,b=10,l=10,r=10),
        )
        col.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
    <b>Confusion Matrix Guide:</b>
    <b>TN</b> = legitimate correctly cleared ·
    <b>TP</b> = fraud correctly flagged ·
    <b>FP</b> = legitimate wrongly flagged (false alarm → customer friction) ·
    <b style="color:{RED};">FN</b> = fraud missed → <b>direct financial loss</b>.
    FN is the highest-cost error type in fraud detection — the overriding goal is to minimise it.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── ROC Curves ────────────────────────────────────────────
    st.markdown("###   ROC Curves - Model Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dot", color=GRAY, width=1.5),
        name="Random Baseline (AUC = 50%)",
    ))
    for name,r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{mi[name]} {name} — AUC {r['auc']}%",
            line=dict(color=mc[name], width=2.5),
        ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        height=440, **LAYOUT_BASE,
        legend=dict(x=0.54, y=0.17, bgcolor="rgba(255,255,255,.88)",
                    bordercolor=LGRAY, borderwidth=1),
        margin=dict(t=20,b=55,l=65,r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Side-by-side bar comparison ───────────────────────────
    st.markdown("###   Side-by-Side Metric Comparison")
    mkeys = ["accuracy","precision","recall","f1","auc"]
    mlabs = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"]
    fig = go.Figure()
    for name,r in results.items():
        fig.add_trace(go.Bar(
            name=f"{mi[name]} {name}",
            x=mlabs, y=[r[m] for m in mkeys],
            marker_color=mc[name],
            text=[f"{r[m]}%" for m in mkeys],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=400, **LAYOUT_BASE,
        yaxis=dict(range=[0,118], gridcolor=LGRAY),
        yaxis_title="Score (%)",
        margin=dict(t=20,b=40,l=65,r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 6 — FRAUD PREDICTOR
# ============================================================
def page_prediction():
    st.markdown('<p class="page-title">  Real-Time Fraud Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Enter transaction details to receive an instant fraud risk score and decision.</p>', unsafe_allow_html=True)

    if "models" not in st.session_state:
        st.warning("  Models not yet trained. Please visit **Model Training** first.")
        return

    models  = st.session_state["models"]
    scaler  = st.session_state["scaler"]
    cat_map = st.session_state["cat_map"]
    gen_map = st.session_state["gen_map"]

    st.markdown("###   Transaction Details")
    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("  Transaction")
        amt  = st.number_input("Amount ($)", 0.01, 100_000.0, 250.0, 0.01)
        hour = st.slider("Hour of Day", 0, 23, 2, help="0=midnight, 23=11PM")
        dow  = st.selectbox("Day of Week", list(range(7)),
                            format_func=lambda x:
                            ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        month = st.selectbox("Month", list(range(1,13)),
                             format_func=lambda x:
                             ["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

    with c2:
        st.markdown("  Cardholder")
        gender   = st.selectbox("Gender", ["F","M"])
        age      = st.slider("Age", 18, 90, 62)
        city_pop = st.number_input("City Population", 100, 5_000_000, 32_000, 1000)
        category = st.selectbox("Merchant Category",
                                sorted(cat_map.keys()) if cat_map else ["misc_net"])

    with c3:
        st.markdown("  Location")
        c_lat = st.number_input("Cardholder Latitude",  -90.0,  90.0,  33.97, 0.001)
        c_lon = st.number_input("Cardholder Longitude",-180.0, 180.0, -80.94, 0.001)
        m_lat = st.number_input("Merchant Latitude",    -90.0,  90.0,  40.71, 0.001)
        m_lon = st.number_input("Merchant Longitude",  -180.0, 180.0, -74.01, 0.001)

    model_choice = st.selectbox("  Prediction Model:", list(models.keys()))

    if st.button("  Analyse This Transaction"):
        # ── Build feature vector ──────────────────────────────
        is_wk    = 1 if dow >= 5 else 0
        log_amt  = float(np.log1p(amt))
        geo_dist = haversine_scalar(c_lat, c_lon, m_lat, m_lon)
        g_enc    = int(gen_map.get(gender, 0))
        c_enc    = int(cat_map.get(category, 0))

        feats    = np.array([[log_amt, hour, dow, month, is_wk,
                              age, city_pop, geo_dist, g_enc, c_enc]])
        feats_sc = scaler.transform(feats)

        model  = models[model_choice]
        pred   = int(model.predict(feats_sc)[0])
        prob   = float(model.predict_proba(feats_sc)[0][1])
        prob_p = prob * 100

        st.markdown("---")
        st.markdown("###   Risk Assessment")

        # Gauge colour mapping
        gauge_col = (RED   if prob_p > 50 else
                     GOLD  if prob_p > 25 else GREEN)

        # ── Risk Gauge ───────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_p,
            number={"suffix":"%",
                    "font":{"size":54,"color":NAVY,"family":"Playfair Display"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickcolor":NAVY,
                        "tickvals":[0,25,50,75,100],
                        "ticktext":["0%","25%","50%","75%","100%"]},
                "bar":{"color":gauge_col,"thickness":0.25},
                "bgcolor":WHITE, "borderwidth":2, "bordercolor":LGRAY,
                "steps":[
                    {"range":[0,25],  "color":"#D4EDDA"},
                    {"range":[25,50], "color":"#FFF3CD"},
                    {"range":[50,75], "color":"#FFD9C0"},
                    {"range":[75,100],"color":"#F8D7DA"},
                ],
                "threshold":{"line":{"color":RED,"width":5},
                             "thickness":0.75,"value":50},
            },
            title={"text":(f"Fraud Probability Score<br>"
                           f"<span style='font-size:.75em;color:{GRAY};'>"
                           f"Model: {model_choice}</span>"),
                   "font":{"size":18,"color":NAVY}},
        ))
        fig.update_layout(
            height=390, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=64,b=24,l=52,r=52),
            font={"color":NAVY,"family":"DM Sans"},
        )

        g_col, d_col = st.columns([1,1])

        with g_col:
            st.plotly_chart(fig, use_container_width=True)

        with d_col:
            verdict = "  FRAUD DETECTED"  if pred==1 else "  LEGITIMATE"
            v_color = RED if pred==1 else GREEN
            risk_lv = ("🔴 CRITICAL" if prob_p > 75 else
                       "🟠 HIGH"     if prob_p > 50 else
                       "🟡 MODERATE" if prob_p > 25 else
                       "🟢 LOW")

            st.markdown(f"""
            <div class="model-card" style="margin-top:16px;border-top-color:{v_color};">
              <h3 style="color:{v_color};font-size:1.6rem;margin:0 0 12px;">{verdict}</h3>
              <hr style="border-color:{LGRAY};margin:0 0 12px;">
              <table style="width:100%;font-size:.91rem;border-collapse:collapse;">
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Fraud Probability</td>
                  <td style="color:{gauge_col};font-weight:800;font-size:1.3rem;
                             text-align:right;">{prob_p:.2f}%</td>
                </tr>
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Risk Level</td>
                  <td style="font-weight:700;text-align:right;">{risk_lv}</td>
                </tr>
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Model</td>
                  <td style="text-align:right;">{model_choice}</td>
                </tr>
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Amount</td>
                  <td style="text-align:right;">${amt:,.2f}</td>
                </tr>
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Geo Distance</td>
                  <td style="text-align:right;">{geo_dist:.1f} km</td>
                </tr>
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Hour</td>
                  <td style="text-align:right;">
                    {" Late Night" if hour<5 or hour>22 else " Business Hours"}
                  </td>
                </tr>
                <tr>
                  <td style="color:{GRAY};padding:6px 0;">Category</td>
                  <td style="text-align:right;">{category}</td>
                </tr>
              </table>
            </div>""", unsafe_allow_html=True)

            if pred == 1:
                st.error("  Recommended: **Block transaction & notify cardholder immediately.**")
            else:
                st.success("  Transaction cleared. No immediate action required.")

        # ── Signal Breakdown ──────────────────────────────────
        st.markdown("####   Transaction Risk Signal Breakdown")
        signals = [
            ("Hour", f"{'🔴 Late night peak' if hour<5 or hour>22 else '🟢 Normal hours'}",
             RED if (hour<5 or hour>22) else GREEN),
            ("Amount", f"{'🔴 Large ($'+str(round(amt,0))[:6]+')' if amt>500 else '🟢 Normal ($'+str(round(amt,2))+')'}",
             RED if amt>500 else GREEN),
            ("Distance", f"{'🔴 '+str(round(geo_dist,1))+' km far' if geo_dist>100 else '🟢 '+str(round(geo_dist,1))+' km near'}",
             RED if geo_dist>100 else GREEN),
            ("Day", f"{'🔴 Weekend' if is_wk else '🟢 Weekday'}",
             RED if is_wk else GREEN),
            ("Category", f"{'🔴 High-risk' if category in ['shopping_net','misc_net','grocery_pos'] else '🟢 Lower-risk'}",
             RED if category in ["shopping_net","misc_net","grocery_pos"] else GREEN),
        ]
        sig_cols = st.columns(5)
        for sc,(sig,status,color) in zip(sig_cols, signals):
            sc.markdown(f"""
            <div style="background:{WHITE};border-radius:10px;padding:14px 10px;
                        border-top:3px solid {color};text-align:center;
                        box-shadow:0 2px 8px rgba(10,35,66,.06);">
              <b style="font-size:.78rem;color:{GRAY};display:block;margin-bottom:6px;">
                {sig}
              </b>
              <span style="font-size:.8rem;color:{NAVY};">{status}</span>
            </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 7 — INSIGHTS & CONCLUSIONS
# ============================================================
def page_conclusions():
    st.markdown('<p class="page-title">  Insights & Conclusions</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Feature analysis · Model trade-offs · Strategic recommendations · Team</p>', unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(
        [" Feature Insights"," Model Trade-offs"," Recommendations"])

    # ── TAB 1: Feature Insights ───────────────────────────────
    with tab1:
        insights = [
            ("","Transaction Amount","HIGH",RED,
             "Fraudulent transactions cluster at two extremes: very small amounts (card-testing — "
             "verifying validity before a major purchase) and very large amounts (maximising "
             "extraction before detection). Log-transform captures both tails and makes this "
             "the strongest engineered signal across all three models."),
            ("","Hour of Day","HIGH",RED,
             "Fraud disproportionately occurs between midnight and 3AM — when cardholders "
             "are asleep and least likely to notice unauthorised charges. Temporal features "
             "consistently rank among the top predictors in production fraud detection systems."),
            ("","Geographic Distance","HIGH",RED,
             "Haversine distance between cardholder home and merchant is a powerful proxy for "
             "stolen-card usage. Transactions >100 km from the cardholder's registered address "
             "are flagged with significantly elevated probability by all three models."),
            ("","Merchant Category","MEDIUM",GOLD,
             "Shopping_net, misc_net, and grocery_pos show the highest fraud rates. Online and "
             "miscellaneous categories are preferred by fraudsters due to minimal physical "
             "verification. Category encoding captures these differential risk profiles."),
            ("👤","Cardholder Age","MEDIUM",GOLD,
             "Age correlates moderately with fraud susceptibility. Older cardholders tend to be "
             "more targeted by specific fraud vectors. Combined with temporal and geographic "
             "signals, age adds demographic segmentation to the model's risk assessment."),
            ("","City Population","LOW-MEDIUM",GRAY,
             "Smaller cities show slightly elevated per-capita fraud rates, possibly due to "
             "less sophisticated local fraud prevention infrastructure. The feature contributes "
             "marginally but consistently across all ensemble models tested."),
        ]
        for ic,feat,impact,color,desc in insights:
            st.markdown(f"""
            <div class="model-card" style="margin-bottom:12px;border-top-color:{color};">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:8px;">
                <b style="color:{NAVY};font-size:1rem;">{ic}  {feat}</b>
                <span style="background:{color};color:white;padding:3px 14px;
                             border-radius:14px;font-size:.74rem;font-weight:700;">
                  {impact}
                </span>
              </div>
              <p style="color:{GRAY};font-size:.87rem;line-height:1.6;margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: Trade-offs ─────────────────────────────────────
    with tab2:
        tradeoffs = [
            ("","SVM",NAVY,
             ["Calibrated probabilities","Strong generalisation","Fast inference","Linear scaling O(n)"],
             ["No native feature importance","Scale-sensitive (mitigated here)","Assumes linear separability","Slower training vs RF"],
             "Best as a high-speed pre-screening layer in a two-stage pipeline."),
            ("","Random Forest",GOLD,
             ["Native feature importances","Handles non-linearity","Robust to noise","Interpretable for audit/compliance"],
             ["Memory-intensive (200 trees)","Conservative probability estimates","Slower inference vs SVM","Can overfit deep trees"],
             "Best when explainability to regulators or cardholders is required."),
            ("","XGBoost",RED,
             ["Top AUC on tabular data","Built-in L1/L2 regularisation","Handles scale_pos_weight","Stochastic subsampling"],
             ["More hyperparameters","Risk of overfitting","Slower training per tree","Less intuitive to interpret"],
             "Best as the primary risk engine where maximum performance is the priority."),
        ]
        cols = st.columns(3)
        for col,(ic,nm,cl,pros,cons,rec) in zip(cols,tradeoffs):
            pros_h = "".join([f"<li style='color:{GREEN};padding:3px 0;'> {p}</li>" for p in pros])
            cons_h = "".join([f"<li style='color:{RED};padding:3px 0;'> {c}</li>" for c in cons])
            col.markdown(f"""
            <div class="model-card" style="border-top-color:{cl};">
              <div style="font-size:2rem;">{ic}</div>
              <h4 style="color:{cl};margin:4px 0 12px;">{nm}</h4>
              <b style="color:{NAVY};font-size:.85rem;">Strengths</b>
              <ul style="margin:6px 0 12px 14px;padding:0;">{pros_h}</ul>
              <b style="color:{NAVY};font-size:.85rem;">Limitations</b>
              <ul style="margin:6px 0 12px 14px;padding:0;">{cons_h}</ul>
              <hr style="border-color:{LGRAY};margin:10px 0;">
              <span style="font-size:.82rem;color:{GOLD};font-weight:600;"> {rec}</span>
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: Recommendations ────────────────────────────────
    with tab3:
        recs = [
            ("","Two-Stage Production Pipeline",
             "Deploy XGBoost as the primary risk scorer (AUC-maximised). Transactions scoring "
             ">30% enter a human review queue; >80% are auto-blocked. SVM serves as a high-speed "
             "pre-filter, clearing obvious legitimate transactions before XGBoost processes edge cases. "
             "This architecture balances cost, speed, and accuracy."),
            ("","Optimise the Decision Threshold",
             "The default 50% threshold is not optimal. Shift to ≈30% to dramatically increase "
             "Recall (catching more fraud) at the cost of some Precision (more false alarms). "
             "The business cost of a missed fraud (financial loss + reputational damage) "
             "far exceeds the cost of a false-positive review."),
            ("","Velocity Feature Engineering for Production",
             "Enrich each transaction in real-time with velocity features: # transactions in "
             "last 1hr / 6hr / 24hr per card, per merchant, per IP, per device. These features "
             "typically outperform all static features and are unavailable in batch-trained models. "
             "Implement via Redis or a streaming platform (Kafka + Flink)."),
            ("","Monitor for Model Drift",
             "Fraudsters adapt their tactics monthly. Implement KS-statistic monitoring on key "
             "feature distributions (amount, distance, hour). Set a retraining trigger when "
             "F1-Score drops >5% from the deployment baseline or KS > 0.15 on any feature. "
             "Aim for monthly retraining cycles with fresh labelled data."),
            ("","Explainable AI for Regulatory Compliance",
             "GDPR Article 22 and financial regulations require explainable automated decisions. "
             "Integrate SHAP (SHapley Additive exPlanations) on the Random Forest to generate "
             "per-transaction plain-English explanations: 'Flagged due to unusual hour (2AM), "
             "large amount ($850), and 245 km distance from home.' Critical for customer "
             "communications and regulator audits."),
        ]
        for ic,title,desc in recs:
            st.markdown(f"""
            <div class="model-card" style="margin-bottom:12px;border-top-color:{GOLD};">
              <div style="display:flex;align-items:flex-start;gap:14px;">
                <span style="font-size:1.9rem;line-height:1;">{ic}</span>
                <div>
                  <b style="color:{NAVY};font-size:1rem;">{title}</b>
                  <p style="color:{GRAY};font-size:.87rem;line-height:1.6;
                             margin:6px 0 0;">{desc}</p>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Conclusion banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{NAVY} 0%,{NAVY2} 55%,{GOLD} 140%);
                    color:white;padding:34px 38px;border-radius:18px;margin-top:30px;">
          <h3 style="font-family:'Playfair Display',serif;color:white;margin:0 0 14px;">
              Final Conclusion
          </h3>
          <p style="color:rgba(255,255,255,.88);font-size:.96rem;line-height:1.78;margin:0;">
            FraudShield demonstrates a complete, production-quality machine learning pipeline
            for credit card fraud detection. Rigorous feature engineering (temporal, geographic,
            and demographic signals) combined with SMOTE-corrected class imbalance and a
            three-model ensemble — <b style="color:{GOLD2};">SVM, Random Forest, and XGBoost</b>
            — achieves strong discriminative performance on a severely imbalanced, real-world
            dataset (555,719 transactions, 0.39% fraud). The interactive dashboard bridges the
            gap between statistical modelling and actionable financial intelligence,
            empowering analysts to explore data, benchmark models, and make real-time
            transaction decisions with confidence.
          </p>
        </div>""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR  +  MAIN  NAVIGATION
# ============================================================
def main():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:26px 0 14px;">
          <div style="font-size:3.4rem;line-height:1;">🛡️</div>
          <h2 style="font-family:'Playfair Display',serif;color:white;
                     font-size:1.65rem;margin:10px 0 4px;letter-spacing:-.015em;">
            FraudShield
          </h2>
          <p style="color:rgba(255,255,255,.5);font-size:.75rem;margin:0;letter-spacing:.1em;">
            CREDIT CARD FRAUD DETECTION
          </p>
        </div>
        <hr style="border-color:rgba(255,255,255,.15);margin:12px 0 20px;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["  Home & Dashboard",
             "  Data Explorer",
             "  Preprocessing",
             "  Model Training",
             "  Evaluation",
             "  Fraud Predictor",
             "  Insights & Conclusions"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:16px 0;'>",
                    unsafe_allow_html=True)

        if "models" in st.session_state:
            st.markdown(f"""
            <div class="status-ok">
               <b>Models Trained</b><br>
              <span style="color:rgba(255,255,255,.65);font-size:.76rem;">
              SVM · Random Forest · XGBoost<br>
              Train {st.session_state.get('train_n',0):,} · Test {st.session_state.get('test_n',0):,}
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-warn">
              ⚠  Models not yet trained<br>
              <span style="color:rgba(255,255,255,.65);font-size:.76rem;">
              Visit Model Training to begin
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="position:fixed;bottom:18px;width:210px;
                    font-size:.68rem;color:rgba(255,255,255,.28);text-align:center;">
          Designed By: <br>Samson Awinsone Ayamga
        </div>""", unsafe_allow_html=True)


    # ── Load Dataset ──────────────────────────────────────────
    # LOCAL = "fraudTest.zip"
    # df = None
    #
    # # Initialize session state to track if the demo button has been clicked
    # if "use_demo" not in st.session_state:
    #     st.session_state.use_demo = False
    #
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("**FraudShield Entry Point**")
    #
    # # Check if the file exists in your local folder/repository
    # if os.path.exists(LOCAL):
    #     if st.sidebar.button(" Run Live Demo (Local File)", use_container_width=True):
    #         st.session_state.use_demo = True
    # else:
    #     st.sidebar.error("⚠ Local 'fraudTest.csv' not found.")
    #
    # st.sidebar.markdown("<p style='text-align: center; color: gray; margin: 0;'> OR </p>", unsafe_allow_html=True)
    #
    # # File uploader section
    # st.sidebar.markdown("**Upload Custom Dataset**")
    # up = st.sidebar.file_uploader("Upload fraudTest.csv", type="csv", label_visibility="collapsed")
    #
    # # If they upload a new file, turn off the demo state so the uploaded file takes priority
    # if up:
    #     st.session_state.use_demo = False
    #
    # # ── Data Routing Logic ────────────────────────────────────
    # if st.session_state.use_demo:
    #     # Route 1: User chose the live demo
    #     with st.spinner("Loading live demo from local directory…"):
    #         df = load_data(file_path=LOCAL)
    #         st.sidebar.success(" Running Live Demo!")
    #
    # elif up:
    #     # Route 2: User chose to upload their own file
    #     with st.spinner("Loading uploaded dataset…"):
    #         df = load_data(uploaded=up)
    #         st.sidebar.success(" Uploaded data loaded!")
    #
    # else:
    #     # Route 3: Neither choice has been made yet
    #     st.info(" Choose **Run Live Demo** or upload **fraudTest.csv** via the sidebar to initialize FraudShield.")
    #     st.stop()


    # ── Remote Dataset Configuration ──────────────────────────
    # Replace with your actual converted direct download link
    DATA_URL = DATA_URL = "https://www.dropbox.com/scl/fi/6qr1quasilstd0omje7j1/fraudTest.csv?rlkey=uezahmd85z5pabbe7625s1nqu&st=u641gjzt&dl=1"
    df = None

    # Track if demo data has been requested
    if "use_demo" not in st.session_state:
        st.session_state.use_demo = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("**FraudShield Entry Point**")

    # The Demo button is always available because the file is hosted in the cloud
    if st.sidebar.button("⚡ Run Live Demo (Cloud Stream)", use_container_width=True):
        st.session_state.use_demo = True

    st.sidebar.markdown("<p style='text-align: center; color: gray; margin: 0;'>- OR -</p>", unsafe_allow_html=True)

    st.sidebar.markdown("**📁 Upload Custom Dataset**")
    up = st.sidebar.file_uploader("Upload fraudTest.csv", type=["csv"], label_visibility="collapsed")

    if up:
        st.session_state.use_demo = False

    # ── Data Routing Logic ────────────────────────────────────
    # if st.session_state.use_demo:
    #     with st.spinner("Streaming live demo dataset from cloud storage..."):
    #         try:
    #             # Pandas can read directly from a live web URL!
    #             df = pd.read_csv(DATA_URL)
    #             st.sidebar.success("Live Demo data streamed successfully!")
    #         except Exception as e:
    #             st.sidebar.error("Cloud stream failed. Please check your internet connection or use manual upload.")
    #
    # elif up:
    #     with st.spinner("Loading uploaded dataset…"):
    #         df = load_data(uploaded=up)
    #         st.sidebar.success("Uploaded data loaded!")
    #
    # else:
    #     st.info(" Choose **Run Live Demo** or **upload fraudTest.csv** via the sidebar to initialize FraudShield.")
    #     st.stop()
    if st.session_state.use_demo:
        with st.spinner("Streaming live demo dataset from cloud storage..."):
            try:
                df = pd.read_csv(DATA_URL)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                # ── Validate the download actually worked ──
                if "is_fraud" not in df.columns:
                    st.sidebar.error(
                        "⚠️ Cloud download returned an invalid file (Google scan block). Please upload the CSV manually instead.")
                    st.session_state.use_demo = False
                    st.stop()
                else:
                    st.sidebar.success("Live Demo data streamed successfully!")
            except Exception as e:
                st.sidebar.error(f"Cloud stream failed: {str(e)[:80]}")
                st.session_state.use_demo = False
                st.stop()

    elif up:
        with st.spinner("Loading uploaded dataset…"):
            df = load_data(uploaded=up)
            if df is None or "is_fraud" not in df.columns:
                st.sidebar.error("⚠️ Uploaded file doesn't look like the fraud dataset.")
                st.stop()
            st.sidebar.success("Uploaded data loaded!")

    else:
        st.info("Choose **Run Live Demo** or **upload fraudTest.csv** via the sidebar to initialize FraudShield.")
        st.stop()

    # ── Final guard ───────────────────────────────────────────
    if df is None:
        st.stop()

    # ── Route to selected page ────────────────────────────────
    if   "Home"        in page: page_home(df)
    elif "Data"        in page: page_data_explorer(df)
    elif "Preprocess"  in page: page_preprocessing(df)
    elif "Training"    in page: page_training(df)
    elif "Evaluation"  in page: page_evaluation()
    elif "Predictor"   in page: page_prediction()
    elif "Conclusions" in page: page_conclusions()


if __name__ == "__main__":
    main()
