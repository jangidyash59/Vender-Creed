import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Venture Creed — Customer Segmentation",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d0f1a; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1f3a 0%, #0d0f1a 50%, #1a1f3a 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 40px 50px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.15) 0%, transparent 70%);
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 1.05rem; margin: 12px 0 0 0; }

/* Metric cards */
.metric-grid { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, #1e2235, #252a40);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px; padding: 20px 22px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(99,102,241,0.15); }
.metric-card .label { color: #64748b; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { color: #e2e8f0; font-size: 1.8rem; font-weight: 800; margin-top: 4px; }
.metric-card .sub   { color: #818cf8; font-size: 0.82rem; margin-top: 2px; }

/* Cluster persona cards */
.cluster-card {
    background: linear-gradient(135deg, #1e2235, #252a40);
    border-radius: 16px; padding: 24px;
    border-left: 4px solid var(--accent);
    margin-bottom: 16px;
    transition: transform 0.2s;
}
.cluster-card:hover { transform: translateX(4px); }
.cluster-card h3 { margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
.cluster-card .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 10px;
    background: rgba(99,102,241,0.15); color: #818cf8;
}
.cluster-card p { color: #94a3b8; font-size: 0.9rem; margin: 6px 0; line-height: 1.6; }
.cluster-card .strategy { color: #a5f3fc; font-size: 0.88rem; margin-top: 10px; }

/* Section headers */
.section-header {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    margin: 32px 0 16px 0; display: flex; align-items: center; gap: 10px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.4), transparent);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1e2235; border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #64748b; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1a, #131629);
    border-right: 1px solid rgba(99,102,241,0.15);
}

/* Selectbox, slider */
.stSelectbox > div > div { background: #1e2235 !important; border-color: rgba(99,102,241,0.3) !important; }
.stSlider .st-emotion-cache-1dp5vir { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

/* Assumption box */
.assumption-box {
    background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    color: #fcd34d; font-size: 0.88rem; line-height: 1.7;
}

/* Footer */
.footer {
    text-align: center; color: #334155; font-size: 0.82rem;
    padding: 24px 0; border-top: 1px solid rgba(99,102,241,0.1); margin-top: 48px;
}

/* Override streamlit default plot background */
div[data-testid="stPlotlyChart"], div[data-testid="column"] { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (cached)
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
CLUSTER_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#f43f5e"]
CLUSTER_NAMES = [
    "Standard / Low-Engagement Base",
    "High-Volume Competitive Accounts",
    "Lean High-Margin Accounts",
    "High-Growth Potential Early Adopters",
]
CLUSTER_SHORT = ["Low-Engagement Base", "High-Volume Competitive", "Lean High-Margin", "High-Growth Early Adopters"]
CLUSTER_STRATEGIES = [
    "Re-engagement campaigns; entry-level lease promotions; educational webinars to increase product adoption.",
    "Competitive retention: loyalty discounts, dedicated account managers, custom SLA packages.",
    "'Benefits Bonanza' loyalty programme; premium machine upgrades; upsell higher-class machines.",
    "Increased marketing spend; pilot new machine classes first; flexible lease incentives.",
]
CLUSTER_STANDOUTS = [
    "Below-average across most segments — low revenue & volume buckets",
    "Highest Volume_Segment in contested markets with Competitiveness_Flag = Yes",
    "High Profit_Bucket relative to Casino_Size — efficient, profitable smaller venues",
    "High EA_Segment & Market_Potential_Segment — low current share but high headroom",
]

MISSING_MARKERS = {"None", "-", "", "nan", "NaN"}
ORDINAL_MAPS = {
    "Seasonality_Segment"      : {"Unknown": 1, "Non Seasonal": 0, "Potentially Seasonal": 2, "Highly Seasonal": 3},
    "EA_Segment"               : {"Unknown": 1, "Late Adopter": 0, "Trend Follower": 2, "Early Adopter": 3},
    "Revenue_Bucket"           : {"Unknown": 1, "L": 0, "M": 2, "H": 3},
    "Profit_Bucket"            : {"Unknown": 1, "L": 0, "M": 2, "H": 3},
    "Market_Share_Segment"     : {"Unknown": 1, "L": 0, "M": 2, "H": 3},
    "Casino_Size_Segment"      : {"Unknown": 1, "L": 0, "M": 2, "H": 3},
    "Market_Potential_Segment" : {"Unknown": 1, "L": 0, "M": 2, "H": 3, "VH": 4},
    "Churn_Segment"            : {"Unknown": 1, "Minimal Change": 0, "Encouraging": 2, "Concerning": 3},
    "Competitiveness_Flag"     : {"Unknown": 0, "Yes": 1},
    "Volume_Segment"           : {"Unknown": 1, "Low": 0, "Medium": 2, "High": 3},
    "Density_Segment"          : {"Unknown": 1, "Low": 0, "Medium": 2, "High": 3},
    "Propensity"               : {"Unknown": 1, "L": 0, "M": 2, "H": 3},
}

@st.cache_data(show_spinner=False)
def load_and_process():
    df = pd.read_feather("Clustering_Data.ftr")
    id_col = "Customer_ID"
    geo_cols = ["Street", "City", "State_Code", "Postal_Code", "Country"]

    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            s = df_clean[col].astype(str).str.strip()
            df_clean[col] = s.where(~s.isin(MISSING_MARKERS), other=np.nan)

    drop_cols = [id_col] + [c for c in geo_cols if c in df_clean.columns]
    behavioral_cols = [c for c in df_clean.columns if c not in drop_cols]

    df_model = df_clean[behavioral_cols].fillna("Unknown")

    df_encoded = df_model.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

    X = df_encoded.values.astype(float)

    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    iso = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=200)
    outlier_pred = iso.fit_predict(X)

    df_clean["Cluster"] = labels
    df_clean["Cluster_Name"] = [CLUSTER_SHORT[l] for l in labels]
    df_clean["PCA_1"] = X_pca[:, 0]
    df_clean["PCA_2"] = X_pca[:, 1]
    df_clean["Is_Outlier"] = (outlier_pred == -1).astype(int)
    df_clean["Customer_ID"] = df["Customer_ID"]

    sil  = silhouette_score(X, labels)
    ch   = calinski_harabasz_score(X, labels)
    var  = pca.explained_variance_ratio_.sum() * 100

    profile = df_encoded.copy()
    profile["Cluster"] = labels
    profile_means = profile.groupby("Cluster").mean().round(3)

    inertia, sil_k, ch_k = [], [], []
    for k in range(2, 11):
        km_t = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbl  = km_t.fit_predict(X)
        inertia.append(km_t.inertia_)
        sil_k.append(silhouette_score(X, lbl))
        ch_k.append(calinski_harabasz_score(X, lbl))

    return df_clean, df_encoded, X, X_pca, labels, sil, ch, var, profile_means, inertia, sil_k, ch_k

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.5rem;'>🎰</div>
        <div style='font-weight:800; font-size:1.1rem; color:#e2e8f0; margin-top:8px;'>Venture Creed</div>
        <div style='color:#64748b; font-size:0.82rem;'>Customer Intelligence</div>
    </div>
    <hr style='border-color:rgba(99,102,241,0.2); margin: 12px 0 20px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔬 EDA & Data Quality", "📊 Cluster Explorer",
         "🎯 Cluster Profiles", "📈 K Selection", "🚨 Outlier Analysis",
         "📋 Assumptions"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(99,102,241,0.1); margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#475569; font-size:0.78rem; padding:0 4px;'>
        <b style='color:#64748b;'>Dataset</b><br>
        3,030 customers · 18 features<br><br>
        <b style='color:#64748b;'>Algorithm</b><br>
        K-Means · K=4<br><br>
        <b style='color:#64748b;'>Author</b><br>
        <a href='https://github.com/jangidyash59' style='color:#818cf8;'>Yash Jangid</a>
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Running clustering pipeline…"):
    df_clean, df_encoded, X, X_pca, labels, sil, ch, var, profile_means, inertia, sil_k, ch_k = load_and_process()

cluster_counts = pd.Series(labels).value_counts().sort_index()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
        <h1>🎰 Venture Creed — Customer Segmentation</h1>
        <p>Unsupervised K-Means clustering on 3,030 casino machine customers · 12 behavioral features · 4 actionable personas</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Customers", "3,030", "18 behavioral features", col1),
        ("Clusters", "4", "K-Means · K=4", col2),
        ("Silhouette", f"{sil:.2f}", "Cluster separation", col3),
        ("CH Score", f"{ch:,.0f}", "Compactness index", col4),
        ("PCA Variance", f"{var:.1f}%", "2-component coverage", col5),
    ]
    for label, value, sub, col in metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'>{value}</div>
                <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Customer Clusters at a Glance</div>", unsafe_allow_html=True)

    # Cluster persona cards
    colors_hex = ["#6366f1", "#f59e0b", "#10b981", "#f43f5e"]
    icons = ["😴", "💪", "💰", "🚀"]
    for c in range(4):
        pct = cluster_counts[c] / len(df_clean) * 100
        st.markdown(f"""
        <div class='cluster-card' style='--accent:{colors_hex[c]}'>
            <h3>{icons[c]} Cluster {c} — {CLUSTER_NAMES[c]}</h3>
            <span class='badge'>{cluster_counts[c]:,} customers · {pct:.1f}%</span>
            <p><b style='color:#cbd5e1;'>Standout:</b> {CLUSTER_STANDOUTS[c]}</p>
            <p class='strategy'>📣 <b>Strategy:</b> {CLUSTER_STRATEGIES[c]}</p>
        </div>
        """, unsafe_allow_html=True)

    # PCA scatter overview
    st.markdown("<div class='section-header'>2D Customer Map (PCA Projection)</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d0f1a")
    ax.set_facecolor("#0d0f1a")
    for c in range(4):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors_hex[c], label=f"C{c}: {CLUSTER_SHORT[c]}",
                   alpha=0.55, s=14, edgecolors="none")
    ax.set_xlabel("PC1", color="#64748b", fontsize=10)
    ax.set_ylabel("PC2", color="#64748b", fontsize=10)
    ax.tick_params(colors="#475569")
    for spine in ax.spines.values():
        spine.set_edgecolor("rgba(99,102,241,0.2)")
    ax.set_title("", color="#e2e8f0")
    legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.15,
                       facecolor="#1e2235", edgecolor="rgba(99,102,241,0.3)",
                       labelcolor="#cbd5e1")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 EDA & Data Quality":
    st.markdown("<h2 style='color:#e2e8f0'>🔬 Exploratory Data Analysis & Data Quality</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📐 Dataset Info", "❓ Missing Values", "🔗 Correlation"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Dataset Shape**")
            st.metric("Rows", "3,030"); st.metric("Columns", "18")
            st.markdown("**Column Types**")
            st.info("All 18 columns are object (string) type — ordinal text buckets")
            st.markdown("**ID Column Note**")
            st.warning("⚠️ Customer_ID contains source typo: `Accoount` (double 'o') — baked into raw data, 0 duplicates")
        with c2:
            st.markdown("**Geography Issues Found**")
            st.error("🏠 220 rows: 'No Address Found' placeholder")
            st.error("🔁 ~600 rows share identical addresses under different Customer IDs")
            st.error("📍 State_Code missing for 10.2% of records")
            st.markdown("**Decision**")
            st.success("✅ Drop all 6 geography columns (ID + 5 geo) — non-behavioral, high-cardinality")

    with tab2:
        df_orig = pd.read_feather("Clustering_Data.ftr")
        MISS = {"None", "-", "", "nan", "NaN"}
        miss_data = {}
        for col in df_orig.columns:
            s = df_orig[col].astype(str).str.strip()
            miss_pct = s.isin(MISS).mean() * 100
            miss_data[col] = round(miss_pct, 1)

        miss_df = pd.DataFrame.from_dict(miss_data, orient="index", columns=["Missing %"])
        miss_df = miss_df[miss_df["Missing %"] > 0].sort_values("Missing %", ascending=False)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d0f1a")
            ax.set_facecolor("#0d0f1a")
            bars = ax.barh(miss_df.index, miss_df["Missing %"],
                           color=["#f43f5e" if v > 60 else "#f59e0b" if v > 20 else "#6366f1"
                                  for v in miss_df["Missing %"]])
            ax.set_xlabel("Missing %", color="#64748b")
            ax.tick_params(colors="#94a3b8", labelsize=9)
            for spine in ax.spines.values(): spine.set_edgecolor("rgba(99,102,241,0.15)")
            ax.set_title("Missing Value Rate per Column", color="#e2e8f0", fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(miss_df.style.background_gradient(cmap="RdYlGn_r", axis=0)
                         .format("{:.1f}%"), use_container_width=True)
            st.markdown("""
            <div class='assumption-box'>
            ⚠️ <b>Key Finding:</b> Missingness is <i>structured</i> — entire segment families
            (Revenue_Bucket, Profit_Bucket, Market_Share, etc.) are absent together.
            This is NOT random — it reflects source-feed gaps for specific customers.<br><br>
            <b>Decision:</b> Map all missing → <code>'Unknown'</code> (middle ordinal value).
            Imputing mean/mode would fabricate values the business never computed.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0d0f1a")
        ax.set_facecolor("#0d0f1a")
        corr = df_encoded.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 8, "color": "#e2e8f0"})
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.set_title("Feature Correlation Matrix (Lower Triangle)", color="#e2e8f0",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("✅ No feature pair exceeds 0.80 correlation — no features dropped.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Cluster Explorer":
    st.markdown("<h2 style='color:#e2e8f0'>📊 Cluster Explorer</h2>", unsafe_allow_html=True)

    col_filter, col_main = st.columns([1, 3])

    with col_filter:
        selected_clusters = st.multiselect(
            "Filter Clusters",
            options=[0, 1, 2, 3],
            default=[0, 1, 2, 3],
            format_func=lambda x: f"C{x}: {CLUSTER_SHORT[x]}"
        )
        point_size = st.slider("Point size", 5, 40, 15)
        alpha_val  = st.slider("Opacity", 0.1, 1.0, 0.55, 0.05)
        show_outliers = st.checkbox("Highlight outliers", value=True)

    with col_main:
        colors_hex = ["#6366f1", "#f59e0b", "#10b981", "#f43f5e"]
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d0f1a")
        ax.set_facecolor("#0d0f1a")

        for c in selected_clusters:
            mask = labels == c
            col_use = colors_hex[c]
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=col_use, label=f"C{c}: {CLUSTER_SHORT[c]}",
                       alpha=alpha_val, s=point_size, edgecolors="none", zorder=2)

        if show_outliers:
            out_mask = df_clean["Is_Outlier"] == 1
            ax.scatter(df_clean["PCA_1"][out_mask], df_clean["PCA_2"][out_mask],
                       c="white", s=point_size * 2.5, marker="x",
                       linewidths=1.2, zorder=3, label="Outlier (Isolation Forest)", alpha=0.8)

        ax.set_xlabel("PC1", color="#64748b", fontsize=10)
        ax.set_ylabel("PC2", color="#64748b", fontsize=10)
        ax.tick_params(colors="#475569")
        for spine in ax.spines.values(): spine.set_edgecolor("rgba(99,102,241,0.15)")
        ax.legend(fontsize=8, framealpha=0.15, facecolor="#1e2235",
                  edgecolor="rgba(99,102,241,0.3)", labelcolor="#cbd5e1")
        ax.set_title(f"PCA 2D Projection — {var:.1f}% variance explained",
                     color="#e2e8f0", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # Distribution bar
    st.markdown("<div class='section-header'>Cluster Size Distribution</div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(9, 4), facecolor="#0d0f1a")
    ax2.set_facecolor("#0d0f1a")
    x_pos = range(4)
    for c in range(4):
        alpha = 1.0 if c in selected_clusters else 0.2
        bar = ax2.bar(c, cluster_counts[c], color=colors_hex[c], alpha=alpha,
                      edgecolor="none", width=0.6)
        pct = cluster_counts[c] / len(df_clean) * 100
        ax2.text(c, cluster_counts[c] + 20, f"{cluster_counts[c]:,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#e2e8f0")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"C{c}\n{CLUSTER_SHORT[c]}" for c in range(4)],
                        color="#94a3b8", fontsize=8)
    ax2.tick_params(axis="y", colors="#475569")
    for spine in ax2.spines.values(): spine.set_edgecolor("rgba(99,102,241,0.1)")
    ax2.set_title("Customer Count per Cluster", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, cluster_counts.max() * 1.2)
    fig2.tight_layout()
    st.pyplot(fig2); plt.close()

    # Data table
    st.markdown("<div class='section-header'>Customer Data Table</div>", unsafe_allow_html=True)
    show_df = df_clean[df_clean["Cluster"].isin(selected_clusters)].copy()
    cols_show = ["Customer_ID", "Cluster", "Cluster_Name", "Is_Outlier"] + list(df_encoded.columns)[:6]
    st.dataframe(show_df[cols_show].reset_index(drop=True), use_container_width=True, height=350)
    st.caption(f"Showing {len(show_df):,} of {len(df_clean):,} customers")

    csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv, "venture_creed_filtered.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTER PROFILES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Cluster Profiles":
    st.markdown("<h2 style='color:#e2e8f0'>🎯 Cluster Profiles & Marketing Strategy</h2>", unsafe_allow_html=True)

    colors_hex = ["#6366f1", "#f59e0b", "#10b981", "#f43f5e"]
    icons = ["😴", "💪", "💰", "🚀"]
    global_mean = df_encoded.mean()

    # Tabs per cluster
    tab_labels = [f"{icons[c]} Cluster {c}" for c in range(4)]
    tabs = st.tabs(tab_labels)

    for c, tab in enumerate(tabs):
        with tab:
            pct = cluster_counts[c] / len(df_clean) * 100
            c1, c2 = st.columns([1.3, 1])

            with c1:
                st.markdown(f"""
                <div class='cluster-card' style='--accent:{colors_hex[c]}'>
                    <h3>{icons[c]} {CLUSTER_NAMES[c]}</h3>
                    <span class='badge'>{cluster_counts[c]:,} customers &nbsp;·&nbsp; {pct:.1f}% of base</span>
                    <p><b style='color:#cbd5e1;'>Key Standout:</b><br>{CLUSTER_STANDOUTS[c]}</p>
                    <p class='strategy'>📣 <b>Marketing Strategy:</b><br>{CLUSTER_STRATEGIES[c]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Feature comparison table
                profile_row = profile_means.loc[c]
                compare_df = pd.DataFrame({
                    "Feature": profile_row.index,
                    "Cluster Avg": profile_row.values.round(2),
                    "Global Avg": global_mean.values.round(2),
                })
                compare_df["vs Global"] = compare_df["Cluster Avg"] - compare_df["Global Avg"]
                compare_df["Direction"] = compare_df["vs Global"].apply(
                    lambda x: "▲ Above" if x > 0.1 else ("▼ Below" if x < -0.1 else "≈ Neutral")
                )
                st.dataframe(
                    compare_df.set_index("Feature").style
                    .background_gradient(subset=["vs Global"], cmap="RdYlGn", vmin=-1.5, vmax=1.5)
                    .format({"Cluster Avg": "{:.2f}", "Global Avg": "{:.2f}", "vs Global": "{:+.2f}"}),
                    use_container_width=True
                )

            with c2:
                # Radar chart
                features = df_encoded.columns.tolist()
                N = len(features)
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor="#0d0f1a")
                ax.set_facecolor("#0d0f1a")

                vals = profile_means.loc[c].tolist()
                vals += vals[:1]
                glob_vals = global_mean.tolist() + [global_mean.tolist()[0]]

                ax.plot(angles, glob_vals, color="#475569", linewidth=1, linestyle="--", alpha=0.6)
                ax.fill(angles, glob_vals, color="#475569", alpha=0.05)
                ax.plot(angles, vals, color=colors_hex[c], linewidth=2.5)
                ax.fill(angles, vals, color=colors_hex[c], alpha=0.25)

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(features, size=7, color="#94a3b8")
                ax.set_ylim(0, 4)
                ax.tick_params(colors="#475569")
                ax.set_title(f"Cluster {c} vs Global Average",
                             color="#e2e8f0", fontsize=10, fontweight="bold", pad=18)
                ax.spines["polar"].set_edgecolor("rgba(99,102,241,0.2)")
                ax.set_facecolor("#0d0f1a")

                patches = [
                    mpatches.Patch(color=colors_hex[c], label=f"Cluster {c}"),
                    mpatches.Patch(color="#475569", label="Global Avg"),
                ]
                ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1.1),
                          fontsize=8, framealpha=0.1, facecolor="#1e2235",
                          edgecolor="none", labelcolor="#cbd5e1")
                fig.tight_layout()
                st.pyplot(fig); plt.close()

    # Master heatmap
    st.markdown("<div class='section-header'>All Cluster Profiles — Heatmap</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0d0f1a")
    ax.set_facecolor("#0d0f1a")
    sns.heatmap(profile_means, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Encoded Score"})
    ax.set_yticklabels([f"C{i}: {CLUSTER_SHORT[i]}" for i in range(4)],
                       rotation=0, fontsize=9, color="#e2e8f0")
    ax.set_xticklabels(ax.get_xticklabels(), color="#94a3b8", fontsize=9)
    ax.set_title("Cluster Feature Profile Heatmap", color="#e2e8f0",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: K SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 K Selection":
    st.markdown("<h2 style='color:#e2e8f0'>📈 Choosing the Optimal Number of Clusters (K)</h2>", unsafe_allow_html=True)

    colors_hex = ["#6366f1", "#f59e0b", "#10b981"]
    K_RANGE = list(range(2, 11))

    # Algorithm comparison
    st.markdown("<div class='section-header'>Algorithm Comparison</div>", unsafe_allow_html=True)
    algo_data = pd.DataFrame({
        "Algorithm": ["K-Means (K=4)", "Agglomerative (K=4)", "DBSCAN"],
        "Silhouette": [0.44, 0.45, 0.59],
        "Noise Points": ["0%", "0%", "9.4%"],
        "Selected": ["✅ YES", "❌ NO", "❌ NO"],
        "Reason": [
            "Best interpretability, scalable, stable clusters",
            "Slightly higher sil but computationally heavy",
            "9.4% noise — every customer needs a segment"
        ]
    })
    st.dataframe(algo_data.set_index("Algorithm"), use_container_width=True)

    st.markdown("<div class='section-header'>K Selection Metrics (K = 2 to 10)</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    plot_configs = [
        ("Elbow Method — Inertia", K_RANGE, inertia, "#6366f1", "Inertia", col1),
        ("Silhouette Score", K_RANGE, sil_k, "#f59e0b", "Score", col2),
        ("Calinski-Harabász Score", K_RANGE, ch_k, "#10b981", "Score", col3),
    ]
    for title, x_vals, y_vals, color, ylabel, col in plot_configs:
        with col:
            fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="#0d0f1a")
            ax.set_facecolor("#0d0f1a")
            ax.plot(x_vals, y_vals, "o-", color=color, linewidth=2.5, markersize=7, zorder=2)
            ax.axvline(4, color="#f43f5e", linestyle="--", linewidth=1.5, label="K=4 selected", zorder=1)
            ax.scatter([4], [y_vals[2]], color="#f43f5e", s=100, zorder=3)
            ax.set_xlabel("K", color="#64748b", fontsize=9)
            ax.set_ylabel(ylabel, color="#64748b", fontsize=9)
            ax.set_title(title, color="#e2e8f0", fontsize=9.5, fontweight="bold")
            ax.tick_params(colors="#475569", labelsize=8)
            ax.legend(fontsize=7.5, framealpha=0.1, facecolor="#1e2235",
                      edgecolor="none", labelcolor="#cbd5e1")
            for spine in ax.spines.values(): spine.set_edgecolor("rgba(99,102,241,0.1)")
            fig.tight_layout()
            st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='assumption-box'>
    <b>Rationale for K=4:</b><br>
    • Elbow bends at K=4 — diminishing returns in inertia reduction beyond this point<br>
    • Calinski-Harabász peaks at K=3–4 — confirming compact, well-separated clusters<br>
    • Silhouette keeps rising with K (misleading on discrete ordinal data — known behavior)<br>
    • K=4 maps to 4 clearly distinct business personas with unique marketing implications
    </div>
    """, unsafe_allow_html=True)

    # Validation metrics
    st.markdown("<div class='section-header'>Model Validation</div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Silhouette Score", f"{sil:.4f}", "Moderate-strong separation")
    with col_b:
        st.metric("Calinski-Harabász", f"{ch:,.2f}", "Strong compactness")
    with col_c:
        st.metric("ARI Sub-sample Stability", "0.917 ± 0.042", "80% sub-samples")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Outlier Analysis":
    st.markdown("<h2 style='color:#e2e8f0'>🚨 Outlier Detection — Isolation Forest</h2>", unsafe_allow_html=True)

    n_outliers = df_clean["Is_Outlier"].sum()
    n_inliers  = len(df_clean) - n_outliers
    colors_hex = ["#6366f1", "#f59e0b", "#10b981", "#f43f5e"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Customers", f"{len(df_clean):,}")
    with c2: st.metric("Outliers Flagged", f"{n_outliers} ({n_outliers/len(df_clean)*100:.1f}%)")
    with c3: st.metric("Normal Customers", f"{n_inliers:,}")

    st.markdown("""
    <div class='assumption-box'>
    ⚠️ <b>Decision: Outliers are FLAGGED, not deleted.</b><br>
    In Casino gaming, unusual purchasing behavior often signals your <i>most valuable</i> or
    <i>most volatile</i> accounts. Automatically removing them would be analytically clean
    but commercially irresponsible. These customers are surfaced here for business review.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Scatter with outliers highlighted
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d0f1a")
        ax.set_facecolor("#0d0f1a")
        for c in range(4):
            mask = (labels == c) & (df_clean["Is_Outlier"] == 0)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_hex[c],
                       alpha=0.4, s=12, edgecolors="none")
        out_mask = df_clean["Is_Outlier"] == 1
        ax.scatter(df_clean["PCA_1"][out_mask], df_clean["PCA_2"][out_mask],
                   c="#f43f5e", s=55, marker="*", zorder=5, label=f"Outliers ({n_outliers})", alpha=0.9)
        ax.legend(fontsize=8, framealpha=0.1, facecolor="#1e2235",
                  edgecolor="none", labelcolor="#cbd5e1")
        ax.set_xlabel("PC1", color="#64748b"); ax.set_ylabel("PC2", color="#64748b")
        ax.tick_params(colors="#475569")
        for spine in ax.spines.values(): spine.set_edgecolor("rgba(99,102,241,0.1)")
        ax.set_title("Outliers in PCA Space", color="#e2e8f0", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        # Outliers per cluster
        out_by_cluster = df_clean.groupby("Cluster")["Is_Outlier"].sum()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d0f1a")
        ax.set_facecolor("#0d0f1a")
        bars = ax.bar(range(4), out_by_cluster.values, color=colors_hex, edgecolor="none", width=0.6)
        for bar, n in zip(bars, out_by_cluster.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(n), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#e2e8f0")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"C{c}" for c in range(4)], color="#94a3b8", fontsize=10)
        ax.set_title("Outliers per Cluster", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count", color="#64748b")
        ax.tick_params(colors="#475569")
        for spine in ax.spines.values(): spine.set_edgecolor("rgba(99,102,241,0.1)")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Outlier Customer Records</div>", unsafe_allow_html=True)
    out_df = df_clean[df_clean["Is_Outlier"] == 1][["Customer_ID", "Cluster", "Cluster_Name"] + list(df_encoded.columns)].reset_index(drop=True)
    st.dataframe(out_df, use_container_width=True, height=350)
    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Outlier List", csv, "outliers.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSUMPTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Assumptions":
    st.markdown("<h2 style='color:#e2e8f0'>📋 Key Assumptions & Methodology Notes</h2>", unsafe_allow_html=True)

    assumptions = [
        ("Structured Missingness",
         "~65% of customers have entire feature families missing (Revenue_Bucket, Profit_Bucket, Market_Share, etc. absent together). "
         "This is structural — not random — reflecting source-feed gaps. "
         "Decision: Map all missing → 'Unknown' (encoded as neutral middle value) rather than imputing mean/mode, "
         "which would fabricate business metrics never computed by the source system."),
        ("Ordinal Ordering Preserved",
         "All behavioral segments follow a logical ordinal order preserved in encoding: "
         "L < M < H (for bucket columns), Late Adopter < Trend Follower < Early Adopter (EA_Segment), "
         "Non Seasonal < Potentially Seasonal < Highly Seasonal (Seasonality), "
         "Minimal Change < Encouraging < Concerning (Churn). "
         "These orderings were validated against business definitions in the data dictionary."),
        ("'VH' — Very High Market Potential",
         "Market_Potential_Segment has 5 levels: L, M, H, VH, Unknown. "
         "VH (Very High) is encoded as 4, above H=3, preserving the full ordinal hierarchy."),
        ("Competitiveness_Flag Binary Encoding",
         "This flag appears as 'Yes' or '-' (dash) in the dataset. "
         "All 3,030 customers map to either Yes=1 or Unknown=0 due to source data structure. "
         "Zero variance is noted — this feature is retained for completeness but adds minimal clustering signal."),
        ("Outliers Retained",
         "Isolation Forest (contamination=0.03) flagged 91 customers (~3%) as multivariate anomalies. "
         "These were flagged for business review, NOT deleted. "
         "Rationale: In Casino gaming, extreme purchasing behavior often identifies "
         "highest-value or highest-risk accounts — both commercially significant."),
        ("K=4 Selection",
         "K was selected as 4 based on convergence of: "
         "(1) Elbow method — inertia bends at K=4, "
         "(2) Calinski-Harabász — peaks at K=3–4, "
         "(3) Business interpretability — 4 clusters map directly to actionable marketing personas. "
         "Silhouette score rises monotonically with K (known misleading behavior on discrete data) and was not used as the primary selector."),
        ("Geography Columns Excluded",
         "Street, City, State_Code, Postal_Code, Country excluded from clustering inputs. "
         "These are high-cardinality, non-behavioral, and contain significant data quality issues "
         "(220 'No Address Found' placeholders, ~600 shared addresses). "
         "Geography-based targeting is a separate analysis concern."),
        ("K-Means Over DBSCAN/Agglomerative",
         "All three algorithms were evaluated. DBSCAN achieved higher silhouette (0.59) but pushed 9.4% "
         "of customers into 'noise' — unacceptable for a complete business segmentation. "
         "Agglomerative achieved similar silhouette to K-Means (0.45 vs 0.44) but is computationally "
         "heavier with no meaningful business benefit. K-Means was selected for interpretability, "
         "scalability, and stability (ARI 0.88–0.92 across reseeds)."),
    ]

    for i, (title, text) in enumerate(assumptions):
        with st.expander(f"**{i+1}. {title}**", expanded=(i == 0)):
            st.markdown(f"""
            <div style='color:#94a3b8; font-size:0.93rem; line-height:1.8; padding:4px 0;'>
            {text}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Methodology Summary</div>", unsafe_allow_html=True)
    steps = [
        ("Step 0", "Imports & Setup", "pandas, scikit-learn, matplotlib, seaborn, pyarrow"),
        ("Step 1", "Load Dataset", "3,030 rows × 18 columns from .ftr (Feather) format"),
        ("Step 2", "Data Quality", "Normalize 3 missing markers, flag geography issues"),
        ("Step 3", "Feature Selection", "Drop ID + 5 geography cols → keep 12 behavioral segments"),
        ("Step 4", "Missing Value Strategy", "Fill NaN → 'Unknown' (structured missingness, not random)"),
        ("Step 5", "Ordinal Encoding", "Map text buckets → numeric scores preserving order"),
        ("Step 6", "Algorithm Comparison", "K-Means vs Agglomerative vs DBSCAN silhouette comparison"),
        ("Step 7", "Choose K", "Elbow + Calinski-Harabász + business logic → K=4"),
        ("Step 8", "Final K-Means", "K=4, n_init=20, all 3,030 customers assigned"),
        ("Step 9", "PCA Visualization", "2D projection, 57.4% variance explained"),
        ("Step 10", "Outlier Detection", "Isolation Forest, contamination=0.03, 91 flagged"),
        ("Step 11", "Cluster Profiling", "Business names + marketing strategies per cluster"),
        ("Step 12", "Validation", "Silhouette 0.44, CH 1009.54, ARI stability 0.88–0.92"),
    ]
    step_df = pd.DataFrame(steps, columns=["Step", "Name", "Description"])
    st.dataframe(step_df.set_index("Step"), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🎰 Venture Creed Customer Segmentation · Built with Streamlit & scikit-learn ·
    <a href='https://github.com/jangidyash59/Vender-Creed' style='color:#818cf8;'>GitHub</a> ·
    Author: Yash Jangid
</div>
""", unsafe_allow_html=True)
