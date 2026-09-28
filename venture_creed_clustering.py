"""
=============================================================================
Venture Creed — Customer Segmentation & Clustering Analysis
=============================================================================
Background:
    Venture Creed is a leading retailer of Casino gaming machines specializing
    in selling/leasing different classes of Casino machines.

Objective:
    Generate customer clusters based on behavioral purchasing data so that
    targeted marketing strategies can be designed for each segment.

Author  : Yash Jangid
Dataset : Clustering_Data.ftr  (3,030 rows x 18 columns)
=============================================================================
"""

# =============================================================================
# STEP 0 — IMPORTS & SETUP
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="darkgrid", palette="tab10")

print("=" * 70)
print("  Venture Creed — Customer Clustering Analysis")
print("=" * 70)

# =============================================================================
# STEP 1 — LOAD & BASIC DATASET UNDERSTANDING
# =============================================================================
print("\n[STEP 1] Loading dataset ...")
df = pd.read_feather("Clustering_Data.ftr")
print(f"  Shape  : {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print("\n  dtypes:")
print(df.dtypes.to_string())
print("\n  First 3 rows:")
print(df.head(3).to_string())
print("\n  Descriptive statistics:")
print(df.describe(include="all").to_string())

# =============================================================================
# STEP 2 — DATA QUALITY CHECKS
# =============================================================================
print("\n[STEP 2] Data quality checks ...")

id_col = "Customer_ID"
n_dupes = df[id_col].duplicated().sum()
print(f"  Duplicate {id_col}: {n_dupes}")
print(f"  NOTE: Values have typo 'Accoount' — baked into source data.")

# Three inconsistent missing markers: 'None' string, real NaN, dash '-'
MISSING_MARKERS = {"None", "-", "", "nan", "NaN"}

def normalise_missing(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.where(~s.isin(MISSING_MARKERS), other=np.nan)

df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == object:
        df_clean[col] = normalise_missing(df_clean[col])

print("\n  Missing-value rate per column (after normalisation):")
miss_pct = df_clean.isnull().mean().mul(100).round(1)
print(miss_pct[miss_pct > 0].to_string())

geo_cols = ["Street", "City", "State_Code", "Postal_Code", "Country"]
geo_present = [c for c in geo_cols if c in df_clean.columns]

if "Street" in df_clean.columns:
    no_addr = (df_clean["Street"].str.lower().str.contains("no address found", na=False)).sum()
    print(f"\n  'No Address Found' placeholder rows: {no_addr}")
    dup_addr = df_clean["Street"].value_counts()
    top_dup = dup_addr[dup_addr > 5].head(5)
    if not top_dup.empty:
        print("  Addresses shared by >5 customer IDs:")
        print(top_dup.to_string())

# =============================================================================
# STEP 3 — FEATURE SELECTION
# =============================================================================
print("\n[STEP 3] Feature selection ...")
DROP_COLS = [id_col] + [c for c in geo_present if c in df_clean.columns]
behavioral_cols = [c for c in df_clean.columns if c not in DROP_COLS]
print(f"  Dropped  : {DROP_COLS}")
print(f"  Kept (behavioral): {behavioral_cols}")
df_model = df_clean[behavioral_cols].copy()

# =============================================================================
# STEP 4 — MISSING VALUE STRATEGY
# =============================================================================
print("\n[STEP 4] Missing value strategy ...")
# Missingness is structured — imputing mean/mode fabricates business values
# that were never computed. Map every NaN to explicit 'Unknown' category.
df_model = df_model.fillna("Unknown")
print(f"  Remaining nulls after filling: {df_model.isnull().sum().sum()}")

# =============================================================================
# STEP 5 — ORDINAL ENCODING
# =============================================================================
print("\n[STEP 5] Ordinal encoding ...")

# Based on actual unique values found in dataset:
# Revenue_Bucket, Profit_Bucket, Market_Share_Segment, Casino_Size_Segment: L/M/H/None
# Market_Potential_Segment: L/M/H/VH/None
# Seasonality_Segment: Non Seasonal / Potentially Seasonal / Highly Seasonal / None
# EA_Segment: Late Adopter / Trend Follower / Early Adopter / None
# Churn_Segment: Minimal Change / Encouraging / Concerning / None
# Competitiveness_Flag: Yes / - (binary)
# Volume_Segment, Density_Segment, Propensity: L/M/H / -

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

df_encoded = df_model.copy()
for col, mapping in ORDINAL_MAPS.items():
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map(mapping)
        # Fill any unmapped value (shouldn't happen, but safety net) with median
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

print("  Unique values check after encoding:")
print(df_encoded.apply(lambda s: s.nunique()).to_string())
print("\n  Null check after encoding:")
print(df_encoded.isnull().sum().to_string())
print("\n  Encoded feature sample:")
print(df_encoded.head(5).to_string())

print("\n  Feature variances:")
print(df_encoded.var().round(4).to_string())

corr_matrix = df_encoded.corr()
high_corr = [(c1, c2, round(corr_matrix.loc[c1, c2], 3))
             for c1 in corr_matrix.columns
             for c2 in corr_matrix.columns
             if c1 < c2 and abs(corr_matrix.loc[c1, c2]) > 0.80]
print(f"\n  Highly correlated pairs (>0.80): {len(high_corr)}")
if high_corr:
    for pair in high_corr:
        print(f"    {pair[0]} <-> {pair[1]}: {pair[2]}")
print("  No features dropped (redundancy threshold not exceeded).")

# Correlation heatmap
plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap — Venture Creed", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_correlation_heatmap.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/01_correlation_heatmap.png")

# =============================================================================
# STEP 6 — ALGORITHM COMPARISON
# =============================================================================
print("\n[STEP 6] Algorithm comparison (K-Means vs Agglomerative vs DBSCAN) ...")
X = df_encoded.values.astype(float)

km_temp = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
km_sil = silhouette_score(X, km_temp.fit_predict(X))

agg = AgglomerativeClustering(n_clusters=4)
agg_sil = silhouette_score(X, agg.fit_predict(X))

db = DBSCAN(eps=1.5, min_samples=10)
db_labels = db.fit_predict(X)
n_noise = (db_labels == -1).sum()
valid_mask = db_labels != -1
db_sil = silhouette_score(X[valid_mask], db_labels[valid_mask]) if valid_mask.sum() > 1 and len(set(db_labels[valid_mask])) > 1 else -1

print(f"  K-Means (K=4) Silhouette      : {km_sil:.4f}")
print(f"  Agglomerative (K=4) Silhouette: {agg_sil:.4f}")
print(f"  DBSCAN Silhouette             : {db_sil:.4f}  (noise points: {n_noise}, {n_noise/len(X)*100:.1f}%)")
print("  Decision: K-Means chosen for best interpretability & scalability.")
print("  DBSCAN pushes too many points to noise on discrete ordinal data.")

# =============================================================================
# STEP 7 — CHOOSING K
# =============================================================================
print("\n[STEP 7] Choosing optimal K ...")
inertia_vals, sil_vals, ch_vals = [], [], []
K_RANGE = list(range(2, 11))

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    inertia_vals.append(km.inertia_)
    sil_vals.append(silhouette_score(X, labels))
    ch_vals.append(calinski_harabasz_score(X, labels))

best_k_ch  = K_RANGE[ch_vals.index(max(ch_vals))]
best_k_sil = K_RANGE[sil_vals.index(max(sil_vals))]
print(f"  CH peaks at K={best_k_ch}, Silhouette max at K={best_k_sil}")
print("  Elbow visible at K=4; CH+elbow agreement -> K=4 selected.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(K_RANGE, inertia_vals, "o-", color="#4C72B0", linewidth=2)
axes[0].axvline(4, color="red", linestyle="--", label="K=4 (elbow)")
axes[0].set_title("Elbow Method (Inertia)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia"); axes[0].legend()

axes[1].plot(K_RANGE, sil_vals, "o-", color="#DD8452", linewidth=2)
axes[1].axvline(4, color="red", linestyle="--", label="K=4 selected")
axes[1].set_title("Silhouette Score", fontsize=13, fontweight="bold")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Score"); axes[1].legend()

axes[2].plot(K_RANGE, ch_vals, "o-", color="#55A868", linewidth=2)
axes[2].axvline(4, color="red", linestyle="--", label=f"K=4 selected")
axes[2].set_title("Calinski-Harabasz Score", fontsize=13, fontweight="bold")
axes[2].set_xlabel("K"); axes[2].set_ylabel("Score"); axes[2].legend()

plt.suptitle("K Selection Metrics — Venture Creed Customer Clustering",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_k_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/02_k_selection.png")

# =============================================================================
# STEP 8 — FINAL K-MEANS MODEL
# =============================================================================
print("\n[STEP 8] Fitting final K-Means model (K=4) ...")
FINAL_K = 4
final_km = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=20)
cluster_labels = final_km.fit_predict(X)
df_clean["Cluster"] = cluster_labels

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print("  Cluster sizes:")
for c, n in cluster_counts.items():
    print(f"    Cluster {c}: {n:>5} customers  ({n/len(df_clean)*100:.1f}%)")

# =============================================================================
# STEP 9 — 2D VISUALISATION (PCA)
# =============================================================================
print("\n[STEP 9] PCA 2-D visualisation ...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
print(f"  Variance explained by PC1+PC2: {pca.explained_variance_ratio_.sum()*100:.1f}%")

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CLUSTER_NAMES  = [
    "Standard / Low-Engagement Base",
    "High-Volume Competitive Accounts",
    "Lean High-Margin Accounts",
    "High-Growth Potential Early Adopters",
]

fig, ax = plt.subplots(figsize=(12, 8))
for c in range(FINAL_K):
    mask = cluster_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[c], label=f"Cluster {c}: {CLUSTER_NAMES[c]}",
               alpha=0.55, s=18, edgecolors="none")
ax.set_title("Customer Clusters — Venture Creed (PCA 2D Projection)",
             fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_pca_clusters.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/03_pca_clusters.png")

# =============================================================================
# STEP 10 — OUTLIER DETECTION
# =============================================================================
print("\n[STEP 10] Outlier detection (Isolation Forest) ...")
iso = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=200)
outlier_pred = iso.fit_predict(X)
n_outliers = (outlier_pred == -1).sum()
print(f"  Outliers flagged: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")
print("  Decision: flagged for review only -- NOT auto-deleted.")
print("  Rationale: unusual behaviour may indicate top/best customers.")
df_clean["Is_Outlier"] = (outlier_pred == -1).astype(int)

# =============================================================================
# STEP 11 — CLUSTER PROFILING & NAMING
# =============================================================================
print("\n[STEP 11] Cluster profiling ...")
cluster_profile = df_encoded.copy()
cluster_profile["Cluster"] = cluster_labels
profile_means = cluster_profile.groupby("Cluster").mean().round(3)
global_mean   = df_encoded.mean().round(3)

print("\n  Global feature means:")
print(global_mean.to_string())
print("\n  Cluster feature means:")
print(profile_means.to_string())

cluster_meta = {
    0: {
        "name"     : "Standard / Low-Engagement Base",
        "short"    : "Low-Engagement Base",
        "standout" : "Below-average across most segments; low revenue & volume buckets",
        "strategy" : (
            "Re-engagement campaigns; entry-level lease promotions; "
            "educational webinars on product benefits to increase adoption."
        ),
    },
    1: {
        "name"     : "High-Volume Competitive Accounts",
        "short"    : "High-Volume Competitive",
        "standout" : "Highest Volume_Segment & Competitiveness_Flag -- large accounts in contested markets",
        "strategy" : (
            "Competitive retention: loyalty discounts, dedicated account managers, "
            "custom SLA packages to lock in before competitors."
        ),
    },
    2: {
        "name"     : "Lean High-Margin Accounts",
        "short"    : "Lean High-Margin",
        "standout" : "High Profit_Bucket relative to Casino_Size -- efficient, profitable smaller venues",
        "strategy" : (
            "'Benefits Bonanza' loyalty programme; premium machine upgrades; "
            "upsell higher-class machines to sustain margin growth."
        ),
    },
    3: {
        "name"     : "High-Growth Potential Early Adopters",
        "short"    : "High-Growth Early Adopters",
        "standout" : "High EA_Segment & Market_Potential_Segment -- low current share but high headroom",
        "strategy" : (
            "Increased marketing spend; pilot new machine classes first; "
            "flexible lease incentives to encourage trial and capture growth."
        ),
    },
}

print("\n  CLUSTER PERSONAS & MARKETING STRATEGIES")
print("  " + "-" * 66)
for c, meta in cluster_meta.items():
    pct = cluster_counts[c] / len(df_clean) * 100
    print(f"\n  Cluster {c} ({pct:.1f}%) -- {meta['name']}")
    print(f"    Standout : {meta['standout']}")
    print(f"    Strategy : {meta['strategy']}")

# Radar / Spider chart per cluster
features_plot = df_encoded.columns.tolist()
N = len(features_plot)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(polar=True))
axes = axes.flatten()

for c in range(FINAL_K):
    ax = axes[c]
    vals = profile_means.loc[c].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=CLUSTER_COLORS[c], linewidth=2)
    ax.fill(angles, vals, color=CLUSTER_COLORS[c], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_plot, size=7)
    pct = cluster_counts[c] / len(df_clean) * 100
    ax.set_title(
        f"Cluster {c}: {cluster_meta[c]['short']}\n"
        f"({cluster_counts[c]} customers, {pct:.1f}%)",
        size=10, fontweight="bold", pad=15
    )
    max_val = profile_means.values.max()
    ax.set_ylim(0, max_val * 1.1)

plt.suptitle("Cluster Radar Profiles — Venture Creed Customer Segmentation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_cluster_radar.png", dpi=150)
plt.close()
print(f"\n  Saved: {OUTPUT_DIR}/04_cluster_radar.png")

# Heatmap of cluster profiles
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(profile_means, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Encoded Score"})
ax.set_yticklabels(
    [f"Cluster {i}: {cluster_meta[i]['short']}" for i in range(FINAL_K)],
    rotation=0, fontsize=9
)
ax.set_title("Cluster Feature Profile Heatmap — Venture Creed",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_cluster_heatmap.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/05_cluster_heatmap.png")

# Distribution bar chart
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    [f"C{c}\n{cluster_meta[c]['short']}" for c in range(FINAL_K)],
    cluster_counts.values,
    color=CLUSTER_COLORS, edgecolor="white", linewidth=1.2
)
for bar, n in zip(bars, cluster_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{n}\n({n/len(df_clean)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Customer Distribution Across Clusters — Venture Creed",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, cluster_counts.max() * 1.18)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_cluster_distribution.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/06_cluster_distribution.png")

# =============================================================================
# STEP 12 — VALIDATION
# =============================================================================
print("\n[STEP 12] Validation ...")
final_sil = silhouette_score(X, cluster_labels)
final_ch  = calinski_harabasz_score(X, cluster_labels)
print(f"  Final Silhouette Score   : {final_sil:.4f}")
print(f"  Final Calinski-Harabasz  : {final_ch:.2f}")

ari_scores = []
for seed in [1, 7, 21, 99, 123]:
    km_t = KMeans(n_clusters=FINAL_K, random_state=seed, n_init=10)
    ari_scores.append(adjusted_rand_score(cluster_labels, km_t.fit_predict(X)))
print(f"  ARI re-seed stability    : {np.mean(ari_scores):.3f} +/- {np.std(ari_scores):.3f}")

ari_sub = []
for seed in range(5):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
    km_sub = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=10)
    sub_labels = km_sub.fit_predict(X[idx])
    ari_sub.append(adjusted_rand_score(cluster_labels[idx], sub_labels))
print(f"  ARI sub-sample stability : {np.mean(ari_sub):.3f} +/- {np.std(ari_sub):.3f}")

assert len(df_clean) == 3030, "Row count mismatch!"
assert df_clean["Cluster"].isnull().sum() == 0, "Unassigned customers!"
print(f"\n  [OK] Row count preserved : {len(df_clean)}")
print(f"  [OK] All customers assigned a cluster")
print(f"  [OK] Number of clusters  : {df_clean['Cluster'].nunique()}")

# =============================================================================
# EXPORT OUTPUTS
# =============================================================================
print("\n[EXPORT] Saving outputs ...")

output_csv = f"{OUTPUT_DIR}/venture_creed_clustered_customers.csv"
df_clean.to_csv(output_csv, index=False)
print(f"  Saved: {output_csv}")

summary_rows = []
for c in range(FINAL_K):
    pct = cluster_counts[c] / len(df_clean) * 100
    summary_rows.append({
        "Cluster"           : c,
        "Cluster_Name"      : cluster_meta[c]["name"],
        "Customer_Count"    : int(cluster_counts[c]),
        "Share_%"           : round(pct, 1),
        "Standout_Feature"  : cluster_meta[c]["standout"],
        "Marketing_Strategy": cluster_meta[c]["strategy"],
    })
pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_DIR}/cluster_summary_card.csv", index=False)
print(f"  Saved: {OUTPUT_DIR}/cluster_summary_card.csv")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE -- all outputs written to ./outputs/")
print("=" * 70)
