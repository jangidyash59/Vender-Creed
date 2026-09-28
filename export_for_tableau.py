"""
Generates a clean, Tableau-ready CSV from the Venture Creed dataset.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42

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

# Human-readable label maps for Tableau
LABEL_MAPS = {
    "Revenue_Bucket"           : {"L": "Low", "M": "Medium", "H": "High"},
    "Profit_Bucket"            : {"L": "Low", "M": "Medium", "H": "High"},
    "Market_Share_Segment"     : {"L": "Low", "M": "Medium", "H": "High"},
    "Casino_Size_Segment"      : {"L": "Low", "M": "Medium", "H": "High"},
    "Market_Potential_Segment" : {"L": "Low", "M": "Medium", "H": "High", "VH": "Very High"},
    "Propensity"               : {"L": "Low", "M": "Medium", "H": "High"},
}

CLUSTER_NAMES = {
    0: "Standard / Low-Engagement Base",
    1: "High-Volume Competitive Accounts",
    2: "Lean High-Margin Accounts",
    3: "High-Growth Potential Early Adopters",
}

# ── Load & normalize missing ──────────────────────────────────────────────────
df = pd.read_feather("Clustering_Data.ftr")
geo_cols = ["Street", "City", "State_Code", "Postal_Code", "Country"]
behavioral_cols = [c for c in df.columns if c not in ["Customer_ID"] + geo_cols]

df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == object:
        s = df_clean[col].astype(str).str.strip()
        df_clean[col] = s.where(~s.isin(MISSING_MARKERS), other=np.nan)

# ── Readable labels layer ──────────────────────────────────────────────────────
df_labels = df_clean[behavioral_cols].copy()
# Fill NaN with "Unknown" for display
for col in behavioral_cols:
    df_labels[col] = df_labels[col].fillna("Unknown")
# Apply label maps
for col, lmap in LABEL_MAPS.items():
    if col in df_labels.columns:
        df_labels[col] = df_labels[col].map(lmap).fillna(df_labels[col])

# ── Encoded numeric layer ─────────────────────────────────────────────────────
df_model = df_clean[behavioral_cols].fillna("Unknown").copy()
df_encoded = df_model.copy()
for col, mapping in ORDINAL_MAPS.items():
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map(mapping)
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

X = df_encoded.values.astype(float)

# ── Models ────────────────────────────────────────────────────────────────────
km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
cluster_labels = km.fit_predict(X)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
v1 = round(pca.explained_variance_ratio_[0]*100, 1)
v2 = round(pca.explained_variance_ratio_[1]*100, 1)

iso = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=200)
outlier_flags = iso.fit_predict(X)

# ── Build final dataframe ─────────────────────────────────────────────────────
final = pd.DataFrame()

# Core identifiers & cluster
final["Customer_ID"]   = df["Customer_ID"].values
final["Cluster_ID"]    = cluster_labels
final["Cluster_Name"]  = [CLUSTER_NAMES[c] for c in cluster_labels]
final["Is_Outlier"]    = (outlier_flags == -1).astype(int)
final["Outlier_Label"] = final["Is_Outlier"].map({0: "Normal", 1: "Outlier"})

# PCA scatter coordinates
final[f"PC1_{v1}pct"] = X_pca[:, 0].round(4)
final[f"PC2_{v2}pct"] = X_pca[:, 1].round(4)

# Numeric scores (for heatmap / aggregation)
for col in df_encoded.columns:
    final[f"{col}_Score"] = df_encoded[col].round(2).values

# Readable text labels (for filters, tooltips, crosstabs)
for col in behavioral_cols:
    final[col] = df_labels[col].values

# Verify no NaN in label columns
null_counts = final[behavioral_cols].isnull().sum()
if null_counts.sum() > 0:
    print("WARNING: NaN in label columns:", null_counts[null_counts > 0])
else:
    print("All label columns: 0 nulls")

# ── Save ──────────────────────────────────────────────────────────────────────
import os; os.makedirs("outputs", exist_ok=True)
out = "outputs/venture_creed_tableau_ready.csv"
final.to_csv(out, index=False)
print(f"Saved: {out}  |  Shape: {final.shape}")
print(f"\nCluster distribution:")
print(final["Cluster_Name"].value_counts().to_string())
print(f"\nColumns ({len(final.columns)}):")
for c in final.columns: print(f"  {c}")
