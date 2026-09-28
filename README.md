# 🎰 Venture Creed — Customer Segmentation & Clustering Analysis

> **Venture Creed** is a leading retailer of Casino gaming machines specializing in selling/leasing different classes of Casino machines across North America and LATAM.

## 🔗 Live Dashboards

| Tool | Link | Description |
|------|------|-------------|
| 🎛️ **Streamlit** | [venture-creed.streamlit.app](https://venture-creed.streamlit.app/) | Interactive ML app — 7 pages, PCA explorer, radar charts, outlier analysis |
| 📊 **Tableau** | [Tableau Public Dashboard](https://public.tableau.com/views/Venture-Creed-Customer-Segmentation/Dashboard1) | BI dashboard — cluster KPIs, feature heatmap, PCA scatter |

---

## 📌 Project Overview

This project performs **unsupervised customer segmentation** on Venture Creed's 3,030-customer base using behavioral purchasing data. The goal is to generate actionable customer clusters that directly inform targeted marketing strategies — enabling the sales team to prioritize accounts and personalize outreach.

**Problem Type:** Unsupervised Learning — Clustering  
**Dataset:** `Clustering_Data.ftr` — 3,030 rows × 18 columns (behavioral segments at customer level)

---

## 🎯 Business Objectives

- Identify distinct customer personas based on purchasing behavior
- Enable targeted marketing strategy per cluster (e.g., *Benefits Bonanza* for loyal accounts)
- Optimize sales team effort allocation
- Flag high-growth potential accounts for increased marketing investment

---

## 📂 Project Structure

```
Venture-Creed/
│
├── Clustering_Data.ftr          # Raw dataset (feather format)
├── Data_Description.xlsx        # Data dictionary
├── venture_creed_clustering.py  # Main analysis script (Steps 0–12)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
└── outputs/
    ├── 01_correlation_heatmap.png          # Feature correlation matrix
    ├── 02_k_selection.png                  # Elbow / Silhouette / CH plots
    ├── 03_pca_clusters.png                 # 2D PCA cluster visualization
    ├── 04_cluster_radar.png                # Radar profiles per cluster
    ├── 05_cluster_heatmap.png              # Cluster feature heatmap
    ├── 06_cluster_distribution.png         # Customer count per cluster
    ├── venture_creed_clustered_customers.csv  # Full output with cluster labels
    └── cluster_summary_card.csv            # Cluster personas & strategies
```

---

## 🔬 Methodology — 12-Step Pipeline

### Step 0 — Setup
Import pandas, scikit-learn (KMeans, Agglomerative, DBSCAN, PCA, IsolationForest), matplotlib, seaborn.

### Step 1 — Basic Dataset Understanding
Load `.ftr` file → **3,030 rows, 18 columns**. Run `.shape`, `.info()`, `.head()`, `.describe()`. Identify ID, geography, and behavioral segment columns using the data dictionary.

### Step 2 — Data Quality Checks
- **No duplicate Customer_IDs** (source bakes a typo: `Accoount` instead of `Account`)
- **3 inconsistent missing markers**: `"None"` string, real `NaN`, dash `"-"` → all normalized
- **Geography anomalies**: 220 rows with *"No Address Found"* placeholder; ~600+ rows sharing identical addresses under different Customer_IDs

### Step 3 — Feature Selection
Drop `Customer_ID` + 5 geography columns (high-cardinality, non-behavioral).  
**Keep 12 behavioral segments** as clustering inputs.

### Step 4 — Missing Value Strategy
> **Key finding:** Missingness is *structured*, not random — entire segment calculation families are absent for some customers at the source-feed level.

Decision: Map every missing marker to explicit `"Unknown"` category instead of imputing mean/mode (which would fabricate values the business never calculated).

### Step 5 — Ordinal Encoding
All 12 features are ordinal (L/M/H, Low/Medium/High, etc.) → map to numeric scores preserving order. `"Unknown"` gets the neutral middle score. Feature correlation check: no pair exceeds 0.80 threshold → no features dropped.

### Step 6 — Algorithm Comparison

| Algorithm | Silhouette Score | Notes |
|-----------|-----------------|-------|
| **K-Means (K=4)** | **0.44** | ✅ Best interpretability & scalability |
| Agglomerative (K=4) | 0.45 | Computationally heavier, similar result |
| DBSCAN | 0.59* | ❌ Pushes 9.4% to noise — unacceptable |

*DBSCAN silhouette computed only on non-noise points.

**Decision:** K-Means chosen.

### Step 7 — Choosing K
Computed Inertia (elbow), Silhouette, and Calinski-Harabász for K=2 to K=10:
- Elbow bends at **K=4**
- CH score peaks around K=3–4
- **K=4 selected** — interpretable, stable, aligned with business segmentation logic

### Step 8 — Final K-Means Model
Fit K-Means with K=4, 20 initializations (`n_init=20`) for stability.  
All 3,030 customers assigned a cluster. Cluster sizes range 12%–58% — no degenerate clusters.

### Step 9 — 2D Visualization (PCA)
PCA reduces 12 features to 2 principal components (**57.4% variance explained**). Clusters are visibly separated in PCA space.

### Step 10 — Outlier Detection (Isolation Forest)
IsolationForest (`contamination=0.03`) flags **91 customers (3.0%)** as multivariate anomalies.  
**Decision:** Flag for review only — *not auto-deleted* (unusual behavior may indicate top/best customers).

### Step 11 — Cluster Profiling & Business Naming

| Cluster | Name | Size | Key Standout | Marketing Strategy |
|---------|------|------|-------------|-------------------|
| **0** | Standard / Low-Engagement Base | 1,770 (58.4%) | Below-average across most segments | Re-engagement campaigns; entry-level lease promos; educational webinars |
| **1** | High-Volume Competitive Accounts | 455 (15.0%) | Highest Volume_Segment; in contested markets | Loyalty discounts; dedicated account managers; custom SLA packages |
| **2** | Lean High-Margin Accounts | 441 (14.6%) | High Profit relative to Casino_Size | **Benefits Bonanza** loyalty programme; premium machine upsell |
| **3** | High-Growth Potential Early Adopters | 364 (12.0%) | High EA_Segment & Market_Potential | Increased marketing spend; pilot new machines; flexible lease incentives |

### Step 12 — Validation

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Silhouette Score | **0.44** | Moderate-strong cluster separation |
| Calinski-Harabász | **1,009.54** | Strong cluster compactness |
| ARI (re-seed stability) | **0.880 ± 0.094** | Highly stable across 5 random seeds |
| ARI (sub-sample stability) | **0.917 ± 0.042** | Excellent stability on 80% sub-samples |

---

## 📊 Key Visualizations

| Plot | Description |
|------|-------------|
| `01_correlation_heatmap.png` | Feature correlation matrix — confirms no redundant features |
| `02_k_selection.png` | Elbow, Silhouette, CH plots — justifies K=4 |
| `03_pca_clusters.png` | 2D PCA scatter — visual cluster separation |
| `04_cluster_radar.png` | Spider charts — feature profile per cluster |
| `05_cluster_heatmap.png` | Heatmap — mean feature values per cluster |
| `06_cluster_distribution.png` | Bar chart — customer count & % per cluster |

---

## 💡 Key Assumptions

1. **Structured missingness** — Entire segment families (Revenue_Bucket, Profit_Bucket, etc.) are absent for ~65% of customers. This is treated as a meaningful unknown state, not random noise.
2. **Ordinal ordering preserved** — L < M < H, Late Adopter < Trend Follower < Early Adopter — validated against business definitions in the data dictionary.
3. **`VH` (Very High)** in `Market_Potential_Segment` encoded as 4 (above H=3).
4. **Competitiveness_Flag** (`Yes`/`-`) treated as binary (1/0) — all values map to 1 due to source data structure.
5. **Outliers retained** — 3% flagged by Isolation Forest are reviewed, not deleted; extreme behavior often indicates high-value accounts.

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/jangidyash59/Vender-Creed.git
cd Vender-Creed

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python venture_creed_clustering.py

# 4. View outputs in ./outputs/
```

**Requirements:** Python 3.8+

---

## 📈 Business Impact

- **Cluster 2 (Lean High-Margin)** — Prime targets for the **"Benefits Bonanza"** loyalty program
- **Cluster 3 (Early Adopters)** — Highest ROI candidates for increased marketing spend
- **Cluster 1 (High-Volume Competitive)** — Retention risk; need dedicated account manager coverage
- **Cluster 0 (Low-Engagement Base)** — Volume segment for re-activation; cost-efficient digital outreach

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas** — Data manipulation
- **scikit-learn** — KMeans, Agglomerative, DBSCAN, PCA, IsolationForest
- **matplotlib / seaborn** — Visualization
- **pyarrow** — Feather file I/O

---

## 👤 Author

**Yash Jangid**  
Data Analyst | Machine Learning Enthusiast  
[GitHub](https://github.com/jangidyash59)

