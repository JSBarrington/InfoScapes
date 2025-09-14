[README.md](https://github.com/user-attachments/files/22317639/README.md)
# InfoScapes Studio — Streamlit Demo App

A working single-file Streamlit app for **information-theoretic species delimitation** and exploratory clustering.  
It bundles **dimensionality reduction**, **information landscapes**, **consensus/persistence**, a **Correlation Explorer**, and **Cluster Distribution Fits**.

> **Entry point:** `Infoscapes_demo_8_next.py`

---

## ✨ Feature Highlights

- **Load data** from CSV (features or square distance matrix) or FASTA (alignment-free k-mer, alignment identity, alignment Hamming).
- **Embeddings**: PCA, t-SNE, MDS/PCoA, Laplacian Eigenmaps, plus optional UMAP/PaCMAP/PHATE/Diffusion Map (auto-detected if installed).
- **Landscapes & Basins**: 2D density landscape with wall projections + interactive basin threshold.
- **Clustering & Coherence/Separation**: KMeans/DBSCAN; silhouette & Calinski–Harabasz; pairwise JSD; barrier height (ΔU) on the landscape.
- **Consensus & Persistence**: co-association matrix and variation-of-information (VI) summaries; batch “Automated Persistence Run” across many DR methods.
- **📈 Correlation Explorer**:
  - Feature↔feature correlation heatmaps (Pearson/Spearman/Kendall), optional Ledoit–Wolf shrinkage and hierarchical ordering
  - Per-cluster correlation heatmaps
  - **Mantel test** between your distance matrix and **1 − co-association** (from consensus)
- **📊 Cluster Distribution Fits**:
  - Fit Normal / Beta / Gamma / Poisson to 1-D variables per cluster, rank by AIC/BIC, and overlay PDFs/PMFs
  - Variables supported: radial distance in embedding, neighbor counts (radius), or a chosen CSV feature
- **Exports**: partitions table, co-association matrix, correlation matrices.

---

## 🚀 Quickstart

### 1) Create the environment (recommended: Conda/Mamba)

```bash
conda env create -f environment.yml
conda activate infoscapes
```

> If you don’t have Conda, install Miniconda or Mambaforge first.

### 2) Run the app

```bash
streamlit run Infoscapes_demo_8_next.py
```

The browser will open at `http://localhost:8501` (or similar).

### 3) (Optional) Install extras

Some methods are **optional** and will only appear in the UI if installed:

```bash
# Optional libraries
pip install umap-learn pacmap phate graphtools scprep hdbscan biopython
```

- **PHATE** requires `graphtools` and often `scprep`.
- **FASTA** options require **Biopython**.

---

## 📂 Expected Inputs

### A) CSV — Feature matrix (recommended to start)

- Rows: samples; Columns: numeric features (choose in UI).  
- The app stores **feature names** so the Distribution Fits can target a specific column.

### B) CSV — Square distance/similarity matrix

- Must be **square** (`n × n`), symmetric, and numeric.
- Works with MDS/PCoA; Correlation Explorer switches to Mantel mode.

### C) FASTA

- Three modes:
  - **Alignment-Free (k-mer)**
  - **Alignment-Based (Identity)** — via BioPython distance calculator
  - **Alignment-Based (Hamming)** — on trimmed/paired sites  
- In all FASTA modes, you’ll get a **distance matrix** (so feature names are not available).

---

## 🧭 UI Walkthrough

1. **Load Data** in the sidebar (CSV or FASTA).  
2. **Choose DR Method** and **Clusterer**; click **Run**.
3. **🗻 Landscape** tab: inspect density landscape, basins, and sample highlights.
4. **🧬 Coherence** & **↔️ Separation** tabs: check silhouette/CH, JSD, and ΔU barrier height.
5. **🔄 Persistence** tab: after “Consensus” or “Automated Persistence Run”, view co-association heatmap and downloads.
6. **📈 Correlations** tab:
   - **Features**: correlation heatmap of numeric columns (with shrinkage + clustering options)
   - **Cluster-wise**: per-cluster correlation heatmaps
   - **Mantel**: if a distance matrix is loaded and a co-association matrix exists
7. **📊 Distributions** tab: pick a variable; inspect best-fit distributions per cluster with overlays and a sortable summary.

---

## 🤖 Automated Persistence Run

In the sidebar, open **“Step 3: Automated Persistence Run”**:

- Select any/all available DR methods (PCA, t-SNE, MDS/PCoA, Laplacian Eigenmaps; UMAP/PaCMAP/PHATE/Diffusion Map appear if installed).
- Choose **KMeans** (K) or **DBSCAN** (eps auto-derived from embedding percentile).
- Run to compute **partitions per method**, build a **co-clustering matrix**, and get a **consensus** via agglomerative clustering.

The Persistence UI includes:
- Sample ordering by **Consensus** or **Hierarchical (co-assoc)** to reveal block structure.
- Optional **upper-triangle mask** and **stable-pair (≥ threshold)** contour overlay.
- **CSV exports** for co-clustering matrix and partitions.

---

## 🧩 Repository Tips (Git, Copilot, Codespaces)

1. **Initialize and push to GitHub**

```bash
git init
git add .
git commit -m "Initial commit: infoscapes app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

2. **Use GitHub Copilot** in VS Code to modularize:
   - Create `infoscapes/` package with `analyzer.py`, `correlations.py`, `distributions.py`, and `ui.py`
   - Update `app.py` to import from those modules.

3. **Codespaces** (optional): open the repo in Codespaces and create the environment from `environment.yml`.

---

## ✅ Tests & CI

- A minimal **smoke test** and a GitHub Actions **CI** workflow are included (if you used the provided repo zip).  
- Extend tests to cover data-loading and basic embedding/cluster flows as you modularize.

---

## 🛠️ Troubleshooting

- **`SyntaxError` near `elif fasta_method`**: Ensure you’re running the fixed file: `Infoscapes_demo_8_next.py`.
- **Missing methods in the UI**: The app hides methods whose libraries aren’t installed (e.g., UMAP/PHATE).
- **FASTA not available**: Install `biopython`.
- **Mantel disabled**: You need both (1) a **square distance matrix** and (2) a **co-association** matrix (run Consensus).

If you hit a specific traceback, paste it into an issue (or here) and include:
- OS + Python version, 
- a small data snippet (10–20 rows), and
- the exact steps leading to the error.

---

## 📜 License

No license selected yet. Consider adding `MIT` or `Apache-2.0` once you’re ready to share.

---

## 🙌 Acknowledgements

- scikit-learn, NumPy, SciPy, Plotly, Streamlit
- (Optional) UMAP-learn, PaCMAP, PHATE, graphtools, scprep
