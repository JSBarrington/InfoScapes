# Infoscapes Studio (Streamlit)

This is a working single-file Streamlit app for **information-theoretic species delimitation** with:
- Dimensionality reduction (PCA, t-SNE, MDS/PCoA, UMAP, etc.)
- Information landscape + basins
- Clustering, consensus, and persistence
- **Correlation Explorer** (feature, per-cluster, Mantel)
- **Cluster Distribution Fits** (Normal/Beta/Gamma/Poisson)

## Run locally

```bash
# (Recommended) using conda
conda env create -f environment.yml
conda activate infoscapes

# Run
streamlit run Infoscapes_demo_8_next.py
```

> If you see a `SyntaxError` on a line with `elif fasta_method`, make sure you're running **this** file (`Infoscapes_demo_8_next.py`) and not an older edited copy.

## Using GitHub + Copilot

1. **Create a new repo** on GitHub (public or private).
2. Initialize locally and push:

```bash
git init
git add .
git commit -m "Initial commit: Infoscapes demo app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

3. **Enable GitHub Copilot** (VS Code > Extensions > GitHub Copilot).  
   - Use **Copilot Chat** to refactor or extract modules (e.g., “extract analyzer into `infoscapes/analyzer.py`”).

## Codespaces (optional)

- Click **Code → Open with Codespaces** on your repo to get a prebuilt dev VM.
- Install conda (`mamba`) and create the env from `environment.yml`.

## Next steps (modularization)

- Convert this single file into a package structure:
```
infoscapes/
  __init__.py
  analyzer.py
  ui.py
  correlations.py
  distributions.py
  utils.py
app.py  # streamlit entrypoint importing from the package
```
- Copilot prompt: *"Create `infoscapes/correlations.py` by moving Correlation Explorer helpers from the current app and export plot functions."*

## Troubleshooting

- If UMAP/PHATE/PaCMAP missing, install them or they’ll be hidden in the UI.
- For FASTA, the app requires BioPython; if not installed, the FASTA option won’t appear.
