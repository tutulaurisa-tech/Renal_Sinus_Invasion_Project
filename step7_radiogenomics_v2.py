# -*- coding: utf-8 -*-
"""
V2 Pipeline - Step 7: Radiogenomics Analysis (v2 Stable Features)
=================================================================
Links 29 bootstrap-stable radiomics features to transcriptomic biology
using matched TCGA/CPTAC3 cohort (n=170, RSI+=32, RSI-=138).

Three layers of analysis:
  Layer A — DEA:    RSI+ vs RSI- differential expression → volcano plot
  Layer B — GSEA:   Reload valid existing Hallmark/KEGG results → replot bubble
  Layer C — Radiogenomics:
      C1: ssGSEA-lite  — 50 Hallmark pathway activity scores per sample
      C2: Feature-pathway correlation — 29 features × 50 pathways Spearman heatmap
      C3: Sphericity focus — scatter vs EMT / IFN-γ pathway scores
      C4: Full genome correlation — 29 features × 17,407 genes, top hits heatmap

Outputs  →  D:\\RSI_Project_Workspace\\6_Radiogenomics\\v2\\
  volcano_DEA_v2.png
  GSEA_hallmark_bubble_v2.png
  feature_pathway_heatmap_v2.png
  sphericity_pathway_scatter_v2.png
  radiogenomics_heatmap_top_v2.png
  feature_pathway_corr_v2.csv
  radiogenomics_full_corr_v2.csv      (29 × 17407 Spearman rho)

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step7_radiogenomics_v2.py

No extra packages needed (uses only numpy, scipy, pandas, matplotlib).
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR = BASE / "3_Extracted_Features"
MOD_DIR  = BASE / "5_Modeling" / "v2_stability"
GEN_DIR  = BASE / "1_Raw_Data" / "Center4_TCGA" / "Genomics"
OLD_GEN  = BASE / "6_Radiogenomics"
OUT_DIR  = BASE / "6_Radiogenomics" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GMT_HALLMARK = OLD_GEN / "GSEA" / "MSigDB_Hallmark_2020" / "gene_sets.gmt"
GMT_KEGG     = OLD_GEN / "GSEA" / "KEGG_2021_Human"      / "gene_sets.gmt"

SEED = 42
np.random.seed(SEED)

print("=" * 65)
print("V2 Pipeline - Step 7: Radiogenomics Analysis")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    idx = np.argsort(pvals)
    fdr = np.ones(n)
    fdr[idx] = pvals[idx] * n / np.arange(1, n + 1)
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        fdr[idx[i]] = min(fdr[idx[i]], fdr[idx[i + 1]])
    return np.clip(fdr, 0, 1)


def spearman_matrix(F, X):
    """
    Vectorised Spearman correlation.
    F : (n_samples, n_features)   — radiomics features
    X : (n_samples, n_genes)      — gene expression
    Returns corr (n_features, n_genes), pval (n_features, n_genes)
    """
    from scipy.stats import rankdata
    n = F.shape[0]
    # rank-transform each column
    Fr = np.apply_along_axis(rankdata, 0, F).astype(float)
    Xr = np.apply_along_axis(rankdata, 0, X).astype(float)
    # standardise
    Fr -= Fr.mean(0); Fr /= (Fr.std(0) + 1e-12)
    Xr -= Xr.mean(0); Xr /= (Xr.std(0) + 1e-12)
    corr = (Fr.T @ Xr) / (n - 1)          # (n_features, n_genes)
    corr = np.clip(corr, -1, 1)
    # t-statistic → p-value
    with np.errstate(divide="ignore", invalid="ignore"):
        t = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-14))
    pval = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return corr, pval


def parse_gmt(path):
    gmt = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                gmt[parts[0]] = parts[2:]
    return gmt


def ssgsea_lite(expr_df, gmt, gene_cols):
    """
    Simplified pathway scoring (mean z-score of pathway genes per sample).
    Returns DataFrame: samples × pathways.
    """
    # z-score genes across samples
    X = expr_df[gene_cols].values.astype(float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-10)
    scores = {}
    for name, genes in gmt.items():
        present = [g for g in genes if g in gene_cols]
        if len(present) < 10:
            continue
        idx = [gene_cols.index(g) for g in present]
        scores[name] = Xz[:, idx].mean(axis=1)
    return pd.DataFrame(scores, index=expr_df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading matched expression + radiomics data...")
expr_df  = pd.read_csv(GEN_DIR / "All_TCGA_Gene_Expression_withLabel.csv")
gene_cols = [c for c in expr_df.columns if c not in ["PatientID", "RSI", "Cohort"]]
y        = expr_df["RSI"].values.astype(int)
n        = len(y)
print(f"    Expression: {expr_df.shape[0]} samples × {len(gene_cols)} genes")
print(f"    RSI+: {y.sum()}   RSI-: {(y==0).sum()}")

# Load 29 stable radiomics features (TCGA cohort, same order as expression)
all_df   = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
tcga_df  = all_df[all_df.Center == "TCGA"].reset_index(drop=True)
sel_df   = pd.read_csv(MOD_DIR / "selected_features_stability.csv")
feat_cols = sel_df["Feature"].tolist()

scaler_df = pd.read_csv(MOD_DIR / "scaler_params_stability.csv")
scaler_df = scaler_df.set_index("Feature").loc[feat_cols].reset_index()
mu = scaler_df["mean"].values
sd = scaler_df["std"].values

X_raw    = tcga_df[feat_cols].values.astype(float)
X_scaled = (X_raw - mu) / sd          # standardised features
print(f"    Radiomics: {X_scaled.shape[0]} cases × {len(feat_cols)} features")

# Short display names
def short_name(feat, max_len=38):
    parts = feat.split("_")
    if feat.startswith("wavelet-"):
        s = parts[0].replace("wavelet-", "w") + "_" + "_".join(parts[1:])
    else:
        s = feat
    return s[:max_len] + ("…" if len(s) > max_len else "")

feat_display = [short_name(f) for f in feat_cols]

# Parse GMT files
gmt_h = parse_gmt(GMT_HALLMARK)
gmt_k = parse_gmt(GMT_KEGG)
print(f"    Hallmark: {len(gmt_h)} gene sets   KEGG: {len(gmt_k)} gene sets")


# ─────────────────────────────────────────────────────────────────────────────
# Layer A: DEA  —  RSI+ vs RSI-  (Welch t-test, BH FDR)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[A] Differential Expression Analysis (RSI+ vs RSI-)...")

X_expr = expr_df[gene_cols].values.astype(float)
pos_mask = y == 1
neg_mask = y == 0

t_stats, pvals_dea = stats.ttest_ind(
    X_expr[pos_mask], X_expr[neg_mask],
    axis=0, equal_var=False
)
log2fc = (X_expr[pos_mask].mean(0) - X_expr[neg_mask].mean(0)) / np.log(2)
fdr_dea = bh_fdr(pvals_dea)

dea_df = pd.DataFrame({
    "Gene": gene_cols,
    "log2FC": log2fc,
    "t_stat": t_stats,
    "pval": pvals_dea,
    "FDR": fdr_dea,
    "neglog10FDR": -np.log10(fdr_dea + 1e-300),
}).sort_values("FDR")

n_sig05 = (dea_df.FDR < 0.05).sum()
n_sig25 = (dea_df.FDR < 0.25).sum()
print(f"    Genes FDR<0.05: {n_sig05}   FDR<0.25: {n_sig25}")
print(f"    Top upregulated (RSI+):")
print(dea_df[dea_df.log2FC > 0].head(5)[["Gene","log2FC","pval","FDR"]].to_string(index=False))
print(f"    Top downregulated (RSI+):")
print(dea_df[dea_df.log2FC < 0].head(5)[["Gene","log2FC","pval","FDR"]].to_string(index=False))

# Volcano plot
fig, ax = plt.subplots(figsize=(8, 6))
lfc   = dea_df.log2FC.values
nlFDR = dea_df.neglog10FDR.values
fdr   = dea_df.FDR.values
gene  = dea_df.Gene.values

# colour by significance + direction
colors_v = np.where(
    (fdr < 0.05) & (lfc > 0), "#E91E63",
    np.where(
        (fdr < 0.05) & (lfc < 0), "#2196F3",
        np.where(fdr < 0.25, "#FF9800", "lightgray")
    )
)
sizes_v = np.where(fdr < 0.05, 40, np.where(fdr < 0.25, 20, 8))
ax.scatter(lfc, nlFDR, c=colors_v, s=sizes_v, alpha=0.7, linewidths=0)
ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

# label top genes
top_label = dea_df.head(12)
for _, row in top_label.iterrows():
    ax.annotate(row.Gene, (row.log2FC, row.neglog10FDR),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

# legend
from matplotlib.patches import Patch
legend_el = [
    Patch(fc="#E91E63", label=f"Up RSI+ (FDR<0.05, n={((fdr<0.05)&(lfc>0)).sum()})"),
    Patch(fc="#2196F3", label=f"Down RSI+ (FDR<0.05, n={((fdr<0.05)&(lfc<0)).sum()})"),
    Patch(fc="#FF9800", label=f"Suggestive (FDR<0.25, n={((fdr<0.25)&(fdr>=0.05)).sum()})"),
    Patch(fc="lightgray", label="Not significant"),
]
ax.legend(handles=legend_el, fontsize=8, loc="upper left")
ax.set_xlabel("log₂ Fold Change (RSI+ / RSI−)", fontsize=11)
ax.set_ylabel("−log₁₀(FDR)", fontsize=11)
ax.set_title(
    "Differential Gene Expression: RSI+ vs RSI−\n"
    f"TCGA/CPTAC3 cohort (n={n}, RSI+={y.sum()}, RSI−={(y==0).sum()})",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "volcano_DEA_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: volcano_DEA_v2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Layer B: GSEA bubble plot  (replot from existing valid results)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[B] Replotting GSEA Hallmark bubble...")
gsea_h = pd.read_csv(OLD_GEN / "GSEA" / "GSEA_Hallmark_results.csv")

# Keep FDR < 0.25 (standard GSEA threshold)
sig_h = gsea_h[gsea_h["FDR q-val"] < 0.25].copy()
sig_h = sig_h.sort_values("NES", ascending=True)  # horizontal order

# Count lead genes
sig_h["n_lead"] = sig_h["Lead_genes"].apply(
    lambda x: len(str(x).split(";")) if pd.notna(x) else 0
)
sig_h["Gene_frac"] = sig_h["Gene %"].str.rstrip("%").astype(float)

fig, ax = plt.subplots(figsize=(9, max(5, len(sig_h) * 0.42)))
scatter = ax.scatter(
    sig_h["NES"],
    range(len(sig_h)),
    c=-np.log10(sig_h["FDR q-val"] + 1e-6),
    s=sig_h["Gene_frac"] * 8,
    cmap="RdYlBu_r",
    alpha=0.9,
    edgecolors="gray", linewidths=0.4,
    vmin=0, vmax=6,
)
ax.set_yticks(range(len(sig_h)))
ax.set_yticklabels(sig_h["Term"], fontsize=9)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=11)
ax.set_title(
    "GSEA Hallmark — RSI+ vs RSI− (FDR < 0.25)\n"
    "Bubble size ∝ gene fraction; color = −log₁₀(FDR)",
    fontsize=10, fontweight="bold"
)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label("−log₁₀(FDR)", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "GSEA_hallmark_bubble_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: GSEA_hallmark_bubble_v2.png  ({len(sig_h)} pathways FDR<0.25)")
print("    Top 5 Hallmark pathways (RSI+ enriched):")
for _, row in sig_h.sort_values("NES", ascending=False).head(5).iterrows():
    print(f"      {row.Term:<40}  NES={row.NES:.2f}  FDR={row['FDR q-val']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer C1: ssGSEA-lite  —  pathway activity scores (170 × 50)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[C1] Computing ssGSEA-lite pathway scores (Hallmark, 50 pathways)...")
pathway_scores = ssgsea_lite(expr_df, gmt_h, gene_cols)
print(f"     Pathway score matrix: {pathway_scores.shape}")
print(f"     Pathways computed: {list(pathway_scores.columns[:5])} ...")


# ─────────────────────────────────────────────────────────────────────────────
# Layer C2: Feature-pathway Spearman correlation (29 × 50)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[C2] Computing feature-pathway Spearman correlations...")
P = pathway_scores.values              # (170, 50)
corr_fp, pval_fp = spearman_matrix(X_scaled, P)   # (29, 50)
fdr_fp = np.apply_along_axis(bh_fdr, 1, pval_fp)  # per-feature FDR
corr_fp_df = pd.DataFrame(corr_fp, index=feat_display,
                            columns=pathway_scores.columns)

# Identify pathways with at least one feature-correlation FDR < 0.05
sig_path_mask = (fdr_fp < 0.05).any(axis=0)
print(f"     Pathways with ≥1 feature FDR<0.05: {sig_path_mask.sum()}")

# Heatmap: all 29 features × selected pathways
# Select pathways: significant ones + GSEA-significant ones
gsea_sig_terms = set(sig_h["Term"].tolist())
path_names = pathway_scores.columns.tolist()
selected_paths = [p for p in path_names
                  if sig_path_mask[path_names.index(p)]
                  or p in gsea_sig_terms]
# Deduplicate and keep up to 20
if len(selected_paths) > 20:
    # Prioritise GSEA significant ones
    gsea_paths = [p for p in selected_paths if p in gsea_sig_terms]
    feat_paths  = [p for p in selected_paths if p not in gsea_sig_terms]
    selected_paths = (gsea_paths + feat_paths)[:20]
print(f"     Pathways shown in heatmap: {len(selected_paths)}")

heat_corr = corr_fp_df[selected_paths].values  # (29, n_sel)
heat_fdr  = fdr_fp[:, [path_names.index(p) for p in selected_paths]]

# Cluster features by hierarchical linkage (simple: sort by mean |corr|)
row_order = np.argsort(np.abs(heat_corr).mean(1))[::-1]

fig, ax = plt.subplots(figsize=(max(12, len(selected_paths) * 0.6),
                                 max(8, len(feat_cols) * 0.35)))
cmap = plt.get_cmap("RdBu_r")
im = ax.imshow(heat_corr[row_order], cmap=cmap,
               aspect="auto", vmin=-0.5, vmax=0.5)

# Asterisks for significant cells
for ri, r in enumerate(row_order):
    for ci in range(len(selected_paths)):
        fv = heat_fdr[r, ci]
        stars = "***" if fv < 0.001 else ("**" if fv < 0.01 else ("*" if fv < 0.05 else ""))
        if stars:
            ax.text(ci, ri, stars, ha="center", va="center",
                    fontsize=6, color="black" if abs(heat_corr[r, ci]) < 0.35 else "white")

ax.set_xticks(range(len(selected_paths)))
ax.set_xticklabels(selected_paths, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(feat_cols)))
ax.set_yticklabels([feat_display[i] for i in row_order], fontsize=7.5)

cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label("Spearman ρ (feature ↔ pathway)", fontsize=9)
ax.set_title(
    "Radiomics Feature — Hallmark Pathway Correlation\n"
    "(v2 Stability Features × ssGSEA-lite scores, TCGA/CPTAC3 n=170)\n"
    "* FDR<0.05   ** FDR<0.01   *** FDR<0.001",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_pathway_heatmap_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: feature_pathway_heatmap_v2.png")

# Save correlation table
corr_fp_df.to_csv(OUT_DIR / "feature_pathway_corr_v2.csv", encoding="utf-8-sig")
print("     Saved: feature_pathway_corr_v2.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Layer C3: Sphericity focus — scatter vs EMT and IFN-γ scores
# ─────────────────────────────────────────────────────────────────────────────
print("\n[C3] Sphericity vs key pathway scores...")
sph_col   = "original_shape_Sphericity"
sph_idx   = feat_cols.index(sph_col)
sph_raw   = X_raw[:, sph_idx]     # unscaled (more interpretable on x-axis)

# Key pathways to highlight
focus_pathways = [
    "Epithelial Mesenchymal Transition",
    "Interferon Gamma Response",
    "G2-M Checkpoint",
    "Inflammatory Response",
]
focus_pathways = [p for p in focus_pathways if p in pathway_scores.columns]
n_focus = len(focus_pathways)

RSI_POS = "#E91E63"
RSI_NEG = "#2196F3"
rsi_colors = np.where(y == 1, RSI_POS, RSI_NEG)

fig, axes = plt.subplots(1, n_focus, figsize=(5.5 * n_focus, 5))
if n_focus == 1:
    axes = [axes]

for ax, path in zip(axes, focus_pathways):
    ps = pathway_scores[path].values
    rho, pv = stats.spearmanr(sph_raw, ps)

    # scatter colored by RSI
    for label, mask, c in [(0, y==0, RSI_NEG), (1, y==1, RSI_POS)]:
        ax.scatter(sph_raw[mask], ps[mask], c=c, s=22, alpha=0.65,
                   label=f"RSI{'+'if label else '−'} (n={mask.sum()})",
                   edgecolors="none")

    # regression line
    m, b, *_ = stats.linregress(sph_raw, ps)
    xs = np.linspace(sph_raw.min(), sph_raw.max(), 100)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    # stat annotation
    sig_str = f"ρ={rho:.3f}\np={'<0.001' if pv < 0.001 else f'{pv:.3f}'}"
    ax.text(0.97, 0.97, sig_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    ax.set_xlabel("Sphericity (raw)", fontsize=10)
    ax.set_ylabel(f"{path}\n(ssGSEA-lite score)", fontsize=9)
    ax.set_title(path, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    print(f"     {path:<42}  ρ={rho:+.3f}  p={pv:.4f}")

fig.suptitle(
    "Sphericity vs. Hallmark Pathway Activity\n"
    f"TCGA/CPTAC3 (n={n}); Spearman ρ with regression line",
    fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_DIR / "sphericity_pathway_scatter_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Saved: sphericity_pathway_scatter_v2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Layer C4: Full genome Spearman correlation (29 × 17,407)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[C4] Full genome radiogenomics correlation (29 × 17407 genes)...")
print("     Computing (this takes ~30 seconds)...")

corr_full, pval_full = spearman_matrix(X_scaled, X_expr)  # (29, 17407)

# BH FDR correction across all gene × feature pairs (global)
fdr_full_flat = bh_fdr(pval_full.flatten())
fdr_full = fdr_full_flat.reshape(pval_full.shape)

# Summary
n_sig_pairs = (fdr_full < 0.05).sum()
n_sig_genes = (fdr_full < 0.05).any(axis=0).sum()
print(f"     Feature-gene pairs FDR<0.05: {n_sig_pairs}")
print(f"     Unique genes with ≥1 feature FDR<0.05: {n_sig_genes}")

# Per-feature summary
print(f"\n     Per-feature significant genes (FDR<0.05):")
for i, fname in enumerate(feat_display):
    n_fi = (fdr_full[i] < 0.05).sum()
    if n_fi > 0:
        top_gene_idx = np.where(fdr_full[i] < 0.05)[0]
        top5 = sorted(top_gene_idx, key=lambda j: abs(corr_full[i,j]), reverse=True)[:3]
        top5_str = ", ".join([f"{gene_cols[j]}({corr_full[i,j]:+.2f})" for j in top5])
        print(f"       {fname:<42}  n={n_fi:4d}  top: {top5_str}")

# Save full correlation
print("\n     Saving full correlation table...")
corr_df = pd.DataFrame(corr_full, index=feat_display, columns=gene_cols)
fdr_df  = pd.DataFrame(fdr_full,  index=feat_display, columns=gene_cols)
corr_df.to_csv(OUT_DIR / "radiogenomics_full_corr_v2.csv", encoding="utf-8-sig")
print("     Saved: radiogenomics_full_corr_v2.csv")

# ── Heatmap: top features × top significant genes ───────────────────────────
# Select top 3 genes per feature (by |rho|, FDR<0.05) → union
top_genes = set()
for i in range(len(feat_cols)):
    sig_idx = np.where(fdr_full[i] < 0.05)[0]
    if len(sig_idx) > 0:
        best = sorted(sig_idx, key=lambda j: abs(corr_full[i, j]), reverse=True)[:5]
        top_genes.update(best)

top_genes = sorted(top_genes, key=lambda j: np.abs(corr_full[:, j]).max(), reverse=True)[:40]

if len(top_genes) >= 5:
    heat_data = corr_full[:, top_genes]          # (29, n_top)
    heat_fdr2 = fdr_full[:, top_genes]

    # Sort features by max |correlation| with these genes
    feat_order = np.argsort(np.abs(heat_data).max(1))[::-1]
    gene_order = np.argsort(np.abs(heat_data).max(0))[::-1]
    heat_data = heat_data[np.ix_(feat_order, gene_order)]
    heat_fdr2 = heat_fdr2[np.ix_(feat_order, gene_order)]
    top_gene_names = [gene_cols[j] for j in np.array(top_genes)[gene_order]]

    fig, ax = plt.subplots(figsize=(max(14, len(top_genes) * 0.42),
                                     max(8, len(feat_cols) * 0.36)))
    im2 = ax.imshow(heat_data, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.6)

    # Stars
    for ri in range(heat_data.shape[0]):
        for ci in range(heat_data.shape[1]):
            fv = heat_fdr2[ri, ci]
            if fv < 0.001:
                s = "***"
            elif fv < 0.01:
                s = "**"
            elif fv < 0.05:
                s = "*"
            else:
                s = ""
            if s:
                ax.text(ci, ri, s, ha="center", va="center", fontsize=5.5,
                        color="white" if abs(heat_data[ri, ci]) > 0.4 else "black")

    ax.set_xticks(range(len(top_gene_names)))
    ax.set_xticklabels(top_gene_names, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels([feat_display[i] for i in feat_order], fontsize=7.5)

    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.5)
    cbar2.set_label("Spearman ρ", fontsize=9)
    ax.set_title(
        f"Radiogenomics Correlation Heatmap — Top Genes (v2 Stability Features)\n"
        f"TCGA/CPTAC3 n={n}  |  * FDR<0.05  ** FDR<0.01  *** FDR<0.001",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "radiogenomics_heatmap_top_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: radiogenomics_heatmap_top_v2.png")
else:
    print("     (Too few significant genes for heatmap at FDR<0.05 — saved CSV only)")


# ─────────────────────────────────────────────────────────────────────────────
# Print biological summary
# ─────────────────────────────────────────────────────────────────────────────
sph_path_corrs = {}
for path in gmt_h:
    if path in pathway_scores.columns:
        ps = pathway_scores[path].values
        rho, pv = stats.spearmanr(sph_raw, ps)
        sph_path_corrs[path] = (rho, pv)

print(f"\n{'='*65}")
print("BIOLOGICAL SUMMARY")
print(f"{'='*65}")
print(f"\nGSEA top Hallmark pathways enriched in RSI+:")
for _, row in sig_h.sort_values("NES", ascending=False).iterrows():
    print(f"  {row.Term:<45}  NES={row.NES:+.2f}  FDR={row['FDR q-val']:.3f}")

print(f"\nSphericity correlation with Hallmark pathways (top |ρ|):")
sph_sorted = sorted(sph_path_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
for path, (rho, pv) in sph_sorted[:10]:
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
    print(f"  {path:<45}  ρ={rho:+.3f}  p={pv:.4f}  {sig}")

print(f"\n{'='*65}")
print("Step 7 Complete.")
print(f"{'='*65}")
print(f"  Output directory: {OUT_DIR}")
print(f"  volcano_DEA_v2.png")
print(f"  GSEA_hallmark_bubble_v2.png")
print(f"  feature_pathway_heatmap_v2.png")
print(f"  sphericity_pathway_scatter_v2.png")
print(f"  radiogenomics_heatmap_top_v2.png")
print(f"  feature_pathway_corr_v2.csv")
print(f"  radiogenomics_full_corr_v2.csv")
