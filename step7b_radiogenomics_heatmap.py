# -*- coding: utf-8 -*-
"""
V2 Pipeline - Step 7b: Radiogenomics Gene Heatmap (per-feature FDR)
=====================================================================
Supplement to step7_radiogenomics_v2.py.

Global FDR correction across 29×17407 pairs is extremely conservative.
Standard radiogenomics practice: per-feature BH FDR (correct within each
feature's 17407 tests independently), then select union of significant genes.

Generates:
  radiogenomics_heatmap_perfdr_v2.png   Feature × gene heatmap (FDR<0.05 per-feature)
  sphericity_keygene_scatter_v2.png     Sphericity vs SFRP1/PBK scatter (biological story)
  radiogenomics_pergene_sig_v2.csv      Significant pairs table

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step7b_radiogenomics_heatmap.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

BASE    = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR = BASE / "3_Extracted_Features"
MOD_DIR  = BASE / "5_Modeling" / "v2_stability"
GEN_DIR  = BASE / "1_Raw_Data" / "Center4_TCGA" / "Genomics"
OUT_DIR  = BASE / "6_Radiogenomics" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("=" * 65)
print("Step 7b: Radiogenomics Heatmap (per-feature FDR)")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    idx = np.argsort(pvals)
    fdr = np.ones(n)
    fdr[idx] = pvals[idx] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        fdr[idx[i]] = min(fdr[idx[i]], fdr[idx[i + 1]])
    return np.clip(fdr, 0, 1)


def spearman_vec(x, Y):
    """Spearman correlation of 1-D array x with each column of Y (n × p)."""
    from scipy.stats import rankdata
    n = len(x)
    xr = rankdata(x).astype(float)
    Yr = np.apply_along_axis(rankdata, 0, Y).astype(float)
    xr -= xr.mean(); xr /= (xr.std() + 1e-12)
    Yr -= Yr.mean(0); Yr /= (Yr.std(0) + 1e-12)
    rho = (xr @ Yr) / (n - 1)
    rho = np.clip(rho, -1, 1)
    t   = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-14))
    pv  = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return rho, pv


def short_name(feat, max_len=40):
    parts = feat.split("_")
    if feat.startswith("wavelet-"):
        s = parts[0].replace("wavelet-", "w") + "_" + "_".join(parts[1:])
    else:
        s = feat
    return s[:max_len] + ("…" if len(s) > max_len else "")


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
expr_df   = pd.read_csv(GEN_DIR / "All_TCGA_Gene_Expression_withLabel.csv")
gene_cols = [c for c in expr_df.columns if c not in ["PatientID", "RSI", "Cohort"]]
y         = expr_df["RSI"].values.astype(int)
X_expr    = expr_df[gene_cols].values.astype(float)
n         = len(y)
print(f"  Expression: {n} samples × {len(gene_cols)} genes")

all_df    = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
tcga_df   = all_df[all_df.Center == "TCGA"].reset_index(drop=True)
sel_df    = pd.read_csv(MOD_DIR / "selected_features_stability.csv")
feat_cols = sel_df["Feature"].tolist()

scaler_df = pd.read_csv(MOD_DIR / "scaler_params_stability.csv")
scaler_df = scaler_df.set_index("Feature").loc[feat_cols].reset_index()
mu = scaler_df["mean"].values
sd = scaler_df["std"].values
X_raw    = tcga_df[feat_cols].values.astype(float)
X_scaled = (X_raw - mu) / sd
feat_display = [short_name(f) for f in feat_cols]

sph_idx  = feat_cols.index("original_shape_Sphericity")
sph_raw  = X_raw[:, sph_idx]
print(f"  Radiomics: {X_scaled.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature Spearman + BH FDR
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing per-feature Spearman correlations + BH FDR...")
print("  (29 features × 17407 genes, ~60s)")

all_rho = np.zeros((len(feat_cols), len(gene_cols)), float)
all_fdr = np.ones ((len(feat_cols), len(gene_cols)), float)
all_pv  = np.ones ((len(feat_cols), len(gene_cols)), float)

for i, feat in enumerate(feat_cols):
    rho, pv = spearman_vec(X_scaled[:, i], X_expr)
    fdr     = bh_fdr(pv)
    all_rho[i] = rho
    all_pv [i] = pv
    all_fdr[i] = fdr
    n_sig = (fdr < 0.05).sum()
    if n_sig > 0:
        print(f"  {feat_display[i]:<42}  FDR<0.05: {n_sig:4d}")

n_sig_pairs = (all_fdr < 0.05).sum()
n_sig_genes = (all_fdr < 0.05).any(axis=0).sum()
print(f"\n  Total pairs FDR<0.05 (per-feature): {n_sig_pairs}")
print(f"  Unique significant genes:           {n_sig_genes}")


# ─────────────────────────────────────────────────────────────────────────────
# Select genes for heatmap
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: union of top genes per feature (FDR<0.05, pick top by |rho|)
#   + always include biological anchor genes regardless of FDR
anchor_genes = ["SFRP1", "FBLN1", "PBK", "CDC45", "NCAPG", "LMNB2",
                "TUBB", "ADSL", "BEGAIN", "NLE1", "EFNB1", "CPXM1"]
anchor_genes = [g for g in anchor_genes if g in gene_cols]

top_gene_set = set()
for i in range(len(feat_cols)):
    sig_mask = all_fdr[i] < 0.05
    if sig_mask.sum() > 0:
        order = np.argsort(np.abs(all_rho[i]) * sig_mask)[::-1]
        chosen = [j for j in order if sig_mask[j]][:8]
        top_gene_set.update(chosen)

# Add anchor gene indices
for g in anchor_genes:
    top_gene_set.add(gene_cols.index(g))

top_gene_idxs = sorted(top_gene_set,
                       key=lambda j: np.abs(all_rho[:, j]).max(),
                       reverse=True)[:60]   # cap at 60 genes

print(f"\n  Genes selected for heatmap: {len(top_gene_idxs)}")
top_gene_names = [gene_cols[j] for j in top_gene_idxs]
print(f"  Includes anchors: {[g for g in anchor_genes if g in top_gene_names]}")


# ─────────────────────────────────────────────────────────────────────────────
# Save significant pairs table
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for i, fname in enumerate(feat_cols):
    for j in range(len(gene_cols)):
        if all_fdr[i, j] < 0.05:
            rows.append({
                "Feature": fname, "Gene": gene_cols[j],
                "Spearman_rho": round(all_rho[i, j], 4),
                "pval": round(all_pv[i, j], 6),
                "FDR_per_feature": round(all_fdr[i, j], 4),
            })
sig_df = pd.DataFrame(rows).sort_values(["Feature", "FDR_per_feature"])
sig_df.to_csv(OUT_DIR / "radiogenomics_pergene_sig_v2.csv",
              index=False, encoding="utf-8-sig")
print(f"  Saved: radiogenomics_pergene_sig_v2.csv  ({len(sig_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap: selected features × selected genes
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating heatmap...")

# Only show features that have ≥1 significant gene (per-feature FDR<0.05)
# or are in top 12 by SHAP importance
shap_top12 = [
    "original_shape_Sphericity",
    "wavelet-HHH_glcm_Imc1",
    "logarithm_firstorder_Skewness",
    "logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "logarithm_ngtdm_Complexity",
    "wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "wavelet-HHH_glcm_MCC",
    "wavelet-LLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "wavelet-HHH_gldm_LowGrayLevelEmphasis",
    "wavelet-LLH_glcm_ClusterTendency",
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
]
feat_show_mask = np.array(
    [(all_fdr[i] < 0.05).any() or feat_cols[i] in shap_top12
     for i in range(len(feat_cols))]
)
feat_show_idx = np.where(feat_show_mask)[0]
if len(feat_show_idx) == 0:
    feat_show_idx = np.arange(min(15, len(feat_cols)))

heat_rho = all_rho[np.ix_(feat_show_idx, top_gene_idxs)]
heat_fdr = all_fdr[np.ix_(feat_show_idx, top_gene_idxs)]

# Sort features: by max |rho| with the selected genes
feat_order = np.argsort(np.abs(heat_rho).max(1))[::-1]
# Sort genes: cluster by correlation pattern (simple: sort by max |rho| across shown features)
gene_order = np.argsort(np.abs(heat_rho).max(0))[::-1]

heat_rho_sorted = heat_rho[np.ix_(feat_order, gene_order)]
heat_fdr_sorted = heat_fdr[np.ix_(feat_order, gene_order)]
feat_labels     = [feat_display[feat_show_idx[i]] for i in feat_order]
gene_labels     = [top_gene_names[j] for j in gene_order]

n_rows, n_cols = heat_rho_sorted.shape
fig_h = max(8,  n_rows * 0.38)
fig_w = max(14, n_cols * 0.45)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(heat_rho_sorted, cmap="RdBu_r", aspect="auto",
               vmin=-0.35, vmax=0.35)

# Significance markers
for ri in range(n_rows):
    for ci in range(n_cols):
        fv = heat_fdr_sorted[ri, ci]
        rv = heat_rho_sorted[ri, ci]
        if fv < 0.001:
            s = "***"
        elif fv < 0.01:
            s = "**"
        elif fv < 0.05:
            s = "*"
        else:
            s = ""
        if s:
            tc = "white" if abs(rv) > 0.24 else "black"
            ax.text(ci, ri, s, ha="center", va="center",
                    fontsize=5.5, color=tc, fontweight="bold")

# Highlight anchor genes with bold x-tick labels
ax.set_xticks(range(n_cols))
xlabels = []
for g in gene_labels:
    if g in anchor_genes:
        xlabels.append(f"★{g}")
    else:
        xlabels.append(g)
ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=7.5)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(feat_labels, fontsize=8)

cbar = plt.colorbar(im, ax=ax, shrink=0.45, pad=0.01)
cbar.set_label("Spearman ρ (feature ↔ gene)", fontsize=9)

# Add box around Sphericity row
sph_row_pos = None
for ri, fi in enumerate(feat_order):
    if feat_cols[feat_show_idx[fi]] == "original_shape_Sphericity":
        sph_row_pos = ri
        break
if sph_row_pos is not None:
    rect = plt.Rectangle((-0.5, sph_row_pos - 0.5), n_cols, 1,
                          fill=False, edgecolor="#E91E63", linewidth=2.0)
    ax.add_patch(rect)
    ax.text(-0.7, sph_row_pos, "★", ha="right", va="center",
            fontsize=11, color="#E91E63")

ax.set_title(
    "Radiogenomics Correlation Heatmap — v2 Stability Features\n"
    f"TCGA/CPTAC3 (n={n});  Per-feature BH FDR  "
    "* <0.05  ** <0.01  *** <0.001\n"
    "★ = biologically annotated anchor gene;  pink box = Sphericity",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "radiogenomics_heatmap_perfdr_v2.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: radiogenomics_heatmap_perfdr_v2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Scatter: Sphericity vs key genes (SFRP1 / PBK / FBLN1)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating key-gene scatter plots...")

key_genes_scatter = [
    ("SFRP1", "Wnt antagonist / tumor suppressor",  "#4CAF50"),
    ("FBLN1", "ECM glycoprotein / invasion barrier", "#FF9800"),
    ("PBK",   "Mitosis kinase / proliferation",      "#9C27B0"),
]
key_genes_scatter = [(g, lbl, c) for g, lbl, c in key_genes_scatter
                     if g in gene_cols]

RSI_POS = "#E91E63"
RSI_NEG = "#2196F3"

fig, axes = plt.subplots(1, len(key_genes_scatter),
                         figsize=(5.5 * len(key_genes_scatter), 5.2))
if len(key_genes_scatter) == 1:
    axes = [axes]

for ax, (gene, bio_label, gene_color) in zip(axes, key_genes_scatter):
    gi  = gene_cols.index(gene)
    ge  = X_expr[:, gi]
    rho = all_rho[sph_idx, gi]
    fdr_v = all_fdr[sph_idx, gi]
    pv  = all_pv[sph_idx, gi]

    # Background colour strip
    for label, mask, c in [(0, y == 0, RSI_NEG), (1, y == 1, RSI_POS)]:
        ax.scatter(sph_raw[mask], ge[mask], c=c, s=22, alpha=0.65,
                   label=f"RSI{'+'if label else '−'} (n={mask.sum()})",
                   edgecolors="none")

    # Regression line
    m, b, *_ = stats.linregress(sph_raw, ge)
    xs = np.linspace(sph_raw.min(), sph_raw.max(), 100)
    ax.plot(xs, m * xs + b, color="black", lw=1.8, ls="--", alpha=0.75)

    # Stat box
    fdr_str = f"FDR={fdr_v:.3f}" if fdr_v >= 0.001 else "FDR<0.001"
    pv_str  = f"p<0.001"  if pv < 0.001 else f"p={pv:.3f}"
    ax.text(0.97, 0.97,
            f"ρ={rho:+.3f}\n{pv_str}\n{fdr_str}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    # Biological label
    ax.set_title(f"{gene}\n({bio_label})", fontsize=10, fontweight="bold",
                 color=gene_color)
    ax.set_xlabel("Sphericity (raw)", fontsize=10)
    ax.set_ylabel(f"{gene} expression\n(log₂-normalised)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    print(f"  {gene:<8}  ρ={rho:+.4f}  pval={pv:.2e}  FDR(per-feat)={fdr_v:.4f}")

fig.suptitle(
    "Sphericity vs. Key Radiogenomic Genes\n"
    "Biological anchor genes for renal sinus invasion pathway",
    fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_DIR / "sphericity_keygene_scatter_v2.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: sphericity_keygene_scatter_v2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Significant feature-gene pairs (per-feature FDR<0.05) summary:")
print(f"{'='*65}")
for fname in feat_cols:
    sub = sig_df[sig_df.Feature == fname]
    if len(sub) > 0:
        top3 = sub.sort_values("FDR_per_feature").head(3)
        genes_str = ", ".join(
            [f"{r.Gene}({r.Spearman_rho:+.2f})" for _, r in top3.iterrows()]
        )
        print(f"  {short_name(fname):<42}  n={len(sub):4d}  top: {genes_str}")

print(f"\n{'='*65}")
print("Step 7b Complete.")
print(f"{'='*65}")
print(f"  Output: {OUT_DIR}")
print(f"  radiogenomics_heatmap_perfdr_v2.png")
print(f"  sphericity_keygene_scatter_v2.png")
print(f"  radiogenomics_pergene_sig_v2.csv")
