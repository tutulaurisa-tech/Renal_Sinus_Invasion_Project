# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 2: Batch Effect Analysis (1316 features)
============================================================
New vs v1:
  - Analyses all 1316 features (7 image types)
  - Per-image-type KS significance breakdown
  - Side-by-side comparison with v1 (107 features)
  - Recommends normalization strategy

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step2_batch_effect.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE     = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR = BASE / "3_Extracted_Features"
OUT_DIR  = BASE / "5_Results" / "v2" / "BatchEffect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_TYPES = ["original", "logarithm", "exponential", "square",
             "squareroot", "gradient", "wavelet"]

COLORS = {"PUTH": "#2196F3", "PUCH": "#4CAF50",
          "TCGA": "#FF9800", "JSPH-RN": "#E91E63"}

print("=" * 60)
print("V2 Batch Effect Analysis (1316 features)")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────
three_ctr = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df   = pd.read_csv(FEAT_DIR / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")
jsph_df["Center"] = "JSPH-RN"

SKIP_EXACT  = {"Case_ID", "PatientID", "Center", "RSI", "Cohort"}
SKIP_PFX    = "diagnostics_"

def feature_cols_from(df):
    return [c for c in df.columns
            if c not in SKIP_EXACT and not c.startswith(SKIP_PFX)]

def img_type_of(feat):
    if feat.startswith("wavelet"):
        return "wavelet"
    for t in IMG_TYPES:
        if feat.startswith(t + "_"):
            return t
    return "unknown"

feat_cols = feature_cols_from(three_ctr)
print(f"Features: {len(feat_cols)}")

# Stack all centers
all_df = pd.concat([three_ctr, jsph_df[["Case_ID", "Center", "RSI"] + feat_cols]],
                   ignore_index=True)

print("\nCohort sizes:")
for ctr in ["PUTH", "PUCH", "TCGA", "JSPH-RN"]:
    sub = all_df[all_df.Center == ctr]
    print(f"  {ctr}: n={len(sub)}  RSI+={int(sub.RSI.sum())}")

# ── Matrix ────────────────────────────────────────────────────────────────
X_all  = all_df[feat_cols].values.astype(float)
c_arr  = all_df.Center.values
rsi_arr = all_df.RSI.fillna(-1).astype(int).values

valid = ~np.isnan(X_all).any(axis=1)
X_all, c_arr, rsi_arr = X_all[valid], c_arr[valid], rsi_arr[valid]
print(f"\nValid samples: {X_all.shape[0]}  (dropped {(~valid).sum()} NaN rows)")

X_scaled = StandardScaler().fit_transform(X_all)


# ══════════════════════════════════════════════════════════════════════════
# 1.  PCA — colored by center
# ══════════════════════════════════════════════════════════════════════════
print("\n1. PCA...")
pca   = PCA(n_components=2, random_state=42)
Xpca  = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for name, color in COLORS.items():
    m = c_arr == name
    ax.scatter(Xpca[m, 0], Xpca[m, 1], c=color,
               label=f"{name} (n={m.sum()})", alpha=0.6, s=20, edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA — by Center (V2, 1316 features)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(Xpca[:, 0], Xpca[:, 1], c="lightgray", alpha=0.2, s=12, edgecolors="none")
m_rn = c_arr == "JSPH-RN"
ax.scatter(Xpca[m_rn, 0], Xpca[m_rn, 1],
           c=np.where(rsi_arr[m_rn] == 1, "#E91E63", "#2196F3"),
           alpha=0.85, s=35, edgecolors="none")
ax.legend(handles=[
    mpatches.Patch(color="#E91E63", label=f"JSPH-RN RSI+ (n={(rsi_arr[m_rn]==1).sum()})"),
    mpatches.Patch(color="#2196F3", label=f"JSPH-RN RSI- (n={(rsi_arr[m_rn]==0).sum()})"),
    mpatches.Patch(color="lightgray", label="PUTH/PUCH/TCGA"),
], fontsize=9)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA — JSPH-RN RSI Labels", fontweight="bold")
ax.grid(True, alpha=0.3)

plt.suptitle("Batch Effect Check: PCA (V2, 1316 features)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: pca_v2.png")


# ══════════════════════════════════════════════════════════════════════════
# 2.  KS test — JSPH-RN vs each reference center
# ══════════════════════════════════════════════════════════════════════════
print("\n2. KS test...")
m_jsph = c_arr == "JSPH-RN"
rows   = []
for i, feat in enumerate(feat_cols):
    img_type = next((t for t in IMG_TYPES if feat.startswith(t + "_")), "unknown")
    row = {"Feature": feat, "ImageType": img_type_of(feat)}
    for ref in ["PUTH", "PUCH", "TCGA"]:
        ks, p = stats.ks_2samp(X_all[m_jsph, i], X_all[c_arr == ref, i])
        row[f"KS_vs_{ref}"] = round(ks, 4)
        row[f"p_vs_{ref}"]  = round(p, 4)
        row[f"sig_vs_{ref}"] = int(p < 0.05)
    rows.append(row)

ks_df = pd.DataFrame(rows)
ks_df["n_sig"] = ks_df[["sig_vs_PUTH", "sig_vs_PUCH", "sig_vs_TCGA"]].sum(axis=1)
ks_df.to_csv(OUT_DIR / "ks_test_v2.csv", index=False, encoding="utf-8-sig")

print(f"\n  JSPH-RN vs reference centers (p<0.05 / {len(ks_df)} features):")
for ref in ["PUTH", "PUCH", "TCGA"]:
    n   = ks_df[f"sig_vs_{ref}"].sum()
    pct = n / len(ks_df) * 100
    print(f"    vs {ref}: {n} ({pct:.1f}%)")
n_clean = (ks_df["n_sig"] == 0).sum()
print(f"  No sig diff vs ANY: {n_clean} ({n_clean/len(ks_df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# 3.  Batch effect breakdown by image type  ← NEW in v2
# ══════════════════════════════════════════════════════════════════════════
print("\n3. Batch effect by image type...")
type_rows = []
for img_type in IMG_TYPES:
    sub = ks_df[ks_df.ImageType == img_type]
    if len(sub) == 0:
        continue
    for ref in ["PUTH", "PUCH", "TCGA"]:
        n_sig = sub[f"sig_vs_{ref}"].sum()
        pct   = n_sig / len(sub) * 100
        type_rows.append({
            "ImageType": img_type,
            "Reference": ref,
            "n_features": len(sub),
            "n_sig": int(n_sig),
            "pct_sig": round(pct, 1),
            "mean_KS": round(sub[f"KS_vs_{ref}"].mean(), 4),
        })

type_df = pd.DataFrame(type_rows)
type_df.to_csv(OUT_DIR / "batch_by_imagetype_v2.csv", index=False, encoding="utf-8-sig")

# Pivot for display
pivot = type_df[type_df.Reference == "PUTH"].set_index("ImageType")[["pct_sig", "mean_KS"]]
pivot.columns = ["% sig vs PUTH", "Mean KS vs PUTH"]
print(pivot.to_string())

# Heatmap of % significant by image type × reference
fig, ax = plt.subplots(figsize=(8, 5))
heat_data = type_df.pivot(index="ImageType", columns="Reference", values="pct_sig")
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd",
            vmin=0, vmax=100, ax=ax,
            cbar_kws={"label": "% features p<0.05"})
ax.set_title("Batch Effect by Image Type\nJSPH-RN vs Reference Centers (V2)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "batch_by_imagetype_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: batch_by_imagetype_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 4.  V1 vs V2 comparison  (if v1 KS results exist)
# ══════════════════════════════════════════════════════════════════════════
v1_ks_path = BASE / "5_Results" / "BatchEffect" / "ks_test_results.csv"
if v1_ks_path.exists():
    print("\n4. V1 vs V2 comparison...")
    v1_ks = pd.read_csv(v1_ks_path)
    v1_pct = v1_ks["sig_vs_PUTH"].sum() / len(v1_ks) * 100
    v2_orig = ks_df[ks_df.ImageType == "original"]
    v2_orig_pct = v2_orig["sig_vs_PUTH"].sum() / len(v2_orig) * 100
    v2_all_pct  = ks_df["sig_vs_PUTH"].sum() / len(ks_df) * 100

    print(f"  V1 (107 original):           {v1_pct:.1f}% sig vs PUTH")
    print(f"  V2 original only ({len(v2_orig)}): {v2_orig_pct:.1f}% sig vs PUTH")
    print(f"  V2 all 1316 features:         {v2_all_pct:.1f}% sig vs PUTH")


# ══════════════════════════════════════════════════════════════════════════
# 5.  Top-40 KS heatmap
# ══════════════════════════════════════════════════════════════════════════
top40 = ks_df.nlargest(40, "KS_vs_PUTH")["Feature"].tolist()
hm    = ks_df.set_index("Feature").loc[top40,
         ["KS_vs_PUTH", "KS_vs_PUCH", "KS_vs_TCGA"]]
hm.columns = ["vs PUTH", "vs PUCH", "vs TCGA"]

fig, ax = plt.subplots(figsize=(7, 12))
sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0, vmax=0.5, ax=ax, cbar_kws={"label": "KS statistic"})
ax.set_title("Top 40 Features by KS Statistic\nJSPH-RN vs Other Centers (V2)",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "ks_heatmap_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: ks_heatmap_v2.png")


# ══════════════════════════════════════════════════════════════════════════
# 6.  Verdict
# ══════════════════════════════════════════════════════════════════════════
pct_vs_puth = ks_df["sig_vs_PUTH"].sum() / len(ks_df) * 100

print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")
if pct_vs_puth < 20:
    verdict = "SMALL"
    rec     = "Direct cross-center validation OK."
elif pct_vs_puth < 50:
    verdict = "MODERATE"
    rec     = "Apply Z-score normalization (PUTH mean/std) before modeling."
else:
    verdict = "LARGE"
    rec     = "Z-score normalization required. Consider ComBat if performance is poor."

print(f"[{verdict}] {pct_vs_puth:.1f}% features differ vs PUTH (p<0.05)")
print(f"Recommendation: {rec}")
print(f"\nOutputs: {OUT_DIR}")
print("Next: python step3_feature_selection.py")
