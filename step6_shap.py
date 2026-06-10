# -*- coding: utf-8 -*-
"""
V2 Pipeline - Step 6: SHAP Interpretability Analysis
=====================================================
Model: Elastic Net (Bootstrap Stability, step3_stability.py)
SHAP:  LinearExplainer  (appropriate for sklearn LogisticRegression)
       SHAP values are in log-odds space (linear model output before sigmoid)

If shap is not installed:
    pip install shap
  or
    conda install -c conda-forge shap

Outputs (5_Results/v2_stability/shap/):
  shap_summary_beeswarm.png     Global beeswarm (training set)
  shap_importance_bar.png       Mean |SHAP| bar chart (all cohorts)
  shap_dependence_top5.png      Dependence plots for top 5 features
  shap_cohort_comparison.png    Per-cohort mean |SHAP| heatmap
  shap_waterfall_rsi_pos.png    Waterfall: high-confidence RSI+ case
  shap_waterfall_rsi_neg.png    Waterfall: high-confidence RSI- case
  shap_values_all.csv           Raw SHAP values for all cases

Usage:
    conda activate radiomics
    pip install shap   # only once
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step6_shap.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import shap
    print(f"shap version: {shap.__version__}")
except ImportError:
    print("ERROR: shap not installed.")
    print("Run: pip install shap")
    raise

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE    = Path(r"D:\RSI_Project_Workspace")
FEAT    = BASE / "3_Extracted_Features"
MOD_DIR = BASE / "5_Modeling" / "v2_stability"
OUT_DIR = BASE / "5_Results"  / "v2_stability" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
L1_RATIO = 0.5
np.random.seed(SEED)

IMG_TYPES  = ["original", "logarithm", "exponential", "square",
              "squareroot", "gradient", "wavelet"]
TYPE_COLOR = {
    "original":    "#2196F3",
    "wavelet":     "#E91E63",
    "gradient":    "#FF9800",
    "logarithm":   "#4CAF50",
    "exponential": "#9C27B0",
    "square":      "#00BCD4",
    "squareroot":  "#FF5722",
}

def img_type_of(feat):
    if feat.startswith("wavelet"):
        return "wavelet"
    for t in IMG_TYPES:
        if feat.startswith(t + "_"):
            return t
    return "unknown"


def short_name(feat, max_len=42):
    """Shorten long feature names for plot labels."""
    parts = feat.split("_")
    # wavelet-HHH_glcm_Imc1  ->  wHHH_glcm_Imc1
    if feat.startswith("wavelet-"):
        band = parts[0].replace("wavelet-", "w")
        rest = "_".join(parts[1:])
        s = f"{band}_{rest}"
    else:
        s = feat
    return s[:max_len] + ("..." if len(s) > max_len else "")


print("=" * 65)
print("V2 Pipeline - Step 6: SHAP Analysis (Elastic Net Stability)")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
all_df  = pd.read_csv(FEAT / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df = pd.read_csv(FEAT / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")

sel_df    = pd.read_csv(MOD_DIR / "selected_features_stability.csv")
feat_cols = sel_df["Feature"].tolist()
n_feats   = len(feat_cols)
print(f"  Stable features: {n_feats}")

scaler_df = pd.read_csv(MOD_DIR / "scaler_params_stability.csv")
scaler_df = scaler_df.set_index("Feature").loc[feat_cols].reset_index()
mu   = scaler_df["mean"].values
sd   = scaler_df["std"].values

SKIP = {"Case_ID", "PatientID", "Center", "RSI", "Cohort"}
puth_df = all_df[all_df.Center == "PUTH"].reset_index(drop=True)
puch_df = all_df[all_df.Center == "PUCH"].reset_index(drop=True)
tcga_df = all_df[all_df.Center == "TCGA"].reset_index(drop=True)

def scale(X_raw):
    return (X_raw - mu) / sd

X_puth_s = scale(puth_df[feat_cols].values.astype(float))
X_jsph_s = scale(jsph_df[feat_cols].values.astype(float))
X_puch_s = scale(puch_df[feat_cols].values.astype(float))
X_tcga_s = scale(tcga_df[feat_cols].values.astype(float))

y_puth = puth_df.RSI.values.astype(int)
y_jsph = jsph_df.RSI.values.astype(int)
y_puch = puch_df.RSI.values.astype(int)
y_tcga = tcga_df.RSI.values.astype(int)

X_train = np.vstack([X_puth_s, X_jsph_s])
y_train = np.concatenate([y_puth, y_jsph])
print(f"  Training: {len(y_train)} cases  RSI+={y_train.sum()}")
print(f"  PUCH: {len(y_puch)}  TCGA: {len(y_tcga)}")


# ─────────────────────────────────────────────────────────────────────────────
# Re-fit final Elastic Net model (same as step3_stability Step C)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRe-fitting Elastic Net on stable features...")
strat = (np.concatenate([np.zeros(len(y_puth), int),
                          np.ones(len(y_jsph), int)]) * 2 + y_train)

cv_model = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 60),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    penalty="elasticnet", solver="saga",
    l1_ratios=[L1_RATIO], class_weight="balanced",
    scoring="roc_auc", max_iter=5000,
    random_state=SEED, n_jobs=-1,
)
cv_model.fit(X_train, y_train)
best_C = float(cv_model.C_[0])
print(f"  Best C = {best_C:.5f}")

model = LogisticRegression(
    C=best_C, penalty="elasticnet", solver="saga",
    l1_ratio=L1_RATIO, class_weight="balanced",
    max_iter=5000, random_state=SEED,
)
model.fit(X_train, y_train)

coefs = model.coef_[0]
n_nonzero = (coefs != 0).sum()
print(f"  Non-zero coefficients: {n_nonzero} / {n_feats}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP LinearExplainer
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing SHAP values (LinearExplainer)...")

# Background = training set (PUTH + JSPH-RN)
# feature_perturbation="interventional" uses marginal distribution (more robust)
explainer = shap.LinearExplainer(
    model, X_train,
    feature_perturbation="interventional"
)

# Compute for all cohorts
shap_train = explainer(X_train)
shap_puch  = explainer(X_puch_s)
shap_tcga  = explainer(X_tcga_s)

# shap_values arrays  [n_samples x n_features]
sv_train = shap_train.values   # shape (576, 29)
sv_puch  = shap_puch.values    # shape (90,  29)
sv_tcga  = shap_tcga.values    # shape (170, 29)

# Per-cohort metadata
cohort_labels = (["PUTH"] * len(y_puth) + ["JSPH-RN"] * len(y_jsph))
cohort_arr    = np.array(cohort_labels)

print(f"  SHAP computed: training={sv_train.shape}, PUCH={sv_puch.shape}, TCGA={sv_tcga.shape}")

# Feature display names
feat_display = [short_name(f) for f in feat_cols]
feat_types   = [img_type_of(f) for f in feat_cols]


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Global Beeswarm (training set)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    sv_train, X_train,
    feature_names=feat_display,
    show=False, max_display=20,
    plot_size=(10, 8),
)
plt.title(
    "SHAP Beeswarm - Elastic Net (Stability)\n"
    "Training set: PUTH + JSPH-RN (576 cases)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary_beeswarm.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Mean |SHAP| bar chart (all cohorts combined, color by type)
# ─────────────────────────────────────────────────────────────────────────────
sv_all  = np.vstack([sv_train, sv_puch, sv_tcga])
mean_abs = np.abs(sv_all).mean(axis=0)
order    = np.argsort(mean_abs)[::-1]

top_n = min(20, n_feats)
idx   = order[:top_n]

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = [TYPE_COLOR.get(feat_types[i], "gray") for i in idx]
ax.barh(range(top_n), mean_abs[idx][::-1], color=colors_bar[::-1], alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels(
    [feat_display[i] for i in idx[::-1]], fontsize=8.5
)
ax.set_xlabel("Mean |SHAP value| (log-odds)", fontsize=10)
ax.set_title(
    "SHAP Feature Importance - Elastic Net (Stability)\n"
    "All cohorts combined (576 train + 90 PUCH + 170 TCGA)",
    fontsize=11, fontweight="bold"
)
legend_patches = [mpatches.Patch(color=c, label=t)
                  for t, c in TYPE_COLOR.items()
                  if t in set(feat_types)]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_importance_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Dependence plots - top 5 features
# ─────────────────────────────────────────────────────────────────────────────
top5_idx = order[:5]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, fi in zip(axes, top5_idx):
    # Find best interaction feature (highest correlation with SHAP residuals)
    shap.dependence_plot(
        fi, sv_all, np.vstack([X_train, X_puch_s, X_tcga_s]),
        feature_names=feat_display,
        ax=ax, show=False,
        dot_size=8, alpha=0.5,
    )
    ax.set_title(feat_display[fi], fontsize=8, fontweight="bold")
    ax.set_xlabel("Feature value (standardized)", fontsize=7)
    ax.set_ylabel("SHAP value" if fi == top5_idx[0] else "", fontsize=7)
    ax.tick_params(labelsize=7)

fig.suptitle(
    "SHAP Dependence Plots - Top 5 Features (all cohorts)\n"
    "Color = interaction feature (auto-selected by SHAP)",
    fontsize=10, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_dependence_top5.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_dependence_top5.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Per-cohort SHAP importance heatmap
# ─────────────────────────────────────────────────────────────────────────────
sv_puth_only = sv_train[:len(y_puth)]
sv_jsph_only = sv_train[len(y_puth):]

cohort_mean_abs = {
    "PUTH":    np.abs(sv_puth_only).mean(axis=0),
    "JSPH-RN": np.abs(sv_jsph_only).mean(axis=0),
    "PUCH":    np.abs(sv_puch).mean(axis=0),
    "TCGA":    np.abs(sv_tcga).mean(axis=0),
}

top12 = order[:12]
heatmap_data = np.array([cohort_mean_abs[c][top12]
                          for c in ["PUTH", "JSPH-RN", "PUCH", "TCGA"]])

# Normalize each feature column to [0,1] for cross-cohort comparison
col_max = heatmap_data.max(axis=0, keepdims=True)
col_max[col_max == 0] = 1
heatmap_norm = heatmap_data / col_max

fig, ax = plt.subplots(figsize=(12, 3.5))
im = ax.imshow(heatmap_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(12))
ax.set_xticklabels([feat_display[i] for i in top12],
                   rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels(["PUTH (train)", "JSPH-RN (train)", "PUCH (ext.)", "TCGA (ext.)"],
                   fontsize=9)

# Annotate cells with raw values
for r in range(4):
    for c in range(12):
        val = cohort_mean_abs[list(cohort_mean_abs.keys())[r]][top12[c]]
        ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                fontsize=6.5, color="black" if heatmap_norm[r, c] < 0.7 else "white")

plt.colorbar(im, ax=ax, label="Relative mean |SHAP| (normalized per feature)")
ax.set_title(
    "Per-Cohort SHAP Importance (Top 12 Features)\n"
    "Color normalized per feature column; values = raw mean |SHAP|",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_cohort_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_cohort_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 & 6: Waterfall plots - representative cases
# ─────────────────────────────────────────────────────────────────────────────
# Select from training set: highest confidence RSI+ and RSI-
probs_train = model.predict_proba(X_train)[:, 1]

# High-confidence RSI+ case (true positive, highest predicted probability)
pos_idx   = np.where(y_train == 1)[0]
best_pos  = pos_idx[np.argmax(probs_train[pos_idx])]
prob_pos  = probs_train[best_pos]

# High-confidence RSI- case (true negative, lowest predicted probability)
neg_idx   = np.where(y_train == 0)[0]
best_neg  = neg_idx[np.argmin(probs_train[neg_idx])]
prob_neg  = probs_train[best_neg]

print(f"\n  Waterfall cases:")
print(f"    RSI+ case: idx={best_pos}  prob={prob_pos:.3f}")
print(f"    RSI- case: idx={best_neg}  prob={prob_neg:.3f}")

for case_idx, label, prob, fname in [
    (best_pos, f"RSI+ case (prob={prob_pos:.3f})", prob_pos, "shap_waterfall_rsi_pos.png"),
    (best_neg, f"RSI- case (prob={prob_neg:.3f})", prob_neg, "shap_waterfall_rsi_neg.png"),
]:
    plt.figure(figsize=(10, 6))
    sv_case = shap.Explanation(
        values=sv_train[case_idx],
        base_values=float(explainer.expected_value),
        data=X_train[case_idx],
        feature_names=feat_display,
    )
    shap.waterfall_plot(sv_case, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {label}\n(log-odds space; base = {explainer.expected_value:.3f})",
              fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Save raw SHAP values
# ─────────────────────────────────────────────────────────────────────────────
rows = []
cohorts_data = [
    ("PUTH",    sv_puth_only, y_puth,
     puth_df.Case_ID.values),
    ("JSPH-RN", sv_jsph_only, y_jsph,
     jsph_df["Case_ID"].values if "Case_ID" in jsph_df.columns
     else jsph_df.index.astype(str).values),
    ("PUCH",    sv_puch, y_puch,
     puch_df.Case_ID.values),
    ("TCGA",    sv_tcga, y_tcga,
     tcga_df.Case_ID.values),
]
for cohort, sv, yt, ids in cohorts_data:
    probs = model.predict_proba(
        X_puth_s if cohort == "PUTH" else
        X_jsph_s if cohort == "JSPH-RN" else
        X_puch_s if cohort == "PUCH" else
        X_tcga_s
    )[:, 1]
    for i in range(len(yt)):
        row = {"CaseID": ids[i], "Cohort": cohort,
               "RSI": int(yt[i]), "PredProb": round(float(probs[i]), 4)}
        for j, feat in enumerate(feat_cols):
            row[f"SHAP_{feat}"] = round(float(sv[i, j]), 5)
        rows.append(row)

shap_csv = pd.DataFrame(rows)
shap_csv.to_csv(OUT_DIR / "shap_values_all.csv", index=False, encoding="utf-8-sig")
print(f"  Saved: shap_values_all.csv  ({len(shap_csv)} rows x {len(feat_cols)} features)")


# ─────────────────────────────────────────────────────────────────────────────
# Print SHAP-based feature importance summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SHAP Feature Importance Summary (Mean |SHAP|, all cohorts)")
print(f"{'='*65}")
print(f"  {'Rank':>4}  {'Feature':<45}  {'Type':>11}  {'MeanAbsSHAP':>11}  {'Coef':>8}")
print(f"  {'-'*4}  {'-'*45}  {'-'*11}  {'-'*11}  {'-'*8}")
for rank, fi in enumerate(order[:15], 1):
    fname = feat_display[fi]
    ftype = feat_types[fi]
    mabs  = mean_abs[fi]
    coef  = coefs[fi]
    print(f"  {rank:>4}  {fname:<45}  {ftype:>11}  {mabs:>11.4f}  {coef:>+8.4f}")

print(f"\n{'='*65}")
print("Step 6 Complete.")
print(f"{'='*65}")
print(f"  Output: {OUT_DIR}")
print(f"  shap_summary_beeswarm.png    (global beeswarm, training set)")
print(f"  shap_importance_bar.png      (mean |SHAP|, all cohorts)")
print(f"  shap_dependence_top5.png     (dependence plots, top 5 features)")
print(f"  shap_cohort_comparison.png   (per-cohort heatmap)")
print(f"  shap_waterfall_rsi_pos.png   (high-confidence RSI+ example)")
print(f"  shap_waterfall_rsi_neg.png   (high-confidence RSI- example)")
print(f"  shap_values_all.csv          (raw SHAP values, all cases)")
