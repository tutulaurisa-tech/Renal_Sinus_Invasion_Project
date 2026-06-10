# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 3: Feature Selection + LASSO Modeling  (V1.1)
=========================================================
0.5 KS cross-center stability filter (on raw features, before normalization)
1.  Normalize with PUTH mean/std
2.  LASSO-LR with 5-fold stratified CV on PUTH
3.  Report selected features + image type breakdown
4.  Compare selected features with v1
5.  Generate predictions for all 4 centers
6.  Save: selected_features_v2.csv, predictions_*_v2.csv

Change log:
  V1.1  2025-05  Added Step0.5 KS stability filter (KS_THRESHOLD=0.20)
                 Filtering done on raw features before PUTH normalization

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step3_feature_selection.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

BASE      = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR  = BASE / "3_Extracted_Features"
OUT_MOD   = BASE / "5_Modeling" / "v2"
OUT_RES   = BASE / "5_Results" / "v2"
OUT_MOD.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

IMG_TYPES = ["original", "logarithm", "exponential", "square",
             "squareroot", "gradient", "wavelet"]
KS_THRESHOLD = 0.25    # 跨中心稳定性阈值（原始特征空间 KS 统计量上限）
# 注意：KS 过滤只对 PUCH / TCGA，不包含 JSPH-RN
# JSPH-RN 的 RSI+ 率(9.5%)与训练中心(~20%)差异显著，是病例构成不同，
# 非扫描仪批次效应，将其纳入过滤会错误删掉大量有判别力的特征。
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("V2 Pipeline — Step 3: Feature Selection + LASSO (V1.1)")
print("=" * 60)


# ── Metrics ───────────────────────────────────────────────────────────────
def auc_score(yt, yp):
    pos = yp[yt == 1]; neg = yp[yt == 0]
    u   = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return u / (len(pos) * len(neg))


def boot_ci(yt, yp, n=1000, seed=42):
    rng = np.random.RandomState(seed); aucs = []; N = len(yt)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(np.unique(yt[idx])) < 2: continue
        aucs.append(auc_score(yt[idx], yp[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ── Load data ─────────────────────────────────────────────────────────────
all_df  = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df = pd.read_csv(FEAT_DIR / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")

SKIP_EXACT = {"Case_ID", "PatientID", "Center", "RSI", "Cohort"}
SKIP_PFX   = "diagnostics_"

def img_type_of(feat):
    if feat.startswith("wavelet"):
        return "wavelet"
    for t in IMG_TYPES:
        if feat.startswith(t + "_"):
            return t
    return "unknown"

feat_cols = [c for c in all_df.columns
             if c not in SKIP_EXACT and not c.startswith(SKIP_PFX)]
print(f"Feature space: {len(feat_cols)}")

puth_df = all_df[all_df.Center == "PUTH"].reset_index(drop=True)
puch_df = all_df[all_df.Center == "PUCH"].reset_index(drop=True)
tcga_df = all_df[all_df.Center == "TCGA"].reset_index(drop=True)

X_puth = puth_df[feat_cols].values.astype(float)
X_puch = puch_df[feat_cols].values.astype(float)
X_tcga = tcga_df[feat_cols].values.astype(float)
X_jsph = jsph_df[feat_cols].values.astype(float)

y_puth = puth_df.RSI.values.astype(int)
y_puch = puch_df.RSI.values.astype(int)
y_tcga = tcga_df.RSI.values.astype(int)
y_jsph = jsph_df.RSI.values.astype(int)

print(f"PUTH: {len(puth_df)}  RSI+={y_puth.sum()}")
print(f"PUCH: {len(puch_df)}  RSI+={y_puch.sum()}")
print(f"TCGA: {len(tcga_df)}  RSI+={y_tcga.sum()}")
print(f"JSPH-RN: {len(jsph_df)}  RSI+={y_jsph.sum()}")

# ══════════════════════════════════════════════════════════════════════════
# 0.5  Cross-center KS stability filter (raw features, before normalization)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- Step 0.5: Cross-center KS stability filter (threshold={KS_THRESHOLD}) ---")
n_total = len(feat_cols)
stable_mask = np.ones(n_total, dtype=bool)

# 以 PUTH 为基准，只与 PUCH / TCGA 做 KS 检验
# JSPH-RN 不参与过滤：其 RSI+ 率(9.5%)远低于训练中心(~20%)，
# 病例构成差异会错误淘汰大量有判别力的特征，保留为纯外部验证集。
for i in range(n_total):
    for ref_name, X_ref in [("PUCH", X_puch), ("TCGA", X_tcga)]:
        v_train = X_puth[:, i][~np.isnan(X_puth[:, i])]
        v_ref   = X_ref[:, i][~np.isnan(X_ref[:, i])]
        if len(v_train) < 5 or len(v_ref) < 5:
            continue
        ks, _ = ks_2samp(v_train, v_ref)
        if ks > KS_THRESHOLD:
            stable_mask[i] = False
            break

n_dropped = int((~stable_mask).sum())
feat_cols_arr = np.array(feat_cols)
feat_cols = feat_cols_arr[stable_mask].tolist()
X_puth = X_puth[:, stable_mask]
X_puch = X_puch[:, stable_mask]
X_tcga = X_tcga[:, stable_mask]
X_jsph = X_jsph[:, stable_mask]

print(f"  Dropped {n_dropped} unstable features (KS > {KS_THRESHOLD} vs PUCH or TCGA)")
print(f"  Remaining: {len(feat_cols)} / {n_total}")

# 保存被剔除的特征列表（诊断用）
pd.DataFrame({
    "Feature": feat_cols_arr[~stable_mask].tolist()
}).to_csv(OUT_MOD / "ks_dropped_features_v2.csv", index=False, encoding="utf-8-sig")

# ── Normalize with PUTH stats ─────────────────────────────────────────────
mu = X_puth.mean(0); sd = X_puth.std(0) + 1e-8
Xp_s = (X_puth - mu) / sd
Xu_s = (X_puch - mu) / sd
Xt_s = (X_tcga - mu) / sd
Xj_s = (X_jsph - mu) / sd
print("\nNormalization: PUTH mean/std")

# Save scaler params
scaler_df = pd.DataFrame({"Feature": feat_cols, "mean": mu, "std": sd})
scaler_df.to_csv(OUT_MOD / "scaler_params_v2.csv", index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════════
# 1.  LASSO feature selection via LassoCV on PUTH
# ══════════════════════════════════════════════════════════════════════════
print("\n--- LASSO Feature Selection (PUTH, 5-fold CV) ---")

lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 2, 60),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    penalty="l1", solver="liblinear",
    class_weight="balanced",
    scoring="roc_auc",
    max_iter=3000,
    random_state=SEED,
)
lasso_cv.fit(Xp_s, y_puth)
best_C = lasso_cv.C_[0]
print(f"  Best C: {best_C:.5f}")

# Final model with best C
lasso = LogisticRegression(
    C=best_C, penalty="l1", solver="liblinear",
    class_weight="balanced", max_iter=3000, random_state=SEED,
)
lasso.fit(Xp_s, y_puth)

# Selected features (non-zero coefficients)
coef = lasso.coef_[0]
sel_mask = coef != 0
sel_feats = [f for f, s in zip(feat_cols, sel_mask) if s]
sel_coefs = coef[sel_mask]

print(f"\n  Selected features: {len(sel_feats)} / {len(feat_cols)}")

# Feature type breakdown
from collections import Counter
type_counts = Counter(img_type_of(f) for f in sel_feats)
print("  By image type:")
for t in IMG_TYPES:
    n = type_counts.get(t, 0)
    if n > 0:
        print(f"    {t:12s}: {n}")

# Save selected features
sel_df = pd.DataFrame({
    "Feature": sel_feats,
    "Coefficient": sel_coefs,
    "ImageType": [img_type_of(f) for f in sel_feats],
    "AbsCoef": np.abs(sel_coefs),
}).sort_values("AbsCoef", ascending=False).reset_index(drop=True)
sel_df.to_csv(OUT_MOD / "selected_features_v2.csv",
              index=False, encoding="utf-8-sig")
print(f"\n  Top 10 features:")
print(sel_df[["Feature", "Coefficient", "ImageType"]].head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# 2.  Compare with v1 selected features
# ══════════════════════════════════════════════════════════════════════════
v1_sel_path = BASE / "5_Modeling" / "selected_features.csv"
if v1_sel_path.exists():
    v1_feats = set(pd.read_csv(v1_sel_path)["Feature"].tolist())
    v2_feats = set(sel_feats)
    overlap  = v1_feats & v2_feats
    only_v1  = v1_feats - v2_feats
    only_v2  = v2_feats - v1_feats
    print(f"\n  V1 vs V2 comparison:")
    print(f"    V1 had {len(v1_feats)} features, V2 has {len(v2_feats)} features")
    print(f"    Overlap (same in both): {len(overlap)}")
    if overlap:
        print(f"      {sorted(overlap)[:5]}")
    print(f"    Only in V1: {len(only_v1)}")
    print(f"    Only in V2: {len(only_v2)}")


# ══════════════════════════════════════════════════════════════════════════
# 3.  PUTH cross-validated predictions (proper internal validation)
# ══════════════════════════════════════════════════════════════════════════
print("\n--- Cross-validated predictions on PUTH ---")

# Use only selected features for CV
Xp_sel = Xp_s[:, sel_mask]
cv_probs = cross_val_predict(
    LogisticRegression(C=best_C, penalty="l1", solver="liblinear",
                       class_weight="balanced", max_iter=3000, random_state=SEED),
    Xp_sel, y_puth,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    method="predict_proba",
)[:, 1]

auc_cv = auc_score(y_puth, cv_probs)
lo, hi = boot_ci(y_puth, cv_probs)
print(f"  PUTH 5-fold CV AUC: {auc_cv:.3f} [{lo:.3f}-{hi:.3f}]")

puth_pred_df = pd.DataFrame({
    "PatientID": puth_df.Case_ID,
    "RSI": y_puth,
    "RSI_Prob": cv_probs,
    "RSI_Pred": (cv_probs >= 0.5).astype(int),
})
puth_pred_df.to_csv(OUT_MOD / "predictions_PUTH_v2.csv",
                    index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════════
# 4.  Retrain on full PUTH → predict PUCH / TCGA / JSPH-RN
# ══════════════════════════════════════════════════════════════════════════
print("\n--- External predictions ---")

# Retrain on full PUTH with selected features
Xp_sel = Xp_s[:, sel_mask]
Xu_sel = Xu_s[:, sel_mask]
Xt_sel = Xt_s[:, sel_mask]
Xj_sel = Xj_s[:, sel_mask]

lasso_final = LogisticRegression(
    C=best_C, penalty="l1", solver="liblinear",
    class_weight="balanced", max_iter=3000, random_state=SEED,
)
lasso_final.fit(Xp_sel, y_puth)

for name, X, y, df_src in [
    ("PUCH", Xu_sel, y_puch, puch_df),
    ("TCGA", Xt_sel, y_tcga, tcga_df),
    ("JSPH_RN", Xj_sel, y_jsph, jsph_df),
]:
    probs = lasso_final.predict_proba(X)[:, 1]
    auc   = auc_score(y, probs)
    lo, hi = boot_ci(y, probs)
    print(f"  {name:8s}  AUC={auc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"RSI+={y.sum()}")

    id_col = "Case_ID" if "Case_ID" in df_src.columns else "PatientID"
    pd.DataFrame({
        "PatientID": df_src[id_col],
        "RSI": y,
        "RSI_Prob": probs,
        "RSI_Pred": (probs >= 0.5).astype(int),
    }).to_csv(OUT_MOD / f"predictions_{name}_v2.csv",
              index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════════
# 5.  Feature importance bar chart
# ══════════════════════════════════════════════════════════════════════════
TYPE_COLORS = {
    "original": "#2196F3", "wavelet": "#E91E63",
    "gradient": "#FF9800", "logarithm": "#4CAF50",
    "exponential": "#9C27B0", "square": "#00BCD4",
    "squareroot": "#FF5722",
}

top_n = min(20, len(sel_df))
plot_df = sel_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
colors = [TYPE_COLORS.get(t, "gray") for t in plot_df.ImageType]
bars = ax.barh(range(top_n), plot_df.AbsCoef.values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(
    [f[:55] + "…" if len(f) > 55 else f for f in plot_df.Feature],
    fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("|LASSO Coefficient|")
ax.set_title(f"V2 LASSO Selected Features (n={len(sel_feats)})\nTop {top_n} by |Coefficient|",
             fontweight="bold")
legend_patches = [mpatches.Patch(color=c, label=t)
                  for t, c in TYPE_COLORS.items()
                  if t in type_counts]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_RES / "lasso_features_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: lasso_features_v2.png")


# ══════════════════════════════════════════════════════════════════════════
# 6.  Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Step 3 Complete.")
print(f"{'='*60}")
print(f"  selected_features_v2.csv  ({len(sel_feats)} features)")
print(f"  predictions_PUTH_v2.csv   AUC={auc_cv:.3f} (5-fold CV)")
print(f"  scaler_params_v2.csv")
print(f"\nNext: python step4_gnn_train.py")
