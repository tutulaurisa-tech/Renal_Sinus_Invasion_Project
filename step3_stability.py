# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 3 (Stability Edition): Bootstrap Stability Selection + Elastic Net
======================================================================================
Strategy:
  - NO KS filter  (eliminated: caused instability, removed biologically-relevant features)
  - Training set : PUTH (346) + JSPH-RN (230) = 576 cases (multi-center training)
  - Test sets    : PUCH (90, internal), TCGA (170, external)
  - Standardize  : StandardScaler fit on combined training set
  - Selection    : Bootstrap Stability Selection (200 iterations, stratified by center+RSI)
                   Elastic Net (l1_ratio=0.5) fitted in each bootstrap subsample
                   Keep features with selection frequency > 50%
  - Final model  : Elastic Net with optimal C (re-tuned on stable features)
  - Evaluation   : 5-fold stratified CV on training set (per-center breakdown)
                   External prediction on PUCH / TCGA

Rationale vs. V2 LASSO:
  - Multi-center training captures center-invariant patterns (higher generalization)
  - Elastic Net is more stable than LASSO when features are correlated
  - Bootstrap stability selection ensures only robustly-selected features are retained
  - Combined training set (576 cases) provides more power for feature selection

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step3_stability.py

Outputs (in 5_Modeling/v2_stability/ and 5_Results/v2_stability/):
  selected_features_stability.csv   — stable features with selection frequency + coef
  scaler_params_stability.csv       — mean/std fit on combined training set
  predictions_PUTH_stability.csv    — CV predictions (training, PUTH center)
  predictions_JSPH_RN_stability.csv — CV predictions (training, JSPH-RN center)
  predictions_PUCH_stability.csv    — external test predictions
  predictions_TCGA_stability.csv    — external test predictions
  stability_plot.png                — feature selection frequency bar chart
  performance_summary.txt           — AUC comparison table
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR  = BASE / "3_Extracted_Features"
OUT_MOD   = BASE / "5_Modeling"  / "v2_stability"
OUT_RES   = BASE / "5_Results"   / "v2_stability"
OUT_MOD.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
BOOT_N       = 200    # bootstrap iterations
BOOT_FRAC    = 0.5    # subsample fraction per iteration (Meinshausen & Bühlmann)
STAB_THRESH  = 0.50   # minimum selection frequency to retain feature
L1_RATIO     = 0.5    # Elastic Net mixing parameter (0=Ridge, 1=LASSO)
SEED         = 42
np.random.seed(SEED)

IMG_TYPES = ["original", "logarithm", "exponential", "square",
             "squareroot", "gradient", "wavelet"]

TYPE_COLORS = {
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


# ─────────────────────────────────────────────────────────────────────────────
# AUC helpers
# ─────────────────────────────────────────────────────────────────────────────
def auc_score(yt, yp):
    """Mann–Whitney AUC (handles ties)."""
    pos = yp[yt == 1]; neg = yp[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return u / (len(pos) * len(neg))


def boot_ci(yt, yp, n=1000, seed=42):
    """Bootstrap 95% CI for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []; N = len(yt)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(np.unique(yt[idx])) < 2:
            continue
        aucs.append(auc_score(yt[idx], yp[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def thresh_metrics(yt, yp, thr=0.5):
    pred = (yp >= thr).astype(int)
    tp = ((pred == 1) & (yt == 1)).sum()
    tn = ((pred == 0) & (yt == 0)).sum()
    fp = ((pred == 1) & (yt == 0)).sum()
    fn = ((pred == 0) & (yt == 1)).sum()
    sen = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spe = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(yt)
    return sen, spe, acc


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("V2 Pipeline — Step 3 Stability: Bootstrap + Elastic Net")
print("=" * 65)

all_df  = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df = pd.read_csv(FEAT_DIR / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")

SKIP_EXACT = {"Case_ID", "PatientID", "Center", "RSI", "Cohort"}
SKIP_PFX   = "diagnostics_"
feat_cols = [c for c in all_df.columns
             if c not in SKIP_EXACT and not c.startswith(SKIP_PFX)]
print(f"\nFeature space: {len(feat_cols)}")

puth_df = all_df[all_df.Center == "PUTH"].reset_index(drop=True)
puch_df = all_df[all_df.Center == "PUCH"].reset_index(drop=True)
tcga_df = all_df[all_df.Center == "TCGA"].reset_index(drop=True)

# Verify JSPH-RN feature columns match
jsph_feat_cols = [c for c in jsph_df.columns
                  if c not in SKIP_EXACT and not c.startswith(SKIP_PFX)]
missing_in_jsph = set(feat_cols) - set(jsph_feat_cols)
if missing_in_jsph:
    print(f"  WARNING: {len(missing_in_jsph)} features missing from JSPH-RN — "
          f"using intersection ({len(feat_cols) - len(missing_in_jsph)} features)")
    feat_cols = [c for c in feat_cols if c in jsph_feat_cols]

X_puth = puth_df[feat_cols].values.astype(float)
X_puch = puch_df[feat_cols].values.astype(float)
X_tcga = tcga_df[feat_cols].values.astype(float)
X_jsph = jsph_df[feat_cols].values.astype(float)

y_puth = puth_df.RSI.values.astype(int)
y_puch = puch_df.RSI.values.astype(int)
y_tcga = tcga_df.RSI.values.astype(int)
y_jsph = jsph_df.RSI.values.astype(int)

print(f"\nCenter summary:")
print(f"  PUTH    (train) : {len(puth_df):3d} cases  RSI+={y_puth.sum():2d}  "
      f"({100*y_puth.mean():.1f}%)")
print(f"  JSPH-RN (train) : {len(jsph_df):3d} cases  RSI+={y_jsph.sum():2d}  "
      f"({100*y_jsph.mean():.1f}%)")
print(f"  PUCH    (test)  : {len(puch_df):3d} cases  RSI+={y_puch.sum():2d}  "
      f"({100*y_puch.mean():.1f}%)")
print(f"  TCGA    (test)  : {len(tcga_df):3d} cases  RSI+={y_tcga.sum():2d}  "
      f"({100*y_tcga.mean():.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Build combined training set
# ─────────────────────────────────────────────────────────────────────────────
# Center labels for stratified bootstrap: PUTH=0, JSPH-RN=1
center_puth = np.zeros(len(puth_df), dtype=int)
center_jsph = np.ones(len(jsph_df), dtype=int)

X_train = np.vstack([X_puth, X_jsph])
y_train = np.concatenate([y_puth, y_jsph])
center_train = np.concatenate([center_puth, center_jsph])

n_train = len(y_train)
print(f"\nCombined training set: {n_train} cases  "
      f"RSI+={y_train.sum()}  ({100*y_train.mean():.1f}%)")

# Stratify label: center * 2 + RSI → 4 groups for stratified CV / bootstrap
strat_train = center_train * 2 + y_train


# ─────────────────────────────────────────────────────────────────────────────
# Standardize (fit on training set only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Standardizing on combined training set ---")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_puth_s  = scaler.transform(X_puth)
X_jsph_s  = scaler.transform(X_jsph)
X_puch_s  = scaler.transform(X_puch)
X_tcga_s  = scaler.transform(X_tcga)

# Save scaler
pd.DataFrame({
    "Feature": feat_cols,
    "mean":    scaler.mean_,
    "std":     scaler.scale_,
}).to_csv(OUT_MOD / "scaler_params_stability.csv",
          index=False, encoding="utf-8-sig")
print(f"  Scaler saved ({len(feat_cols)} features)")


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Find optimal C via 5-fold CV on full training set
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step A: Elastic Net CV to find optimal C ---")
enet_cv = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 60),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    penalty="elasticnet",
    solver="saga",
    l1_ratios=[L1_RATIO],
    class_weight="balanced",
    scoring="roc_auc",
    max_iter=5000,
    random_state=SEED,
    n_jobs=-1,
)
enet_cv.fit(X_train_s, y_train)
best_C = float(enet_cv.C_[0])
print(f"  Best C = {best_C:.5f}  (l1_ratio={L1_RATIO})")


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Bootstrap Stability Selection
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Step B: Bootstrap Stability Selection "
      f"(n={BOOT_N}, frac={BOOT_FRAC}, C={best_C:.4f}) ---")

n_feats = len(feat_cols)
selection_counts = np.zeros(n_feats, dtype=float)
rng = np.random.RandomState(SEED)

for b in range(BOOT_N):
    # Stratified subsample: maintain center×RSI proportions
    idx_boot = []
    for grp in np.unique(strat_train):
        grp_idx = np.where(strat_train == grp)[0]
        n_sample = max(2, int(np.ceil(len(grp_idx) * BOOT_FRAC)))
        sampled  = rng.choice(grp_idx, size=n_sample, replace=False)
        idx_boot.extend(sampled.tolist())

    idx_boot = np.array(idx_boot)
    X_b = X_train_s[idx_boot]
    y_b = y_train[idx_boot]

    # Skip if only one class present (shouldn't happen with stratification)
    if len(np.unique(y_b)) < 2:
        continue

    m = LogisticRegression(
        C=best_C,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=L1_RATIO,
        class_weight="balanced",
        max_iter=5000,
        random_state=b,
    )
    m.fit(X_b, y_b)
    selection_counts += (m.coef_[0] != 0).astype(float)

    if (b + 1) % 50 == 0:
        n_sel_now = int((selection_counts / (b + 1) >= STAB_THRESH).sum())
        print(f"  [{b+1:3d}/{BOOT_N}]  features currently stable: {n_sel_now}")

selection_freq = selection_counts / BOOT_N
n_stable = int((selection_freq >= STAB_THRESH).sum())
print(f"\n  Bootstrap complete.")
print(f"  Features with freq ≥ {STAB_THRESH:.0%}: {n_stable} / {n_feats}")
print(f"  Freq distribution: "
      f">90%={int((selection_freq>0.9).sum())}  "
      f"70-90%={int(((selection_freq>0.7)&(selection_freq<=0.9)).sum())}  "
      f"50-70%={int(((selection_freq>=0.5)&(selection_freq<=0.7)).sum())}  "
      f"<50%={int((selection_freq<0.5).sum())}")

# Extract stable feature indices
stable_mask = selection_freq >= STAB_THRESH
stable_feats = [f for f, s in zip(feat_cols, stable_mask) if s]
stable_freq  = selection_freq[stable_mask]

# Image type breakdown
type_counts = Counter(img_type_of(f) for f in stable_feats)
print(f"\n  Stable features by image type:")
for t in IMG_TYPES:
    n = type_counts.get(t, 0)
    if n > 0:
        print(f"    {t:12s}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Final Elastic Net model on stable features
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Step C: Final model on {n_stable} stable features ---")

# Re-tune C on stable feature subset for best final model
feat_mask_arr = np.array(feat_cols)
X_train_sel = X_train_s[:, stable_mask]
X_puth_sel  = X_puth_s[:, stable_mask]
X_jsph_sel  = X_jsph_s[:, stable_mask]
X_puch_sel  = X_puch_s[:, stable_mask]
X_tcga_sel  = X_tcga_s[:, stable_mask]

enet_final_cv = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 60),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    penalty="elasticnet",
    solver="saga",
    l1_ratios=[L1_RATIO],
    class_weight="balanced",
    scoring="roc_auc",
    max_iter=5000,
    random_state=SEED,
    n_jobs=-1,
)
enet_final_cv.fit(X_train_sel, y_train)
best_C_final = float(enet_final_cv.C_[0])
print(f"  Re-tuned C on stable features: {best_C_final:.5f}")

enet_final = LogisticRegression(
    C=best_C_final,
    penalty="elasticnet",
    solver="saga",
    l1_ratio=L1_RATIO,
    class_weight="balanced",
    max_iter=5000,
    random_state=SEED,
)
enet_final.fit(X_train_sel, y_train)

# Non-zero features in final model
final_coefs = enet_final.coef_[0]
nonzero_mask = final_coefs != 0
n_nonzero = nonzero_mask.sum()
print(f"  Non-zero coefs in final model: {n_nonzero} / {n_stable}")

# Save selected features with stability scores + coefficients
sel_df = pd.DataFrame({
    "Feature":        stable_feats,
    "SelectionFreq":  stable_freq,
    "FinalCoef":      final_coefs,
    "AbsCoef":        np.abs(final_coefs),
    "ImageType":      [img_type_of(f) for f in stable_feats],
    "Active":         nonzero_mask.astype(int),
}).sort_values("SelectionFreq", ascending=False).reset_index(drop=True)
sel_df.to_csv(OUT_MOD / "selected_features_stability.csv",
              index=False, encoding="utf-8-sig")
print(f"\n  Top 15 features by stability frequency:")
print(sel_df[["Feature", "SelectionFreq", "FinalCoef", "ImageType"]].head(15).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Step D: Cross-validated predictions on training set (per-center breakdown)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step D: 5-fold CV on combined training set ---")

cv_probs_train = cross_val_predict(
    LogisticRegression(
        C=best_C_final,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=L1_RATIO,
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
    ),
    X_train_sel, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    method="predict_proba",
)[:, 1]

# Overall training CV AUC
auc_train_cv = auc_score(y_train, cv_probs_train)
lo_tr, hi_tr = boot_ci(y_train, cv_probs_train)
sen_tr, spe_tr, acc_tr = thresh_metrics(y_train, cv_probs_train)
print(f"\n  Combined training set CV:")
print(f"    AUC = {auc_train_cv:.3f}  [{lo_tr:.3f}-{hi_tr:.3f}]")
print(f"    SEN={sen_tr:.3f}  SPE={spe_tr:.3f}  ACC={acc_tr:.3f}")

# Per-center breakdown
n_puth = len(y_puth)
cv_puth = cv_probs_train[:n_puth]
cv_jsph = cv_probs_train[n_puth:]

auc_puth_cv = auc_score(y_puth, cv_puth)
lo_pu, hi_pu = boot_ci(y_puth, cv_puth)
sen_pu, spe_pu, acc_pu = thresh_metrics(y_puth, cv_puth)
print(f"\n  PUTH (within CV):")
print(f"    AUC = {auc_puth_cv:.3f}  [{lo_pu:.3f}-{hi_pu:.3f}]")
print(f"    SEN={sen_pu:.3f}  SPE={spe_pu:.3f}  ACC={acc_pu:.3f}")

auc_jsph_cv = auc_score(y_jsph, cv_jsph)
lo_js, hi_js = boot_ci(y_jsph, cv_jsph)
sen_js, spe_js, acc_js = thresh_metrics(y_jsph, cv_jsph)
print(f"\n  JSPH-RN (within CV):")
print(f"    AUC = {auc_jsph_cv:.3f}  [{lo_js:.3f}-{hi_js:.3f}]")
print(f"    SEN={sen_js:.3f}  SPE={spe_js:.3f}  ACC={acc_js:.3f}")

# Save CV predictions for PUTH
pd.DataFrame({
    "PatientID": puth_df.Case_ID,
    "RSI":       y_puth,
    "RSI_Prob":  cv_puth,
    "RSI_Pred":  (cv_puth >= 0.5).astype(int),
}).to_csv(OUT_MOD / "predictions_PUTH_stability.csv",
          index=False, encoding="utf-8-sig")

# Save CV predictions for JSPH-RN
jsph_id_col = "Case_ID" if "Case_ID" in jsph_df.columns else "PatientID"
pd.DataFrame({
    "PatientID": jsph_df[jsph_id_col],
    "RSI":       y_jsph,
    "RSI_Prob":  cv_jsph,
    "RSI_Pred":  (cv_jsph >= 0.5).astype(int),
}).to_csv(OUT_MOD / "predictions_JSPH_RN_stability.csv",
          index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# Step E: External predictions (model trained on full training set)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step E: External predictions (full model) ---")

perf_rows = []

for name, X_ext, y_ext, id_series in [
    ("PUCH", X_puch_sel, y_puch, puch_df.Case_ID),
    ("TCGA", X_tcga_sel, y_tcga, tcga_df.Case_ID),
]:
    probs = enet_final.predict_proba(X_ext)[:, 1]
    auc   = auc_score(y_ext, probs)
    lo, hi = boot_ci(y_ext, probs)
    sen, spe, acc = thresh_metrics(y_ext, probs)
    print(f"  {name:8s}  AUC={auc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"SEN={sen:.3f}  SPE={spe:.3f}  ACC={acc:.3f}")
    perf_rows.append({
        "Cohort": name, "AUC": round(auc, 3),
        "CI_lo": round(lo, 3), "CI_hi": round(hi, 3),
        "SEN": round(sen, 3), "SPE": round(spe, 3), "ACC": round(acc, 3),
    })
    pd.DataFrame({
        "PatientID": id_series,
        "RSI":       y_ext,
        "RSI_Prob":  probs,
        "RSI_Pred":  (probs >= 0.5).astype(int),
    }).to_csv(OUT_MOD / f"predictions_{name}_stability.csv",
              index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# Step F: Compare with V2 LASSO results
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step F: Performance comparison ---")

v2_lasso_res = {
    "PUTH (Train)": 0.793, "PUCH (Int.Test)": 0.688,
    "TCGA (Ext.Val)": 0.733, "JSPH-RN (Ext.Val)": 0.698,
}

# Stability model results summary
stab_results = {
    "PUTH (CV)":      (auc_puth_cv, lo_pu, hi_pu),
    "JSPH-RN (CV)":   (auc_jsph_cv, lo_js, hi_js),
}
for row in perf_rows:
    key = f"{row['Cohort']} (Ext.)"
    stab_results[key] = (row["AUC"], row["CI_lo"], row["CI_hi"])

summary_lines = [
    "=" * 70,
    "Performance Summary: Stability Model vs V2 LASSO",
    "=" * 70,
    f"  Training set  : PUTH ({len(y_puth)}) + JSPH-RN ({len(y_jsph)}) = {n_train} cases",
    f"  Stable features selected: {n_stable}",
    f"  Final non-zero features : {n_nonzero}",
    f"  Bootstrap iterations    : {BOOT_N}  (frac={BOOT_FRAC})",
    f"  Stability threshold     : {STAB_THRESH:.0%}",
    "",
    f"  {'Cohort':<22s}  {'Stability AUC':<20s}  {'V2 LASSO AUC':<15s}  {'Delta':>7s}",
    f"  {'-'*22}  {'-'*20}  {'-'*15}  {'-'*7}",
]

mapping = [
    ("PUTH (CV)",     "PUTH (Train)"),
    ("JSPH-RN (CV)",  "JSPH-RN (Ext.Val)"),
    ("PUCH (Ext.)",   "PUCH (Int.Test)"),
    ("TCGA (Ext.)",   "TCGA (Ext.Val)"),
]
for stab_key, lasso_key in mapping:
    if stab_key not in stab_results:
        continue
    auc_s, lo_s, hi_s = stab_results[stab_key]
    auc_l = v2_lasso_res.get(lasso_key, float("nan"))
    delta  = auc_s - auc_l if not np.isnan(auc_l) else float("nan")
    delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "N/A"
    auc_str   = f"{auc_s:.3f} ({lo_s:.3f}-{hi_s:.3f})"
    lasso_str = f"{auc_l:.3f}" if not np.isnan(auc_l) else "N/A"
    summary_lines.append(
        f"  {stab_key:<22s}  {auc_str:<20s}  {lasso_str:<15s}  {delta_str:>7s}"
    )
    print(f"  {stab_key:<22s}  Stability={auc_s:.3f}  LASSO={auc_l:.3f}  Δ={delta:+.3f}")

summary_lines += [
    "",
    f"  V2 LASSO feature count: see selected_features_v2.csv",
    f"  Stability feature count: {n_stable} (bootstrap stable), "
    f"{n_nonzero} (non-zero in final model)",
]
summary_text = "\n".join(summary_lines)
print()
print(summary_text)
with open(OUT_RES / "performance_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)


# ─────────────────────────────────────────────────────────────────────────────
# Step G: Stability frequency plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step G: Stability frequency plot ---")

# Plot top-N stable features by selection frequency
TOP_N_PLOT = min(30, len(sel_df))
plot_df = sel_df.head(TOP_N_PLOT)

fig, ax = plt.subplots(figsize=(11, max(6, TOP_N_PLOT * 0.42)))
colors = [TYPE_COLORS.get(t, "gray") for t in plot_df.ImageType]
ax.barh(range(TOP_N_PLOT), plot_df.SelectionFreq.values, color=colors, alpha=0.85)
ax.axvline(STAB_THRESH, color="red", linestyle="--", linewidth=1.2,
           label=f"Stability threshold ({STAB_THRESH:.0%})")
ax.set_yticks(range(TOP_N_PLOT))
ax.set_yticklabels(
    [f[:58] + "…" if len(f) > 58 else f for f in plot_df.Feature],
    fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Bootstrap Selection Frequency")
ax.set_xlim(0, 1.05)
ax.set_title(
    f"Bootstrap Stability Selection  (n_stable={n_stable}, non-zero={n_nonzero})\n"
    f"Training: PUTH+JSPH-RN ({n_train} cases)  |  Elastic Net l1_ratio={L1_RATIO}  |  "
    f"Bootstrap={BOOT_N}×{int(BOOT_FRAC*100)}%",
    fontweight="bold", fontsize=10)

legend_patches = [mpatches.Patch(color=c, label=t)
                  for t, c in TYPE_COLORS.items()
                  if t in type_counts]
legend_patches.append(
    plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({STAB_THRESH:.0%})"))
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()

out_fig = OUT_RES / "stability_plot.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_fig.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Compare overlapping features with V2 LASSO
# ─────────────────────────────────────────────────────────────────────────────
v2_sel_path = BASE / "5_Modeling" / "v2" / "selected_features_v2.csv"
if v2_sel_path.exists():
    v2_feats = set(pd.read_csv(v2_sel_path)["Feature"].tolist())
    stab_feats_set = set(stable_feats)
    overlap = stab_feats_set & v2_feats
    print(f"\n--- Feature overlap with V2 LASSO ---")
    print(f"  V2 LASSO: {len(v2_feats)}  |  Stability: {len(stab_feats_set)}")
    print(f"  Overlap: {len(overlap)}")
    if overlap:
        print(f"  Shared features: {sorted(overlap)[:10]}")
        overlap_df = sel_df[sel_df.Feature.isin(overlap)][
            ["Feature", "SelectionFreq", "FinalCoef", "ImageType"]]
        print(overlap_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Step 3 Stability Complete.")
print(f"{'='*65}")
print(f"  Output directory: {OUT_MOD}")
print(f"  selected_features_stability.csv  ({n_stable} bootstrap-stable features)")
print(f"  scaler_params_stability.csv      (fit on PUTH+JSPH-RN)")
print(f"  predictions_PUTH_stability.csv   (CV, AUC={auc_puth_cv:.3f})")
print(f"  predictions_JSPH_RN_stability.csv (CV, AUC={auc_jsph_cv:.3f})")
for row in perf_rows:
    print(f"  predictions_{row['Cohort']}_stability.csv  "
          f"(AUC={row['AUC']:.3f} [{row['CI_lo']:.3f}-{row['CI_hi']:.3f}])")
print(f"\n  Result plots: {OUT_RES}")
print(f"  stability_plot.png")
print(f"  performance_summary.txt")
print(f"\nNext: python step4_gnn_train.py  (update DATA_DIR to v2_stability)")
