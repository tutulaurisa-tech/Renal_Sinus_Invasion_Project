# -*- coding: utf-8 -*-
"""
V2 Pipeline - Step 6b: SHAP Supplemental Figures
=================================================
Replaces the two less informative plots from step6_shap.py:

  1. shap_sphericity_stratification.png
       Sphericity quartile stratification:
       - Left panel:  violin plot of Sphericity by RSI status (training set)
       - Middle panel: RSI+ rate (%) per Sphericity quartile (bar, with CI)
       - Right panel:  mean |SHAP| contribution per quartile (all cohorts)

  2. shap_waterfall_rsi_neg_typical.png
       Waterfall for a TYPICAL RSI- case (prob 0.02-0.10), replacing the
       extreme prob=0.000 case which is not representative.

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step6b_shap_supplement.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

try:
    import shap
except ImportError:
    print("ERROR: pip install shap")
    raise

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

# ─────────────────────────────────────────────────────────────────────────────
# Paths (identical to step6)
# ─────────────────────────────────────────────────────────────────────────────
BASE    = Path(r"D:\RSI_Project_Workspace")
FEAT    = BASE / "3_Extracted_Features"
MOD_DIR = BASE / "5_Modeling" / "v2_stability"
OUT_DIR = BASE / "5_Results"  / "v2_stability" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
L1_RATIO = 0.5
np.random.seed(SEED)

print("=" * 65)
print("V2 Pipeline - Step 6b: SHAP Supplemental Figures")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Load data (same as step6)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
all_df  = pd.read_csv(FEAT / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df = pd.read_csv(FEAT / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")

sel_df    = pd.read_csv(MOD_DIR / "selected_features_stability.csv")
feat_cols = sel_df["Feature"].tolist()
n_feats   = len(feat_cols)

scaler_df = pd.read_csv(MOD_DIR / "scaler_params_stability.csv")
scaler_df = scaler_df.set_index("Feature").loc[feat_cols].reset_index()
mu = scaler_df["mean"].values
sd = scaler_df["std"].values

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

# Raw (unscaled) Sphericity for the training set — needed for violin plot
sph_col = "original_shape_Sphericity"
sph_raw_puth = puth_df[sph_col].values.astype(float)
sph_raw_jsph = jsph_df[sph_col].values.astype(float)
sph_raw_all  = np.concatenate([sph_raw_puth, sph_raw_jsph])

# ─────────────────────────────────────────────────────────────────────────────
# Re-fit Elastic Net (identical to step6)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRe-fitting Elastic Net...")
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
sph_idx  = feat_cols.index(sph_col)
sph_coef = model.coef_[0][sph_idx]
print(f"  Sphericity coefficient: {sph_coef:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP LinearExplainer
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer  = shap.LinearExplainer(model, X_train,
                                   feature_perturbation="interventional")
shap_train = explainer(X_train)
shap_puch  = explainer(X_puch_s)
shap_tcga  = explainer(X_tcga_s)

sv_train = shap_train.values
sv_puch  = shap_puch.values
sv_tcga  = shap_tcga.values
sv_all   = np.vstack([sv_train, sv_puch, sv_tcga])

probs_train = model.predict_proba(X_train)[:, 1]
print(f"  Done — training SHAP shape: {sv_train.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Sphericity Stratification (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 1: Sphericity stratification...")

# Quartile labels based on training set distribution
q_edges = np.quantile(sph_raw_all, [0, 0.25, 0.50, 0.75, 1.0])
q_labels = ["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"]
q_idx    = np.digitize(sph_raw_all, q_edges[1:-1])  # 0-based quartile

# --- colours
RSI_POS_COLOR = "#E91E63"   # pink
RSI_NEG_COLOR = "#2196F3"   # blue
QUARTILE_COLORS = ["#FFF176", "#FFD54F", "#FF8A65", "#EF5350"]  # light→dark warm

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle(
    "Sphericity Stratification Analysis\n"
    "Training set: PUTH + JSPH-RN (n=576)",
    fontsize=12, fontweight="bold", y=1.02
)

# ── Panel A: Violin plot by RSI status ──────────────────────────────────────
ax = axes[0]
sph_pos = sph_raw_all[y_train == 1]
sph_neg = sph_raw_all[y_train == 0]

parts = ax.violinplot([sph_neg, sph_pos], positions=[0, 1],
                       showmedians=True, showextrema=True, widths=0.6)
for i, (pc, c) in enumerate(zip(parts["bodies"],
                                  [RSI_NEG_COLOR, RSI_POS_COLOR])):
    pc.set_facecolor(c); pc.set_alpha(0.55)
for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
    parts[key].set_colors(["gray"])

# overlay jitter
for xi, (vals, c) in enumerate([(sph_neg, RSI_NEG_COLOR),
                                  (sph_pos, RSI_POS_COLOR)]):
    jitter = np.random.uniform(-0.12, 0.12, len(vals))
    ax.scatter(xi + jitter, vals, color=c, alpha=0.25, s=6, zorder=2)

# Mann-Whitney U
stat, pval = stats.mannwhitneyu(sph_pos, sph_neg, alternative="two-sided")
sig_str = f"p={pval:.3f}" if pval >= 0.001 else f"p<0.001"
ax.text(0.5, 0.97, f"Mann–Whitney {sig_str}",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xticks([0, 1])
ax.set_xticklabels(["RSI−\n(n={})".format(len(sph_neg)),
                     "RSI+\n(n={})".format(len(sph_pos))], fontsize=10)
ax.set_ylabel("Sphericity (raw)", fontsize=10)
ax.set_title("A  Distribution by RSI Status", fontsize=10, fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.3)

print(f"  Sphericity: RSI+ median={np.median(sph_pos):.3f}  "
      f"RSI- median={np.median(sph_neg):.3f}  {sig_str}")

# ── Panel B: RSI+ rate per quartile (bar + Wilson 95% CI) ──────────────────
ax = axes[1]

def wilson_ci(pos, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = pos / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - half), min(1, center + half)

rsi_rates, ci_lo, ci_hi, ns = [], [], [], []
for q in range(4):
    mask  = q_idx == q
    n_tot = mask.sum()
    n_pos = y_train[mask].sum()
    rate  = n_pos / n_tot if n_tot > 0 else 0
    lo, hi = wilson_ci(n_pos, n_tot)
    rsi_rates.append(rate * 100)
    ci_lo.append((rate - lo) * 100)
    ci_hi.append((hi - rate) * 100)
    ns.append(n_tot)

bars = ax.bar(range(4), rsi_rates, color=QUARTILE_COLORS, alpha=0.85,
              edgecolor="gray", linewidth=0.8)
ax.errorbar(range(4), rsi_rates,
            yerr=[ci_lo, ci_hi],
            fmt="none", color="black", capsize=5, linewidth=1.5)

for i, (b, n) in enumerate(zip(bars, ns)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + ci_hi[i] + 1.5,
            f"n={n}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(range(4))
ax.set_xticklabels(q_labels, fontsize=9)
ax.set_xlabel("Sphericity Quartile", fontsize=10)
ax.set_ylabel("RSI+ Rate (%)", fontsize=10)
ax.set_ylim(0, max(rsi_rates) + max(ci_hi) + 12)
ax.set_title("B  RSI+ Rate by Sphericity Quartile\n(95% Wilson CI)",
             fontsize=10, fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.3)

# Trend annotation
slope, intercept, r, p_trend, se = stats.linregress(range(4), rsi_rates)
ax.text(0.97, 0.97,
        f"Trend: {'↓' if slope < 0 else '↑'}{abs(slope):.1f}%/quartile\n"
        f"(p_trend={p_trend:.3f})",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

print(f"  RSI+ rate by quartile: {[f'{r:.1f}%' for r in rsi_rates]}")
print(f"  Trend p={p_trend:.4f}")

# ── Panel C: Mean |SHAP| contribution per quartile ─────────────────────────
ax = axes[2]

# Use training SHAP only (quartile is defined on training data)
sph_shap_all = sv_train[:, sph_idx]  # SHAP values for Sphericity, all train cases

q_shap_pos, q_shap_neg = [], []
for q in range(4):
    mask  = q_idx == q
    shaps = sph_shap_all[mask]
    q_shap_pos.append(shaps[shaps > 0].mean() if (shaps > 0).any() else 0)
    q_shap_neg.append(shaps[shaps < 0].mean() if (shaps < 0).any() else 0)

x = np.arange(4)
ax.bar(x, q_shap_pos, color=RSI_POS_COLOR, alpha=0.75, label="RSI+ push (positive)")
ax.bar(x, q_shap_neg, color=RSI_NEG_COLOR, alpha=0.75, label="RSI− push (negative)")
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(q_labels, fontsize=9)
ax.set_xlabel("Sphericity Quartile", fontsize=10)
ax.set_ylabel("Mean SHAP value (log-odds)", fontsize=10)
ax.set_title("C  Sphericity SHAP Contribution\nby Quartile (Training set)",
             fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Add coefficient annotation
ax.text(0.03, 0.97,
        f"Coef = {sph_coef:+.3f}\n(negative → low sph = RSI+)",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))

plt.tight_layout()
plt.savefig(OUT_DIR / "shap_sphericity_stratification.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_sphericity_stratification.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Typical RSI- waterfall (prob 0.02 - 0.10)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 2: Typical RSI- waterfall...")

# Short display names (same logic as step6)
def short_name(feat, max_len=42):
    parts = feat.split("_")
    if feat.startswith("wavelet-"):
        band = parts[0].replace("wavelet-", "w")
        rest = "_".join(parts[1:])
        s = f"{band}_{rest}"
    else:
        s = feat
    return s[:max_len] + ("..." if len(s) > max_len else "")

feat_display = [short_name(f) for f in feat_cols]

# Select typical RSI- case: true negative with prob in [0.02, 0.10]
# If none found in that range, widen to [0.01, 0.15]
neg_mask  = y_train == 0
neg_idx   = np.where(neg_mask)[0]
neg_probs = probs_train[neg_idx]

typical_mask = (neg_probs >= 0.02) & (neg_probs <= 0.10)
if typical_mask.sum() == 0:
    typical_mask = (neg_probs >= 0.01) & (neg_probs <= 0.15)
if typical_mask.sum() == 0:
    typical_mask = neg_probs < 0.20  # fallback

# Among candidates, pick one closest to prob=0.05 (median typical)
candidates = neg_idx[typical_mask]
cand_probs = probs_train[candidates]
target_p   = 0.05
best_typical = candidates[np.argmin(np.abs(cand_probs - target_p))]
prob_typical = probs_train[best_typical]

print(f"  Typical RSI- case: idx={best_typical}  prob={prob_typical:.3f}")
print(f"  (from {typical_mask.sum()} candidates in target prob range)")

plt.figure(figsize=(10, 6))
sv_case = shap.Explanation(
    values      = sv_train[best_typical],
    base_values = float(explainer.expected_value),
    data        = X_train[best_typical],
    feature_names = feat_display,
)
shap.waterfall_plot(sv_case, max_display=15, show=False)
plt.title(
    f"SHAP Waterfall — Typical RSI− case (prob={prob_typical:.3f})\n"
    f"(log-odds space; base = {explainer.expected_value:.3f})",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_waterfall_rsi_neg_typical.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_waterfall_rsi_neg_typical.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print SHAP summary for the two new waterfall cases
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Sphericity stratification summary:")
print(f"  Quartile edges: {[f'{e:.3f}' for e in q_edges]}")
for q in range(4):
    mask = q_idx == q
    print(f"  Q{q+1}  n={mask.sum():3d}  "
          f"RSI+={y_train[mask].sum():3d} ({rsi_rates[q]:.1f}%)  "
          f"median_sph={np.median(sph_raw_all[mask]):.3f}  "
          f"mean_SHAP={sph_shap_all[mask].mean():+.3f}")

print(f"\nCoefficient for Sphericity: {sph_coef:+.4f}")
print("(negative → low sphericity → RSI+, i.e. irregular shape = invasion risk)")
print(f"\n{'='*65}")
print("Step 6b Complete.")
print(f"  Output: {OUT_DIR}")
print(f"  shap_sphericity_stratification.png")
print(f"  shap_waterfall_rsi_neg_typical.png")
print(f"{'='*65}")
