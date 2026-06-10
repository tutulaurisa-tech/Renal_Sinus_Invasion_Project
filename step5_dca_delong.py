# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 5: DeLong Test + Decision Curve Analysis
============================================================
Inputs:
  5_Modeling/v2_stability/predictions_*_stability.csv   (Elastic Net)
  7_GNN/v2_stability/predictions_GNN_*_stability.csv    (GNN-PSN)

Tests performed:
  1. DeLong test   — GNN-PSN vs Elastic Net, per cohort
  2. DeLong test   — Stability vs V2 LASSO (main improvement test)
  3. DCA curves    — per cohort: GNN / ElasticNet / Treat-all / Treat-none
  4. Combined performance table (AUC + SEN + SPE + DeLong p-value)

External sets (fair comparison): PUCH, TCGA
Training sets (reference only):  PUTH, JSPH-RN

Outputs (5_Results/v2_stability/):
  delong_results.csv
  dca_PUCH.png
  dca_TCGA.png
  dca_combined.png   (2×2 all cohorts)
  final_performance_table.csv
  final_performance_table.txt

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step5_dca_delong.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE    = Path(r"D:\RSI_Project_Workspace")
MOD_DIR = BASE / "5_Modeling" / "v2_stability"
GNN_DIR = BASE / "7_GNN"      / "v2_stability"
OLD_DIR = BASE / "5_Modeling" / "v2"            # V2 LASSO predictions
OUT_RES = BASE / "5_Results"  / "v2_stability"
OUT_RES.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("V2 Pipeline — Step 5: DeLong Test + Decision Curve Analysis")
print("=" * 65)


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════
def auc_score(yt, yp):
    """Vectorized Mann-Whitney AUC."""
    pos = yp[yt == 1]; neg = yp[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Broadcasting: pos[:,None] vs neg[None,:]
    mat = (pos[:, None] > neg[None, :]).astype(float) \
        + 0.5 * (pos[:, None] == neg[None, :]).astype(float)
    return float(mat.mean())


def boot_ci(yt, yp, n=1000, seed=42):
    rng = np.random.RandomState(seed); aucs = []; N = len(yt)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(np.unique(yt[idx])) < 2:
            continue
        aucs.append(auc_score(yt[idx], yp[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def youden_threshold(yt, yp):
    thresholds = np.sort(np.unique(yp))[::-1]
    best_j, best_thr = -1, 0.5
    P = yt.sum(); N = len(yt) - P
    if P == 0 or N == 0:
        return 0.5
    for thr in thresholds:
        pred = (yp >= thr).astype(int)
        tp = ((pred==1)&(yt==1)).sum()
        tn = ((pred==0)&(yt==0)).sum()
        j  = tp/P + tn/N - 1
        if j > best_j:
            best_j = j; best_thr = thr
    return best_thr


def thresh_metrics(yt, yp, thr=None):
    if thr is None:
        thr = youden_threshold(yt, yp)
    pred = (yp >= thr).astype(int)
    tp = ((pred==1)&(yt==1)).sum(); tn = ((pred==0)&(yt==0)).sum()
    fp = ((pred==1)&(yt==0)).sum(); fn = ((pred==0)&(yt==1)).sum()
    sen = tp/(tp+fn) if (tp+fn)>0 else float("nan")
    spe = tn/(tn+fp) if (tn+fp)>0 else float("nan")
    acc = (tp+tn)/len(yt)
    ppv = tp/(tp+fp) if (tp+fp)>0 else float("nan")
    npv = tn/(tn+fn) if (tn+fn)>0 else float("nan")
    return {"SEN":round(sen,3),"SPE":round(spe,3),"ACC":round(acc,3),
            "PPV":round(ppv,3),"NPV":round(npv,3),"Threshold":round(thr,3)}


# ═════════════════════════════════════════════════════════════════════════════
# DeLong Test  (DeLong 1988 / Sun & Xu 2014)
# ═════════════════════════════════════════════════════════════════════════════
def _placement_values(yt, yp):
    """V10: placement value of positive vs negative (structural component)."""
    pos = yp[yt == 1]; neg = yp[yt == 0]
    V10 = np.array([np.mean((p > neg) + 0.5*(p == neg)) for p in pos])
    V01 = np.array([np.mean((neg_j < pos) + 0.5*(neg_j == pos)) for neg_j in neg])
    return V10, V01


def delong_test(yt, yp1, yp2):
    """
    Two-sided DeLong test: H0: AUC1 == AUC2.
    Returns: (z_stat, p_value, auc1, auc2, delta_auc).
    """
    auc1 = auc_score(yt, yp1)
    auc2 = auc_score(yt, yp2)
    n1 = (yt == 1).sum()   # positives
    n0 = (yt == 0).sum()   # negatives

    V10_1, V01_1 = _placement_values(yt, yp1)
    V10_2, V01_2 = _placement_values(yt, yp2)

    # Covariance matrix S (2×2) of [AUC1, AUC2]
    S10 = np.cov(V10_1, V10_2) / n1   # contribution from positives
    S01 = np.cov(V01_1, V01_2) / n0   # contribution from negatives
    S   = S10 + S01                     # 2×2 covariance matrix

    # Var(AUC1 - AUC2) = S[0,0] + S[1,1] - 2*S[0,1]
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0, auc1, auc2, auc1 - auc2

    z  = (auc1 - auc2) / np.sqrt(var_diff)
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p), float(auc1), float(auc2), float(auc1 - auc2)


# ═════════════════════════════════════════════════════════════════════════════
# Decision Curve Analysis
# ═════════════════════════════════════════════════════════════════════════════
def net_benefit(yt, yp, thresholds):
    """Net benefit curve for a prediction model."""
    N = len(yt)
    nb = []
    for pt in thresholds:
        pred = (yp >= pt).astype(int)
        tp   = ((pred==1) & (yt==1)).sum()
        fp   = ((pred==1) & (yt==0)).sum()
        nb.append((tp - fp * pt / (1 - pt + 1e-10)) / N)
    return np.array(nb)


def net_benefit_all(yt, thresholds):
    """Net benefit of 'treat all' strategy."""
    prev = yt.mean()
    return np.array([prev - (1 - prev) * pt / (1 - pt + 1e-10)
                     for pt in thresholds])


def plot_dca(yt, yp_dict, title, ax, thr_range=(0.05, 0.85)):
    """
    yp_dict: {label: (probs, color, linestyle)}
    """
    thresholds = np.linspace(thr_range[0], thr_range[1], 200)

    # Treat none (always 0)
    ax.axhline(0, color="black", lw=1.2, linestyle=":", label="Treat None")

    # Treat all
    nb_all = net_benefit_all(yt, thresholds)
    ax.plot(thresholds * 100, np.clip(nb_all, -0.05, None),
            color="gray", lw=1.5, linestyle="--", label="Treat All")

    # Models
    for label, (yp, color, ls) in yp_dict.items():
        nb = net_benefit(yt, yp, thresholds)
        ax.plot(thresholds * 100, np.clip(nb, -0.05, None),
                color=color, lw=2, linestyle=ls, label=label)

    ax.set_xlim(thr_range[0] * 100, thr_range[1] * 100)
    ax.set_ylim(-0.05, yt.mean() * 1.3)
    ax.set_xlabel("Threshold Probability (%)", fontsize=10)
    ax.set_ylabel("Net Benefit", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25)


# ═════════════════════════════════════════════════════════════════════════════
# Load predictions
# ═════════════════════════════════════════════════════════════════════════════
print("\nLoading predictions...")

# Elastic Net (step3_stability)
en_puth = pd.read_csv(MOD_DIR / "predictions_PUTH_stability.csv")
en_jsph = pd.read_csv(MOD_DIR / "predictions_JSPH_RN_stability.csv")
en_puch = pd.read_csv(MOD_DIR / "predictions_PUCH_stability.csv")
en_tcga = pd.read_csv(MOD_DIR / "predictions_TCGA_stability.csv")

# GNN-PSN (step4_gnn_stability)
gnn_puth = pd.read_csv(GNN_DIR / "predictions_GNN_PUTH_stability.csv")
gnn_jsph = pd.read_csv(GNN_DIR / "predictions_GNN_JSPH_RN_stability.csv")
gnn_puch = pd.read_csv(GNN_DIR / "predictions_GNN_PUCH_stability.csv")
gnn_tcga = pd.read_csv(GNN_DIR / "predictions_GNN_TCGA_stability.csv")

# V2 LASSO (step3 original) — for improvement comparison
lasso_puth = lasso_puch = lasso_tcga = lasso_jsph = None
try:
    lasso_puth = pd.read_csv(OLD_DIR / "predictions_PUTH_v2.csv")
    lasso_puch = pd.read_csv(OLD_DIR / "predictions_PUCH_v2.csv")
    lasso_tcga = pd.read_csv(OLD_DIR / "predictions_TCGA_v2.csv")
    lasso_jsph = pd.read_csv(OLD_DIR / "predictions_JSPH_RN_v2.csv")
    print("  V2 LASSO predictions loaded for comparison")
except FileNotFoundError:
    print("  V2 LASSO predictions not found — skipping LASSO comparison")

# Ground truth
y = {
    "PUTH":    en_puth.RSI.values.astype(int),
    "JSPH_RN": en_jsph.RSI.values.astype(int),
    "PUCH":    en_puch.RSI.values.astype(int),
    "TCGA":    en_tcga.RSI.values.astype(int),
}
yp_en  = {"PUTH":    en_puth.RSI_Prob.values,
           "JSPH_RN": en_jsph.RSI_Prob.values,
           "PUCH":    en_puch.RSI_Prob.values,
           "TCGA":    en_tcga.RSI_Prob.values}
yp_gnn = {"PUTH":    gnn_puth.GNN_Prob.values,
           "JSPH_RN": gnn_jsph.GNN_Prob.values,
           "PUCH":    gnn_puch.GNN_Prob.values,
           "TCGA":    gnn_tcga.GNN_Prob.values}
yp_lasso = {}
if lasso_puch is not None:
    yp_lasso = {"PUTH":    lasso_puth.RSI_Prob.values,
                "JSPH_RN": lasso_jsph.RSI_Prob.values,
                "PUCH":    lasso_puch.RSI_Prob.values,
                "TCGA":    lasso_tcga.RSI_Prob.values}


# ═════════════════════════════════════════════════════════════════════════════
# DeLong Tests
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- DeLong Tests ---")

delong_rows = []

COHORT_LABELS = {
    "PUTH":    "PUTH (Train, in-sample)",
    "JSPH_RN": "JSPH-RN (Train, in-sample)",
    "PUCH":    "PUCH (Int.Test, external)",
    "TCGA":    "TCGA (Ext.Val, external)",
}

# Test 1: GNN-PSN vs Elastic Net
print("\n  [1] GNN-PSN vs Elastic Net (Stability features)")
for ckey, clabel in COHORT_LABELS.items():
    z, p, a1, a2, delta = delong_test(y[ckey], yp_gnn[ckey], yp_en[ckey])
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    print(f"    {clabel:<35s}  "
          f"GNN={a1:.3f} vs EN={a2:.3f}  Δ={delta:+.3f}  "
          f"z={z:+.3f}  p={p:.4f}  {sig}")
    delong_rows.append({
        "Comparison": "GNN-PSN vs ElasticNet",
        "Cohort": clabel,
        "AUC_model1": round(a1, 3), "AUC_model2": round(a2, 3),
        "Delta": round(delta, 3), "Z": round(z, 3), "P_value": round(p, 4),
        "Significance": sig,
    })

# Test 2: Elastic Net (Stability) vs V2 LASSO
if yp_lasso:
    print("\n  [2] Elastic Net (Stability) vs V2 LASSO")
    for ckey, clabel in COHORT_LABELS.items():
        z, p, a1, a2, delta = delong_test(y[ckey], yp_en[ckey], yp_lasso[ckey])
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    {clabel:<35s}  "
              f"Stab={a1:.3f} vs LASSO={a2:.3f}  Δ={delta:+.3f}  "
              f"z={z:+.3f}  p={p:.4f}  {sig}")
        delong_rows.append({
            "Comparison": "ElasticNet(Stab) vs V2 LASSO",
            "Cohort": clabel,
            "AUC_model1": round(a1, 3), "AUC_model2": round(a2, 3),
            "Delta": round(delta, 3), "Z": round(z, 3), "P_value": round(p, 4),
            "Significance": sig,
        })

# Test 3: GNN-PSN (Stability) vs V2 LASSO
if yp_lasso:
    print("\n  [3] GNN-PSN (Stability) vs V2 LASSO")
    for ckey, clabel in COHORT_LABELS.items():
        z, p, a1, a2, delta = delong_test(y[ckey], yp_gnn[ckey], yp_lasso[ckey])
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    {clabel:<35s}  "
              f"GNN={a1:.3f} vs LASSO={a2:.3f}  Δ={delta:+.3f}  "
              f"z={z:+.3f}  p={p:.4f}  {sig}")
        delong_rows.append({
            "Comparison": "GNN-PSN(Stab) vs V2 LASSO",
            "Cohort": clabel,
            "AUC_model1": round(a1, 3), "AUC_model2": round(a2, 3),
            "Delta": round(delta, 3), "Z": round(z, 3), "P_value": round(p, 4),
            "Significance": sig,
        })

delong_df = pd.DataFrame(delong_rows)
delong_df.to_csv(OUT_RES / "delong_results.csv", index=False, encoding="utf-8-sig")
print(f"\n  Saved: delong_results.csv")


# ═════════════════════════════════════════════════════════════════════════════
# Final Performance Table
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Final Performance Table ---")

perf_rows = []
all_models = [("GNN-PSN",     yp_gnn),
              ("ElasticNet",  yp_en)]
if yp_lasso:
    all_models.append(("V2-LASSO", yp_lasso))

COHORT_ORDER = [
    ("PUTH",    "PUTH (Train)"),
    ("JSPH_RN", "JSPH-RN (Train)"),
    ("PUCH",    "PUCH (Int.Test)"),
    ("TCGA",    "TCGA (Ext.Val)"),
]

for ckey, clabel in COHORT_ORDER:
    for mname, yp_dict in all_models:
        yt = y[ckey]; yp = yp_dict[ckey]
        auc = auc_score(yt, yp)
        lo, hi = boot_ci(yt, yp)
        m = thresh_metrics(yt, yp)
        perf_rows.append({
            "Cohort":  clabel,
            "Model":   mname,
            "AUC":     round(auc, 3),
            "CI_lo":   round(lo, 3),
            "CI_hi":   round(hi, 3),
            "AUC_str": f"{auc:.3f} ({lo:.3f}-{hi:.3f})",
            **m,
        })

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(OUT_RES / "final_performance_table.csv",
               index=False, encoding="utf-8-sig")

# Pretty-print
header = f"  {'Cohort':<22s}  {'Model':>12s}  {'AUC (95% CI)':<22s}  {'SEN':>5s}  {'SPE':>5s}  {'ACC':>5s}"
sep    = "  " + "-" * (len(header) - 2)
lines  = ["\n" + "=" * 65, "  Final Performance Table", "=" * 65, header, sep]
prev_cohort = ""
for _, row in perf_df.iterrows():
    if row.Cohort != prev_cohort:
        if prev_cohort:
            lines.append(sep)
        prev_cohort = row.Cohort
    lines.append(f"  {row.Cohort:<22s}  {row.Model:>12s}  "
                 f"{row.AUC_str:<22s}  {row.SEN:>5.3f}  {row.SPE:>5.3f}  {row.ACC:>5.3f}")
lines.append("=" * 65)
lines.append("  * PUTH / JSPH-RN: in-sample for GNN, cross-validated for ElasticNet")
lines.append("  * PUCH / TCGA: fully external for both models")
table_str = "\n".join(lines)
print(table_str)

with open(OUT_RES / "final_performance_table.txt", "w", encoding="utf-8") as f:
    f.write(table_str)
print(f"\n  Saved: final_performance_table.csv / .txt")


# ═════════════════════════════════════════════════════════════════════════════
# Decision Curve Analysis — Individual plots
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Decision Curve Analysis ---")

PLOT_COHORTS = [
    ("PUTH",    "PUTH (Training, in-sample)"),
    ("PUCH",    "PUCH (Internal Test)"),
    ("TCGA",    "TCGA (External Validation)"),
    ("JSPH_RN", "JSPH-RN (Training, in-sample)"),
]

MODEL_STYLE = {
    "GNN-PSN":    ("#E91E63", "-"),
    "ElasticNet": ("#2196F3", "-"),
    "V2-LASSO":   ("#FF9800", "--"),
}

for ckey, clabel in [("PUCH", "PUCH (Internal Test)"),
                      ("TCGA", "TCGA (External Validation)")]:
    yt = y[ckey]
    yp_dict_dca = {}
    for mname, yp_d in all_models:
        color, ls = MODEL_STYLE[mname]
        yp_dict_dca[mname] = (yp_d[ckey], color, ls)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_dca(yt, yp_dict_dca, clabel, ax)
    plt.tight_layout()
    fname = f"dca_{ckey}.png"
    plt.savefig(OUT_RES / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ═════════════════════════════════════════════════════════════════════════════
# Decision Curve Analysis — Combined 2×2 figure (all cohorts)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for ax, (ckey, clabel) in zip(axes, PLOT_COHORTS):
    yt = y[ckey]
    yp_dict_dca = {}
    for mname, yp_d in all_models:
        color, ls = MODEL_STYLE[mname]
        yp_dict_dca[mname] = (yp_d[ckey], color, ls)
    plot_dca(yt, yp_dict_dca, clabel, ax)

fig.suptitle(
    "Decision Curve Analysis — Stability Edition\n"
    "GNN-PSN  vs  Elastic Net (Bootstrap-Stable)  vs  V2 LASSO\n"
    "Train: PUTH+JSPH-RN (576)  |  External: PUCH, TCGA",
    fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_RES / "dca_combined.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: dca_combined.png")


# ═════════════════════════════════════════════════════════════════════════════
# ROC comparison — all models, external cohorts only (clean figure for paper)
# ═════════════════════════════════════════════════════════════════════════════
def roc_curve_np(yt, yp):
    desc = np.argsort(yp)[::-1]; ys = yt[desc]
    P = yt.sum(); N = len(yt) - P
    tpr, fpr = [0.], [0.]; tp = fp = 0; prev = None
    for label, score in zip(ys, yp[desc]):
        if score != prev:
            tpr.append(tp/P); fpr.append(fp/N); prev = score
        if label == 1: tp += 1
        else: fp += 1
    tpr.append(tp/P); fpr.append(fp/N)
    return np.array(fpr), np.array(tpr)


ROC_STYLE = {
    "GNN-PSN":    ("#E91E63", "-",  2.0),
    "ElasticNet": ("#2196F3", "-",  2.0),
    "V2-LASSO":   ("#FF9800", "--", 1.5),
}
ext_cohorts = [("PUCH", "PUCH (Internal Test)"),
               ("TCGA", "TCGA (External Validation)")]

fig3, ax3s = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (ckey, clabel) in zip(ax3s, ext_cohorts):
    yt = y[ckey]
    for mname, yp_d in all_models:
        color, ls, lw = ROC_STYLE[mname]
        yp = yp_d[ckey]
        fpr, tpr = roc_curve_np(yt, yp)
        auc = auc_score(yt, yp); lo, hi = boot_ci(yt, yp)
        ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                label=f"{mname}: {auc:.3f} [{lo:.3f}-{hi:.3f}]")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel("1 – Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity",    fontsize=11)
    ax.set_title(clabel, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

fig3.suptitle(
    "ROC Curves — External Cohorts\n"
    "GNN-PSN (Stability) vs Elastic Net (Stability) vs V2 LASSO",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_RES / "roc_external_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: roc_external_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# Final summary print
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("Step 5 Complete.")
print(f"{'='*65}")
print(f"  Output: {OUT_RES}")
print(f"  delong_results.csv")
print(f"  final_performance_table.csv / .txt")
print(f"  dca_PUCH.png  dca_TCGA.png  dca_combined.png")
print(f"  roc_external_comparison.png")

# Quick DeLong summary for external sets
print(f"\n  DeLong summary (external sets only):")
ext_rows = delong_df[delong_df.Cohort.str.contains("external")]
for _, row in ext_rows.iterrows():
    print(f"    [{row.Comparison}]  {row.Cohort.split('(')[0].strip()}: "
          f"Δ={row.Delta:+.3f}  p={row.P_value:.4f}  {row.Significance}")

print(f"\n  Pipeline complete. All outputs in {OUT_RES}")
print(f"  Have a good rest — 辛苦了！")
