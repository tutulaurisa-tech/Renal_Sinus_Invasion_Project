# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 4: GNN-PSN Training
========================================
Same GraphSAGE architecture as v1.
Input features: from selected_features_v2.csv (Step 3 output).
Train on PUTH, validate on PUCH (inductive), TCGA (inductive), JSPH-RN (inductive).

Outputs:
  graphsage_best_v2.pt          Saved model weights
  GNN_performance_v2.csv        AUC table (all cohorts)
  predictions_GNN_*_v2.csv      Per-case GNN probabilities
  GNN_vs_LASSO_ROC_v2.png       ROC comparison figure

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step4_gnn_train.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

BASE     = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR = BASE / "3_Extracted_Features"
OUT_GNN  = BASE / "7_GNN" / "v2"
OUT_MOD  = BASE / "5_Modeling" / "v2"
OUT_RES  = BASE / "5_Results" / "v2"
OUT_GNN.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_NEIGHBORS = 5
HIDDEN_DIM  = 64
DROPOUT     = 0.5
LR          = 5e-4
EPOCHS      = 500
PATIENCE    = 50
VAL_RATIO   = 0.2
SEED        = 42
IMG_TYPES   = ["original", "logarithm", "exponential", "square",
               "squareroot", "gradient", "wavelet"]

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}")
print("=" * 60)
print("V2 Pipeline — Step 4: GNN-PSN Training")
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

def roc_curve_np(yt, yp):
    desc = np.argsort(yp)[::-1]; ys = yt[desc]; P = yt.sum(); N = len(yt) - P
    tpr, fpr, thr = [0.], [0.], [np.inf]
    tp = fp = 0; prev = None
    for label, score in zip(ys, yp[desc]):
        if score != prev:
            tpr.append(tp / P); fpr.append(fp / N)
            thr.append(score); prev = score
        if label == 1: tp += 1
        else: fp += 1
    tpr.append(tp / P); fpr.append(fp / N); thr.append(-np.inf)
    return np.array(fpr), np.array(tpr), np.array(thr)

def youden_thr(yt, yp):
    fpr, tpr, thr = roc_curve_np(yt, yp)
    return thr[np.argmax(tpr - fpr)]

def eval_metrics(yt, yp, name):
    auc = auc_score(yt, yp); lo, hi = boot_ci(yt, yp)
    thr = youden_thr(yt, yp); pred = (yp >= thr).astype(int)
    tp = ((yt==1)&(pred==1)).sum(); tn = ((yt==0)&(pred==0)).sum()
    fp = ((yt==0)&(pred==1)).sum(); fn = ((yt==1)&(pred==0)).sum()
    sen = tp/(tp+fn) if tp+fn>0 else 0
    spe = tn/(tn+fp) if tn+fp>0 else 0
    acc = (tp+tn)/len(yt)
    print(f"  {name}: AUC={auc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"SEN={sen:.3f}  SPE={spe:.3f}")
    return {"Model": name, "AUC": auc, "CI_lo": lo, "CI_hi": hi,
            "AUC_str": f"{auc:.3f} ({lo:.3f}-{hi:.3f})",
            "SEN": sen, "SPE": spe, "ACC": acc}, yp


# ── Graph utils ───────────────────────────────────────────────────────────
def cosine_sim(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return (X / norm) @ (X / norm).T

def knn_edges(sim, k, src_idx, dst_idx):
    rows, cols = [], []
    candidates = np.array(dst_idx)
    for s in src_idx:
        scores = sim[s, candidates].copy()
        scores[candidates == s] = -1
        top_k = candidates[np.argsort(scores)[-k:]]
        for t in top_k:
            rows += [s, t]; cols += [t, s]
    return torch.unique(
        torch.tensor([rows, cols], dtype=torch.long), dim=1)


# ── GNN architecture ──────────────────────────────────────────────────────
class GraphSAGE_RSI(nn.Module):
    def __init__(self, in_ch, hidden=64, drop=0.5):
        super().__init__()
        self.c1   = SAGEConv(in_ch, hidden)
        self.c2   = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, self.drop, self.training)
        x = F.relu(self.c2(x, ei))
        x = F.dropout(x, self.drop, self.training)
        return torch.sigmoid(self.head(x)).squeeze(-1)


# ── Load data ─────────────────────────────────────────────────────────────
print("\nLoading data...")
all_df  = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df = pd.read_csv(FEAT_DIR / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")
sel_df  = pd.read_csv(OUT_MOD / "selected_features_v2.csv")

feat_cols = sel_df["Feature"].tolist()
print(f"Selected features: {len(feat_cols)}")

def img_type_of(feat):
    if feat.startswith("wavelet"):
        return "wavelet"
    for t in IMG_TYPES:
        if feat.startswith(t + "_"):
            return t
    return "unknown"

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

print(f"PUTH:{len(puth_df)}  PUCH:{len(puch_df)}  "
      f"TCGA:{len(tcga_df)}  JSPH-RN:{len(jsph_df)}")

# Normalize
mu = X_puth.mean(0); sd = X_puth.std(0) + 1e-8
Xp_s = (X_puth - mu) / sd
Xu_s = (X_puch - mu) / sd
Xt_s = (X_tcga - mu) / sd
Xj_s = (X_jsph - mu) / sd

n_puth = len(puth_df); n_puch = len(puch_df)
n_tcga = len(tcga_df); n_jsph = len(jsph_df)
puth_r = np.arange(n_puth)


# ── Train/val split (stratified 80/20 within PUTH) ───────────────────────
rng = np.random.RandomState(SEED)
pos_all = np.where(y_puth == 1)[0]; neg_all = np.where(y_puth == 0)[0]
rng.shuffle(pos_all); rng.shuffle(neg_all)
nv_pos = max(1, int(len(pos_all) * VAL_RATIO))
nv_neg = max(1, int(len(neg_all) * VAL_RATIO))
val_idx = np.concatenate([pos_all[:nv_pos], neg_all[:nv_neg]])
tr_idx  = np.concatenate([pos_all[nv_pos:], neg_all[nv_neg:]])
print(f"\nPUTH split -> train:{len(tr_idx)} RSI+={y_puth[tr_idx].sum()}  "
      f"val:{len(val_idx)} RSI+={y_puth[val_idx].sum()}")


# ── Build graphs ──────────────────────────────────────────────────────────
print("\nBuilding graphs...")
sim_p = cosine_sim(Xp_s)

ei_trval = torch.unique(torch.cat([
    knn_edges(sim_p, K_NEIGHBORS, tr_idx, tr_idx),
    knn_edges(sim_p, K_NEIGHBORS, tr_idx, val_idx),
], dim=1), dim=1)

ei_full_puth = knn_edges(sim_p, K_NEIGHBORS, puth_r, puth_r)
print(f"  PUTH full graph: {n_puth} nodes, {ei_full_puth.shape[1]} edges")

# PUCH inductive
Xpu_f  = np.vstack([Xp_s, Xu_s])
puch_r = np.arange(n_puth, n_puth + n_puch)
ei_puch = torch.unique(torch.cat([
    knn_edges(cosine_sim(Xpu_f), K_NEIGHBORS, puth_r, puth_r),
    knn_edges(cosine_sim(Xpu_f), K_NEIGHBORS, puth_r, puch_r),
], dim=1), dim=1)
print(f"  PUCH graph: {n_puth+n_puch} nodes, {ei_puch.shape[1]} edges")

# TCGA inductive
Xt_f   = np.vstack([Xp_s, Xt_s])
tcga_r = np.arange(n_puth, n_puth + n_tcga)
ei_tcga = torch.unique(torch.cat([
    knn_edges(cosine_sim(Xt_f), K_NEIGHBORS, puth_r, puth_r),
    knn_edges(cosine_sim(Xt_f), K_NEIGHBORS, puth_r, tcga_r),
], dim=1), dim=1)
print(f"  TCGA graph: {n_puth+n_tcga} nodes, {ei_tcga.shape[1]} edges")

# JSPH-RN inductive
Xj_f   = np.vstack([Xp_s, Xj_s])
jsph_r = np.arange(n_puth, n_puth + n_jsph)
ei_jsph = torch.unique(torch.cat([
    knn_edges(cosine_sim(Xj_f), K_NEIGHBORS, puth_r, puth_r),
    knn_edges(cosine_sim(Xj_f), K_NEIGHBORS, puth_r, jsph_r),
], dim=1), dim=1)
print(f"  JSPH-RN graph: {n_puth+n_jsph} nodes, {ei_jsph.shape[1]} edges")


# ── Tensors ───────────────────────────────────────────────────────────────
x_puth_t = torch.tensor(Xp_s,  dtype=torch.float32).to(DEVICE)
x_puch_t = torch.tensor(Xpu_f, dtype=torch.float32).to(DEVICE)
x_tcga_t = torch.tensor(Xt_f,  dtype=torch.float32).to(DEVICE)
x_jsph_t = torch.tensor(Xj_f,  dtype=torch.float32).to(DEVICE)
y_puth_t = torch.tensor(y_puth, dtype=torch.float32).to(DEVICE)

ei_trval_t    = ei_trval.to(DEVICE)
ei_full_puth_t = ei_full_puth.to(DEVICE)
ei_puch_t     = ei_puch.to(DEVICE)
ei_tcga_t     = ei_tcga.to(DEVICE)
ei_jsph_t     = ei_jsph.to(DEVICE)

tr_idx_t  = torch.tensor(tr_idx, dtype=torch.long)
val_idx_t = torch.tensor(val_idx, dtype=torch.long)

# Class weight
y_tr_np  = y_puth[tr_idx]
pw       = (y_tr_np == 0).sum() / (y_tr_np == 1).sum()
pos_wt   = torch.tensor([pw], dtype=torch.float32).to(DEVICE)
print(f"\nClass imbalance (neg/pos): {pw:.2f}")


# ── Training ──────────────────────────────────────────────────────────────
model = GraphSAGE_RSI(len(feat_cols), HIDDEN_DIM, DROPOUT).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
crit  = nn.BCEWithLogitsLoss(pos_weight=pos_wt)

# Use sigmoid output → need raw logits for BCEWithLogitsLoss
# Override forward for training (no sigmoid) then restore
class GraphSAGE_RSI_logit(nn.Module):
    def __init__(self, in_ch, hidden=64, drop=0.5):
        super().__init__()
        self.c1   = SAGEConv(in_ch, hidden)
        self.c2   = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, self.drop, self.training)
        x = F.relu(self.c2(x, ei))
        x = F.dropout(x, self.drop, self.training)
        return self.head(x).squeeze(-1)   # raw logits

    def predict_proba(self, x, ei):
        return torch.sigmoid(self.forward(x, ei))

model = GraphSAGE_RSI_logit(len(feat_cols), HIDDEN_DIM, DROPOUT).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

best_val_auc = 0.0; best_epoch = 0; patience_cnt = 0
best_state   = None
train_aucs   = []; val_aucs = []

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optim.zero_grad()
    logits = model(x_puth_t, ei_trval_t)
    loss   = crit(logits[tr_idx_t], y_puth_t[tr_idx_t])
    loss.backward(); optim.step()

    model.eval()
    with torch.no_grad():
        probs_full = model.predict_proba(x_puth_t, ei_full_puth_t).cpu().numpy()

    tr_auc  = auc_score(y_puth[tr_idx],  probs_full[tr_idx])
    val_auc = auc_score(y_puth[val_idx], probs_full[val_idx])
    train_aucs.append(tr_auc); val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc; best_epoch = epoch
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}  "
              f"train_AUC={tr_auc:.3f}  val_AUC={val_auc:.3f}  "
              f"best_val={best_val_auc:.3f} @{best_epoch}")

    if patience_cnt >= PATIENCE:
        print(f"  Early stop @epoch {epoch}  best_val_AUC={best_val_auc:.3f} @{best_epoch}")
        break

# Restore best weights
model.load_state_dict(best_state)
torch.save(best_state, OUT_GNN / "graphsage_best_v2.pt")
print(f"  Model saved: graphsage_best_v2.pt")


# ── Evaluation ────────────────────────────────────────────────────────────
print("\n--- Evaluation ---")
model.eval()

with torch.no_grad():
    prob_gnn_puth = model.predict_proba(x_puth_t, ei_full_puth_t).cpu().numpy()
    prob_all_puch = model.predict_proba(x_puch_t, ei_puch_t).cpu().numpy()
    prob_all_tcga = model.predict_proba(x_tcga_t, ei_tcga_t).cpu().numpy()
    prob_all_jsph = model.predict_proba(x_jsph_t, ei_jsph_t).cpu().numpy()

prob_gnn_puch = prob_all_puch[puch_r]
prob_gnn_tcga = prob_all_tcga[tcga_r]
prob_gnn_jsph = prob_all_jsph[jsph_r]

print("\n[PUTH - Train (in-sample)]")
mg_puth, _ = eval_metrics(y_puth, prob_gnn_puth, "GNN-PSN (PUTH)")

print("\n[PUCH - Internal Test]")
mg_puch, _ = eval_metrics(y_puch, prob_gnn_puch, "GNN-PSN (PUCH)")

print("\n[TCGA - External Val]")
mg_tcga, _ = eval_metrics(y_tcga, prob_gnn_tcga, "GNN-PSN (TCGA)")

print("\n[JSPH-RN - External Val]")
mg_jsph, _ = eval_metrics(y_jsph, prob_gnn_jsph, "GNN-PSN (JSPH-RN)")

# LASSO comparison
l_puth = pd.read_csv(OUT_MOD / "predictions_PUTH_v2.csv")
l_puch = pd.read_csv(OUT_MOD / "predictions_PUCH_v2.csv")
l_tcga = pd.read_csv(OUT_MOD / "predictions_TCGA_v2.csv")
l_jsph = pd.read_csv(OUT_MOD / "predictions_JSPH_RN_v2.csv")

print("\n[LASSO comparison]")
for name, y, ldf in [("PUTH(CV)", y_puth, l_puth),
                      ("PUCH",     y_puch, l_puch),
                      ("TCGA",     y_tcga, l_tcga),
                      ("JSPH-RN",  y_jsph, l_jsph)]:
    a = auc_score(y, ldf.RSI_Prob.values)
    lo, hi = boot_ci(y, ldf.RSI_Prob.values)
    print(f"  LASSO ({name}): AUC={a:.3f} [{lo:.3f}-{hi:.3f}]")


# ── Save predictions ──────────────────────────────────────────────────────
for name, y, probs, df_src in [
    ("PUTH",    y_puth, prob_gnn_puth, puth_df),
    ("PUCH",    y_puch, prob_gnn_puch, puch_df),
    ("TCGA",    y_tcga, prob_gnn_tcga, tcga_df),
    ("JSPH_RN", y_jsph, prob_gnn_jsph, jsph_df),
]:
    id_col = "Case_ID" if "Case_ID" in df_src.columns else "PatientID"
    pd.DataFrame({
        "PatientID": df_src[id_col],
        "RSI": y,
        "GNN_Prob": probs,
    }).to_csv(OUT_GNN / f"predictions_GNN_{name}_v2.csv",
              index=False, encoding="utf-8-sig")


# ── Performance table ─────────────────────────────────────────────────────
perf_rows = []
for cohort_label, mg, ml_df, y in [
    ("PUTH (Train)",          mg_puth, l_puth, y_puth),
    ("PUCH (Int.Test)",       mg_puch, l_puch, y_puch),
    ("TCGA (Ext.Val)",        mg_tcga, l_tcga, y_tcga),
    ("JSPH-RN (Ext.Val)",     mg_jsph, l_jsph, y_jsph),
]:
    la = auc_score(y, ml_df.RSI_Prob.values)
    llo, lhi = boot_ci(y, ml_df.RSI_Prob.values)
    perf_rows.append({"Cohort": cohort_label, "Model": "GNN-PSN",
                      **{k: round(v, 3) if isinstance(v, float) else v
                         for k, v in mg.items() if k != "Model"}})
    perf_rows.append({"Cohort": cohort_label, "Model": "LASSO",
                      "AUC": round(la, 3),
                      "AUC_str": f"{la:.3f} ({llo:.3f}-{lhi:.3f})"})

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(OUT_GNN / "GNN_performance_v2.csv",
               index=False, encoding="utf-8-sig")
print(f"\n{'='*60}")
print("Performance Summary (V2)")
print(f"{'='*60}")
for _, row in perf_df.iterrows():
    print(f"  {row.Cohort:25s}  {row.Model:8s}  {row.get('AUC_str', row.AUC)}")


# ── Training curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_aucs, label="Train AUC", color="#2196F3", lw=1.5)
ax.plot(val_aucs,   label="Val AUC",   color="#E91E63", lw=1.5)
ax.axvline(best_epoch - 1, color="gray", lw=1, linestyle="--",
           label=f"Best val epoch {best_epoch}")
ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
ax.set_title("GNN-PSN V2 Training Curve", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_GNN / "training_curve_v2.png", dpi=150, bbox_inches="tight")
plt.close()

# ── ROC comparison figure ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()
roc_sets = [
    ("PUTH (Training)",       y_puth, prob_gnn_puth, l_puth.RSI_Prob.values),
    ("PUCH (Internal Test)",  y_puch, prob_gnn_puch, l_puch.RSI_Prob.values),
    ("TCGA (External Val)",   y_tcga, prob_gnn_tcga, l_tcga.RSI_Prob.values),
    ("JSPH-RN (External Val)",y_jsph, prob_gnn_jsph, l_jsph.RSI_Prob.values),
]
for ax, (title, yt, pg, pl) in zip(axes2, roc_sets):
    for yp, label, color, ls in [
        (pg, "GNN-PSN", "#E91E63", "-"),
        (pl, "LASSO",   "#2196F3", "--"),
    ]:
        fpr, tpr, _ = roc_curve_np(yt, yp)
        auc = auc_score(yt, yp); lo, hi = boot_ci(yt, yp)
        ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                label=f"{label}: {auc:.3f} [{lo:.3f}-{hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 – Specificity", fontsize=10)
    ax.set_ylabel("Sensitivity", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

fig2.suptitle("V2 Pipeline: GNN-PSN vs LASSO ROC (All Cohorts)\n"
              "*PUTH LASSO = cross-validated; PUTH GNN = in-sample",
              fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_GNN / "GNN_vs_LASSO_ROC_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: GNN_vs_LASSO_ROC_v2.png")


print(f"\n{'='*60}")
print("Step 4 Complete.")
print(f"{'='*60}")
print(f"  Outputs: {OUT_GNN}")
print(f"\nNext steps (optional):")
print(f"  python step5_validate_jsph.py    # Detailed JSPH-RN validation")
print(f"  python step6_dca_delong.py       # DCA + DeLong test")
