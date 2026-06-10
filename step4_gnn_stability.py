# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 4 (Stability Edition): GNN-PSN Training
===========================================================
Training set : PUTH (346) + JSPH-RN (230)  — same as step3_stability
Features     : 29 bootstrap-stable features from selected_features_stability.csv
Scaler       : fit on PUTH + JSPH-RN (scaler_params_stability.csv)
Architecture : GraphSAGE (2-layer), same as V2
Graph        : Patient Similarity Network (cosine KNN, K=5)
               Training graph : PUTH + JSPH-RN combined
               Test graphs    : [train nodes] + [PUCH or TCGA nodes] (inductive)

Evaluation:
  PUTH     — in-sample (training cohort) — reported for comparison only
  JSPH-RN  — in-sample (training cohort) — reported for comparison only
  PUCH     — external test (inductive)
  TCGA     — external validation (inductive)

Comparison: GNN-PSN vs Elastic Net (step3_stability predictions)

Outputs (7_GNN/v2_stability/ and 5_Results/v2_stability/):
  graphsage_best_stability.pt
  GNN_performance_stability.csv
  predictions_GNN_{center}_stability.csv
  training_curve_stability.png
  GNN_vs_ElasticNet_ROC_stability.png

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step4_gnn_stability.py
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

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE     = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR = BASE / "3_Extracted_Features"
OUT_MOD  = BASE / "5_Modeling" / "v2_stability"     # step3 outputs
OUT_GNN  = BASE / "7_GNN"     / "v2_stability"
OUT_RES  = BASE / "5_Results" / "v2_stability"
OUT_GNN.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_NEIGHBORS = 5
HIDDEN_DIM  = 64
DROPOUT     = 0.5
LR          = 5e-4
EPOCHS      = 500
PATIENCE    = 50
VAL_RATIO   = 0.2
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}")
print("=" * 65)
print("V2 Pipeline — Step 4 Stability: GNN-PSN (PUTH+JSPH-RN train)")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def auc_score(yt, yp):
    pos = yp[yt == 1]; neg = yp[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return u / (len(pos) * len(neg))


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


def roc_curve_np(yt, yp):
    desc = np.argsort(yp)[::-1]; ys = yt[desc]
    P = yt.sum(); N = len(yt) - P
    tpr, fpr = [0.], [0.]
    tp = fp = 0; prev = None
    for label, score in zip(ys, yp[desc]):
        if score != prev:
            tpr.append(tp / P); fpr.append(fp / N); prev = score
        if label == 1: tp += 1
        else: fp += 1
    tpr.append(tp / P); fpr.append(fp / N)
    return np.array(fpr), np.array(tpr)


def youden_thr(yt, yp):
    fpr, tpr, thr = _roc_full(yt, yp)
    return thr[np.argmax(tpr - fpr)]


def _roc_full(yt, yp):
    desc = np.argsort(yp)[::-1]; ys = yt[desc]; yps = yp[desc]
    P = yt.sum(); N = len(yt) - P
    tpr, fpr, thr = [0.], [0.], [np.inf]
    tp = fp = 0; prev = None
    for label, score in zip(ys, yps):
        if score != prev:
            tpr.append(tp / P); fpr.append(fp / N); thr.append(score)
            prev = score
        if label == 1: tp += 1
        else: fp += 1
    tpr.append(tp / P); fpr.append(fp / N); thr.append(-np.inf)
    return np.array(fpr), np.array(tpr), np.array(thr)


def eval_metrics(yt, yp, name):
    auc = auc_score(yt, yp); lo, hi = boot_ci(yt, yp)
    fpr_arr, tpr_arr, thr_arr = _roc_full(yt, yp)
    thr = thr_arr[np.argmax(tpr_arr - fpr_arr)]
    pred = (yp >= thr).astype(int)
    tp = ((yt==1)&(pred==1)).sum(); tn = ((yt==0)&(pred==0)).sum()
    fp_n = ((yt==0)&(pred==1)).sum(); fn = ((yt==1)&(pred==0)).sum()
    sen = tp/(tp+fn) if tp+fn>0 else 0
    spe = tn/(tn+fp_n) if tn+fp_n>0 else 0
    acc = (tp+tn)/len(yt)
    print(f"  {name:30s}  AUC={auc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"SEN={sen:.3f}  SPE={spe:.3f}  ACC={acc:.3f}")
    return {"Model": name, "AUC": auc, "CI_lo": lo, "CI_hi": hi,
            "AUC_str": f"{auc:.3f} ({lo:.3f}-{hi:.3f})",
            "SEN": round(sen, 3), "SPE": round(spe, 3), "ACC": round(acc, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Graph utilities
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# GNN architecture
# ─────────────────────────────────────────────────────────────────────────────
class GraphSAGE_RSI(nn.Module):
    def __init__(self, in_ch, hidden=64, drop=0.5):
        super().__init__()
        self.c1   = SAGEConv(in_ch, hidden)
        self.c2   = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, x, ei):
        """Returns raw logits (for BCEWithLogitsLoss)."""
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, self.drop, self.training)
        x = F.relu(self.c2(x, ei))
        x = F.dropout(x, self.drop, self.training)
        return self.head(x).squeeze(-1)

    def predict_proba(self, x, ei):
        return torch.sigmoid(self.forward(x, ei))


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")
all_df  = pd.read_csv(FEAT_DIR / "All_Centers_Radiomics_v2_withLabel.csv")
jsph_df = pd.read_csv(FEAT_DIR / "Radiomics_CSVs" / "JSPH_RN_v2_withLabel.csv")

# Load stable features (all 29 bootstrap-stable, GNN will learn weights)
sel_df    = pd.read_csv(OUT_MOD / "selected_features_stability.csv")
feat_cols = sel_df["Feature"].tolist()
print(f"Stable features loaded: {len(feat_cols)}")

# Load scaler parameters (fit on PUTH+JSPH-RN in step3)
# Filter to only the 29 stable features (scaler_params has all 1316 features)
scaler_df = pd.read_csv(OUT_MOD / "scaler_params_stability.csv")
scaler_df = scaler_df.set_index("Feature").loc[feat_cols].reset_index()
mu_train  = scaler_df["mean"].values
sd_train  = scaler_df["std"].values
print(f"Scaler loaded (fit on PUTH+JSPH-RN, filtered to {len(feat_cols)} stable features)")

puth_df = all_df[all_df.Center == "PUTH"].reset_index(drop=True)
puch_df = all_df[all_df.Center == "PUCH"].reset_index(drop=True)
tcga_df = all_df[all_df.Center == "TCGA"].reset_index(drop=True)

X_puth_raw = puth_df[feat_cols].values.astype(float)
X_puch_raw = puch_df[feat_cols].values.astype(float)
X_tcga_raw = tcga_df[feat_cols].values.astype(float)
X_jsph_raw = jsph_df[feat_cols].values.astype(float)

y_puth = puth_df.RSI.values.astype(int)
y_puch = puch_df.RSI.values.astype(int)
y_tcga = tcga_df.RSI.values.astype(int)
y_jsph = jsph_df.RSI.values.astype(int)

print(f"\n  PUTH    (train): {len(puth_df):3d}  RSI+={y_puth.sum()} ({100*y_puth.mean():.1f}%)")
print(f"  JSPH-RN (train): {len(jsph_df):3d}  RSI+={y_jsph.sum()} ({100*y_jsph.mean():.1f}%)")
print(f"  PUCH    (test) : {len(puch_df):3d}  RSI+={y_puch.sum()} ({100*y_puch.mean():.1f}%)")
print(f"  TCGA    (test) : {len(tcga_df):3d}  RSI+={y_tcga.sum()} ({100*y_tcga.mean():.1f}%)")

# Apply step3 scaler
Xp_s = (X_puth_raw - mu_train) / sd_train
Xu_s = (X_puch_raw - mu_train) / sd_train
Xt_s = (X_tcga_raw - mu_train) / sd_train
Xj_s = (X_jsph_raw - mu_train) / sd_train

# Combined training matrix (PUTH then JSPH-RN, preserving order)
X_train_s = np.vstack([Xp_s, Xj_s])
y_train   = np.concatenate([y_puth, y_jsph])
n_puth    = len(y_puth)
n_jsph    = len(y_jsph)
n_train   = len(y_train)

print(f"\n  Combined training: {n_train}  RSI+={y_train.sum()} ({100*y_train.mean():.1f}%)")

# Combined center label for stratification: PUTH=0, JSPH-RN=1
center_train = np.concatenate([np.zeros(n_puth, int), np.ones(n_jsph, int)])
strat_train  = center_train * 2 + y_train   # 4 groups


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split — stratified by center×RSI
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.RandomState(SEED)
val_idx_list = []
tr_idx_list  = []
for grp in np.unique(strat_train):
    grp_idx = np.where(strat_train == grp)[0]
    rng.shuffle(grp_idx)
    n_val = max(1, int(len(grp_idx) * VAL_RATIO))
    val_idx_list.extend(grp_idx[:n_val].tolist())
    tr_idx_list.extend(grp_idx[n_val:].tolist())

val_idx = np.array(val_idx_list)
tr_idx  = np.array(tr_idx_list)
train_r = np.arange(n_train)

print(f"\n  Train split: {len(tr_idx)}  RSI+={y_train[tr_idx].sum()}")
print(f"  Val split  : {len(val_idx)}  RSI+={y_train[val_idx].sum()}")
print(f"  Val: PUTH={int((val_idx < n_puth).sum())}  "
      f"JSPH-RN={int((val_idx >= n_puth).sum())}")


# ─────────────────────────────────────────────────────────────────────────────
# Build graphs
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding graphs...")
sim_train = cosine_sim(X_train_s)

# Training graph: edges among train+val nodes (all training cases)
ei_trval = torch.unique(torch.cat([
    knn_edges(sim_train, K_NEIGHBORS, tr_idx,  tr_idx),
    knn_edges(sim_train, K_NEIGHBORS, tr_idx,  val_idx),
], dim=1), dim=1)

# Full training graph (for in-sample evaluation)
ei_full_train = knn_edges(sim_train, K_NEIGHBORS, train_r, train_r)
print(f"  Training graph: {n_train} nodes, {ei_full_train.shape[1]} edges")

# PUCH inductive: training nodes + PUCH nodes
n_puch    = len(y_puch)
Xpu_f     = np.vstack([X_train_s, Xu_s])
puch_r    = np.arange(n_train, n_train + n_puch)
ei_puch   = torch.unique(torch.cat([
    knn_edges(cosine_sim(Xpu_f), K_NEIGHBORS, train_r, train_r),
    knn_edges(cosine_sim(Xpu_f), K_NEIGHBORS, train_r, puch_r),
], dim=1), dim=1)
print(f"  PUCH graph:     {n_train+n_puch} nodes, {ei_puch.shape[1]} edges")

# TCGA inductive: training nodes + TCGA nodes
n_tcga    = len(y_tcga)
Xt_f      = np.vstack([X_train_s, Xt_s])
tcga_r    = np.arange(n_train, n_train + n_tcga)
ei_tcga   = torch.unique(torch.cat([
    knn_edges(cosine_sim(Xt_f), K_NEIGHBORS, train_r, train_r),
    knn_edges(cosine_sim(Xt_f), K_NEIGHBORS, train_r, tcga_r),
], dim=1), dim=1)
print(f"  TCGA graph:     {n_train+n_tcga} nodes, {ei_tcga.shape[1]} edges")


# ─────────────────────────────────────────────────────────────────────────────
# Tensors
# ─────────────────────────────────────────────────────────────────────────────
x_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
x_puch_t  = torch.tensor(Xpu_f,    dtype=torch.float32).to(DEVICE)
x_tcga_t  = torch.tensor(Xt_f,     dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train,  dtype=torch.float32).to(DEVICE)

ei_trval_t    = ei_trval.to(DEVICE)
ei_full_t     = ei_full_train.to(DEVICE)
ei_puch_t     = ei_puch.to(DEVICE)
ei_tcga_t     = ei_tcga.to(DEVICE)

tr_idx_t  = torch.tensor(tr_idx,  dtype=torch.long)
val_idx_t = torch.tensor(val_idx, dtype=torch.long)

# Positive class weight based on training subset
y_tr_np = y_train[tr_idx]
pw      = (y_tr_np == 0).sum() / max((y_tr_np == 1).sum(), 1)
pos_wt  = torch.tensor([pw], dtype=torch.float32).to(DEVICE)
print(f"\n  Class imbalance (neg/pos ratio in train split): {pw:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
model = GraphSAGE_RSI(len(feat_cols), HIDDEN_DIM, DROPOUT).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
crit  = nn.BCEWithLogitsLoss(pos_weight=pos_wt)

best_val_auc = 0.0; best_epoch = 0; patience_cnt = 0; best_state = None
train_aucs = []; val_aucs = []

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optim.zero_grad()
    logits = model(x_train_t, ei_trval_t)
    loss   = crit(logits[tr_idx_t], y_train_t[tr_idx_t])
    loss.backward(); optim.step()

    model.eval()
    with torch.no_grad():
        probs_full = model.predict_proba(x_train_t, ei_full_t).cpu().numpy()

    tr_auc  = auc_score(y_train[tr_idx],  probs_full[tr_idx])
    val_auc = auc_score(y_train[val_idx], probs_full[val_idx])
    train_aucs.append(tr_auc); val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc; best_epoch = epoch; patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}  loss={loss.item():.4f}  "
              f"train_AUC={tr_auc:.3f}  val_AUC={val_auc:.3f}  "
              f"best_val={best_val_auc:.3f} @{best_epoch}")

    if patience_cnt >= PATIENCE:
        print(f"  Early stop @epoch {epoch}  best_val_AUC={best_val_auc:.3f} @{best_epoch}")
        break

model.load_state_dict(best_state)
torch.save(best_state, OUT_GNN / "graphsage_best_stability.pt")
print(f"  Model saved: graphsage_best_stability.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Evaluation ---")
model.eval()
with torch.no_grad():
    prob_all_train = model.predict_proba(x_train_t, ei_full_t).cpu().numpy()
    prob_all_puch  = model.predict_proba(x_puch_t,  ei_puch_t).cpu().numpy()
    prob_all_tcga  = model.predict_proba(x_tcga_t,  ei_tcga_t).cpu().numpy()

prob_gnn_puth = prob_all_train[:n_puth]
prob_gnn_jsph = prob_all_train[n_puth:]
prob_gnn_puch = prob_all_puch[puch_r]
prob_gnn_tcga = prob_all_tcga[tcga_r]

print("\n[PUTH — in-sample (training cohort)]")
mg_puth = eval_metrics(y_puth, prob_gnn_puth, "GNN-PSN (PUTH in-sample)")

print("\n[JSPH-RN — in-sample (training cohort)]")
mg_jsph = eval_metrics(y_jsph, prob_gnn_jsph, "GNN-PSN (JSPH-RN in-sample)")

print("\n[PUCH — External Test]")
mg_puch = eval_metrics(y_puch, prob_gnn_puch, "GNN-PSN (PUCH ext.)")

print("\n[TCGA — External Validation]")
mg_tcga = eval_metrics(y_tcga, prob_gnn_tcga, "GNN-PSN (TCGA ext.)")


# ─────────────────────────────────────────────────────────────────────────────
# Load Elastic Net (step3_stability) predictions for comparison
# ─────────────────────────────────────────────────────────────────────────────
en_puth = pd.read_csv(OUT_MOD / "predictions_PUTH_stability.csv")
en_jsph = pd.read_csv(OUT_MOD / "predictions_JSPH_RN_stability.csv")
en_puch = pd.read_csv(OUT_MOD / "predictions_PUCH_stability.csv")
en_tcga = pd.read_csv(OUT_MOD / "predictions_TCGA_stability.csv")

print("\n[Elastic Net (step3_stability) — for comparison]")
for name, y, edf in [
    ("PUTH (CV)",   y_puth, en_puth),
    ("JSPH-RN (CV)",y_jsph, en_jsph),
    ("PUCH",        y_puch, en_puch),
    ("TCGA",        y_tcga, en_tcga),
]:
    a = auc_score(y, edf.RSI_Prob.values)
    lo, hi = boot_ci(y, edf.RSI_Prob.values)
    print(f"  ElasticNet ({name:15s}): AUC={a:.3f} [{lo:.3f}-{hi:.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Save predictions
# ─────────────────────────────────────────────────────────────────────────────
for name, y, probs, df_src in [
    ("PUTH",    y_puth, prob_gnn_puth, puth_df),
    ("JSPH_RN", y_jsph, prob_gnn_jsph, jsph_df),
    ("PUCH",    y_puch, prob_gnn_puch, puch_df),
    ("TCGA",    y_tcga, prob_gnn_tcga, tcga_df),
]:
    id_col = "Case_ID" if "Case_ID" in df_src.columns else "PatientID"
    pd.DataFrame({
        "PatientID": df_src[id_col],
        "RSI":       y,
        "GNN_Prob":  probs,
        "GNN_Pred":  (probs >= 0.5).astype(int),
    }).to_csv(OUT_GNN / f"predictions_GNN_{name}_stability.csv",
              index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# Performance table
# ─────────────────────────────────────────────────────────────────────────────
perf_rows = []
for cohort_label, mg, en_df, y in [
    ("PUTH (Train)",      mg_puth, en_puth, y_puth),
    ("JSPH-RN (Train)",   mg_jsph, en_jsph, y_jsph),
    ("PUCH (Int.Test)",   mg_puch, en_puch, y_puch),
    ("TCGA (Ext.Val)",    mg_tcga, en_tcga, y_tcga),
]:
    la = auc_score(y, en_df.RSI_Prob.values)
    llo, lhi = boot_ci(y, en_df.RSI_Prob.values)
    gnn_row = {"Cohort": cohort_label, "Model": "GNN-PSN"}
    gnn_row.update({k: round(v, 3) if isinstance(v, float) else v
                    for k, v in mg.items() if k != "Model"})
    perf_rows.append(gnn_row)
    perf_rows.append({
        "Cohort":   cohort_label,
        "Model":    "ElasticNet",
        "AUC":      round(la, 3),
        "CI_lo":    round(llo, 3),
        "CI_hi":    round(lhi, 3),
        "AUC_str":  f"{la:.3f} ({llo:.3f}-{lhi:.3f})",
    })

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(OUT_GNN / "GNN_performance_stability.csv",
               index=False, encoding="utf-8-sig")

print(f"\n{'='*65}")
print("Performance Summary (Stability Edition)")
print(f"  * PUTH / JSPH-RN = in-sample; PUCH / TCGA = external")
print(f"{'='*65}")
for _, row in perf_df.iterrows():
    print(f"  {row.Cohort:22s}  {row.Model:12s}  {row.get('AUC_str', row.AUC)}")


# ─────────────────────────────────────────────────────────────────────────────
# Training curve
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_aucs, label="Train AUC", color="#2196F3", lw=1.5)
ax.plot(val_aucs,   label="Val AUC",   color="#E91E63", lw=1.5)
ax.axvline(best_epoch - 1, color="gray", lw=1, linestyle="--",
           label=f"Best val epoch {best_epoch}")
ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
ax.set_title("GNN-PSN Stability Training Curve\n"
             "(Train = PUTH+JSPH-RN, Val = 20% stratified)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_GNN / "training_curve_stability.png", dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ROC comparison: GNN-PSN vs Elastic Net
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()
roc_sets = [
    ("PUTH (Training, in-sample)",
     y_puth, prob_gnn_puth, en_puth.RSI_Prob.values),
    ("PUCH (Internal Test)",
     y_puch, prob_gnn_puch, en_puch.RSI_Prob.values),
    ("TCGA (External Val)",
     y_tcga, prob_gnn_tcga, en_tcga.RSI_Prob.values),
    ("JSPH-RN (Training, in-sample)",
     y_jsph, prob_gnn_jsph, en_jsph.RSI_Prob.values),
]
for ax, (title, yt, pg, pe) in zip(axes2, roc_sets):
    for yp, label, color, ls in [
        (pg, "GNN-PSN",     "#E91E63", "-"),
        (pe, "Elastic Net", "#2196F3", "--"),
    ]:
        fpr, tpr = roc_curve_np(yt, yp)
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

fig2.suptitle(
    "Stability Edition: GNN-PSN vs Elastic Net (Bootstrap Stable Features)\n"
    "Train: PUTH+JSPH-RN (576)  |  External: PUCH, TCGA  |  "
    "*in-sample AUC is optimistic for training cohorts",
    fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_GNN / "GNN_vs_ElasticNet_ROC_stability.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: GNN_vs_ElasticNet_ROC_stability.png")


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Step 4 Stability Complete.")
print(f"{'='*65}")
print(f"  Output: {OUT_GNN}")
print(f"  graphsage_best_stability.pt")
print(f"  GNN_performance_stability.csv")
print(f"  GNN_vs_ElasticNet_ROC_stability.png")
print(f"  training_curve_stability.png")
print(f"\nKey results (external sets):")
for row in perf_df[perf_df.Cohort.isin(["PUCH (Int.Test)", "TCGA (Ext.Val)"])].itertuples():
    print(f"  {row.Cohort:22s}  {row.Model:12s}  {row.AUC_str}")
print(f"\nNext: python step5_dca_delong.py  (DeLong test + Decision Curve)")
