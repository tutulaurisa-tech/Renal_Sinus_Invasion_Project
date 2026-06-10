# -*- coding: utf-8 -*-
"""
V2 Pipeline — Step 1: Merge Features + Labels
==============================================
Input:
  RadiomicsFeatures_Center1_PUTH_v2.csv   (1316 features + diagnostics_*)
  RadiomicsFeatures_Center2_PUCH_v2.csv
  RadiomicsFeatures_Center4_TCGA_v2.csv
  JSPH_RN_ALL_radiomics.csv               (already clean, 1316 features)

Output:
  All_Centers_Radiomics_v2_withLabel.csv  (PUTH+PUCH+TCGA, 606 cases)
  JSPH_RN_v2_withLabel.csv                (231 cases, for external val)
  merge_report_v2.txt                     (feature counts, missing cases)

Usage:
    conda activate radiomics
    cd D:\\RSI_Project_Workspace\\4_Scripts\\v2_pipeline
    python step1_merge_labels.py
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(r"D:\RSI_Project_Workspace")
FEAT_DIR  = BASE / "3_Extracted_Features" / "Radiomics_CSVs"
LABEL_DIR = BASE / "2_Clinical_Labels"
OUT_DIR   = BASE / "3_Extracted_Features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = BASE / "5_Results" / "v2"
REPORT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("V2 Pipeline — Step 1: Merge Features + Labels")
print("=" * 60)

# ── Image type prefixes (used throughout pipeline) ────────────────────────
IMG_TYPES = ["original", "logarithm", "exponential", "square",
             "squareroot", "gradient", "wavelet"]


SKIP_EXACT    = {"Case_ID", "PatientID", "Center", "RSI", "Cohort"}
SKIP_PREFIX   = "diagnostics_"

def drop_diagnostics(df):
    diag = [c for c in df.columns if c.startswith(SKIP_PREFIX)]
    return df.drop(columns=diag)


def feature_cols(df):
    """Return all radiomics feature columns (handles wavelet-LLL_ hyphen format)."""
    return [c for c in df.columns
            if c not in SKIP_EXACT and not c.startswith(SKIP_PREFIX)]


def img_type_of(feat):
    if feat.startswith("wavelet"):          # wavelet-LLL_, wavelet-LLH_, …
        return "wavelet"
    for t in IMG_TYPES:
        if feat.startswith(t + "_"):
            return t
    return "unknown"


def count_by_type(cols):
    from collections import Counter
    ct = Counter(img_type_of(c) for c in cols)
    return dict(ct)


# ══════════════════════════════════════════════════════════════════════════
# 1.  Load v2 CSVs for PUTH / PUCH / TCGA
# ══════════════════════════════════════════════════════════════════════════
centers_cfg = {
    "PUTH": {
        "feat_csv":    FEAT_DIR / "RadiomicsFeatures_Center1_PUTH_v2.csv",
        "label_src":   LABEL_DIR / "PUTH_RSI_LABEL.xlsx",
        "label_id":    "原数据库序号",
        "label_rsi":   "侵犯肾窦",
    },
    "PUCH": {
        "feat_csv":    FEAT_DIR / "RadiomicsFeatures_Center2_PUCH_v2.csv",
        "label_src":   LABEL_DIR / "PUCH_RSI_LABEL.xlsx",
        "label_id":    "ID",
        "label_rsi":   "LABEL",
    },
    "TCGA": {
        "feat_csv":    FEAT_DIR / "RadiomicsFeatures_Center4_TCGA_v2.csv",
        "label_src":   LABEL_DIR / "TCGA_RSI_LABEL.csv",
        "label_id":    "PatientID",
        "label_rsi":   "RSI",
    },
}

report_lines = []
dfs = []

for center, cfg in centers_cfg.items():
    feat_path = cfg["feat_csv"]
    if not feat_path.exists():
        msg = f"  !! {center}: {feat_path.name} NOT FOUND — skipping"
        print(msg); report_lines.append(msg)
        continue

    df = pd.read_csv(feat_path)
    df = drop_diagnostics(df)
    df["Case_ID"] = df["Case_ID"].astype(str)

    # Load label
    lsrc = cfg["label_src"]
    ldf  = pd.read_excel(lsrc) if str(lsrc).endswith(".xlsx") else pd.read_csv(lsrc)
    ldf  = ldf.rename(columns={cfg["label_id"]: "Case_ID",
                                cfg["label_rsi"]: "RSI"})
    ldf["Case_ID"] = ldf["Case_ID"].astype(str)

    # Merge
    n_before = len(df)
    df = df.merge(ldf[["Case_ID", "RSI"]], on="Case_ID", how="left")
    n_missing_rsi = df["RSI"].isna().sum()
    df["Center"] = center

    fcols = feature_cols(df)
    msg = (f"  {center}: {len(df)} cases  "
           f"RSI+={int(df.RSI.sum())}  "
           f"missing_RSI={n_missing_rsi}  "
           f"features={len(fcols)}")
    print(msg); report_lines.append(msg)
    dfs.append(df)


# ══════════════════════════════════════════════════════════════════════════
# 2.  Find common feature set across all loaded centers
# ══════════════════════════════════════════════════════════════════════════
if not dfs:
    print("ERROR: No center data loaded. Run extraction first.")
    exit(1)

all_feat_sets = [set(feature_cols(d)) for d in dfs]
common_feats  = sorted(set.intersection(*all_feat_sets))
print(f"\nCommon features across {len(dfs)} centers: {len(common_feats)}")

by_type = count_by_type(common_feats)
for t, n in sorted(by_type.items()):
    print(f"  {t:12s}: {n}")
report_lines.append(f"\nCommon features: {len(common_feats)}")
report_lines.extend([f"  {t}: {n}" for t, n in sorted(by_type.items())])

# ── Check against JSPH-RN feature set ─────────────────────────────────────
jsph_all = pd.read_csv(FEAT_DIR / "JSPH_RN_ALL_radiomics.csv", nrows=1)
jsph_feats = feature_cols(jsph_all)
only_in_3ctr  = set(common_feats) - set(jsph_feats)
only_in_jsph  = set(jsph_feats)  - set(common_feats)
final_feats   = sorted(set(common_feats) & set(jsph_feats))

print(f"\nJSPH-RN features: {len(jsph_feats)}")
print(f"  Only in 3-center: {len(only_in_3ctr)}")
print(f"  Only in JSPH-RN:  {len(only_in_jsph)}")
print(f"  FINAL shared:     {len(final_feats)}")

if only_in_3ctr:
    print(f"  3-center-only examples: {list(only_in_3ctr)[:3]}")
if only_in_jsph:
    print(f"  JSPH-only examples:     {list(only_in_jsph)[:3]}")

report_lines += [
    f"\nFinal shared features (all 4 centers): {len(final_feats)}",
    f"Dropped (3-center only): {len(only_in_3ctr)}",
    f"Dropped (JSPH-only):     {len(only_in_jsph)}",
]


# ══════════════════════════════════════════════════════════════════════════
# 3.  Build All_Centers_Radiomics_v2_withLabel.csv  (PUTH + PUCH + TCGA)
# ══════════════════════════════════════════════════════════════════════════
META_COLS = ["Case_ID", "Center", "RSI"]
keep_cols = META_COLS + final_feats

combined = pd.concat(
    [d[keep_cols] for d in dfs if all(c in d.columns for c in META_COLS)],
    ignore_index=True
)

# Drop rows with NaN RSI
n_before = len(combined)
combined = combined.dropna(subset=["RSI"]).reset_index(drop=True)
combined["RSI"] = combined["RSI"].astype(int)
print(f"\nCombined (PUTH+PUCH+TCGA): {len(combined)} cases "
      f"(dropped {n_before-len(combined)} missing RSI)")
for ctr in combined.Center.unique():
    sub = combined[combined.Center == ctr]
    print(f"  {ctr}: n={len(sub)}  RSI+={sub.RSI.sum()}")

out_3ctr = OUT_DIR / "All_Centers_Radiomics_v2_withLabel.csv"
combined.to_csv(out_3ctr, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_3ctr.name}  ({combined.shape})")
report_lines.append(f"\nAll_Centers_v2: {combined.shape}")


# ══════════════════════════════════════════════════════════════════════════
# 4.  Build JSPH_RN_v2_withLabel.csv
# ══════════════════════════════════════════════════════════════════════════
jsph_df   = pd.read_csv(FEAT_DIR / "JSPH_RN_ALL_radiomics.csv")
# v3: 使用重新病理评估的统一标准标签（剔除RN-157尿毒症萎缩肾）
jsph_lbl  = pd.read_csv(LABEL_DIR / "JSPH_RN_RSI_LABEL_v3_clean.csv")
# jsph_lbl 已含 Case_ID / RSI 两列，无需 rename

jsph_df["Case_ID"] = jsph_df["Case_ID"].astype(str)
jsph_lbl["Case_ID"] = jsph_lbl["Case_ID"].astype(str)

jsph_merged = jsph_df.merge(jsph_lbl[["Case_ID", "RSI"]],
                             on="Case_ID", how="inner")
jsph_merged = jsph_merged.dropna(subset=["RSI"]).reset_index(drop=True)
jsph_merged["RSI"] = jsph_merged["RSI"].astype(int)

# Keep only final shared features
jsph_keep = ["Case_ID"] + final_feats + ["RSI"]
jsph_merged = jsph_merged[[c for c in jsph_keep if c in jsph_merged.columns]]

out_jsph = FEAT_DIR / "JSPH_RN_v2_withLabel.csv"
jsph_merged.to_csv(out_jsph, index=False, encoding="utf-8-sig")
print(f"Saved: {out_jsph.name}  ({jsph_merged.shape})")
print(f"  JSPH-RN: n={len(jsph_merged)}  "
      f"RSI+={jsph_merged.RSI.sum()}  "
      f"RSI-={(jsph_merged.RSI==0).sum()}")
report_lines.append(f"JSPH_RN_v2: {jsph_merged.shape}")


# ══════════════════════════════════════════════════════════════════════════
# 5.  Check for NaN values
# ══════════════════════════════════════════════════════════════════════════
print("\n--- NaN check ---")
for name, df in [("3-center", combined), ("JSPH-RN", jsph_merged)]:
    feat_df = df[final_feats]
    nan_cols = feat_df.columns[feat_df.isna().any()].tolist()
    nan_rows = feat_df.isna().any(axis=1).sum()
    print(f"  {name}: {nan_rows} rows with NaN  |  {len(nan_cols)} cols with NaN")
    if nan_cols:
        print(f"    First NaN cols: {nan_cols[:5]}")
    report_lines.append(f"{name} NaN rows: {nan_rows}")


# ══════════════════════════════════════════════════════════════════════════
# 6.  Save feature list + report
# ══════════════════════════════════════════════════════════════════════════
feat_list_df = pd.DataFrame({"Feature": final_feats})
feat_list_df["ImageType"] = feat_list_df.Feature.apply(
    lambda x: next((t for t in IMG_TYPES if x.startswith(t + "_")), "unknown"))
feat_list_df.to_csv(REPORT_PATH / "feature_list_v2.csv",
                    index=False, encoding="utf-8-sig")

with open(REPORT_PATH / "merge_report_v2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n{'='*60}")
print("Step 1 Complete.")
print(f"{'='*60}")
print(f"  {out_3ctr}")
print(f"  {out_jsph}")
print(f"  feature_list_v2.csv  ({len(final_feats)} features)")
print(f"\nNext: python step2_batch_effect.py")
