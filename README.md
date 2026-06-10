# RSI-Radiomics: Multi-centre CT Radiomics for Renal Sinus Invasion Prediction

This repository contains the source code accompanying the manuscript:

> **Multi-Centre CT Radiomics for Preoperative Prediction of Renal Sinus Invasion in Clear Cell Renal Cell Carcinoma: A Bootstrap-Stabilised Elastic Net and Graph Neural Network Approach with Radiogenomic Validation**  
> Ye Yan†, Chao Liang†, Peichen Duan†, Yichang Hao†, Yunhe Guan, Haibin Zhu*, Pengfei Shao*, Shudong Zhang*  
> *eClinicalMedicine* (under review)

†Equal contribution. *Co-corresponding authors.

---

## Pipeline Overview

The analysis pipeline is organised as sequential steps under `v2_pipeline/`:

| Script | Description |
|--------|-------------|
| `step1_merge_labels.py` | Merge RSI labels across four cohorts (PUTH, JSPH-RN, PUCH, TCGA/CPTAC3) |
| `step2_batch_effect.py` | Inter-scanner batch effect assessment and z-score normalisation |
| `step3_stability.py` | Bootstrap stability selection (200 iterations, threshold ≥ 0.50) |
| `step3_feature_selection.py` | Final 29-feature selection and coefficient extraction |
| `step4_gnn_train.py` | GraphSAGE graph neural network training (k=5 KNN graph) |
| `step4_gnn_stability.py` | GNN stability analysis across bootstrap seeds |
| `step5_dca_delong.py` | DeLong AUC comparison and Decision Curve Analysis |
| `step5b_sensitivity.py` | Sensitivity / specificity analysis and calibration |
| `step6_shap.py` | SHAP LinearExplainer feature attribution (all four cohorts) |
| `step6b_shap_supplement.py` | Sphericity quartile stratification and waterfall plots |
| `step7_radiogenomics_v2.py` | GSEA (Hallmark), ssGSEA pathway correlation, genome-wide gene correlation |
| `step7b_radiogenomics_heatmap.py` | Radiomic–transcriptomic heatmap visualisation |

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.10. nnU-Net v2 must be installed separately following the [official instructions](https://github.com/MIC-DKFZ/nnUNet).

## Data Availability

Clinical and imaging data are available upon reasonable request to the corresponding author (S. Zhang; shudong_zhang_PUTH@outlook.com). Public TCGA-KIRC and CPTAC-3 data are freely available at [The Cancer Imaging Archive](https://www.cancerimagingarchive.net).

## Citation

If you use this code, please cite:

```
[Citation to be added upon acceptance]
```

## Licence

MIT Licence — see [LICENSE](LICENSE) for details.
