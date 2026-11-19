# Model card

**For education only. Not for clinical use.** The classifier produces *screening flags*, not diagnoses. Every number
below is generated from the files in `reports/` (written by the DVC pipeline) and from the running demo.

## 1. Chest X-ray classifier

### Model

| | |
|---|---|
| Architecture | DenseNet121 (ImageNet weights), new linear head 1024 → 14 logits |
| Input | one frontal chest X-ray, grayscale resized to 224 x 224, copied to 3 channels, ImageNet normalisation |
| Output | 14 logits (one per finding) and one Grad-CAM heatmap per flagged label |
| Decision rule | a label is flagged when `logit >= threshold`; thresholds per label, chosen on validation (Youden's J) |
| Training | `BCEWithLogitsLoss` with per-label `pos_weight`, AdamW (lr 1e-4, weight decay 1e-5), `ReduceLROnPlateau` (factor 0.1, patience 1), batch 32, 10 epochs, Apple M4 (MPS) |
| Selected checkpoint | epoch 9 (best validation mean AUROC 0.8285) |
| Checkpoint | `models/densenet121_best.pt`, sha256 `2db1b3f90229b5b2fd3366c6dab4d1acb1464357358a9540ff88bd22667d059e` (release `assets-v1`) |
| Serving | CPU: 0.35 s/image on the Mac, about 1.1-1.3 s/image in Docker |

### Intended use

- Teaching and demonstrating a complete imaging-AI workflow: DICOM storage, inference, explanation, human review and
  an audit trail.
- **Out of scope:** any clinical use, triage or diagnosis; images other than adult frontal chest X-rays; populations or
  hospitals unlike the NIH Clinical Center data.

### Data

NIH ChestX-ray14, split by patient: train 69,625 / val 16,899 / test 25,596 images (the
official NIH test list). Details in [`DATA.md`](DATA.md).

### Performance per label (test unless marked)

| Label | Val AUROC | Test AUROC | Test positives | Threshold (logit) | Sensitivity | Specificity | PPV |
|---|---|---|---|---|---|---|---|
| Hernia | 0.894 | 0.935 | 86 (0.3%) | -3.81 | 0.85 | 0.87 | 0.021 |
| Emphysema | 0.878 | 0.902 | 1,093 (4.3%) | -0.80 | 0.86 | 0.78 | 0.147 |
| Cardiomegaly | 0.917 | 0.890 | 1,069 (4.2%) | -1.50 | 0.90 | 0.68 | 0.110 |
| Pneumothorax | 0.849 | 0.844 | 2,665 (10.4%) | -0.70 | 0.85 | 0.68 | 0.235 |
| Edema | 0.911 | 0.842 | 925 (3.6%) | -0.54 | 0.92 | 0.58 | 0.075 |
| Effusion | 0.897 | 0.820 | 4,658 (18.2%) | -0.04 | 0.89 | 0.56 | 0.312 |
| Fibrosis | 0.783 | 0.816 | 435 (1.7%) | -0.20 | 0.72 | 0.77 | 0.052 |
| Mass | 0.848 | 0.812 | 1,748 (6.8%) | -0.32 | 0.81 | 0.63 | 0.140 |
| Pleural_Thickening | 0.804 | 0.764 | 1,143 (4.5%) | -0.11 | 0.77 | 0.63 | 0.089 |
| Atelectasis | 0.810 | 0.759 | 3,279 (12.8%) | -0.01 | 0.80 | 0.57 | 0.216 |
| Consolidation | 0.808 | 0.750 | 1,815 (7.1%) | -0.18 | 0.93 | 0.38 | 0.102 |
| Nodule | 0.755 | 0.736 | 1,623 (6.3%) | -0.04 | 0.68 | 0.66 | 0.118 |
| Pneumonia | 0.748 | 0.699 | 555 (2.2%) | 0.33 | 0.65 | 0.65 | 0.040 |
| Infiltration | 0.695 | 0.694 | 6,112 (23.9%) | 0.15 | 0.77 | 0.51 | 0.328 |

**Mean AUROC: validation 0.8285, test 0.8045.** Validation chose both
the checkpoint and the thresholds, so it is optimistic. The test set is also sicker and differently imaged: Effusion
prevalence 18.2% vs 10.0% on validation, and AP (bedside) views 57% vs 34%.

**Thresholds favour sensitivity.** Youden's J balances sensitivity and specificity, so for rare findings most flags are
false positives (low PPV, e.g. Hernia 0.021). PPV also depends on prevalence, so it would
change in another population. The 42 demo images get a median of 6 flags (range 0-10).
The logits are not calibrated probabilities.

### Subgroups (test split)

Mean test AUROC over the labels with at least 30 positive images in that subgroup:

| Attribute | Group | Images | Mean AUROC | Labels |
|---|---|---|---|---|
| sex | F | 10,714 | 0.805 | 14 |
| sex | M | 14,882 | 0.804 | 14 |
| age | <40 | 8,290 | 0.801 | 13 |
| age | 40-59 | 11,224 | 0.793 | 13 |
| age | 60+ | 6,078 | 0.796 | 14 |
| view | AP | 14,500 | 0.769 | 13 |
| view | PA | 11,096 | 0.798 | 14 |

No gap by sex and a slight decline with age. Only sex, age and view are available in the dataset (no ethnicity, no
scanner information).

**View-position shortcut:** the Edema logit separates AP from PA views (AUROC 0.862) better than it separates Edema from
no Edema (test AUROC 0.842). AP images come from bedside exams of sicker patients, so the
model partly learned "AP view" as a sign of disease.

### Explanations: Grad-CAM localisation check

Grad-CAM on the last DenseNet feature block (7 x 7, upsampled), so the maps are coarse. Check against the
984 radiologist boxes in the test split: a *hit* means the heatmap peak lies within 15 px of the
box. The *shuffled* baseline uses the heatmap of another image with the same box; *lift* is the mean heat inside the
box relative to the whole image.

Overall: **51% hits vs 36% for the shuffled baseline.**

| Box label flagged? | Boxes | Hit | Shuffled hit | Lift | Shuffled lift |
|---|---|---|---|---|---|
| not flagged | 165 | 19% | 13% | 0.45 | 0.23 |
| flagged | 819 | 57% | 40% | 2.61 | 1.80 |

| Label | Boxes | Flagged | Hit | Shuffled hit | Lift | Shuffled lift |
|---|---|---|---|---|---|---|
| Cardiomegaly | 146 | 96% | 98% | 92% | 2.87 | 2.56 |
| Infiltration | 123 | 81% | 57% | 38% | 1.93 | 1.49 |
| Effusion | 153 | 97% | 56% | 31% | 2.37 | 1.51 |
| Pneumonia | 120 | 65% | 52% | 38% | 1.83 | 1.32 |
| ALL | 984 | 83% | 51% | 36% | 2.46 | 1.59 |
| Mass | 85 | 81% | 49% | 24% | 3.52 | 1.82 |
| Atelectasis | 180 | 81% | 36% | 21% | 2.48 | 1.56 |
| Pneumothorax | 98 | 81% | 21% | 15% | 1.55 | 0.50 |
| Nodule | 79 | 73% | 15% | 8% | 2.39 | 1.19 |

The heatmap points to the right region far more often when the model also flags the label. It is weakest for small
findings (Nodule). A heatmap in the right place does not prove the model's reasoning is right.

## 2. Spleen segmentation (results only, not served)

MONAI 3D U-Net (4.8 M parameters), Dice loss, MSD Task09 Spleen, 33 training / 8
validation volumes (seed 42), 600 epochs on a Kaggle Tesla T4.
Dice is measured resampled 1.5x1.5x2.0 mm, spleen only, hard (argmax).

| Checkpoint | Mean Dice | Notes |
|---|---|---|
| Best (epoch 425) | 0.942 | SD 0.010, worst volume 0.926 |
| Plateau (epoch 400+, 40 evaluations) | median 0.895 | IQR 0.875-0.906 |
| Last (epoch 600) | 0.823 | weak volumes: `spleen_26` 0.26, `spleen_49` 0.65 |

With 8 validation volumes that also chose the checkpoint and no test set (MSD test labels are not public), the best
Dice is optimistic; the plateau median is the more honest number.

## 3. Limitations and bias (complete list)

**Model and data**
- Not a medical device. Labels are text-mined from reports and noisy; all images come from one hospital and are
  frontal views only.
- The view-position shortcut (above) means part of the signal is "how the image was taken", not the disease.
- Thresholds were tuned on validation and transfer imperfectly to test; specificity drops for several labels.
- Low PPV for rare findings; the flags are for screening a worklist, not for diagnosis.
- Grad-CAM is coarse (7 x 7) and shows where the evidence is, not why.
- Demo DICOMs are NIH PNGs converted to 8-bit Secondary Capture; real scanner DICOMs are untested.
- Subgroup analysis is limited to sex, age and view.
- Spleen results: small validation set used for selection, no test set.

**System**
- No authentication: the reviewer name is self-declared; ports are bound to `127.0.0.1` only.
- No idempotency key: two separate POSTs record two decisions (the viewer locks its form after a save).
- Removing rows from the end of the audit log is only detected against an externally kept anchor hash.
- The worker retries every 4xx answer forever, and its "processed N" log line undercounts after a transient error.
- The worklist is not paginated; the API logs browser-cancelled requests as errors.
- One nginx request hang was observed once and not reproduced; requests to the API are now bounded by 2 s timeouts.
- Inference runs on CPU in Docker (about 1.1-1.3 s per image).
