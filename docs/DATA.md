# Data

For education only, not for clinical use. This page describes every dataset the project uses, how it is split and
preprocessed, which pipeline stage produces which file, and what is shipped for the demo.

## 1. NIH ChestX-ray14 (classification)

- **Source:** NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC (Kaggle mirror `nih-chest-xrays/data`,
  about 45 GB). Citation in the README. Terms: see the dataset's own README.
- **Content:** 112,120 frontal chest X-rays (PNG, 1024 x 1024) of 30,805 patients, with 14 finding labels
  text-mined from the radiology reports (so the labels are noisy) or "No Finding".
- **Location:** `data/raw/nih/`, tracked by DVC. There is no DVC remote: the data is re-downloaded from the source
  (README, *Full reproduction*).
- **Files used:** `Data_Entry_2017.csv` (labels, patient id, age, sex, view position), `test_list.txt` (the official
  test split) and `BBox_List_2017.csv` (984 radiologist bounding boxes, used only for the Grad-CAM localisation check).

### Splits (`make_splits` stage, output `data/processed/splits.csv`)

| Split | Images | Patients | How it is chosen |
|---|---|---|---|
| train | 69,625 | 22,406 | the rest of the non-test images |
| val | 16,899 | 5,602 | 20 % of the non-test **patients** (`GroupShuffleSplit`, seed 42) |
| test | 25,596 | 2,797 | the official NIH `test_list.txt` |

No patient appears in more than one split. Validation is used for checkpoint selection (best epoch) and for the per-label thresholds; the test
split is used only for the final numbers.

### Preprocessing

- `resize` stage: each PNG is converted to 8-bit grayscale and LANCZOS-resized to 256 x 256
  (`data/processed/images_resized/`, parameter `resize.size`).
- Training: random affine transform (+-7 degrees rotation, +-5 % shift, 0.95-1.05 scale), resize to 224, grayscale copied
  to 3 channels, ImageNet normalisation. Evaluation and serving use the same steps without the random transform.

### Pipeline (`dvc.yaml`)

| Stage | Command | Main inputs | Outputs |
|---|---|---|---|
| `resize` | `python -m src.data.resize_images` | `data/raw/nih` | `data/processed/images_resized/` |
| `make_splits` | `python -m src.data.make_splits` | `Data_Entry_2017.csv`, `test_list.txt` | `data/processed/splits.csv` |
| `train` | `python -m src.train.train` | splits, resized images | `models/densenet121_best.pt`, `reports/train_metrics.json`, `reports/train_history.csv` |
| `predict` | `python -m src.evaluation.predict` | checkpoint, splits, resized images | `data/processed/predictions/` |
| `eval_metrics` | `python -m src.evaluation.metrics` | predictions, splits | `reports/eval_metrics.json` |
| `thresholds` | `python -m src.evaluation.thresholds` | predictions, splits | `reports/thresholds.json` |
| `subgroups` | `python -m src.evaluation.subgroups` | predictions, splits | `reports/subgroup_metrics.json` |
| `localization` | `python -m src.explain.localization` | checkpoint, thresholds, `BBox_List_2017.csv`, resized images | `reports/gradcam_localization.json`, `reports/gradcam_localization_boxes.csv` |

Parameters (`params.yaml`): `splits.val_frac 0.2`, `splits.seed 42`, `resize.size 256`; `train`: 10 epochs,
batch size 32, lr 1e-4, weight decay 1e-5, `ReduceLROnPlateau` with factor 0.1 and patience 1, seed 42.

- The small reports are tracked in git (`cache: false`); images, predictions and the checkpoint live in the DVC cache.
- Stages also list the source files they import constants from, so changing such a constant re-runs the stage.
- On macOS, delete Finder's `.DS_Store` files in `data/raw/nih` before `dvc repro`: DVC hashes every file in the folder.

## 2. Demo set (Orthanc)

**Seeded subset**, `data/demo/demo_subset.csv` (40 images of 37 patients, seed 42):

| Group | Images | Meaning |
|---|---|---|
| `box` | 20 | images with a radiologist bounding box (`box_labels`) |
| `no_finding` | 10 | labelled "No Finding" |
| `random` | 10 | random picks |

View: AP 19, PA 21. Sex: F 15, M 25. Age 21-72 (median 57).

**Two extra images,** `00000003_000` and `00000003_001` (one patient), were used for the first Orthanc test and are
kept in the demo. `00000003_000` is the *Try it* example in the README. All 42 demo images are from the **test** split, so none of them was used for training, checkpoint selection or threshold choice.

**Conversion to DICOM** (`src/data/`): DICOM Secondary Capture, 8-bit MONOCHROME2, deterministic UIDs (converting the
same image again gives the same UIDs, so Orthanc answers `AlreadyStored`), `SeriesDescription` = the NIH image name,
which the worker reports as `image_index`. These are converted PNGs, not scanner output (see the README limitations).

**Release `assets-v1`** (downloaded by `scripts/fetch_assets.sh`):

| Asset | Unpacked to | Notes |
|---|---|---|
| `densenet121_best.pt` | `models/densenet121_best.pt` | classifier checkpoint (about 27 MB), built into the worker image |
| `demo_dicom.tar.gz` | `data/demo/dicom/*.dcm` | the 42 demo DICOMs |

`assets/SHA256SUMS` lists all 43 files at their destination paths. `fetch_assets.sh` verifies them with `sha256sum`
or `shasum`, skips the download if they already verify, and fails if any file does not match.
`scripts/load_demo.sh` uploads the 42 files with the login from `.env` and prints how many were new
(`Success`) and how many Orthanc already had (`AlreadyStored`).

## 3. MSD Task09 Spleen (segmentation, results only)

- **Source:** Medical Segmentation Decathlon, Task09 Spleen: 41 labelled abdominal CT volumes. Citation in the
  README.
- **Split:** 33 training / 8 validation volumes; no separate test set (the validation set also chose the checkpoint).
- **Training:** `notebooks/04_spleen_unet.ipynb` on a Kaggle T4. The raw data is not in the repository; the trained
  model is tracked by DVC in `models/spleen/` (3 files, 38.6 MB) but is not published, because there is no DVC remote.
- **Results:** `reports/spleen_metrics.json`, `spleen_failures.csv`, `spleen_curve.png`, `spleen_errors.png`,
  `spleen_overlays.png`.

## 4. Privacy

The NIH images were de-identified by the provider; patient ids are dataset ids, not hospital ids. The demo DICOMs
are generated from these images and their NIH metadata. Nothing in this repository comes from real patients of this
project.
