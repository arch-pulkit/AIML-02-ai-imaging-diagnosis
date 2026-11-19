# Architecture

AIML-02 · AI-Assisted Medical Imaging & Diagnosis System. **For education only. Not for clinical use.**

This document describes the system in ten diagrams. Each one names the files it was drawn from, so it can be checked
against the code. All diagrams are Mermaid, which GitHub renders.

1. [System overview](#1-system-overview)
2. [Data flow, level 0 (context)](#2-data-flow-level-0-context)
3. [Data flow, level 1 (serving)](#3-data-flow-level-1-serving)
4. [Data flow, ML pipeline (offline)](#4-data-flow-ml-pipeline-offline)
5. [Model: DenseNet121 classifier](#5-model-densenet121-classifier)
6. [Model: Grad-CAM](#6-model-grad-cam)
7. [Model: 3D U-Net (spleen segmentation)](#7-model-3d-u-net-spleen-segmentation)
8. [One image end to end (sequence)](#8-one-image-end-to-end-sequence)
9. [Results database (ER diagram)](#9-results-database-er-diagram)
10. [Deployment](#10-deployment)

**Data-flow diagram (DFD) notation used in sections 2–4**

| Shape | Meaning |
|---|---|
| Rectangle | External entity (a person or system outside our control) |
| Rounded box, numbered | Process (transforms data) |
| Cylinder, `D`-numbered | Data store |
| Labelled arrow | Data flow. The label is the data that moves, never a control signal |

## 1. System overview

```mermaid
flowchart LR
  L["scripts/load_demo.sh<br/>(upload DICOM)"] --> O[("Orthanc 1.13<br/>DICOM server :8042")]
  W["Worker (Python, CPU)<br/>DenseNet121 + Grad-CAM"] -- "poll /changes, fetch file" --> O
  W -- "POST /api/results" --> A["Results API (Go)<br/>SQLite + hash-chained audit log :8080"]
  A -- "preview proxy" --> O
  V["Viewer (React, nginx) :3000"] -- "/api/*" --> A
  R(("Reviewer")) --> V
```

| Component | Technology | Responsibility | Source |
|---|---|---|---|
| Orthanc | Orthanc 1.13 (official image) | Stores DICOM; change feed; preview rendering | `compose.yaml` |
| Worker | Python 3.11, PyTorch (CPU) | Follows the change feed, analyses each new instance, POSTs the result | `src/serve/worker.py`, `src/serve/inference.py` |
| Results API | Go standard library, SQLite (`modernc.org/sqlite`) | The single authority for results, decisions and the audit log | `services/api/` |
| Viewer | React 19 + Vite, served by nginx | Worklist, review page, decisions, audit page | `frontend/` |

The browser only ever talks to nginx (port 3000), which serves the built app and forwards `/api/` to the API, so
everything runs on one origin.

## 2. Data flow, level 0 (context)

```mermaid
flowchart LR
  NIH["NIH Clinical Center<br/>(dataset, via Kaggle)"]
  UP["Image sender<br/>(scripts/load_demo.sh)"]
  REV["Reviewer"]
  SYS(["0 · AIML-02 system"])
  NIH -- "112,120 X-ray PNGs, labels, boxes (offline, for training)" --> SYS
  UP -- "DICOM instances" --> SYS
  SYS -- "worklist, X-ray, heatmap overlays, flags" --> REV
  REV -- "decision: approve or reject, comment, name" --> SYS
  SYS -- "audit log, chain verification result" --> REV
```

## 3. Data flow, level 1 (serving)

```mermaid
flowchart LR
  UP["Image sender"]
  REV["Reviewer"]
  P1(["1 · Store DICOM<br/>(Orthanc)"])
  P2(["2 · Analyse image<br/>(worker)"])
  P3(["3 · Record results and decisions<br/>(results API)"])
  P4(["4 · Present for review<br/>(viewer)"])
  P5(["5 · Verify audit chain<br/>(results API)"])
  D1[("D1 · Orthanc store<br/>volume orthanc-db")]
  D2[("D2 · Results DB, SQLite<br/>results, heatmaps, decisions, audit_log<br/>volume api-data")]
  D3[("D3 · Worker cursor<br/>volume worker-state")]
  D4[("D4 · Checkpoint + thresholds<br/>inside the worker image")]

  UP -- "DICOM file" --> P1
  P1 -- "stored instance" --> D1
  D1 -- "change feed, DICOM bytes" --> P2
  D4 -- "weights, per-label thresholds" --> P2
  D3 -- "last processed change" --> P2
  P2 -- "result: logits, flags, heatmap PNGs, model hash (or error)" --> P3
  P2 -- "new cursor, written only after the API accepted" --> D3
  P3 -- "result rows + audit row, one transaction" --> D2
  D1 -- "X-ray preview PNG" --> P3
  D2 -- "summaries, detail, heatmaps, decision history" --> P3
  P3 -- "JSON and PNG over /api" --> P4
  P4 -- "worklist, overlay, flags" --> REV
  REV -- "approve or reject, comment, name" --> P4
  P4 -- "decision" --> P3
  P3 -- "decision row + audit row" --> D2
  D2 -- "audit_log rows" --> P5
  P5 -- "ok + anchor hash, or first broken seq" --> P4
```

| Store | Written by | Read by | Rules |
|---|---|---|---|
| D1 Orthanc store | P1 | P2, P3 (preview proxy) | Stored instances are never overwritten (`OverwriteInstances: false`) |
| D2 Results DB | P3 only | P3, P5 | `ok` results are write-once; heatmaps, decisions and the audit log are append-only (triggers, section 9) |
| D3 Worker cursor | P2 | P2 | Advanced only after a 2xx answer, so delivery is at-least-once |
| D4 Checkpoint + thresholds | image build (`fetch_assets.sh`, `reports/thresholds.json`) | P2 | Its sha256 is sent with every result and stored with it |

## 4. Data flow, ML pipeline (offline)

The edges between stages are exactly those of `dvc dag` (checked by a script). The labels name the artifacts.

```mermaid
flowchart TD
  nih[("data/raw/nih<br/>112,128 files, DVC-tracked")]
  make_splits(["make_splits"])
  resize(["resize"])
  train(["train"])
  predict(["predict"])
  eval_metrics(["eval_metrics"])
  thresholds(["thresholds"])
  subgroups(["subgroups"])
  localization(["localization"])
  r_train[("reports/train_metrics.json<br/>train_history.csv")]
  r_eval[("reports/eval_metrics.json")]
  r_thr[("reports/thresholds.json")]
  r_sub[("reports/subgroup_metrics.json")]
  r_loc[("reports/gradcam_localization.json<br/>+ per-box CSV")]
  worker[("worker image (D4)")]

  nih -->|"labels, official test list"| make_splits
  nih -->|"112,120 PNGs"| resize
  nih -->|"radiologist boxes"| localization
  make_splits -->|"splits.csv"| train
  make_splits -->|"splits.csv"| predict
  make_splits -->|"splits.csv"| eval_metrics
  make_splits -->|"splits.csv"| thresholds
  make_splits -->|"splits.csv"| subgroups
  make_splits -->|"splits.csv"| localization
  resize -->|"256 px PNGs"| train
  resize -->|"256 px PNGs"| predict
  resize -->|"256 px PNGs"| localization
  train -->|"densenet121_best.pt"| predict
  train -->|"densenet121_best.pt"| localization
  predict -->|"logits per image"| eval_metrics
  predict -->|"logits per image"| thresholds
  predict -->|"logits per image"| subgroups
  thresholds -->|"per-label thresholds"| localization
  train -.-> r_train
  eval_metrics -.-> r_eval
  thresholds -.-> r_thr
  subgroups -.-> r_sub
  localization -.-> r_loc
  train -. "checkpoint" .-> worker
  thresholds -. "thresholds.json" .-> worker
```

- Splits are **by patient** (`GroupShuffleSplit`, validation fraction 0.2, seed 42). The official NIH test list is the
  test set: 69,625 train / 16,899 validation / 25,596 test images. Source: `src/data/make_splits.py`, `params.yaml`.
- The spleen U-Net (section 7) is trained separately in `notebooks/04_spleen_unet.ipynb` on Kaggle; its outputs are
  tracked with `dvc add models/spleen`.

## 5. Model: DenseNet121 classifier

```mermaid
flowchart TB
  A["DICOM, 8-bit MONOCHROME2"] --> B["decode to a grayscale image"]
  B --> C["LANCZOS resize to 256 × 256<br/>(same function as the offline resize stage)"]
  C --> D["Resize 224 · grayscale → 3 channels · ImageNet normalisation<br/>tensor 3 × 224 × 224"]
  D --> E["conv0 7×7, stride 2 + max-pool → 64 × 56 × 56"]
  E --> F1["dense block 1 (6 layers) + transition → 128 × 28 × 28"]
  F1 --> F2["dense block 2 (12 layers) + transition → 256 × 14 × 14"]
  F2 --> F3["dense block 3 (24 layers) + transition → 512 × 7 × 7"]
  F3 --> F4["dense block 4 (16 layers) + norm5 → 1024 × 7 × 7<br/>(= model.features, the Grad-CAM layer)"]
  F4 --> G["ReLU → global average pool → 1024"]
  G --> H["Linear 1024 → 14 (new head)"]
  H --> I["14 logits"]
  I --> J{"logit ≥ the label's threshold?"}
  J -- "yes" --> K["flagged"]
  J -- "no" --> L["not flagged"]
```

| | |
|---|---|
| Backbone | torchvision `densenet121` with ImageNet weights; classifier replaced by `Linear(1024, 14)` (`src/models/classifier.py`) |
| Size | about 7.0 M parameters |
| Loss | `BCEWithLogitsLoss` with per-label `pos_weight` = negatives / positives on the training split (`src/train/weights.py`) |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-5; `ReduceLROnPlateau` (factor 0.1, patience 1) (`src/train/train.py`, `params.yaml`) |
| Training | 10 epochs, batch 32, Apple M4 (MPS); best epoch 9 by validation |
| Augmentation | `RandomAffine` only: ±7° rotation, ±5 % shift, 0.95–1.05 scale (`src/data/dataset.py`) |
| Thresholds | one per label, Youden's J on validation (`src/evaluation/thresholds.py` → `reports/thresholds.json`) |

The same resize function and eval transforms are used in training, evaluation and serving (`src/serve/inference.py`),
so the served model sees exactly what it was evaluated on.

## 6. Model: Grad-CAM

```mermaid
flowchart LR
  X["input 3 × 224 × 224"] --> F["model.features"]
  F -- "forward hook: activations A,<br/>cloned, 1024 × 7 × 7" --> A["A"]
  F --> R["ReLU → pool → Linear<br/>→ logit of label k"]
  R -- "backward from logit k;<br/>gradient hook" --> GR["∂ logit k / ∂ A<br/>1024 × 7 × 7"]
  GR --> W["channel weights α<br/>= mean over the 7 × 7 grid → 1024"]
  A --> S["ReLU( Σ α · A ) over channels<br/>→ 7 × 7 map"]
  W --> S
  S --> U["bilinear upsample → 224 × 224<br/>→ 8-bit PNG"]
```

- Own implementation with PyTorch hooks (`src/explain/gradcam.py`). The activations are **cloned** inside the hook,
  because DenseNet applies an in-place ReLU to that tensor straight afterwards.
- Heatmaps are made for every **flagged** label, or for the top-scoring label if none is flagged.
- The 7 × 7 grid explains the blobby maps. Against 984 radiologist boxes, the peak lies within 15 px of the box in 51 %
  of cases, against 36 % for shuffled boxes (`src/explain/localization.py`, `reports/gradcam_localization.json`).

## 7. Model: 3D U-Net (spleen segmentation)

```mermaid
flowchart TB
  IN["CT volume → spacing 1.5 × 1.5 × 2.0 mm → HU window scaled to 0–1<br/>training: random crops balanced around the spleen"] --> E1
  E1["level 1 · 16 channels"] -- "stride 2" --> E2["level 2 · 32 channels"]
  E2 -- "stride 2" --> E3["level 3 · 64 channels"]
  E3 -- "stride 2" --> E4["level 4 · 128 channels"]
  E4 -- "stride 2" --> B["bottom · 256 channels"]
  B -- "up" --> U4["128 channels"]
  U4 -- "up" --> U3["64 channels"]
  U3 -- "up" --> U2["32 channels"]
  U2 -- "up" --> U1["16 channels"]
  U1 --> OUT["2 channels: background, spleen → softmax"]
  E4 -. "skip" .-> U4
  E3 -. "skip" .-> U3
  E2 -. "skip" .-> U2
  E1 -. "skip" .-> U1
```

| | |
|---|---|
| Network | MONAI `UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm=BATCH)`, 4.8 M parameters |
| Data | Medical Segmentation Decathlon Task09 Spleen: 41 CTs, 33 train / 8 validation |
| Training | `DiceLoss(to_onehot_y=True, softmax=True)`, Adam lr 1e-4, 600 epochs on a Kaggle T4 |
| Inference | `sliding_window_inference` over the whole volume |
| Result | best Dice 0.942 (epoch 425), plateau median 0.895, last 0.823 |

Source: `notebooks/04_spleen_unet.ipynb`. This model is evaluated but not served.

## 8. One image end to end (sequence)

```mermaid
sequenceDiagram
  autonumber
  actor S as Image sender
  participant O as Orthanc
  participant W as Worker
  participant A as Results API
  participant DB as SQLite
  participant V as Viewer (nginx)
  actor R as Reviewer
  S->>O: POST /instances (DICOM)
  O-->>S: Success
  W->>O: GET /changes since cursor (polling)
  O-->>W: NewInstance + instance id
  W->>O: GET /instances/id/file
  O-->>W: DICOM bytes
  Note over W: preprocess → DenseNet121 → thresholds → Grad-CAM
  W->>A: POST /api/results (logits, flags, heatmap PNGs, model hash)
  alt API down or 5xx
    A--xW: error
    Note over W: retry after 4, 8, 16 … s, cursor unchanged
  else accepted
    A->>DB: INSERT result + heatmaps + audit row (one transaction)
    A-->>W: 2xx (also for a duplicate)
    Note over W: save the cursor only now (at-least-once)
  end
  R->>V: open the worklist
  V->>A: GET /api/results
  A-->>V: summaries
  R->>V: open one result
  V->>A: GET /api/results/id, /preview, /heatmaps/label
  A->>O: fetch the preview PNG
  A-->>V: detail JSON + PNGs
  R->>V: Approve (name, comment)
  V->>A: POST /api/results/id/decisions
  A->>DB: INSERT decision + audit row (one transaction)
  A-->>V: 201 Created
  R->>V: open the audit page
  V->>A: GET /api/audit/verify and GET /api/audit?order=desc
  A->>DB: recompute every hash in order
  A-->>V: ok + anchor hash (or the first broken seq)
```

## 9. Results database (ER diagram)

```mermaid
erDiagram
  results ||--o{ heatmaps : "has (ok results only)"
  results ||--o{ decisions : "reviewed in"
  results |o..o{ audit_log : "logged in (no foreign key)"
  results {
    TEXT orthanc_instance_id PK
    TEXT status "ok or error"
    TEXT error "set only on error rows"
    TEXT model_name
    TEXT checkpoint_sha256
    TEXT processed_at "worker clock"
    TEXT received_at "API clock"
    TEXT image_index
    TEXT patient_id
    TEXT study_uid
    TEXT series_uid
    TEXT sop_uid
    TEXT logits_json "JSON object"
    TEXT thresholds_json "JSON object"
    TEXT flagged_json "JSON array"
  }
  heatmaps {
    TEXT orthanc_instance_id PK, FK
    TEXT label PK
    BLOB png "PNG signature checked"
  }
  decisions {
    INTEGER id PK
    TEXT orthanc_instance_id FK
    TEXT decision "approve or reject"
    TEXT comment "required for reject"
    TEXT reviewer "must not be blank"
    TEXT created_at
  }
  audit_log {
    INTEGER seq PK
    TEXT at
    TEXT actor "worker or reviewer name"
    TEXT action "result_created, result_replaced, result_duplicate, decision"
    TEXT orthanc_instance_id "nullable"
    TEXT details_json
    TEXT prev_hash "64 hex"
    TEXT hash UK "SHA-256, 64 hex"
  }
```

All tables are `STRICT`, and the rules live in the database itself, not only in Go (`services/api/schema.sql`):

| Rule | Enforced by |
|---|---|
| An `ok` result is write-once; no result is ever deleted | triggers `results_ok_write_once`, `results_no_delete` |
| Heatmaps and decisions only for `ok` results | triggers `heatmaps_require_ok`, `decisions_require_ok` |
| Heatmaps, decisions and audit rows are append-only | `*_no_update` and `*_no_delete` triggers (6) |
| An `ok` row has valid JSON logits, thresholds and flags plus the model hash; an `error` row has an error message | `CHECK` constraints |
| A reject needs a comment; the reviewer name can't be blank | `CHECK` constraints |
| Every audit row links to the previous one | `prev_hash` = the previous row's `hash` (64 zeros for seq 1); `hash` is SHA-256 over the row's contents including `prev_hash` (`services/api/audit.go`) |

`GET /api/audit/verify` recomputes the whole chain. An edited or forged row breaks it at that row. Deleting the newest
rows (truncation) is only detected by comparing against the anchor hash kept outside the system.

## 10. Deployment

```mermaid
flowchart LR
  subgraph host["Host, ports bound to 127.0.0.1 only"]
    BR["Browser"]
    SH["scripts/load_demo.sh"]
  end
  subgraph net["Compose network"]
    O["orthanc<br/>orthancteam/orthanc, pinned<br/>8042 REST + web, 4242 DICOM"]
    W["worker<br/>python:3.11-slim base, CPU, non-root<br/>no ports"]
    A["api<br/>distroless static, uid 65532<br/>8080"]
    F["frontend<br/>nginx-unprivileged, uid 101<br/>8080 inside the container"]
    ST["stub-api<br/>profile stub: captures worker payloads for testing"]
  end
  V1[("volume orthanc-db")] --- O
  V2[("volume worker-state")] --- W
  V3[("volume api-data")] --- A
  BR -- "127.0.0.1:3000" --> F
  BR -. "127.0.0.1:8080, raw JSON" .-> A
  SH -- "127.0.0.1:8042" --> O
  F -- "/api → api:8080" --> A
  A -- "preview → orthanc:8042" --> O
  W -- "change feed, files → orthanc:8042" --> O
  W -- "results → api:8080" --> A
```

- Every base image is pinned by digest (`services/*/Dockerfile`, `frontend/Dockerfile`, `compose.yaml`).
- Start order: the worker and the frontend wait until the API reports healthy.
- nginx resolves `api` at request time (Docker DNS), so a recreated API container is found without restarting nginx.
  Connections time out after 2 s, so a missing API gives a 502 at once.
- The Orthanc login comes from `.env` (see `.env.example`); `docker compose` refuses to start without it.
