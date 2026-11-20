# Reflection

## What I learned

Four things were new to me on this project: DenseNet, DVC, Grad-CAM, and container networking.

**DenseNet.** How a chest X-ray classifier works as a multi-label problem: one output per finding, a weighted loss so rare findings still count, and a separate threshold per finding instead of one cut-off for all.

**DVC.** Data versioned outside git, and a pipeline where each stage records its inputs, so a change reruns only what it affects.

**Grad-CAM.** I implemented it myself. The heatmaps look convincing, and that is the trap. I learned to test them against expert boxes and a shuffled baseline instead of judging them by eye.

**Networking.** Containers find each other by service name, nginx forwards API calls from the viewer, and a recreated container can leave nginx pointing at an old address.

The biggest surprise was the labels. They were mined from radiology reports, not read image by image, so some of them are wrong. A score measures agreement with noisy labels, not with the truth. Now a strong number gets a second look before I trust it.

## What I owned

- **Decisions and scope:** what to build, how to demo it, and what to leave out.
- **Data:** parsing the NIH labels and the patient-level split. This is the decision I'm proudest of. If one patient's images are in both training and test, the model partly recognises people it has already seen, and the results look better than they are. Splitting by patient keeps the evaluation honest.
- **Model:** I chose DenseNet121, the weighted loss and the per-label thresholds, then ran the training, tuned it and evaluated it.
- **Go Results API:** I wrote every Go file: storage, duplicate handling, the hash-chained audit log and its tests.
- **Worker and Docker setup:** I decided how they behave (at-least-once delivery, the cursor saved only after the API accepts a result, pinned images) and debugged them until the stack ran from a fresh clone.
- **Segmentation:** a MONAI U-Net for spleen CT, trained on Kaggle as a second experiment; reported, not served.

AI was leveraged throughout as a researcher: it explained concepts and drafted the React viewer. I ran every step, checked each output against what was expected, and fixed what was wrong. React and nginx are the parts I would still need help to change confidently on my own.

## What I'd do next

- Login and roles, so every decision is tied to a real doctor.
- Idempotent decisions, so a retried request can never record twice.
- Real scanner DICOMs instead of converted PNGs.
- Calibration and confidence intervals, not just point scores.
- Serve the CT model, and bring Python test coverage up to the Go API's level.
- Learn React and nginx properly, so I can own the parts I currently lean on AI for.
