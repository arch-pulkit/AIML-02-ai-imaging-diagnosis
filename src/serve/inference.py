"""
Inference for one DICOM: same preprocessing as training/eval, logits, flags, Grad-CAM.

For education only, not clinical use.
"""

import io
import json
from dataclasses import dataclass

import numpy as np
import pydicom
import torch
import yaml
from PIL import Image

from src.data.dataset import build_transforms
from src.data.nih import LABELS, PROJECT_ROOT
from src.evaluation.thresholds import THRESHOLDS
from src.explain.gradcam import gradcam

PARAMS = PROJECT_ROOT / "params.yaml"


@dataclass
class Result:
    logits: dict[str, float]  # label -> raw logit
    flagged: list[str]  # labels with logit >= threshold
    heatmaps: dict[str, np.ndarray]  # label -> (224, 224) Grad-CAM in [0, 1]


def offline_size() -> int:
    """
    Size of the offline resize (resize stage), read from params.yaml.
    """
    return int(yaml.safe_load(PARAMS.read_text())["resize"]["size"])


def load_thresholds() -> dict[str, float]:
    data = json.loads(THRESHOLDS.read_text())
    assert set(data) == set(LABELS), f"thresholds.json labels differ: {sorted(data)}"
    return {label: float(data[label]["threshold_logit"]) for label in LABELS}


def dicom_to_image(data: bytes) -> tuple[Image.Image, dict]:
    """
    Decode DICOM bytes to an 8-bit grayscale PIL image plus identifiers.
    """
    ds = pydicom.dcmread(io.BytesIO(data))
    photometric = ds.get("PhotometricInterpretation")
    bits = ds.get("BitsStored")
    assert photometric == "MONOCHROME2" and bits == 8, (
        f"only 8-bit MONOCHROME2 is supported (demo data), got {photometric}, {bits} bits"
    )
    arr = ds.pixel_array
    assert arr.dtype == np.uint8 and arr.ndim == 2, (arr.dtype, arr.shape)
    meta = {
        "image_index": str(ds.get("SeriesDescription", "")),
        "patient_id": str(ds.get("PatientID", "")),
        "study_uid": str(ds.StudyInstanceUID),
        "series_uid": str(ds.SeriesInstanceUID),
        "sop_uid": str(ds.SOPInstanceUID),
    }

    return Image.fromarray(arr), meta


def preprocess(gray: Image.Image, size: int) -> torch.Tensor:
    """
    Mirror the pipeline: resize_one (LANCZOS to `size`), then eval transforms.

    Returns a (1, 3, 224, 224) tensor.
    """
    assert gray.mode == "L", gray.mode
    small = gray.resize(
        (size, size), Image.Resampling.LANCZOS
    )  # = resize_images.resize_one
    return build_transforms(train=False)(small).unsqueeze(0)


def analyze(
    model: torch.nn.Module, x: torch.Tensor, thresholds: dict[str, float]
) -> Result:
    assert x.ndim == 4 and x.shape[0] == 1, (
        f"expected (1, C, H, W), got {tuple(x.shape)}"
    )
    with torch.no_grad():
        values = model(x)[0].cpu().numpy()
    logits = {label: float(v) for label, v in zip(LABELS, values, strict=True)}
    flagged = [label for label in LABELS if logits[label] >= thresholds[label]]

    cam_labels = flagged or [max(logits, key=logits.get)]
    heatmaps = {}
    for label in cam_labels:
        cam, cam_logit = gradcam(model, x[0], LABELS.index(label))
        assert abs(cam_logit - logits[label]) < 1e-4, (label, cam_logit, logits[label])
        heatmaps[label] = cam
    return Result(logits=logits, flagged=flagged, heatmaps=heatmaps)
