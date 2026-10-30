import numpy as np
import pydicom


def apply_window(img, center, width):

    if width == 0:
        return np.zeros_like(img, dtype=np.uint8)
        
    low = center - width/2
    high = center + width/2
    
    new_img = np.clip(img, low, high)

    scaled_img = ((new_img - low) / width) *255

    return scaled_img.astype(np.uint8)
import numpy as np
import pydicom
from pydicom.multival import MultiValue


def _get(ds, tag, cast=str):
    """Read a DICOM tag as a plain Python value. Returns None if missing or empty."""
    value = ds.get(tag)
    if value is None:
        return None

    # Multi-value tags like PixelSpacing: cast each element
    if isinstance(value, (MultiValue, list, tuple)):
        if len(value) == 0:
            return None
        return [cast(v) for v in value]

    text = str(value).strip()
    if not text:
        return None
    return text if cast is str else cast(value)


def load_dicom(path) -> tuple[np.ndarray, dict]:
    """Load a DICOM file.

    Returns the image in real units (HU for CT) and metadata as plain Python types.
    """
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # The DICOM standard says missing rescale tags mean "use pixels as-is"
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    img = img * slope + intercept

    # MONOCHROME1 means higher values are darker; flip so higher = brighter
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        img = img.max() + img.min() - img

    meta = {
        "patient_id":    _get(ds, "PatientID"),
        "patient_sex":   _get(ds, "PatientSex"),
        "patient_age":   _get(ds, "PatientAge"),
        "modality":      _get(ds, "Modality"),
        "view_position": _get(ds, "ViewPosition"),
        "rows":          _get(ds, "Rows", int),
        "columns":       _get(ds, "Columns", int),
        "pixel_spacing": _get(ds, "PixelSpacing", float),
        "study_uid":     _get(ds, "StudyInstanceUID"),
        "series_uid":    _get(ds, "SeriesInstanceUID"),
        "sop_uid":       _get(ds, "SOPInstanceUID"),
    }
    return img, meta

def to_display(img, center=None, width=None) -> np.ndarray:
    """Convert an image to uint8 for display. Uses min-max scaling if no window is given."""
    if center is None or width is None:
        lo, hi = float(img.min()), float(img.max())
        center, width = (lo + hi) / 2, hi - lo
    return apply_window(img, center, width)

