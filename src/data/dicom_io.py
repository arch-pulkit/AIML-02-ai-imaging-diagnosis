import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from pathlib import Path
from pydicom.multival import MultiValue


def apply_window(img, center, width):

    if width == 0:
        return np.zeros_like(img, dtype=np.uint8)
        
    low = center - width/2
    high = center + width/2
    
    new_img = np.clip(img, low, high)

    scaled_img = ((new_img - low) / width) *255

    return scaled_img.astype(np.uint8)

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

def png_to_dicom(png_path, row, out_path) -> None:

    arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)

    #Separaing UIDs
    image_index = str(row["image_index"])
    study_uid = generate_uid(entropy_srcs=[str(row["patient_id"]), str(row["follow_up"])])
    series_uid = generate_uid(entropy_srcs=[image_index, "series"])
    sop_uid = generate_uid(entropy_srcs=[image_index, "sop"])

    #DICOM header container, explicitly tells any DICOM viewer what standard format the data is stored in and how bytes are encoded
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()

    ds.file_meta = file_meta

    #Organize the file within the secondary medical imaging tree
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    #Patient
    #attaches the clean metadata from csv row directly to the file
    ds.PatientID = str(row["patient_id"])
    ds.PatientSex = str(row["patient_sex"])
    ds.PatientAge = f"{int(row['patient_age']) :03d}Y"

    #Image info
    ds.Modality = "DX"                  # Digital X-Ray
    ds.ConversionType = "WSD"           # Workstation
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.ViewPosition = str(row["view_position"])

    #Adjusts pixel physical spacing cuz NIH png images were resized to 1024*1024; so recalculating mm for image
    # DICOM order is [row spacing / column spacing]
    #Original X-Rays were around 2500-3000 pixels wide and NIH sharnk them, CSV gives the spacing of ORIGINAL, hence
    rows, cols = arr.shape
    spacing_y = float(row["spacing_y"]) * float(row["height"]) / rows
    spacing_x = float(row["spacing_x"]) * float(row["width"]) / cols

    ds.PixelSpacing = [round(spacing_y, 6), round(spacing_x, 6)]

    #Tellss software how to read raw pixel blocks; Pixel Description
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7                      # top bit of 8 used: bits 0 to 7
    ds.PixelRepresentation = 0          #0 = unsigned
    ds.PixelData = arr.tobytes()

    #Save
    #writes out a fully compliant .dcm file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(out_path, enforce_file_format=True)


