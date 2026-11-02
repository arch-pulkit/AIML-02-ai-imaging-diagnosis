import numpy as np
import pandas as pd
from PIL import Image

from src.data.dicom_io import apply_window, load_dicom, png_to_dicom, to_display


def test_apply_window_known_values():
    x = np.array([-1000, -160, 40, 240, 1000], dtype=np.float32)
    out = apply_window(x, center=40, width=400)
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 127, 255, 255]


def test_apply_window_zero_width_is_black():
    out = apply_window(np.ones((4, 4)), center=0, width=0)
    assert (out == 0).all()


def test_to_display_uses_full_range():
    out = to_display(np.linspace(-500, 500, 100, dtype=np.float32))
    assert out.min() == 0 and out.max() == 255


def _fake_row():
    """A made-up CSV row, so tests don't depend on the real dataset."""
    return pd.Series({
        "image_index": "99999999_000.png", "follow_up": 0,
        "patient_id": 99999999, "patient_sex": "F", "patient_age": 7,
        "view_position": "AP", "width": 2048, "height": 2048,
        "spacing_x": 0.15, "spacing_y": 0.15,
    })


def _fake_png(tmp_path, size=64):
    arr = (np.arange(size * size) % 256).astype(np.uint8).reshape(size, size)
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path)
    return path, arr


def test_png_dicom_round_trip(tmp_path):
    png, arr = _fake_png(tmp_path)
    dcm = tmp_path / "img.dcm"
    png_to_dicom(png, _fake_row(), dcm)
    img, meta = load_dicom(dcm)

    assert np.array_equal(img.astype(np.uint8), arr)
    assert meta["patient_id"] == "99999999"
    assert meta["patient_sex"] == "F"
    assert meta["patient_age"] == "007Y"          # zero-padded
    assert meta["modality"] == "DX"
    assert meta["view_position"] == "AP"
    assert meta["pixel_spacing"] == [4.8, 4.8]    # 0.15 * 2048 / 64: resize correction


def test_uids_are_deterministic(tmp_path):
    png, _ = _fake_png(tmp_path)
    png_to_dicom(png, _fake_row(), tmp_path / "a.dcm")
    png_to_dicom(png, _fake_row(), tmp_path / "b.dcm")
    _, a = load_dicom(tmp_path / "a.dcm")
    _, b = load_dicom(tmp_path / "b.dcm")
    for key in ("study_uid", "series_uid", "sop_uid"):
        assert a[key] == b[key]
