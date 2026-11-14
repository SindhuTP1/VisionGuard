import io
import math

import numpy as np
import pytest
from PIL import Image

from app.vision_utils import (
    BBox, ImageDecodeError, joint_angle_deg, landmarks_to_array, load_image_bytes,
    roll_compensate, select_primary_box,
)


def _png(w=64, h=48, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, (w, h)).save(buf, "PNG")
    return buf.getvalue()


def test_roll_compensation_levels_eye_axis():
    pts = np.array([[0, 0, 0], [10, 10, 0], [5, 5, 3.0]])
    out, roll = roll_compensate(pts, 0, 1)
    assert math.degrees(roll) == pytest.approx(45)
    assert out[0, 1] == pytest.approx(out[1, 1])
    # 3D distances are rotation invariant
    assert np.linalg.norm(out[0] - out[2]) == pytest.approx(np.linalg.norm(pts[0] - pts[2]))


def test_joint_angle():
    assert joint_angle_deg(np.array([1, 0, 0]), np.zeros(3), np.array([0, 1, 0])) == pytest.approx(90)
    assert math.isnan(joint_angle_deg(np.zeros(3), np.zeros(3), np.ones(3)))


def test_landmarks_scaled_isotropically():
    class L:
        def __init__(s, x, y, z):
            s.x, s.y, s.z = x, y, z

    pts = landmarks_to_array([L(0.5, 0.5, 0.1)], width=200, height=100)
    assert pts.tolist() == [[100, 50, 20]]


def test_bbox_padding_clips_to_image():
    assert BBox(10, 10, 50, 90).padded(0.1, 60, 100).as_list() == [6, 2, 54, 98]
    assert BBox(0, 0, 60, 100).padded(0.5, 60, 100).as_list() == [0, 0, 60, 100]


def test_select_primary_box():
    big, small = BBox(0, 0, 100, 100, 0.5), BBox(0, 0, 10, 10, 0.99)
    assert select_primary_box([small, big]) == big
    assert select_primary_box([small], min_area=500) is None


def test_load_image_bytes_converts_and_downscales():
    arr = load_image_bytes(_png(400, 200, "RGBA"), max_side=100)
    assert arr.shape == (50, 100, 3) and arr.dtype == np.uint8


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8garbage"])
def test_load_image_bytes_rejects_garbage(data):
    with pytest.raises(ImageDecodeError):
        load_image_bytes(data)


@pytest.mark.parametrize("size", [(1024, 1024), (640, 442), (150, 100)])
def test_classifier_transform_never_resizes_and_fits_any_size(size):
    from app.vision_utils import build_classifier_transform

    img = Image.new("RGB", size, (120, 60, 30))
    assert tuple(build_classifier_transform(True)(img).shape) == (3, 224, 224)
    assert tuple(build_classifier_transform(False)(img).shape) == (5, 3, 224, 224)


def test_jpeg_normalize_equalizes_quality():
    from app.vision_utils import JpegNormalize

    hi = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 30)).save(hi, "JPEG", quality=95)
    lo = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 30)).save(lo, "JPEG", quality=40)
    a, b = (JpegNormalize(75)(Image.open(io.BytesIO(x.getvalue()))) for x in (hi, lo))
    assert a.quantization == b.quantization


def test_load_image_bytes_rejects_decompression_bomb():
    with pytest.raises(ImageDecodeError):
        load_image_bytes(_png(1000, 1000), max_pixels=10_000)
