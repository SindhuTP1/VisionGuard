import pytest
import torch

from app.config import settings
from app.model_loader import (
    ModelUnavailableError, build_classifier, load_classifier, save_classifier_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "m.pt"
    save_classifier_checkpoint(build_classifier(pretrained=False), path, run_id="unit")
    clf = load_classifier(settings, path)
    assert clf.run_id == "unit" and clf.labels == ("real", "fake")


def test_refuses_checkpoint_with_different_preprocessing(tmp_path):
    path = tmp_path / "old.pt"
    save_classifier_checkpoint(build_classifier(pretrained=False), path, run_id="old")
    ckpt = torch.load(path, weights_only=True)
    ckpt["preprocessing"] = "resize_center_crop_v0"
    torch.save(ckpt, path)
    with pytest.raises(ModelUnavailableError, match="preprocessing"):
        load_classifier(settings, path)


def test_missing_checkpoint_is_unavailable_not_faked(tmp_path):
    with pytest.raises(ModelUnavailableError, match="not found"):
        load_classifier(settings, tmp_path / "nope.pt")
