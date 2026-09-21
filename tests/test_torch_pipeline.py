"""Tests that run a real (tiny) PyTorch model. Skip on constrained machines: `pytest -m "not torch"`."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from src import dataset, synth  # noqa: E402
from src.config import load_config  # noqa: E402
from src.infer import AFE, afe_from_checkpoint  # noqa: E402
from src.model import CoAtNetLite, SmallCNN, build_model, complexity, count_params  # noqa: E402
from src.schema import validate_document  # noqa: E402


def test_smallcnn_forward_shape_and_budget():  # T3 / NFR-1
    m = SmallCNN().eval()
    assert m(torch.zeros(3, 1, 64, 48)).shape == (3, 38)
    c = complexity(m)
    assert c["int8_weight_bytes_upper_bound"] <= 1_000_000 and c["params"] == count_params(m)
    assert 1e6 < c["macs"] < 5e7


def test_coatnetlite_forward_shape():
    assert CoAtNetLite().eval()(torch.zeros(2, 1, 64, 48)).shape == (2, 38)


def test_build_model_unknown():
    with pytest.raises(ValueError):
        build_model("resnet")


def test_train_one_epoch_then_infer(tmp_path):  # T4 done-when + T5 wiring
    from src.train import train

    root = synth.generate(tmp_path / "data", participants=3, sessions_per_scenario=1, per_session=40, seed=0)
    cfg = load_config()
    cfg["train"].update(batch_size=16, val_frac=0.2, test_frac=0.2, epochs=1)
    res = train(cfg, root, tmp_path / "run", epochs=1, max_samples=60)
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "split.json").exists()
    assert res["params"] > 0

    afe = afe_from_checkpoint(tmp_path / "run" / "best.pt", top_n=5)
    samples = dataset.load_samples(root)
    x, _ = dataset.LogMelDataset(samples[:6]).arrays()
    doc = afe.run_arrays(x[:, 0], [s.onset_s for s in samples[:6]], "smoke")
    validate_document(doc)
    assert doc["meta"]["calibrated"] is False
    assert doc == afe.run_arrays(x[:, 0], [s.onset_s for s in samples[:6]], "smoke")  # AC-7 with the real model


def test_calibration_json_is_picked_up(tmp_path):
    from src.train import train

    root = synth.generate(tmp_path / "data", participants=3, sessions_per_scenario=1, per_session=30, seed=1)
    cfg = load_config()
    cfg["train"].update(batch_size=16)
    train(cfg, root, tmp_path / "run", epochs=1, max_samples=40)
    (tmp_path / "run" / "calibration.json").write_text(json.dumps({"temperature": 2.5}))
    afe = afe_from_checkpoint(tmp_path / "run" / "best.pt")
    assert isinstance(afe, AFE) and afe.temperature == 2.5 and afe.calibrated
