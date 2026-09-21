"""Numpy-only parts of the export path (AC-6 bookkeeping). The ONNX/TFLite conversion itself needs the export toolchain."""
import numpy as np

from src.export_esp32 import check_ac6, dequantize, estimate_input_quant, pick_calibration, quantize


def test_input_quant_covers_range_and_roundtrips():
    x = np.random.default_rng(0).normal(size=(50, 1, 64, 48)).astype(np.float32) * 1.5
    q = estimate_input_quant(x)
    assert -128 <= q["zero_point"] <= 127 and q["scale"] > 0
    err = np.abs(dequantize(quantize(x, q["scale"], q["zero_point"]), q["scale"], q["zero_point"]) - x).max()
    assert err <= q["scale"] / 2 + 1e-6


def test_quantize_saturates():
    assert quantize(np.array([1e9, -1e9], np.float32), 0.1, 0).tolist() == [127, -128]


def test_constant_input_does_not_divide_by_zero():
    q = estimate_input_quant(np.zeros((2, 1, 64, 48), np.float32))
    assert q["scale"] == 1.0


def test_ac6_verdicts():
    ok = check_ac6(0.90, 0.885, 400_000)
    assert ok["pass_accuracy"] and ok["pass_size"] and abs(ok["top1_drop_pp"] - 1.5) < 1e-9
    assert not check_ac6(0.90, 0.85, 400_000)["pass_accuracy"]
    assert not check_ac6(0.90, 0.90, 2_000_000)["pass_size"]


def test_calibration_subset_is_deterministic_and_bounded():
    x = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1, 1)
    a, b = pick_calibration(x, 300), pick_calibration(x, 300)
    assert a.shape[0] == 300 and np.array_equal(a, b)
    assert pick_calibration(x[:10], 300).shape[0] == 10
