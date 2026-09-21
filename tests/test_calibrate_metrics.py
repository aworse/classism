import numpy as np
import pytest

from src import labels
from src.calibrate import calibrate, ece, fit_temperature, nll, softmax
from src.evaluate import acceptance, compute_metrics


def overconfident(n=4000, c=38, acc=0.7, sharp=12.0, seed=0):
    """Logits whose confidence (~sharp) far exceeds their accuracy (acc)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    pred = np.where(rng.random(n) < acc, y, (y + rng.integers(1, c, n)) % c)
    logits = rng.normal(0, 1, (n, c))
    logits[np.arange(n), pred] += sharp
    return logits, y


def test_fit_temperature_softens_overconfident_model():
    logits, y = overconfident()
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert nll(logits, y, t) < nll(logits, y, 1.0)


def sharpened(factor, n=4000, seed=0):
    """A calibrated model whose logits were multiplied by `factor` (over-confident by exactly T = factor)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, (n, 38))
    y = np.array([rng.choice(38, p=p) for p in softmax(z)])
    return z * factor, y


def test_calibration_recovers_temperature_and_meets_ac5():
    val, yv = sharpened(3.0, seed=1)
    test, yt = sharpened(3.0, seed=2)
    res = calibrate(val, yv)
    assert res["temperature"] == pytest.approx(3.0, rel=0.1)
    assert res["ece_before"] > 0.1 and res["ece_after"] < res["ece_before"]
    assert ece(softmax(test, res["temperature"]), yt) <= 0.05  # AC-5 on held-out data


def test_temperature_for_calibrated_model_is_near_one():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, (3000, 38))
    y = np.array([rng.choice(38, p=p) for p in softmax(logits)])  # labels sampled from the model itself
    t = fit_temperature(logits, y)
    assert 0.85 < t < 1.15


def test_ece_perfect_and_worst_cases():
    y = np.arange(10) % 38
    sure = np.full((10, 38), 1e-9)
    sure[np.arange(10), y] = 1.0
    assert ece(sure / sure.sum(1, keepdims=True), y) < 1e-6
    wrong = np.full((10, 38), 1e-9)
    wrong[np.arange(10), (y + 1) % 38] = 1.0
    assert ece(wrong / wrong.sum(1, keepdims=True), y) > 0.99


def test_fit_temperature_rejects_empty():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((0, 38)), np.zeros(0, int))


# ---- metrics --------------------------------------------------------------------------------
def test_metrics_hand_checked():
    sp, bs, ga = labels.SYMBOL_TO_IDX["<sp>"], labels.SYMBOL_TO_IDX["<bs>"], labels.SYMBOL_TO_IDX["ㄱ"]
    kk = labels.SYMBOL_TO_IDX["ㄲ"]
    logits = np.zeros((4, 38))
    logits[0, sp] = 9            # correct (space)
    logits[1, kk] = 9; logits[1, ga] = 5    # true ㄱ, top-1 wrong (ㄲ), but ㄱ in top-2 -> both normal keys
    logits[2, ga] = 9            # correct
    logits[3, sp] = 9            # true <bs> predicted <sp>: keytype wrong
    y = np.array([sp, ga, ga, bs])
    m = compute_metrics(logits, y, ["near", "near", "far", "far"], top_n=2)
    assert m["top1"] == 0.5 and m["top_n_coverage"] == 0.75 and m["top_n_accuracy"] == m["top_n_coverage"]
    assert m["keytype_acc"] == 0.75
    assert m["per_scenario"]["near"]["top1"] == 0.5 and m["per_scenario"]["far"]["n"] == 2
    assert m["per_keytype"]["backspace"]["recall"] == 0.0 and m["per_keytype"]["space"]["recall"] == 1.0
    cm = np.array(m["confusion"])
    assert cm.sum() == 4 and cm[ga, kk] == 1 and cm[bs, sp] == 1
    assert len(m["labels"]) == 38


def test_acceptance_verdicts():
    logits, y = overconfident(n=600, acc=0.95, sharp=8)
    scen = np.array(["near"] * 600)
    m = compute_metrics(logits, y, scen, top_n=5)
    ac = acceptance(m)
    assert ac["AC-3_near_top1"]["pass"] is True and ac["AC-5_ece"]["pass"] is None  # not calibrated -> not measured
    m2 = compute_metrics(logits, y, ["far"] * 600, top_n=5, calibrated=True)
    assert acceptance(m2)["AC-3_near_top1"]["pass"] is None  # no near samples -> not measured
    assert isinstance(acceptance(m2)["AC-5_ece"]["pass"], bool)


def test_metrics_reject_empty():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((0, 38)), np.zeros(0, int))
