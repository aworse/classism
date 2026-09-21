"""AC-1, AC-2, AC-7, AC-8, AC-9 and FR-3/FR-4 — the AFE post-processing, with a numpy stand-in for the CNN."""
import copy
import json

import numpy as np
import pytest

from src import labels
from src.infer import AFE, top_n_renorm
from src.melio import MelSession, RegimeError
from src.schema import SchemaError, load_schema, validate_document


@pytest.fixture
def afe(fake_predictor):
    return AFE(fake_predictor, top_n=5, batch_size=3)


@pytest.fixture
def doc(afe, logmels, onsets):
    return afe.run(MelSession.from_arrays(logmels, onsets, "p1_near_s0"))


def test_schema_symbols_match_label_space():
    assert set(load_schema()["$defs"]["symbol"]["enum"]) == set(labels.LABELS)


def test_ac1_document_validates(doc):
    validate_document(doc)
    assert doc["schema"] == "kdaa.afe.v1" and doc["sample_rate"] == 48000 and doc["lang"] == "ko"
    assert doc["meta"] == {"model_id": "smallcnn-fp32-v1", "topN": 5, "input_shape": [1, 64, 48], "calibrated": False}
    json.dumps(doc, ensure_ascii=False)  # serialisable


def test_ac2_count_order_and_onset_passthrough(doc, onsets):
    assert len(doc["tokens"]) == len(onsets)
    assert [t["idx"] for t in doc["tokens"]] == list(range(len(onsets)))
    assert [t["onset_s"] for t in doc["tokens"]] == onsets


def test_ac9_alts_sum_to_one_and_descend(doc):
    for t in doc["tokens"]:
        ps = [a["p"] for a in t["alts"]]
        assert len(ps) == 5 and abs(sum(ps) - 1.0) <= 1e-6 and ps == sorted(ps, reverse=True)


def test_ac7_deterministic(afe, logmels, onsets):
    a = afe.run_arrays(logmels, onsets)
    b = afe.run_arrays(logmels, onsets)
    assert a == b


def test_batching_does_not_change_output(fake_predictor, logmels, onsets):
    small = AFE(fake_predictor, batch_size=2).run_arrays(logmels, onsets)
    big = AFE(fake_predictor, batch_size=64).run_arrays(logmels, onsets)
    assert [[a["jamo"] for a in t["alts"]] for t in small["tokens"]] == [[a["jamo"] for a in t["alts"]] for t in big["tokens"]]


def test_ac8_bad_shape_raises(afe, logmels, onsets):
    with pytest.raises(RegimeError):
        afe.run_arrays([np.zeros((64, 40), np.float32)], [0.0])
    with pytest.raises(RegimeError):  # one bad clip among good ones must not be silently accepted
        afe.run_arrays([logmels[0], np.zeros((63, 48), np.float32)], [0.0, 1.0])


def test_length_mismatch_raises(afe, logmels):
    with pytest.raises(ValueError):
        afe.run_arrays(logmels, [0.0])


def test_empty_session_is_valid(afe):
    d = afe.run_arrays([], [])
    validate_document(d)
    assert d["tokens"] == []


@pytest.mark.parametrize("top", ["<sp>", "<bs>", "<shift>", "<caps>", "<other>", "ㄱ"])
def test_keytype_derived_from_top1(top):
    logits = np.full((1, 38), -5.0, np.float32)
    logits[0, labels.SYMBOL_TO_IDX[top]] = 5.0
    afe = AFE(lambda x: logits, top_n=3)
    (tok,) = afe.tokens_from_logits(logits, [0.0])
    assert tok["alts"][0]["jamo"] == top and tok["keytype"] == labels.keytype_of(top)


def test_temperature_and_calibrated_flag(fake_predictor, logmels, onsets):
    hot = AFE(fake_predictor, temperature=1.0).run_arrays(logmels, onsets)
    cold = AFE(fake_predictor, temperature=4.0, calibrated=True).run_arrays(logmels, onsets)
    assert cold["meta"]["calibrated"] is True
    validate_document(cold)
    # softer distribution -> lower top-1 probability, same ranking
    assert all(c["alts"][0]["p"] <= h["alts"][0]["p"] + 1e-12 for c, h in zip(cold["tokens"], hot["tokens"]))
    assert [t["alts"][0]["jamo"] for t in cold["tokens"]] == [t["alts"][0]["jamo"] for t in hot["tokens"]]


def test_top_n_renorm_ties_are_deterministic():
    p = np.full((1, 38), 1 / 38)
    idx, q = top_n_renorm(p, 5)
    assert idx[0].tolist() == [0, 1, 2, 3, 4] and abs(q.sum() - 1) < 1e-12


def test_invalid_config_rejected(fake_predictor):
    for kw in ({"top_n": 0}, {"top_n": 39}, {"temperature": 0.0}):
        with pytest.raises(ValueError):
            AFE(fake_predictor, **kw)


# ---- validator catches malformed documents --------------------------------------------------
def _mut(doc, fn):
    d = copy.deepcopy(doc)
    fn(d)
    return d


@pytest.mark.parametrize("mutate", [
    lambda d: d["tokens"][0]["alts"][0].__setitem__("p", 0.99),          # sum != 1
    lambda d: d["tokens"][1].__setitem__("idx", 5),                       # idx gap
    lambda d: d["tokens"][0].__setitem__("keytype", "space" if d["tokens"][0]["keytype"] == "normal" else "normal"),  # keytype != top-1
    lambda d: d["tokens"][0]["alts"].reverse(),                           # not descending
    lambda d: d["tokens"][0]["alts"][0].__setitem__("jamo", "x"),         # unknown symbol
    lambda d: d.__setitem__("schema", "kdaa.afe.v2"),
    lambda d: d["meta"].__setitem__("input_shape", [1, 64, 47]),
    lambda d: d.pop("meta"),
])
def test_validator_rejects_bad_documents(doc, mutate):
    with pytest.raises(SchemaError):
        validate_document(_mut(doc, mutate))
