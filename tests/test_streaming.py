"""FR-10 / T10: incremental emission gives the same tokens as batch mode."""
import pytest

from src.infer import AFE
from src.melio import RegimeError
from src.schema import validate_document


@pytest.fixture
def afe(fake_predictor):
    return AFE(fake_predictor, top_n=5)


def test_stream_matches_batch(afe, logmels, onsets):
    batch = afe.run_arrays(logmels, onsets, "s")
    stream = afe.stream("s")
    emitted = list(stream.run(zip(onsets, logmels)))
    assert len(emitted) == len(onsets)
    for e, b in zip(emitted, batch["tokens"]):
        assert e["idx"] == b["idx"] and e["onset_s"] == b["onset_s"] and e["keytype"] == b["keytype"]
        assert [a["jamo"] for a in e["alts"]] == [a["jamo"] for a in b["alts"]]
        # float32 logits: batched vs single-row matmul accumulate in a different order (~1e-6), so not bit-equal
        assert [a["p"] for a in e["alts"]] == pytest.approx([a["p"] for a in b["alts"]], abs=1e-4)
    doc = stream.document()
    validate_document(doc)
    assert len(stream.latencies_ms) == len(onsets)


def test_tokens_available_before_session_ends(afe, logmels, onsets):
    s = afe.stream()
    first = s.push(onsets[0], logmels[0])
    assert first["idx"] == 0 and len(s.document()["tokens"]) == 1


def test_stream_rejects_gaps_disorder_and_bad_shapes(afe, logmels, onsets):
    s = afe.stream()
    s.push(onsets[0], logmels[0])
    with pytest.raises(ValueError):
        s.push(onsets[1], logmels[1], idx=3)
    with pytest.raises(ValueError):
        s.push(onsets[0] - 1.0, logmels[1])
    import numpy as np
    with pytest.raises(RegimeError):
        s.push(onsets[1], np.zeros((64, 10), np.float32))
    assert len(s.tokens) == 1  # failed pushes leave no partial state
