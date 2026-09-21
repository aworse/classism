import numpy as np
import pytest

from src.melio import MEL_SPEC_VERSION, REGIME, MelSession, RegimeError, check_logmel


def make_session(logmels, onsets, **kw):
    return MelSession.from_arrays(logmels, onsets, "p1_near_s0")


def test_regime_frozen_values():
    assert REGIME.input_shape == (1, 64, 48)
    assert REGIME.sample_rate == 48000 and REGIME.win_samples == 384 and REGIME.hop_samples == 96
    assert len(REGIME.param_hash()) == 8 and REGIME.param_hash() == REGIME.param_hash()


def test_check_logmel_accepts_both_layouts(logmels):
    assert check_logmel(logmels[0]).shape == (64, 48)
    assert check_logmel(logmels[0][None]).shape == (64, 48)


@pytest.mark.parametrize("shape", [(64, 47), (63, 48), (1, 48, 64), (2, 64, 48), (64,), (1, 1, 64, 48)])
def test_wrong_shape_raises_ac8(shape):
    with pytest.raises(RegimeError):
        check_logmel(np.zeros(shape, np.float32))


def test_rejects_nan_and_ints():
    bad = np.zeros((64, 48), np.float32)
    bad[3, 3] = np.nan
    with pytest.raises(RegimeError):
        check_logmel(bad)
    with pytest.raises(RegimeError):
        check_logmel(np.zeros((64, 48), np.int16))


def test_strict_norm(logmels):
    check_logmel(logmels[0], strict_norm=True)
    with pytest.raises(RegimeError):
        check_logmel(logmels[0] * 5 + 3, strict_norm=True)


def test_session_check_ok(logmels, onsets):
    s = make_session(logmels, onsets).check()
    assert s.logmels().shape == (7, 64, 48) and list(s.onsets()) == onsets


def test_session_hash_and_envelope_mismatch(logmels, onsets):
    s = make_session(logmels, onsets)
    s.param_hash = "deadbeef"
    with pytest.raises(RegimeError, match="param_hash"):
        s.check()
    s.check(verify_hash=False)
    s2 = make_session(logmels, onsets)
    s2.n_mels = 80
    with pytest.raises(RegimeError, match="n_mels"):
        s2.check()
    s3 = make_session(logmels, onsets)
    s3.mel_spec_version = "2.0"
    with pytest.raises(RegimeError, match="version"):
        s3.check()


def test_session_idx_gap_and_unordered_onsets(logmels, onsets):
    s = make_session(logmels, onsets)
    s.records[3].idx = 4
    with pytest.raises(RegimeError, match="contiguous"):
        s.check()
    with pytest.raises(RegimeError, match="time-ordered"):
        make_session(logmels[:3], [1.0, 0.5, 2.0]).check()


def test_record_with_wrong_shape_in_session(logmels, onsets):
    s = make_session(logmels, onsets)
    s.records[2].logmel = np.zeros((64, 40), np.float32)
    with pytest.raises(RegimeError):
        s.check()


@pytest.mark.parametrize("encode", ["list", "base64"])
def test_json_roundtrip(logmels, onsets, encode):
    d = make_session(logmels, onsets).to_dict(encode)
    back = MelSession.from_dict(d).check()
    np.testing.assert_allclose(back.logmels(), logmels, rtol=0, atol=0)
    assert back.mel_spec_version == MEL_SPEC_VERSION


def test_npz_roundtrip(tmp_path, logmels, onsets):
    p = tmp_path / "s.npz"
    make_session(logmels, onsets).to_npz(p)
    back = MelSession.from_npz(p).check()
    assert back.session_id == "p1_near_s0" and len(back.records) == 7
    np.testing.assert_array_equal(back.logmels(), logmels)


def test_missing_envelope_field():
    with pytest.raises(RegimeError, match="missing"):
        MelSession.from_dict({"records": []})
