import numpy as np
import pytest

from src import dataset, labels, synth
from src.config import load_config
from src.melio import RegimeError


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return synth.generate(tmp_path_factory.mktemp("synth"), participants=3, sessions_per_scenario=2, per_session=60, seed=0)


@pytest.fixture(scope="module")
def samples(root):
    return dataset.load_samples(root)


def test_load_samples(samples):
    assert len(samples) == 3 * 3 * 2 * 60
    assert {s.scenario for s in samples} == {"near", "far", "noise"}
    assert all(s.mel_path.exists() and 0 <= s.label < 38 for s in samples)
    assert samples[0].clip_id == f"{samples[0].session_id}_00000"


def test_every_class_present(samples):
    assert (dataset.class_counts(samples) > 0).all()


def test_dataset_items_are_contract_shaped(samples):
    ds = dataset.LogMelDataset(samples[:5])
    x, y = ds[0]
    assert x.shape == (1, 64, 48) and x.dtype == np.float32 and y == samples[0].label
    xs, ys = ds.arrays()
    assert xs.shape == (5, 1, 64, 48) and ys.tolist() == [s.label for s in samples[:5]]
    assert abs(float(x.mean())) < 1e-4 and abs(float(x.std()) - 1) < 1e-3  # z-scored like the Mel output


@pytest.mark.parametrize("mode,key", [("session", "session_id"), ("participant", "participant")])
def test_group_splits_never_leak(samples, mode, key):
    sp = dataset.make_split(samples, mode, seed=7, val_frac=0.2, test_frac=0.2)
    groups = {k: {getattr(samples[i], key) for i in v} for k, v in sp.items()}
    assert groups["train"].isdisjoint(groups["val"]) and groups["train"].isdisjoint(groups["test"]) and groups["val"].isdisjoint(groups["test"])
    assert sorted(sum(sp.values(), [])) == list(range(len(samples)))  # a partition
    assert all(sp[k] for k in sp)


def test_split_is_reproducible_and_seed_dependent(samples):
    a = dataset.make_split(samples, "session", 1)
    assert a == dataset.make_split(samples, "session", 1)
    assert any(a != dataset.make_split(samples, "session", s) for s in range(2, 8))


def test_random_split_partitions(samples):
    sp = dataset.make_split(samples, "random", 0)
    assert sorted(sum(sp.values(), [])) == list(range(len(samples)))


def test_too_few_groups_is_an_error(samples):
    few = [s for s in samples if s.participant == "p1"]
    with pytest.raises(ValueError, match="participant"):
        dataset.make_split(few, "participant")


def test_split_file_roundtrip(tmp_path, samples):
    sp = dataset.make_split(samples, "session", 3)
    dataset.save_split(tmp_path / "split.json", samples, sp, "session", 3)
    assert dataset.load_split(tmp_path / "split.json", samples) == sp


def test_coverage_report(samples):
    ok = dataset.coverage_report(samples, min_per_jamo=1, min_per_special=1)
    assert ok["ok"] and ok["n_samples"] == len(samples)
    bad = dataset.coverage_report(samples, min_per_jamo=10_000, min_per_special=1)
    assert not bad["ok"] and any(r["participant"] == "p1" for r in bad["short"])
    only_normal = [s for s in samples if s.keytype == "normal"]
    rep = dataset.coverage_report(only_normal, 1, 1)
    assert {r["symbol"] for r in rep["short"]} == set(labels.SPECIAL)


def test_class_weights(samples):
    w = dataset.class_weights(dataset.class_counts(samples))
    assert w.shape == (38,) and (w > 0).all() and abs(w.mean() - 1) < 1e-5


def test_missing_mel_is_loud(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "p1_near_s0.csv").write_text(
        "onset_s,jamo,keytype,shift,scenario,participant\n1.0,ㄱ,normal,0,near,p1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        dataset.load_samples(tmp_path)


def test_bad_csv_rows_are_loud(tmp_path):
    (tmp_path / "labels").mkdir()
    p = tmp_path / "labels" / "s.csv"
    p.write_text("onset_s,jamo,keytype\n1.0,ㄱ,normal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        dataset.load_samples(tmp_path)
    p.write_text("onset_s,jamo,keytype,shift,scenario,participant\n1.0,,normal,0,near,p1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.load_samples(tmp_path, check_files=False)


def test_wrong_shape_mel_is_rejected(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "mel").mkdir()
    (tmp_path / "labels" / "s.csv").write_text(
        "onset_s,jamo,keytype,shift,scenario,participant\n1.0,ㄱ,normal,0,near,p1\n", encoding="utf-8")
    np.save(tmp_path / "mel" / "s_00000.npy", np.zeros((64, 30), np.float32))
    ds = dataset.LogMelDataset(dataset.load_samples(tmp_path))
    with pytest.raises(RegimeError):
        ds[0]


def test_config_defaults_and_validation(tmp_path):
    cfg = load_config()
    assert cfg["labels"]["num_classes"] == 38 and cfg["infer"]["top_n"] == 5 and cfg["train"]["split"] == "session"
    bad = tmp_path / "c.yaml"
    bad.write_text("input:\n  n_mels: 80\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen regime"):
        load_config(bad)
