"""Training data: label CSVs + pre-computed log-mel tensors, session/participant splits (FR-6, T4, T8).

Layout (option A of §3.1, "pre-computed tensors keyed by clip_id"):
    <root>/labels/<session_id>.csv   columns: onset_s, jamo, keytype, shift, scenario, participant [, clip_id]
    <root>/mel/<clip_id>.npy         float32 (64,48) or (1,64,48), final z-scored log-mel

`session_id` is the CSV file stem. If the CSV has no `clip_id` column, clip_id is `<session_id>_<row:05d>`
(row = 0-based data row), which is the name the Mel component is expected to write. Pure numpy/stdlib.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import labels
from .melio import REGIME, load_clip

REQUIRED_COLUMNS = ("onset_s", "jamo", "keytype", "shift", "scenario", "participant")


@dataclass(frozen=True)
class Sample:
    clip_id: str
    session_id: str
    participant: str
    scenario: str
    keytype: str
    symbol: str
    label: int
    onset_s: float
    mel_path: Path


def load_samples(root: str | Path, *, check_files: bool = True) -> list[Sample]:
    root = Path(root)
    csvs = sorted((root / "labels").glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no label CSVs under {root / 'labels'}")
    out: list[Sample] = []
    missing: list[str] = []
    for path in csvs:
        sid = path.stem
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            absent = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if absent:
                raise ValueError(f"{path.name}: missing columns {absent}")
            for i, row in enumerate(reader):
                symbol = labels.symbol_of(row["keytype"].strip(), (row["jamo"] or "").strip())
                cid = (row.get("clip_id") or "").strip() or f"{sid}_{i:05d}"
                mp = root / "mel" / f"{cid}.npy"
                if check_files and not mp.exists():
                    missing.append(str(mp))
                out.append(Sample(cid, sid, row["participant"].strip(), row["scenario"].strip(), row["keytype"].strip(),
                                  symbol, labels.label_index(symbol), float(row["onset_s"]), mp))
    if missing:
        raise FileNotFoundError(f"{len(missing)} log-mel files missing, e.g. {missing[:3]}")
    return out


# ---- splitting ------------------------------------------------------------------------------
def _group_split(keys: list[str], val_frac: float, test_frac: float, seed: int, what: str) -> dict[str, list[int]]:
    groups = sorted(set(keys))
    n_test = max(1, round(len(groups) * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(len(groups) * val_frac)) if val_frac > 0 else 0
    if len(groups) - n_test - n_val < 1:
        raise ValueError(f"a {what} split into train/val/test needs at least {n_test + n_val + 1} {what}s, found {len(groups)}")
    perm = np.random.default_rng(seed).permutation(len(groups))
    which = {}
    for rank, g in enumerate(perm):
        which[groups[g]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, k in enumerate(keys):
        split[which[k]].append(i)
    return split


def make_split(samples: list[Sample], mode: str = "session", seed: int = 1234, val_frac: float = 0.1,
               test_frac: float = 0.2) -> dict[str, list[int]]:
    """Index lists {train,val,test}. `session`: a recording session never spans two subsets (FR-6);
    `participant`: cross-user generalisation; `random`: per clip (optimistic — sanity checks only)."""
    if mode == "session":
        return _group_split([s.session_id for s in samples], val_frac, test_frac, seed, "session")
    if mode == "participant":
        return _group_split([s.participant for s in samples], val_frac, test_frac, seed, "participant")
    if mode == "random":
        perm = np.random.default_rng(seed).permutation(len(samples))
        n_test, n_val = int(round(len(samples) * test_frac)), int(round(len(samples) * val_frac))
        return {"test": sorted(perm[:n_test].tolist()), "val": sorted(perm[n_test:n_test + n_val].tolist()),
                "train": sorted(perm[n_test + n_val:].tolist())}
    raise ValueError(f"unknown split mode {mode!r}")


def save_split(path: str | Path, samples: list[Sample], split: dict[str, list[int]], mode: str, seed: int) -> None:
    doc = {"mode": mode, "seed": seed, **{k: [samples[i].clip_id for i in v] for k, v in split.items()}}
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_split(path: str | Path, samples: list[Sample]) -> dict[str, list[int]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    pos = {s.clip_id: i for i, s in enumerate(samples)}
    try:
        return {k: [pos[c] for c in doc[k]] for k in ("train", "val", "test")}
    except KeyError as e:
        raise ValueError(f"split file {path} refers to clip {e} that is not in the dataset") from None


# ---- dataset --------------------------------------------------------------------------------
class LogMelDataset:
    """Map-style dataset (works with torch DataLoader without importing torch). Yields ((1,64,48) f32, int)."""

    def __init__(self, samples: list[Sample], *, cache: bool = True):
        self.samples, self._cache = samples, ({} if cache else None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        s = self.samples[i]
        x = self._cache.get(i) if self._cache is not None else None
        if x is None:
            x = load_clip(s.mel_path, REGIME)[None]
            if self._cache is not None:
                self._cache[i] = x
        return x, s.label

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.samples:
            return np.zeros((0, *REGIME.input_shape), np.float32), np.zeros(0, np.int64)
        xs = np.stack([self[i][0] for i in range(len(self))])
        return xs, np.array([s.label for s in self.samples], dtype=np.int64)


def class_counts(samples: list[Sample]) -> np.ndarray:
    return np.bincount([s.label for s in samples], minlength=labels.num_classes())


def class_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse-sqrt-frequency weights, mean 1 over present classes; absent classes get 0."""
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return (w / w[counts > 0].mean()).astype(np.float32)


def coverage_report(samples: list[Sample], min_per_jamo: int = 25, min_per_special: int = 25) -> dict:
    """T8/§7: >= min_per_jamo keystrokes per jamo *per participant*, and >= min_per_special per special token overall."""
    per_part = defaultdict(Counter)
    total = Counter()
    for s in samples:
        per_part[s.participant][s.symbol] += 1
        total[s.symbol] += 1
    short: list[dict] = []
    for part, cnt in sorted(per_part.items()):
        short += [{"participant": part, "symbol": j, "have": cnt[j], "need": min_per_jamo}
                  for j in labels.JAMO if cnt[j] < min_per_jamo]
    short += [{"participant": "*", "symbol": t, "have": total[t], "need": min_per_special}
              for t in labels.SPECIAL if total[t] < min_per_special]
    return {"ok": not short, "n_samples": len(samples), "counts": {k: total[k] for k in labels.LABELS}, "short": short}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(description="Check label/mel coverage of a dataset root")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep = coverage_report(load_samples(args.data), cfg["data"]["min_per_jamo"], cfg["data"]["min_per_special"])
    print(json.dumps({k: rep[k] for k in ("ok", "n_samples")}, indent=1))
    for row in rep["short"][:40]:
        print(f"  short: {row['participant']:>10} {row['symbol']:>8} have {row['have']} < {row['need']}")
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
