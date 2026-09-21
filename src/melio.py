"""Mel -> AFE input adapter and regime validation (spec §3.1; FR-1, NFR-2, NFR-6, AC-8).

The AFE never touches audio. It receives final, already z-scored log-mel tensors and refuses (raises)
anything that does not match the frozen regime instead of silently mis-inferring.

`param_hash` definition (proposal, to be agreed with the Mel component — see O-O3): first 8 hex chars of
sha256 over the canonical JSON (sorted keys, no whitespace) of `Regime`'s fields.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MEL_SPEC_VERSION = "1.0"


class RegimeError(ValueError):
    """Input does not conform to the frozen log-mel regime."""


@dataclass(frozen=True)
class Regime:
    sample_rate: int = 48000
    n_fft: int = 1024
    win_samples: int = 384  # 8 ms @ 48 kHz
    hop_samples: int = 96  # 2 ms @ 48 kHz
    n_mels: int = 64
    fmin: float = 200.0
    fmax: float = 20000.0
    clip_ms: int = 100  # onset -5 ms ... +95 ms
    onset_pre_ms: int = 5
    frames: int = 48
    log_eps: float = 1e-6
    normalization: str = "per_clip_zscore"
    dtype: str = "float32"

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (1, self.n_mels, self.frames)

    def param_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


REGIME = Regime()


def check_logmel(x, regime: Regime = REGIME, *, strict_norm: bool = False) -> np.ndarray:
    """Validate one keystroke's log-mel; return it as float32 `(n_mels, frames)`.

    Accepts `(n_mels, frames)` or `(1, n_mels, frames)`. Raises RegimeError on any mismatch (AC-8).
    `strict_norm` additionally checks the per-clip z-score (mean~0, std~1 or an all-zero flat clip).
    """
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.shape != (regime.n_mels, regime.frames):
        raise RegimeError(
            f"log-mel shape {tuple(np.shape(x))} does not match the frozen regime "
            f"(1, {regime.n_mels}, {regime.frames})"
        )
    if not np.issubdtype(a.dtype, np.floating):
        raise RegimeError(f"log-mel must be floating point, got {a.dtype}")
    a = a.astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        raise RegimeError("log-mel contains NaN/Inf")
    if strict_norm:
        mean, std = float(a.mean()), float(a.std())
        if abs(mean) > 1e-2 or not (abs(std - 1.0) <= 5e-2 or std < 1e-6):
            raise RegimeError(f"log-mel is not per-clip z-scored (mean={mean:.4f}, std={std:.4f})")
    return a


@dataclass
class MelRecord:
    idx: int
    onset_s: float
    logmel: np.ndarray  # (n_mels, frames) float32


@dataclass
class MelSession:
    session_id: str
    records: list[MelRecord]
    mel_spec_version: str = MEL_SPEC_VERSION
    param_hash: str | None = None
    sample_rate: int = REGIME.sample_rate
    n_mels: int = REGIME.n_mels
    frames: int = REGIME.frames
    meta: dict = field(default_factory=dict)

    # ---- validation -------------------------------------------------------------------------
    def check(self, regime: Regime = REGIME, *, verify_hash: bool = True, strict_norm: bool = False) -> "MelSession":
        """Validate the envelope and every record; normalises each `logmel` in place. Returns self."""
        if self.mel_spec_version != MEL_SPEC_VERSION:
            raise RegimeError(f"mel_spec_version {self.mel_spec_version!r} != {MEL_SPEC_VERSION!r}")
        for name, got, want in (("sample_rate", self.sample_rate, regime.sample_rate),
                                ("n_mels", self.n_mels, regime.n_mels),
                                ("frames", self.frames, regime.frames)):
            if got != want:
                raise RegimeError(f"{name} {got} != frozen regime value {want}")
        if verify_hash:
            want_hash = regime.param_hash()
            if self.param_hash != want_hash:
                raise RegimeError(f"param_hash {self.param_hash!r} != expected {want_hash!r} (NFR-2)")
        prev_onset = -np.inf
        for i, rec in enumerate(self.records):
            if rec.idx != i:
                raise RegimeError(f"record idx must be contiguous from 0; position {i} has idx {rec.idx}")
            if not np.isfinite(rec.onset_s) or rec.onset_s < prev_onset:
                raise RegimeError(f"record {i}: onset_s {rec.onset_s!r} is not finite / time-ordered")
            prev_onset = rec.onset_s
            rec.logmel = check_logmel(rec.logmel, regime, strict_norm=strict_norm)
        return self

    # ---- array views ------------------------------------------------------------------------
    def logmels(self) -> np.ndarray:
        """`(N, n_mels, frames)` float32."""
        if not self.records:
            return np.zeros((0, self.n_mels, self.frames), dtype=np.float32)
        return np.stack([r.logmel for r in self.records]).astype(np.float32, copy=False)

    def onsets(self) -> np.ndarray:
        return np.array([r.onset_s for r in self.records], dtype=np.float64)

    # ---- construction / serialisation -------------------------------------------------------
    @classmethod
    def from_arrays(cls, logmels, onsets, session_id: str = "session", regime: Regime = REGIME) -> "MelSession":
        logmels, onsets = list(logmels), list(onsets)
        if len(logmels) != len(onsets):
            raise RegimeError(f"{len(logmels)} log-mels but {len(onsets)} onsets")
        recs = [MelRecord(i, float(o), np.asarray(m)) for i, (m, o) in enumerate(zip(logmels, onsets))]
        return cls(session_id, recs, MEL_SPEC_VERSION, regime.param_hash(), regime.sample_rate, regime.n_mels, regime.frames)

    @classmethod
    def from_dict(cls, d: dict) -> "MelSession":
        """Session envelope of §3.1. `logmel` is a nested list or a base64 little-endian float32 string."""
        try:
            recs = [MelRecord(int(r["idx"]), float(r["onset_s"]), _decode_logmel(r["logmel"], d["n_mels"], d["frames"]))
                    for r in d["records"]]
            return cls(str(d["session_id"]), recs, str(d["mel_spec_version"]), d.get("param_hash"),
                       int(d["sample_rate"]), int(d["n_mels"]), int(d["frames"]))
        except KeyError as e:
            raise RegimeError(f"session envelope is missing field {e}") from None

    @classmethod
    def from_json(cls, path: str | Path) -> "MelSession":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_npz(cls, path: str | Path) -> "MelSession":
        """npz with `logmel (N,n_mels,frames)`, `onset_s (N,)` and scalar meta (see `to_npz`)."""
        with np.load(path, allow_pickle=False) as z:
            lm, on = z["logmel"], z["onset_s"]
            meta = {k: z[k].item() for k in ("session_id", "mel_spec_version", "param_hash", "sample_rate", "n_mels", "frames")
                    if k in z.files}
        if lm.ndim != 3 or len(lm) != len(on):
            raise RegimeError(f"npz logmel {lm.shape} / onset_s {on.shape} are inconsistent")
        recs = [MelRecord(i, float(on[i]), lm[i]) for i in range(len(on))]
        return cls(str(meta.get("session_id", Path(path).stem)), recs, str(meta.get("mel_spec_version", MEL_SPEC_VERSION)),
                   meta.get("param_hash"), int(meta.get("sample_rate", REGIME.sample_rate)),
                   int(meta.get("n_mels", lm.shape[1])), int(meta.get("frames", lm.shape[2])))

    def to_dict(self, encode: str = "list") -> dict:
        def enc(a):
            a = np.ascontiguousarray(a, dtype="<f4")
            return a.tolist() if encode == "list" else base64.b64encode(a.tobytes()).decode("ascii")
        return {"mel_spec_version": self.mel_spec_version, "param_hash": self.param_hash,
                "sample_rate": self.sample_rate, "n_mels": self.n_mels, "frames": self.frames,
                "session_id": self.session_id,
                "records": [{"idx": r.idx, "onset_s": r.onset_s, "logmel": enc(r.logmel)} for r in self.records]}

    def to_npz(self, path: str | Path) -> None:
        np.savez(path, logmel=self.logmels(), onset_s=self.onsets(), session_id=self.session_id,
                 mel_spec_version=self.mel_spec_version, param_hash=self.param_hash or "",
                 sample_rate=self.sample_rate, n_mels=self.n_mels, frames=self.frames)


def _decode_logmel(v, n_mels: int, frames: int) -> np.ndarray:
    if isinstance(v, str):
        raw = base64.b64decode(v)
        if len(raw) != 4 * n_mels * frames:
            raise RegimeError(f"base64 logmel has {len(raw)} bytes, expected {4 * n_mels * frames}")
        return np.frombuffer(raw, dtype="<f4").reshape(n_mels, frames).astype(np.float32)
    return np.asarray(v, dtype=np.float32)  # shape is validated by check_logmel


def load_clip(path: str | Path, regime: Regime = REGIME) -> np.ndarray:
    """Load one pre-computed training tensor (`mel/<clip_id>.npy`) and validate it."""
    return check_logmel(np.load(path, allow_pickle=False), regime)
