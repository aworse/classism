"""log-mel -> `kdaa.afe.v1` (FR-2..FR-5, FR-10; T5, T10).

The AFE is a pure function: validated log-mel in, symbols + probabilities out. It never composes
syllables (D6). All post-processing (softmax, top-N, renormalisation, keytype) is numpy-only, so the
model behind it is just a callable `predictor((B,1,64,48) float32) -> (B,38) logits`; a PyTorch
predictor and (in export_esp32) a TFLite predictor both fit.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from . import labels
from .calibrate import read_temperature, softmax
from .melio import REGIME, MelSession, Regime, check_logmel

Predictor = Callable[[np.ndarray], np.ndarray]
SCHEMA_ID = "kdaa.afe.v1"


def top_n_renorm(probs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-n class indices (probability-descending, ties -> lower index) and their renormalised probs (FR-3)."""
    order = np.argsort(-probs, axis=-1, kind="stable")[..., :n]
    p = np.take_along_axis(probs, order, axis=-1)
    return order, p / p.sum(axis=-1, keepdims=True)


class AFE:
    def __init__(self, predictor: Predictor, *, model_id: str = "smallcnn-fp32-v1", top_n: int = 5,
                 temperature: float = 1.0, calibrated: bool = False, batch_size: int = 64,
                 regime: Regime = REGIME, lang: str = "ko"):
        if not 1 <= top_n <= labels.num_classes():
            raise ValueError(f"top_n must be in [1, {labels.num_classes()}]")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.predictor, self.model_id, self.top_n = predictor, model_id, top_n
        self.temperature, self.calibrated, self.batch_size = float(temperature), bool(calibrated), batch_size
        self.regime, self.lang = regime, lang

    # ---- core -------------------------------------------------------------------------------
    def logits(self, logmels: Iterable) -> np.ndarray:
        """Validate every log-mel (raises RegimeError on mismatch, AC-8) and return `(N, 38)` logits."""
        x = [check_logmel(m, self.regime) for m in logmels]
        if not x:
            return np.zeros((0, labels.num_classes()), dtype=np.float32)
        batch = np.stack(x)[:, None]  # (N,1,n_mels,frames)
        out = [np.asarray(self.predictor(batch[i:i + self.batch_size]), dtype=np.float32)
               for i in range(0, len(batch), self.batch_size)]
        logits = np.concatenate(out)
        if logits.shape != (len(x), labels.num_classes()):
            raise ValueError(f"predictor returned {logits.shape}, expected {(len(x), labels.num_classes())}")
        return logits

    def tokens_from_logits(self, logits: np.ndarray, onsets, start_idx: int = 0) -> list[dict]:
        order, p = top_n_renorm(softmax(logits, self.temperature), self.top_n)
        toks = []
        for k, (row, prow) in enumerate(zip(order, p)):
            alts = [{"jamo": labels.IDX_TO_SYMBOL[int(c)], "p": float(q)} for c, q in zip(row, prow)]
            toks.append({"idx": start_idx + k, "onset_s": float(onsets[k]),
                         "keytype": labels.keytype_of(alts[0]["jamo"]), "alts": alts})
        return toks

    def document(self, session_id: str, tokens: list[dict]) -> dict:
        return {"schema": SCHEMA_ID, "sample_rate": self.regime.sample_rate, "session_id": session_id,
                "lang": self.lang, "tokens": tokens,
                "meta": {"model_id": self.model_id, "topN": self.top_n,
                         "input_shape": list(self.regime.input_shape), "calibrated": self.calibrated}}

    # ---- batch (session) mode ---------------------------------------------------------------
    def run_arrays(self, logmels, onsets, session_id: str = "session") -> dict:
        logmels, onsets = list(logmels), list(onsets)
        if len(logmels) != len(onsets):
            raise ValueError(f"{len(logmels)} log-mels but {len(onsets)} onsets")
        return self.document(session_id, self.tokens_from_logits(self.logits(logmels), onsets))

    def run(self, session: MelSession, *, verify_hash: bool = True) -> dict:
        """FR-5: a Mel session envelope -> kdaa.afe.v1 document (one token per keystroke, order and onset_s preserved)."""
        session.check(self.regime, verify_hash=verify_hash)
        return self.run_arrays(session.logmels(), session.onsets(), session.session_id)

    # ---- streaming mode (FR-10 / T10) -------------------------------------------------------
    def stream(self, session_id: str = "stream") -> "AFEStream":
        return AFEStream(self, session_id)


class AFEStream:
    """Incremental emission: one token per pushed keystroke, no need to wait for the session to end."""

    def __init__(self, afe: AFE, session_id: str):
        self.afe, self.session_id = afe, session_id
        self.tokens: list[dict] = []
        self.latencies_ms: list[float] = []  # host wall-clock per push; NOT the ESP32 figure (NFR-1)

    def push(self, onset_s: float, logmel, idx: int | None = None) -> dict:
        if idx is not None and idx != len(self.tokens):
            raise ValueError(f"idx must be contiguous: expected {len(self.tokens)}, got {idx}")
        if self.tokens and onset_s < self.tokens[-1]["onset_s"]:
            raise ValueError("onset_s must be time-ordered")
        t0 = time.perf_counter()
        (tok,) = self.afe.tokens_from_logits(self.afe.logits([logmel]), [onset_s], start_idx=len(self.tokens))
        self.latencies_ms.append((time.perf_counter() - t0) * 1e3)
        self.tokens.append(tok)
        return tok

    def run(self, keystrokes: Iterable[tuple[float, np.ndarray]]) -> Iterator[dict]:
        for onset_s, logmel in keystrokes:
            yield self.push(onset_s, logmel)

    def document(self) -> dict:
        return self.afe.document(self.session_id, list(self.tokens))


# ---- PyTorch glue (torch imported lazily so the numpy paths above stay light) ------------------
class TorchPredictor:
    def __init__(self, model, device: str = "cpu"):
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            t = self._torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
            return self.model(t).float().cpu().numpy()


def load_predictor(ckpt_path: str | Path, device: str = "cpu") -> tuple[TorchPredictor, dict]:
    """Rebuild the model saved by train.py. Returns (predictor, checkpoint metadata)."""
    import torch

    from .model import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck.get("param_hash") != REGIME.param_hash():
        raise ValueError(f"checkpoint param_hash {ck.get('param_hash')!r} != current regime {REGIME.param_hash()!r} (NFR-2)")
    if list(ck["labels"]) != list(labels.LABELS):
        raise ValueError("checkpoint label order differs from src/labels.py")
    model = build_model(ck["model_name"], ck["num_classes"], **ck.get("model_kwargs", {}))
    model.load_state_dict(ck["state_dict"])
    return TorchPredictor(model, device), {k: v for k, v in ck.items() if k != "state_dict"}


def afe_from_checkpoint(ckpt_path: str | Path, *, top_n: int = 5, use_calibration: bool = True,
                        batch_size: int = 64, device: str = "cpu") -> AFE:
    """AFE around a trained checkpoint; applies `calibration.json` beside it when present (FR-8)."""
    predictor, meta = load_predictor(ckpt_path, device)
    t = read_temperature(Path(ckpt_path).parent / "calibration.json") if use_calibration else None
    return AFE(predictor, model_id=meta.get("model_id", "smallcnn-fp32-v1"), top_n=top_n,
               temperature=t or 1.0, calibrated=t is not None, batch_size=batch_size)


def predict_dataset(predictor: Predictor, ds, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Logits `(N,38)` and int labels `(N,)` for a LogMelDataset (raw, un-tempered)."""
    x, y = ds.arrays()
    if len(x) == 0:
        return np.zeros((0, labels.num_classes()), np.float32), y
    logits = np.concatenate([np.asarray(predictor(x[i:i + batch_size]), np.float32) for i in range(0, len(x), batch_size)])
    return logits, y
