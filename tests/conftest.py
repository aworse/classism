import numpy as np
import pytest

from src import labels
from src.melio import REGIME


class FakePredictor:
    """Deterministic stand-in for the CNN: a fixed random linear map of the flattened log-mel.
    Lets the AFE post-processing (top-N, schema, streaming) be tested without any ML framework."""

    def __init__(self, seed: int = 0, scale: float = 0.05):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, scale, (REGIME.n_mels * REGIME.frames, labels.num_classes())).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4 and x.shape[1:] == REGIME.input_shape
        return x.reshape(len(x), -1) @ self.w


@pytest.fixture
def fake_predictor():
    return FakePredictor()


@pytest.fixture
def logmels():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, REGIME.n_mels, REGIME.frames)).astype(np.float32)
    return (x - x.mean(axis=(1, 2), keepdims=True)) / x.std(axis=(1, 2), keepdims=True)


@pytest.fixture
def onsets():
    return [0.5, 0.9, 1.234, 1.402, 2.0, 2.0, 3.5]  # non-decreasing, includes a tie
