"""Single-head CNN classifiers (FR-2, T3). Input `(B,1,64,48)` -> `(B,38)` logits.

`SmallCNN` is the deployed baseline (conv/BN/ReLU/maxpool/FC only, so it converts cleanly to TFLite Micro
int8). `CoAtNetLite` is a heavier conv+attention variant for offline comparison ONLY (spec §8).
"""
from __future__ import annotations

import torch
from torch import nn

from .labels import num_classes as _num_classes
from .melio import REGIME


class SmallCNN(nn.Module):
    """4 x [conv3x3 - BN - ReLU - maxpool2] then flatten -> dropout -> linear (~123k params, ~10M MACs)."""

    def __init__(self, num_classes: int = _num_classes(), widths: tuple[int, ...] = (16, 32, 64, 96),
                 dropout: float = 0.2, n_mels: int = REGIME.n_mels, frames: int = REGIME.frames):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        h, wd = n_mels // 2 ** len(widths), frames // 2 ** len(widths)
        if h < 1 or wd < 1:
            raise ValueError("too many pooling stages for the input size")
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in * h * wd, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class _MBConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, expand: int = 2):
        super().__init__()
        mid = c_in * expand
        self.block = nn.Sequential(
            nn.Conv2d(c_in, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.res = stride == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.res else y


class CoAtNetLite(nn.Module):
    """Conv stem + 2 MBConv stages + 2 transformer layers over the 8x6 feature map (offline comparison only)."""

    def __init__(self, num_classes: int = _num_classes(), dim: int = 64, depth: int = 2, heads: int = 4,
                 dropout: float = 0.1, n_mels: int = REGIME.n_mels, frames: int = REGIME.frames):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU())
        self.stages = nn.Sequential(_MBConv(24, 32, 2), _MBConv(32, dim, 2))
        n_tokens = (n_mels // 8) * (frames // 8)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stages(self.stem(x))  # (B, dim, H, W)
        t = f.flatten(2).transpose(1, 2) + self.pos  # (B, H*W, dim)
        return self.head(self.norm(self.encoder(t)).mean(dim=1))


MODELS = {"smallcnn": SmallCNN, "coatnetlite": CoAtNetLite}


def build_model(name: str, num_classes: int = _num_classes(), **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def complexity(model: nn.Module, input_shape: tuple[int, ...] = (1, *REGIME.input_shape)) -> dict:
    """Parameters, int8 weight size upper bound (1 byte/param) and conv/linear MACs (attention matmuls excluded)."""
    macs = 0

    def hook(m, _inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv2d):
            macs += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features * (out.numel() // out.shape[-1] // out.shape[0] if out.dim() > 2 else 1)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    model.train(was_training)
    for h in handles:
        h.remove()
    n = count_params(model)
    return {"params": n, "int8_weight_bytes_upper_bound": n, "macs": int(macs)}
