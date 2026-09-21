"""38-symbol label space (spec §4): 33 base jamo + 5 special tokens, one flat classifier output (D3)."""
from __future__ import annotations

import numpy as np

from . import hangul

JAMO: tuple[str, ...] = tuple(hangul.LABELS)
SPECIAL: tuple[str, ...] = ("<sp>", "<bs>", "<shift>", "<caps>", "<other>")
LABELS: tuple[str, ...] = JAMO + SPECIAL

KEYTYPES: tuple[str, ...] = ("normal", "space", "backspace", "shift", "caps", "other")
_KEYTYPE_OF_SPECIAL = {"<sp>": "space", "<bs>": "backspace", "<shift>": "shift", "<caps>": "caps", "<other>": "other"}
_SPECIAL_OF_KEYTYPE = {v: k for k, v in _KEYTYPE_OF_SPECIAL.items()}

SYMBOL_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(LABELS)}
IDX_TO_SYMBOL: dict[int, str] = dict(enumerate(LABELS))
KEYTYPE_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(KEYTYPES)}


def num_classes() -> int:
    return len(LABELS)


def keytype_of(symbol: str) -> str:
    """Keytype derived from a symbol (§3.2): special tokens map to their type, base jamo to 'normal'."""
    if symbol in _KEYTYPE_OF_SPECIAL:
        return _KEYTYPE_OF_SPECIAL[symbol]
    if symbol in SYMBOL_TO_IDX:
        return "normal"
    raise ValueError(f"unknown symbol {symbol!r}")


def symbol_of(keytype: str, jamo: str | None = None) -> str:
    """Label-table row -> symbol. `jamo` is meaningful only when keytype == 'normal' (§4)."""
    if keytype == "normal":
        if not jamo or jamo not in JAMO:
            raise ValueError(f"keytype 'normal' needs a base jamo, got {jamo!r}")
        return jamo
    if keytype not in _SPECIAL_OF_KEYTYPE:
        raise ValueError(f"unknown keytype {keytype!r}; expected one of {KEYTYPES}")
    return _SPECIAL_OF_KEYTYPE[keytype]


def label_index(symbol: str) -> int:
    try:
        return SYMBOL_TO_IDX[symbol]
    except KeyError:
        raise ValueError(f"unknown symbol {symbol!r}") from None


# class index -> keytype index, for vectorised keytype accuracy
KEYTYPE_OF_CLASS = np.array([KEYTYPE_TO_IDX[keytype_of(s)] for s in LABELS], dtype=np.int64)

assert len(LABELS) == 38 and len(set(LABELS)) == 38, "label space must be 38 unique symbols"
