"""Validation of `kdaa.afe.v1` documents: JSON Schema + the semantic rules the schema cannot express."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import labels
from .melio import REGIME

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "kdaa_afe_v1.schema.json"


class SchemaError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_document(doc: dict, *, tol: float = 1e-6) -> None:
    """Raise SchemaError unless `doc` is a valid kdaa.afe.v1 document (AC-1, AC-2 structure, AC-9)."""
    import jsonschema

    try:
        jsonschema.Draft202012Validator(load_schema()).validate(doc)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path)
        raise SchemaError(f"schema violation at '{path}': {e.message}") from e

    top_n = doc["meta"]["topN"]
    if list(doc["meta"]["input_shape"]) != list(REGIME.input_shape):
        raise SchemaError(f"meta.input_shape {doc['meta']['input_shape']} != regime {list(REGIME.input_shape)}")
    for i, tok in enumerate(doc["tokens"]):
        if tok["idx"] != i:
            raise SchemaError(f"tokens[{i}].idx is {tok['idx']}; idx must be contiguous from 0")
        alts = tok["alts"]
        if len(alts) > top_n:
            raise SchemaError(f"tokens[{i}] has {len(alts)} alts > topN {top_n}")
        ps = [a["p"] for a in alts]
        if abs(sum(ps) - 1.0) > tol:
            raise SchemaError(f"tokens[{i}].alts probabilities sum to {sum(ps)!r}, expected 1 (+-{tol})")
        if any(ps[k] < ps[k + 1] for k in range(len(ps) - 1)):
            raise SchemaError(f"tokens[{i}].alts are not probability-descending")
        if len({a["jamo"] for a in alts}) != len(alts):
            raise SchemaError(f"tokens[{i}].alts contains duplicate symbols")
        want = labels.keytype_of(alts[0]["jamo"])
        if tok["keytype"] != want:
            raise SchemaError(f"tokens[{i}].keytype {tok['keytype']!r} != {want!r} derived from top-1 {alts[0]['jamo']!r}")
