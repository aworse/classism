"""Hangul jamo tables plus a small composition / beam-search demo.

Spec D6: `LABELS` is the AFE's 33-jamo label list. Composition (`compose`) and beam search
(`beam_decode`) exist ONLY for standalone AFE demos/evaluation; the live pipeline delegates all
syllable composition to the LM Server, and nothing in `infer.py` imports them.
"""
from __future__ import annotations

import json
import math
import sys

CONSONANTS = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")  # 14
TENSE = list("ㄲㄸㅃㅆㅉ")  # 5
VOWELS = list("ㅏㅐㅑㅓㅔㅕㅗㅛㅜㅠㅡㅣ")  # 12
SHIFT_VOWELS = list("ㅒㅖ")  # 2

LABELS = CONSONANTS + TENSE + VOWELS + SHIFT_VOWELS  # 33 base jamo

# Unicode syllable layout: 0xAC00 + (lead * 21 + vowel) * 28 + tail
_LEADS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWELS_ALL = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_TAILS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
          "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_COMPOUND_V = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ",
               ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ", ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}
_COMPOUND_T = {("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ", ("ㄹ", "ㄱ"): "ㄺ",
               ("ㄹ", "ㅁ"): "ㄻ", ("ㄹ", "ㅂ"): "ㄼ", ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ",
               ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ", ("ㅂ", "ㅅ"): "ㅄ"}
_SPLIT_T = {v: k for k, v in _COMPOUND_T.items()}
_SINGLE_TAILS = {t for t in _TAILS if len(t) == 1 and t not in _COMPOUND_T.values()}

# Shift on the Dubeolsik layout.
SHIFT_MAP = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ", "ㅐ": "ㅒ", "ㅔ": "ㅖ"}

_IS_VOWEL = set(_VOWELS_ALL)


def _render(lead: str | None, vowel: str | None, tail: str | None) -> str:
    if lead and vowel:
        code = 0xAC00 + (_LEADS.index(lead) * 21 + _VOWELS_ALL.index(vowel)) * 28 + _TAILS.index(tail or "")
        return chr(code)
    return (lead or "") + (vowel or "") + (tail or "")  # stray jamo stay as-is


def compose(jamos: list[str]) -> str:
    """Compose a jamo sequence into syllables (Dubeolsik rules, incl. batchim re-attachment)."""
    out: list[str] = []
    lead = vowel = tail = None

    def flush() -> None:
        nonlocal lead, vowel, tail
        if lead or vowel or tail:
            out.append(_render(lead, vowel, tail))
        lead = vowel = tail = None

    for j in jamos:
        if j in _IS_VOWEL:
            if vowel is None and tail is None:
                if lead is None:
                    out.append(j)  # vowel with no lead: stray
                else:
                    vowel = j
            elif tail is None:
                comb = _COMPOUND_V.get((vowel, j))
                if comb:
                    vowel = comb
                else:
                    flush()
                    out.append(j)
            else:  # the tail moves to the front of the next syllable
                if tail in _SPLIT_T:
                    tail, moved = _SPLIT_T[tail]
                else:
                    tail, moved = None, tail
                flush()
                lead, vowel = moved, j
        else:
            if lead is None:
                lead = j
            elif vowel is None:
                flush()
                lead = j
            elif tail is None:
                if j in _SINGLE_TAILS:
                    tail = j
                else:
                    flush()
                    lead = j
            else:
                comb = _COMPOUND_T.get((tail, j))
                if comb:
                    tail = comb
                else:
                    flush()
                    lead = j
    flush()
    return "".join(out)


def is_hangul_syllable(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def apply_tokens(symbols: list[str]) -> str:
    """Interpret a symbol sequence (jamo + special tokens) as text. Demo-only semantics."""
    words: list[str] = []
    cur: list[str] = []
    shift = False
    for s in symbols:
        if s == "<sp>":
            words.append(compose(cur))
            cur = []
        elif s == "<bs>":
            if cur:
                cur.pop()
            elif words:
                cur = _decompose_word(words.pop())
        elif s == "<shift>":
            shift = True
            continue
        elif s in ("<caps>", "<other>"):
            pass
        else:
            cur.append(SHIFT_MAP.get(s, s) if shift else s)
        shift = False
    words.append(compose(cur))
    return " ".join(words)


def _decompose_word(word: str) -> list[str]:
    out: list[str] = []
    for ch in word:
        if is_hangul_syllable(ch):
            n = ord(ch) - 0xAC00
            lead, rest = divmod(n, 21 * 28)
            vowel, tail = divmod(rest, 28)
            out.append(_LEADS[lead])
            v = _VOWELS_ALL[vowel]
            out.extend(next(([a, b] for (a, b), c in _COMPOUND_V.items() if c == v), [v]))
            if tail:
                t = _TAILS[tail]
                out.extend(_SPLIT_T.get(t, (t,)))
        else:
            out.append(ch)
    return out


def _score_text(text: str) -> float:
    """Penalty for stray (uncomposed) jamo; the only 'language model' this demo has."""
    return -sum(1.0 for ch in text if ch != " " and not is_hangul_syllable(ch))


def beam_decode(token_alts: list[list[tuple[str, float]]], beam: int = 8, stray_penalty: float = 1.5) -> list[tuple[str, float]]:
    """Demo beam search over per-keystroke alternatives. Returns [(text, score)] best-first."""
    beams: list[tuple[float, list[str]]] = [(0.0, [])]
    for alts in token_alts:
        nxt = []
        for logp, seq in beams:
            for sym, p in alts:
                nxt.append((logp + math.log(max(p, 1e-12)), seq + [sym]))
        # prune on acoustic score + composition plausibility
        nxt.sort(key=lambda b: -(b[0] + stray_penalty * _score_text(apply_tokens(b[1]))))
        beams = nxt[:beam]
    scored = [(apply_tokens(seq), lp + stray_penalty * _score_text(apply_tokens(seq))) for lp, seq in beams]
    return sorted(scored, key=lambda t: -t[1])


def decode_document(doc: dict, beam: int = 8) -> list[tuple[str, float]]:
    """Run the demo decoder over a `kdaa.afe.v1` document."""
    alts = [[(a["jamo"], a["p"]) for a in tok["alts"]] for tok in doc["tokens"]]
    return beam_decode(alts, beam=beam)


if __name__ == "__main__":  # python -m src.hangul afe_output.json
    with open(sys.argv[1], encoding="utf-8") as f:
        for text, score in decode_document(json.load(f))[:5]:
            print(f"{score:8.3f}  {text}")
