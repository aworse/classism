import pytest

from src import hangul, labels


def test_label_space_is_38():
    assert labels.num_classes() == 38
    assert len(hangul.LABELS) == 33 and len(set(labels.LABELS)) == 38
    assert (len(hangul.CONSONANTS), len(hangul.TENSE), len(hangul.VOWELS), len(hangul.SHIFT_VOWELS)) == (14, 5, 12, 2)


def test_keytype_mapping():
    assert labels.keytype_of("ㄱ") == "normal"
    assert [labels.keytype_of(s) for s in labels.SPECIAL] == ["space", "backspace", "shift", "caps", "other"]
    with pytest.raises(ValueError):
        labels.keytype_of("x")


def test_index_maps_roundtrip():
    for i, s in enumerate(labels.LABELS):
        assert labels.SYMBOL_TO_IDX[s] == i and labels.IDX_TO_SYMBOL[i] == s
    assert all(labels.KEYTYPE_OF_CLASS[i] == labels.KEYTYPE_TO_IDX[labels.keytype_of(s)] for i, s in enumerate(labels.LABELS))


def test_symbol_of_row():
    assert labels.symbol_of("normal", "ㅏ") == "ㅏ"
    assert labels.symbol_of("space", "") == "<sp>"
    with pytest.raises(ValueError):
        labels.symbol_of("normal", "")
    with pytest.raises(ValueError):
        labels.symbol_of("bogus", "ㄱ")


@pytest.mark.parametrize("jamos,text", [
    ("ㅎㅏㄴㄱㅡㄹ", "한글"),
    ("ㅅㅏㄹㅏㅁ", "사람"),
    ("ㅇㅏㄴㄴㅕㅇ", "안녕"),
    ("ㄷㅏㄹㄱㅣ", "달기"),          # tail ㄹ + ㄱ then vowel: ㄱ moves to the next syllable
    ("ㄷㅏㄹㄱ", "닭"),              # compound batchim
    ("ㅇㅗㅏ", "와"),                # compound vowel
    ("ㄱㅏㅂㅅㅇㅣ", "값이"),        # compound batchim ㅄ stays, ㅇ starts the next syllable
])
def test_compose(jamos, text):
    assert hangul.compose(list(jamos)) == text


def test_apply_tokens_space_backspace_shift():
    assert hangul.apply_tokens(list("ㅎㅏㄴ") + ["<sp>"] + list("ㄱㅡㄹ")) == "한 글"
    assert hangul.apply_tokens(list("ㅎㅏㄴㅇ") + ["<bs>"]) == "한"
    assert hangul.apply_tokens(["<shift>", "ㄱ", "ㅏ"]) == "까"


def test_beam_prefers_wellformed_syllables():
    alts = [[("ㅎ", 0.5), ("ㅏ", 0.5)], [("ㅏ", 0.6), ("ㅎ", 0.4)]]
    text, _ = hangul.beam_decode(alts)[0]
    assert text == "하"
