from __future__ import annotations

import pytest
import regex
import numpy as np

import mojoregex
from mojoregex import _lib


def signature(found):
    if found is None:
        return None
    return (
        found.span(),
        found.group(),
        found.groups(),
        found.groupdict(),
        found.lastindex,
        found.lastgroup,
    )


@pytest.mark.parametrize(
    "pattern,text",
    [
        ("abc", "xxabczz"),
        ("a|ab", "xxab"),
        ("ab|a", "xxab"),
        ("a.*b", "xa12b34b"),
        ("a.*?b", "xa12b34b"),
        ("colou?r", "color colour"),
        ("ab+c", "xxabbbc"),
        ("ab*c", "xxac"),
        ("a{2,4}", "xaaaaa"),
        ("a{2,4}?", "xaaaaa"),
        ("(?:foo|bar)+", "xxfoobarbar!"),
        ("([A-Za-z_][A-Za-z_0-9]*)", "9 alpha_2!"),
        (r"(\w+)-(\d+)", "xxAb_9-123z"),
        (r"[^a-z]+", "abc123_Xyz"),
        (r"\D+\d+", "abc-123"),
        (r"\s+\S+", "x \t word!"),
        (r"\bcat\b", "a cat! scatter"),
        (r"\Bcat", "scatter cat"),
        (r"(a|ab)+b", "xxaabyy"),
        (r"(a+)\1", "xaaaa!"),
        (r"(?P<word>[a-z]+)-(?P=word)", "x test-test y"),
        (r"(a(b)?)+", "xxaba!"),
        (r"(ab|a)c", "xxac"),
        (r"(a|(b))c", "xxbc"),
        (r"\Aabc", "abc\nabc"),
        (r"abc\Z", "abc"),
        ("é.+界", "xéaβ界z"),
    ],
)
def test_search_parity(pattern, text):
    assert signature(mojoregex.search(pattern, text)) == signature(regex.search(pattern, text))


@pytest.mark.parametrize(
    "pattern,text,flags",
    [
        ("hello", "HeLLo", regex.I),
        ("[a-z]+", "ABCxyz", regex.I),
        ("^two", "one\ntwo\nthree", regex.M),
        ("two$", "one\ntwo\nthree", regex.M),
        ("a.b", "a\nb", regex.S),
        (r"^\w+$", "alpha_123", regex.A),
        (r"^\w+$", "élan", regex.A),
        ("a b  # words\n c", "abc", regex.X),
        ("^a.*z$", "A\nZ", regex.I | regex.S),
    ],
)
def test_flag_parity(pattern, text, flags):
    assert signature(mojoregex.search(pattern, text, flags)) == signature(
        regex.search(pattern, text, flags)
    )


@pytest.mark.parametrize(
    "pattern,text",
    [
        ("abc", "abcdef"),
        ("a+", "aaab"),
        ("a+?", "aaab"),
        ("(ab|a)b", "ab"),
        ("abc", "xabc"),
        ("", "abc"),
        (r"\b", "word"),
    ],
)
def test_match_parity(pattern, text):
    assert signature(mojoregex.match(pattern, text)) == signature(regex.match(pattern, text))


@pytest.mark.parametrize(
    "pattern,text",
    [
        ("abc", "abc"),
        ("abc", "abcd"),
        ("a*", "aaaa"),
        ("(ab|cd)+", "abcdab"),
        (r"(\w+)-\1", "same-same"),
        ("^abc$", "abc\n"),
    ],
)
def test_fullmatch_parity(pattern, text):
    assert signature(mojoregex.fullmatch(pattern, text)) == signature(
        regex.fullmatch(pattern, text)
    )


def test_pos_and_endpos_parity():
    ours = mojoregex.compile(r"^b|c$")
    theirs = regex.compile(r"^b|c$")
    for pos, endpos in [(0, 3), (1, 3), (1, 2), (-4, 99)]:
        assert signature(ours.search("abc", pos, endpos)) == signature(
            theirs.search("abc", pos, endpos)
        )


@pytest.mark.parametrize(
    "pattern,text",
    [
        (r"\w+", "one 22 three"),
        (r"(\w+)-(\d+)", "a-1 b-22 bad c-333"),
        (r"(a)|(b)", "abba"),
        (r".*?", "ab"),
        (r"\b", "two words"),
    ],
)
def test_findall_and_finditer_parity(pattern, text):
    assert mojoregex.findall(pattern, text) == regex.findall(pattern, text)
    assert [
        (match.span(), match.group(), match.groups())
        for match in mojoregex.finditer(pattern, text)
    ] == [
        (match.span(), match.group(), match.groups())
        for match in regex.finditer(pattern, text)
    ]


def test_overlapped_finditer_parity():
    pattern = mojoregex.compile("aba")
    expected = [(match.span(), match.group()) for match in regex.finditer("aba", "ababa", overlapped=True)]
    actual = [(match.span(), match.group()) for match in pattern.finditer("ababa", overlapped=True)]
    assert actual == expected


@pytest.mark.parametrize(
    "pattern,text,maxsplit",
    [
        (r"\s+", " one  two\tthree ", 0),
        (r"([,;])", "a,b;c,d", 0),
        (r",", "a,b,c,d", 2),
        (r"\b", "two words", 0),
    ],
)
def test_split_parity(pattern, text, maxsplit):
    assert mojoregex.split(pattern, text, maxsplit) == regex.split(pattern, text, maxsplit)


@pytest.mark.parametrize(
    "pattern,repl,text,count",
    [
        (r"(\w+)-(\d+)", r"\2:\1", "a-1 b-22", 0),
        (r"(?P<a>\w+):(?P<b>\w+)", r"\g<b>/\g<a>", "x:y a:b", 1),
        (r"(a)|(b)", r"<\1\2>", "abba", 0),
        (r"\s+", "_", " one  two ", 2),
        (r".*?", "-", "ab", 0),
    ],
)
def test_sub_and_subn_parity(pattern, repl, text, count):
    assert mojoregex.sub(pattern, repl, text, count) == regex.sub(pattern, repl, text, count)
    assert mojoregex.subn(pattern, repl, text, count) == regex.subn(pattern, repl, text, count)


def test_callable_sub_parity():
    transform = lambda match: f"[{match.group().upper()}]"
    text = "one 22"
    assert mojoregex.sub(r"\w+", transform, text) == regex.sub(r"\w+", transform, text)


def test_bytes_parity():
    pattern = br"(?P<key>[A-Z]+)=([^\s]+)"
    text = b"x ABC=\xff\x80 tail"
    assert signature(mojoregex.search(pattern, text)) == signature(regex.search(pattern, text))
    assert mojoregex.sub(br"([A-Z]+)", br"<\1>", text) == regex.sub(
        br"([A-Z]+)", br"<\1>", text
    )


def test_match_methods_and_expand():
    found = mojoregex.search(r"(?P<left>\w+)-(?P<right>\d+)", "x abc-12 y")
    assert found is not None
    assert found[0] == "abc-12"
    assert found["left"] == "abc"
    assert found.span("right") == (6, 8)
    assert found.groups("missing") == ("abc", "12")
    assert found.groupdict() == {"left": "abc", "right": "12"}
    assert found.expand(r"\g<right>:\g<left>") == "12:abc"


def test_scanner_parity():
    ours = mojoregex.compile(r"\w+").scanner("one 22 three")
    theirs = regex.compile(r"\w+").scanner("one 22 three")
    assert [ours.search().group(), ours.search().group(), ours.search().group()] == [
        theirs.search().group(), theirs.search().group(), theirs.search().group()
    ]
    assert ours.search() is None
    ours = mojoregex.compile(r"\w+").scanner("one 22")
    theirs = regex.compile(r"\w+").scanner("one 22")
    assert signature(ours.match()) == signature(theirs.match())
    assert ours.match() is None and theirs.match() is None


def test_escape_roundtrip():
    value = "a+b (c)\t[d]"
    escaped = mojoregex.escape(value)
    assert mojoregex.fullmatch(escaped, value)
    assert escaped == regex.escape(value)


def test_splititer_and_purge_api():
    assert list(mojoregex.splititer(r"([,])", "a,b,c", 1)) == regex.split(
        r"([,])", "a,b,c", 1
    )
    assert mojoregex.purge() is None


def test_simd_tail_and_repeat_backtracking_parity():
    repeated = "a" * 37
    cases = [
        (r"(a+)-\1", repeated + "-" + repeated),
        (r"[a-z]+z", "a" * 36 + "z"),
        (r"a+a", "a" * 37),
    ]
    for pattern, text in cases:
        assert signature(mojoregex.fullmatch(pattern, text)) == signature(
            regex.fullmatch(pattern, text)
        )


def test_prefix_parallel_threshold_parity():
    pattern = mojoregex.compile("needle")
    upstream = regex.compile("needle")
    for length in (3_999_990, 4_000_000):
        text = "x" * length + "needle"
        assert signature(pattern.search(text)) == signature(upstream.search(text))


def test_findall_multiple_native_batches_parity():
    text = " ".join(f"word{i % 100}" for i in range(17_000))
    assert mojoregex.findall(r"\w+", text) == regex.findall(r"\w+", text)


def test_native_buffer_contract_rejects_wrong_dtype_and_stride():
    with pytest.raises(TypeError):
        mojoregex._address(np.zeros(4, dtype=np.int32), np.int64)
    with pytest.raises(TypeError):
        mojoregex._address(np.zeros(8, dtype=np.int64)[::2], np.int64)


def test_native_entry_point_rejects_null_buffers():
    assert _lib.lib().mrx_exec(*([0] * 20)) == -4


def test_findall_retries_when_scratch_is_too_small(monkeypatch):
    original = mojoregex.Pattern._scratch

    def tiny_scratch(self, span):
        scratch = original(self, span)
        return [
            scratch[0],
            *(np.empty(8, dtype=np.int64) for _ in range(4)),
            *(np.empty(1, dtype=np.int64) for _ in range(2)),
        ]

    monkeypatch.setattr(mojoregex.Pattern, "_scratch", tiny_scratch)
    text = " ".join(["ab"] * 200)
    assert mojoregex.findall(r"((a|b)+)", text) == regex.findall(r"((a|b)+)", text)


@pytest.mark.parametrize(
    "pattern",
    [
        "(",
        "[abc",
        "a{4,2}",
        r"\q",
        r"(a*)*",
        "(?=a)",
        "a*+",
    ],
)
def test_invalid_or_unsupported_patterns_raise(pattern):
    with pytest.raises(mojoregex.error):
        mojoregex.compile(pattern)


def test_compiled_pattern_metadata():
    pattern = mojoregex.compile(r"(?P<a>a)(b)", mojoregex.I)
    assert pattern.pattern == r"(?P<a>a)(b)"
    assert pattern.flags & mojoregex.I
    assert pattern.groups == 2
    assert pattern.groupindex == {"a": 1}
    assert mojoregex.compile(pattern) is pattern
    with pytest.raises(ValueError):
        mojoregex.compile(pattern, mojoregex.M)


def test_unsupported_engine_modes_are_explicit():
    for pattern, flags in [
        ("abc", mojoregex.POSIX),
        ("abc", mojoregex.DEBUG),
        ("abc", mojoregex.V0 | mojoregex.V1),
        ("abc", mojoregex.ASCII | mojoregex.UNICODE),
        (b"abc", mojoregex.UNICODE),
        ("abc", 1 << 30),
    ]:
        with pytest.raises(mojoregex.error):
            mojoregex.compile(pattern, flags)
    with pytest.raises(NotImplementedError):
        mojoregex.search("abc", "abc", partial=True)
    with pytest.raises(NotImplementedError):
        mojoregex.search("abc", "abc", timeout=0.1)
