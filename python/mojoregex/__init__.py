"""A Python-compatible API backed by a Mojo ordered-backtracking engine."""

from __future__ import annotations

from array import array
import ctypes
from enum import IntFlag
from functools import lru_cache
import sys
from typing import Any, Callable, Iterator

import numpy as np

from ._compiler import CompileError, compile_pattern
from ._lib import lib


class RegexFlag(IntFlag):
    A = ASCII = 0x80
    B = BESTMATCH = 0x1000
    D = DEBUG = 0x200
    E = ENHANCEMATCH = 0x8000
    F = FULLCASE = 0x4000
    I = IGNORECASE = 0x2
    L = LOCALE = 0x4
    M = MULTILINE = 0x8
    P = POSIX = 0x10000
    R = REVERSE = 0x400
    S = DOTALL = 0x10
    U = UNICODE = 0x20
    V0 = VERSION0 = 0x2000
    V1 = VERSION1 = 0x100
    W = WORD = 0x800
    X = VERBOSE = 0x40


A = ASCII = RegexFlag.ASCII
B = BESTMATCH = RegexFlag.BESTMATCH
D = DEBUG = RegexFlag.DEBUG
E = ENHANCEMATCH = RegexFlag.ENHANCEMATCH
F = FULLCASE = RegexFlag.FULLCASE
I = IGNORECASE = RegexFlag.IGNORECASE
L = LOCALE = RegexFlag.LOCALE
M = MULTILINE = RegexFlag.MULTILINE
P = POSIX = RegexFlag.POSIX
R = REVERSE = RegexFlag.REVERSE
S = DOTALL = RegexFlag.DOTALL
U = UNICODE = RegexFlag.UNICODE
V0 = VERSION0 = RegexFlag.VERSION0
V1 = VERSION1 = RegexFlag.VERSION1
W = WORD = RegexFlag.WORD
X = VERBOSE = RegexFlag.VERBOSE
DEFAULT_VERSION = VERSION0

error = CompileError

_UNSUPPORTED_FLAGS = (
    BESTMATCH | DEBUG | ENHANCEMATCH | FULLCASE | LOCALE | POSIX | REVERSE | WORD
)
_KNOWN_FLAGS = (
    ASCII | BESTMATCH | DEBUG | ENHANCEMATCH | FULLCASE | IGNORECASE | LOCALE
    | MULTILINE | POSIX | REVERSE | DOTALL | UNICODE | VERSION0 | VERSION1
    | WORD | VERBOSE
)

_unicode_utf8 = ctypes.pythonapi.PyUnicode_AsUTF8AndSize
_unicode_utf8.argtypes = [ctypes.py_object, ctypes.POINTER(ctypes.c_ssize_t)]
_unicode_utf8.restype = ctypes.c_void_p


def _address(values, dtype) -> int:
    if isinstance(values, np.ndarray):
        if values.dtype != dtype or not values.flags.c_contiguous:
            raise TypeError(f"native buffer must be contiguous {np.dtype(dtype).name}")
        address = int(values.ctypes.data)
    elif isinstance(values, array):
        expected_typecode = "q" if np.dtype(dtype) == np.dtype(np.int64) else None
        if values.typecode != expected_typecode or values.itemsize != np.dtype(dtype).itemsize:
            raise TypeError(f"native buffer must contain {np.dtype(dtype).name} values")
        address = values.buffer_info()[0]
    else:
        raise TypeError("native buffer must be a NumPy array or array.array")
    if address == 0:
        raise RuntimeError("native buffer has a null address")
    return address


def _text_array(value: str | bytes, force_wide: bool = False) -> np.ndarray:
    if isinstance(value, str):
        if value.isascii() and not force_wide:
            size = ctypes.c_ssize_t()
            address = _unicode_utf8(value, ctypes.byref(size))
            if not address:
                raise RuntimeError("could not access the string buffer")
            buffer = (ctypes.c_uint8 * size.value).from_address(address)
            values = np.ctypeslib.as_array(buffer)
        else:
            encoding = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
            values = np.frombuffer(
                value.encode(encoding, errors="surrogatepass"), dtype=np.uint32
            )
    elif isinstance(value, bytes):
        values = np.frombuffer(value, dtype=np.uint8)
        if force_wide:
            values = values.astype(np.uint32)
    else:
        raise TypeError("expected string or bytes-like object")
    if len(values) == 0:
        values = np.zeros(1, dtype=np.uint32)
    return values


def _slice(value: str | bytes, start: int, end: int):
    return value[start:end]


class Match:
    __slots__ = ("re", "string", "pos", "endpos", "_spans", "lastindex", "lastgroup")

    def __init__(
        self,
        pattern: Pattern,
        string: str | bytes,
        pos: int,
        endpos: int,
        spans: tuple[tuple[int, int], ...],
        lastindex: int | None,
        lastgroup: str | None,
    ):
        self.re = pattern
        self.string = string
        self.pos = pos
        self.endpos = endpos
        self._spans = spans
        self.lastindex = lastindex
        self.lastgroup = lastgroup

    def _group_number(self, group: int | str) -> int:
        if isinstance(group, str):
            try:
                return self.re.groupindex[group]
            except KeyError:
                raise IndexError(f"unknown group name {group!r}") from None
        if not isinstance(group, int) or group < 0 or group > self.re.groups:
            raise IndexError("no such group")
        return group

    def group(self, *groups):
        if not groups:
            groups = (0,)
        values = []
        for group in groups:
            number = self._group_number(group)
            start, end = self._spans[number]
            values.append(None if start < 0 else _slice(self.string, start, end))
        return values[0] if len(values) == 1 else tuple(values)

    __getitem__ = group

    def groups(self, default=None):
        return tuple(
            default if self._spans[i][0] < 0 else _slice(self.string, *self._spans[i])
            for i in range(1, self.re.groups + 1)
        )

    def groupdict(self, default=None):
        return {
            name: default if self._spans[number][0] < 0 else self.group(number)
            for name, number in self.re.groupindex.items()
        }

    def start(self, group=0) -> int:
        return self._spans[self._group_number(group)][0]

    def end(self, group=0) -> int:
        return self._spans[self._group_number(group)][1]

    def span(self, group=0) -> tuple[int, int]:
        return self._spans[self._group_number(group)]

    def expand(self, template):
        return _expand(template, self)

    def __bool__(self):
        return True

    def __repr__(self) -> str:
        return f"<mojoregex.Match object; span={self.span()}, match={self.group()!r}>"


class Pattern:
    __slots__ = (
        "pattern", "flags", "groups", "groupindex", "_indexgroup", "_code",
        "_ranges", "_prefix", "_bytes", "_has_loop", "_has_repeat_class",
        "_cycle_writes",
    )

    def __init__(self, pattern: str | bytes, flags: int = 0):
        if not isinstance(pattern, (str, bytes)):
            raise TypeError("first argument must be string or compiled pattern")
        flags = int(flags)
        unknown = flags & ~int(_KNOWN_FLAGS)
        if unknown:
            raise CompileError(f"unknown regex flags: 0x{unknown:x}")
        if flags & int(_UNSUPPORTED_FLAGS):
            unsupported = RegexFlag(flags) & _UNSUPPORTED_FLAGS
            raise CompileError(f"unsupported regex flags: {unsupported!s}")
        if flags & int(VERSION0) and flags & int(VERSION1):
            raise CompileError("VERSION0 and VERSION1 are mutually exclusive")
        if isinstance(pattern, bytes) and flags & int(UNICODE):
            raise CompileError("cannot use UNICODE with a bytes pattern")
        if flags & int(ASCII) and flags & int(UNICODE):
            raise CompileError("ASCII and UNICODE are mutually exclusive")
        if not flags & int(VERSION0 | VERSION1):
            flags |= int(DEFAULT_VERSION)
        if isinstance(pattern, str) and not flags & int(ASCII | LOCALE):
            flags |= int(UNICODE)
        elif isinstance(pattern, bytes):
            flags |= int(ASCII)
        source = pattern.decode("latin1") if isinstance(pattern, bytes) else pattern
        code, ranges, groups, groupindex, flags, prefix = compile_pattern(source, flags)
        self.pattern = pattern
        self.flags = RegexFlag(flags)
        self.groups = groups
        self.groupindex = dict(groupindex)
        self._indexgroup = {number: name for name, number in groupindex.items()}
        self._code = array("q", code)
        self._ranges = array("q", ranges or [0, 0])
        self._prefix = prefix
        self._bytes = isinstance(pattern, bytes)
        instructions = [code[i:i + 3] for i in range(0, len(code), 3)]
        loops = [
            (instruction[1], i)
            for i, instruction in enumerate(instructions)
            if instruction[0] == 4 and instruction[1] <= i
        ]
        self._has_loop = bool(loops)
        self._has_repeat_class = any(instruction[0] == 15 for instruction in instructions)
        cyclic = set()
        for start, stop in loops:
            cyclic.update(range(start, stop + 1))
        self._cycle_writes = sum(
            1 if instructions[i][0] == 6 else 1 + bool(instructions[i][2])
            if instructions[i][0] == 14 else 0
            for i in cyclic
        )

    def _check_string(self, string):
        if self._bytes != isinstance(string, bytes):
            expected = "bytes" if self._bytes else "string"
            raise TypeError(f"cannot use this pattern on a {type(string).__name__}; expected {expected}")

    @staticmethod
    def _bounds(string, pos, endpos):
        length = len(string)
        if pos is None:
            pos = 0
        pos = max(0, int(pos))
        if pos > length:
            pos = length
        if endpos is None:
            endpos = length
        else:
            endpos = max(0, min(length, int(endpos)))
        return pos, endpos

    def _scratch(self, span):
        slot_count = 2 * (self.groups + 1) + 2
        instruction_count = len(self._code) // 3
        stack_capacity = (
            max(64, span + instruction_count + 8)
            if self._has_loop else max(64, instruction_count + 8)
        )
        log_capacity = max(
            64, instruction_count * 2 + self._cycle_writes * (span + 1)
        )
        return [
            np.empty(slot_count, dtype=np.int64),
            np.empty(stack_capacity, dtype=np.int64),
            np.empty(stack_capacity, dtype=np.int64),
            np.empty(stack_capacity, dtype=np.int64),
            np.empty(stack_capacity, dtype=np.int64),
            np.empty(log_capacity, dtype=np.int64),
            np.empty(log_capacity, dtype=np.int64),
        ]

    def _exec(self, string, pos=0, endpos=None, mode=0, _prepared=None) -> Match | None:
        self._check_string(string)
        pos, endpos = self._bounds(string, pos, endpos)
        slot_count = 2 * (self.groups + 1) + 2
        span = max(0, endpos - pos)
        instruction_count = len(self._code) // 3
        if _prepared is None:
            text = _text_array(
                string,
                force_wide=(
                    mode == 2 and span >= 65_536 and self._has_repeat_class
                ),
            )
            scratch = self._scratch(span)
        else:
            text, scratch = _prepared
        captures, stack_pc, stack_pos, stack_log, stack_min, log_slot, log_old = scratch
        stack_capacity = len(stack_pc)
        log_capacity = len(log_slot)
        while True:
            status = lib().mrx_exec(
                _address(self._code, np.int64), instruction_count,
                _address(self._ranges, np.int64),
                _address(text, text.dtype), len(string), text.itemsize,
                pos, endpos, mode,
                int(self.flags), self._prefix, _address(captures, np.int64),
                slot_count, _address(stack_pc, np.int64),
                _address(stack_pos, np.int64), _address(stack_log, np.int64),
                _address(stack_min, np.int64), stack_capacity,
                _address(log_slot, np.int64), _address(log_old, np.int64),
                log_capacity,
            )
            if status != -2:
                break
            stack_capacity *= 2
            log_capacity *= 2
            stack_pc = np.empty(stack_capacity, dtype=np.int64)
            stack_pos = np.empty(stack_capacity, dtype=np.int64)
            stack_log = np.empty(stack_capacity, dtype=np.int64)
            stack_min = np.empty(stack_capacity, dtype=np.int64)
            log_slot = np.empty(log_capacity, dtype=np.int64)
            log_old = np.empty(log_capacity, dtype=np.int64)
            scratch[:] = [
                captures, stack_pc, stack_pos, stack_log, stack_min,
                log_slot, log_old,
            ]
        if status == 0:
            return None
        if status < 0:
            raise RuntimeError(f"Mojo regex engine failed with status {status}")
        spans = tuple(
            (int(captures[i * 2]), int(captures[i * 2 + 1]))
            for i in range(self.groups + 1)
        )
        lastindex = int(captures[-2]) if captures[-2] >= 0 else None
        lastgroup = self._indexgroup.get(int(captures[-1])) if captures[-1] >= 0 else None
        return Match(self, string, pos, endpos, spans, lastindex, lastgroup)

    def search(self, string, pos=0, endpos=None, *, concurrent=None, partial=False, timeout=None):
        _options(concurrent, partial, timeout)
        return self._exec(string, pos, endpos, 0)

    def match(self, string, pos=0, endpos=None, *, concurrent=None, partial=False, timeout=None):
        _options(concurrent, partial, timeout)
        return self._exec(string, pos, endpos, 1)

    def fullmatch(self, string, pos=0, endpos=None, *, concurrent=None, partial=False, timeout=None):
        _options(concurrent, partial, timeout)
        return self._exec(string, pos, endpos, 2)

    def finditer(self, string, pos=0, endpos=None, *, overlapped=False, concurrent=None, partial=False, timeout=None):
        _options(concurrent, partial, timeout)
        self._check_string(string)
        pos, endpos = self._bounds(string, pos, endpos)
        prepared = (_text_array(string), self._scratch(max(0, endpos - pos)))
        cursor = pos
        while cursor <= endpos:
            found = self._exec(string, cursor, endpos, 0, prepared)
            if found is None:
                break
            yield found
            start, stop = found.span()
            if overlapped:
                cursor = start + 1
            elif stop > start:
                cursor = stop
            else:
                nonempty = self._exec(string, start, endpos, 3, prepared)
                if nonempty is not None and nonempty.start() == start:
                    yield nonempty
                    cursor = nonempty.end()
                else:
                    cursor = start + 1

    def findall(self, string, pos=0, endpos=None, *, overlapped=False, concurrent=None, timeout=None):
        _options(concurrent, False, timeout)
        self._check_string(string)
        pos, endpos = self._bounds(string, pos, endpos)
        text = _text_array(string)
        scratch = self._scratch(max(0, endpos - pos))
        captures, stack_pc, stack_pos, stack_log, stack_min, log_slot, log_old = scratch
        slot_count = 2 * (self.groups + 1) + 2
        span_slots = slot_count - 2
        instruction_count = len(self._code) // 3
        stack_capacity = len(stack_pc)
        log_capacity = len(log_slot)
        batch_capacity = 16384
        results = np.empty((batch_capacity, span_slots), dtype=np.int64)
        cursor = np.array([pos], dtype=np.int64)
        empty = b"" if self._bytes else ""
        values = []
        while cursor[0] <= endpos:
            count = lib().mrx_find_all(
                _address(self._code, np.int64), instruction_count,
                _address(self._ranges, np.int64),
                _address(text, text.dtype), len(string), text.itemsize,
                pos, endpos, 0,
                int(self.flags), self._prefix, _address(captures, np.int64),
                slot_count, _address(stack_pc, np.int64),
                _address(stack_pos, np.int64), _address(stack_log, np.int64),
                _address(stack_min, np.int64), stack_capacity,
                _address(log_slot, np.int64), _address(log_old, np.int64),
                log_capacity, _address(results, np.int64), batch_capacity,
                _address(cursor, np.int64), int(overlapped),
            )
            if count == -2:
                stack_capacity *= 2
                log_capacity *= 2
                stack_pc = np.empty(stack_capacity, dtype=np.int64)
                stack_pos = np.empty(stack_capacity, dtype=np.int64)
                stack_log = np.empty(stack_capacity, dtype=np.int64)
                stack_min = np.empty(stack_capacity, dtype=np.int64)
                log_slot = np.empty(log_capacity, dtype=np.int64)
                log_old = np.empty(log_capacity, dtype=np.int64)
                continue
            if count < 0:
                raise RuntimeError(f"Mojo regex engine failed with status {count}")
            rows = results[:count].tolist()
            if self.groups == 0:
                batch_values = [string[row[0]:row[1]] for row in rows]
            elif self.groups == 1:
                batch_values = [
                    empty if row[2] < 0 else string[row[2]:row[3]]
                    for row in rows
                ]
            else:
                batch_values = [tuple(
                    empty if row[group * 2] < 0 else string[
                        row[group * 2]:row[group * 2 + 1]
                    ]
                    for group in range(1, self.groups + 1)
                ) for row in rows]
            if values:
                values.extend(batch_values)
            else:
                values = batch_values
            if count == 0:
                break
        return values

    def split(self, string, maxsplit=0, *, concurrent=None, timeout=None):
        _options(concurrent, False, timeout)
        pieces = []
        last = 0
        splits = 0
        for found in self.finditer(string):
            if maxsplit and splits >= maxsplit:
                break
            pieces.append(_slice(string, last, found.start()))
            pieces.extend(found.groups())
            last = found.end()
            splits += 1
        pieces.append(_slice(string, last, len(string)))
        return pieces

    def sub(self, repl, string, count=0, pos=0, endpos=None, *, concurrent=None, timeout=None):
        return self.subn(repl, string, count, pos, endpos, concurrent=concurrent, timeout=timeout)[0]

    def subn(self, repl, string, count=0, pos=0, endpos=None, *, concurrent=None, timeout=None):
        _options(concurrent, False, timeout)
        self._check_string(string)
        pos, endpos = self._bounds(string, pos, endpos)
        pieces = [_slice(string, 0, pos)]
        last = pos
        replaced = 0
        for found in self.finditer(string, pos, endpos):
            if count and replaced >= count:
                break
            pieces.append(_slice(string, last, found.start()))
            pieces.append(repl(found) if callable(repl) else _expand(repl, found))
            last = found.end()
            replaced += 1
        pieces.append(_slice(string, last, len(string)))
        separator = b"" if self._bytes else ""
        try:
            result = separator.join(pieces)
        except TypeError:
            raise TypeError("replacement must have the same string type as the pattern") from None
        return result, replaced

    def scanner(self, string, pos=0, endpos=None, *, overlapped=False, concurrent=None, partial=False, timeout=None):
        _options(concurrent, partial, timeout)
        return Scanner(self, string, pos, endpos, overlapped)

    def __repr__(self) -> str:
        return f"mojoregex.compile({self.pattern!r}, {self.flags!r})"


class Scanner:
    def __init__(self, pattern: Pattern, string, pos=0, endpos=None, overlapped=False):
        self.pattern = pattern
        self.string = string
        self.pos, self.endpos = pattern._bounds(string, pos, endpos)
        self.overlapped = overlapped

    def search(self):
        if self.pos > self.endpos:
            return None
        found = self.pattern.search(self.string, self.pos, self.endpos)
        if found is None:
            self.pos = self.endpos + 1
        else:
            start, end = found.span()
            self.pos = start + 1 if self.overlapped else end if end > start else start + 1
        return found

    def match(self):
        if self.pos > self.endpos:
            return None
        found = self.pattern.match(self.string, self.pos, self.endpos)
        if found is None:
            return None
        start, end = found.span()
        self.pos = start + 1 if self.overlapped else end if end > start else start + 1
        return found


def _options(concurrent, partial, timeout):
    if partial:
        raise NotImplementedError("partial matching is not supported")
    if timeout is not None:
        raise NotImplementedError("match timeouts are not supported")


def _expand(template, found: Match):
    is_bytes = isinstance(template, bytes)
    if is_bytes != isinstance(found.string, bytes):
        raise TypeError("replacement must have the same string type as the pattern")
    text = template.decode("latin1") if is_bytes else template
    result = []
    i = 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v", "a": "\a", "\\": "\\"}
    while i < len(text):
        if text[i] != "\\":
            result.append(text[i])
            i += 1
            continue
        i += 1
        if i >= len(text):
            raise CompileError("bad escape at end of replacement")
        if text[i] == "g" and i + 1 < len(text) and text[i + 1] == "<":
            close = text.find(">", i + 2)
            if close < 0:
                raise CompileError("missing group name terminator")
            key = text[i + 2:close]
            group: int | str = int(key) if key.isdigit() else key
            result.append(found.group(group) or (b"" if is_bytes else ""))
            i = close + 1
        elif text[i].isdigit():
            start = i
            while i < len(text) and text[i].isdigit():
                i += 1
            result.append(found.group(int(text[start:i])) or (b"" if is_bytes else ""))
        elif text[i] in escapes:
            result.append(escapes[text[i]])
            i += 1
        else:
            result.append("\\" + text[i])
            i += 1
    if is_bytes:
        return b"".join(part if isinstance(part, bytes) else part.encode("latin1") for part in result)
    return "".join(result)


@lru_cache(maxsize=512)
def _compile_cached(pattern: str | bytes, flags: int) -> Pattern:
    return Pattern(pattern, flags)


def compile(pattern, flags=0, ignore_unused=False, cache_pattern=None, **kwargs) -> Pattern:
    if kwargs:
        raise ValueError(f"unused keyword arguments: {', '.join(kwargs)}")
    if isinstance(pattern, Pattern):
        if flags:
            raise ValueError("cannot process flags argument with a compiled pattern")
        return pattern
    flags = int(flags)
    if cache_pattern is False:
        return Pattern(pattern, flags)
    return _compile_cached(pattern, flags)


def search(pattern, string, flags=0, pos=None, endpos=None, partial=False, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).search(
        string, pos, endpos, partial=partial, concurrent=concurrent, timeout=timeout
    )


def match(pattern, string, flags=0, pos=None, endpos=None, partial=False, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).match(
        string, pos, endpos, partial=partial, concurrent=concurrent, timeout=timeout
    )


def fullmatch(pattern, string, flags=0, pos=None, endpos=None, partial=False, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).fullmatch(
        string, pos, endpos, partial=partial, concurrent=concurrent, timeout=timeout
    )


def findall(pattern, string, flags=0, pos=None, endpos=None, overlapped=False, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).findall(
        string, pos, endpos, overlapped=overlapped, concurrent=concurrent, timeout=timeout
    )


def finditer(pattern, string, flags=0, pos=None, endpos=None, overlapped=False, partial=False, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).finditer(
        string, pos, endpos, overlapped=overlapped, partial=partial,
        concurrent=concurrent, timeout=timeout
    )


def split(pattern, string, maxsplit=0, flags=0, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).split(
        string, maxsplit, concurrent=concurrent, timeout=timeout
    )


def splititer(pattern, string, maxsplit=0, flags=0, concurrent=None, timeout=None, **kwargs):
    yield from split(pattern, string, maxsplit, flags, concurrent, timeout, **kwargs)


def sub(pattern, repl, string, count=0, flags=0, pos=None, endpos=None, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).sub(
        repl, string, count, pos, endpos, concurrent=concurrent, timeout=timeout
    )


def subn(pattern, repl, string, count=0, flags=0, pos=None, endpos=None, concurrent=None, timeout=None, **kwargs):
    return compile(pattern, flags, **kwargs).subn(
        repl, string, count, pos, endpos, concurrent=concurrent, timeout=timeout
    )


def escape(pattern, special_only=True, literal_spaces=False):
    if not isinstance(pattern, (str, bytes)):
        raise TypeError("pattern must be a string or bytes")
    is_bytes = isinstance(pattern, bytes)
    text = pattern.decode("latin1") if is_bytes else pattern
    special = frozenset(r"\|()[]{}?*+.^$")
    result = []
    for char in text:
        if char == " " and literal_spaces:
            result.append(" ")
        elif char in special or (not special_only and not char.isalnum() and char != "_"):
            result.append("\\" + char)
        elif char.isspace():
            result.append("\\" + char)
        else:
            result.append(char)
    escaped = "".join(result)
    return escaped.encode("latin1") if is_bytes else escaped


def purge():
    _compile_cached.cache_clear()
    return None


template = compile

__all__ = [
    "A", "ASCII", "B", "BESTMATCH", "D", "DEBUG", "DEFAULT_VERSION", "E",
    "ENHANCEMATCH", "F", "FULLCASE", "I", "IGNORECASE", "L", "LOCALE", "M",
    "MULTILINE", "P", "POSIX", "R", "REVERSE", "S", "DOTALL", "U", "UNICODE",
    "V0", "VERSION0", "V1", "VERSION1", "W", "WORD", "X", "VERBOSE",
    "RegexFlag", "Pattern", "Match", "Scanner", "compile", "escape", "findall",
    "finditer", "fullmatch", "match", "purge", "search", "split", "splititer",
    "sub", "subn", "template", "error",
]
