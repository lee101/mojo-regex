from __future__ import annotations

from dataclasses import dataclass


OP_MATCH = 0
OP_CHAR = 1
OP_ANY = 2
OP_CLASS = 3
OP_JMP = 4
OP_SPLIT = 5
OP_SAVE = 6
OP_BOL = 7
OP_EOL = 8
OP_BACKREF = 9
OP_BOUNDARY = 10
OP_NOT_BOUNDARY = 11
OP_BEGIN = 12
OP_END = 13
OP_MARK = 14
OP_REPEAT_CLASS = 15
OP_REPEAT_BOUNDS = 16
OP_REPEAT_CHAR = 17

IGNORECASE = 2
VERBOSE = 64


class CompileError(Exception):
    pass


@dataclass
class Node:
    kind: str
    value: object = None

    def nullable(self) -> bool:
        if self.kind in {"empty", "bol", "eol", "begin", "end", "boundary", "nboundary"}:
            return True
        if self.kind in {"char", "any", "class", "backref"}:
            return False
        if self.kind in {"group"}:
            return self.value[1].nullable()
        if self.kind == "seq":
            return all(node.nullable() for node in self.value)
        if self.kind == "alt":
            return any(node.nullable() for node in self.value)
        if self.kind == "repeat":
            node, minimum, _, _ = self.value
            return minimum == 0 or node.nullable()
        raise AssertionError(self.kind)


_DIGIT = [(48, 57)]
_WORD = [(48, 57), (65, 90), (95, 95), (97, 122)]
_SPACE = [(9, 13), (32, 32)]
_UNIVERSE = [(0, 0x10FFFF)]


def _complement(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = _merge(ranges)
    result = []
    cursor = 0
    for lo, hi in merged:
        if cursor < lo:
            result.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
    if cursor <= 0x10FFFF:
        result.append((cursor, 0x10FFFF))
    return result


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if result and lo <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


def _case_expand(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = list(ranges)
    for lo, hi in ranges:
        a, b = max(lo, 65), min(hi, 90)
        if a <= b:
            result.append((a + 32, b + 32))
        a, b = max(lo, 97), min(hi, 122)
        if a <= b:
            result.append((a - 32, b - 32))
    return _merge(result)


class Parser:
    def __init__(self, pattern: str, flags: int):
        self.pattern = pattern
        self.flags = flags
        self.i = 0
        self.groups = 0
        self.groupindex: dict[str, int] = {}

    def error(self, message: str) -> CompileError:
        return CompileError(f"{message} at position {self.i}")

    def _skip_verbose(self) -> None:
        if not self.flags & VERBOSE:
            return
        while self.i < len(self.pattern):
            if self.pattern[self.i] in " \t\n\r\f\v":
                self.i += 1
            elif self.pattern[self.i] == "#":
                newline = self.pattern.find("\n", self.i)
                self.i = len(self.pattern) if newline < 0 else newline + 1
            else:
                break

    def peek(self) -> str | None:
        self._skip_verbose()
        return self.pattern[self.i] if self.i < len(self.pattern) else None

    def take(self) -> str:
        if self.i >= len(self.pattern):
            raise self.error("unexpected end of pattern")
        char = self.pattern[self.i]
        self.i += 1
        return char

    def parse(self) -> Node:
        node = self.parse_alt()
        self._skip_verbose()
        if self.i != len(self.pattern):
            raise self.error("unbalanced parenthesis")
        return node

    def parse_alt(self) -> Node:
        alternatives = [self.parse_seq()]
        while self.peek() == "|":
            self.take()
            alternatives.append(self.parse_seq())
        return alternatives[0] if len(alternatives) == 1 else Node("alt", alternatives)

    def parse_seq(self) -> Node:
        nodes = []
        while self.peek() not in {None, "|", ")"}:
            nodes.append(self.parse_piece())
        if not nodes:
            return Node("empty")
        return nodes[0] if len(nodes) == 1 else Node("seq", nodes)

    def parse_piece(self) -> Node:
        node = self.parse_atom()
        char = self.peek()
        minimum = maximum = None
        if char == "*":
            self.take()
            minimum, maximum = 0, None
        elif char == "+":
            self.take()
            minimum, maximum = 1, None
        elif char == "?":
            self.take()
            minimum, maximum = 0, 1
        elif char == "{":
            saved = self.i
            self.take()
            digits = self._raw_digits()
            if not digits:
                self.i = saved
                return node
            minimum = int(digits)
            maximum = minimum
            if self.i < len(self.pattern) and self.pattern[self.i] == ",":
                self.i += 1
                digits = self._raw_digits()
                maximum = int(digits) if digits else None
            if self.i >= len(self.pattern) or self.pattern[self.i] != "}":
                self.i = saved
                return node
            self.i += 1
            if maximum is not None and maximum < minimum:
                raise self.error("bad repeat range")
            if minimum > 10000 or (maximum is not None and maximum > 10000):
                raise self.error("repeat count is too large")
        if minimum is None:
            return node
        greedy = True
        if self.peek() == "?":
            self.take()
            greedy = False
        elif self.peek() == "+":
            raise self.error("possessive quantifiers are not supported")
        if maximum is None and node.nullable():
            raise self.error("unbounded repetition of an empty expression is not supported")
        return Node("repeat", (node, minimum, maximum, greedy))

    def _raw_digits(self) -> str:
        start = self.i
        while self.i < len(self.pattern) and self.pattern[self.i].isdigit():
            self.i += 1
        return self.pattern[start:self.i]

    def parse_atom(self) -> Node:
        char = self.take()
        if char in "*+?":
            raise self.error("nothing to repeat")
        if char == ".":
            return Node("any")
        if char == "^":
            return Node("bol")
        if char == "$":
            return Node("eol")
        if char == "[":
            return self.parse_class()
        if char == "\\":
            return self.parse_escape(False)
        if char != "(":
            return Node("char", ord(char))

        capturing = True
        name = None
        if self.i < len(self.pattern) and self.pattern[self.i] == "?":
            self.i += 1
            if self.pattern.startswith(":", self.i):
                self.i += 1
                capturing = False
            elif self.pattern.startswith("P<", self.i):
                self.i += 2
                close = self.pattern.find(">", self.i)
                if close < 0:
                    raise self.error("missing group name terminator")
                name = self.pattern[self.i:close]
                self.i = close + 1
            elif self.pattern.startswith("<", self.i) and not self.pattern.startswith(("<=", "<!"), self.i):
                self.i += 1
                close = self.pattern.find(">", self.i)
                if close < 0:
                    raise self.error("missing group name terminator")
                name = self.pattern[self.i:close]
                self.i = close + 1
            elif self.pattern.startswith("P=", self.i):
                self.i += 2
                close = self.pattern.find(")", self.i)
                if close < 0:
                    raise self.error("missing named backreference terminator")
                ref = self.pattern[self.i:close]
                self.i = close + 1
                if ref not in self.groupindex:
                    raise self.error(f"unknown group name {ref!r}")
                return Node("backref", self.groupindex[ref])
            elif self.pattern.startswith("#", self.i):
                close = self.pattern.find(")", self.i + 1)
                if close < 0:
                    raise self.error("unterminated comment")
                self.i = close + 1
                return Node("empty")
            else:
                raise self.error("unsupported group extension")

        group = None
        if capturing:
            self.groups += 1
            group = self.groups
            if name is not None:
                if not name.isidentifier() or name in self.groupindex:
                    raise self.error("invalid or duplicate group name")
                self.groupindex[name] = group
        child = self.parse_alt()
        if self.peek() != ")":
            raise self.error("missing closing parenthesis")
        self.take()
        return Node("group", (group, child, name)) if capturing else child

    def parse_escape(self, in_class: bool) -> Node:
        if self.i >= len(self.pattern):
            raise self.error("bad escape at end of pattern")
        char = self.take()
        classes = {
            "d": _DIGIT,
            "w": _WORD,
            "s": _SPACE,
            "D": _complement(_DIGIT),
            "W": _complement(_WORD),
            "S": _complement(_SPACE),
        }
        if char in classes:
            ranges = classes[char]
            if self.flags & IGNORECASE:
                ranges = _case_expand(ranges)
            return Node("class", (ranges, False))
        escapes = {
            "n": 10, "r": 13, "t": 9, "f": 12, "v": 11, "a": 7,
            "\\": 92,
        }
        if char in escapes:
            return Node("char", escapes[char])
        if char == "b":
            return Node("char", 8) if in_class else Node("boundary")
        if char == "B" and not in_class:
            return Node("nboundary")
        if char == "A" and not in_class:
            return Node("begin")
        if char in {"Z", "z"} and not in_class:
            return Node("end")
        if char == "x":
            return Node("char", self._hex(2))
        if char == "u":
            return Node("char", self._hex(4))
        if char == "U":
            return Node("char", self._hex(8))
        if char.isdigit() and not in_class:
            digits = char + self._raw_digits()
            group = int(digits)
            if group == 0 or group > self.groups:
                raise self.error("invalid group reference")
            return Node("backref", group)
        if char.isalnum():
            raise self.error(f"unsupported escape \\{char}")
        return Node("char", ord(char))

    def _hex(self, count: int) -> int:
        chunk = self.pattern[self.i:self.i + count]
        if len(chunk) != count or any(c not in "0123456789abcdefABCDEF" for c in chunk):
            raise self.error("incomplete hex escape")
        self.i += count
        value = int(chunk, 16)
        if value > 0x10FFFF:
            raise self.error("invalid Unicode code point")
        return value

    def parse_class(self) -> Node:
        negated = self.i < len(self.pattern) and self.pattern[self.i] == "^"
        if negated:
            self.i += 1
        ranges: list[tuple[int, int]] = []
        first = True
        while self.i < len(self.pattern):
            if self.pattern[self.i] == "]" and not first:
                self.i += 1
                if self.flags & IGNORECASE:
                    ranges = _case_expand(ranges)
                return Node("class", (_merge(ranges), negated))
            first = False
            left = self._class_item()
            if (
                len(left) == 1
                and left[0][0] == left[0][1]
                and self.i < len(self.pattern)
                and self.pattern[self.i] == "-"
                and self.i + 1 < len(self.pattern)
                and self.pattern[self.i + 1] != "]"
            ):
                self.i += 1
                right = self._class_item()
                if len(right) != 1 or right[0][0] != right[0][1]:
                    raise self.error("bad character range")
                lo, hi = left[0][0], right[0][0]
                if hi < lo:
                    raise self.error("bad character range")
                ranges.append((lo, hi))
            else:
                ranges.extend(left)
        raise self.error("unterminated character set")

    def _class_item(self) -> list[tuple[int, int]]:
        char = self.take()
        if char == "\\":
            node = self.parse_escape(True)
            if node.kind == "class":
                return node.value[0]
            return [(node.value, node.value)]
        return [(ord(char), ord(char))]


class Compiler:
    def __init__(self):
        self.code: list[list[int]] = []
        self.ranges: list[tuple[int, int]] = []

    def emit(self, op: int, a: int = 0, b: int = 0) -> int:
        self.code.append([op, a, b])
        return len(self.code) - 1

    def compile(self, node: Node) -> None:
        kind = node.kind
        if kind == "empty":
            return
        if kind == "char":
            self.emit(OP_CHAR, node.value)
        elif kind == "any":
            self.emit(OP_ANY)
        elif kind == "class":
            ranges, negated = node.value
            offset = len(self.ranges)
            self.ranges.extend(ranges)
            count = len(ranges)
            self.emit(OP_CLASS, offset, -count - 1 if negated else count)
        elif kind == "bol":
            self.emit(OP_BOL)
        elif kind == "eol":
            self.emit(OP_EOL)
        elif kind == "begin":
            self.emit(OP_BEGIN)
        elif kind == "end":
            self.emit(OP_END)
        elif kind == "boundary":
            self.emit(OP_BOUNDARY)
        elif kind == "nboundary":
            self.emit(OP_NOT_BOUNDARY)
        elif kind == "backref":
            self.emit(OP_BACKREF, node.value)
        elif kind == "seq":
            for child in node.value:
                self.compile(child)
        elif kind == "group":
            group, child, name = node.value
            self.emit(OP_SAVE, group * 2)
            self.compile(child)
            self.emit(OP_SAVE, group * 2 + 1)
            self.emit(OP_MARK, group, int(name is not None))
        elif kind == "alt":
            jumps = []
            alternatives = node.value
            for alternative in alternatives[:-1]:
                split = self.emit(OP_SPLIT)
                first = len(self.code)
                self.compile(alternative)
                jumps.append(self.emit(OP_JMP))
                second = len(self.code)
                self.code[split][1:] = [first, second]
            self.compile(alternatives[-1])
            end = len(self.code)
            for jump in jumps:
                self.code[jump][1] = end
        elif kind == "repeat":
            child, minimum, maximum, greedy = node.value
            if child.kind == "char" and maximum is None and greedy:
                self.emit(OP_REPEAT_CHAR, child.value)
                self.emit(OP_REPEAT_BOUNDS, minimum)
                return
            if child.kind == "class" and maximum is None and greedy:
                ranges, negated = child.value
                offset = len(self.ranges)
                self.ranges.extend(ranges)
                count = len(ranges)
                self.emit(
                    OP_REPEAT_CLASS,
                    offset,
                    -count - 1 if negated else count,
                )
                self.emit(OP_REPEAT_BOUNDS, minimum)
                return
            for _ in range(minimum):
                self.compile(child)
            if maximum is None:
                split = self.emit(OP_SPLIT)
                body = len(self.code)
                self.compile(child)
                self.emit(OP_JMP, split)
                end = len(self.code)
                self.code[split][1:] = [body, end] if greedy else [end, body]
            else:
                for _ in range(maximum - minimum):
                    split = self.emit(OP_SPLIT)
                    body = len(self.code)
                    self.compile(child)
                    end = len(self.code)
                    self.code[split][1:] = [body, end] if greedy else [end, body]
        else:
            raise AssertionError(kind)


def _prefix(node: Node) -> int:
    if node.kind == "char":
        return node.value
    if node.kind == "group":
        return _prefix(node.value[1])
    if node.kind == "seq":
        for child in node.value:
            prefix = _prefix(child)
            if prefix >= 0:
                return prefix
            if not _zero_width(child):
                return -1
        return -1
    if node.kind == "alt":
        prefixes = {_prefix(child) for child in node.value}
        return prefixes.pop() if len(prefixes) == 1 else -1
    if node.kind == "repeat" and node.value[1] > 0 and not node.value[0].nullable():
        return _prefix(node.value[0])
    return -1


def _zero_width(node: Node) -> bool:
    if node.kind in {"empty", "bol", "eol", "begin", "end", "boundary", "nboundary"}:
        return True
    if node.kind == "seq":
        return all(_zero_width(child) for child in node.value)
    if node.kind == "alt":
        return all(_zero_width(child) for child in node.value)
    if node.kind == "group":
        return _zero_width(node.value[1])
    if node.kind == "repeat":
        return node.value[2] == 0 or _zero_width(node.value[0])
    return False


def compile_pattern(pattern: str, flags: int):
    parser = Parser(pattern, flags)
    tree = parser.parse()
    compiler = Compiler()
    compiler.emit(OP_SAVE, 0)
    compiler.compile(tree)
    compiler.emit(OP_SAVE, 1)
    compiler.emit(OP_MATCH)
    flat_code = [value for instruction in compiler.code for value in instruction]
    flat_ranges = [value for pair in compiler.ranges for value in pair]
    return (
        flat_code,
        flat_ranges,
        parser.groups,
        parser.groupindex,
        parser.flags,
        _prefix(tree),
    )
