# mojo-regex

`mojo-regex` is an open-source ordered-backtracking regular-expression engine
implemented in Mojo, with a Python API named `mojoregex`. It targets the
compute-heavy matching loop rather than wrapping Python's `re` or `regex`
matcher: Python parses a pattern into bytecode and Mojo executes that bytecode,
including backtracking, capture rollback, and backreference comparison.

The API follows the third-party Python
[`regex`](https://pypi.org/project/regex/) package for the covered subset.
Use `import mojoregex as regex` when substituting it in covered code.

## Coverage

Implemented:

- `compile`, `search`, `match`, `fullmatch`, `findall`, `finditer`,
  `split`, `splititer`, `sub`, `subn`, `escape`, and `purge`
- `Pattern`, `Match`, and stateful `Scanner` objects, including named groups,
  match spans, dictionaries, replacement expansion, and `lastindex` /
  `lastgroup`
- literals, `.`, ordered alternation, capturing and non-capturing groups
- greedy and lazy `*`, `+`, `?`, and `{m,n}` repetition
- character sets, negated sets, ranges, and ASCII `\d`, `\w`, `\s` families
- `^`, `$`, `\A`, `\Z`, `\b`, and `\B`
- numeric and named backreferences
- `IGNORECASE` for ASCII, `MULTILINE`, `DOTALL`, `VERBOSE`, bytes patterns,
  Unicode literals, and the `overlapped` iterator option

The parity suite compares all of those behaviors directly with the installed
upstream `regex` package.

Not implemented:

- lookahead, lookbehind, conditionals, atomic or possessive groups
- recursion, subroutines, branch reset, capture histories, callouts
- fuzzy matching, partial matches, timeouts, reverse, POSIX, best-match, or
  enhanced-match modes
- Unicode property escapes, grapheme clusters, Unicode word/digit/space
  classification, or non-ASCII case folding
- nested set algebra and scoped inline flags

Built-in character classes intentionally use ASCII definitions. Non-ASCII
literals and `.` retain Python character offsets, but code that needs full
Unicode classification should keep using upstream `regex`. Unsupported engine
modes fail explicitly instead of silently selecting different semantics.

## Install and run

The repository pins the Mojo nightly dialect it was developed against.
Installation is currently from a source checkout with Pixi; no Python wheel is
published.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-regex.so`. Run a script inside the
Pixi environment so `python/` is on `PYTHONPATH`.

```python
import mojoregex as regex

record = regex.compile(r"(?P<key>[A-Za-z_]\w*)=(?P<value>\d+)")
match = record.search("mode=fast retries=12")

assert match is not None
assert match.groupdict() == {"key": "retries", "value": "12"}
assert record.sub(r"\g<key>:\g<value>", "retries=12") == "retries:12"
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14. Times are the best of five runs
(three for the larger repetition cases). The ratio is upstream time divided
by Mojo time, so values below 1 mean Mojo is slower.

| case | mojoregex | regex | upstream / Mojo | result |
|---|---:|---:|---:|:---|
| literal search, match at 1M | 0.404 ms | 0.684 ms | 1.69x | Mojo faster |
| identifier fullmatch, 500k | 0.902 ms | 8.946 ms | 9.92x | Mojo faster |
| structured search, 720k | 0.282 ms | 0.644 ms | 2.28x | Mojo faster |
| capturing backreference, 200k | 0.087 ms | 0.538 ms | 6.18x | Mojo faster |
| findall, 10k words | 3.637 ms | 3.852 ms | 1.06x | Mojo faster |

Prefix scans, simple greedy repeats, character-class repeats, and backreference
comparisons use SIMD with scalar remainder loops. Prefix scans above four
million characters use independent parallel chunks; smaller scans stay serial.
Repeated module-level calls reuse a bounded compiled-pattern cache, and
`findall` returns matches in native batches instead of crossing the FFI once per
result.

No GPU path is included. Regex scanning is comparison- and memory-bound, with
well under the roughly two arithmetic operations per byte needed to offset GPU
transfer and launch costs. The `max` dependency supplies CPU `parallelize`; it
is not used for GPU execution.

## How it works

The Python parser lowers the pattern into fixed-width instructions
`(opcode, a, b)` stored as contiguous signed 64-bit words. ASCII strings and
bytes use their native one-byte representation; non-ASCII input uses a
contiguous `UInt32` code-point buffer. Match and capture offsets therefore stay
Python character indices. Character-set ranges are a second contiguous `Int64`
buffer.

The ctypes boundary passes buffers as integer addresses after Python verifies
their exact dtype, contiguous layout, and non-null address. ASCII strings use
CPython's UTF-8 storage directly, bytes use a NumPy `frombuffer` view, and the
inputs remain strongly referenced for the entire native call. The exported Mojo
functions
validate lengths, bounds, capacities, modes, and addresses before reconstructing
them with
`UnsafePointer[..., AnyOrigin[mut=True]]`, as required by this Mojo dialect.
The VM follows the first branch of each split and saves the second branch in a
caller-owned stack. Capture writes use a rollback log, so failed alternatives
restore the exact prior capture state without copying every capture at every
branch. Python owns all code, input, stack, log, and output memory; the native
call allocates nothing and returns either a match, no match, or a retry signal
if a scratch arena must grow. Python creates uninitialized scratch arenas with
NumPy; those buffers cross the FFI boundary zero-copy.

`finditer`, substitution, and splitting prepare the input view and scratch
arena once and reuse them for the whole scan. `findall` additionally
collects up to 16,384 results per native call.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds `/tmp/mojo-bench.lock` so concurrent jobs on the
machine do not contaminate measurements.
