"""Benchmarks against the upstream ``regex`` package on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojoregex  # noqa: E402
import regex  # noqa: E402


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


CASES = []


def benchmark(name, ours, upstream, repeat=5):
    CASES.append((name, ours, upstream, repeat))


haystack = "a" * 1_000_000 + "needle"
benchmark(
    "literal search, match at 1M",
    lambda: mojoregex.search("needle", haystack),
    lambda: regex.search("needle", haystack),
)

identifier = "alpha_" + "x9" * 250_000
ours_identifier = mojoregex.compile(r"[A-Za-z_][A-Za-z_0-9]+")
regex_identifier = regex.compile(r"[A-Za-z_][A-Za-z_0-9]+")
benchmark(
    "identifier fullmatch, 500k",
    lambda: ours_identifier.fullmatch(identifier),
    lambda: regex_identifier.fullmatch(identifier),
    3,
)

lines = ("INFO 200: healthy\n" * 40_000) + "ERROR 503: unavailable\n"
ours_log = mojoregex.compile(r"ERROR\s+\d+:\s+[A-Za-z]+")
regex_log = regex.compile(r"ERROR\s+\d+:\s+[A-Za-z]+")
benchmark(
    "structured search, 720k",
    lambda: ours_log.search(lines),
    lambda: regex_log.search(lines),
)

left = "a" * 100_000
backref_text = left + "-" + left
ours_backref = mojoregex.compile(r"(a+)-\1")
regex_backref = regex.compile(r"(a+)-\1")
benchmark(
    "capturing backreference, 200k",
    lambda: ours_backref.fullmatch(backref_text),
    lambda: regex_backref.fullmatch(backref_text),
    3,
)

words = " ".join(f"word{i % 100}" for i in range(10_000))
ours_words = mojoregex.compile(r"\w+")
regex_words = regex.compile(r"\w+")
benchmark(
    "findall, 10k words",
    lambda: ours_words.findall(words),
    lambda: regex_words.findall(words),
    3,
)


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| case | mojoregex | regex | upstream / Mojo | result |")
    print("|---|---:|---:|---:|:---|")
    for name, ours, upstream, repeat in CASES:
        ours()
        upstream()
        mojo_seconds = best_time(ours, repeat)
        upstream_seconds = best_time(upstream, repeat)
        ratio = upstream_seconds / mojo_seconds
        result = "Mojo faster" if ratio > 1 else "Mojo slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.3f} ms | "
            f"{upstream_seconds * 1000:.3f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
