from __future__ import annotations

import ctypes
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "dist" / "libmojo-regex.so"
I = ctypes.c_int64

_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise RuntimeError("Mojo library is missing; run `pixi run build`")
        _lib = ctypes.CDLL(os.fspath(LIB))
        fn = _lib.mrx_exec
        fn.argtypes = [I] * 21
        fn.restype = I
        find_all = _lib.mrx_find_all
        find_all.argtypes = [I] * 25
        find_all.restype = I
    return _lib
