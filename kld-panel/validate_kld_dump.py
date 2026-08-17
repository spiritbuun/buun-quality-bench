#!/usr/bin/env python3
"""Validate a TURBO_KLD_DUMP and optionally require every value to be exactly zero."""

import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--exact-zero", action="store_true")
    args = parser.parse_args()

    data = args.dump.read_bytes()
    if len(data) < 8:
        parser.error("truncated dump header")
    n_pos, n_chunk = struct.unpack_from("<ii", data)
    if n_pos <= 0 or n_chunk <= 0:
        parser.error(f"invalid shape {n_chunk}x{n_pos}")
    expected = 8 + 4 * n_pos * n_chunk
    if len(data) != expected:
        parser.error(f"file has {len(data)} bytes; expected {expected}")
    values = np.frombuffer(data, dtype="<f4", offset=8)
    if not np.isfinite(values).all():
        parser.error("dump contains non-finite values")
    if args.exact_zero and np.any(values != 0.0):
        nonzero = values[values != 0.0]
        parser.error(
            f"dump contains {nonzero.size} nonzero values; maximum absolute value "
            f"is {float(np.max(np.abs(nonzero))):.9g}"
        )
    print(f"valid dump: chunks={n_chunk} positions={n_pos} values={values.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
