#!/usr/bin/env python3
"""Small deterministic stand-in used only by the shell-driver tests."""

import os
import struct
import sys
from pathlib import Path


def value_after(flag, default):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


key_type = value_after("-ctk", "f16")
value_type = value_after("-ctv", "f16")

if "--help" in sys.argv:
    print("--cache-type-k TYPE allowed values: f16, turbo4")
elif "-v" in sys.argv:
    key_mib = 1.0 if key_type == "f16" else 0.25
    value_mib = 1.0 if value_type == "f16" else 0.25
    print(
        f"llama_kv_cache: size = {key_mib + value_mib:.2f} MiB "
        f"(2048 cells, 1 layers), K ({key_type}): {key_mib:.2f} MiB, "
        f"V ({value_type}): {value_mib:.2f} MiB"
    )
else:
    anchor = key_type == "f16" and value_type == "f16"
    value = 0.0 if anchor else 0.01
    same = 100.0 if anchor else 99.0
    print("Mean PPL(Q)                   :   5.000000 ±   0.010000")
    print(f"Mean ln(PPL(Q)/PPL(base))     : {value:10.6f} ±   0.000100")
    print(f"Mean    KLD: {value:10.6f} ± {0.0 if anchor else 0.000001:10.6f}")
    print(f"Maximum KLD: {value:10.6f}")
    print(f"99.9%   KLD: {value:10.6f}")
    print(f"Median  KLD: {value:10.6f}")
    print(f"RMS Δp    : {value:6.3f} ± 0.001 %")
    print(f"Same top p: {same:6.3f} ± 0.010 %")
    dump_path = os.environ.get("TURBO_KLD_DUMP")
    if anchor and dump_path:
        Path(dump_path).write_bytes(struct.pack("<ii2f", 2, 1, 0.0, 0.0))
        Path(dump_path + ".meta").write_text(
            "format_version=2\nn_pos=2\nn_chunk=1\nscore_last_k=0\n"
        )
