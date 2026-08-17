#!/usr/bin/env python3
"""Build layer-pricing TSVs from paired TURBO_KLD_DUMP files.

Expected names under --dumps (the prefix is configurable):

  pxr_TAG_base_fp16.kld
  pxr_TAG_base_t8.kld
  pxr_TAG_t8-t4_l12_k.kld
  pxr_TAG_t8-t4_l12_v.kld

Each cell is paired with base_<high tier>. The output schemas are consumed
directly by build_degrade_order.py.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


TRANSITIONS = ("fp16-t8", "t8-t4", "t4-t3", "t3-t2", "t2-t1")
SIDES = ("k", "v")
STAT_COLUMNS = {
    "median": "excess_median",
    "trim1": "excess_trim0.01",
    "frac": "frac_gt_0.001",
    "mean": "excess_mean",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_dump(path, expected_last_k=None, allow_legacy=False):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: truncated dump header")
    n_pos, n_chunk = struct.unpack_from("<ii", data)
    if n_pos <= 0 or n_chunk <= 0:
        raise ValueError(f"{path}: invalid shape {n_chunk}x{n_pos}")
    expected = 8 + 4 * n_pos * n_chunk
    if len(data) != expected:
        raise ValueError(
            f"{path}: size {len(data)} does not match {n_chunk}x{n_pos} dump ({expected})"
        )
    metadata_path = Path(str(path) + ".meta")
    if metadata_path.exists():
        metadata = dict(
            line.split("=", 1) for line in metadata_path.read_text().splitlines() if "=" in line
        )
        if metadata.get("format_version") != "2":
            raise ValueError(f"{metadata_path}: unsupported dump metadata version")
        if int(metadata.get("n_pos", -1)) != n_pos or int(metadata.get("n_chunk", -1)) != n_chunk:
            raise ValueError(f"{metadata_path}: shape does not match dump")
        recorded_last_k = int(metadata.get("score_last_k", -1))
        requested_last_k = expected_last_k or 0
        if recorded_last_k != requested_last_k:
            raise ValueError(
                f"{path}: dump score_last_k={recorded_last_k}, requested {requested_last_k}"
            )
    elif expected_last_k is not None and not allow_legacy:
        raise ValueError(
            f"{path}: no metadata proves this is a last-K dump; "
            "use --allow-legacy-dumps only after verification"
        )
    return np.frombuffer(data, dtype="<f4", offset=8).reshape(n_chunk, n_pos).astype(np.float64)


def trimmed_mean(values, fraction=0.01):
    flat = np.sort(np.asarray(values).reshape(-1))
    trim = int(flat.size * fraction)
    if trim == 0:
        return float(flat.mean())
    if 2 * trim >= flat.size:
        raise ValueError("not enough samples for requested trim")
    return float(flat[trim:-trim].mean())


def summarize(excess, threshold):
    return {
        "median": float(np.median(excess)),
        "trim1": trimmed_mean(excess),
        "frac": float((excess > threshold).mean()),
        "mean": float(excess.mean()),
    }


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left, right):
    if len(left) < 3 or len(left) != len(right):
        return float("nan")
    rank_left = average_ranks(left)
    rank_right = average_ranks(right)
    if np.ptp(rank_left) == 0 or np.ptp(rank_right) == 0:
        return float("nan")
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def chunk_groups(n_chunk):
    if n_chunk < 2:
        raise ValueError("at least two chunks are required for split-half reliability")
    return {
        "all": np.arange(n_chunk),
        "even": np.arange(0, n_chunk, 2),
        "odd": np.arange(1, n_chunk, 2),
        "firsthalf": np.arange(0, n_chunk // 2),
        "secondhalf": np.arange(n_chunk // 2, n_chunk),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--campaign-id",
        required=True,
        help="immutable build/model/inference-settings identifier embedded in the outputs",
    )
    parser.add_argument("--prefix", default="pxr_")
    parser.add_argument("--out-cells", required=True, type=Path)
    parser.add_argument("--out-reliability", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument(
        "--last-k",
        type=int,
        help=(
            "score only the final K positions; required when dumps were zero-filled by "
            "TURBO_SCORE_LAST_K"
        ),
    )
    parser.add_argument(
        "--allow-legacy-dumps",
        action="store_true",
        help="accept last-K dumps without metadata after manual verification",
    )
    args = parser.parse_args()

    if args.last_k is not None and args.last_k <= 0:
        parser.error("--last-k must be positive")
    if not math.isfinite(args.threshold):
        parser.error("--threshold must be finite")

    escaped = re.escape(args.prefix + args.tag + "_")
    cell_pattern = re.compile(
        rf"^{escaped}(fp16-t8|t8-t4|t4-t3|t3-t2|t2-t1)_l([0-9]+)_([kv])\.kld$"
    )
    cells = {}
    for path in sorted(args.dumps.iterdir()):
        match = cell_pattern.match(path.name)
        if not match:
            continue
        transition, layer_text, side = match.groups()
        key = (transition, int(layer_text), side)
        if key in cells:
            raise SystemExit(f"duplicate cell for {key}: {cells[key]} and {path}")
        cells[key] = path
    if not cells:
        raise SystemExit(
            f"no cells matched {args.prefix}{args.tag}_<transition>_l<LAYER>_<k|v>.kld"
        )

    layers = sorted({layer for _, layer, _ in cells})
    expected_keys = {
        (transition, layer, side)
        for transition in TRANSITIONS
        for layer in layers
        for side in SIDES
    }
    missing = sorted(expected_keys - cells.keys())
    if missing:
        preview = ", ".join(
            f"{transition}/l{layer}/{side}" for transition, layer, side in missing[:8]
        )
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise SystemExit(f"incomplete dump matrix: {len(missing)} missing cells: {preview}{more}")

    anchors = {}
    for transition in TRANSITIONS:
        high_tier = transition.split("-", 1)[0]
        path = args.dumps / f"{args.prefix}{args.tag}_base_{high_tier}.kld"
        if not path.is_file():
            raise SystemExit(f"missing anchor for {transition}: {path}")
        anchors[transition] = read_dump(path, args.last_k, args.allow_legacy_dumps)

    if np.any(anchors["fp16-t8"] != 0.0):
        raise SystemExit("fp16 anchor is not exactly zero; the campaign is invalid")

    grouped = {}
    panel_rows = []
    for transition, layer, side in sorted(cells):
        anchor = anchors[transition]
        cell = read_dump(cells[(transition, layer, side)], args.last_k, args.allow_legacy_dumps)
        if cell.shape != anchor.shape:
            raise SystemExit(
                f"shape mismatch for {cells[(transition, layer, side)]}: "
                f"{cell.shape} vs anchor {anchor.shape}"
            )
        if args.last_k is not None:
            if args.last_k > cell.shape[1]:
                raise SystemExit(f"--last-k {args.last_k} exceeds dump width {cell.shape[1]}")
            cell = cell[:, -args.last_k :]
            anchor = anchor[:, -args.last_k :]
        excess = cell - anchor
        groups = chunk_groups(excess.shape[0])
        grouped[(transition, layer, side)] = {
            group: summarize(excess[index], args.threshold) for group, index in groups.items()
        }
        stats = grouped[(transition, layer, side)]["all"]
        panel_rows.append(
            {
                "tag": args.tag,
                "campaign_id": args.campaign_id,
                "group": "all",
                "side": side,
                "layer": layer,
                "transition": transition,
                **{column: stats[stat] for stat, column in STAT_COLUMNS.items()},
            }
        )

    args.out_cells.parent.mkdir(parents=True, exist_ok=True)
    with args.out_cells.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "tag",
                "campaign_id",
                "group",
                "side",
                "layer",
                "transition",
                *STAT_COLUMNS.values(),
            ],
        )
        writer.writeheader()
        writer.writerows(panel_rows)

    reliability_rows = []
    for transition in TRANSITIONS:
        for side in SIDES:
            for stat in STAT_COLUMNS:
                even = [grouped[(transition, layer, side)]["even"][stat] for layer in layers]
                odd = [grouped[(transition, layer, side)]["odd"][stat] for layer in layers]
                first = [grouped[(transition, layer, side)]["firsthalf"][stat] for layer in layers]
                second = [
                    grouped[(transition, layer, side)]["secondhalf"][stat] for layer in layers
                ]
                reliability_rows.append(
                    {
                        "tag": args.tag,
                        "campaign_id": args.campaign_id,
                        "transition": transition,
                        "side": side,
                        "stat": stat,
                        "rho_even_odd": spearman(even, odd),
                        "rho_half": spearman(first, second),
                    }
                )

    args.out_reliability.parent.mkdir(parents=True, exist_ok=True)
    with args.out_reliability.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "tag",
                "campaign_id",
                "transition",
                "side",
                "stat",
                "rho_even_odd",
                "rho_half",
            ],
        )
        writer.writeheader()
        writer.writerows(reliability_rows)

    anchor_paths = {
        args.dumps / f"{args.prefix}{args.tag}_base_{transition.split('-', 1)[0]}.kld"
        for transition in TRANSITIONS
    }
    input_paths = sorted(set(cells.values()) | anchor_paths)
    manifest = {
        "schema_version": 1,
        "tag": args.tag,
        "campaign_id": args.campaign_id,
        "threshold": args.threshold,
        "last_k": args.last_k,
        "inputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "metadata_sha256": sha256_file(Path(str(path) + ".meta"))
                if Path(str(path) + ".meta").exists()
                else None,
            }
            for path in input_paths
        ],
        "outputs": {
            "cells_sha256": sha256_file(args.out_cells),
            "reliability_sha256": sha256_file(args.out_reliability),
        },
    }
    Path(str(args.out_cells) + ".meta.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"wrote {len(panel_rows)} cells to {args.out_cells}")
    print(f"wrote {len(reliability_rows)} reliability rows to {args.out_reliability}")


if __name__ == "__main__":
    main()
