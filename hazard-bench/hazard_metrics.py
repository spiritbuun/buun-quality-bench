#!/usr/bin/env python3
"""Offline hazard metrics from a llama-perplexity base and KLD dumps.

The hazard definition matches frontier-hazard.cpp:

    R = KL / (0.5 * (p_top1 - p_top2)**2 + 1e-6)

The base stores quantized log probabilities, from which p_top1 and p_top2 are
reconstructed. Cached margins are bound to the base path, size and mtime, with
an optional full SHA-256 verification.
"""

import argparse
import hashlib
import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


MARGIN_FORMAT_VERSION = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def base_identity(path, include_hash):
    resolved = str(Path(path).resolve())
    stat = os.stat(resolved)
    identity = {
        "path": resolved,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        identity["sha256"] = sha256_file(resolved)
    return identity


def extract_probability_margins(base_path, out_path):
    identity_before = base_identity(base_path, include_hash=False)
    with open(base_path, "rb") as handle:
        if handle.read(8) != b"_logits_":
            raise SystemExit("not a logits base file")
        (n_ctx,) = struct.unpack("<I", handle.read(4))
        n_vocab, n_chunk = struct.unpack("<ii", handle.read(8))
        if n_ctx < 2 or n_vocab < 2 or n_chunk < 1:
            raise SystemExit(
                f"invalid base dimensions: ctx={n_ctx} vocab={n_vocab} chunks={n_chunk}"
            )
        n_scored = n_ctx - 1 - n_ctx // 2
        row_width = 2 * ((n_vocab + 1) // 2) + 4
        handle.seek(8 + 4 + 8 + 4 * n_ctx * n_chunk)
        print(
            f"base: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk} "
            f"n_scored={n_scored} row_width={row_width}"
        )
        margins = np.zeros((n_chunk, n_scored), dtype=np.float32)
        slice_rows = 1024
        for chunk in range(n_chunk):
            done = 0
            while done < n_scored:
                rows = min(slice_rows, n_scored - done)
                raw = np.fromfile(handle, dtype=np.uint16, count=rows * row_width)
                if raw.size != rows * row_width:
                    raise SystemExit(f"truncated base in chunk {chunk} at scored row {done}")
                raw = raw.reshape(rows, row_width)
                header = raw[:, :4].copy().view(np.float32).reshape(rows, 2)
                scale = header[:, 0]
                min_log_probability = header[:, 1]
                quantized = raw[:, 4 : 4 + n_vocab]
                top2 = np.partition(quantized, n_vocab - 2, axis=1)[:, -2:]
                logp_second = scale * top2[:, 0].astype(np.float32) + min_log_probability
                logp_first = scale * top2[:, 1].astype(np.float32) + min_log_probability
                margins[chunk, done : done + rows] = np.exp(logp_first) - np.exp(logp_second)
                done += rows
            print(
                f"  chunk {chunk}: median probability margin {np.median(margins[chunk]):.6g}, "
                f"frac<=0 {float((margins[chunk] <= 0).mean()):.4f}"
            )

    identity_after_extract = base_identity(base_path, include_hash=False)
    if identity_after_extract != identity_before:
        raise SystemExit("base file changed while margins were being extracted")
    base_hash = sha256_file(base_path)
    identity_after_hash = base_identity(base_path, include_hash=False)
    if identity_after_hash != identity_before:
        raise SystemExit("base file changed while its identity was being hashed")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    margins.tofile(out_path)
    metadata = {
        "format_version": MARGIN_FORMAT_VERSION,
        "margin": "top2_probability_gap",
        "n_chunk": n_chunk,
        "n_scored": n_scored,
        "base": {**identity_after_hash, "sha256": base_hash},
    }
    Path(str(out_path) + ".meta").write_text(json.dumps(metadata, indent=2) + "\n")
    return margins


def load_probability_margins(base_path, margin_path, verify_hash):
    meta_path = Path(str(margin_path) + ".meta")
    try:
        metadata = json.loads(meta_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot validate cached margins ({error}); rebuild with --rebuild-margins"
        )
    if (
        metadata.get("format_version") != MARGIN_FORMAT_VERSION
        or metadata.get("margin") != "top2_probability_gap"
    ):
        raise SystemExit("cached margins use an obsolete formula; rebuild with --rebuild-margins")
    current = base_identity(base_path, include_hash=verify_hash)
    recorded = metadata.get("base", {})
    for field in ("path", "size", "mtime_ns"):
        if current[field] != recorded.get(field):
            raise SystemExit(
                f"cached margins belong to a different base ({field} mismatch); "
                "rebuild with --rebuild-margins"
            )
    if verify_hash and current["sha256"] != recorded.get("sha256"):
        raise SystemExit("cached margin base SHA-256 mismatch; rebuild with --rebuild-margins")
    n_chunk = int(metadata["n_chunk"])
    n_scored = int(metadata["n_scored"])
    margins = np.fromfile(margin_path, dtype=np.float32)
    if margins.size != n_chunk * n_scored:
        raise SystemExit(
            f"cached margins have {margins.size} values; expected {n_chunk * n_scored}"
        )
    return margins.reshape(n_chunk, n_scored)


def read_dump(path, expected_last_k=None, allow_legacy=False):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: truncated dump header")
    n_pos, n_chunk = struct.unpack_from("<ii", data)
    if n_pos <= 0 or n_chunk <= 0:
        raise ValueError(f"{path}: invalid shape {n_chunk}x{n_pos}")
    expected = 8 + 4 * n_pos * n_chunk
    if len(data) != expected:
        raise ValueError(f"{path}: size {len(data)} does not match shape {n_chunk}x{n_pos}")
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


def groups(n_chunk):
    if n_chunk < 2:
        raise ValueError("at least two chunks are required for reliability splits")
    return {
        "all": np.arange(n_chunk),
        "even": np.arange(0, n_chunk, 2),
        "odd": np.arange(1, n_chunk, 2),
        "firsthalf": np.arange(0, n_chunk // 2),
        "secondhalf": np.arange(n_chunk // 2, n_chunk),
    }


def trimmed(values, fraction=0.01):
    flat = np.sort(np.asarray(values).reshape(-1))
    trim = int(flat.size * fraction)
    if trim == 0:
        return float(flat.mean())
    if 2 * trim >= flat.size:
        raise ValueError("not enough values for requested trim")
    return float(flat[trim:-trim].mean())


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


def discover_dumps(directory, tag):
    anchors = {}
    cells = defaultdict(list)
    prefix = f"pxr_{tag}_"
    cell_pattern = re.compile(r"^(fp16-t8|t8-t4|t4-t3|t3-t2|t2-t1)_l([0-9]+)_([kv])$")
    for path in sorted(Path(directory).glob(f"{prefix}*.kld")):
        suffix = path.stem[len(prefix) :]
        if suffix.startswith("base_"):
            tier = suffix.removeprefix("base_")
            if tier in anchors:
                raise SystemExit(f"duplicate anchor for {tier}")
            anchors[tier] = path
            continue
        match = cell_pattern.fullmatch(suffix)
        if not match:
            continue
        transition, layer, side = match.groups()
        high, low = transition.split("-", 1)
        cells[(high, low)].append((int(layer), side, path))
    return anchors, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="./base_f16kv.kld")
    parser.add_argument("--margins", default="./margins.f32")
    parser.add_argument("--dumps", default="./dumps")
    parser.add_argument("--tag", default="q27")
    parser.add_argument(
        "--campaign-id",
        required=True,
        help="immutable build/model/inference-settings identifier embedded in the outputs",
    )
    parser.add_argument("--out", default="./hazard_out")
    parser.add_argument(
        "--last-k",
        type=int,
        help="analyze only the final K positions; required for TURBO_SCORE_LAST_K dumps",
    )
    parser.add_argument(
        "--allow-legacy-dumps",
        action="store_true",
        help="accept last-K dumps without metadata after manual verification",
    )
    parser.add_argument("--rebuild-margins", action="store_true")
    parser.add_argument(
        "--verify-base-hash",
        action="store_true",
        help="rehash the full base before using cached margins",
    )
    args = parser.parse_args()

    if args.last_k is not None and args.last_k <= 0:
        parser.error("--last-k must be positive")
    if args.rebuild_margins or not os.path.exists(args.margins):
        margins = extract_probability_margins(args.base, args.margins)
    else:
        margins = load_probability_margins(args.base, args.margins, args.verify_base_hash)

    anchors, cells = discover_dumps(args.dumps, args.tag)
    if not cells:
        raise SystemExit(f"no cell dumps found for tag {args.tag!r}")
    probability_margin = margins.astype(np.float64)
    denominator = 0.5 * probability_margin * probability_margin + 1e-6

    rows = []
    values = defaultdict(dict)
    anchor_hazard = {}
    for (high, low), entries in sorted(cells.items()):
        if high not in anchors:
            raise SystemExit(f"missing base anchor for transition {high}-{low}")
        anchor = read_dump(anchors[high], args.last_k, args.allow_legacy_dumps)
        if anchor.shape != denominator.shape:
            raise SystemExit(
                f"anchor shape {anchor.shape} does not match margins {denominator.shape}"
            )
        if args.last_k is not None:
            if args.last_k > anchor.shape[1]:
                raise SystemExit(f"--last-k {args.last_k} exceeds dump width {anchor.shape[1]}")
            anchor = anchor[:, -args.last_k :]
            transition_denominator = denominator[:, -args.last_k :]
        else:
            transition_denominator = denominator
        if high == "fp16" and np.any(anchor != 0.0):
            raise SystemExit("fp16 anchor is not exactly zero; the campaign is invalid")
        if high not in anchor_hazard:
            anchor_hazard[high] = anchor / transition_denominator >= 1.0
        anchor_flip_hazard = anchor_hazard[high]
        transition = f"{high}-{low}"
        for layer, side, path in sorted(entries):
            cell = read_dump(path, args.last_k, args.allow_legacy_dumps)
            if cell.shape != denominator.shape:
                raise SystemExit(
                    f"cell shape {cell.shape} does not match margins {denominator.shape}: {path}"
                )
            if args.last_k is not None:
                cell = cell[:, -args.last_k :]
            hazard = cell / transition_denominator
            weighted_excess = (cell - anchor) / transition_denominator
            for group_name, chunk_indices in groups(cell.shape[0]).items():
                risk_crossing_excess = float(
                    (hazard[chunk_indices] >= 1.0).mean() - anchor_flip_hazard[chunk_indices].mean()
                )
                stats = {
                    "risk_crossing_excess": risk_crossing_excess,
                    "mw_excess_mean": float(weighted_excess[chunk_indices].mean()),
                    "mw_excess_trim1": trimmed(weighted_excess[chunk_indices]),
                    "mw_excess_median": float(np.median(weighted_excess[chunk_indices])),
                }
                rows.append((path.stem, transition, layer, side, group_name, stats))
                for key, value in stats.items():
                    values[(transition, side, group_name, key)][layer] = value

    keys = ["risk_crossing_excess", "mw_excess_mean", "mw_excess_trim1", "mw_excess_median"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + "_cells.tsv", "w") as handle:
        handle.write("campaign_id\tname\ttransition\tlayer\tside\tgroup\t" + "\t".join(keys) + "\n")
        for name, transition, layer, side, group_name, stats in rows:
            handle.write(
                f"{args.campaign_id}\t{name}\t{transition}\t{layer}\t{side}\t{group_name}\t"
                + "\t".join(f"{stats[key]:.9g}" for key in keys)
                + "\n"
            )
    with open(args.out + "_reliability.tsv", "w") as handle:
        handle.write("campaign_id\ttransition\tside\tstat\trho_even_odd\trho_half\n")
        transition_sides = sorted({(transition, side) for transition, side, _, _ in values})
        for transition, side in transition_sides:
            for key in keys:
                even = values.get((transition, side, "even", key), {})
                odd = values.get((transition, side, "odd", key), {})
                first = values.get((transition, side, "firsthalf", key), {})
                second = values.get((transition, side, "secondhalf", key), {})
                common_even = sorted(set(even) & set(odd))
                common_half = sorted(set(first) & set(second))
                rho_even = spearman(
                    [even[layer] for layer in common_even],
                    [odd[layer] for layer in common_even],
                )
                rho_half = spearman(
                    [first[layer] for layer in common_half],
                    [second[layer] for layer in common_half],
                )
                handle.write(
                    f"{args.campaign_id}\t{transition}\t{side}\t{key}\t"
                    f"{rho_even:.3f}\t{rho_half:.3f}\n"
                )
    dump_paths = sorted(
        set(anchors.values()) | {entry[2] for entries in cells.values() for entry in entries}
    )
    margin_metadata = json.loads(Path(str(args.margins) + ".meta").read_text())
    output_metadata = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "tag": args.tag,
        "last_k": args.last_k,
        "base": margin_metadata["base"],
        "margin_cache": str(Path(args.margins).resolve()),
        "dumps": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "metadata_sha256": sha256_file(Path(str(path) + ".meta"))
                if Path(str(path) + ".meta").exists()
                else None,
            }
            for path in dump_paths
        ],
        "outputs": {
            "cells_sha256": sha256_file(args.out + "_cells.tsv"),
            "reliability_sha256": sha256_file(args.out + "_reliability.tsv"),
        },
    }
    Path(args.out + "_meta.json").write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        "probability-margin stats: median",
        float(np.median(probability_margin)),
        "p10",
        float(np.percentile(probability_margin, 10)),
        "frac<=0",
        float((probability_margin <= 0).mean()),
    )
    print("wrote", args.out + "_cells.tsv / _reliability.tsv")


if __name__ == "__main__":
    main()
