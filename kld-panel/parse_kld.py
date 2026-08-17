#!/usr/bin/env python3
"""Summarize llama-perplexity KLD logs across candidates and context depths."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


STAT_PATTERNS = {
    "mean": r"Mean\s+KLD:\s*(-?[0-9.eE+]+)",
    "median": r"Median\s+KLD:\s*(-?[0-9.eE+]+)",
    "p95": r"95\.0%\s+KLD:\s*(-?[0-9.eE+]+)",
    "p99": r"99\.0%\s+KLD:\s*(-?[0-9.eE+]+)",
    "p999": r"99\.9%\s+KLD:\s*(-?[0-9.eE+]+)",
    "max": r"Maximum KLD:\s*(-?[0-9.eE+]+)",
    "rms": r"RMS Δp\s*:\s*(-?[0-9.eE+]+)",
    "same": r"Same top p:\s*(-?[0-9.eE+]+)",
}
REQUIRED = tuple(STAT_PATTERNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--prefix", default="", help="only consider filenames with this prefix")
    parser.add_argument(
        "--pattern",
        help=(
            "regular expression for the filename after --prefix; must define named "
            "groups 'candidate' and 'depth'"
        ),
    )
    parser.add_argument(
        "--depths",
        help="comma-separated required depths; default: infer the union found in the logs",
    )
    parser.add_argument("--top", type=int, default=8, help="leaderboard row count")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report complete candidates and exit successfully despite incomplete ones",
    )
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    stats: dict[str, float] = {}
    for key, pattern in STAT_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            stats[key] = float(match.group(1))
    missing = [key for key in REQUIRED if key not in stats]
    if missing:
        raise ValueError(f"missing summary fields: {', '.join(missing)}")
    stats["flip"] = 100.0 - stats["same"]
    return stats


def identify(filename: str, prefix: str, pattern: re.Pattern[str] | None) -> tuple[str, int] | None:
    if not filename.startswith(prefix):
        return None
    name = filename[len(prefix) :]
    patterns = (
        [pattern]
        if pattern
        else [
            re.compile(r"(?P<candidate>.+)_ctx(?P<depth>\d+)\.log$"),
            re.compile(r"(?P<candidate>.+)_(?P<depth>\d+)\.log$"),
        ]
    )
    for candidate_pattern in patterns:
        if candidate_pattern is None:
            continue
        match = candidate_pattern.fullmatch(name)
        if match:
            return match.group("candidate"), int(match.group("depth"))
    return None


def mean_at_depths(
    data: dict[str, dict[int, dict[str, float]]],
    candidate: str,
    depths: list[int],
    key: str,
) -> float:
    return sum(data[candidate][depth][key] for depth in depths) / len(depths)


def display_name(candidate: str) -> str:
    return f"iter{int(candidate):03d}" if candidate.isdigit() else candidate


def main() -> int:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be positive")
    custom_pattern = re.compile(args.pattern) if args.pattern else None
    if custom_pattern and not {"candidate", "depth"}.issubset(custom_pattern.groupindex):
        raise SystemExit("--pattern must define named groups 'candidate' and 'depth'")

    data: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    errors: list[str] = []
    for path in sorted(args.log_dir.glob(f"{args.prefix}*.log")):
        identity = identify(path.name, args.prefix, custom_pattern)
        if identity is None:
            continue
        candidate, depth = identity
        if depth in data[candidate]:
            errors.append(f"{path}: duplicate candidate/depth {candidate!r}/{depth}")
            continue
        try:
            data[candidate][depth] = parse_log(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")

    if not data:
        print(f"No parseable KLD logs found in {args.log_dir}", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2

    if args.depths:
        depths = sorted({int(value) for value in args.depths.split(",") if value.strip()})
    else:
        depths = sorted({depth for candidate in data.values() for depth in candidate})
    if not depths:
        print("No context depths selected", file=sys.stderr)
        return 2

    complete = sorted(
        candidate for candidate, values in data.items() if all(depth in values for depth in depths)
    )
    incomplete = {
        candidate: [depth for depth in depths if depth not in values]
        for candidate, values in data.items()
        if candidate not in complete
    }
    print(
        f"candidates parsed: {len(data)}; complete at {len(depths)} depths "
        f"({', '.join(map(str, depths))}): {len(complete)}"
    )
    if incomplete:
        print("incomplete candidates:")
        for candidate, missing in sorted(incomplete.items()):
            print(f"  {display_name(candidate)}: missing {', '.join(map(str, missing))}")
    if errors:
        print("malformed logs:")
        for error in errors:
            print(f"  {error}")
    print()

    if not complete:
        print("No candidate has a complete panel", file=sys.stderr)
        return 2

    def leaderboard(key: str, depth: int | None, label: str) -> None:
        rows = [
            (
                data[candidate][depth][key]
                if depth is not None
                else mean_at_depths(data, candidate, depths, key),
                candidate,
            )
            for candidate in complete
        ]
        rows.sort(key=lambda row: (row[0], row[1]))
        scope = f"@{depth}" if depth is not None else "(equal-weight depth mean)"
        print(f"--- best {label} {scope} ---")
        for value, candidate in rows[: args.top]:
            print(f"  {display_name(candidate):24s} {value:.6f}")
        print()

    deepest = depths[-1]
    print("=" * 72)
    print("LEADERBOARDS (lower is better)")
    print("=" * 72)
    leaderboard("mean", None, "mean KLD")
    leaderboard("median", None, "median KLD")
    leaderboard("p99", None, "p99 KLD")
    leaderboard("p999", None, "p99.9 KLD")
    leaderboard("mean", deepest, "mean KLD")
    leaderboard("flip", deepest, "argmax flip %")

    axes = {
        "MEAN (depth mean)": min(complete, key=lambda c: mean_at_depths(data, c, depths, "mean")),
        "MEDIAN (depth mean)": min(
            complete, key=lambda c: mean_at_depths(data, c, depths, "median")
        ),
        "P99 (depth mean)": min(complete, key=lambda c: mean_at_depths(data, c, depths, "p99")),
        f"DEEP (mean@{deepest})": min(complete, key=lambda c: data[c][deepest]["mean"]),
    }
    print("=" * 72)
    print("AXIS WINNERS — full per-depth panels")
    print("=" * 72)
    for axis, candidate in axes.items():
        print(f"\n### {axis}: {display_name(candidate)}")
        print(
            f"{'depth':>8} {'mean':>10} {'median':>10} {'p95':>10} {'p99':>10} "
            f"{'p99.9':>10} {'max':>10} {'RMSdp%':>9} {'flip%':>9}"
        )
        for depth in depths:
            stats = data[candidate][depth]
            print(
                f"{depth:>8} {stats['mean']:>10.6f} {stats['median']:>10.6f} "
                f"{stats['p95']:>10.6f} {stats['p99']:>10.6f} "
                f"{stats['p999']:>10.6f} {stats['max']:>10.6f} "
                f"{stats['rms']:>9.3f} {stats['flip']:>9.3f}"
            )

    distinct = sorted({candidate for candidate in axes.values()})
    print(f"\nDistinct axis winners: {', '.join(display_name(c) for c in distinct)}")
    return 2 if (errors or incomplete) and not args.allow_incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
