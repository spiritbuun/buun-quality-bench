#!/usr/bin/env python3
"""Compare routing-probe task outcomes, then paired confidence margins.

Task correctness is the primary verdict. Target-span margin is interpreted only
as a confidence tie-breaker when the compared configurations have identical
per-case correctness outcomes.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def display_name(path, records):
    labels = {record.get("label") for record in records.values() if record.get("label")}
    if len(labels) == 1:
        return labels.pop()
    name = Path(path).stem
    return name.removeprefix("lp_")


def target_span_margin(record):
    tokens = record.get("lp") or []
    target = record.get("expected_target")
    if not tokens or not target:
        return None
    text = "".join(token[0] for token in tokens)
    marker = text.rfind("FINAL_TARGET")
    start_target = text.find(target, marker if marker >= 0 else 0)
    if start_target < 0:
        return None
    end_target = start_target + len(target)
    offset = 0
    margins = []
    for token in tokens:
        if len(token) < 3:
            return None
        token_text, logprob, runner_up = token
        start_token, end_token = offset, offset + len(token_text)
        offset = end_token
        if end_token <= start_target or start_token >= end_target:
            continue
        if runner_up is not None:
            margins.append(logprob - runner_up)
    return min(margins) if margins else None


def answer_margin(record):
    margins = [
        token[1] - token[2]
        for token in (record.get("lp") or [])
        if len(token) >= 3 and token[2] is not None
    ]
    return min(margins) if margins else None


def load(path, scope):
    records = {}
    duplicate_ids = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record:
                raise SystemExit(f"{path}:{line_number}: record has no id")
            if record["id"] in records:
                duplicate_ids.add(record["id"])
            records[record["id"]] = record
    if not records:
        raise SystemExit(f"{path}: no records")
    scorer = target_span_margin if scope == "target" else answer_margin
    for record in records.values():
        record["_margin"] = scorer(record)
    return records, duplicate_ids


def validate_pair(name_a, records_a, name_b, records_b, allow_incomplete):
    ids_a, ids_b = set(records_a), set(records_b)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)
        only_b = sorted(ids_b - ids_a)
        message = (
            f"coverage mismatch {name_a} vs {name_b}: "
            f"only-{name_a}={only_a[:8]} only-{name_b}={only_b[:8]}"
        )
        if not allow_incomplete:
            raise SystemExit(message)
        print("WARNING:", message)
    common = sorted(ids_a & ids_b)
    for case_id in common:
        left, right = records_a[case_id], records_b[case_id]
        left_fingerprint = left.get("case_sha256")
        right_fingerprint = right.get("case_sha256")
        if left_fingerprint and right_fingerprint:
            if left_fingerprint != right_fingerprint:
                raise SystemExit(f"case content mismatch for {case_id}: {name_a} vs {name_b}")
        else:
            left_expected = (
                left.get("expected_action"),
                left.get("expected_target"),
                left.get("expected_rank"),
            )
            right_expected = (
                right.get("expected_action"),
                right.get("expected_target"),
                right.get("expected_rank"),
            )
            if left_expected != right_expected:
                raise SystemExit(f"expected answer mismatch for {case_id}: {name_a} vs {name_b}")
    return common


def paired_statistics(left, right, seed, bootstrap_samples):
    delta = left - right
    mean = float(delta.mean())
    if delta.size < 2:
        t_value = float("nan")
    else:
        standard_deviation = float(delta.std(ddof=1))
        if standard_deviation == 0:
            t_value = 0.0 if mean == 0 else math.copysign(float("inf"), mean)
        else:
            t_value = mean / (standard_deviation / math.sqrt(delta.size))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(bootstrap_samples, delta.size))
    bootstrap_means = delta[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return mean, t_value, float(ci_low), float(ci_high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="two or more probe JSONL files")
    parser.add_argument(
        "--scope",
        choices=("target", "answer"),
        default="target",
        help="margin span (default: retrieved target only)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="diagnostic only: compare the ID intersection instead of failing",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    if len(args.results) < 2:
        parser.error("provide at least two result files")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    cells = {}
    for path in args.results:
        records, duplicates = load(path, args.scope)
        name = display_name(path, records)
        if name in cells:
            raise SystemExit(f"duplicate configuration label {name!r}")
        cells[name] = records
        if duplicates:
            print(f"NOTE: {name}: using latest records for {len(duplicates)} retried case IDs")

    print(f"margin scope: {args.scope}")
    print(f"{'config':22s} {'cases':>5s} {'exact':>9s} {'errors':>6s} {'scorable':>8s}")
    for name, records in cells.items():
        exact = sum(bool(record.get("exact")) for record in records.values())
        errors = sum(bool(record.get("error")) for record in records.values())
        scorable = sum(record.get("_margin") is not None for record in records.values())
        print(
            f"{name:22s} {len(records):5d} {exact:4d}/{len(records):<4d} {errors:6d} {scorable:8d}"
        )
        if errors and not args.allow_incomplete:
            raise SystemExit(f"{name}: contains {errors} request errors")

    names = list(cells)
    print()
    print(
        f"{'A vs B':<32} {'task verdict':<23} {'margin n':>8} {'mean dA-B':>10} "
        f"{'95% boot CI':>22} {'t':>8} {'A>B':>5} {'B>A':>5} {'minA':>8} {'minB':>8}"
    )
    for pair_index, (name_a, name_b) in enumerate(itertools.combinations(names, 2)):
        records_a, records_b = cells[name_a], cells[name_b]
        common = validate_pair(name_a, records_a, name_b, records_b, args.allow_incomplete)
        exact_a = {case_id: bool(records_a[case_id].get("exact")) for case_id in common}
        exact_b = {case_id: bool(records_b[case_id].get("exact")) for case_id in common}
        count_a, count_b = sum(exact_a.values()), sum(exact_b.values())
        if count_a > count_b:
            task_verdict = f"A wins exact {count_a}-{count_b}"
        elif count_b > count_a:
            task_verdict = f"B wins exact {count_b}-{count_a}"
        elif exact_a != exact_b:
            task_verdict = "exact tied; cases differ"
        else:
            task_verdict = "identical exact outcomes"

        margin_ids = [
            case_id
            for case_id in common
            if exact_a[case_id]
            and exact_b[case_id]
            and records_a[case_id]["_margin"] is not None
            and records_b[case_id]["_margin"] is not None
        ]
        expected_margin_ids = [
            case_id for case_id in common if exact_a[case_id] and exact_b[case_id]
        ]
        if len(margin_ids) != len(expected_margin_ids) and not args.allow_incomplete:
            missing = sorted(set(expected_margin_ids) - set(margin_ids))
            raise SystemExit(f"unscorable correct cases for {name_a} vs {name_b}: {missing[:8]}")

        if len(margin_ids) < 2:
            print(
                f"{name_a + ' vs ' + name_b:<32} {task_verdict:<23} {len(margin_ids):8d} "
                f"{'-':>10} {'-':>22} {'-':>8} {'-':>5} {'-':>5} {'-':>8} {'-':>8}"
            )
            continue
        values_a = np.array([records_a[case_id]["_margin"] for case_id in margin_ids])
        values_b = np.array([records_b[case_id]["_margin"] for case_id in margin_ids])
        mean, t_value, ci_low, ci_high = paired_statistics(
            values_a, values_b, args.seed + pair_index, args.bootstrap_samples
        )
        print(
            f"{name_a + ' vs ' + name_b:<32} {task_verdict:<23} {len(margin_ids):8d} "
            f"{mean:10.4f} {'[' + format(ci_low, '.4f') + ', ' + format(ci_high, '.4f') + ']':>22} "
            f"{t_value:8.2f} {(values_a > values_b).sum():5d} {(values_b > values_a).sum():5d} "
            f"{values_a.min():8.3f} {values_b.min():8.3f}"
        )
        if exact_a != exact_b:
            print("  note: margin is diagnostic only because per-case task outcomes differ")

    print("\nworst 6 correct, scorable cases per cell (id: target min-margin):")
    for name, records in cells.items():
        scored = [
            (case_id, record["_margin"])
            for case_id, record in records.items()
            if record.get("exact") and record.get("_margin") is not None
        ]
        worst = sorted(scored, key=lambda item: item[1])[:6]
        print(f"  {name:<16} " + "  ".join(f"{case_id}:{value:.2f}" for case_id, value in worst))


if __name__ == "__main__":
    main()
