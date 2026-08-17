#!/usr/bin/env python3
"""Summarize expected-target token margins from router probe results."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def target_span_margins(record):
    tokens = record.get("lp") or []
    if not tokens:
        return None
    text = "".join(token[0] for token in tokens)
    target = record["expected_target"]
    marker = text.rfind("FINAL_TARGET")
    target_start = text.find(target, marker if marker >= 0 else 0)
    if target_start < 0:
        return None
    target_end = target_start + len(target)
    margins, logprobs = [], []
    offset = 0
    for token in tokens:
        if len(token) < 3:
            return None
        token_text, logprob, runner_up = token
        token_start, token_end = offset, offset + len(token_text)
        offset = token_end
        if token_end <= target_start or token_start >= target_end:
            continue
        logprobs.append(logprob)
        if runner_up is not None:
            margins.append(logprob - runner_up)
    if not margins:
        return None
    return margins, logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    print(
        f"{'label':22s} {'total':>5s} {'exact':>5s} {'errors':>6s} {'scored':>6s} "
        f"{'mean_margin':>11s} {'mean_case_min':>13s} {'p10_case_min':>12s} "
        f"{'worst_case':>10s} {'mean_lp':>8s} {'frac_m<5':>8s}"
    )
    exit_status = 0
    for path in args.results:
        latest = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    latest[record["id"]] = record
        if not latest:
            print(f"{path}: no records", file=sys.stderr)
            exit_status = 2
            continue
        labels = {record.get("label") for record in latest.values() if record.get("label")}
        label = labels.pop() if len(labels) == 1 else Path(path).stem.removeprefix("lp_")
        case_min, case_mean, case_lp = [], [], []
        n_low = n_token = 0
        for record in latest.values():
            if not record.get("exact"):
                continue
            result = target_span_margins(record)
            if result is None:
                continue
            margins, logprobs = result
            case_min.append(min(margins))
            case_mean.append(sum(margins) / len(margins))
            case_lp.append(sum(logprobs) / len(logprobs))
            n_low += sum(1 for margin in margins if margin < 5.0)
            n_token += len(margins)
        total = len(latest)
        exact = sum(bool(record.get("exact")) for record in latest.values())
        errors = sum(bool(record.get("error")) for record in latest.values())
        scored = len(case_min)
        if not case_min:
            print(
                f"{label:22s} {total:5d} {exact:5d} {errors:6d} {scored:6d} "
                f"{'-':>11s} {'-':>13s} {'-':>12s} {'-':>10s} {'-':>8s} {'-':>8s}"
            )
            exit_status = 2
            continue
        p10 = float(np.percentile(case_min, 10))
        print(
            f"{label:22s} {total:5d} {exact:5d} {errors:6d} {scored:6d} "
            f"{sum(case_mean) / scored:11.3f} {sum(case_min) / scored:13.3f} {p10:12.3f} "
            f"{min(case_min):10.3f} {sum(case_lp) / scored:8.4f} "
            f"{n_low / max(1, n_token):8.4f}"
        )
        if errors or scored != exact:
            print(
                f"warning: {label}: expected one target-margin score per exact case; "
                f"exact={exact}, scored={scored}, errors={errors}",
                file=sys.stderr,
            )
            exit_status = 2
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
