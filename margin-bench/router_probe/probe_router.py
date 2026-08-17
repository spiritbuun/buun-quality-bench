#!/usr/bin/env python3
"""Run deterministic routing probes against an OpenAI-compatible endpoint."""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path


PATTERN = re.compile(
    r"FINAL_ACTION\s*=\s*([A-Za-z_]+)\s*;\s*"
    r"FINAL_TARGET\s*=\s*([A-Za-z0-9_:/\.\-]+)\s*;\s*"
    r"SOURCE_RANK\s*=\s*(\d+)"
)


def query(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 4,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    choice = result["choices"][0]
    usage = result.get("usage", {})
    logprobs = []
    for token in (choice.get("logprobs") or {}).get("content") or []:
        runner_up = None
        skipped_chosen = False
        for alternative in token.get("top_logprobs", []):
            if (
                not skipped_chosen
                and alternative["token"] == token["token"]
                and alternative["logprob"] == token["logprob"]
            ):
                skipped_chosen = True
                continue
            runner_up = alternative["logprob"]
            break
        logprobs.append([token["token"], token["logprob"], runner_up])
    return (
        choice["message"]["content"],
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        logprobs,
    )


def case_fingerprint(case):
    payload = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def completed_record(record):
    return not record.get("error") and "exact" in record and bool(record.get("lp"))


def load_cases(path):
    with open(path, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if not cases:
        raise SystemExit("case file is empty")
    required = {"id", "user", "expected_action", "expected_target", "expected_rank"}
    for index, case in enumerate(cases, 1):
        missing = required - case.keys()
        if missing:
            raise SystemExit(f"case {index} lacks fields: {', '.join(sorted(missing))}")
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("case file contains duplicate IDs")
    return cases


def latest_records(path):
    latest = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "id" not in record:
                    raise SystemExit(f"{path}:{line_number}: record has no id")
                latest[record["id"]] = record
    except FileNotFoundError:
        pass
    return latest


def bind_campaign(args):
    output = Path(args.out)
    metadata_path = Path(str(output) + ".meta.json")
    case_bytes = Path(args.data).read_bytes()
    expected = {
        "schema_version": 1,
        "config_id": args.config_id,
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "max_tokens": args.max_tokens,
        "request_timeout": args.timeout,
        "temperature": 0.0,
        "top_logprobs": 4,
        "enable_thinking": False,
    }
    if metadata_path.exists():
        recorded = json.loads(metadata_path.read_text())
        if recorded != expected:
            raise SystemExit(
                f"refusing to resume {output}: campaign settings differ from {metadata_path}"
            )
    elif output.exists() and output.stat().st_size:
        raise SystemExit(
            f"refusing to adopt result file without campaign metadata: {output}; use a fresh --out"
        )
    else:
        metadata_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--label", required=True, help="short display label for this configuration")
    parser.add_argument(
        "--config-id",
        required=True,
        help="immutable build/model/server-flag identifier recorded for safe resume",
    )
    args = parser.parse_args()

    cases = load_cases(args.data)
    bind_campaign(args)
    latest = latest_records(args.out)
    done = {case_id: record for case_id, record in latest.items() if completed_record(record)}
    for case in cases:
        if case["id"] not in done:
            continue
        fingerprint = case_fingerprint(case)
        recorded_fingerprint = done[case["id"]].get("case_sha256")
        if recorded_fingerprint is None:
            raise SystemExit(
                f"existing result {case['id']} has no case fingerprint; use a fresh --out file"
            )
        if recorded_fingerprint != fingerprint:
            raise SystemExit(
                f"existing result {case['id']} belongs to different case content; "
                "use a fresh --out file"
            )

    started = time.time()
    with open(args.out, "a", encoding="utf-8") as output:
        for index, case in enumerate(cases, 1):
            if case["id"] in done:
                continue
            record = {
                "id": case["id"],
                "label": args.label,
                "case_sha256": case_fingerprint(case),
                "expected_action": case["expected_action"],
                "expected_target": case["expected_target"],
                "expected_rank": case["expected_rank"],
            }
            try:
                text, prompt_tokens, completion_tokens, logprobs = query(
                    args.base_url,
                    args.model,
                    case["user"],
                    args.max_tokens,
                    args.timeout,
                )
                match = PATTERN.search(text or "")
                record.update(
                    raw=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    lp=logprobs,
                )
                if match:
                    action, target, rank = (
                        match.group(1),
                        match.group(2),
                        int(match.group(3)),
                    )
                    record.update(
                        got_action=action,
                        got_target=target,
                        got_rank=rank,
                        ok_action=(action == case["expected_action"]),
                        ok_target=(target == case["expected_target"]),
                        ok_rank=(rank == case["expected_rank"]),
                    )
                    record["exact"] = (
                        record["ok_action"] and record["ok_target"] and record["ok_rank"]
                    )
                else:
                    record.update(
                        exact=False,
                        ok_action=False,
                        ok_target=False,
                        ok_rank=False,
                        parse_fail=True,
                    )
            except Exception as error:  # Keep the case retryable and preserve the diagnostic.
                record.update(error=f"{type(error).__name__}: {error}", exact=False)
            output.write(json.dumps(record) + "\n")
            output.flush()
            if index % 10 == 0:
                print(
                    f"  [{index}/{len(cases)}] elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    latest = latest_records(args.out)
    total = len(cases)
    exact = sum(bool(latest.get(case["id"], {}).get("exact")) for case in cases)
    action = sum(bool(latest.get(case["id"], {}).get("ok_action")) for case in cases)
    target = sum(bool(latest.get(case["id"], {}).get("ok_target")) for case in cases)
    rank = sum(bool(latest.get(case["id"], {}).get("ok_rank")) for case in cases)
    errors = sum(bool(latest.get(case["id"], {}).get("error")) for case in cases)
    missing = sum(case["id"] not in latest for case in cases)
    print(
        f"SUMMARY label={args.label} n={total} exact={exact}/{total} ({100 * exact / total:.1f}%) "
        f"action={action}/{total} target={target}/{total} rank={rank}/{total} "
        f"errors={errors} missing={missing}"
    )
    if errors or missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
