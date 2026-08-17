#!/usr/bin/env python3
"""
Build a measured per-(layer, side) KV degrade order from a price panel.

Input  : a price-panel TSV (one row per measured cell) + a reliability TSV.
Output : an order file in `VBR_DEGRADE_ORDER` token format, plus optionally a C
         array you can bake into a runtime.

The order is what a variable-bit-rate KV controller walks when it needs to free
memory: "which (layer, side) gives up a tier next?" — cheapest quality loss per
byte saved, first.

Construction (two stages):

  1. The whole fp16->t8 band first, cheapest-first. In our runtime this is a
     contract, not just an ordering preference: it is the one rung that is both
     cheap and domain-reversible, so a controller may spend it under transient
     pressure without imprinting irreversible re-encode error onto tokens already
     in the cache. Keep it contiguous at the front.
     This band is ALWAYS priced with the `frac` lens — the only statistic that
     resolves a transition that fine (see ../METHODOLOGY.md).

  2. Then a price-per-bit water-fill over the rest of the chain: repeatedly take
     the unit whose next hop has the lowest  excess_price / bits_saved,  advance
     it one tier, repeat until everything sits at the floor.

Lens selection: pass --lens explicitly, or --lens auto to pick the statistic with
the best split-half reliability ON THIS MODEL (recommended). The mean is never
selected automatically — it was not the best statistic on any model we measured,
and the water-fill amplifies lens noise.

Usage:
  ./build_degrade_order.py --cells panel.tsv --reliability rel.tsv \
        --lens auto --out order.txt [--emit-c mymodel]
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict

# Effective bits/value per tier, including block overhead. Adjust to your codec
# ladder; only the RATIOS matter to the water-fill.
BITS = {"t8": 8.125, "t4": 4.125, "t3": 3.25, "t2": 2.25, "t1": 1.25}
CHAIN = [("t8", "t4"), ("t4", "t3"), ("t3", "t2"), ("t2", "t1")]
ENTRY_BAND = "fp16-t8"
ALL_TRANSITIONS = [ENTRY_BAND] + [f"{hi}-{lo}" for hi, lo in CHAIN]
SIDES = ("k", "v")
LENS_COL = {
    "median": "excess_median",
    "trim1": "excess_trim0.01",
    "frac": "frac_gt_0.001",
    "mean": "excess_mean",
}
TIER_ENUM = {
    "t8": "VBR_TIER_T8",
    "t4": "VBR_TIER_T4",
    "t3": "VBR_TIER_T3_TCQ",
    "t2": "VBR_TIER_T2_TCQ",
    "t1": "VBR_TIER_T1_TCQ",
}


def load_cells(path, tag):
    """prices[lens][(layer, side, transition)] = float ; plus the sorted layer list."""
    prices, layers, seen, campaigns = defaultdict(dict), set(), set(), set()
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if tag and r.get("tag") != tag:
                continue
            # 'group' lets you carry chunk-subset splits in one file; price on 'all'.
            if r.get("group", "all") != "all":
                continue
            if r.get("campaign_id"):
                campaigns.add(r["campaign_id"])
            # side == 'kv' rows are joint K+V swaps: informative (they are
            # super-additive vs k+v alone) but not per-side prices. Never price on them.
            side = r["side"]
            if side == "kv":
                continue
            if side not in SIDES:
                sys.exit(f"invalid side {side!r}; expected 'k', 'v', or optional 'kv'")
            transition = r["transition"]
            if transition not in ALL_TRANSITIONS:
                sys.exit(f"invalid transition {transition!r}; expected one of {ALL_TRANSITIONS}")
            layer = int(r["layer"])
            if layer < 0 or layer > 255:
                sys.exit(f"layer {layer} is outside the runtime's uint8 layer-id range")
            key = (layer, side, transition)
            if key in seen:
                sys.exit(
                    f"duplicate priced row for layer={layer} side={side} transition={transition}"
                )
            seen.add(key)
            layers.add(layer)
            for lens, col in LENS_COL.items():
                if col in r and r[col] not in ("", "-"):
                    value = float(r[col])
                    if not math.isfinite(value):
                        sys.exit(
                            f"non-finite {col} for layer={layer} side={side} "
                            f"transition={transition}"
                        )
                    prices[lens][key] = value
    if len(campaigns) > 1:
        sys.exit(f"cell panel mixes campaign IDs: {sorted(campaigns)}")
    return prices, sorted(layers), next(iter(campaigns), None)


def load_reliability(rel_path, tag):
    values, campaigns = {}, set()
    with open(rel_path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if tag and r.get("tag") != tag:
                continue
            if r.get("campaign_id"):
                campaigns.add(r["campaign_id"])
            stat = r["stat"]
            if stat not in LENS_COL:
                continue
            key = (r["transition"], r["side"], stat)
            if key in values:
                sys.exit(
                    f"duplicate reliability row for transition={key[0]} side={key[1]} stat={key[2]}"
                )
            value = float(r["rho_half"])
            if not math.isfinite(value):
                sys.exit(f"non-finite rho_half for transition={key[0]} side={key[1]} stat={key[2]}")
            values[key] = value
    if len(campaigns) > 1:
        sys.exit(f"reliability panel mixes campaign IDs: {sorted(campaigns)}")
    return values, next(iter(campaigns), None)


def reliability_score(values, stat, transitions):
    expected = {(transition, side, stat) for transition in transitions for side in SIDES}
    missing = sorted(expected - values.keys())
    if missing:
        return None, missing
    return min(values[key] for key in expected), []


def check_entry_reliability(values, min_rho):
    score, missing = reliability_score(values, "frac", [ENTRY_BAND])
    if missing:
        sys.exit(
            "entry-band frac reliability is incomplete: "
            + ", ".join(f"{transition}/{side}" for transition, side, _ in missing)
        )
    if score < min_rho:
        sys.exit(
            f"entry-band frac reliability only reaches rho={score:+.2f} (< {min_rho}); "
            "do not ship this order"
        )
    print(f"[lens] entry-band frac worst-side rho={score:+.2f}", file=sys.stderr)


def pick_lens(values, min_rho):
    """Choose the statistic with the best WORST-rung split-half reliability."""
    scored = {}
    incomplete = {}
    chain_transitions = [f"{hi}-{lo}" for hi, lo in CHAIN]
    for stat in ("median", "trim1", "frac"):
        score, missing = reliability_score(values, stat, chain_transitions)
        if missing:
            incomplete[stat] = missing
        else:
            scored[stat] = score
    if not scored:
        details = "; ".join(
            f"{stat}: {len(rows)} missing" for stat, rows in sorted(incomplete.items())
        )
        sys.exit(
            "no non-mean lens has a complete transition x side reliability matrix"
            + (f" ({details})" if details else "")
        )
    print(
        "[lens] worst-rung split-half rho:  "
        + "  ".join(f"{k}={v:+.2f}" for k, v in sorted(scored.items())),
        file=sys.stderr,
    )
    best = max(scored, key=scored.get)
    if scored[best] < min_rho:
        sys.exit(
            f"[lens] best statistic '{best}' only reaches rho={scored[best]:+.2f} "
            f"(< {min_rho}). The panel does not replicate — do not ship this order. "
            f"Add chunks/prompts, or check the gates in ../METHODOLOGY.md first."
        )
    print(
        f"[lens] selected '{best}' (worst-rung rho={scored[best]:+.2f})",
        file=sys.stderr,
    )
    return best


def band_order(prices, layers, lens, transition):
    """One transition, both sides, cheapest (least protected) first."""
    items = []
    for layer in layers:
        for side in SIDES:
            p = prices[lens].get((layer, side, transition))
            if p is not None:
                items.append((max(p, 0.0), layer, side))
    items.sort()
    return [(layer, side) for _, layer, side in items]


def waterfill(prices, layers, lens):
    """Greedy price-per-bit descent from t8 to the floor."""
    state = {(layer, side): "t8" for layer in layers for side in SIDES}
    nxt_of, order = dict(CHAIN), []
    while True:
        best = bkey = None
        for (layer, side), cur in state.items():
            nxt = nxt_of.get(cur)
            if nxt is None:
                continue
            p = prices[lens].get((layer, side, f"{cur}-{nxt}"))
            if p is None:
                continue
            key = max(p, 0.0) / (BITS[cur] - BITS[nxt])  # excess per bit saved
            if best is None or key < best:
                best, bkey = key, (layer, side, cur, nxt)
        if bkey is None:
            return order
        layer, side, cur, nxt = bkey
        order.append((layer, side, nxt))
        state[(layer, side)] = nxt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cells", required=True, help="price panel TSV")
    ap.add_argument("--reliability", help="split-half reliability TSV (needed for --lens auto)")
    ap.add_argument("--tag", default="", help="model tag, if the TSVs hold several models")
    ap.add_argument("--lens", default="auto", choices=["auto", "median", "trim1", "frac", "mean"])
    ap.add_argument(
        "--min-rho",
        type=float,
        default=0.8,
        help="reject the panel if the best statistic is below this (default 0.8)",
    )
    ap.add_argument("--out", default="order.txt", help="VBR_DEGRADE_ORDER file to write")
    ap.add_argument(
        "--emit-c",
        metavar="NAME",
        help="also print a C array named vbr_order_<NAME> for baking in",
    )
    ap.add_argument(
        "--allow-unreliable",
        action="store_true",
        help="diagnostic only: skip reliability checks for an explicitly selected lens",
    )
    a = ap.parse_args()
    if not math.isfinite(a.min_rho) or not -1.0 <= a.min_rho <= 1.0:
        ap.error("--min-rho must be finite and between -1 and 1")

    prices, layers, cell_campaign = load_cells(a.cells, a.tag)
    if not layers:
        sys.exit("no rows matched — check --tag, and that 'side' is 'k'/'v'")

    required_keys = {
        (layer, side, transition)
        for layer in layers
        for side in SIDES
        for transition in ALL_TRANSITIONS
    }
    observed_keys = set().union(*(set(rows) for rows in prices.values()))
    missing_rows = sorted(required_keys - observed_keys)
    if missing_rows:
        preview = ", ".join(
            f"l{layer}/{side}/{transition}" for layer, side, transition in missing_rows[:8]
        )
        more = f" (+{len(missing_rows) - 8} more)" if len(missing_rows) > 8 else ""
        sys.exit(f"price panel is incomplete: {len(missing_rows)} missing rows: {preview}{more}")

    lens = a.lens
    rel_values = None
    reliability_campaign = None
    if a.reliability:
        rel_values, reliability_campaign = load_reliability(a.reliability, a.tag)
        if not rel_values:
            sys.exit("reliability file has no rows matching --tag")
        if cell_campaign != reliability_campaign:
            sys.exit(
                "cells and reliability provenance differ: "
                f"{cell_campaign!r} vs {reliability_campaign!r}"
            )
    if lens == "auto":
        if rel_values is None:
            sys.exit("--lens auto needs --reliability")
        check_entry_reliability(rel_values, a.min_rho)
        lens = pick_lens(rel_values, a.min_rho)
    elif not a.allow_unreliable:
        if rel_values is None:
            sys.exit(
                "an explicit --lens still needs --reliability; use --allow-unreliable "
                "for diagnostics"
            )
        check_entry_reliability(rel_values, a.min_rho)
        score, missing = reliability_score(rel_values, lens, [f"{hi}-{lo}" for hi, lo in CHAIN])
        if missing:
            sys.exit(f"selected lens {lens!r} has {len(missing)} missing reliability rows")
        if score < a.min_rho:
            sys.exit(f"selected lens {lens!r} only reaches rho={score:+.2f} (< {a.min_rho})")

    required_prices = {(layer, side, ENTRY_BAND) for layer in layers for side in SIDES}
    missing_entry_prices = sorted(required_prices - prices["frac"].keys())
    chain_transitions = [f"{hi}-{lo}" for hi, lo in CHAIN]
    required_chain_prices = {
        (layer, side, transition)
        for layer in layers
        for side in SIDES
        for transition in chain_transitions
    }
    missing_chain_prices = sorted(required_chain_prices - prices[lens].keys())
    if missing_entry_prices or missing_chain_prices:
        missing = [("frac", *key) for key in missing_entry_prices]
        missing += [(lens, *key) for key in missing_chain_prices]
        preview = ", ".join(
            f"{stat}:l{layer}/{side}/{transition}" for stat, layer, side, transition in missing[:8]
        )
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        sys.exit(f"price panel lacks values required by the selected lenses: {preview}{more}")

    steps = [(layer, side, "t8") for layer, side in band_order(prices, layers, "frac", ENTRY_BAND)]
    n_band = len(steps)
    steps += waterfill(prices, layers, lens)

    expected = len(layers) * len(SIDES) * (1 + len(CHAIN))
    if len(steps) != expected:
        sys.exit(f"internal error: constructed {len(steps)} steps, expected {expected}")

    if a.emit_c and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", a.emit_c):
        sys.exit("--emit-c NAME must be a valid C identifier suffix")

    with open(a.out, "w") as fh:
        for i in range(0, len(steps), 12):
            fh.write(
                " ".join(f"{layer}{side}:{tier}" for layer, side, tier in steps[i : i + 12]) + "\n"
            )
    print(
        f"[out] {a.out}: {len(steps)} steps, lens={lens}, fp16->t8 band={n_band}",
        file=sys.stderr,
    )

    if a.emit_c:
        print(f"// lens={lens} (fp16->t8 band: frac), {len(steps)} steps, layers={len(layers)}")
        print(f"static const vbr_degrade_step vbr_order_{a.emit_c}[] = {{")
        for i in range(0, len(steps), 6):
            row = ", ".join(
                f"{{{layer:2d}, {1 if side == 'v' else 0}, {TIER_ENUM[tier]}}}"
                for layer, side, tier in steps[i : i + 6]
            )
            print(f"    {row},")
        print("};")


if __name__ == "__main__":
    main()
