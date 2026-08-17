# Layer pricing — build a measured degrade order

For variable-bit-rate KV schemes that demote layers **individually**. Answers: *which
(layer, side) should give up a tier next?*

Different layers tolerate quantization very differently, and the ordering is not guessable
— it is not simply "deep layers first", and it differs between architectures. Two models we
measured with the same layer count had essentially **opposite** price structures. So it has
to be measured per model.

> **Depends on the KLD panel.** The prices come from per-token dumps of one-tier-at-a-time
> cells (needs `TURBO_KLD_DUMP` — see `../patches/`). Read `../METHODOLOGY.md` §6 before generating a panel — the cell design and the
> acceptance gate are the whole ballgame.

## The pipeline

```
KLD panel (per-token dumps, one (layer,side) moved one tier per cell)
      -> build_price_panel.py -> panel.tsv + reliability.tsv
      -> build_degrade_order.py
      -> order.txt   (VBR_DEGRADE_ORDER format)  [+ optional C array]
      -> validate with margin-bench
```

## What you measure

One cell per **(layer, side, transition)**, byte-identical to its anchor except that single
unit moved down one tier:

- sides: `k`, `v` (and optionally `kv` = both together — see the interaction note below)
- transitions: `fp16-t8`, `t8-t4`, `t4-t3`, `t3-t2`, `t2-t1`

For an N-KV-layer model that is `N × 2 × 5` cells. The price of a cell is the per-token diff
against its anchor, reduced to four statistics ("lenses"):

| lens | column | what it is |
|---|---|---|
| `median` | `excess_median` | median per-token excess — the flip predictor |
| `trim1` | `excess_trim0.01` | 1% trimmed mean |
| `frac` | `frac_gt_0.001` | fraction of tokens with excess > τ (catastrophe rate) |
| `mean` | `excess_mean` | carried for completeness — **never price on it** |

## Input schema

`panel.tsv` (tab-separated; `examples/panel_example.tsv` is a runnable 4-layer toy):

```
tag  campaign_id  group  side  layer  transition  excess_median  excess_trim0.01  frac_gt_0.001  excess_mean
```

- `tag` — model identifier, so several models can share one file (or leave constant).
- `campaign_id` — immutable build/model/settings identity; generated panels require it.
- `group` — chunk-subset label; use `all` for the priced rows and keep `even`/`odd`/half
  splits in the same file if you want, they are ignored by the builder.
- `side` — `k` or `v`. Rows with `side=kv` are skipped for pricing (see below).

`reliability.tsv` (`examples/reliability_example.tsv`):

```
tag  campaign_id  transition  side  stat  rho_half
```

`rho_half` is the split-half Spearman of the per-layer ranking for that statistic — even/odd
or first/second-half chunks, ranked separately, correlated. This is the acceptance gate.

Build both TSVs directly from the dump directory:

```bash
./build_price_panel.py --dumps ./dumps --tag mymodel \
  --campaign-id 'build=<sha>;model=<file hash>;ctx=...;chunks=...;fa=...' \
  --out-cells panel.tsv --out-reliability reliability.tsv
```

The reducer expects `pxr_mymodel_base_<high-tier>.kld` anchors and
`pxr_mymodel_<high-low>_l<LAYER>_<k|v>.kld` cells, for example
`pxr_mymodel_t8-t4_l12_v.kld`. Keep each hook-generated `.meta` beside its dump. For a
frontier panel, generate every artifact with the same `TURBO_SCORE_LAST_K=N` and pass
`--last-k N` to the reducer. The required `--campaign-id` is embedded in both TSVs; the
reducer also writes `panel.tsv.meta.json` with SHA-256 identities for every input and both
outputs.

## Build the order

```bash
./build_degrade_order.py \
    --cells panel.tsv --reliability reliability.tsv \
    --tag mymodel --lens auto --out order.txt
```

`--lens auto` picks the statistic with the best **worst-rung** reliability on your model and
**refuses to emit an order** if even the best one falls below `--min-rho` (default 0.8). A
panel that cannot reproduce its own ranking on two halves of one run cannot price anything.

Add `--emit-c mymodel` to also print a C array for baking into a runtime.

## How the order is constructed

1. **The whole `fp16→t8` band first**, cheapest-first, always priced with the `frac` lens —
   the only statistic that resolves a rung that fine.
2. **Price-per-bit water-fill** over the rest: repeatedly advance whichever unit has the
   lowest `excess / bits_saved`, until everything sits at the floor.

**Keep the entry band contiguous at the front.** In our runtime that is a contract, not a
preference: it is the one rung that is cheap *and* domain-reversible, so a controller can
spend it under transient pressure without imprinting irreversible re-encode error onto
tokens already in the cache. A hand-edited order that breaks the band forfeits that.

## Output format

`order.txt` is whitespace-separated tokens, `<layer><k|v>:<tier>`:

```
63k:t8 63v:t8 59v:t8 59k:t8 ... 11v:t2 3k:t1
```

Consumed via `VBR_DEGRADE_ORDER=order.txt`. Parsing is strict — one malformed token and the
runtime warns and falls back to its built-in order, silently discarding your work. Check the
log line on first use.

In buun-llama-cpp, installing a custom order also disables runtime co-tenancy demand
shedding for that cache. The custom schedule can still degrade under its own capacity
controller, but it will not publish/service the default cross-context shedding policy.
Treat that behavior as part of deployment validation, not merely a file-format detail.

## Gotchas that will silently ruin a panel

- **Partial offload.** If any layer's KV lives on CPU it may fall back to a fixed type and
  be *pinned* — it never degrades, so its cells are meaningless and the water-fill is built
  on a hole. **Measure with all KV on GPU**, even if that means a smaller quant or shorter
  context for the measurement runs.
- **Interaction terms.** K+V swapped together is super-additive (1.61× of k+v at `t8→t4`,
  1.10× at the 1-bit rung). Per-side cells alone *under-price* joint swaps. Measure `kv`
  cells if you can; they are excluded from pricing but tell you how wrong the additive
  assumption is on your model.
- **Same build for the whole campaign.** Ride an unchanged exact-zero fp16 anchor through
  every campaign and stop if it moves at all.
- **Architectures with mixed attention** (e.g. full-attention layers interleaved with
  sliding-window ones) still need **one order covering all layers**; a runtime that splits
  the cache into sub-caches will filter the order to each one's own layers.
- **Deployment-true configuration.** Measure with the same tap/codec configuration you ship;
  a mismatch re-prices everything.

## Validate before shipping

An order is a *hypothesis about quality*, so confirm it with a task-grounded run
(`../margin-bench/`) rather than on fidelity numbers alone. The water-fill amplifies lens
noise — a schedule built from an unreliable statistic can be worse than no schedule at all.
