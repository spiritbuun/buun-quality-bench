# Methodology — how to measure model-quality changes

Worked example throughout is KV-cache quantization, but the reasoning applies to any change
that perturbs the output distribution (weight quants, fine-tunes, kernel rewrites).

---

## 1. Fidelity ≠ goodness

KLD, hazard, and flip-rate measure *distance from the fp16 model*, not quality. Margin
measures task-local confidence on answers the model got right; it is still secondary to the
task outcome. A codec can beat fp16 on a task while scoring worse on fidelity — we have a
weight-quant config that beats BF16 on a maths benchmark (26/30 vs 18/30) with *worse* KLD,
and a turbo-class codec that beats q4_0 on perplexity yet loses to it on hazard.

Task outcomes are the authority for that task. Fidelity metrics are cheap, dense proxies;
they do not prove broader usefulness or correctness.

### Historical example: better KLD, weaker task confidence on Laguna

A static-KV comparison on Laguna-S-2.1 (UD-IQ4_NL, 2× RTX 3090) produced opposite codec
orderings under a reduced 16k KLD panel and the 8k routing/NIAH margin bench. Build:
buun-llama-cpp
`5d7dc397783bb1a5f84eb3ef2d838b0cb35af596`.

| measure | turbo3 | turbo3_tcq | direction |
|---|---:|---:|---|
| median KLD @16k, 4 chunks | 0.026127 | **0.017500** | TCQ better by 33.02% |
| mean KLD @16k, 4 chunks | 0.088375 | **0.064200** | TCQ better by 27.36% |
| same-top @16k | 89.589% | **91.543%** | TCQ better by 1.954pt |
| routing/NIAH exact @8k | 120/120 | 120/120 | tied |
| mean per-case minimum answer margin | **5.9300** | 5.6479 | scalar better by 0.2821 |
| median per-case minimum answer margin | **5.9005** | 5.7830 | scalar better |
| worst per-case minimum answer margin | **3.6935** | 0.2623 | TCQ has a thin-tail outlier |

The routing panel was `margin-bench/router_probe/cases/rd_8192_c2.jsonl`; paired
scalar-minus-TCQ minimum-margin difference was +0.2821 logits, `t=2.20`, with 67/120
cases favoring scalar and 53/120 favoring TCQ. That is an **ambiguous task-confidence
signal, not an accuracy inversion**: both codecs answered every case exactly, and `t=2.20`
is only a descriptive statistic for this fixed synthetic set. The legacy answer-span
analysis also placed two of TCQ's three weakest minima on the fixed opening token `FINAL`,
not the retrieved target; the current scripts default to target-span margins to avoid that
confound.

---

## 2. The mean was unreliable for fine codec comparisons. Start with the median.

**The anatomy.** A per-layer mean KLD-excess of −2901 μnats decomposed as: top-50 negative
tokens −766 nats, top-50 positive +644 nats, net −285 over 98,280 tokens; signed median
+0.46 μnats; ~71% of tokens moved >100 μnats in *both* directions. The mean is a small
residual of two huge cancelling tails, carried by ~0.1% of tokens. That is why it can flip
sign across corpus halves and model quants.

**The reliability table.** Split-half Spearman of per-layer rankings — same run, 16 layers:

| statistic | t8→t4 | t4→t3 | t3→t2 | t2→t1 | fp16→t8 |
|---|---|---|---|---|---|
| excess **mean** | +0.10 | +0.27 | −0.24 | +0.89 | −0.22 |
| excess trim 0.1% | −0.35 | +0.16 | −0.26 | +0.73 | −0.05 |
| excess **trim 1%** | +0.88 | +0.91 | +0.81 | +0.96 | −0.05 |
| excess **median** | **+0.91** | **+0.96** | **+0.92** | **+0.98** | +0.02 |
| **frac tokens > 1e-3** | +0.88 | +0.96 | +0.93 | +0.97 | **+0.98** |

Notes: 0.1% trim is *not* enough — the unstable tail is fatter than that; 1% is the working
level. The mean only becomes usable at very coarse transitions (the ~1-bit rung) where the
signal is enormous. Note also that only `frac` resolves the finest (fp16→t8) rung at all —
which is why the pricing tool always uses `frac` for that band regardless of the lens you
pick for the rest.

### Codebook-training example

A sweep of 3 codebook pools × 100 training iterations, read under **mean** KLD, appeared
non-monotonic: iteration 15 sometimes beat converged iteration 60, within-family MSE↔KLD
correlation was approximately zero, and seed rankings changed.

Re-reading the **same archives** — no new runs — under both labels gave the following
ρ(statistic, iteration):

| pool | median@16k | mean@16k (same runs) |
|---|---|---|
| pool A | **−0.66** | −0.29 |
| pool B | **−0.92** | +0.09 |
| pool C | **−0.72** | −0.19 |

(Lower KLD = better, so negative ρ = training monotonically improves quality.) Training
improves the robust label essentially monotonically in all three pools; the mean label is
noise on identical data. Under median labels, train-MSE became predictive again (ρ ≈ +0.66).

---

## 3. median@16k predicted flips in our measured campaigns

The question a pricing statistic must answer: does it predict actual decision changes?
Two independent validations:

- **Run level.** median-KLD vs decision flips: **ρ = 0.76**, surviving restriction to
  converged codebooks (0.77). Mean-KLD collapsed with depth: 8k 0.49 → 16k 0.19 →
  32k **−0.16** — sign inverted. That was a convergence confound plus tail noise.
- **Layer level** (TRUE argmax flips from `llama-perplexity`'s `Same top p`, 144 cells):
  trueflip~median **+0.82** (t8→t4), **+0.84** (t3→t2), **+0.91/+0.93** (t1) — versus
  trueflip~mean **−0.39** at t8→t4. The mean was *anti-correlated* with real flips at the
  fine rung. Hazard-L (`KL/(½·margin²)`) matched the median family: +0.84…+0.96.

The lenses form two camps: **{median, trim1%, hazard-L, TRUE FLIPS}** vs **{frac>τ, mean}**.
Mechanism: catastrophe tokens are big KL hits on big-margin (confident) tokens — they do
not flip decisions. Flips live where margins are thin and a broad typical-token elevation
crosses them. The median family measures exactly that.

**Honest counterweight.** `frac>τ` replicates too, and measures a real but *different*
structure: large distortions on confident tokens — a compounding/drift candidate that does
not flip today. It is also the only lens that resolves the finest transitions. Carry it in
every table next to the median; just don't price on it alone.

### Pick the lens per model
On a new model, choose the statistic with the best split-half reliability **on that model**,
then confirm with the cheap flip bench. Our first model said "trim1 wins, mean last" — and
that did *not* replicate: on a second model the frac-built schedule dominated every budget
while the flip-built one failed badly, because its per-layer prices came from that model's
least-reliable rungs and **the water-fill amplifies lens noise**. On a shallow 10-layer
model every lens tied within 0.5pt — with few layers, order barely matters. Expect the
lens choice to matter most on sensitive, many-layer models. The mean was never best
anywhere.

### The depth-dilution artifact — why deep KLD "improves"

Teacher-forced mean KLD *goes down* as context grows, and it is an artifact:

- The mean is set by sparse catastrophic spikes (>1 nat), and the spike-token fraction
  **falls with depth: 2.3% @2k → 0.96% @32k**. More context = more-constrained predictions
  = ever more teacher-forced-easy filler tokens crowding the average, while the per-token
  median sits flat (~0.0018) in every position band at every depth. Deep windows don't hurt
  less; they contain proportionally more tokens that were never at risk.
- The literature reports the same failure mode under averaged perplexity:
  [**LongPPL**](https://arxiv.org/abs/2410.23771) reports that most long-context tokens are
  context-agnostic in its experiments, making
  averaged PPL correlate ~0 with downstream while key-token loss correlates −0.96. At
  *coarse* scale mean-KLD can still track well
  (["Accuracy is Not All You Need"](https://arxiv.org/abs/2407.09141) reports strong
  correlation across its compressed-model comparisons); the failure studied here is *fine
  discrimination among similar-quality candidates* — the pricing regime.
- **Last-k does not rescue the teacher-forced mean.** The last-64 zone is the *lowest*-KLD
  region (0.09–0.35× the full-window mean) — recency = most context = most constrained —
  even though those queries attend over the most-quantized cache. Neither full-window nor
  last-k teacher-forced KLD can see autoregressive compounding; only autoregressive judges
  (trajectory survival, live-cache generation, task runs) can.

---

## 4. Score where the model actually decodes (last-K / "l64")

`l64` = KLD over only the **last 64 positions** of each window — the true decode frontier —
instead of the whole scored half. *(buun-llama-cpp hook: `TURBO_SCORE_LAST_K=64`.)*

**The experiment that forced it.** We tested protecting the *tail* of KV rows (most recent
positions) at high precision — a "bathtub" allocation: fp16 on attention-sink positions
[0,128) plus the recent tail, cheap tier in the middle — against a control holding the same
byte multiset spread evenly across depth. Matched 0.483 bytes/value, ctx 8192:

| config | B/val | full-window KLD | last-64 KLD |
|---|---|---|---|
| flat t4 (baseline) | 0.516 | 0.0254 | 0.0219 |
| **bathtub** (sink+recent protected) | 0.483 | 0.0540 | **0.0232** |
| spread evenly (same bytes) | 0.483 | 0.0578 | 0.0516 |

Under full-window scoring, bathtub vs spread is **+6.6% — invisible (~1.5σ)** — and bathtub
even "loses" 2.1× to the flat baseline. Under last-64 it beats the spread control by **55%**
and ties the flat baseline with **6.4% fewer bytes**.

Full-window scoring heavily weights positions the model does not decode from in production,
so it can dilute or invert the apparent value of position-targeted protection. **Do not
judge a positional / recency / VBR scheme on full-window KLD alone.**

(The recency wall is soft and wide: protecting the last <64 positions buys nothing; the knee
is ~128–512 recent positions, with ~512 capturing ~80% of recoverable damage.)

Keep the two findings straight: last-K fixes *where you score* (relative comparisons at the
frontier); it does not fix teacher-forcing itself.

---

## 5. Building a layer price panel (what `layer-pricing/` consumes)

**Cell design — paired anchors.** Every cell matches its anchor in ctx, chunks, batch,
flash-attention setting, model file, prompt, and scoring positions, with exactly **one
variable changed**: one (layer, side) moved down one tier. Per-token dump for each cell; the
price is the per-token diff cell − anchor, so the harness floor differences out.

`n_KV_layers × 2 sides × 5 transitions` cells: `fp16-t8`, `t8-t4`, `t4-t3`, `t3-t2`, `t2-t1`.

**Difference-of-medians ≠ median-of-differences.** Printed per-run statistics are triage;
conclusions come from paired per-token dumps.

**Also measure interaction terms.** K+V swapped together is super-additive — 1.61× of
(k+v) at t8→t4, 1.10× at the 1-bit rung. Per-side cells alone *under-price* joint swaps.

**Gates before reading anything:**
1. completeness + return codes;
2. distinctness — bit-identical values between two cells mean a parse failure, not a tie;
3. negative-excess audit;
4. printed-stat vs dump cross-check;
5. determinism vs prior identical cells.

**Acceptance:** split-half reliability (even/odd and first/second-half chunk splits,
Spearman the two half-rankings) — our current heuristic is **publish orders only at
ρ ≳ 0.8**. This is the gate that
killed the mean. `build_degrade_order.py --lens auto` enforces it and refuses to emit an
order from a panel that doesn't replicate.

---

## 6. The checklist

1. **Coherence gate first.** Generate a paragraph and *read it* before trusting any metric.
   Numerically-dead paths can score as "mild degradation" — we had a broken kernel show
   batch-1 PPL 15.6 vs 4.6 while batched metrics looked fine.
2. **Anchor must be exactly zero** (fp16-vs-fp16) before reading any cell.
3. **Paired anchors, per-token dumps.** Every cell against a same-config higher-tier anchor.
4. **Split-half reliability gate**, ρ ≳ 0.8, before publishing any order.
5. **A clean fp16 anchor is necessary, not sufficient, for portability.** Within one engine,
   compare codecs against the same base artifact and identical tokens/scored positions.
   Across builds or engines, first compare their reference logits directly. Separate local
   anchors define different coordinate systems when the reference forward passes differ;
   relative KLD values are not automatically interchangeable.
6. **Measure repeatability for the judge you use.** Deterministic self-comparisons must be
   exactly zero. For stochastic or task-based judges, repeat fixed controls and report the
   observed variation rather than assuming a universal noise floor.
7. **Depth is part of the metric.** Statistics that agree at 2k can invert by 32k. Validate
   at deployment depth.
8. **Single draws are coin flips.** Pass/fail deltas on one seed regress to the mean on
   reseed — we "recovered" 6 hard problems with a better codec, and 4 re-passed under the
   *old* codec with a new seed. Paired per-case statistics on a fixed case set, or
   multi-seed.
