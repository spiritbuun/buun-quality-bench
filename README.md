# buun-quality-bench

Four harnesses for measuring **what a change does to a model's output quality** in
llama.cpp — plus the measurement lessons we paid for getting them right.

They were built for KV-cache quantization, and several drivers deliberately expose
KV-specific controls. The measurement principles also apply to weight quantization,
fine-tunes, sampler changes, and kernel rewrites, but those uses need adapters rather than
being drop-in promises.

We wrote these after several rounds of confidently reaching the wrong conclusion.
**Read `METHODOLOGY.md` before trusting any number these tools print** — especially
anything that is a mean.

**Ground rule (rule 0):** everything here except the margin bench measures *fidelity to a
reference model*, not goodness. A change can beat the reference on a downstream task while
scoring worse on fidelity. Direct task outcomes answer goodness for that task; the rest are
cheap, dense proxies. Always say which one you mean in a verdict.

The basic model runs use upstream llama.cpp APIs and tools. Exact raw-anchor certification,
last-K scoring, and layer-pricing need the small per-token dump hooks in `patches/`. The
core hooks also exist in [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp),
with the sidecar-version caveat documented in `patches/README.md`.

---

## The four tools

| | what it answers | cost | needs |
|---|---|---|---|
| **`kld-panel/`** | "How far from the reference, across configs × depths?" | minutes–hours | upstream; patch for exact raw anchor |
| **`hazard-bench/`** | "Will this change flip decisions?" | minutes | upstream (offline variant: per-token dumps) |
| **`margin-bench/`** | "Is this change actually *good*?" (task-grounded) | minutes per config | upstream server |
| **`layer-pricing/`** | "Which layer should give up precision first?" | consumes KLD-panel output | per-token dumps |

They form a ladder of cost and of authority. The KLD panel is the workhorse you run
constantly; the margin bench is the arbiter you run before believing a win.

### 1. `kld-panel/` — teacher-forced KL divergence vs a reference
The workhorse. A `{config} × {context depth}` matrix wrapping
`llama-perplexity --kl-divergence`. Resumable; a bad cell does not stop later cells, but
the campaign exits nonzero until every cell succeeds. Gives you
mean/median/p99/max KLD, RMS Δp, and a free true-flip rate (`Same top p`) per cell.
Judge on the **median**. Start here.

### 2. `hazard-bench/` — dense per-token decision-risk panel
Answers the flip question directly. For each scored position it computes KL per unit of
decision margin (`R = KL / (½·margin²)`) and margin erosion — graded signals that can be
more informative than a saturated pass/fail panel. Ships as a
standalone tool using only the public `llama.h` API, so it drops into any llama.cpp tree.

### 3. `margin-bench/` — task-grounded arbiter
The one that measures *goodness*. Synthetic long-context routing cases: an evidence package
buried at depth behind confusable distractors, which the model must read back through
whatever you changed. Scores the **minimum logprob margin** per case — calibrated
distance-to-flip — and compares configs by paired per-case differences. Treat margins as
secondary diagnostics after exact task outcomes, not as task accuracy.

### 4. `layer-pricing/` — build a measured degrade order
For variable-bit-rate schemes that demote layers individually. Takes a per-(layer, side)
price panel and emits the order in which layers should give up precision — cheapest quality
loss per byte saved, first. **Depends on the KLD panel**: prices come from per-token dumps
of one-tier-at-a-time cells.

---

## Which do I use?

- **Ranking configs / gating a build** → KLD panel, judged on median.
- **"Does it flip decisions?"** → hazard bench, or the free `Same top p` already in every
  KLD cell log.
- **"Is the win real?"** → margin bench. Do not ship on fidelity metrics alone.
- **Position-targeted schemes** (sink/recency protection, VBR, anything spending bits
  unevenly across positions) → include decode-frontier scoring (last-K; see
  `METHODOLOGY.md` §4). Full-window scoring can dilute or invert the effect.
- **Fine transitions near the noise floor** → catastrophe *fractions* from paired dumps,
  not any magnitude statistic.
- **Per-layer bit allocation** → layer-pricing, fed by KLD-panel dumps.

---

## Quickstart

Requires Python 3.10+, NumPy, Bash, and a llama.cpp build. Install the Python dependency
into your preferred environment with `python -m pip install -r requirements.txt`.

```bash
# 0. one-time: get a corpus
bash /path/to/llama.cpp/scripts/get-wikitext-2.sh

# 1. generate a reference logit base (once per model × depth)
mkdir -p bases
llama-perplexity -m model.gguf -f wiki.test.raw \
  -ctk f16 -ctv f16 -fa on -ngl 99 -c 16384 --chunks 18 \
  --save-all-logits bases/base_f16kv_ctx16384_18ch.kld

# 2. sweep candidates against it (the fp16/fp16 exact-zero anchor is automatic)
BIN_DIR=/path/to/build/bin MODEL=model.gguf DATASET=wiki.test.raw \
BASE_DIR=./bases TYPES="q8_0 q4_0" \
KV_TIERS='16384:base_f16kv_ctx16384_18ch.kld:18' \
  ./kld-panel/kv_kld_sweep.sh ./run1

# 3. confirm a win is real before believing it
#    (serve each config, probe, then compare — see margin-bench/README.md)
python3 margin-bench/paired_margins.py lp_candidate.jsonl lp_baseline.jsonl
```

**Before reading any number:** run the anchor. A reference-vs-reference cell must score
exactly zero (hazard: `flip_rate = mean_R = 0.0000`). If it doesn't, the instrument is
broken — fix that first. This has caught more real problems for us than any single metric.

---

## Repo layout

```
README.md              this file — what each tool is, which to reach for
METHODOLOGY.md         how to measure without fooling yourself; read before use
LICENSE                MIT
patches/               the two llama-perplexity hooks, for non-buun builds
kld-panel/             KLD matrix driver + offline statistics
hazard-bench/          standalone flip/decision-risk tool + offline variant
margin-bench/          task-grounded routing probe + paired statistics
layer-pricing/         degrade-order builder + panel schema examples
```

## Status

Shared as-is, in the hope it's useful. Issues and PRs are welcome but unsupported — we
maintain this against our own needs. If you extend it (other architectures, other codecs,
a better arbiter), we'd genuinely like to hear about it.

## Provenance & credit

Extracted from the [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) eval
stack. The true-flip grounding comes from llama.cpp's own `llama-perplexity`
(`Same top p`); the trajectory/flip framing traces to
[TurboQuant discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
(contributor **sztlink**'s `trajectory` metric — percentage of greedy steps whose
argmax matches the reference), and the routing-case design is a depth/confusability
extension of the same author's kv-score router probes. The second-order flip proxy
`R = KL/(½·margin²)`, the reliability gating, and the pricing water-fill are ours.

Reference numbers quoted throughout were measured on Qwen3.6-class models on consumer
NVIDIA hardware. They illustrate *shape*, not targets — margins and KLD are model-relative
and must be re-measured per model.

## License

MIT — see `LICENSE`.
