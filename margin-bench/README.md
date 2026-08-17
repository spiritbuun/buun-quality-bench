# Margin bench — the task-grounded arbiter

This is what we use to decide whether a KV codec change is actually *good*, alongside (not
instead of) KLD. We built it after KLD burned us several times. If you are staring at KLD
tables wondering why your "wins" don't feel like wins, this is the way out we found.

## The four traps (each cost us real time)

**Trap 1 — KLD is fidelity-to-fp16, not goodness.** A codec change that *routes around*
fp16's own quirks raises KLD while improving real outcomes. We have a weight-quant config
that beats BF16 on a maths benchmark (26/30 vs 18/30) with *worse* KLD. Gate on KLD alone
and you will reject genuinely good changes. The final arbiter must be task-grounded.

**Trap 2 — mean KLD is tail-dominated.** The mean is dragged by a few catastrophic
positions; bulk improvements vanish into it. One of our best changes (a V-mean subtraction
tap) moved mean KLD by ~1% — t ≈ 0.3, pure noise — while tripling worst-case task margins
(t = +8.5, 117-vs-3 paired wins). A mean-KLD gate would have rejected it. If you must use
KLD: paired per-position differencing, and read median / p99 / last-window separately.

**Trap 3 — exact-match saturates.** A capable model aces a task across every codec and
depth (we scored 120/120 on every config — zero discrimination). The signal is not
*whether* the model answers correctly, it is **how close it came to flipping**. That is
what logprob margins measure.

**Trap 4 — single draws are coin flips.** Pass/fail deltas on one seed regress to the mean
on reseed: we "recovered" 6 hard problems with a better codec, and 4 of them re-passed
under the *old* codec with a new seed. Use paired per-case statistics on a fixed case set,
or multi-seed. Never trust one draw.

## What it does

Synthetic long-context **routing cases**: an evidence package (action / target /
source-rank) is buried at depth behind tunable noise and confusable distractors, so the
model must read it back *through your quantized KV cache*. Greedy decode, fixed cases.

For every answer token it records `logprob(chosen) − logprob(runner-up)`; a case's score is
the **minimum margin** across its answer tokens — a calibrated distance-to-flip. Comparing
two configs is then paired per-case differences on identical cases → a t-statistic.

In our original 4B campaigns it separated several KV codecs at depths where exact match
was saturated. That effect size is model- and case-set-specific; calibrate the supplied
cases on your own target before drawing a conclusion.

## Quickstart

1. Serve your config — one slot, no batching, so runs stay deterministic:

```bash
llama-server -m model.gguf -ngl 999 -c 12288 \
  -ctk <your_kv_type> -ctv <your_kv_type> -fa on -np 1 \
  --host 127.0.0.1 --port 8099 --jinja
```

2. Probe it (~120 cases, sequential, greedy):

```bash
python3 router_probe/probe_router.py \
  --data router_probe/cases/rd_8192_c2.jsonl \
  --out lp_myconfig.jsonl --label myconfig \
  --config-id 'build=<sha>;model=<file hash>;flags=-ctk ... -ctv ... -fa on'
```

3. Repeat for your baseline (`lp_baseline.jsonl`), then compare:

```bash
python3 paired_margins.py lp_myconfig.jsonl lp_baseline.jsonl
```

You get: exact task outcomes first, then the mean paired margin delta, a bootstrap interval,
a descriptive t-statistic, per-case win counts, and the closest-to-flip correct cases.

## Reading the output

- Treat a bootstrap interval that excludes zero, a lopsided paired win count, and a stable
  rerun as converging evidence. The printed t-statistic is descriptive; this case set is
  synthetic and was not randomly sampled from a deployment population.
- **minA / minB** (worst-case margin) is the number that predicts production failures. A
  config whose worst case sits at 0.3 nats is one unlucky sample from a flip; 2.5 nats is
  comfortable. We weight this over the mean.
- Margins are **model-relative**: compare configs on the same model, same build, same cases.
  Never compare absolute margins across models.

## Notes

- Cases: `rd_2048_c2` / `rd_8192_c2` / `rd_32768_c2` (depth sweep; `c2` = confusability
  tier). `gen_router_cases.py` regenerates them or makes harder tiers if your model
  saturates these — **fix the seed and reuse the same file for every config you ever
  compare.**
- `depth_tokens` is a character-budget approximation, not a tokenizer guarantee. The probe
  records the server's actual `prompt_tokens`; verify that distribution before describing a
  panel by depth, especially across tokenizers or chat templates.
- The probe sends `chat_template_kwargs: {"enable_thinking": false}` (Qwen-style).
  Other servers or templates may ignore or reject it; adjust the request builder and record
  that change in `--config-id` when it is not appropriate.
- The `.meta.json` beside each output binds resume to the case file, endpoint parameters,
  label, and your required `--config-id`. Put the build, model artifact, and complete
  quality-affecting server flags in that ID. The endpoint cannot discover those flags for
  you.
- 32k cases need server `-c` ≳ 36000.

## Credit

Case design is a depth/confusability extension of the kv-score router probes by **sztlink**
in [TurboQuant discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969).
