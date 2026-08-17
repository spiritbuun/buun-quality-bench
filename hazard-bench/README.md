# Hazard bench

Dense per-token flip / decision-risk panel vs fp16, teacher-forced.

Complementary to the KLD panel. In the campaigns described in `../METHODOLOGY.md`, its
aggregate ranking tracked true flips well; that is empirical calibration, not a guarantee
for a new model. It is a dense central statistic rather than a first-divergence measure.

## Files

- **`frontier-hazard.cpp`** + **`CMakeLists.txt`** — standalone tool. Uses only the public
  `llama.h` API, so it drops into **any** llama.cpp tree: copy this folder to
  `examples/frontier-hazard/`, add `add_subdirectory(frontier-hazard)` to
  `examples/CMakeLists.txt`, build target `llama-frontier-hazard`.
- **`hazard_metrics.py`** — offline approximation. Computes the `R` normalization from artifacts you
  already have (the base logits file + per-cell per-token dumps), no extra GPU runs. Also
  contains the streaming margin extractor (~7 min for a 32 GB base). *Needs buun-llama-cpp per-token dumps.*

## How it works

Loads the model once, builds two contexts — reference fp16/fp16 KV and the quantized KV
under test — and feeds both the **same real token prefix**, requesting logits at scored
positions. One `llama_decode`, no autoregression: per-step error never compounds into a
one-shot flip, which keeps the signal graded where autoregressive harnesses saturate.

Per scored position (fp16 distribution `P`, quant `Q`, full vocab):

```
a       = argmax P                        # fp16 top-1
F_t     = [argmax Q != a]                 # top-1 flip
margin  = p[a] - p[2nd]                   # fp16 decision margin (probability)
KL      = sum_v p_v (log p_v - log q_v)
R_t     = KL / (0.5*margin^2 + eps)       # decision danger: KL per unit of margin
L_t     = (gapP - gapQ) / (gapP + eps)    # margin erosion; >=1 means the fp16 margin is erased
```

Per prompt it reports `flip_rate, mean_R, cvar95_R` (mean of the worst 5%), `mean_L`,
`frac_Lge1`, `mean_KL`, plus depth-band aggregates (0-128 / 128-512 / 512-2k / 2k-8k / 8k+)
for a hazard-vs-depth curve.

`mean_L` is noisy — **`frac_Lge1` is the usable margin-erosion statistic.**

The offline reducer is not numerically interchangeable with this executable. Its KLD dump
comes from `llama-perplexity`'s quantized log-probability base and probability cutoff,
whereas the executable compares two live full-vocabulary logit arrays. Offline
`risk_crossing_excess` is the change in the fraction of positions where `R >= 1`; it is a
necessary-risk threshold, not an observed argmax flip. Use the executable's `flip_rate`
when actual teacher-forced flips are the question.

Run the offline reducer with an explicit provenance identity:

```bash
./hazard_metrics.py --base base_f16kv.kld --margins margins.f32 \
  --dumps ./dumps --tag mymodel \
  --campaign-id 'build=<sha>;model=<file hash>;ctx=...;chunks=...;fa=...' \
  --out ./hazard_out/mymodel
```

It writes cells, reliability, and a metadata JSON containing SHA-256 identities for the
base, dump inputs, and output tables.

## Run

```bash
./build/bin/llama-frontier-hazard -m model.gguf -f prompts.txt \
  -ctk q4_0 -ctv q4_0 -ngl 99 --n-prefix 8192 --n-score 256 --max-prompts 128
```

Each nonempty line in `prompts.txt` is one prompt and must tokenize to at least
`n-prefix` tokens. The tool reports short/tokenization/decode failures and exits
nonzero if any selected prompt cannot be scored; do not mistake a shallow smoke corpus for
a long-context validation. `--allow-shallow` is an explicit smoke-test escape hatch.

**Always run the anchor first.** `-ctk f16 -ctv f16` must print all zeros
(`flip_rate = mean_R = ... = 0.0000` exactly). If it does not, something is broken — fix
that before reading any codec number. Anchor mode also compares every returned float logit
bit-for-bit and exits nonzero on the first mismatch; the printed decimal precision is not
treated as a tolerance.

Illustrative ladder (128 wikitext prefixes, `n_prefix=128`) — shape, not targets:

| codec | flip_rate | mean_R | frac_Lge1 | mean_KL |
|---|---|---|---|---|
| f16 (anchor) | 0.0000 | 0.00 | 0.0000 | 0.00000 |
| q8_0 | 0.0115 | 6.3 | 0.0113 | 0.00041 |
| q4_0 | 0.0261 | 40.5 | 0.0251 | 0.00261 |

## Provenance

True flip-rate grounding comes from llama.cpp's own `llama-perplexity` (`Same top p`); the
trajectory/flip framing traces to
[TurboQuant discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
(contributor **sztlink**'s `trajectory` metric — percentage of greedy steps whose argmax
matches fp16).
`R_t = KL/(½·margin²)` is our decision-risk proxy layered on top; in our layer-pricing
campaign its rank correlation with observed argmax flips ranged from +0.84 to +0.96.
