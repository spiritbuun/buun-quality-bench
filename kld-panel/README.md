# KLD panel

Teacher-forced KL divergence vs an fp16-KV logit base, over a `{KV type} × {context depth}`
matrix. The workhorse — this is what you run constantly.

Wraps `llama-perplexity --kl-divergence`, which is **upstream**: the basic panel needs no
fork. The per-token dump hooks (needed for `layer-pricing/`) are buun-llama-cpp-only and marked below.

> Judge on the **median**. Read `../METHODOLOGY.md` §2 before using the mean for anything.

## Files

- **`kv_kld_sweep.sh`** — the matrix driver. Resumable (OK cells skipped on re-run);
  failures are recorded and the remaining cells continue, but the final exit status stays
  nonzero until all cells succeed. A manifest refuses cross-campaign resume. Emits
  `results.tsv` + `results.md` per run.
- **`kv_common.sh`** — shared config/lib, sourced by the driver; keep them together. Every
  knob is env-overridable: `BIN_DIR`, `MODEL`, `DATASET`, `BASE_DIR`, `KV_TIERS`, `TYPES`,
  `KTYPE` (pin K, sweep V only), `CELL_TIMEOUT`, `FA`, `NGL`.
- **`parse_kld.py`** — per-depth panel + leaderboards from a directory of per-cell logs.
  It infers arbitrary candidate names and depths; run `./parse_kld.py --help` for custom
  filename patterns and explicit depth requirements. Prints per candidate per depth:
  `meanKLD  median  p95  p99  p99.9  maxKLD  RMSdp%  flip%`.
- **`validate_kld_dump.py`** — validates dump shape/finiteness and can require every raw
  KLD value to equal zero exactly.

## Step 1 — generate logit bases (once per model × depth)

Run with **fp16 KV** and save the full logit distribution per scored token.
`--save-all-logits` and `--kl-divergence-base` are the same upstream flag: without
`--kl-divergence` it *writes* the base, with it, it *reads* it.

```bash
mkdir -p bases
llama-perplexity -m model.gguf -f wiki.test.raw \
  -ctk f16 -ctv f16 -fa on -ngl 99 \
  -c 16384 --chunks 18 \
  --save-all-logits bases/base_f16kv_ctx16384_18ch.kld
```

Repeat per depth tier. A reasonable ladder: 2048/32ch, 8192/24ch, 16384/18ch, 32768/9ch.

The base file also carries the token stream, and the comparison reads those stored tokens.
The sweep still records the supplied corpus identity as provenance and refuses to resume if
it changes. Bases are large (tens of GB at depth); plan disk before starting.

## Step 2 — run the sweep

```bash
BIN_DIR=/path/to/build/bin \
MODEL=/path/model.gguf \
DATASET=/path/wiki.test.raw \
BASE_DIR=./bases \
TYPES="q8_0 q4_0" \
  ./kv_kld_sweep.sh ./run1            # or --shallow for the first tier only
```

By default `TYPES` auto-detects the build's *custom* KV types (anything `--help` advertises
beyond the standard set); on stock upstream, set `TYPES` explicitly.

Built in: an **effective-BPW probe** that reads the bytes each type actually allocates
(calibrated against fp16) and flags silent K/V substitutions. Trust it over name-based bpw
tables — a type that quietly falls back costs nothing and looks free. Probe failure stops
the campaign by default; `ALLOW_BPW_PROBE_FAILURE=1` is an explicit diagnostic bypass.

The matrix driver intentionally rejects inherited `TURBO_SCORE_LAST_K`,
`TURBO_SCORE_LAST_ONLY`, and `TURBO_KLD_DUMP`: its online percentile columns are a
full-window report, and a fixed dump path would be overwritten cell by cell. Use uniquely
named hook dumps with the offline reducers for frontier or layer-pricing campaigns.

## What each cell reports

From the `llama-perplexity` output: Mean KLD (± standard error), Median KLD, 99.9% KLD, Max KLD, PPL(Q),
ln(PPL ratio), RMS Δp, **Same top p**, seconds.

`Same top p` is a free true-flip rate: **flip% = 100 − same-top-p**. It is the closest thing
here to ground truth, and it costs nothing.

Only the **deep half of each window is scored** (`n_pos = n_ctx/2 − 1`, e.g. 8191 at 16k) —
bake that into any token-count arithmetic.

## buun-llama-cpp hooks

- **`TURBO_KLD_DUMP=<path>`** — per-token KLD dump (see `../patches/`)
  (`i32 n_pos, i32 n_chunk, f32[chunk][pos]`) plus a `.meta` sidecar. Turns one run into every statistic offline:
  median, trimmed means, `frac>τ`, positional buckets, split-half reliability. **This is
  what `layer-pricing/` consumes.**
- **`TURBO_SCORE_LAST_K=64`** — score only the last K positions per window, i.e. the true
  decode frontier. Required for judging any position-targeted scheme; see
  `../METHODOLOGY.md` §4 for the experiment that forced it into existence.

## Before you read anything

The sweep runs an fp16-vs-fp16 cell at every depth automatically and asks the dump hook for
its raw per-token KLD values. Every raw float must equal zero; otherwise the cell is
`ANCHORFAIL`. If the build lacks the hook, the default is `ANCHORUNVERIFIED` rather than
pretending six-decimal summary output proves exact equality. `REQUIRE_RAW_ANCHOR=0` permits
an upstream-only smoke run, but does not relax the methodological rule or certify the
anchor. Do not introduce a numeric tolerance: a nonzero self-comparison indicates
nondeterminism, misalignment, or corrupted artifacts that must be fixed.
