# Patches

Small additions to `llama-perplexity` that some of these tools depend on. Everything else
in this suite runs on stock upstream; these two hooks are the only things that don't.

The core environment hooks are present in
[buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp). This patch also adds the
dump-metadata sidecar used by the public reducers; builds predating that sidecar can only
use last-K artifacts with the explicit `--allow-legacy-dumps` verification bypass.

## `kld-dump-hooks.patch`

Three hunks, all in `tools/perplexity/perplexity.cpp`, adding two environment-gated hooks.
Both are inert when the variables are unset, so a patched binary behaves exactly like an
unpatched one by default.

```bash
cd /path/to/llama.cpp
git apply /path/to/patches/kld-dump-hooks.patch
cmake --build build --target llama-perplexity -j
```

Verified to apply cleanly against upstream `34af94cd9ab277632e27caeec2d41de2fd091b31`
on 2026-08-17. If upstream drifts, the
three anchors are easy to re-find: the `int counter = 0;` in `process_logits`, the
`lock.unlock();` in the same worker loop, and the `if (kld.count < 100) return;` guard in
the KL-divergence reporting path.

### `TURBO_KLD_DUMP=<path>` — per-token KLD dump

Writes the per-position KLD array **before** it is sorted for percentiles, so two runs
against the same base and token stream can be differenced position-by-position offline.

This is the important one: it uses **common random numbers**. Shared per-token difficulty
cancels in the paired difference, so the paired-delta standard error is far below
`sqrt(SE_cand² + SE_base²)`. It also turns a single run into every statistic offline —
median, trimmed means, `frac>τ`, positional buckets, split-half reliability — without
re-running the model.

Binary layout:

```
int32   n_pos_per_chunk
int32   n_chunk
float32 kld[n_chunk * n_pos_per_chunk]     // chunk-major
```

The hook also writes `<path>.meta`, recording the shape and `score_last_k` value. Offline
tools reject a last-K dump without matching metadata unless you explicitly pass
`--allow-legacy-dumps` after manually verifying an older artifact.

Read it with e.g.:

```python
import numpy as np
with open(path, "rb") as f:
    n_pos, n_chunk = np.fromfile(f, dtype=np.int32, count=2)
    kld = np.fromfile(f, dtype=np.float32).reshape(n_chunk, n_pos)
```

**Required by `layer-pricing/`** — the per-(layer, side) price panel is built from these
dumps. Also what `hazard-bench/hazard_metrics.py` consumes in its offline mode.

### `TURBO_SCORE_LAST_K=N` — score at the decode frontier

Scores only the last `N` positions of each window instead of the whole scored half.
`TURBO_SCORE_LAST_ONLY=1` is shorthand for `N=1`.

Full-window scoring heavily weights positions the model does not decode from in production,
so it can dilute or invert position-targeted protection — sink/recency schemes,
variable-bit-rate caches, anything that spends bits unevenly across positions. See
`../METHODOLOGY.md` §4 for the experiment that forced this into existence: a config that
looks like a 2.1× *loss* under full-window scoring is a tie-at-6.4%-fewer-bytes at the
frontier.

Skipped positions are zero-filled in the raw array. The online mean over `kld.count` stays
exact, but online percentiles are meaningless in this mode. The offline reducers require
the matching `--last-k N` and slice away the zero-filled prefix.
