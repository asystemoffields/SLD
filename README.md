# SLD — Speculative Looped Decoding

**Lossless acceleration of looped / recurrent-depth transformers by transplanting
speculative decoding from the token axis to the depth axis.**

A looped transformer reasons by running a shared **core** block many times —
`h_{t+1} = core(h_t + e)` — which is powerful but **sequential**: a depth-`k`
answer costs `k` core calls, one after another. SLD makes those calls (mostly)
parallel. A cheap **draft** proposes a whole *trajectory* of future loop states;
the true core **verifies all of them in one batched pass**; SLD **accepts the
longest prefix** the true core agrees with and continues from there. A depth-`k`
computation collapses from `k` sequential rounds to `ceil(k / accepted)` — `O(1)`
when the draft is good — and the answer is **identical to the full loop**.

This is the lossless accept-longest-prefix protocol of speculative / blockwise
parallel decoding (Stern 2018; Leviathan, Chen 2023), applied for the first time
to a single token's **depth/loop trajectory** on a **frozen** looped transformer.
See [`SLD_SPEC.md`](SLD_SPEC.md) for the method, the precise losslessness
condition, prior-art positioning, and the falsification bar.

## Result in one table

Fixed-permutation pointer-chasing (`answer = pi^k(start)`), a 0.3M-param looped
teacher trained from scratch on CPU (100% exact match), depth `k` swept. **Every
row is lossless vs the full loop, asserted byte-for-byte.**

| k | full-loop rounds | early-exit rounds | **SLD rounds** | mean accept | wall-clock speedup @ batch-1 |
|--:|--:|--:|--:|--:|--:|
| 2  | 2  | 1.87  | **1.00** | 2.00  | 0.60× |
| 4  | 4  | 3.80  | **1.00** | 4.00  | 0.86× |
| 8  | 8  | 7.70  | **1.00** | 8.00  | 1.28× |
| 12 | 12 | 11.08 | **1.00** | 12.00 | 1.67× |
| 16 | 16 | 15.12 | **1.00** | 16.00 | **2.07×** |

```
sequential core rounds vs depth k        (full=#   SLD=O)
k= 2 |OO##........................   full=2   SLD=1
k= 4 |OO#####.....................   full=4   SLD=1
k= 8 |OO############..............   full=8   SLD=1
k=12 |OO###################.......   full=12  SLD=1
k=16 |OO##########################   full=16  SLD=1
```

The headline: **full-loop sequential core rounds grow linearly with depth `k`;
SLD stays at one round** (a single batched verification swallows the whole
trajectory), while remaining exactly lossless. The batch-1 wall-clock speedup
grows monotonically with `k`, crossing 1× near `k=6` and reaching **2.07× at
`k=16`** (SLD is ~constant at ~1.6 ms; the full loop grows to 3.3 ms). Early-exit
cannot help on this advancing (non-converging) loop — only a draft that
*predicts where the loop will be* can skip ahead, which is exactly what SLD
verifies and exploits.

**Why lossless matters (length generalization).** Push `k` past the draft's
horizon and the draft goes out-of-distribution. The lossy one-shot jump
collapses to chance; SLD stays exactly lossless by spending one more verified
round:

| k | in draft horizon? | draft-only acc (lossy) | **SLD acc** | SLD rounds |
|--:|:--:|--:|--:|--:|
| 16 | yes | 1.000 | **1.000** | 1.0 |
| 18 | **no (OOD)** | 0.029 | **1.000** | 2.0 |
| 20 | **no (OOD)** | 0.035 | **1.000** | 2.0 |

## Install & run

SLD builds on the [`jumprec`](https://github.com/asystemoffields/jumprec)
substrate (the looped transformer + recurrence task). Either install it, or keep
a sibling `../SMOKE` checkout / set `JUMPREC_PATH` (see `sld/substrate.py`).

```bash
# CPU torch + numpy; jumprec on the path (pip install -e ../SMOKE, or PYTHONPATH)
pip install -e .
PYTHONPATH=../SMOKE python -m pytest tests/ -q          # losslessness invariants
PYTHONPATH=../SMOKE:. python bench/experiment.py --tag main   # the full frontier
```

The experiment trains (and caches) a teacher, a learned draft, and the original
JumpRec jump baseline, then writes the depth sweep, horizon sweep, baselines, and
wall-clock numbers to `results/frontier_main.json`.

## What's here

```
sld/
  substrate.py   bridge to the jumprec looped-transformer substrate
  draft.py       LearnedDraft, AndersonDraft (training-free), Identity/Blind controls, JumpModule
  specloop.py    sld_decode + full-loop / early-exit / confidence-jump baselines
  training.py    distill the draft on the frozen teacher's trajectory tape
bench/
  experiment.py  the frontier experiment (depth, horizon, baselines, wall-clock)
  common.py      checkpoint + timing helpers
tests/           losslessness + control invariants
SLD_SPEC.md      the method, losslessness proof sketch, prior art, falsification bar
```

## Honest scope

- Losslessness is rigorous for **discrete-readout** recurrence (we verify on the
  argmax answer, which is bit-stable); we assert it holds 100% across all `k`.
- The wall-clock win is a **batch-1 latency** result on a **compact** core, where
  batched verification stays in the CPU "flat" region. The hardware-free
  **counted core rounds** metric is where the result is airtight; on parallel
  hardware those rounds are the latency. At large serving batch the win shrinks —
  reported honestly in `results/`.

## Results

Full numbers in `results/frontier_main.json`; regenerate the tables with
`python bench/summarize.py main`. Highlights below (32 symbols, teacher 0.3M
params @ 100% exact match, draft 0.14M params, horizon 16, 6 threads).

**Horizon sweep (k=16)** — rounds are exactly `ceil(k/H)`, FLOP-matched (16 core
rows either way), all lossless:

| horizon H | SLD rounds | core rows/example | mean accept |
|--:|--:|--:|--:|
| 1 | 16.00 | 16.0 | 1.00 |
| 2 | 8.00  | 16.0 | 2.00 |
| 4 | 4.00  | 16.0 | 4.00 |
| 8 | 2.00  | 16.0 | 8.00 |
| 16 | 1.00 | 16.0 | 16.00 |

**Controls (all lossless)** — isolate the source of the win. No-draft (identity)
and blind drafts save nothing (rounds ≈ k); the training-free Anderson draft is
weak on the advancing phase (≈0.8·k rounds), exactly as a fixed-point
extrapolator should be when there is no fixed point to chase yet. Only the
learned draft collapses rounds to 1.

**Wall-clock.** Batch-1 latency speedup grows with `k` (0.60× → **2.07×** at
k=16) — the round-count win shows through once the saved core calls outweigh the
per-round overhead. Batch-64 throughput does **not** win (0.6–0.7×): the verify
batch `B·H` leaves the CPU flat region and the extra readout pass costs ~1×. This
is the expected latency-not-throughput / discrete-readout scope; the
hardware-free **counted-core-rounds** result (constant vs linear in `k`) is where
the contribution is airtight — on parallel hardware those rounds *are* the
latency.

## License

MIT.
