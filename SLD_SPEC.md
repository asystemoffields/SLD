# Speculative Looped Decoding (SLD)

## The one-paragraph idea

Looped / recurrent-depth transformers (Universal Transformer, parcae, Huginn)
get extra reasoning depth by running a shared **core** block many times:
`h_{t+1} = core(h_t + e)`. The cost is that those iterations are **sequential** —
you pay one core call per loop step even when the computation is predictable.
SLD removes the sequentiality by transplanting **speculative decoding** from the
token axis to the **depth/loop axis**: a cheap **draft** proposes a whole
*trajectory* of future loop states `g_1..g_H`; the true core **verifies all of
them in one batched pass**; we **accept the longest prefix** on which the draft
is self-consistent with the true core, and continue from the verified state. A
depth-`k` looped computation that costs `k` sequential core rounds collapses to
`ceil(k / accepted)` rounds — `O(1)` when the draft is good — **losslessly**.

## Why this is new

The acceptance protocol (draft a block, accept the longest validated prefix,
take the first true output past it as a free correction) is exactly
Blockwise Parallel Decoding (Stern et al. 2018) / speculative decoding
(Leviathan, Chen et al. 2023) — but those operate on the **token** sequence.
Fixed-point acceleration of implicit models (DEQ + Anderson/Broyden, Bai et al.
2019; HyperSolver, ICLR 2022) accelerates a **root-solve**, not a pretrained
looped transformer, and has no verify/accept-prefix step. Recent depth-axis
parallelism (StagFormer, Cross-Loop Parallelism 2025) **retrains** a new
architecture and overlaps loops **across different tokens**. The closest
recurrent-depth work, "Efficient Parallel Samplers for Recurrent-Depth Models"
(2025), parallelizes **diagonally across token positions** and is **lossy**
(~5× at ~1% quality loss).

**SLD is, to our knowledge, the first method that speculatively drafts and
verifies a single token's depth/loop trajectory on a *frozen* looped
transformer, in one batched true-core pass, with *lossless* longest-prefix
acceptance.** What is genuinely ours is (1) the axis transposition to one token's
depth trajectory, (2) the lossless **discrete-readout** acceptance test that
makes one batched verification "free" on the depth axis, and (3) a
training-free fixed-point-extrapolation draft variant for any pretrained loop.
The verification *algorithm* is borrowed and cited, not reinvented.

## The method

Notation: `encode` runs the prelude once to produce `h_0`; `step(h) = core(h + e)`
is one time-invariant loop iteration with prompt reinjection `e`; `readout(h)`
is the model's argmax answer head (a discrete projection).

One SLD round, from the current verified true state `h` at loop step `t`:

1. **Draft** `g_1..g_H = draft(h)` — guesses of `h_{t+1..t+H}`.
2. **Verify (one batched core call)** `u_i = step(g_i)` for the window, with
   `g_0 := h` so `u_0 = step(h)` is the *exact* next true state. All `H` calls
   are a single batched forward (batch dimension `H`).
3. **Accept longest prefix.** Accept `g_{i+1}` iff
   `readout(g_{i+1}) == readout(u_i)` and `g_i` was accepted. Let `a` be the
   accepted length.
4. **Advance** `a+1` steps: carry `u_a` (whose readout equals the true symbol at
   step `t+a+1`), set `t ← t + a + 1`. The first unverified true output `u_a` is
   the free correction (the "bonus token" analog).
5. Repeat until `t` reaches the target depth `k`.

If the draft is useless, `a = 0` every round and SLD degrades **exactly** to the
full loop (one true step per round) — never worse, always lossless.

## Why it is lossless (and the precise condition)

Acceptance is on the **discrete readout**, not the continuous hidden state. This
matters: comparing continuous states would be fragile to BLAS reduction-order
noise (batched-verify vs serial differ by ULPs) and would force a tolerance that
**amplifies along the depth axis**. The readout (argmax symbol) is bit-stable, so
`tol = 0` is meaningful.

On a **symbolic recurrence** task the readout is **Markov**:
`readout(step(s))` depends only on `readout(s)` (advancing one hop is
`symbol → pi(symbol)`). Then every state SLD advances to has a readout equal to
the true iterate's readout *exactly*, regardless of continuous drift in the
nuisance dimensions, so the final answer is identical to the full loop. We
**assert** this byte-for-byte in `bench/experiment.py` and the test suite (it
holds at 100% across all `k` and seeds).

Honest scope (what the red-team flagged and we respect):
- Losslessness is rigorous for **discrete-readout** recurrence; on a fully
  continuous-state loop SLD is "verified to a tolerance" (lossy in kind, like the
  original confidence verifier). We benchmark the discrete-readout regime.
- The wall-clock win lives in the **batch-1 latency** regime on a **compact**
  core, where a batched verify of `H` states stays in the CPU "flat" region
  (measured: verifying 17 states ≈ 1.3× one call at d=96, 6 threads). At large
  serving batch `B`, the verify batch `B·H` leaves the flat region and the win
  shrinks — we report this honestly. The **counted-core-rounds** metric is
  hardware-free and is where the result is airtight; on parallel hardware (the
  real home of looped LLMs) those rounds are the latency.

## Baselines (all share the frozen teacher, seeds, and data)

- **Full loop** — `k` sequential core calls; the lossless reference.
- **Early exit / convergence halt** — lossless but cannot skip the *advancing*
  phase; on a non-converging loop it offers nothing (the point).
- **Original JumpRec** — predict the final state in one jump + a confidence
  verifier; **lossy** (quality trades with the threshold). SLD must dominate it.
- **No-draft control** (identity draft) — isolates the draft from the verify
  trick; should accept ~0 and save nothing.
- **Blind-draft control** (random) — confirms wins come from prediction; must
  still be lossless.
- **Training-free Anderson draft** — works on any frozen loop; expected weak on
  the advancing phase (a fixed-point extrapolator has no fixed point to chase
  there). The "works on any pretrained loop" control.
- **Parallel-scan oracle** — `ceil(log2 k)` rounds, the associativity ceiling.

## Falsification bar (must all survive)

1. Lossless byte-for-byte vs the full loop, every `k`.
2. Mean accepted-prefix length `> 1` and **rising** with `k`.
3. Speedup (rounds and wall-clock) **monotone increasing** in `k`.
4. Wall-clock agrees with core-rounds at the calibrated compact size.
5. Pareto-dominates original JumpRec (≥ quality at ≤ cost).
6. Net of draft cost, speedup `> 1×`.
