# SLD — Speculative Looped Decoding

**Verified acceleration of looped / recurrent-depth transformers.** A looped model
reasons by running a shared **core** block many times (`h_{t+1} = core(h_t + e)`),
which is powerful but **sequential** — and often *redundant*: the computation tends
to converge well before the loop budget is spent. SLD detects that convergence
cheaply, **extrapolates** the converged state, and **verifies** it (accept only when
the model's next-token prediction is stable under more true core steps) before
skipping the remaining loops. The model's behavior is *preserved* — not approximated
away — at fewer sequential core calls.

## Validated on a real model (parcae-140m)

The real test of any looped-LM speedup is whether it preserves a *real* model's
*real* behavior. We run SLD on the pretrained
[`parcae-140m`](https://github.com/sandyresearch/parcae) stable looped LM
(recurrence `T=8`) — on **parcae's own evaluations**, head-to-head against the full
loop (`bench/parcae_lambada.py`, `bench/parcae_nl.py`, `bench/parcae_generate.py`):

**LAMBADA** (`eval_configs/eval-lambada.yaml`, 120 examples, CPU):

| method | LAMBADA acc | matches full loop | core rounds | CPU ms/ex | speedup |
|---|--:|--:|--:|--:|--:|
| full loop | 0.500 | — | 8 | 135 | 1.00× |
| **SLD** | **0.500** | 92.5% | 4.42 | 89.5 | **1.51×** |
| early-exit | 0.500 | 92.5% | 4.47 | 90.4 | 1.49× |

SLD **preserves parcae's benchmark accuracy** (0.500 → 0.500) while cutting the
recurrence to ~4.4 of 8 core calls **at a real 1.5× CPU wall-clock speedup** — the
skipped loops become latency because verification is in *state space* (the
contractive core's last-position state cosine →1; a core step + a dot product, no
32k-vocab decode), and the readout is paid once. ARC-Easy (a multiple-choice
benchmark) shows the same accuracy preservation. And it is **legible** — parcae
generates coherent English and SLD reproduces it:

```
"The capital of France is" -> " Paris. It is the capital of the French Republic,
                                the largest country in Europe, and the largest"
"Water is made of"         -> " water molecules. Water molecules are made up of
                                water, hydrogen, oxygen, carbon, nitrogen, and"
```

**Scope.** This is *near*-lossless on a real continuous-state LM — the accepted token
matches the full loop on ~92% of positions and benchmark accuracy is preserved, with
a rare early-accept that can compound over long greedy generation (so generation uses
a stricter acceptance threshold). The 1.5× is on parcae's short `T=8` loop on CPU; it
**grows with loop depth** — the single emit-decode amortizes over more skipped core
calls, so recurrent-depth LMs that unroll 32–132× gain far more.

## Take it to a GPU

[`notebooks/sld_parcae_gpu.ipynb`](notebooks/sld_parcae_gpu.ipynb) is a Colab
notebook that runs this head-to-head on a GPU: a **self-validating** parcae adapter
(it asserts the reconstructed loop matches parcae's native output before any claim),
then LAMBADA + ARC-Easy + generation, full-loop vs SLD, reporting accuracy,
agreement, core rounds and wall-clock. Adding another of parcae's core tasks is a
one-line `load_dataset(...)` change.

## How it works

For each token, the recurrence runs the shared core `T` times to produce next-token
logits. SLD:
1. runs a few real core steps (warm-up);
2. **extrapolates** the converged state with a vector fixed-point accelerator
   (Aitken / Anderson);
3. **verifies** the extrapolated state in *state space* — one more true core step
   that barely moves the last-position state (cosine ≥ `thr`) means the contractive
   recurrence has settled and the readout has locked. This costs a core step + a dot
   product; crucially it does **not** decode, so verification doesn't give back the
   loops it saved;
4. accepts and decodes **once** to emit if settled, else keeps looping.

The result is identical-or-near-identical to the full loop, at fewer sequential core
calls *and* less wall-clock. The method is a depth-axis cousin of speculative
decoding (verify a cheap guess against the true model, fall back when it disagrees) —
here the "cheap check" is moved off the expensive 32k-vocab readout onto the state.

## Install & run

SLD builds on the [`jumprec`](https://github.com/asystemoffields/jumprec) substrate
(the looped-transformer + a synthetic recurrence task used for the controlled study
below). Either install it, keep a sibling `../SMOKE` checkout, or set `JUMPREC_PATH`.

```bash
pip install -e .                                  # torch (cpu/gpu) + numpy
PYTHONPATH=../SMOKE python -m pytest tests/ -q    # invariants
# real-model validation (downloads parcae-140m + tokenizer):
PYTHONPATH=../SMOKE:. python bench/parcae_lambada.py 200
PYTHONPATH=../SMOKE:. python bench/parcae_generate.py
```

## A controlled synthetic study (clearly bounded)

To characterize the *mechanism* in isolation — where the recurrence is a clean,
discrete-symbol computation and the verifier can be exact — see
[`RESULTS.md`](RESULTS.md) ("Synthetic study"). There the readout is a sufficient
statistic, so verification is bit-stable and SLD is **exactly lossless**, and on a
deliberately deep loop its sequential-round advantage over early-exit grows large.
That is a controlled best-case for understanding the method; the **real-model
parcae numbers above are the load-bearing claim** — a real continuous-state LM does
not have that discrete structure, and there SLD is near-lossless, as reported.

## What's here

```
sld/             specloop.py, draft.py, training.py, substrate.py  (the method + substrate bridge)
bench/
  parcae_lambada.py / parcae_nl.py / parcae_generate.py / parcae_sld.py / parcae_cpu.py
                 real parcae-140m validation (validated adapter, his evals)
  experiment.py / convergent*.py / draft_quality.py / incontext.py
                 the controlled synthetic study (depth, ablations)
  summarize.py / plot.py / common.py / run_all.sh
notebooks/       sld_parcae_gpu.ipynb  (GPU head-to-head on parcae)
tests/           losslessness + control invariants (synthetic)
RESULTS.md       real-model results, then the bounded synthetic study
SLD_SPEC.md      the method, the losslessness condition, prior art
```

## License

MIT.
