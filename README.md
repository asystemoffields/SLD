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

**LAMBADA** (`eval_configs/eval-lambada.yaml`, 200 examples, CPU):

| method | LAMBADA acc | matches full loop | core rounds | CPU ms/ex |
|---|--:|--:|--:|--:|
| full loop | 0.535 | — | 8 | 181 |
| **SLD** | **0.535** | 96.5% | 5.25 | 277 |
| early-exit | 0.520 | 93.0% | 3.68 | 269 |

SLD **preserves parcae's benchmark accuracy** (0.535 → 0.535) while cutting the
recurrence to ~5 of 8 core calls, and holds it *more faithfully than the un-verified
early-exit baseline*, which drops to 0.520. ARC-Easy (a multiple-choice benchmark)
shows the same accuracy preservation. And it is **legible** — parcae generates
coherent English and SLD reproduces it:

```
"The capital of France is" -> " Paris. It is the capital of the French Republic,
                                the largest country in Europe, and the largest"
"Water is made of"         -> " water molecules. Water molecules are made up of
                                water, hydrogen, oxygen, carbon, nitrogen, and"
```

**Scope.** This is *near*-lossless on a real continuous-state LM — predictions
are essentially identical per recurrence, with a rare early-accept that can compound
over long greedy generation. On parcae's short `T=8` loop the per-step verification
can offset the saved core calls in *CPU wall-clock*; the saving converts to **latency
on a GPU** (each forward is a kernel launch) and **grows with loop depth** —
recurrent-depth LMs unroll 32–132×, where the full loop is far more wasteful.

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
3. **verifies** the extrapolated state by checking its next-token prediction is
   stable under a couple more true core steps;
4. accepts and stops if stable, else keeps looping.

The result is identical-or-near-identical to the full loop, at fewer sequential core
calls. The method is a depth-axis cousin of speculative decoding (verify a cheap
guess against the true model, fall back when it disagrees).

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
