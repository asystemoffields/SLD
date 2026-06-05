# notebooks

**`sld_parcae_gpu.ipynb`** — take Speculative Looped Decoding to a GPU on a real
pretrained looped LM, [`parcae`](https://github.com/sandyresearch/parcae).

Open it in Google Colab (Runtime ▸ GPU) and run top to bottom. It:

1. loads `parcae-140m` and finds its recurrent core + loop count;
2. builds a **self-validating adapter** (recovers `encode`/`step`/`decode` via
   hooks and **asserts** the reconstructed loop reproduces parcae's native
   next-token output — so no SLD number is reported on a wrong wiring);
3. confirms parcae's loop **converges** (so convergence early-exit is a *fair*
   baseline, not a strawman);
4. runs **SLD** (verified fixed-point form: Anderson-extrapolate the converged
   state, accept only if its next token is stable under one more *true* core
   step) vs full-loop and early-exit — reporting losslessness and sequential
   core rounds;
5. times the **recurrence wall-clock** at batch 1 / 16 / 64 on GPU.

The notebook is **model-agnostic except one `LoopedAdapter` cell**. If parcae's
internals differ from the assumptions (the core is the module called `T` times;
additive input injection; decode via overriding the core's final output), the
self-validation assertion fails loudly with guidance — fix that one cell, the SLD
logic is untouched.

`build_notebook.py` generates the `.ipynb` from Python sources (`python
notebooks/build_notebook.py`); edit there and regenerate rather than editing the
JSON by hand.

Why GPU: on CPU the batched verify of `T` states leaves the "flat" region at any
real serving batch, so SLD's fewer *sequential* rounds don't translate to a
wall-clock win there. On a GPU the batched verify is one parallel kernel, so the
round-count reduction becomes a real latency reduction at batch — the point of
this notebook. Method, lossless CPU results, and ablations are in the repo root
(`README.md`, `RESULTS.md`, `SLD_SPEC.md`).
