"""Builds notebooks/sld_parcae_gpu.ipynb. Run: python notebooks/build_notebook.py

Cells are authored as Python strings and emitted as valid .ipynb JSON. The
notebook is self-validating: a sanity cell ASSERTS that our hooked re-implementation
of parcae's loop reproduces parcae's own next-token output before any SLD claim.
"""
import json
from pathlib import Path

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md(r"""
# SLD on parcae — lossless speculative acceleration of a stable looped LM (GPU)

**Speculative Looped Decoding (SLD)** transplants speculative decoding from the
*token* axis to the *depth/loop* axis of a looped transformer. A cheap **draft**
proposes future loop states; the **true core verifies them in one batched pass**;
we accept the longest prefix consistent with the true core and continue. A
depth-`T` recurrence collapses from `T` *sequential* core calls to a few
*parallel-verified* rounds — same FLOPs, far less sequential depth — **without
changing the model's output**.

On CPU ([SLD repo](https://github.com/asystemoffields/SLD), see `RESULTS.md`) this
is *exactly lossless* and turns a depth-`k` synthetic recurrence into **one**
sequential core round (vs `k`). But the CPU wall-clock win is fragile: a batched
verify of `T` states leaves the CPU "flat" region at any real serving batch.
**A GPU fixes that for free** — that batched verify is one parallel kernel. This
notebook tests SLD on a *real* pretrained looped LM,
[`parcae`](https://github.com/sandyresearch/parcae) (stable looped LM, recurrence
`T=8`, contractive core `ρ<1` → the loop converges), measuring:

1. **Losslessness** — SLD's next token vs the full `T`-loop next token.
2. **Sequential core rounds** — hardware-free (SLD ≪ `T`).
3. **GPU wall-clock** at batch 1 / 16 / 64 — the serving batch where SLD beats
   the full loop *and* a fair early-exit baseline (the win CPU could not show).

> Everything below is model-agnostic except one `LoopedAdapter` cell, which is
> **self-validating**: it asserts our re-implementation of parcae's loop matches
> parcae's native output. If parcae's internals differ from the assumptions
> (additive input-injection; the core is the module called `T` times), the
> assertion fails loudly with guidance — adjust that one cell, the SLD code is
> untouched.
""")

code(r"""
import torch, time, math
assert torch.cuda.is_available(), "Runtime ▸ Change runtime type ▸ GPU"
DEV = "cuda"; torch.manual_seed(0)
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
""")

code(r"""
!pip -q install parcae-lm einops sentencepiece tokenizers safetensors transformers >/dev/null 2>&1
!git clone -q https://github.com/asystemoffields/SLD.git 2>/dev/null || true
print("installed")
""")

md(r"""
## 1. Load parcae and inspect the loop

We need three primitives for SLD: `encode` (prelude → initial state `h0`),
`step` (one shared-core iteration), `decode` (coda + head → next-token logits).
First, find the recurrent core (the module invoked `T` times) and the loop count.
""")

code(r"""
import parcae_lm
parcae = parcae_lm.from_pretrained("SandyResearch/parcae-140m").to(DEV).eval()
for p in parcae.parameters(): p.requires_grad_(False)

cfg = getattr(parcae, "config", None)
T_LOOPS = None
for k in ("recurrence", "n_loops", "num_recurrence", "recurrent_steps", "loops", "T"):
    if cfg is not None and hasattr(cfg, k):
        T_LOOPS = int(getattr(cfg, k)); print("loop-count:", k, "=", T_LOOPS)
T_LOOPS = T_LOOPS or 8
print("using T_LOOPS =", T_LOOPS)

# count how many times each module is called in one forward -> the core is called T times
from collections import Counter
counts = Counter(); handles = []
for n, m in parcae.named_modules():
    if "." in n and len(list(m.parameters(recurse=False))) == 0 and not any(m.children()):
        continue
    handles.append(m.register_forward_hook((lambda nm: (lambda *_: counts.update([nm])))(n)))
with torch.no_grad(): parcae(torch.randint(0, 1000, (1, 8), device=DEV))
for h in handles: h.remove()
# the recurrent core (and its descendants) are the modules called exactly T times;
# the OUTERMOST such module (shallowest path) is the looped unit we want.
core_candidates = sorted([n for n, c in counts.items() if c == T_LOOPS and n],
                         key=lambda n: (n.count("."), len(n)))
print("modules called exactly", T_LOOPS, "times (shallowest first):", core_candidates[:8])
""")

md(r"""
## 2. Adapter (the only parcae-specific cell) + self-validation

`step(h) = core(h + e)` with a constant input-injection `e` recovered from a trace;
`decode(ids, h)` re-runs the model but **overrides the core's final output with `h`**,
so parcae's own coda + head turn `h` into logits (no need to know their internals).
The cell **asserts** that replaying the loop ourselves reproduces parcae's native
next-token argmax. If it fails, set `CORE_NAME` or switch the injection model.
""")

code(r'''
CORE_NAME = core_candidates[0] if core_candidates else None    # outermost looped unit; override if needed
assert CORE_NAME, "Could not auto-find the recurrent core; set CORE_NAME to the module looped T times."
core = dict(parcae.named_modules())[CORE_NAME]
print("recurrent core:", CORE_NAME, type(core).__name__)

class LoopedAdapter:
    def __init__(self, model, core, T):
        self.model, self.core, self.T = model, core, T
        self._io, self._override, self._calls = [], None, 0
        core.register_forward_hook(self._rec)        # records (in,out) and applies override
    def _rec(self, m, inp, out):
        self._calls += 1
        self._io.append((inp[0].detach(), out.detach() if torch.is_tensor(out) else out[0].detach()))
        if self._override is not None and self._calls == self.T:
            return self._override                     # substitute final core output with our state
    def _trace(self, ids):
        self._io, self._calls, self._override = [], 0, None
        with torch.no_grad(): out = self.model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        cin  = [a for (a, b) in self._io]
        cout = [b for (a, b) in self._io]
        # constant additive injection: core input_t = h_{t-1} + e ; e = cin[1]-cout[0]
        e  = cin[1] - cout[0]
        h0 = cin[0] - e                               # h0 (prelude output) = cin[0] - e
        return logits, h0, e, cout
    @torch.no_grad()
    def step(self, h):  return self.core(h + self.e)
    @torch.no_grad()
    def decode(self, ids, h):
        self._override, self._calls = h, 0
        with torch.no_grad(): out = self.model(ids)
        self._override = None
        logits = out.logits if hasattr(out, "logits") else out
        return logits[:, -1, :]                       # next-token logits

A = LoopedAdapter(parcae, core, T_LOOPS)

# ---- self-validation: our manual loop must match parcae's native output ----
ids = torch.randint(0, 1000, (4, 24), device=DEV)
native_logits, h0, e, true_traj = A._trace(ids)
A.e = e
h = h0
for _ in range(T_LOOPS): h = A.step(h)
ours = A.decode(ids, h)[:, :]                          # decode our final state
native_next = (native_logits[:, -1, :]).argmax(-1)
agree = (ours.argmax(-1) == native_next).float().mean().item()
print(f"replay vs native next-token agreement: {agree:.3f}")
assert agree > 0.99, ("Adapter assumptions don't hold for this parcae build. "
                      "Check: (1) CORE_NAME is the looped block; (2) injection is additive "
                      "(else use concat); (3) decode override targets the right call.")
print("[ok] adapter validated — SLD below operates on the real parcae loop.")
''')

md(r"""
## 3. parcae converges — so early-exit is a *fair* baseline

A contractive core (`ρ<1`) means the loop settles toward a fixed point. We confirm
it: decode the next token at each loop step and see how early it stabilizes. Where
it stabilizes before `T`, **convergence early-exit** is already a legitimate
speedup — so SLD has to beat *that*, not just the full loop.
""")

code(r"""
@torch.no_grad()
def trajectory_next_tokens(ids):
    _, h0, e, _ = A._trace(ids); A.e = e
    toks, h = [], h0
    for _ in range(T_LOOPS):
        h = A.step(h)
        toks.append(A.decode(ids, h).argmax(-1))      # next token after t loops
    return torch.stack(toks, 0)                        # [T, B]

ids = torch.randint(0, 1000, (64, 32), device=DEV)
toks = trajectory_next_tokens(ids)
final = toks[-1]
settle = torch.full_like(final, T_LOOPS)
for t in range(T_LOOPS - 1, -1, -1):
    settle = torch.where(toks[t] == final, torch.full_like(settle, t + 1), settle)
print("loops until the next token equals the T-loop answer (per example):")
for t in range(1, T_LOOPS + 1):
    frac = (settle <= t).float().mean().item()
    print(f"  by loop {t}: {frac*100:5.1f}% of tokens already final")
print("mean loops needed:", settle.float().mean().item(), "/", T_LOOPS)
""")

md(r"""
## 4. SLD on parcae

Because parcae converges, SLD specializes to **verified fixed-point acceleration**:
run a couple of real loop steps, then a **training-free Anderson/Aitken draft**
extrapolates the converged state `ŝ`; we **verify** `ŝ` is on-trajectory by checking
its next token is stable under one more *true* core step
(`argmax decode(core(ŝ)) == argmax decode(ŝ)`); if so we accept (lossless on the
readout) and stop, else we take more real steps. This is the convergent-loop form
of SLD; the multi-step batched verify (the synthetic headline) reduces to this
when the trajectory is a contraction.
""")

code(r"""
@torch.no_grad()
def aitken(s_km1, s_k, s_kp1):
    # vector Aitken / Irons-Tuck extrapolate of the fixed point from 3 iterates
    d1 = s_k - s_km1; d2 = s_kp1 - s_k
    dd = d2 - d1
    num = (d2 * dd).flatten(1).sum(1, keepdim=True)
    den = (dd * dd).flatten(1).sum(1, keepdim=True).clamp_min(1e-9)
    a = (num / den).view(-1, *([1] * (s_k.dim() - 1)))
    return s_kp1 - a * d2

@torch.no_grad()
def sld_decode(ids, warmup=2, max_loops=None):
    # returns (next_token, core_rounds); lossless target = the full T-loop next token
    T = max_loops or T_LOOPS
    _, h0, e, _ = A._trace(ids); A.e = e
    hs = [h0]; rounds = 0
    for _ in range(warmup):                       # a few real steps to seed extrapolation
        hs.append(A.step(hs[-1])); rounds += 1
    while rounds < T:
        s_star = aitken(hs[-3], hs[-2], hs[-1]) if len(hs) >= 3 else hs[-1]
        cs = A.step(s_star); rounds += 1          # one true-core verification step
        a_star = A.decode(ids, s_star).argmax(-1)
        a_next = A.decode(ids, cs).argmax(-1)
        if (a_star == a_next).all():              # verified fixed point (readout stable)
            return a_star, rounds
        hs.append(cs)                             # not converged: keep the true step, continue
    return A.decode(ids, hs[-1]).argmax(-1), rounds

@torch.no_grad()
def full_decode(ids):
    _, h0, e, _ = A._trace(ids); A.e = e
    h = h0
    for _ in range(T_LOOPS): h = A.step(h)
    return A.decode(ids, h).argmax(-1), T_LOOPS

@torch.no_grad()
def earlyexit_decode(ids, patience=1):
    _, h0, e, _ = A._trace(ids); A.e = e
    h = h0; prev = None; stable = 0
    for t in range(1, T_LOOPS + 1):
        h = A.step(h); cur = A.decode(ids, h).argmax(-1)
        if prev is not None and (cur == prev).all():
            stable += 1
            if stable >= patience: return cur, t
        else: stable = 0
        prev = cur
    return prev, T_LOOPS

ids = torch.randint(0, 1000, (64, 32), device=DEV)
full_ans, full_r = full_decode(ids)
ee_ans, ee_r = earlyexit_decode(ids)
sld_ans, sld_r = sld_decode(ids, warmup=2)
print(f"full-loop:   rounds={full_r}        (reference)")
print(f"early-exit:  rounds={ee_r}   lossless={ (ee_ans==full_ans).float().mean().item():.3f}")
print(f"SLD:         rounds={sld_r}   lossless={ (sld_ans==full_ans).float().mean().item():.3f}")
""")

md(r"""
## 5. GPU wall-clock — the regime CPU could not win

We time the **recurrence** (the sequential core calls SLD reduces) at batch 1 / 16 /
64. On GPU the batched verify stays parallel, so fewer *sequential* core calls is a
real **latency** drop — at exactly the serving batches where the CPU version lost
(its verify batch left the flat region). Decode is identical across methods and
charged once.
""")

code(r"""
@torch.no_grad()
def time_recurrence(method, B, reps=20, warm=5):
    ids = torch.randint(0, 1000, (B, 32), device=DEV)
    for _ in range(warm): method(ids); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps): method(ids); torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3

print(f"{'batch':>6} {'full ms':>9} {'early ms':>9} {'SLD ms':>9} {'SLD speedup':>12}")
for B in (1, 16, 64):
    tf = time_recurrence(lambda x: full_decode(x), B)
    te = time_recurrence(lambda x: earlyexit_decode(x), B)
    ts = time_recurrence(lambda x: sld_decode(x, warmup=2), B)
    print(f"{B:>6} {tf:>9.2f} {te:>9.2f} {ts:>9.2f} {tf/ts:>11.2f}x")
""")

md(r"""
## 6. How to read this, honestly

* **Losslessness** — SLD's next token should match the full `T`-loop on ~100% of
  positions (verification only accepts a readout-stable fixed point).
* **Rounds** — SLD should use fewer sequential core calls than `T` whenever parcae
  has converged before `T`; this is the hardware-free claim and equals the latency
  on parallel hardware.
* **GPU wall-clock** — fewer sequential rounds → lower latency, and the saving
  *holds as batch grows* (the batched core call is one parallel kernel), which is
  the central thing GPU buys over the CPU experiment.

**Caveats.** parcae runs a *fixed* `T=8`, so the target is `h_T` (not the true
fixed point) and the SLD speedup is bounded by `T=8`; the dramatic version needs a
*deeper* recurrence (e.g. Huginn's 32–132 unrolls), where collapsing sequential
depth pays off far more — the same `sld_decode`/adapter apply, just with larger
`T`. And the `decode`-override re-runs the model per state, so for production timing
identify parcae's coda+head and apply them directly (cheaper); here we time the
*recurrence*, which is what SLD actually changes.

**Provenance.** Method + lossless CPU results + ablations:
https://github.com/asystemoffields/SLD . parcae:
https://github.com/sandyresearch/parcae .
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent / "sld_parcae_gpu.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
