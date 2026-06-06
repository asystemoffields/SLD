"""Real SLD on parcae-140m (CPU). Builds encode/step/decode from parcae's own
modules, VALIDATES the manual loop reproduces parcae's native output, then runs
SLD (verified fixed-point acceleration) and checks it is lossless on the real
next token. Slow-ish but real; the same adapter goes into the GPU notebook.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import torch

RES = Path(__file__).resolve().parents[1] / "results"


class ParcaeLoop:
    """encode / step / decode for parcae, reconstructed from its forward()."""
    def __init__(self, m):
        self.m = m
        self.off_coda = m.config.n_layers_in_prelude + m.config.n_layers_in_recurrent_block
        self.logit_scale = m.config.init.logit_scale

    @torch.no_grad()
    def encode(self, ids):
        m = self.m
        m._current_input_ids = ids
        fc = m.freqs_cis[:, : ids.shape[1]]
        e = m.transformer.wte(ids)
        if getattr(m, "emb_scale", 1) != 1:
            e = e * m.emb_scale
        for i, block in enumerate(m.transformer.prelude):
            ve = m.value_embeds[str(i)](ids) if str(i) in m.value_embeds else None
            e = block(e, fc, None, ve=ve)
        if m.config.prelude_norm:
            e = m.transformer.ln_prelude(e)
        x0 = m.initialize_state(e)
        return x0, e, fc

    @torch.no_grad()
    def step(self, x, e, fc):
        return self.m.core_block_forward(x, e, fc, None, torch.tensor(0), torch.tensor(0))

    @torch.no_grad()
    def decode(self, x, ids, fc):
        """Next-token readout: coda over all positions (a causal stack needs them),
        but ln_f + lm_head on the LAST position only -- we never read the rest, and
        the 32k-vocab projection over a full sequence is ~80% of a naive decode."""
        m = self.m
        x = m.transformer.C(x)
        for i, block in enumerate(m.transformer.coda):
            k = str(self.off_coda + i)
            ve = m.value_embeds[k](ids) if k in m.value_embeds else None
            x = block(x, fc, None, ve=ve)
        x = m.transformer.ln_f(x[:, -1:, :])
        return (m.lm_head(x).float() * self.logit_scale)[:, -1, :]


@torch.no_grad()
def aitken(a, b, c):  # vector Aitken extrapolation of the fixed point from 3 iterates
    d1, d2 = b - a, c - b
    dd = d2 - d1
    coef = (d2 * dd).flatten(1).sum(1, keepdim=True) / (dd * dd).flatten(1).sum(1, keepdim=True).clamp_min(1e-9)
    return c - coef.view(-1, *([1] * (c.dim() - 1))) * d2


@torch.no_grad()
def lastcos(a, b):  # cosine of the last-position state between two iterates -> [B]
    return torch.nn.functional.cosine_similarity(a[:, -1], b[:, -1], dim=-1)


@torch.no_grad()
def sld(loop, ids, T, warmup=2, thr=0.999):
    """Verified fixed-point SLD with STATE-SPACE verification. The recurrent core is
    contractive: the last-position state converges (cosine -> 1) and the readout
    locks well before. Warm up, extrapolate the fixed point, and accept once one
    true core step barely moves the last-position state (cosine >= thr). The check
    is a core step + a dot product -- no 32k-vocab decode -- so the skipped loops
    become wall-clock. Decode ONCE, to emit. Returns (token, rounds)."""
    x, e, fc = loop.encode(ids)
    hs = [x]; rounds = 0
    for _ in range(warmup):
        x = loop.step(x, e, fc); hs.append(x); rounds += 1
    while rounds < T:
        s = aitken(hs[-3], hs[-2], hs[-1]) if len(hs) >= 3 else hs[-1]
        s1 = loop.step(s, e, fc); rounds += 1
        if (lastcos(s1, s) >= thr).all():
            return loop.decode(s1, ids, fc).argmax(-1), rounds
        hs.append(s1)
    return loop.decode(hs[-1], ids, fc).argmax(-1), rounds


@torch.no_grad()
def earlyexit(loop, ids, T, thr=0.999, patience=1):
    """State-space convergence early-exit on the TRUE iterates: stop once the
    last-position state stops moving (cosine >= thr for `patience` steps), then
    decode ONCE. Same convergence signal as SLD, without the extrapolation jump."""
    x, e, fc = loop.encode(ids)
    prev = x; stable = 0
    for t in range(1, T + 1):
        x = loop.step(x, e, fc)
        if (lastcos(x, prev) >= thr).all():
            stable += 1
            if stable >= patience:
                return loop.decode(x, ids, fc).argmax(-1), t
        else:
            stable = 0
        prev = x
    return loop.decode(x, ids, fc).argmax(-1), T


def main():
    import parcae_lm
    torch.set_num_threads(6)
    print("loading parcae-140m ...", flush=True)
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    T = int(getattr(m.config, "mean_recurrence", 8))
    loop = ParcaeLoop(m)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("SandyResearch/parcae-140m")
        prompts = [tok(t, return_tensors="pt").input_ids for t in
                   ["The capital of France is", "Water is made of hydrogen and",
                    "The opposite of hot is", "Two plus two equals",
                    "The quick brown fox jumps over the lazy", "She opened the door and saw a"]]
    except Exception as e:
        print("tokenizer not bundled with the 140m checkpoint; using valid random token ids "
              "(the loop's convergence is a property of the model on its input regardless)", flush=True)
        tok = None
        g = torch.Generator().manual_seed(0)
        prompts = [torch.randint(0, 30000, (1, 12), generator=g) for _ in range(16)]

    # ---- validate the adapter against parcae's native forward ----
    print("\n[validate] manual loop vs native forward:", flush=True)
    ok = True
    for ids in prompts[:4]:
        x, e, fc = loop.encode(ids)
        for _ in range(T): x = loop.step(x, e, fc)
        ours = loop.decode(x, ids, fc).argmax(-1)
        native = m(ids, num_steps_pair=torch.tensor([T, 0]), return_logits=True)["logits"][:, -1, :].argmax(-1)
        match = (ours == native).all().item()
        ok = ok and match
        print(f"   T={T}: ours=={native.tolist()} native -> {'MATCH' if match else 'MISMATCH'}", flush=True)
    assert ok, "adapter does not reproduce native parcae output; do not trust SLD numbers"
    print("[ok] adapter validated on real parcae.", flush=True)

    # ---- full-loop vs lossless early-exit vs SLD on real parcae ----
    print("\n[accelerate] full-loop vs lossless early-exit vs SLD on real parcae:", flush=True)
    r_ee, ll_ee, r_sld, ll_sld = [], [], [], []
    for ids in prompts:
        full = m(ids, num_steps_pair=torch.tensor([T, 0]), return_logits=True)["logits"][:, -1, :].argmax(-1)
        ee_a, ee_r = earlyexit(loop, ids, T)
        sld_a, sld_r = sld(loop, ids, T)
        r_ee.append(ee_r); ll_ee.append((ee_a == full).all().item())
        r_sld.append(sld_r); ll_sld.append((sld_a == full).all().item())
    import statistics as st
    print(f"   full-loop:           {T} sequential core rounds (reference)", flush=True)
    print(f"   early-exit (lossless, sequential): mean {st.mean(r_ee):.2f} rounds, "
          f"lossless {sum(ll_ee)}/{len(ll_ee)}", flush=True)
    print(f"   SLD  (verified):                   mean {st.mean(r_sld):.2f} rounds, "
          f"lossless {sum(ll_sld)}/{len(ll_sld)}", flush=True)
    print("   note: on parcae's short T=8 loop, sequential early-exit already captures the", flush=True)
    print("   convergence headroom on CPU; SLD's edge is verifying depths IN PARALLEL (one", flush=True)
    print("   batched core pass) -> fewer SEQUENTIAL rounds at GPU batch, and it scales to", flush=True)
    print("   deep recurrence (e.g. Huginn's 32-132 loops). See the GPU notebook.", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "parcae_sld.json").write_text(json.dumps(
        {"T": T, "n_prompts": len(prompts),
         "early_exit_rounds": r_ee, "early_exit_lossless": sum(ll_ee),
         "sld_rounds": r_sld, "sld_lossless": sum(ll_sld),
         "mean_early_exit_rounds": st.mean(r_ee), "mean_sld_rounds": st.mean(r_sld)}, indent=2))
    print("[saved] results/parcae_sld.json", flush=True)


if __name__ == "__main__":
    main()
