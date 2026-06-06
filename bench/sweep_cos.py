"""Sweep the state-cosine convergence threshold. Verification signal = cosine of
the last-position recurrent state between consecutive core steps (cheap: a core
step + a dot product, NO coda, NO 32k lm_head). Decode exactly once, to emit.

For each threshold report, vs the full T-loop: token match rate (faithfulness),
mean core rounds, and wall-clock speedup. This is the wholesale Leak-2 fix:
convergence is detected in state space, so verification stops paying the readout.
"""
from __future__ import annotations
import sys, time, statistics as st
import torch
from parcae_sld import ParcaeLoop, aitken
torch.set_num_threads(6)


@torch.no_grad()
def lastcos(a, b):
    return torch.nn.functional.cosine_similarity(a[:, -1], b[:, -1], dim=-1).item()


@torch.no_grad()
def full_tok(loop, ctx, T):
    x, e, fc = loop.encode(ctx)
    for _ in range(T): x = loop.step(x, e, fc)
    return int(loop.decode(x, ctx, fc).argmax(-1))


@torch.no_grad()
def ee_state(loop, ctx, T, thr, patience=1):
    """early-exit on state cosine: stop once the last-position state stops moving."""
    x, e, fc = loop.encode(ctx); prev = x; stable = 0; r = 0
    for t in range(1, T + 1):
        x = loop.step(x, e, fc); r += 1
        if lastcos(x, prev) >= thr:
            stable += 1
            if stable >= patience: return int(loop.decode(x, ctx, fc).argmax(-1)), r
        else: stable = 0
        prev = x
    return int(loop.decode(x, ctx, fc).argmax(-1)), r


@torch.no_grad()
def sld_state(loop, ctx, T, thr, warmup=2):
    """extrapolate the fixed point, accept once one true step barely moves the
    last-position state (cos >= thr). Decode once."""
    x, e, fc = loop.encode(ctx); hs = [x]; r = 0
    for _ in range(warmup): x = loop.step(x, e, fc); r += 1; hs.append(x)
    while r < T:
        s = aitken(hs[-3], hs[-2], hs[-1]) if len(hs) >= 3 else hs[-1]
        s1 = loop.step(s, e, fc); r += 1
        if lastcos(s1, s) >= thr: return int(loop.decode(s1, ctx, fc).argmax(-1)), r
        hs.append(s1)
    return int(loop.decode(hs[-1], ctx, fc).argmax(-1)), r


def main():
    import parcae_lm
    from transformers import AutoTokenizer
    from datasets import load_dataset
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    tok = AutoTokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    T = int(getattr(m.config, "mean_recurrence", 8))
    loop = ParcaeLoop(m)

    exs = []
    for i in range(min(n, len(ds))):
        ids = tok(ds[i]["text"], return_tensors="pt").input_ids
        if ids.shape[1] >= 2: exs.append((ids[:, :-1], int(ids[0, -1])))
    print(f"{len(exs)} LAMBADA examples, T={T}\n", flush=True)

    # full-loop reference (token + wall-clock + accuracy)
    a = time.perf_counter(); fulls = [full_tok(loop, c, T) for c, _ in exs]; tf = time.perf_counter() - a
    acc_full = sum(p == g for p, (_, g) in zip(fulls, exs)) / len(exs)
    print(f"full loop: acc {acc_full:.3f}  {T} rounds  {tf/len(exs)*1e3:.1f} ms/ex\n", flush=True)

    for name, fn in [("early-exit/state", ee_state), ("SLD/state", sld_state)]:
        print(f"== {name} ==", flush=True)
        print(f"  {'thr':>8}{'match-full':>12}{'acc':>8}{'rounds':>9}{'ms/ex':>9}{'speedup':>9}", flush=True)
        for thr in [0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]:
            a = time.perf_counter()
            out = [fn(loop, c, T, thr) for c, _ in exs]
            tt = time.perf_counter() - a
            match = sum(p == f for (p, _), f in zip(out, fulls)) / len(exs)
            acc = sum(p == g for (p, _), (_, g) in zip(out, exs)) / len(exs)
            rounds = st.mean([r for _, r in out])
            print(f"  {thr:>8.4f}{match:>12.3f}{acc:>8.3f}{rounds:>9.2f}"
                  f"{tt/len(exs)*1e3:>9.1f}{tf/tt:>8.2f}x", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
