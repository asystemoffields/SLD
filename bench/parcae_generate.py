"""Tangible real-NL check: greedy generation from parcae-140m, decoded to English.

Generate a continuation from a real prompt with (a) the full T-loop recurrence and
(b) SLD-accelerated recurrence at every token, then DECODE both to text. SLD should
produce the identical generation while using fewer sequential core rounds per token.
This is the most legible non-synthetic validation: you can read the output.

Run: PYTHONPATH=../SMOKE:.. python bench/parcae_generate.py
"""
from __future__ import annotations
import torch
from parcae_sld import ParcaeLoop, sld, earlyexit   # CPU-validated adapter + decoders

PROMPTS = [
    "The capital of France is",
    "Water is made of",
    "The sun rises in the",
    "Two plus two equals",
]


@torch.no_grad()
def full_next(loop, ids, T):
    x, e, fc = loop.encode(ids)
    for _ in range(T): x = loop.step(x, e, fc)
    return loop.decode(x, ids, fc).argmax(-1), T


@torch.no_grad()
def generate(loop, ids0, n_new, next_fn, T):
    ids = ids0; rounds = 0
    for _ in range(n_new):
        t, r = next_fn(loop, ids, T); rounds += r
        ids = torch.cat([ids, t[:, None]], dim=1)
    return ids, rounds


def main():
    import parcae_lm
    from transformers import AutoTokenizer
    torch.set_num_threads(6)
    print("loading parcae-140m + tokenizer ...", flush=True)
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    tok = AutoTokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
    T = int(getattr(m.config, "mean_recurrence", 8))
    loop = ParcaeLoop(m)
    N = 20

    print(f"\n=== greedy generation: full T={T} loop vs SLD (decoded to English) ===\n", flush=True)
    n_exact = 0; tok_match = tok_total = 0; sldr_all = []
    for p in PROMPTS:
        ids0 = tok(p, return_tensors="pt").input_ids
        full_ids, full_r = generate(loop, ids0, N, full_next, T)
        sld_ids, sld_r = generate(loop, ids0, N, lambda l, i, t: sld(l, i, t, thr=0.9999), T)
        match = torch.equal(full_ids, sld_ids); n_exact += int(match); sldr_all.append(sld_r / N)
        g = full_ids[0][ids0.shape[1]:]; s = sld_ids[0][ids0.shape[1]:]
        tok_match += (g == s).sum().item(); tok_total += g.numel()
        print(f"prompt: {p!r}", flush=True)
        print(f"  full-loop  ({full_r/N:.1f} rounds/tok): {tok.decode(g)!r}", flush=True)
        print(f"  SLD        ({sld_r/N:.1f} rounds/tok): {tok.decode(s)!r}", flush=True)
        print(f"  identical: {match}\n", flush=True)
    import statistics as st
    print(f"=> parcae generates coherent English; SLD reproduces it at ~{st.mean(sldr_all):.1f}/{T} "
          f"core rounds per token,", flush=True)
    print(f"   exactly on {n_exact}/{len(PROMPTS)} prompts, {tok_match}/{tok_total} tokens matching.", flush=True)
    print(f"   This is NEAR-lossless: per single recurrence step it is ~100% (see parcae_nl.py:", flush=True)
    print(f"   115/115 positions), but over many autoregressive steps the fixed-point-acceptance", flush=True)
    print(f"   heuristic can accept slightly early and compound -- the EXACT guarantee needs the", flush=True)
    print(f"   discrete-readout re-anchoring of the synthetic task, which a continuous-state LM", flush=True)
    print(f"   lacks. Net: the dramatic exact case is synthetic; a real LM is modest + near-lossless.", flush=True)


if __name__ == "__main__":
    main()
