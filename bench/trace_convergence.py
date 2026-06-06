"""Look at the convergence geometry per step, to find a CHEAP signal that locks
when the token locks. Candidate signals, last position only:
  - x-cos   : cosine(x_t[-1], x_{t-1}[-1])           (raw recurrent state)
  - x-rel   : ||dx|| / ||x||  (raw recurrent state)
  - h-cos   : cosine(h_t, h_{t-1}), h = ln_f(coda(x))[-1]   (pre-lm_head hidden)
  - h-rel   : ||dh|| / ||h||
  - margin  : top1 - top2 logit gap at the last position
vs whether the argmax token == the full-T-loop token.
"""
from __future__ import annotations
import torch
from parcae_sld import ParcaeLoop
torch.set_num_threads(6)


@torch.no_grad()
def cos(a, b): return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
@torch.no_grad()
def rel(a, b): return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-9)).item()


@torch.no_grad()
def hidden(loop, x, ids, fc):
    m = loop.m; h = m.transformer.C(x)
    for i, blk in enumerate(m.transformer.coda):
        k = str(loop.off_coda + i)
        ve = m.value_embeds[k](ids) if k in m.value_embeds else None
        h = blk(h, fc, None, ve=ve)
    return m.transformer.ln_f(h)[:, -1, :]                  # [1, H]


def main():
    import parcae_lm
    from transformers import AutoTokenizer
    from datasets import load_dataset
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    tok = AutoTokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    T = int(getattr(m.config, "mean_recurrence", 8))
    loop = ParcaeLoop(m)

    for i in range(5):
        ids = tok(ds[i]["text"], return_tensors="pt").input_ids
        ctx = ids[:, :-1]
        x, e, fc = loop.encode(ctx)
        xs, hs, toks = [x], [], []
        for t in range(1, T + 1):
            x = loop.step(x, e, fc); xs.append(x)
            h = hidden(loop, x, ctx, fc); hs.append(h)
            lg = (m.lm_head(h).float() * loop.logit_scale)[0]
            top2 = lg.topk(2).values
            toks.append((int(lg.argmax()), (top2[0] - top2[1]).item()))
        final = toks[-1][0]
        print(f"\n--- ex {i} (seq={ctx.shape[1]}) final_token={final} ---", flush=True)
        print(f"{'t':>2}{'tok':>7}{'lock':>5}{'x-cos':>8}{'x-rel':>9}{'h-cos':>8}{'h-rel':>9}{'margin':>9}", flush=True)
        for t in range(T):
            xc = cos(xs[t+1][:, -1], xs[t][:, -1]); xr = rel(xs[t+1][:, -1], xs[t][:, -1])
            hc = cos(hs[t], hs[t-1]) if t > 0 else float('nan')
            hr = rel(hs[t], hs[t-1]) if t > 0 else float('nan')
            tk, mg = toks[t]
            print(f"{t+1:>2}{tk:>7}{'Y' if tk==final else '.':>5}{xc:>8.4f}{xr:>9.3f}"
                  f"{hc:>8.4f}{hr:>9.3f}{mg:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
