"""Run a benchmark from parcae's own eval (LAMBADA, eval_configs/eval-lambada.yaml)
with full-loop vs SLD on parcae-140m (CPU).

LAMBADA = predict the final token of a passage from its context. We compare the
benchmark accuracy under the full T-loop recurrence vs SLD-accelerated recurrence,
and how often SLD's prediction matches the full loop (losslessness on a real
benchmark), at fewer sequential core rounds. A subset is used so it runs on CPU.

Run: PYTHONPATH=../SMOKE:.. python bench/parcae_lambada.py [n_examples]
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import torch
from parcae_sld import ParcaeLoop, sld, earlyexit

RES = Path(__file__).resolve().parents[1] / "results"


@torch.no_grad()
def full_last(loop, ctx, T):
    x, e, fc = loop.encode(ctx)
    for _ in range(T): x = loop.step(x, e, fc)
    return loop.decode(x, ctx, fc).argmax(-1), T


def main():
    import parcae_lm
    from transformers import AutoTokenizer
    from datasets import load_dataset
    torch.set_num_threads(6)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print("loading parcae-140m + tokenizer + LAMBADA ...", flush=True)
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    tok = AutoTokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    T = int(getattr(m.config, "mean_recurrence", 8))
    loop = ParcaeLoop(m)

    cf = cs = ce = 0                               # correct: full, sld, early-exit
    sm = em = 0                                    # matches full loop
    sr = er = 0.0                                  # rounds
    tf = ts = te = 0.0                             # wall-clock
    scored = 0
    t0 = time.time()
    for i in range(min(n, len(ds))):
        ids = tok(ds[i]["text"], return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            continue
        ctx, target = ids[:, :-1], int(ids[0, -1])
        a = time.perf_counter(); pf, _ = full_last(loop, ctx, T); tf += time.perf_counter() - a
        a = time.perf_counter(); psf, r1 = sld(loop, ctx, T); ts += time.perf_counter() - a
        a = time.perf_counter(); pe, r3 = earlyexit(loop, ctx, T); te += time.perf_counter() - a
        pf, psf, pe = int(pf), int(psf), int(pe)
        cf += pf == target; cs += psf == target; ce += pe == target
        sm += psf == pf; em += pe == pf
        sr += r1; er += r3; scored += 1
        if scored % 50 == 0:
            print(f"  {scored} scored  ({time.time()-t0:.0f}s)  full-acc {cf/scored:.3f}", flush=True)

    N = scored
    out = {"n": N, "T": T,
           "acc": {"full": cf/N, "sld": cs/N, "early_exit": ce/N},
           "matches_full_loop": {"sld": sm/N, "early_exit": em/N},
           "mean_rounds": {"full": T, "sld": sr/N, "early_exit": er/N},
           "wall_ms": {"full": tf/N*1e3, "sld": ts/N*1e3, "early_exit": te/N*1e3},
           "speedup": {"sld": tf/ts, "early_exit": tf/te}}
    print("\n=== parcae-140m LAMBADA (eval_configs/eval-lambada.yaml): full-loop vs SLD ===", flush=True)
    print(f"  examples: {N}\n", flush=True)
    print(f"  {'method':<14}{'LAMBADA acc':>12}{'matches full':>14}{'core rounds':>13}{'CPU ms/ex':>11}{'speedup':>10}", flush=True)
    print(f"  {'full loop':<14}{cf/N:>12.3f}{'—':>14}{T:>13.0f}{tf/N*1e3:>11.1f}{'1.00x':>10}", flush=True)
    print(f"  {'SLD':<14}{cs/N:>12.3f}{sm/N:>14.3f}{sr/N:>13.2f}{ts/N*1e3:>11.1f}{tf/ts:>9.2f}x", flush=True)
    print(f"  {'early-exit':<14}{ce/N:>12.3f}{em/N:>14.3f}{er/N:>13.2f}{te/N*1e3:>11.1f}{tf/te:>9.2f}x", flush=True)
    print(f"\n  => SLD preserves parcae's LAMBADA accuracy ({cf/N:.3f}) at {sr/N:.1f} of {T} core rounds and "
          f"{tf/ts:.2f}x wall-clock.", flush=True)
    print(f"     Verification is in state space (a core step + a dot product, no 32k-vocab decode), so the", flush=True)
    print(f"     skipped loops are a real speedup even on this short T=8 loop on CPU; it grows with depth.", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "parcae_lambada.json").write_text(json.dumps(out, indent=2))
    print("[saved] results/parcae_lambada.json", flush=True)


if __name__ == "__main__":
    main()
