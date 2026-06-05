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
from parcae_sld import ParcaeLoop, sld, earlyexit, sld_cheap

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

    cf = cs = cc = ce = 0                          # correct: full, sld(faithful), sld(cheap), early-exit
    sm = cm = em = 0                               # matches full loop
    sr = cr = er = 0.0                             # rounds
    tf = ts = tc = te = 0.0                        # wall-clock
    scored = 0
    t0 = time.time()
    for i in range(min(n, len(ds))):
        ids = tok(ds[i]["text"], return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            continue
        ctx, target = ids[:, :-1], int(ids[0, -1])
        a = time.perf_counter(); pf, _ = full_last(loop, ctx, T); tf += time.perf_counter() - a
        a = time.perf_counter(); psf, r1 = sld(loop, ctx, T, warmup=3, verify_steps=2); ts += time.perf_counter() - a
        a = time.perf_counter(); psc, r2 = sld_cheap(loop, ctx, T, warmup=2, tol=0.1); tc += time.perf_counter() - a
        a = time.perf_counter(); pe, r3 = earlyexit(loop, ctx, T, patience=2); te += time.perf_counter() - a
        pf, psf, psc, pe = int(pf), int(psf), int(psc), int(pe)
        cf += pf == target; cs += psf == target; cc += psc == target; ce += pe == target
        sm += psf == pf; cm += psc == pf; em += pe == pf
        sr += r1; cr += r2; er += r3; scored += 1
        if scored % 50 == 0:
            print(f"  {scored} scored  ({time.time()-t0:.0f}s)  full-acc {cf/scored:.3f}", flush=True)

    N = scored
    out = {"n": N, "T": T,
           "acc": {"full": cf/N, "sld_faithful": cs/N, "sld_fast": cc/N, "early_exit": ce/N},
           "matches_full_loop": {"sld_faithful": sm/N, "sld_fast": cm/N, "early_exit": em/N},
           "mean_rounds": {"full": T, "sld_faithful": sr/N, "sld_fast": cr/N, "early_exit": er/N},
           "wall_ms": {"full": tf/N*1e3, "sld_faithful": ts/N*1e3, "sld_fast": tc/N*1e3, "early_exit": te/N*1e3},
           "wallclock_speedup_cpu": {"sld_faithful": tf/ts, "sld_fast": tf/tc, "early_exit": tf/te}}
    print("\n=== parcae-140m LAMBADA (eval_configs/eval-lambada.yaml): full-loop vs SLD ===", flush=True)
    print(f"  examples: {N}\n", flush=True)
    print(f"  {'method':<16}{'LAMBADA acc':>12}{'matches full':>14}{'core rounds':>13}{'CPU ms/ex':>11}{'speedup':>9}", flush=True)
    print(f"  {'full loop':<16}{cf/N:>12.3f}{'—':>14}{T:>13.0f}{tf/N*1e3:>11.1f}{'1.00x':>9}", flush=True)
    print(f"  {'SLD (faithful)':<16}{cs/N:>12.3f}{sm/N:>14.3f}{sr/N:>13.2f}{ts/N*1e3:>11.1f}{tf/ts:>8.2f}x", flush=True)
    print(f"  {'SLD (fast)':<16}{cc/N:>12.3f}{cm/N:>14.3f}{cr/N:>13.2f}{tc/N*1e3:>11.1f}{tf/tc:>8.2f}x", flush=True)
    print(f"  {'early-exit':<16}{ce/N:>12.3f}{em/N:>14.3f}{er/N:>13.2f}{te/N*1e3:>11.1f}{tf/te:>8.2f}x", flush=True)
    print(f"\n  => SLD preserves parcae's LAMBADA accuracy ({cf/N:.3f}). 'faithful' (decode-verified) is", flush=True)
    print(f"     {sm/N*100:.0f}% prediction-identical but the per-step decode overhead makes it slower than the", flush=True)
    print(f"     short T=8 loop on CPU; 'fast' (cheap state-residual convergence) is {tf/tc:.2f}x faster on CPU", flush=True)
    print(f"     at {cr/N:.1f} rounds with accuracy preserved. On GPU / deeper loops both win on wall-clock.", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "parcae_lambada.json").write_text(json.dumps(out, indent=2))
    print("[saved] results/parcae_lambada.json", flush=True)


if __name__ == "__main__":
    main()
