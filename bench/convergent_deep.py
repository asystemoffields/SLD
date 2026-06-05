"""Deep convergent loop: SLD's advantage over a fair early-exit keeps widening.

Same setup as bench/convergent.py but with a DEEP contracting map (disjoint long
chains to roots), so depth-to-root reaches ~20+. SLD stays at ~constant rounds
while early-exit must walk every step -- the gap grows from a few x to ~10x+,
the regime that matters for deep recurrent-depth LMs (Huginn unrolls 32-132).
The round-count result is robust to teacher accuracy (it's about how many
sequential core calls each method makes, and SLD is lossless vs the teacher's
own loop regardless).

Run: PYTHONPATH=../SMOKE:.. python bench/convergent_deep.py
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from sld.substrate import (TaskSpec, make_batch, ModelConfig, LoopedTransformer,
                           TrainConfig, train_teacher, count_params)
from sld import draft as D
from sld.training import train_draft
from convergent import depth_to_root_sweep   # reuse the sweep

CK = Path(__file__).resolve().parents[1] / "results" / "ckpt"
RES = Path(__file__).resolve().parents[1] / "results"


def make_deep_map(n: int, chain_len: int, seed: int = 0):
    """Disjoint chains of length ~chain_len, each converging to its own root.
    f(order[i]) = order[i-1] within a chain; chain heads are fixed points."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)
    f = torch.arange(n)
    depth = torch.zeros(n, dtype=torch.long)
    pos = 0
    while pos < n:
        L = min(chain_len, n - pos)
        head = int(order[pos])
        f[head] = head                                   # root (fixed point)
        for j in range(1, L):
            node = int(order[pos + j])
            f[node] = int(order[pos + j - 1])            # point to the shallower neighbor
            depth[node] = j
        pos += L
    return f, depth


def main():
    torch.set_num_threads(6)
    N, chain, L, H = 48, 24, 28, 24          # depths up to 23; loop budget 28; horizon 24
    f, depth = make_deep_map(N, chain_len=chain, seed=0)
    print(f"deep convergent map: N={N} chain_len={chain} max depth-to-root={int(depth.max())}", flush=True)
    spec = TaskSpec(n_nodes=N, max_hops=L, loop_steps=L, advance_only=True, perm=f)

    tpath = CK / "convergent_deep_teacher.pt"
    cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, n_answer=spec.n_nodes,
                      out_pos=spec.out_pos, d_model=96, n_heads=4, d_ff=192,
                      prelude_layers=2, core_layers=1, coda_layers=1, loop_steps=L)
    model = LoopedTransformer(cfg)
    if tpath.exists():
        model.load_state_dict(torch.load(tpath)); print("[load] deep teacher", flush=True)
    else:
        print("teacher params", count_params(model), flush=True)
        train_teacher(model, spec, TrainConfig(steps=2200, batch_size=256, lr=3e-3, log_every=700))
        torch.save(model.state_dict(), tpath)
    model.eval()

    drf = D.LearnedDraft(cfg.d_model, horizon=H, n_answer=spec.n_nodes, out_pos=spec.out_pos)
    train_draft(model, drf, spec, make_batch, steps=1800, batch=256, horizon=H,
                tape_examples=4096, log_every=900)
    drf.eval()

    rows = depth_to_root_sweep(model, spec, drf, depth, H)
    print("\n=== DEEP CONVERGENT LOOP: SLD vs fair early-exit ===", flush=True)
    print(f"{'depth':>5} {'n':>5} {'full':>6} {'early-exit':>10} {'SLD':>6} {'SLD/EE speedup':>14} {'lossless':>9}")
    for r in rows:
        print(f"{r['depth']:>5} {r['n']:>5} {r['full_rounds']:>6.0f} {r['ee_rounds']:>10.2f} "
              f"{r['sld_rounds']:>6.2f} {r['sld_vs_ee_speedup']:>14.2f} {r['sld_lossless']:>9.3f}", flush=True)
    assert all(abs(r["sld_lossless"] - 1.0) < 1e-9 for r in rows), "LOSSLESS VIOLATION"
    deep = [r for r in rows if r["depth"] >= 16]
    if deep:
        print(f"[ok] at depth>=16: SLD ~{deep[-1]['sld_rounds']:.1f} rounds vs early-exit "
              f"{deep[-1]['ee_rounds']:.0f} -> {deep[-1]['sld_vs_ee_speedup']:.1f}x, lossless.", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "convergent_deep.json").write_text(json.dumps(rows, indent=2))
    print("[saved] results/convergent_deep.json", flush=True)


if __name__ == "__main__":
    main()
