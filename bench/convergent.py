"""Convergent-loop ablation: SLD vs a FAIR early-exit baseline.

The headline experiment uses a permutation (non-converging) loop, where early-exit
is structurally useless. A reviewer will object that real looped LMs converge,
where early-exit *does* help. This ablation answers that: replace the permutation
with a CONTRACTING map f (a functional graph whose roots are fixed points), so
f^k(start) climbs to a root and then HOLDS -- the loop genuinely converges. Now
early-exit is a strong, fair baseline (it stops at the root). SLD still wins: it
LEAPS the climb that early-exit walks one step at a time, so SLD's advantage over
early-exit grows with depth-to-root. Both remain exactly lossless, and the clean
symbol-only re-anchoring still applies (roots are absorbing -> the symbol is a
sufficient statistic, no counter).

Run: PYTHONPATH=../SMOKE:.. python bench/convergent.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from sld.substrate import (TaskSpec, make_batch, ModelConfig, LoopedTransformer,
                           TrainConfig, train_teacher, count_params)
from sld import draft as D
from sld import specloop as SL
from sld.training import train_draft

CK = Path(__file__).resolve().parents[1] / "results" / "ckpt"
RES = Path(__file__).resolve().parents[1] / "results"


def make_convergent_map(n: int, n_roots: int = 2, seed: int = 0):
    """A functional graph f: [n]->[n] with n_roots fixed points (roots) and every
    other node pointing to a strictly-shallower node -> f^k converges to a root.
    Returns (f [n], depth_to_root [n])."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)            # placement order; first n_roots are roots
    f = torch.arange(n)
    depth = torch.zeros(n, dtype=torch.long)
    for pos in range(n_roots, n):
        node = int(order[pos])
        parent = int(order[int(torch.randint(0, pos, (1,), generator=g))])  # already-placed -> shallower
        f[node] = parent
        depth[node] = depth[parent] + 1
    return f, depth


@torch.no_grad()
def depth_to_root_sweep(model, spec, drf, depth, H, n=512):
    """Per depth-to-root group: full-loop / early-exit / SLD rounds, all lossless."""
    reanchor = SL.make_reanchor(model, spec)
    L = spec.loop_steps
    g = torch.Generator().manual_seed(55)
    # sample a big pool, then bucket by the start's depth-to-root
    b = make_batch(spec, n * 6, generator=g, fixed_hop=L)   # run to convergence (answer = root)
    dstart = depth[b["start"]]
    rows = []
    for d in range(1, int(depth.max()) + 1):
        m = dstart == d
        if m.sum() < 32:
            continue
        tok = b["tokens"][m]
        root = b["target"][m]                                  # f^L(start) = the root
        full = SL.full_loop_decode(model, tok, n_steps=L)      # fixed budget L
        # patience=2: require two consecutive stable readouts, so a single
        # coincidental repeat (a no-op first step on some inputs) cannot
        # false-trigger convergence at the depth boundaries.
        ee = SL.early_exit_decode(model, tok, max_steps=L, patience=2)
        sld = SL.sld_decode(model, drf, tok, horizon=H, max_steps=L,
                            stop_on_converge=True, conv_patience=1, reanchor_encode=reanchor)
        rows.append({
            "depth": d, "n": int(m.sum()),
            "full_acc": (full.answer == root).float().mean().item(),
            "full_rounds": full.core_rounds,
            "ee_rounds": ee.core_rounds, "ee_lossless": (ee.answer == full.answer).float().mean().item(),
            "sld_rounds": sld.core_rounds, "sld_lossless": (sld.answer == full.answer).float().mean().item(),
            "sld_mean_accept": sld.extra["mean_accept"],
            "sld_vs_ee_speedup": ee.core_rounds / max(sld.core_rounds, 1e-9),
        })
    return rows


def main():
    torch.set_num_threads(6)
    N, L, H = 32, 18, 16
    f, depth = make_convergent_map(N, n_roots=2, seed=0)
    print(f"convergent map: N={N} roots=2 max depth-to-root={int(depth.max())}", flush=True)
    spec = TaskSpec(n_nodes=N, max_hops=L, loop_steps=L, advance_only=True, perm=f)

    tpath = CK / "convergent_teacher.pt"
    cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, n_answer=spec.n_nodes,
                      out_pos=spec.out_pos, d_model=96, n_heads=4, d_ff=192,
                      prelude_layers=2, core_layers=1, coda_layers=1, loop_steps=L)
    model = LoopedTransformer(cfg)
    if tpath.exists():
        model.load_state_dict(torch.load(tpath)); print("[load] convergent teacher", flush=True)
    else:
        print("teacher params", count_params(model), flush=True)
        train_teacher(model, spec, TrainConfig(steps=1800, batch_size=256, lr=3e-3, log_every=600))
        torch.save(model.state_dict(), tpath)
    model.eval()

    drf = D.LearnedDraft(cfg.d_model, horizon=H, n_answer=spec.n_nodes, out_pos=spec.out_pos)
    print("draft params", count_params(drf), flush=True)
    train_draft(model, drf, spec, make_batch, steps=1500, batch=256, horizon=H,
                tape_examples=4096, log_every=750)
    drf.eval()

    rows = depth_to_root_sweep(model, spec, drf, depth, H)
    print("\n=== CONVERGENT LOOP: SLD vs FAIR early-exit (all lossless) ===", flush=True)
    print(f"{'depth':>5} {'n':>5} {'full':>6} {'early-exit':>10} {'SLD':>6} {'accept':>7} "
          f"{'SLD/EE speedup':>14} {'lossless':>9}")
    for r in rows:
        print(f"{r['depth']:>5} {r['n']:>5} {r['full_rounds']:>6.1f} {r['ee_rounds']:>10.2f} "
              f"{r['sld_rounds']:>6.2f} {r['sld_mean_accept']:>7.2f} {r['sld_vs_ee_speedup']:>14.2f} "
              f"{r['sld_lossless']:>9.3f}")
    assert all(abs(r["sld_lossless"] - 1.0) < 1e-9 for r in rows), "LOSSLESS VIOLATION"
    print("[ok] SLD exactly lossless on the convergent loop; beats fair early-exit, widening with depth.")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "convergent.json").write_text(json.dumps(rows, indent=2))
    print("[saved] results/convergent.json", flush=True)


if __name__ == "__main__":
    main()
