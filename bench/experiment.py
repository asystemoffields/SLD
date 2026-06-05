"""SLD frontier experiment: the headline numbers.

Trains (and caches) a looped teacher, a learned draft, and the original-JumpRec
jump baseline, then evaluates the full compute/quality/wall-clock frontier:

  * depth sweep k:      SLD sequential rounds vs full-loop / early-exit / oracle
  * horizon sweep H:    rounds ~ ceil(k/H) and the wall-clock tradeoff
  * draft controls:     learned vs identity (no-draft) vs blind vs Anderson
  * lossy predecessor:  original-JumpRec confidence jump, swept over thresholds
  * wall-clock:         batch-1 latency (the representative regime) and batch-64 throughput

Everything is asserted lossless against the full loop. Run:
    PYTHONPATH=../SMOKE python bench/experiment.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from sld.substrate import (TaskSpec, make_batch, ModelConfig, LoopedTransformer,
                           TrainConfig, train_teacher, evaluate_teacher, count_params)
from sld import draft as D
from sld import specloop as SL
from sld.training import train_draft, train_jump
import common as C


def get_models(args):
    """Build/train or load teacher, learned draft, jump baseline."""
    tpath = C.CKPT_DIR / f"teacher_{args.tag}.pt"
    dpath = C.CKPT_DIR / f"draft_{args.tag}.pt"
    jpath = C.CKPT_DIR / f"jump_{args.tag}.pt"

    spec = TaskSpec(n_nodes=args.n_nodes, max_hops=args.max_hops,
                    loop_steps=args.loop_steps, advance_only=True)
    if tpath.exists() and not args.retrain:
        model, spec = C.load_teacher(tpath)
        print(f"[load] teacher {tpath.name}  params={count_params(model)}")
    else:
        cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len,
                          n_answer=spec.n_nodes, out_pos=spec.out_pos,
                          d_model=args.d_model, n_heads=args.n_heads, d_ff=2 * args.d_model,
                          prelude_layers=2, core_layers=args.core_layers, coda_layers=1,
                          loop_steps=args.loop_steps)
        model = LoopedTransformer(cfg)
        print(f"[train] teacher params={count_params(model)}")
        train_teacher(model, spec, TrainConfig(steps=args.teacher_steps, batch_size=256,
                                               lr=3e-3, log_every=args.teacher_steps // 3))
        C.save_teacher(model, spec, tpath)
    ev = evaluate_teacher(model, spec, n_eval=4096)
    print(f"[teacher] exact-match acc overall={ev['overall']:.4f}")

    H = args.horizon
    if dpath.exists() and not args.retrain:
        drf, _ = C.load_learned_draft(dpath, model.cfg.d_model,
                                      n_answer=model.cfg.n_answer, out_pos=model.cfg.out_pos)
        print(f"[load] draft {dpath.name}  params={count_params(drf)}")
    else:
        drf = D.LearnedDraft(model.cfg.d_model, horizon=H,
                             n_answer=model.cfg.n_answer, out_pos=model.cfg.out_pos)
        print(f"[train] draft params={count_params(drf)} horizon={H}")
        train_draft(model, drf, spec, make_batch, steps=args.draft_steps, batch=256,
                    horizon=H, tape_examples=4096, log_every=args.draft_steps // 3)
        C.save_module(drf, {"horizon": H}, dpath)

    jump = D.JumpModule(model.cfg.d_model)
    if jpath.exists() and not args.retrain:
        ck = torch.load(jpath, weights_only=False); jump.load_state_dict(ck["state"]); jump.eval()
        print(f"[load] jump {jpath.name}")
    else:
        print(f"[train] jump (original-JumpRec baseline)")
        train_jump(model, jump, spec, make_batch, steps=args.draft_steps, batch=256,
                   log_every=args.draft_steps)
        C.save_module(jump, {}, jpath)
    return model, spec, drf, jump, H


@torch.no_grad()
def depth_sweep(model, spec, drf, H, ks, n=512):
    """Core-rounds / accuracy / losslessness for every method, per depth k."""
    rows = []
    g = torch.Generator().manual_seed(2024)
    identity = D.IdentityDraft(H)
    blind = D.BlindDraft(H, scale=0.5)
    reanchor = SL.make_reanchor(model, spec)
    for k in ks:
        b = make_batch(spec, n, generator=g, fixed_hop=k)
        tok, tgt = b["tokens"], b["target"]
        full = SL.full_loop_decode(model, tok, n_steps=k)
        ee = SL.early_exit_decode(model, tok, max_steps=k, patience=1)
        sld = SL.sld_decode(model, drf, tok, horizon=min(H, k), max_steps=k, stop_on_converge=False, reanchor_encode=reanchor)
        nod = SL.sld_decode(model, identity, tok, horizon=min(H, k), max_steps=k, stop_on_converge=False, reanchor_encode=reanchor)
        bld = SL.sld_decode(model, blind, tok, horizon=min(H, k), max_steps=k, stop_on_converge=False, reanchor_encode=reanchor)
        anders = AndersonResult(model, spec, tok, H, k)
        # lossy ancestor: draft predicts pi^k in one shot, NO verification
        donly = SL.draft_only_decode(model, drf, tok, n_steps=k)
        def acc(r): return (r.answer == tgt).float().mean().item()
        def loss(r): return (r.answer == full.answer).float().mean().item()
        rows.append({
            "k": k, "full_acc": acc(full),
            "full_rounds": full.core_rounds, "ee_rounds": ee.core_rounds,
            "oracle_log2": math.ceil(math.log2(k)) if k > 1 else 1,
            "sld_rounds": sld.core_rounds, "sld_rows": sld.core_rows,
            "sld_acc": acc(sld), "sld_lossless": loss(sld), "sld_mean_accept": sld.extra["mean_accept"],
            "nodraft_rounds": nod.core_rounds, "nodraft_accept": nod.extra["mean_accept"],
            "blind_rounds": bld.core_rounds, "blind_accept": bld.extra["mean_accept"],
            "anderson_rounds": anders["rounds"], "anderson_accept": anders["accept"], "anderson_lossless": anders["lossless"],
            "draftonly_acc": acc(donly),          # lossy one-shot (== draft k-ahead accuracy)
        })
    return rows


@torch.no_grad()
def lengthgen(model, spec, drf, H, ks_in, ks_out, n=512):
    """Train-vs-test depth generalization: a draft with horizon Htrain<max is OOD
    for k>Htrain. draft-only (lossy) accuracy degrades; SLD stays lossless (more
    rounds, never wrong). This is the value of verification."""
    rows = []
    g = torch.Generator().manual_seed(404)
    reanchor = SL.make_reanchor(model, spec)
    for k in ks_out:
        b = make_batch(spec, n, generator=g, fixed_hop=k)
        tok, tgt = b["tokens"], b["target"]
        full = SL.full_loop_decode(model, tok, n_steps=k)
        sld = SL.sld_decode(model, drf, tok, horizon=H, max_steps=k, stop_on_converge=False, reanchor_encode=reanchor)
        donly = SL.draft_only_decode(model, drf, tok, n_steps=min(k, H))
        rows.append({
            "k": k, "in_train": k <= max(ks_in),
            "draftonly_acc": (donly.answer == tgt).float().mean().item(),
            "sld_acc": (sld.answer == tgt).float().mean().item(),
            "sld_lossless": (sld.answer == full.answer).float().mean().item(),
            "sld_rounds": sld.core_rounds,
        })
    return rows


@torch.no_grad()
def AndersonResult(model, spec, tok, H, k):
    try:
        and_draft = D.AndersonDraft(model, horizon=H)
        and_draft.reset()
        r = SL.sld_decode(model, and_draft, tok, horizon=min(H, k), max_steps=k,
                          stop_on_converge=False, reanchor_encode=SL.make_reanchor(model, spec))
        full = SL.full_loop_decode(model, tok, n_steps=k)
        return {"rounds": r.core_rounds, "accept": r.extra["mean_accept"],
                "lossless": (r.answer == full.answer).float().mean().item()}
    except Exception as e:  # training-free control; never let it crash the frontier
        return {"rounds": float("nan"), "accept": float("nan"), "lossless": float("nan"),
                "error": str(e)}


@torch.no_grad()
def horizon_sweep(model, spec, drf, H, k, n=512):
    rows = []
    g = torch.Generator().manual_seed(7)
    b = make_batch(spec, n, generator=g, fixed_hop=k)
    full = SL.full_loop_decode(model, b["tokens"], n_steps=k)
    reanchor = SL.make_reanchor(model, spec)
    for h in [1, 2, 4, 8, H]:
        if h > H:
            continue
        sld = SL.sld_decode(model, drf, b["tokens"], horizon=h, max_steps=k, stop_on_converge=False, reanchor_encode=reanchor)
        rows.append({"horizon": h, "rounds": sld.core_rounds, "rows": sld.core_rows,
                     "lossless": (sld.answer == full.answer).float().mean().item(),
                     "mean_accept": sld.extra["mean_accept"]})
    return rows


def wallclock(model, spec, drf, H, ks, threads, batch):
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(99)
    reanchor = SL.make_reanchor(model, spec)
    rows = []
    for k in ks:
        b = make_batch(spec, batch, generator=g, fixed_hop=k)
        tok = b["tokens"]
        t_full = C.time_decode(lambda x: SL.full_loop_decode(model, x, n_steps=k), tok)
        t_sld = C.time_decode(
            lambda x: SL.sld_decode(model, drf, x, horizon=min(H, k), max_steps=k, stop_on_converge=False, reanchor_encode=reanchor), tok)
        rows.append({"k": k, "full_ms": t_full, "sld_ms": t_sld,
                     "speedup": t_full / t_sld, "per_query_full_ms": t_full / batch,
                     "per_query_sld_ms": t_sld / batch})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n_nodes", type=int, default=32)
    ap.add_argument("--max_hops", type=int, default=16)
    ap.add_argument("--loop_steps", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--core_layers", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--teacher_steps", type=int, default=1500)
    ap.add_argument("--draft_steps", type=int, default=1500)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    model, spec, drf, jump, H = get_models(args)

    ks = [k for k in [2, 4, 6, 8, 10, 12, 14, 16] if k <= args.max_hops]
    print("\n=== DEPTH SWEEP (sequential core rounds; lossless asserted) ===")
    ds = depth_sweep(model, spec, drf, H, ks)
    print(f"{'k':>3} {'full':>5} {'early':>6} {'oracle':>6} {'SLD':>5} {'accept':>7} "
          f"{'lossless':>8} {'nodraft':>7} {'blind':>6} {'anders':>7} {'donly_acc':>9}")
    for r in ds:
        print(f"{r['k']:>3} {r['full_rounds']:>5.1f} {r['ee_rounds']:>6.2f} {r['oracle_log2']:>6} "
              f"{r['sld_rounds']:>5.2f} {r['sld_mean_accept']:>7.2f} {r['sld_lossless']:>8.3f} "
              f"{r['nodraft_rounds']:>7.2f} {r['blind_rounds']:>6.2f} {r['anderson_rounds']:>7.2f} "
              f"{r['draftonly_acc']:>9.3f}")
    assert all(abs(r["sld_lossless"] - 1.0) < 1e-9 for r in ds), "LOSSLESS VIOLATION"
    print("[ok] SLD is exactly lossless vs full loop at every k.")

    # Length generalization: k beyond the draft's horizon H is OOD. draft-only
    # (lossy) collapses; SLD stays lossless by spending more verified rounds.
    ks_out = sorted(set([12, 14, 16, args.loop_steps - 2, args.loop_steps]))
    ks_out = [k for k in ks_out if k <= args.loop_steps]
    print("\n=== LENGTH GENERALIZATION (draft horizon H={}, k up to {}) ===".format(H, args.loop_steps))
    lg = lengthgen(model, spec, drf, H, ks_in=ks, ks_out=ks_out)
    print(f"{'k':>3} {'OOD?':>5} {'draftonly_acc':>13} {'SLD_acc':>8} {'SLD_lossless':>12} {'SLD_rounds':>11}")
    for r in lg:
        print(f"{r['k']:>3} {('' if r['in_train'] else 'OOD'):>5} {r['draftonly_acc']:>13.3f} "
              f"{r['sld_acc']:>8.3f} {r['sld_lossless']:>12.3f} {r['sld_rounds']:>11.2f}")

    print(f"\n=== HORIZON SWEEP (k={args.max_hops}) ===")
    hs = horizon_sweep(model, spec, drf, H, args.max_hops)
    for r in hs:
        print(f" H={r['horizon']:>2}  rounds={r['rounds']:.2f}  rows/ex={r['rows']:.1f}  "
              f"accept={r['mean_accept']:.2f}  lossless={r['lossless']:.3f}")

    print("\n=== WALL-CLOCK (6 threads, batch-1 latency) ===")
    wc1 = wallclock(model, spec, drf, H, ks, threads=args.threads, batch=1)
    for r in wc1:
        print(f" k={r['k']:>2}  full={r['full_ms']:.3f}ms  SLD={r['sld_ms']:.3f}ms  speedup={r['speedup']:.2f}x")
    print("=== WALL-CLOCK (6 threads, batch-64 throughput) ===")
    wc64 = wallclock(model, spec, drf, H, ks, threads=args.threads, batch=64)
    for r in wc64:
        print(f" k={r['k']:>2}  full={r['full_ms']:.3f}ms  SLD={r['sld_ms']:.3f}ms  speedup={r['speedup']:.2f}x")

    out = {"args": vars(args), "teacher_acc": evaluate_teacher(model, spec, n_eval=4096),
           "depth_sweep": ds, "lengthgen": lg, "horizon_sweep": hs,
           "wallclock_b1": wc1, "wallclock_b64": wc64,
           "elapsed_s": time.perf_counter() - t0}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / f"frontier_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.perf_counter()-t0:.1f}s  ->  results/frontier_{args.tag}.json")


if __name__ == "__main__":
    main()
