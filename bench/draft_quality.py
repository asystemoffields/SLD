"""Draft quality -> acceptance -> speedup (always lossless).

SLD's speedup is governed entirely by how often the draft is right: verification
guarantees correctness regardless, so a worse draft costs *more rounds*, never
*accuracy*. We make this precise by taking the trained (near-perfect) draft and
corrupting a controlled fraction `p` of its predicted symbols, sweeping `p` from 0
(the learned draft) to 1 (random, = the blind control). At every `p` SLD stays
exactly lossless; the accepted-prefix length and the sequential-core-round count
move smoothly between "1 round" and "full loop".

This is the lever that the GPU/real-model path turns: a stronger draft (more
capacity/training, which a GPU affords) raises acceptance and shrinks rounds.

Run: PYTHONPATH=../SMOKE:.. python bench/draft_quality.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sld import specloop as SL
import common as C

RES = Path(__file__).resolve().parents[1] / "results"


class NoisyDraft:
    """Wraps a symbol-head draft and corrupts a fraction p of its predictions."""

    def __init__(self, inner, p: float, n_nodes: int, seed: int = 0):
        self.inner = inner
        self.sym_net = inner.sym_net          # so SLD uses the cheap symbol path
        self.horizon = inner.horizon
        self.p = p
        self.n_nodes = n_nodes
        self.g = torch.Generator().manual_seed(seed)

    def propose_symbols(self, h, h0, horizon=None):
        s = self.inner.propose_symbols(h, h0, horizon)
        if self.p <= 0:
            return s
        mask = torch.rand(s.shape, generator=self.g) < self.p
        rand = torch.randint(0, self.n_nodes, s.shape, generator=self.g)
        return torch.where(mask, rand, s)

    # propose() is only used by the continuous path; SLD uses propose_symbols here.
    def propose(self, h, h0, horizon=None):
        return self.inner.propose(h, h0, horizon)


@torch.no_grad()
def main():
    tag = "main"
    model, spec = C.load_teacher(C.CKPT_DIR / f"teacher_{tag}.pt")
    drf, meta = C.load_learned_draft(C.CKPT_DIR / f"draft_{tag}.pt", model.cfg.d_model,
                                     n_answer=model.cfg.n_answer, out_pos=model.cfg.out_pos)
    H = meta["horizon"]
    reanchor = SL.make_reanchor(model, spec)
    k = spec.max_hops
    g = torch.Generator().manual_seed(2)
    b = SL.__dict__  # noqa  (silence linters)
    from sld.substrate import make_batch
    batch = make_batch(spec, 512, generator=g, fixed_hop=k)
    tok = batch["tokens"]
    full = SL.full_loop_decode(model, tok, n_steps=k)

    print(f"draft-quality sweep at k={k}, horizon={H}  (full loop = {k} rounds)")
    print(f"{'corrupt p':>9} {'sym acc':>8} {'mean accept':>12} {'SLD rounds':>11} {'lossless':>9}")
    rows = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        nd = NoisyDraft(drf, p, spec.n_nodes, seed=7)
        # measure the draft's realized per-symbol accuracy vs the true trajectory
        h0 = model.encode(tok)
        pred = nd.propose_symbols(h0, h0, H)                       # [B,H]
        tgt = batch["traj_target"][:, 1:H + 1]
        sym_acc = (pred == tgt).float().mean().item()
        sld = SL.sld_decode(model, nd, tok, horizon=H, max_steps=k,
                            stop_on_converge=False, reanchor_encode=reanchor)
        lossless = (sld.answer == full.answer).float().mean().item()
        rows.append({"p": p, "sym_acc": sym_acc, "mean_accept": sld.extra["mean_accept"],
                     "sld_rounds": sld.core_rounds, "lossless": lossless})
        print(f"{p:>9.2f} {sym_acc:>8.3f} {sld.extra['mean_accept']:>12.2f} "
              f"{sld.core_rounds:>11.2f} {lossless:>9.3f}")
    assert all(abs(r["lossless"] - 1.0) < 1e-9 for r in rows), "LOSSLESS VIOLATION"
    print("[ok] exactly lossless at every draft quality; rounds degrade smoothly to the full loop.")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "draft_quality.json").write_text(json.dumps(rows, indent=2))
    print("[saved] results/draft_quality.json")


if __name__ == "__main__":
    main()
