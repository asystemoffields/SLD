"""In-context-map SLD experiment (non-memorizable).

Unlike the fixed-permutation task, here the permutation is DIFFERENT for every
example and is given IN the prompt as shuffled (key, value) pairs:

    [BOS][MAP] k0 v0 k1 v1 ... k_{N-1} v_{N-1} [Q] start [OUT]

The answer is pi^k(start) for THIS example's pi. A draft cannot memorize pi^i;
it must read the map from the recurrent state and compose it. This is the
stress test that SLD's win is not a memorization artifact: the looped teacher
genuinely chases an arbitrary in-context pointer, the draft does in-context
composition, and SLD stays exactly lossless (re-anchoring re-encodes the prompt
with the current node as the new start; the map -- the rest of the prompt -- is
the sufficient statistic alongside the node).

This module is self-contained (its own task generator + attention draft) so the
clean fixed-permutation substrate is untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sld.substrate import ModelConfig, LoopedTransformer, count_params
from jumprec.model import Block


@dataclass
class ICSpec:
    n_nodes: int = 8
    loop_steps: int = 12

    BOS = 0
    MAP = 1
    Q = 2
    OUT = 3
    PAD = 4

    @property
    def node_base(self): return 5

    @property
    def vocab_size(self): return self.node_base + self.n_nodes

    @property
    def seq_len(self): return 2 + 2 * self.n_nodes + 3   # BOS MAP pairs Q start OUT

    @property
    def start_pos(self): return 2 + 2 * self.n_nodes + 1  # the start node token position

    @property
    def out_pos(self): return self.seq_len - 1


def ic_batch(spec: ICSpec, B, *, generator=None, fixed_hop=None, hop_high=None, device="cpu"):
    N, L = spec.n_nodes, spec.loop_steps
    hop_high = L if hop_high is None else hop_high
    # vectorized random permutations (one per example): pi[b,i] = pi_b(i)
    perms = torch.rand(B, N, generator=generator).argsort(1)
    starts = torch.randint(0, N, (B,), generator=generator)
    if fixed_hop is not None:
        hops = torch.full((B,), fixed_hop, dtype=torch.long)
    else:
        hops = torch.randint(1, hop_high + 1, (B,), generator=generator)
    # trajectory pi^j(start)
    traj = torch.empty(B, L + 1, dtype=torch.long)
    cur = starts.clone(); traj[:, 0] = cur
    for j in range(1, L + 1):
        cur = perms.gather(1, cur.unsqueeze(1)).squeeze(1)
        traj[:, j] = cur
    target = traj.gather(1, hops.unsqueeze(1)).squeeze(1)

    tokens = torch.full((B, spec.seq_len), spec.PAD, dtype=torch.long)
    tokens[:, 0] = spec.BOS
    tokens[:, 1] = spec.MAP
    # shuffled (key,value) pairs so the model must use attention, not position
    order = torch.rand(B, N, generator=generator).argsort(1)          # presentation order per row
    keys = spec.node_base + order                                     # [B,N]
    values = spec.node_base + perms.gather(1, order)                  # pi(order)
    tokens[:, 2:2 + 2 * N:2] = keys
    tokens[:, 3:2 + 2 * N:2] = values
    tokens[:, 2 + 2 * N] = spec.Q
    tokens[:, spec.start_pos] = spec.node_base + starts
    tokens[:, spec.out_pos] = spec.OUT
    return {"tokens": tokens.to(device), "target": target.to(device), "hops": hops.to(device),
            "start": starts.to(device), "perm": perms.to(device), "traj_target": traj.to(device)}


def make_ic_reanchor(model, spec, base_tokens):
    """Re-anchor for in-context: rebuild each example's prompt with start := current
    node (same map), then encode. base_tokens are this batch's prompts (carry the map)."""
    @torch.no_grad()
    def reanchor(nodes, idx):
        toks = base_tokens[idx].clone()
        toks[:, spec.start_pos] = spec.node_base + nodes
        return model.encode(toks)
    return reanchor


def train_ic_teacher(model, spec, steps=2500, batch=256, lr=2e-3, log_every=500, seed=0, device="cpu"):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train(); t0 = time.perf_counter()
    import math
    for step in range(steps):
        lr_t = lr * (0.5 * (1 + math.cos(math.pi * step / steps)))
        for pg in opt.param_groups: pg["lr"] = lr_t
        b = ic_batch(spec, batch, generator=g, device=device)
        logits, traj = model.run_loop(b["tokens"], record=True)
        tgt = b["traj_target"]
        loss = F.cross_entropy(logits, tgt[:, -1])
        ds = 0.0
        for tstep in range(1, len(traj)):
            ds = ds + F.cross_entropy(model.decode(traj[tstep]), tgt[:, tstep])
        loss = loss + ds / max(1, len(traj) - 1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % log_every == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == tgt[:, -1]).float().mean().item()
            print(f"[ic-teacher] step {step:5d} loss {loss.item():.4f} acc {acc:.3f} "
                  f"{time.perf_counter()-t0:.1f}s")
    return model


@torch.no_grad()
def ic_eval(model, spec, n=2048, seed=999, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    b = ic_batch(spec, n, generator=g, device=device)
    # run per-example k steps
    h0 = model.encode(b["tokens"]); h = h0
    kmax = int(b["hops"].max())
    pred = b["start"].clone()
    for j in range(1, kmax + 1):
        h = model.step(h, h0)
        r = model.decode(h).argmax(-1)
        pred = torch.where(b["hops"] == j, r, pred)
    out = {"overall": (pred == b["target"]).float().mean().item()}
    for k in sorted(set(b["hops"].tolist())):
        m = b["hops"] == k
        out[f"hop{k}"] = (pred[m] == b["target"][m]).float().mean().item()
    return out


class ICDraft(nn.Module):
    """Attention-based draft for the in-context map: it must READ the per-example
    permutation out of the recurrent state and compose it. One attention block over
    the prompt, then an offset-conditioned symbol head reading the OUT position."""

    def __init__(self, cfg: ModelConfig, horizon: int, n_answer: int, out_pos: int, n_blocks: int = 2):
        super().__init__()
        self.horizon = horizon
        self.out_pos = out_pos
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_blocks)])
        d = cfg.d_model
        hid = 2 * d
        self.sym_off = nn.Embedding(horizon, d)
        self.sym_net = nn.Sequential(nn.Linear(2 * d, hid), nn.GELU(), nn.Linear(hid, n_answer))
        nn.init.normal_(self.sym_off.weight, std=0.02)

    def sym_logits(self, h, horizon=None):
        H = self.horizon if horizon is None else horizon
        x = h
        for blk in self.blocks:                    # attend over map + pointer
            x = blk(x)
        v = x[:, self.out_pos]                     # [B,d]
        B, d = v.shape
        vexp = v.unsqueeze(1).expand(B, H, d)
        off = self.sym_off(torch.arange(H, device=h.device)).view(1, H, d).expand(B, H, d)
        return self.sym_net(torch.cat([vexp, off], dim=-1))   # [B,H,n_answer]

    def propose_symbols(self, h, h0, horizon=None):
        return self.sym_logits(h, horizon).argmax(-1)

    def propose(self, h, h0, horizon=None):        # unused in re-anchor mode
        H = self.horizon if horizon is None else horizon
        return h.unsqueeze(1).expand(h.shape[0], H, *h.shape[1:])


def train_ic_draft(model, draft, spec, steps=1500, batch=256, lr=2e-3, horizon=12,
                   log_every=300, seed=0, device="cpu"):
    """Distill the draft on STEP-0 states (matched to re-anchored inference)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    g = torch.Generator().manual_seed(seed + 5)
    opt = torch.optim.AdamW(draft.parameters(), lr=lr, weight_decay=1e-4)
    H = horizon
    t0 = time.perf_counter()
    for step in range(steps):
        b = ic_batch(spec, batch, generator=g, device=device)
        with torch.no_grad():
            h0 = model.encode(b["tokens"])                 # step-0 states (carry the map)
        tgt = b["traj_target"][:, 1:H + 1]                 # pi^1..pi^H(start)
        Hn = tgt.shape[1]
        logits = draft.sym_logits(h0, Hn)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == tgt).float().mean().item()
            print(f"[ic-draft] step {step:5d} loss {loss.item():.4f} sym-acc {acc:.3f} "
                  f"{time.perf_counter()-t0:.1f}s")
    return draft


@torch.no_grad()
def ic_run(model, spec, draft, H, ks, n=512, device="cpu"):
    from sld import specloop as SL
    g = torch.Generator().manual_seed(321)
    rows = []
    for k in ks:
        b = ic_batch(spec, n, generator=g, fixed_hop=k, device=device)
        tok, tgt = b["tokens"], b["target"]
        reanchor = make_ic_reanchor(model, spec, tok)
        full = SL.full_loop_decode(model, tok, n_steps=k)
        sld = SL.sld_decode(model, draft, tok, horizon=min(H, k), max_steps=k,
                            stop_on_converge=False, reanchor_encode=reanchor)
        donly = SL.draft_only_decode(model, draft, tok, n_steps=min(k, H))
        rows.append({"k": k,
                     "full_acc": (full.answer == tgt).float().mean().item(),
                     "sld_acc": (sld.answer == tgt).float().mean().item(),
                     "sld_lossless": (sld.answer == full.answer).float().mean().item(),
                     "sld_rounds": sld.core_rounds, "mean_accept": sld.extra["mean_accept"],
                     "draftonly_acc": (donly.answer == tgt).float().mean().item()})
    return rows


if __name__ == "__main__":
    import json
    from pathlib import Path
    torch.set_num_threads(6)
    CK = Path(__file__).resolve().parents[1] / "results" / "ckpt"
    CK.mkdir(parents=True, exist_ok=True)
    spec = ICSpec(n_nodes=6, loop_steps=5)
    H = 5
    print(f"in-context: N={spec.n_nodes} vocab={spec.vocab_size} seq_len={spec.seq_len} "
          f"start_pos={spec.start_pos} out_pos={spec.out_pos}", flush=True)
    cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, n_answer=spec.n_nodes,
                      out_pos=spec.out_pos, d_model=96, n_heads=4, d_ff=192,
                      prelude_layers=2, core_layers=2, coda_layers=1, loop_steps=spec.loop_steps)
    tpath = CK / "ic_teacher.pt"
    model = LoopedTransformer(cfg)
    if tpath.exists():
        model.load_state_dict(torch.load(tpath)); print("[load] ic teacher", flush=True)
    else:
        print("teacher params", count_params(model), flush=True)
        train_ic_teacher(model, spec, steps=2500, batch=256, lr=1.5e-3, log_every=400)
        torch.save(model.state_dict(), tpath)
    model.eval()
    print("IC teacher eval:", {k: round(v, 3) for k, v in ic_eval(model, spec).items()}, flush=True)

    draft = ICDraft(cfg, horizon=H, n_answer=spec.n_nodes, out_pos=spec.out_pos, n_blocks=2)
    print("ic draft params", count_params(draft), flush=True)
    train_ic_draft(model, draft, spec, steps=2000, batch=256, horizon=H, lr=1.5e-3, log_every=400)
    draft.eval()

    ks = [k for k in [1, 2, 3, 4, 5] if k <= spec.loop_steps]
    rows = ic_run(model, spec, draft, H, ks)
    print("\n=== IN-CONTEXT SLD (non-memorizable map) ===")
    print(f"{'k':>3} {'full_acc':>9} {'SLD_acc':>8} {'lossless':>9} {'SLD_rounds':>11} "
          f"{'accept':>7} {'draftonly':>10}")
    for r in rows:
        print(f"{r['k']:>3} {r['full_acc']:>9.3f} {r['sld_acc']:>8.3f} {r['sld_lossless']:>9.3f} "
              f"{r['sld_rounds']:>11.2f} {r['mean_accept']:>7.2f} {r['draftonly_acc']:>10.3f}")
    (Path(__file__).resolve().parents[1] / "results" / "incontext.json").write_text(json.dumps(rows, indent=2))
    print("[saved] results/incontext.json")
