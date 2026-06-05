"""Train the learned draft against the frozen teacher's trajectory tape.

"Freeze the expensive object, train the cheap peripheral": the teacher never
moves. We record its loop trajectory on training queries and distill the draft
to predict the next ``horizon`` states from any state on the trajectory. The
loss combines state MSE with cross-entropy on the *readout* at each drafted step
-- because the readout is exactly what the speculative verifier checks, so
optimizing it directly maximizes the accepted-prefix length.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def record_tape(model, spec, make_batch, n_examples: int, *, generator=None, batch: int = 512):
    """Return (states, readouts) tapes.

    states  : [N, L+1, T, d]  the loop trajectory h_0..h_L for each query
    readouts: [N, L+1]        the argmax readout at each loop step
    """
    model.eval()
    L = model.cfg.loop_steps
    states, readouts = [], []
    done = 0
    while done < n_examples:
        bs = min(batch, n_examples - done)
        b = make_batch(spec, bs, generator=generator)
        h0 = model.encode(b["tokens"])
        h = h0
        traj = [h0]
        for _ in range(L):
            h = model.step(h, h0)
            traj.append(h)
        S = torch.stack(traj, dim=1)                       # [bs, L+1, T, d]
        R = model.decode(S.reshape(bs * (L + 1), *S.shape[2:])).argmax(-1).reshape(bs, L + 1)
        states.append(S)
        readouts.append(R)
        done += bs
    return torch.cat(states), torch.cat(readouts)


def train_draft(
    model, draft, spec, make_batch, *,
    steps: int = 1500, batch: int = 256, lr: float = 2e-3, horizon: int | None = None,
    ce_weight: float = 1.0, tape_examples: int = 4096, seed: int = 0, log_every: int = 250,
    verbose: bool = True,
):
    """Distill the draft on the frozen teacher's tape. Teacher weights never change."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    H = draft.horizon if horizon is None else horizon
    L = model.cfg.loop_steps

    g = torch.Generator().manual_seed(seed + 7)
    states, readouts = record_tape(model, spec, make_batch, tape_examples, generator=g)
    N = states.shape[0]
    opt = torch.optim.AdamW(draft.parameters(), lr=lr, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed + 11)
    t0 = time.perf_counter()
    draft.train()
    for step in range(steps):
        # Train from STEP-0 (re-anchored) states only: at inference, re-anchoring
        # always feeds the draft the canonical step-0 state encode(symbol), so the
        # training distribution must match (else the symbol head sees off-distribution
        # trajectory states and acceptance collapses). Random starts already cover
        # all symbols, so step-0 states span the whole input space the draft sees.
        ei = torch.randint(0, N, (batch,), generator=rng)
        ti = torch.zeros(batch, dtype=torch.long)                    # step-0 canonical states
        h = states[ei, ti]                                            # [B,T,d]
        # targets: next H states (clamped at L) and their readouts
        offs = torch.arange(1, H + 1)
        tgt_idx = (ti.unsqueeze(1) + offs.unsqueeze(0)).clamp_max(L)  # [B,H]
        tgt_state = states[ei.unsqueeze(1), tgt_idx]                  # [B,H,T,d]
        tgt_read = readouts[ei.unsqueeze(1), tgt_idx]                 # [B,H]

        G = draft.propose(h, h, H)                                   # [B,H,T,d]
        mse = F.mse_loss(G, tgt_state)
        # readout CE through the FROZEN decoder (keeps predicted states on-manifold)
        Bn, Hn, T, d = G.shape
        logits = model.decode(G.reshape(Bn * Hn, T, d))             # [B*H, n_answer]
        ce = F.cross_entropy(logits, tgt_read.reshape(Bn * Hn))
        loss = mse + ce_weight * ce
        # cheap symbol head CE (the fast inference path used by re-anchored SLD)
        sym_acc = float("nan")
        if getattr(draft, "sym_net", None) is not None:
            sym_logits = draft.sym_logits(h, H)                      # [B,H,n_answer]
            sym_ce = F.cross_entropy(sym_logits.reshape(Bn * Hn, -1), tgt_read.reshape(Bn * Hn))
            loss = loss + ce_weight * sym_ce
            with torch.no_grad():
                sym_acc = (sym_logits.argmax(-1) == tgt_read).float().mean().item()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (step % log_every == 0 or step == steps - 1):
            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt_read.reshape(-1)).float().mean().item()
            print(f"[draft] step {step:5d}  loss {loss.item():.4f}  mse {mse.item():.4f}  "
                  f"ce {ce.item():.4f}  readout-acc {acc:.3f}  sym-acc {sym_acc:.3f}  "
                  f"{time.perf_counter()-t0:.1f}s")
    return {"train_time_s": time.perf_counter() - t0}


def train_jump(model, jump, spec, make_batch, *, steps=1500, batch=256, lr=2e-3,
               seed=0, log_every=250, verbose=True):
    """Train the original-JumpRec jump module to predict the FINAL state from h0."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    L = model.cfg.loop_steps
    opt = torch.optim.AdamW(jump.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 3)
    t0 = time.perf_counter()
    for step in range(steps):
        b = make_batch(spec, batch, generator=g)
        with torch.no_grad():
            h0 = model.encode(b["tokens"])
            h = h0
            for _ in range(L):
                h = model.step(h, h0)
            target_state = h
            target_read = model.decode(h).argmax(-1)
        hj = jump(h0)
        mse = F.mse_loss(hj, target_state)
        logits = model.decode(model.step(hj, h0))
        ce = F.cross_entropy(logits, target_read)
        loss = mse + ce
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (step % log_every == 0 or step == steps - 1):
            with torch.no_grad():
                acc = (logits.argmax(-1) == target_read).float().mean().item()
            print(f"[jump] step {step:5d}  loss {loss.item():.4f}  acc {acc:.3f}  "
                  f"{time.perf_counter()-t0:.1f}s")
    return {"train_time_s": time.perf_counter() - t0}
