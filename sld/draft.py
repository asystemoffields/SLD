"""Draft modules: cheap proposers of future loop states.

A draft maps the current recurrent state ``h`` (= true ``h_t``) to a *trajectory*
of the next ``horizon`` states ``g_1..g_H`` (guesses of ``h_{t+1..t+H}``). The
speculative decoder verifies these against the true core in one batched pass.

Two flavors:
  * ``LearnedDraft`` -- an offset-conditioned, *non-recurrent*, attention-free
    predictor trained to imitate the frozen teacher's recorded trajectory. It is
    deliberately far cheaper than one core call (no attention), so its cost does
    not eat the speculative speedup. For a fixed permutation it only has to learn
    the lookup ``symbol -> pi^i(symbol)``, which a pointwise MLP can represent.
  * ``AndersonDraft`` -- a *training-free* fixed-point extrapolator (Anderson /
    Irons-Tuck) that works on any frozen looped transformer. It only has signal
    in the converging/settling phase; on the advancing phase it is expected to be
    weak. Included as the "works on any pretrained loop" control.

Also: ``IdentityDraft`` (no-draft control) and ``BlindDraft`` (random control)
to isolate where speculative wins come from, and ``JumpModule`` -- the original
JumpRec lossy jump (predict the final state in one shot) used by the
confidence-verifier baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDraft(nn.Module):
    """Offset-conditioned pointwise predictor of the next ``horizon`` states.

    g_i = h + MLP([h ; offset_emb[i]]), computed for all i in one batched pass.
    No attention -> much cheaper than the core (which has attention).
    """

    def __init__(self, d_model: int, horizon: int, n_answer: int | None = None,
                 out_pos: int = 0, hidden_mult: int = 2):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.out_pos = out_pos
        self.offset_emb = nn.Embedding(horizon, d_model)
        hid = hidden_mult * d_model
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        nn.init.normal_(self.offset_emb.weight, std=0.02)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Cheap symbol-trajectory head: predict the next H readout symbols directly
        # from the current readout-position vector (no [T,d] state, no attention).
        # In re-anchor mode this is all SLD needs from the draft, so a round costs
        # ~one core call of wall-clock (the states are gathered from a cached table).
        self.sym_off = nn.Embedding(horizon, d_model)
        self.sym_net = (None if n_answer is None else nn.Sequential(
            nn.Linear(2 * d_model, hid), nn.GELU(), nn.Linear(hid, n_answer)))
        if self.sym_net is not None:
            nn.init.normal_(self.sym_off.weight, std=0.02)

    def propose(self, h: torch.Tensor, h0: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """h: [B,T,d] current state. Returns G: [B,H,T,d] guesses g_1..g_H."""
        H = self.horizon if horizon is None else horizon
        B, T, d = h.shape
        hexp = h.unsqueeze(1).expand(B, H, T, d)                       # [B,H,T,d]
        off = self.offset_emb(torch.arange(H, device=h.device))        # [H,d]
        off = off.view(1, H, 1, d).expand(B, H, T, d)
        inp = torch.cat([hexp, off], dim=-1)                           # [B,H,T,2d]
        resid = self.net(inp)
        return hexp + resid

    def sym_logits(self, h: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """Predict the next H readout symbols from h. Returns [B,H,n_answer]."""
        H = self.horizon if horizon is None else horizon
        B, T, d = h.shape
        v = h[:, self.out_pos]                                         # [B,d]
        vexp = v.unsqueeze(1).expand(B, H, d)
        off = self.sym_off(torch.arange(H, device=h.device)).view(1, H, d).expand(B, H, d)
        return self.sym_net(torch.cat([vexp, off], dim=-1))           # [B,H,n_answer]

    def propose_symbols(self, h: torch.Tensor, h0: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        """Cheap path: predicted next-H symbols [B,H] (argmax of the symbol head)."""
        return self.sym_logits(h, horizon).argmax(-1)


class IdentityDraft(nn.Module):
    """No-draft control: g_i = h (copy current state). Accept length should be ~0."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def propose(self, h, h0, horizon=None):
        H = self.horizon if horizon is None else horizon
        return h.unsqueeze(1).expand(h.shape[0], H, *h.shape[1:])


class BlindDraft(nn.Module):
    """Random-draft control: confirms wins come from prediction, not the verify trick."""

    def __init__(self, horizon: int, scale: float = 1.0):
        super().__init__()
        self.horizon = horizon
        self.scale = scale

    def propose(self, h, h0, horizon=None):
        H = self.horizon if horizon is None else horizon
        B, T, d = h.shape
        g = h.unsqueeze(1).expand(B, H, T, d).clone()
        return g + self.scale * torch.randn_like(g)


class AndersonDraft:
    """Training-free fixed-point extrapolation draft (Irons-Tuck / Aitken, AA(1)).

    Uses the iterate-recurrence form  s_bar + beta * r_bar  (NOT the unstable
    's0 - alpha*ds' shortcut). Re-normalizes onto the residual-stream norm shell.
    Maintains a tiny history of (s, core(s)) across rounds. Only has signal on a
    contracting/settling trajectory; expected weak on the advancing phase.
    """

    def __init__(self, model, horizon: int, beta: float = 1.0):
        self.model = model
        self.horizon = horizon
        self.beta = beta
        self.prev_s = None
        self.prev_r = None

    def reset(self):
        self.prev_s = None
        self.prev_r = None

    @torch.no_grad()
    def propose(self, h: torch.Tensor, h0: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        H = self.horizon if horizon is None else horizon
        B, T, d = h.shape
        gh = self.model.step(h, h0)            # one core eval: g(s_k)
        r = gh - h                             # residual r_k
        # SLD's active batch shrinks as examples finish; reset history on a size
        # change (this is a training-free control, so a graceful reset is fine).
        if self.prev_s is not None and self.prev_s.shape[0] != B:
            self.prev_s, self.prev_r = None, None
        if self.prev_s is None:
            # not enough history: roll forward by repeated core (geometric guess = current step)
            self.prev_s, self.prev_r = h, r
            base = gh
            return base.unsqueeze(1).expand(B, H, T, d)
        dr = r - self.prev_r
        denom = (dr * dr).flatten(1).sum(1, keepdim=True).clamp_min(1e-12)   # <dr,dr>
        num = (r * dr).flatten(1).sum(1, keepdim=True)                       # <r,dr>
        alpha = (num / denom).view(B, 1, 1)
        s_bar = (1 - alpha) * h + alpha * self.prev_s
        r_bar = (1 - alpha) * r + alpha * self.prev_r
        s_star = s_bar + self.beta * r_bar     # extrapolated fixed point
        # geometric rate lambda_hat from residual norms for rolling forward
        rn = r.flatten(1).norm(dim=1, keepdim=True)
        prn = self.prev_r.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-9)
        lam = (rn / prn).clamp(0.0, 0.999).view(B, 1, 1)
        self.prev_s, self.prev_r = h, r
        # draft trajectory: s_star + lam^j (h - s_star), j=1..H  (settling-phase model)
        offs = torch.arange(1, H + 1, device=h.device).view(1, H, 1, 1).float()
        lam_e = lam.view(B, 1, 1, 1)
        G = s_star.unsqueeze(1) + lam_e.pow(offs) * (h.unsqueeze(1) - s_star.unsqueeze(1))
        return G


class JumpModule(nn.Module):
    """Original-JumpRec jump: predict the FINAL recurrent state in one shot, plus a
    confidence head. Lossy. Used by the confidence-verifier baseline we must beat."""

    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        hid = hidden_mult * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        return h0 + self.net(h0)
