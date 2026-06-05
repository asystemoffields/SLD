"""Speculative Looped Decoding (SLD) and baselines.

SLD transplants speculative decoding from the *token* axis to the *depth/loop*
axis of a single token's trajectory in a looped transformer:

  1. From the current verified state ``h`` (= true ``h_t``) a cheap DRAFT proposes
     a trajectory of future loop states ``g_1..g_H``.
  2. The TRUE core verifies them in ONE batched forward pass: it computes
     ``u_i = core(g_i)`` for the window in a single call (plus ``u_0 = core(h)``,
     the exact next true state).
  3. We accept the longest prefix on which the draft is self-consistent with the
     true core's *discrete readout*: ``readout(g_{i+1}) == readout(u_i)``. The
     first true output past the accepted prefix is a free, exact correction (the
     "bonus token" analog).
  4. We advance ``a+1`` true loop steps for the cost of ONE batched core call,
     and repeat.

Losslessness. The acceptance test is on the *argmax readout* (a discrete
projection), so it is bit-stable (immune to BLAS reduction-order noise) and
``tol = 0`` is meaningful. For symbolic recurrence the readout is Markov
(``readout(core(s))`` depends only on ``readout(s)``), so every advanced state's
readout equals the true iterate's readout *exactly*, regardless of continuous
drift in nuisance dimensions -- the final answer is identical to the full loop.
We assert this byte-for-byte in the benchmark.

Counting. The scientific (hardware-free) metric is the number of *sequential
core rounds*; this is the latency on parallel hardware. We also report total
core rows (a FLOP proxy: rejected drafts cost extra rows but no extra rounds),
draft calls, and the accepted-prefix-length distribution. Wall-clock is reported
separately, in the regime (compact state, multi-thread) where batched verify is
flat and the round count and wall-clock agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DecodeStats:
    answer: torch.Tensor                       # [B] predicted answer symbols
    core_rounds: float = 0.0                   # mean sequential core invocations / example
    core_rows: float = 0.0                     # mean total seq-rows through core / example
    draft_calls: float = 0.0                   # mean draft invocations / example
    readout_calls: float = 0.0                 # mean readout(decode) invocations / example
    accept_lengths: list = field(default_factory=list)   # accepted prefix per round (all)
    per_example_rounds: torch.Tensor = None    # [B] rounds each example actually used
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def _readout(model, states: torch.Tensor) -> torch.Tensor:
    """Discrete readout (argmax answer symbol) for a stack of states [N,T,d] -> [N]."""
    return model.decode(states).argmax(-1)


def make_reanchor(model, spec):
    """A reanchor_encode callable for the pointer-chase task: map a batch of
    symbols to the canonical on-manifold step-0 state ``encode(start=symbol)``.

    There are only ``n_nodes`` possible single-symbol inputs, so we precompute the
    prelude over all of them ONCE (a cached input encoding -- not the recurrent
    core, which is still run live for verification). Re-anchoring is then a free
    gather, which keeps each speculative round at ~one core call of wall-clock.
    """
    with torch.no_grad():
        syms = torch.arange(spec.n_nodes)
        toks = torch.empty(spec.n_nodes, spec.seq_len, dtype=torch.long)
        toks[:, 0] = spec.node_base + syms
        toks[:, 1] = spec.REG1 if spec.advance_only else (spec.hop_base + spec.max_hops)
        if spec.seq_len > 2:
            toks[:, 2] = spec.REG0
        if spec.seq_len > 3:
            toks[:, 3] = spec.REG1
        for p in range(4, spec.seq_len):
            toks[:, p] = spec.REG0
        table = model.encode(toks)               # [N, T, d] cached input encodings

    def reanchor(symbols: torch.Tensor, ex_idx=None) -> torch.Tensor:
        return table[symbols]                    # map is global -> example index unused
    return reanchor


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

@torch.no_grad()
def draft_only_decode(model, draft, tokens, *, n_steps: int) -> DecodeStats:
    """Lossy ancestor (original-JumpRec style): trust the draft's prediction of
    pi^k(start) in ONE shot, with NO verification. Cost ~0 core calls, but accuracy
    is only the draft's k-ahead accuracy -- it has no correctness guarantee. This
    is exactly what SLD's lossless verification fixes."""
    h0 = model.encode(tokens)
    if hasattr(draft, "propose_symbols") and getattr(draft, "sym_net", None) is not None:
        sym = draft.propose_symbols(h0, h0, n_steps)        # [B, n_steps]
        ans = sym[:, n_steps - 1]
    else:
        G = draft.propose(h0, h0, n_steps)                  # [B, n_steps, T, d]
        ans = _readout(model, G[:, n_steps - 1])
    B = tokens.shape[0]
    return DecodeStats(answer=ans, core_rounds=0.0, core_rows=0.0, draft_calls=1.0,
                       readout_calls=1.0, per_example_rounds=torch.zeros(B))


@torch.no_grad()
def full_loop_decode(model, tokens, *, n_steps: int) -> DecodeStats:
    """Run the true core exactly ``n_steps`` times (the lossless reference).

    Cost = n_steps sequential core calls (batch B each)."""
    B = tokens.shape[0]
    h0 = model.encode(tokens)
    h = h0
    for _ in range(n_steps):
        h = model.step(h, h0)
    ans = _readout(model, h)
    return DecodeStats(answer=ans, core_rounds=float(n_steps), core_rows=float(n_steps),
                       draft_calls=0.0, readout_calls=1.0,
                       per_example_rounds=torch.full((B,), float(n_steps)))


@torch.no_grad()
def early_exit_decode(model, tokens, *, max_steps: int, patience: int = 1) -> DecodeStats:
    """Convergence early-exit: stop once the readout stops changing for ``patience``
    consecutive steps. Lossless w.r.t. the held full loop; free only when the loop
    converges. On the *advancing* phase it cannot skip ahead -- that is the point."""
    B = tokens.shape[0]
    h0 = model.encode(tokens)
    h = h0
    prev = _readout(model, h)
    stable = torch.zeros(B, dtype=torch.long)
    done_at = torch.full((B,), max_steps, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool)
    cur_read = prev.clone()
    for t in range(1, max_steps + 1):
        h = model.step(h, h0)
        r = _readout(model, h)
        same = (r == prev) & ~finished
        stable = torch.where(same, stable + 1, torch.zeros_like(stable))
        newly = (stable >= patience) & ~finished
        # answer for newly-finished examples is the stable readout (prev)
        cur_read = torch.where(newly, prev, cur_read)
        done_at = torch.where(newly, torch.full_like(done_at, t - patience), done_at)
        finished = finished | newly
        prev = r
        if finished.all():
            break
    # any unfinished -> use final readout
    cur_read = torch.where(finished, cur_read, _readout(model, h))
    rounds = done_at.float()
    return DecodeStats(answer=cur_read, core_rounds=rounds.mean().item(),
                       core_rows=rounds.mean().item(), draft_calls=0.0,
                       readout_calls=1.0, per_example_rounds=rounds)


# --------------------------------------------------------------------------- #
# Speculative Looped Decoding
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sld_decode(
    model, draft, tokens, *,
    horizon: int, max_steps: int, stop_on_converge: bool = True,
    conv_patience: int = 1, count_active_rows: bool = True,
    reanchor_encode=None,
) -> DecodeStats:
    """Speculative Looped Decoding.

    Each example carries its own verified true state ``h_b`` at loop-step ``t_b``.
    Per round: draft a window, verify with one batched core call, accept the
    longest readout-consistent prefix, advance ``a+1`` steps, repeat until each
    example reaches ``max_steps`` or its readout converges.

    ``reanchor_encode`` (optional): a callable ``symbols[N] -> states[N,T,d]`` that
    returns the canonical ON-MANIFOLD state for a discrete readout (e.g. re-running
    the prelude with that symbol as the start). When provided, the verified state
    is snapped back onto the manifold every round, which makes SLD *exactly*
    lossless regardless of draft quality: the carried state never drifts off the
    trajectory manifold where the readout-Markov property would otherwise fail.
    This is valid precisely when the readout is a sufficient statistic of the
    recurrent state (the discrete-readout regime).
    """
    B, T = tokens.shape[0], tokens.shape[1]
    d = model.cfg.d_model
    h0 = model.encode(tokens)                       # [B,T,d] fixed reinjection
    h = h0.clone()                                  # current verified true state
    t = torch.zeros(B, dtype=torch.long)            # verified loop-step per example
    finished = torch.zeros(B, dtype=torch.bool)
    answer = _readout(model, h)                     # readout at step 0
    stable = torch.zeros(B, dtype=torch.long)
    prev_read = answer.clone()

    rounds = 0
    total_rows = 0
    draft_calls = 0
    readout_calls = 0
    per_ex_rounds = torch.zeros(B, dtype=torch.long)
    accept_lengths: list = []

    while not finished.all() and rounds < max_steps + 2:
        active = ~finished
        idx = active.nonzero(as_tuple=True)[0]
        nb = idx.numel()
        # window: cannot draft past the per-example budget; use the smallest active remaining
        remaining = (max_steps - t[idx]).clamp_min(1)
        win = int(min(horizon, int(remaining.max().item())))
        if win < 1:
            break

        ha = h[idx]                                  # [nb,T,d]  on-manifold current state
        h0a = h0[idx]
        draft_calls += 1

        if reanchor_encode is not None:
            # DISCRETE-READOUT LOSSLESS MODE. The draft only needs to predict the
            # next-symbol trajectory (states are gathered from the cached encode
            # table), so use the cheap symbol head when available. Snapping each
            # drafted symbol back to its canonical on-manifold state makes every
            # parallel verify step the TRUE core on a real trajectory state, so
            # readout-Markov holds exactly and SLD is lossless regardless of the
            # draft. Verify inputs: [h, encode(g_1), ..., encode(g_{win-1})].
            if hasattr(draft, "propose_symbols") and getattr(draft, "sym_net", None) is not None:
                read_g = draft.propose_symbols(ha, h0a, win)                  # [nb,win] cheap
            else:
                G = draft.propose(ha, h0a, win)
                read_g = _readout(model, G.reshape(nb * win, T, d)).reshape(nb, win)
            # re-anchor each drafted symbol to its canonical on-manifold state. For
            # per-example context (e.g. an in-context map), reanchor_encode also takes
            # the example index so it can reconstruct the right state.
            if win > 1:
                ex = idx.unsqueeze(1).expand(nb, win - 1).reshape(-1)
                Ghat = reanchor_encode(read_g[:, :win - 1].reshape(-1), ex).reshape(nb, win - 1, T, d)
            else:
                Ghat = ha[:, :0].reshape(nb, 0, T, d)
            verify_in = torch.cat([ha.unsqueeze(1), Ghat], dim=1)            # [nb,win,T,d]
            h0_rep = verify_in.reshape(nb * win, T, d)                       # each on-manifold state reinjects itself
        else:
            G = draft.propose(ha, h0a, win)                                  # [nb,win,T,d]
            read_g = _readout(model, G.reshape(nb * win, T, d)).reshape(nb, win)
            verify_in = torch.cat([ha.unsqueeze(1), G[:, :win - 1]], dim=1)  # continuous (approx) path
            h0_rep = h0a.unsqueeze(1).expand(nb, win, T, d).reshape(nb * win, T, d)  # fixed reinjection

        flat = verify_in.reshape(nb * win, T, d)
        U = model.step(flat, h0_rep).reshape(nb, win, T, d)                  # one batched core call
        rounds += 1
        per_ex_rounds[idx] += 1
        total_rows += (nb * win) if count_active_rows else (B * win)
        read_u = _readout(model, U.reshape(nb * win, T, d)).reshape(nb, win) # true next symbols
        readout_calls += 1

        # accept g_{i+1} iff its symbol matches the true next symbol read_u[:,i]
        match = (read_g == read_u)                   # [nb,win]
        notmatch = ~match
        big = torch.where(notmatch, torch.arange(win).expand(nb, win), torch.full((nb, win), win))
        a = big.min(dim=1).values                    # accepted prefix length in [0,win]
        accept_lengths.extend(a.tolist())

        # advance a+1 steps, but never past this example's OWN remaining budget
        # (win is set by the slowest active example; faster ones must not overshoot).
        rem = (max_steps - t[idx]).clamp_min(1)                # per-example budget left
        adv = torch.minimum(torch.clamp(a + 1, max=win), rem)  # steps to advance, per example
        carry_idx = (adv - 1).clamp(min=0, max=win - 1)        # read_u col for symbol at t+adv
        new_read = read_u[torch.arange(nb), carry_idx]
        if reanchor_encode is not None:
            new_h = reanchor_encode(new_read, idx)    # canonical on-manifold state, exact
        else:
            new_h = U[torch.arange(nb), carry_idx]    # continuous carry (approx)
        h[idx] = new_h
        h0[idx] = new_h
        t[idx] = t[idx] + adv
        answer[idx] = new_read

        # convergence / budget stop
        same = (new_read == prev_read[idx])
        stable[idx] = torch.where(same, stable[idx] + adv, torch.zeros_like(stable[idx]))
        prev_read[idx] = new_read
        reach_budget = t[idx] >= max_steps
        converged = stable[idx] >= conv_patience if stop_on_converge else torch.zeros_like(reach_budget)
        newly_done = reach_budget | converged
        fin_global = idx[newly_done]
        finished[fin_global] = True

    return DecodeStats(
        answer=answer,
        core_rounds=per_ex_rounds.float().mean().item(),
        core_rows=float(total_rows) / B,
        draft_calls=float(draft_calls),
        readout_calls=float(readout_calls),
        accept_lengths=accept_lengths,
        per_example_rounds=per_ex_rounds,
        extra={"mean_accept": (sum(accept_lengths) / max(1, len(accept_lengths)))},
    )


# --------------------------------------------------------------------------- #
# Original-JumpRec confidence verifier (lossy predecessor baseline)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def confidence_jump_decode(
    model, jump, tokens, *, max_steps: int, threshold: float, correct_steps: int = 1,
) -> DecodeStats:
    """Predict the final state in one jump; accept if the readout margin clears
    ``threshold`` else fall back to the full loop. Lossy (quality trades with the
    threshold). This is the original JumpRec mechanism SLD must Pareto-dominate."""
    B = tokens.shape[0]
    h0 = model.encode(tokens)
    hj = jump(h0)
    for _ in range(correct_steps):
        hj = model.step(hj, h0)
    logits = model.decode(hj)
    prob = logits.softmax(-1)
    top2 = prob.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    accept = margin >= threshold
    ans_jump = logits.argmax(-1)
    # fallback: full loop for rejected
    rounds = torch.where(accept, torch.full((B,), float(1 + correct_steps)),
                         torch.full((B,), float(max_steps)))
    ans = ans_jump.clone()
    if (~accept).any():
        full = full_loop_decode(model, tokens[~accept], n_steps=max_steps)
        ans[~accept] = full.answer
    return DecodeStats(answer=ans, core_rounds=rounds.mean().item(),
                       core_rows=rounds.mean().item(), draft_calls=1.0,
                       readout_calls=1.0, per_example_rounds=rounds,
                       extra={"accept_rate": accept.float().mean().item()})
