"""Correctness tests for Speculative Looped Decoding.

The load-bearing property is LOSSLESSNESS: on the advance-only symbolic task the
recurrent readout is Markov, so SLD's answer must equal the full loop's answer
exactly, for every depth k and regardless of draft quality. We also check the
controls behave (no-draft accepts ~nothing; SLD never costs more rounds than the
full loop; a random draft is still lossless).

Run:  PYTHONPATH=../SMOKE:.. python -m pytest tests/ -q
   or: PYTHONPATH=../SMOKE:.. python tests/test_specloop.py
"""

from __future__ import annotations

import torch

from sld.substrate import (TaskSpec, make_batch, ModelConfig, LoopedTransformer,
                           TrainConfig, train_teacher)
from sld import draft as D
from sld import specloop as SL
from sld.training import train_draft


def _tiny_trained(seed=0, steps=400):
    torch.manual_seed(seed)
    spec = TaskSpec(n_nodes=8, max_hops=5, loop_steps=7, advance_only=True)
    cfg = ModelConfig(vocab_size=spec.vocab_size, seq_len=spec.seq_len, n_answer=spec.n_nodes,
                      out_pos=spec.out_pos, d_model=32, n_heads=4, d_ff=64,
                      prelude_layers=1, core_layers=1, coda_layers=1, loop_steps=spec.loop_steps)
    model = LoopedTransformer(cfg)
    train_teacher(model, spec, TrainConfig(steps=steps, batch_size=256, lr=4e-3,
                                           log_every=10_000), verbose=False)
    model.eval()
    return model, spec, cfg


def test_lossless_learned_draft():
    model, spec, cfg = _tiny_trained()
    H = 5
    re = SL.make_reanchor(model, spec)
    drf = D.LearnedDraft(cfg.d_model, horizon=H, n_answer=spec.n_nodes, out_pos=spec.out_pos)
    train_draft(model, drf, spec, make_batch, steps=400, batch=256, horizon=H,
                tape_examples=1024, log_every=10_000, verbose=False)
    drf.eval()
    g = torch.Generator().manual_seed(11)
    for k in [1, 2, 3, 4, 5]:
        b = make_batch(spec, 128, generator=g, fixed_hop=k)
        full = SL.full_loop_decode(model, b["tokens"], n_steps=k)
        sld = SL.sld_decode(model, drf, b["tokens"], horizon=min(H, k), max_steps=k,
                            stop_on_converge=False, reanchor_encode=re)
        assert torch.equal(sld.answer, full.answer), f"lossless violated at k={k}"
        assert sld.core_rounds <= full.core_rounds + 1e-6, f"SLD slower than full at k={k}"


def test_lossless_with_random_draft():
    """Verification must hold the answer correct even with a useless (blind) draft."""
    model, spec, cfg = _tiny_trained(seed=1)
    re = SL.make_reanchor(model, spec)
    blind = D.BlindDraft(horizon=5, scale=1.0)
    g = torch.Generator().manual_seed(3)
    for k in [1, 3, 5]:
        b = make_batch(spec, 128, generator=g, fixed_hop=k)
        full = SL.full_loop_decode(model, b["tokens"], n_steps=k)
        sld = SL.sld_decode(model, blind, b["tokens"], horizon=min(5, k), max_steps=k,
                            stop_on_converge=False, reanchor_encode=re)
        assert torch.equal(sld.answer, full.answer), f"blind-draft lossless violated at k={k}"


def test_no_draft_accepts_nothing():
    """Identity draft proposes the current state -> no leap -> rounds == full loop."""
    model, spec, cfg = _tiny_trained(seed=2)
    re = SL.make_reanchor(model, spec)
    ident = D.IdentityDraft(horizon=5)
    g = torch.Generator().manual_seed(5)
    k = 5
    b = make_batch(spec, 128, generator=g, fixed_hop=k)
    full = SL.full_loop_decode(model, b["tokens"], n_steps=k)
    sld = SL.sld_decode(model, ident, b["tokens"], horizon=k, max_steps=k,
                        stop_on_converge=False, reanchor_encode=re)
    assert torch.equal(sld.answer, full.answer)
    assert sld.extra["mean_accept"] < 0.5, "identity draft should accept ~nothing"
    assert sld.core_rounds >= k - 1e-6, "no-draft should not save rounds"


def test_shapes_and_counting():
    model, spec, cfg = _tiny_trained(seed=3, steps=50)
    drf = D.LearnedDraft(cfg.d_model, horizon=4)
    b = make_batch(spec, 16, fixed_hop=4)
    sld = SL.sld_decode(model, drf, b["tokens"], horizon=4, max_steps=4, stop_on_converge=False)
    assert sld.answer.shape == (16,)
    assert sld.per_example_rounds.shape == (16,)
    assert sld.core_rows >= sld.core_rounds  # rows >= rounds always


if __name__ == "__main__":
    for fn in [test_lossless_learned_draft, test_lossless_with_random_draft,
               test_no_draft_accepts_nothing, test_shapes_and_counting]:
        fn()
        print(f"[pass] {fn.__name__}")
    print("all tests passed")
