"""Shared helpers for SLD benchmarks: checkpoint save/load and wall-clock timing."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from sld.substrate import TaskSpec, ModelConfig, LoopedTransformer
from sld import draft as D

CKPT_DIR = Path(__file__).resolve().parents[1] / "results" / "ckpt"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def save_teacher(model: LoopedTransformer, spec: TaskSpec, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": vars(model.cfg),
        "spec": {k: getattr(spec, k) for k in
                 ["n_nodes", "max_hops", "loop_steps", "perm_seed", "seq_len", "advance_only"]},
        "perm": spec.perm,
    }, path)


def load_teacher(path: Path):
    ck = torch.load(path, weights_only=False)
    spec = TaskSpec(**ck["spec"], perm=ck["perm"])
    cfg = ModelConfig(**ck["model_cfg"])
    model = LoopedTransformer(cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, spec


def save_module(module: torch.nn.Module, meta: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": module.state_dict(), "meta": meta}, path)


def load_learned_draft(path: Path, d_model: int, n_answer: int | None = None, out_pos: int = 0):
    ck = torch.load(path, weights_only=False)
    drf = D.LearnedDraft(d_model, horizon=ck["meta"]["horizon"],
                         n_answer=n_answer, out_pos=out_pos)
    drf.load_state_dict(ck["state"])
    drf.eval()
    return drf, ck["meta"]


@torch.no_grad()
def time_decode(fn, tokens, *, repeats: int = 50, warmup: int = 5) -> float:
    """Median ms per call of a decode fn(tokens). tokens fixed across repeats."""
    for _ in range(warmup):
        fn(tokens)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(tokens)
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]
