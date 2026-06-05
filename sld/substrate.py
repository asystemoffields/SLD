"""Bridge to the JumpRec substrate (the looped-transformer + recurrence tasks).

SLD is its own project but builds on the ``jumprec`` substrate package (looped
transformer model, fixed-permutation pointer-chasing task, teacher training).
To stay trivially runnable before ``jumprec`` is pip-installed, this module adds
the local jumprec checkout to ``sys.path`` if it is not already importable.

Resolution order for the jumprec checkout:
  1. already importable (e.g. ``pip install -e path/to/jumprec``);
  2. the ``JUMPREC_PATH`` environment variable;
  3. the default sibling checkout ``../SMOKE`` next to this repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_jumprec_importable() -> None:
    try:
        import jumprec  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    candidates = []
    env = os.environ.get("JUMPREC_PATH")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    # repo root is SLD/, jumprec substrate default lives at ../SMOKE
    candidates.append(here.parents[2] / "SMOKE")
    for c in candidates:
        if c and (c / "jumprec" / "__init__.py").exists():
            sys.path.insert(0, str(c))
            return
    raise ModuleNotFoundError(
        "Could not locate the 'jumprec' substrate. Install it "
        "(pip install -e path/to/jumprec) or set JUMPREC_PATH."
    )


_ensure_jumprec_importable()

from jumprec import tasks, model, train  # noqa: E402,F401
from jumprec.tasks import TaskSpec, make_batch, grouped_accuracy, make_permutation  # noqa: E402,F401
from jumprec.model import ModelConfig, LoopedTransformer, count_params  # noqa: E402,F401
from jumprec.train import TrainConfig, train_teacher, evaluate_teacher  # noqa: E402,F401

__all__ = [
    "tasks", "model", "train",
    "TaskSpec", "make_batch", "grouped_accuracy", "make_permutation",
    "ModelConfig", "LoopedTransformer", "count_params",
    "TrainConfig", "train_teacher", "evaluate_teacher",
]
