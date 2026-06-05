"""SLD -- Speculative Looped Decoding.

Lossless acceleration of looped / recurrent-depth transformers by transplanting
speculative decoding from the token axis to the depth/loop axis: draft a
trajectory of future loop states, verify them all in one batched true-core pass,
accept the longest discrete-readout-consistent prefix, and continue from the
verified state. Built on the ``jumprec`` substrate (see ``sld.substrate``).
"""

from . import substrate  # noqa: F401  (ensures jumprec is importable)
from . import draft, specloop, training  # noqa: F401

__all__ = ["substrate", "draft", "specloop", "training"]
