"""Global deterministic seeding.

Author: 晨星
"""

from __future__ import annotations

import random

import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """Set every RNG we use from a single seed.

    Returns the numpy Generator that should be threaded through the generator
    so that *all* stochastic draws in one run are reproducible.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
