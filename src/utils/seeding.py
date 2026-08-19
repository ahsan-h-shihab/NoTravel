"""Deterministic seeding.

Why this exists: RANDOMNESS requires that whenever randomness affects results
the seed is recorded, and REPRODUCIBILITY CONSTITUTION treats loss of
reproducibility as scientific failure. Every stochastic operation in this project draws from
a generator created here, so there is exactly one place where randomness enters.
"""

from __future__ import annotations

import os
import random

import numpy as np

#: Project-wide default seed. Recorded in every provenance record.
DEFAULT_SEED = 20260802


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed all global RNGs and pin thread counts that affect float reduction order.

    Thread counts are pinned because BLAS reduction order depends on the number of
    threads, which can change low-order bits of embeddings and therefore of scores.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU inference; full determinism not needed
    except ImportError:
        pass

    return seed


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh, explicitly seeded NumPy generator.

    Prefer this over the legacy global `np.random` state: passing an explicit generator
    makes the dependence of a result on its seed visible in the call site.
    """
    return np.random.default_rng(seed)


def derived_seed(base: int, *parts: str | int) -> int:
    """Deterministically derive a sub-seed from a base seed and identifying parts.

    Used so that, e.g., the bootstrap for (language=am, method=quantile, replicate=17) has
    its own reproducible stream that does not depend on execution order. Order independence
    matters: it means results are identical whether languages are processed sequentially or
    in any other order.
    """
    h = base
    for part in parts:
        for ch in str(part):
            h = (h * 1_000_003 + ord(ch)) % (2**31 - 1)
        h = (h * 31 + 17) % (2**31 - 1)
    return h
