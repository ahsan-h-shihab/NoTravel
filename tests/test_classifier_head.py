"""Tests for deployed-classifier head interpretation (Arm A).

Two ways to silently produce a meaningless experiment while raising no error:

  1. Score the wrong class, inverting every result for that model.
  2. Apply softmax to a single-logit sigmoid head, which returns exactly 1.0 for EVERY
     input and yields a degenerate score vector.

The second is what Detoxify's multilingual classifier would have triggered: it exposes
num_labels = 1 with id2label = {0: 'toxic'}. Both are regression-tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.classifiers import _activation_for, _toxic_index


class _Config:
    def __init__(self, id2label, num_labels=None, problem_type=None):
        self.id2label = id2label
        self.num_labels = num_labels if num_labels is not None else len(id2label)
        self.problem_type = problem_type


# ------------------------------------------------------------------ activation selection

def test_single_logit_head_uses_sigmoid():
    """THE regression test. Detoxify exposes num_labels=1, id2label={0:'toxic'}.
    Softmax over one logit is identically 1.0 -- a silent, total failure."""
    assert _activation_for(_Config({0: "toxic"}, num_labels=1)) == "sigmoid"


def test_two_class_head_uses_softmax():
    assert _activation_for(_Config({0: "non-toxic", 1: "toxic"})) == "softmax"


def test_multi_label_head_uses_sigmoid():
    cfg = _Config({0: "toxic", 1: "obscene", 2: "threat"},
                  problem_type="multi_label_classification")
    assert _activation_for(cfg) == "sigmoid"


def test_softmax_over_one_logit_is_degenerate():
    """Demonstrates the failure the guard prevents, so nobody 'simplifies' it away."""
    import torch

    logits = torch.tensor([[-4.0], [0.0], [4.0]])          # wildly different inputs
    assert torch.allclose(torch.softmax(logits, dim=-1), torch.ones(3, 1))
    sig = torch.sigmoid(logits).squeeze()
    assert sig[0] < 0.1 and sig[1] == pytest.approx(0.5) and sig[2] > 0.9
    assert len(np.unique(sig.numpy())) == 3


# ---------------------------------------------------------------- positive-class lookup

def test_finds_toxic_class_regardless_of_index():
    assert _toxic_index(_Config({0: "non-toxic", 1: "toxic"})) == 1
    assert _toxic_index(_Config({0: "toxic", 1: "non-toxic"})) == 0


def test_does_not_mistake_non_toxic_for_toxic():
    """'non-toxic' contains 'toxic' as a substring; a naive match inverts the model."""
    assert _toxic_index(_Config({0: "non-toxic", 1: "toxic"})) == 1


def test_single_label_head_resolves():
    assert _toxic_index(_Config({0: "toxic"}, num_labels=1)) == 0


def test_refuses_to_guess_on_uninformative_labels():
    """Better to fail loudly than to score the wrong class."""
    with pytest.raises(ValueError):
        _toxic_index(_Config({0: "LABEL_0", 1: "LABEL_2"}))


def test_refuses_when_no_label_map():
    with pytest.raises(ValueError):
        _toxic_index(_Config({}))
