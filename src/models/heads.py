"""Arm-B classification head: logistic regression on frozen embeddings.

Why logistic regression and not something stronger: the head is an *instrument*, not the
contribution. It must (a) produce calibrated-ish probabilities so that a threshold is
meaningful, (b) train in seconds on CPU, (c) be simple enough that no reviewer questions
whether the reported effect is an artefact of head capacity. IMPLEMENTATION
PHILOSOPHY: implementation exists to answer scientific questions, and is not itself the
contribution.

Regularisation strength is selected once on the SOURCE language's own dev split. Selecting
it per target language would leak target information into the model and would confound the
threshold effect this project is trying to isolate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..utils.seeding import DEFAULT_SEED

#: Candidate inverse-regularisation strengths, swept on source dev only.
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class TrainedHead:
    model: LogisticRegression
    source_language: str
    C: float
    seed: int
    n_train: int
    source_dev_auroc: float
    #: Class prior of the data the head was TRAINED on. The head's posteriors encode this
    #: prior, so every prior-shift correction must use it as the reference -- NOT the
    #: prevalence of whatever validation sample a threshold happened to be tuned on.
    #: Getting this wrong made SLD/EM report an estimated prior of 1.0 even for the SOURCE
    #: language, where no shift exists; that would have been published as a cross-lingual
    #: failure of a established method rather than a misuse of it.
    train_prior: float = 0.5

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return p(y=1 | x) for each row."""
        return self.model.predict_proba(embeddings)[:, 1]


def train_head(train_emb: np.ndarray, train_labels: np.ndarray,
               dev_emb: np.ndarray, dev_labels: np.ndarray,
               source_language: str, seed: int = DEFAULT_SEED,
               C_grid: tuple[float, ...] = C_GRID) -> TrainedHead:
    """Fit a logistic head on the source language, selecting C on source dev by AUROC.

    AUROC is used for model selection because it is threshold-independent: choosing C by a
    thresholded metric would entangle head selection with the very threshold question under
    study.
    """
    from ..eval.metrics import auroc

    best: tuple[float, float, LogisticRegression] | None = None
    for C in C_grid:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=seed, n_jobs=1,
        )
        clf.fit(train_emb, train_labels)
        score = auroc(clf.predict_proba(dev_emb)[:, 1], dev_labels)
        if best is None or score > best[1]:
            best = (C, score, clf)

    assert best is not None
    C, dev_auroc, clf = best
    return TrainedHead(
        model=clf,
        source_language=source_language,
        C=float(C),
        seed=seed,
        n_train=int(train_emb.shape[0]),
        source_dev_auroc=float(dev_auroc),
        train_prior=float(np.mean(np.asarray(train_labels) == 1)),
    )
