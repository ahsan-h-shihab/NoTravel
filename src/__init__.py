"""Project source package.

WHY THIS FILE IMPORTS TORCH FIRST
---------------------------------
On this Windows environment, importing scikit-learn before PyTorch makes torch fail with:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
    Error loading "...\\torch\\lib\\c10.dll" or one of its dependencies.

Verified directly (2026-08-02):

    python -c "import torch, sklearn.linear_model, pandas"   -> OK
    python -c "import sklearn.linear_model, pandas, torch"   -> WinError 1114
    python -c "import pandas, torch"                         -> OK

The cause is an OpenMP runtime conflict: scikit-learn and torch each vendor an OpenMP
runtime, and whichever loads second fails to initialise. Importing torch here -- before any
module that pulls in scikit-learn -- makes every entry point safe regardless of its own
import order.

This is load-bearing. Removing it reintroduces a failure that presents as a mysterious DLL
error hundreds of lines away from its actual cause.

WHY THIS FILE PINS THE HUGGINGFACE CACHE
----------------------------------------
The system drive on the reference machine has very little free space, so model and dataset
caches must not land in the default per-user location. Setting HF_HOME here -- before
`huggingface_hub` is imported anywhere -- makes every entry point use a single repo-local
cache with no external environment setup, which is both a storage requirement and a
reproducibility win: there is exactly one cache, and it travels with the repository.

An externally-set HF_HOME is always respected, so a user with a shared cache elsewhere is
not forced to re-download.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Must happen BEFORE huggingface_hub / datasets / transformers are imported anywhere.
_os.environ.setdefault("HF_HOME", str(_Path(__file__).resolve().parents[1] / "data" / "cache"))
# Symlinks require elevated privileges on this Windows setup. `load_dataset` merely warns
# and falls back, but `snapshot_download` hard-fails with
#   OSError: [WinError 1314] A required privilege is not held by the client
# so symlinks are disabled outright rather than only silencing the warning. Files are copied
# into the cache instead of linked; at ~27 MB for the SIB-200 repo this costs nothing.
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Tokenizer fork-parallelism warnings are noise for single-process CPU runs.
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # pragma: no cover - environment guard, not logic
    import torch as _torch  # noqa: F401
except ImportError:
    # torch is genuinely optional for the pure-analysis modules (metrics, thresholds),
    # which must stay importable in a minimal environment.
    pass
