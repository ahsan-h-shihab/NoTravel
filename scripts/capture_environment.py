"""Capture a machine-readable snapshot of the execution environment.

Why this exists (CONFIGURATION MANAGEMENT, § REPRODUCIBILITY CONSTITUTION):
a scientific result without its environment is incomplete. This script is the single
authoritative source for the environment record; any prose description is derived
from its output, never hand-maintained independently.

Usage
-----
    python scripts/capture_environment.py
    python scripts/capture_environment.py --out environment/environment_snapshot.json

The snapshot is deliberately dependency-light so it runs even when the scientific stack is
broken -- diagnosing a broken environment is exactly when it is most needed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Packages whose versions can plausibly change a reported number.
# Keep this list explicit rather than dumping `pip freeze`: an explicit list documents
# which dependencies we consider scientifically load-bearing.
TRACKED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "statsmodels",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "huggingface-hub",
    "sentence-transformers",
    "matplotlib",
    "seaborn",
    "nltk",
    "tqdm",
]


def _pkg_version(name: str) -> str | None:
    """Return an installed package version, or None if genuinely absent.

    Distribution metadata is not always trustworthy. On this machine matplotlib's
    installed METADATA carries no `Version` field, so both `pip list` and
    `importlib.metadata.version()` report a null version for a package that imports and
    renders correctly. A falsy metadata result is therefore treated as "unknown", not as
    "absent", and we fall back to the module's own `__version__`.
    """
    version = None
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = None
    except Exception:  # corrupt or unreadable dist-info
        version = None
    if version:
        return version

    module_name = name.replace("-", "_")
    try:
        module = __import__(module_name)
    except Exception:
        return None  # genuinely unavailable
    return getattr(module, "__version__", None) or "installed-version-unknown"


def _git(*args: str) -> str | None:
    """Run a git command, returning stripped stdout or None.

    Returns None on non-zero exit so that an empty repository (where `rev-parse HEAD`
    fails but still echoes "HEAD" on stdout) records a null commit rather than the
    literal string "HEAD".
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _torch_info() -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
    }


def _cpu_name() -> str:
    # platform.processor() is uninformative on Linux; try a few sources.
    name = platform.processor() or ""
    if not name and sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return name or "unknown"


def collect() -> dict:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": _cpu_name(),
            "logical_cores": os.cpu_count(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {name: _pkg_version(name) for name in TRACKED_PACKAGES},
        "torch": _torch_info(),
        "thread_env": {
            var: os.environ.get(var)
            for var in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            # `symbolic-ref` works on an empty repository; `rev-parse --abbrev-ref` does not.
            "branch": _git("symbolic-ref", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "environment" / "environment_snapshot.json",
        help="Destination JSON file.",
    )
    args = parser.parse_args()

    snapshot = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    print(f"Environment snapshot written to {args.out}")
    missing = [name for name, ver in snapshot["packages"].items() if ver is None]
    if missing:
        print(f"WARNING: tracked packages not installed: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
