"""Provenance capture for every experiment run.

Why this exists: EXPERIMENT TRACKING requires that every experiment carry a
permanent identity and a fifteen-field provenance record, and RESULT
TRACEABILITY requires every reported number to trace back through it. Provenance is written
by code, never by hand, so it cannot drift from what actually ran.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "statsmodels",
    "torch", "transformers", "tokenizers", "datasets", "sentence-transformers",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file, streamed so large artifacts do not need to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str | None:
    """Current commit SHA, or None.

    Returns None (never the literal string "HEAD") when the repository has no commits --
    a silently wrong commit field is worse than a missing one, because it looks valid.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_is_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes.

    A dirty tree means the recorded commit does not fully describe the code that ran, so
    this flag is surfaced rather than hidden.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def software_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        version = None
        try:
            version = metadata.version(name)
        except Exception:
            version = None
        if not version:  # corrupt dist-info: fall back to the module itself
            try:
                version = getattr(__import__(name.replace("-", "_")), "__version__", None)
            except Exception:
                version = None
        versions[name] = version
    return versions


def environment() -> dict[str, Any]:
    import os

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "python_executable": sys.executable,
    }


class RunRecord:
    """Accumulates a provenance record for one experiment run and writes it atomically.

    Usage::

        rec = RunRecord("EXP-001", objective="...", hypothesis="H1")
        rec.set(dataset="...", dataset_version="...", config={...}, random_seed=42)
        ...
        rec.add_output("results/tables/x.csv")
        rec.finish(status="COMPLETE", interpretation="...")
    """

    def __init__(self, experiment_id: str, objective: str, hypothesis: str,
                 run_dir: str | Path | None = None) -> None:
        self.experiment_id = experiment_id
        self.run_dir = Path(run_dir) if run_dir else REPO_ROOT / "experiments" / "runs" / experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "experiment_id": experiment_id,
            "objective": objective,
            "hypothesis": hypothesis,
            "dataset": None,
            "dataset_version": None,
            "model": None,
            "model_version": None,
            "config": None,
            "random_seed": None,
            "software_versions": software_versions(),
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "environment": environment(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
            "wall_clock_s": None,
            "outputs": [],
            "status": "RUNNING",
            "interpretation": None,
        }
        self._t0 = datetime.now(timezone.utc)

    def set(self, **fields: Any) -> "RunRecord":
        unknown = set(fields) - set(self._data)
        if unknown:
            raise KeyError(f"Unknown provenance field(s): {sorted(unknown)}")
        self._data.update(fields)
        return self

    def add_output(self, path: str | Path) -> "RunRecord":
        p = Path(path)
        rel = p.relative_to(REPO_ROOT) if p.is_absolute() and p.is_relative_to(REPO_ROOT) else p
        self._data["outputs"].append({
            "path": str(rel).replace("\\", "/"),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })
        return self

    def finish(self, status: str = "COMPLETE", interpretation: str | None = None) -> Path:
        if status not in {"COMPLETE", "FAILED", "ABANDONED"}:
            raise ValueError(f"Invalid terminal status: {status}")
        now = datetime.now(timezone.utc)
        self._data.update({
            "status": status,
            "interpretation": interpretation,
            "finished_at": now.isoformat(timespec="seconds"),
            "wall_clock_s": round((now - self._t0).total_seconds(), 2),
        })
        out = self.run_dir / "provenance.json"
        out.write_text(json.dumps(self._data, indent=2) + "\n", encoding="utf-8")
        return out

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)
