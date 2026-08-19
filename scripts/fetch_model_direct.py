"""Resumable direct download of a pinned model into a local directory.

Why this exists: `hf_hub_download` repeatedly stalled at zero bytes on this machine for one
1.1 GB weight file, while a direct HTTP range request to the same URL returned data
immediately. The failure is in the library's resume/lock handling, not the network, and it
was blocking a pre-registered experiment (EXP-009).

This downloader is deliberately minimal and resumable: it requests explicit byte ranges,
appends to a partial file, and can be re-run to continue. Every file is fetched from a
`resolve/<revision>/` URL, so the pinning guarantee is identical to the library's -- nothing
about reproducibility is weakened by bypassing it.

Usage:
    python scripts/fetch_model_direct.py --repo unitary/multilingual-toxic-xlm-roberta \\
        --revision 4ad6f5c1... --out data/models/unitary-multilingual-toxic-xlm-roberta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Files a sequence-classification model needs. Missing optional ones are skipped, not fatal:
#: repositories differ in which tokenizer artifacts they ship.
REQUIRED = ("config.json", "pytorch_model.bin")
OPTIONAL = ("sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.txt", "merges.txt", "added_tokens.json")

CHUNK = 8 * 1024 * 1024  # 8 MB ranges: small enough to survive a flaky connection


def _url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _remote_size(url: str, timeout: int = 60) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "fetch-direct"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch(repo: str, revision: str, filename: str, dest: Path,
          max_retries: int = 40, timeout: int = 120) -> bool:
    """Download one file, resuming from whatever is already on disk."""
    url = _url(repo, revision, filename)
    total = _remote_size(url, timeout=timeout)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and total is not None and dest.stat().st_size == total:
        print(f"  {filename}: already complete ({total/1e6:.1f} MB)")
        return True

    have = dest.stat().st_size if dest.exists() else 0
    if total is None:
        print(f"  {filename}: size unknown; single-shot fetch")

    t0 = time.perf_counter()
    stalls = 0
    while total is None or have < total:
        end = have + CHUNK - 1 if total is None else min(have + CHUNK - 1, total - 1)
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={have}-{end}", "User-Agent": "fetch-direct"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            stalls += 1
            if stalls > max_retries:
                print(f"  {filename}: FAILED after {stalls} retries ({exc})")
                return False
            time.sleep(min(2 ** min(stalls, 6), 30))
            continue

        if not data:
            stalls += 1
            if stalls > max_retries:
                print(f"  {filename}: FAILED — no data after {stalls} attempts")
                return False
            time.sleep(2)
            continue

        with open(dest, "ab") as fh:
            fh.write(data)
        have += len(data)
        stalls = 0

        if total:
            pct = 100.0 * have / total
            rate = have / max(time.perf_counter() - t0, 1e-9) / 1e6
            print(f"  {filename}: {have/1e6:7.1f}/{total/1e6:.1f} MB "
                  f"({pct:5.1f}%) {rate:.2f} MB/s", flush=True)
        if total is None:
            break

    print(f"  {filename}: done in {time.perf_counter()-t0:.0f}s")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.repo} @ {args.revision[:12]} -> {out}")

    ok = True
    for name in REQUIRED:
        if not fetch(args.repo, args.revision, name, out / name):
            ok = False
    for name in OPTIONAL:
        url = _url(args.repo, args.revision, name)
        if _remote_size(url) is None:
            continue  # not present in this repository
        fetch(args.repo, args.revision, name, out / name)

    # Record provenance so a locally-fetched model is as traceable as a cached one.
    manifest = {"repo": args.repo, "revision": args.revision, "files": {}}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "download_manifest.json":
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    h.update(chunk)
            manifest["files"][p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    (out / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("COMPLETE" if ok else "INCOMPLETE — required file missing")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
