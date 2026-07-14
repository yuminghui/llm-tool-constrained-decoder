"""
Download raw open-source datasets and convert them to the experiment contract.

Usage::

    # download + convert one or all datasets (full)
    python -m experiments.others.prepare_data --dataset toolalpaca
    python -m experiments.others.prepare_data --dataset all --limit 150

    # skip download if raw files are already present
    python -m experiments.others.prepare_data --dataset seal_tools --skip-download

    # (re)build the tiny committed offline fixtures from converted data
    python -m experiments.others.prepare_data --dataset sample --limit 5

Converted files land in ``experiments/others/data/<dataset>/{evaluate,tools}.json``
(git-ignored); raw downloads cache under ``experiments/others/data/_raw/<dataset>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Callable, Dict, List, Tuple

from experiments.others.converters import toolalpaca, seal_tools, api_bank
from experiments.others.converters import write_converted, dedup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
RAW_ROOT = os.path.join(DATA_ROOT, "_raw")
SAMPLE_ROOT = os.path.join(DATA_ROOT, "sample")

MODULES = {
    "toolalpaca": toolalpaca,
    "seal_tools": seal_tools,
    "api_bank": api_bank,
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, retries: int = 3, backoff: float = 1.5) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ptc-decoder-eval"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last}")


def _download_to(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = _fetch(url)
    with open(dest, "wb") as f:
        f.write(data)


def _download_dataset(name: str, raw_dir: str, limit: int) -> None:
    module = MODULES[name]
    os.makedirs(raw_dir, exist_ok=True)
    if name == "api_bank":
        files = api_bank.remote_file_urls(limit=limit, opener=_fetch)
        print(f"  [api_bank] {len(files)} dialog files to download...")
        for i, (url, rel) in enumerate(files, 1):
            dest = os.path.join(raw_dir, rel)
            if not os.path.isfile(dest):
                _download_to(url, dest)
            if i % 25 == 0:
                print(f"    ...{i}/{len(files)}")
    else:
        for url, fname in module.SOURCES:
            dest = os.path.join(raw_dir, fname)
            print(f"  [{name}] downloading {fname} ...")
            _download_to(url, dest)


# ---------------------------------------------------------------------------
# Convert / sample
# ---------------------------------------------------------------------------

def prepare(name: str, limit: int, seed: int, n_distractors: int, skip_download: bool) -> Dict[str, int]:
    raw_dir = os.path.join(RAW_ROOT, name)
    out_dir = os.path.join(DATA_ROOT, name)
    if not skip_download:
        _download_dataset(name, raw_dir, limit)

    module = MODULES[name]
    kwargs = {"limit": limit, "seed": seed}
    if name in ("seal_tools", "api_bank"):
        kwargs["n_distractors"] = n_distractors
    stats = module.convert(raw_dir, out_dir, **kwargs)
    print(f"  [{name}] wrote {stats['tasks']} tasks / {stats['tools']} tools → {out_dir}")
    return stats


def make_sample(name: str, n: int) -> Dict[str, int]:
    """Build a tiny committed fixture by truncating the converted dataset."""
    src = os.path.join(DATA_ROOT, name)
    ev_path = os.path.join(src, "evaluate.json")
    tj_path = os.path.join(src, "tools.json")
    if not (os.path.isfile(ev_path) and os.path.isfile(tj_path)):
        print(f"  [sample:{name}] no converted data at {src} — run prepare first; skipping")
        return {"tasks": 0, "tools": 0}
    with open(ev_path, "r", encoding="utf-8") as f:
        records = json.load(f)[:n]
    with open(tj_path, "r", encoding="utf-8") as f:
        tool_defs = {d["name"]: d for d in json.load(f)}
    out_dir = os.path.join(SAMPLE_ROOT, name)
    stats = write_converted(out_dir, records, tool_defs)
    print(f"  [sample:{name}] wrote {stats['tasks']} tasks / {stats['tools']} tools → {out_dir}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare external datasets for PTC-Decoder experiments")
    p.add_argument("--dataset", required=True,
                   choices=["toolalpaca", "seal_tools", "api_bank", "all", "sample"])
    p.add_argument("--limit", type=int, default=0, help="max tasks (0 = all); for api_bank also caps files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-distractors", type=int, default=12)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    if args.dataset == "sample":
        n = args.limit or 5
        for name in MODULES:
            make_sample(name, n)
        return

    names = list(MODULES) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"\n=== preparing {name} ===")
        prepare(name, args.limit, args.seed, args.n_distractors, args.skip_download)


if __name__ == "__main__":
    main()
