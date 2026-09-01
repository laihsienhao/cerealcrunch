#!/usr/bin/env python
"""Selectively extract a stratified sample from ArtiFact (bitmind/ArtiFact).

Usage:
    python scripts/extract_artifact.py --n_per_fake_generator 280 --n_total_real 7000

Lists the remote archive's ~2.5M entries (a few tens of seconds, no payload
download), then fetches only `n_per_fake_generator` images from each of the
25 fake generators and `n_total_real` spread evenly across the 8 real
sources - via `remotezip` HTTP range requests, not a bulk 31.7GB download.
Resumable: safe to re-run after an interruption, already-fetched files are
skipped. Follow with scripts/prepare_artifact_split.py to produce a
stratified train/validation/test split from the extracted pool.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigc_detect.artifact_data import extract_selected, group_by_top_folder, list_remote_entries, select_members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a stratified sample from ArtiFact.")
    parser.add_argument("--output_root", type=Path, default=Path("data/raw/artifact_crossgen/test"))
    parser.add_argument("--n_per_fake_generator", type=int, default=280)
    parser.add_argument("--n_total_real", type=int, default=7000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Listing remote archive contents (no payload download)...", flush=True)
    names = list_remote_entries()
    fake_groups, real_groups = group_by_top_folder(names)
    print(f"Fake generators found: {len(fake_groups)}, Real sources found: {len(real_groups)}", flush=True)

    selected = select_members(fake_groups, real_groups, args.n_per_fake_generator, args.n_total_real)
    n_fake = sum(1 for _, label, _ in selected if label == "FAKE")
    n_real = sum(1 for _, label, _ in selected if label == "REAL")
    print(f"Selected {len(selected)} files total ({n_fake} FAKE, {n_real} REAL)", flush=True)

    counts = extract_selected(selected, args.output_root)
    print(
        f"Done: fetched {counts['fetched_this_run']} new files "
        f"({counts['skipped']} already present, {counts['real']} REAL + {counts['fake']} FAKE total on disk)."
    )


if __name__ == "__main__":
    main()
