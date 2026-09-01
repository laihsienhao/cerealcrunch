"""Selective extraction from ArtiFact (bitmind/ArtiFact on Hugging Face).

ArtiFact ships as a single ~31.7GB monolithic zip with no per-generator
split, unlike SID_Set's per-file Parquet shards - downloading it in full
just to keep a small stratified sample isn't practical. Instead this uses
`remotezip` (HTTP range requests) to list the archive's ~2.5M entries
without downloading the payload, then selectively fetches only the files
belonging to the requested per-generator/per-source sample sizes.

See data/README.md for the dataset background and why a stratified sample
(not the full archive) is what we actually want here.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from remotezip import RemoteZip

ARTIFACT_URL = "https://huggingface.co/datasets/bitmind/ArtiFact/resolve/main/ArtiFact.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_remote_entries(url: str = ARTIFACT_URL) -> list[str]:
    """Lists every entry in the remote zip's central directory - a few tens
    of seconds over the network, but doesn't touch the 31.7GB payload."""
    with RemoteZip(url) as rz:
        return rz.namelist()


def group_by_top_folder(names: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Splits ArtiFact/{Real,Fake}/<source_or_generator>/... entries into two
    dicts keyed by the 25 fake-generator or 8 real-source folder names."""
    fake_groups: dict[str, list[str]] = defaultdict(list)
    real_groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if name.endswith("/") or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        label, top = parts[1], parts[2]
        if label == "Fake":
            fake_groups[top].append(name)
        elif label == "Real":
            real_groups[top].append(name)
    return dict(fake_groups), dict(real_groups)


def select_members(
    fake_groups: dict[str, list[str]],
    real_groups: dict[str, list[str]],
    n_per_fake_generator: int,
    n_total_real: int,
) -> list[tuple[str, str, str]]:
    """Picks a stratified sample: `n_per_fake_generator` from each of the 25
    fake generators, and `n_total_real` spread as evenly as possible across
    the 8 real sources. Returns (zip_member_name, local_label, source_name)
    tuples. Selection is deterministic (sorted filenames, first N) - not
    randomized, since remotezip's per-file fetch cost doesn't depend on
    position in the archive, so there's no efficiency reason to prefer one
    over the other, but a fixed, reproducible choice is easier to reason about.
    """
    selected: list[tuple[str, str, str]] = []
    for generator, names in sorted(fake_groups.items()):
        for name in sorted(names)[:n_per_fake_generator]:
            selected.append((name, "FAKE", generator))

    real_sources = sorted(real_groups.keys())
    base, remainder = divmod(n_total_real, len(real_sources))
    for i, source in enumerate(real_sources):
        n = base + (1 if i < remainder else 0)
        for name in sorted(real_groups[source])[:n]:
            selected.append((name, "REAL", source))
    return selected


def extract_selected(
    selected: list[tuple[str, str, str]],
    output_root: Path,
    url: str = ARTIFACT_URL,
    log_every: int = 200,
) -> dict[str, int]:
    """Fetches each selected member and writes it to
    `output_root/{REAL,FAKE}/<source>__<basename>.jpg`.

    Resumable: skips anything already on disk (destination filenames are
    deterministic), so an interrupted run (this dataset is large enough that
    network hiccups are a real risk) can just be restarted rather than
    re-fetching everything.
    """
    for label in ("REAL", "FAKE"):
        (output_root / label).mkdir(parents=True, exist_ok=True)

    todo = []
    n_skipped = 0
    for member, label, source in selected:
        dest_path = output_root / label / f"{source}__{Path(member).name}"
        if dest_path.exists():
            n_skipped += 1
        else:
            todo.append((member, label, dest_path))
    print(f"Already on disk: {n_skipped}. Remaining to fetch: {len(todo)}.", flush=True)

    start = time.time()
    n_done = 0
    with RemoteZip(url) as rz:
        for member, label, dest_path in todo:
            dest_path.write_bytes(rz.read(member))
            n_done += 1
            if n_done % log_every == 0:
                elapsed = time.time() - start
                rate = elapsed / n_done
                remaining = (len(todo) - n_done) * rate
                print(
                    f"  {n_done}/{len(todo)} fetched ({elapsed:.0f}s elapsed, "
                    f"~{remaining:.0f}s remaining at {rate:.2f}s/file)",
                    flush=True,
                )

    n_real = sum(1 for p in output_root.glob("REAL/*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    n_fake = sum(1 for p in output_root.glob("FAKE/*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return {"real": n_real, "fake": n_fake, "fetched_this_run": n_done, "skipped": n_skipped}
