"""SID_Set dataset extraction (saberzl/SID_Set on HuggingFace, see data/README.md).

Label convention: 0 = REAL (authentic, from OpenImages V7), 1 = FAKE (fully
synthetic, generated via FLUX) - matching aigc_detect.data's CIFAKEDataset
convention. Label 2 (tampered: partially-edited real photos with edit
masks) is excluded - it doesn't map to whole-image AIGC detection, which is
this project's scope.

Rather than persisting raw Parquet (which mixes in the tampered images/masks
we don't want, and stores synthetic images as lossless PNG), export_split_to_folders()
extracts just the kept images into a CIFAKE-style REAL/FAKE JPEG folder
layout, so the rest of the pipeline reuses aigc_detect.data.list_samples/
CIFAKEDataset directly instead of a separate dataset wrapper.

Every Parquet file (whether already local or fetched on demand) is read
directly via pyarrow, never through datasets.load_dataset(): the latter
writes a persistent Arrow-cache copy under ~/.cache/huggingface/datasets/
for every file it loads - confirmed empirically that this happens even
with keep_in_memory=True/disable_caching(), and that it doesn't shrink back
down after the source file is deleted, silently consuming disk space.
Streaming mode avoids that cache, but its Image feature can't be forced to
stay undecoded (cast_column(..., decode=False) is a no-op there in
practice), which would force a lossy decode/re-encode round-trip on
already-JPEG images - reintroducing the double-compression problem
_save_image is specifically designed to avoid. Downloading each needed file
by exact name via huggingface_hub.hf_hub_download, then deleting the
downloaded blob after processing, sidesteps both problems at once.
"""
from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image as PILImage

DATASET_NAME = "saberzl/SID_Set"
EXCLUDED_LABEL = 2  # tampered - out of scope, see data/README.md
LABEL_TO_FOLDER = {0: "REAL", 1: "FAKE"}
JPEG_MAGIC = b"\xff\xd8\xff"


def _list_remote_parquet_filenames(split: str) -> list[str]:
    """List Parquet filenames (e.g. 'train-00000-of-00249.parquet') for a split."""
    all_files = list_repo_files(DATASET_NAME, repo_type="dataset")
    prefix = f"data/{split}-"
    return sorted(Path(f).name for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))


def _save_image(raw_bytes: bytes, dest_path: Path) -> None:
    """Save one image as JPEG at dest_path.

    Already-JPEG bytes are written as-is (no decode/re-encode) - this avoids
    a second round of JPEG compression, which would otherwise leave real
    (JPEG-native) images with two compression passes and synthetic
    (PNG-native) images with only one, a detectable and label-correlated
    artifact that would be a spurious shortcut for the model. Anything else
    (PNG) is decoded and re-encoded as JPEG quality 90, exactly once.
    """
    if raw_bytes[:3] == JPEG_MAGIC:
        dest_path.write_bytes(raw_bytes)
    else:
        image = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        image.save(dest_path, format="JPEG", quality=90)


def export_split_to_folders(
    split: str,
    output_root: Path,
    target_per_label: dict[int, int] | None = None,
    local_parquet_dir: Path | None = None,
    log_every: int = 1000,
) -> dict[int, int]:
    """Extract non-tampered images from a SID_Set split into output_root/{REAL,FAKE}/.

    Processes already-downloaded local Parquet files matching
    `{split}-*.parquet` under local_parquet_dir first, deleting each one
    once its images are saved (only after a successful write) to reclaim
    space as it goes. If target_per_label counts aren't met from local
    files alone (or none were given), downloads whichever remaining
    Parquet files for this split aren't already local (by exact filename,
    via huggingface_hub), processing and deleting each the same way, until
    targets are met. With target_per_label=None, processes every available
    file (all local, then everything remaining on the hub).
    """
    # Count what's already there so a resumed/re-run call doesn't collect
    # target_per_label images on top of a previous (possibly interrupted)
    # run's output instead of up to it.
    counts = {}
    for label, folder in LABEL_TO_FOLDER.items():
        folder_path = output_root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        counts[label] = sum(1 for _ in folder_path.glob("*.jpg"))

    def targets_met() -> bool:
        if target_per_label is None:
            return False
        return all(counts[label] >= target for label, target in target_per_label.items())

    def label_target_met(label: int) -> bool:
        return target_per_label is not None and counts[label] >= target_per_label.get(
            label, float("inf")
        )

    total_saved = 0

    def process_table(table: pq.Table) -> None:
        nonlocal total_saved
        for row in table.to_pylist():
            if targets_met():
                return
            label = row["label"]
            if label == EXCLUDED_LABEL or label_target_met(label):
                continue
            dest = output_root / LABEL_TO_FOLDER[label] / f"{row['img_id']}.jpg"
            _save_image(row["image"]["bytes"], dest)
            counts[label] += 1
            total_saved += 1
            if total_saved % log_every == 0:
                print(f"  saved {total_saved} images (real={counts[0]}, fake={counts[1]})", flush=True)

    # Persistent record of which source Parquet filenames have already been
    # fully extracted, surviving across separate calls/process restarts -
    # local files are deleted right after processing, so without this we'd
    # have no way to tell "already done" from "not fetched yet" on a resumed
    # run, and would re-download/re-extract (silently overwriting, since
    # filenames are the source's own img_id) files we already had.
    manifest_path = output_root / ".processed_sources.txt"
    processed_filenames: set[str] = set()
    if manifest_path.exists():
        processed_filenames = set(manifest_path.read_text().splitlines())

    def mark_processed(filename: str) -> None:
        processed_filenames.add(filename)
        with open(manifest_path, "a") as f:
            f.write(filename + "\n")

    if local_parquet_dir is not None:
        local_files = sorted(local_parquet_dir.glob(f"{split}-*.parquet"))
        for parquet_file in local_files:
            if targets_met():
                break
            if parquet_file.name not in processed_filenames:
                process_table(pq.read_table(parquet_file))
                mark_processed(parquet_file.name)
            parquet_file.unlink()  # already extracted (now or previously) either way

    if not targets_met():
        remaining = [
            name for name in _list_remote_parquet_filenames(split) if name not in processed_filenames
        ]
        for filename in remaining:
            if targets_met():
                break
            downloaded_path = Path(
                hf_hub_download(DATASET_NAME, filename=f"data/{filename}", repo_type="dataset")
            )
            process_table(pq.read_table(downloaded_path))
            mark_processed(filename)
            # hf_hub_download's returned path is typically a symlink into the
            # shared HF cache - resolve it to delete the actual blob, not
            # just the symlink, so the space is genuinely reclaimed.
            real_path = downloaded_path.resolve()
            downloaded_path.unlink(missing_ok=True)
            if real_path.exists():
                real_path.unlink()

    return counts
