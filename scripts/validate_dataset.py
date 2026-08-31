#!/usr/bin/env python
"""Validate image integrity across all dataset folders before a long training run.

Opens and fully decodes every image (catching truncation that a lazy
Image.open() alone would miss), and cross-checks folder counts against
known-good values. Run this once before committing to a real, unrepeatable
training run.

Usage:
    python scripts/validate_dataset.py
"""
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

FOLDERS = {
    "cifake_train_real": Path("data/raw/cifake/train/REAL"),
    "cifake_train_fake": Path("data/raw/cifake/train/FAKE"),
    "cifake_test_real": Path("data/raw/cifake/test/REAL"),
    "cifake_test_fake": Path("data/raw/cifake/test/FAKE"),
    "sid_train_real": Path("data/raw/sid_set/train/REAL"),
    "sid_train_fake": Path("data/raw/sid_set/train/FAKE"),
    "sid_validation_real": Path("data/raw/sid_set/validation/REAL"),
    "sid_validation_fake": Path("data/raw/sid_set/validation/FAKE"),
    "sid_test_real": Path("data/raw/sid_set/test/REAL"),
    "sid_test_fake": Path("data/raw/sid_set/test/FAKE"),
}

EXPECTED_COUNTS = {
    "cifake_train_real": 50000,
    "cifake_train_fake": 50000,
    "cifake_test_real": 10000,
    "cifake_test_fake": 10000,
    "sid_train_real": 50000,
    "sid_train_fake": 50000,
    "sid_validation_real": 10000,
    "sid_validation_fake": 10000,
    "sid_test_real": 10000,
    "sid_test_fake": 10000,
}


def _check_image(path: Path) -> str | None:
    """Return an error string if the image fails to fully decode, else None."""
    try:
        with Image.open(path) as img:
            img.convert("RGB").load()
        return None
    except Exception as exc:
        return str(exc)


def validate_folder(folder: Path, max_workers: int = 8) -> tuple[int, list[Path]]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    bad_files: list[Path] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path, error in zip(files, executor.map(_check_image, files)):
            if error is not None:
                bad_files.append(path)
                print(f"  CORRUPT: {path} ({error})", flush=True)

    return len(files), bad_files


def main() -> None:
    total_bad = 0
    for name, folder in FOLDERS.items():
        if not folder.exists():
            print(f"{name}: folder not found, skipping ({folder})")
            continue

        count, bad_files = validate_folder(folder)
        expected = EXPECTED_COUNTS.get(name)
        status = "OK" if expected is None or count == expected else f"MISMATCH (expected {expected})"
        print(f"{name}: {count} files, {len(bad_files)} corrupt - {status}", flush=True)
        total_bad += len(bad_files)

    print()
    if total_bad == 0:
        print("All images validated successfully - no corrupt files found.")
    else:
        print(f"WARNING: {total_bad} corrupt file(s) found - review the list above.")


if __name__ == "__main__":
    main()
