import io

import numpy as np
from datasets import Dataset as HFDataset
from datasets import Image as HFImage
from PIL import Image

from aigc_detect.sid_data import _save_image, export_split_to_folders


def _make_pil_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_save_image_writes_jpeg_bytes_as_is(tmp_path):
    buf = io.BytesIO()
    _make_pil_image(0).save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()

    dest = tmp_path / "out.jpg"
    _save_image(jpeg_bytes, dest)

    assert dest.read_bytes() == jpeg_bytes  # written as-is, no re-encoding


def test_save_image_converts_png_to_jpeg(tmp_path):
    buf = io.BytesIO()
    _make_pil_image(0).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    dest = tmp_path / "out.jpg"
    _save_image(png_bytes, dest)

    result = Image.open(dest)
    assert result.format == "JPEG"
    assert result.size == (8, 8)


def test_export_split_to_folders_filters_and_stops_at_target(tmp_path):
    examples = (
        [{"img_id": f"real_{i}", "image": _make_pil_image(i), "label": 0} for i in range(3)]
        + [{"img_id": f"fake_{i}", "image": _make_pil_image(10 + i), "label": 1} for i in range(3)]
        + [
            {"img_id": f"tampered_{i}", "image": _make_pil_image(20 + i), "label": 2}
            for i in range(3)
        ]
    )

    ds = HFDataset.from_list(examples)
    ds = ds.cast_column("image", HFImage())

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    parquet_path = parquet_dir / "train-00000-of-00001.parquet"
    ds.to_parquet(str(parquet_path))

    output_root = tmp_path / "out"
    counts = export_split_to_folders(
        split="train",
        output_root=output_root,
        target_per_label={0: 2, 1: 2},
        local_parquet_dir=parquet_dir,
    )

    assert counts == {0: 2, 1: 2}
    assert len(list((output_root / "REAL").glob("*.jpg"))) == 2
    assert len(list((output_root / "FAKE").glob("*.jpg"))) == 2
    assert not any("tampered" in p.name for p in output_root.rglob("*.jpg"))
    assert not parquet_path.exists()  # deleted once its images were extracted


def test_export_split_to_folders_resumes_from_existing_output(tmp_path):
    """A target already satisfied by pre-existing output shouldn't collect more."""
    output_root = tmp_path / "out"
    (output_root / "REAL").mkdir(parents=True)
    (output_root / "FAKE").mkdir(parents=True)
    for i in range(2):
        (output_root / "REAL" / f"existing_real_{i}.jpg").write_bytes(b"fake-jpeg-bytes")
        (output_root / "FAKE" / f"existing_fake_{i}.jpg").write_bytes(b"fake-jpeg-bytes")

    counts = export_split_to_folders(
        split="train",
        output_root=output_root,
        target_per_label={0: 2, 1: 2},
        local_parquet_dir=None,
    )

    assert counts == {0: 2, 1: 2}
    # Nothing new should have been added - targets were already met.
    assert len(list((output_root / "REAL").glob("*.jpg"))) == 2
    assert len(list((output_root / "FAKE").glob("*.jpg"))) == 2


def _write_local_parquet(path, examples):
    ds = HFDataset.from_list(examples)
    ds = ds.cast_column("image", HFImage())
    ds.to_parquet(str(path))


def test_export_split_to_folders_manifest_prevents_reprocessing_deleted_file(tmp_path):
    """A source file already recorded as processed shouldn't be re-extracted,
    even if a file with that same name reappears locally (e.g. re-downloaded) -
    this is what actually prevents overwrite-without-progress on a resumed run,
    distinct from the "target already met" short-circuit tested above."""
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    file_a = parquet_dir / "train-00000-of-00002.parquet"
    _write_local_parquet(file_a, [{"img_id": f"a_{i}", "image": _make_pil_image(i), "label": 0} for i in range(2)])

    output_root = tmp_path / "out"
    counts = export_split_to_folders(
        split="train", output_root=output_root, target_per_label={0: 2}, local_parquet_dir=parquet_dir
    )
    assert counts == {0: 2, 1: 0}

    manifest = output_root / ".processed_sources.txt"
    assert manifest.exists()
    assert file_a.name in manifest.read_text().splitlines()

    # Recreate a file with the SAME name (simulating a re-download) plus a
    # genuinely new file B; raise the target so the run must look for more.
    _write_local_parquet(file_a, [{"img_id": f"a_dup_{i}", "image": _make_pil_image(i), "label": 0} for i in range(2)])
    file_b = parquet_dir / "train-00001-of-00002.parquet"
    _write_local_parquet(file_b, [{"img_id": f"b_{i}", "image": _make_pil_image(50 + i), "label": 0} for i in range(2)])

    counts2 = export_split_to_folders(
        split="train", output_root=output_root, target_per_label={0: 4}, local_parquet_dir=parquet_dir
    )

    # Only file B's 2 images should have been added - file A's images were
    # skipped (already in the manifest), not re-extracted under new names.
    assert counts2 == {0: 4, 1: 0}
    assert len(list((output_root / "REAL").glob("*.jpg"))) == 4
    assert not file_a.exists() and not file_b.exists()  # both deleted regardless
