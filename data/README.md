# Data

This directory is gitignored (except this file and `.gitkeep` placeholders) —
datasets are large and are rebuilt from the sources below, not committed.

## CIFAKE (current)

Source: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Download the zip from Kaggle (requires a free Kaggle account) and extract it
so the layout looks like:

    data/raw/cifake/
        train/
            REAL/
            FAKE/
        test/
            REAL/
            FAKE/

Note: CIFAKE images are 32x32 (derived from CIFAR-10). That's fast for
getting the pipeline (data -> train -> eval) working end-to-end, but the
resolution doesn't match realistic photo sizes, so robustness-transform
severity and results here shouldn't be treated as representative. A
higher-resolution dataset (SID_Set) is planned before drawing real
conclusions.

## SID_Set (higher resolution, for real numbers)

Source: https://huggingface.co/datasets/saberzl/SID_Set (also documented in the
underlying paper, arXiv:2412.04292, "SIDA," CVPR 2025).

300K photo-realistic images (1024px-scale, not CIFAKE's 32x32) across 3 labels:

- **0 = real** (100K, from OpenImages V7)
- **1 = fully synthetic / AIGC** (100K, generated via FLUX)
- **2 = tampered** (100K: 80K object-tampered + 20K partially-tampered real
  photos with edit masks) — **excluded**. This project scopes AIGC detection
  at the whole-image level (per PROBLEM.md), and tampered images are a
  region-level manipulation-localization problem, not a fit for our binary
  REAL/FAKE framing.

On HuggingFace, SID_Set only ships `train` (249 Parquet files) and
`validation` (34 files) - there's no `test` split available there (the
dataset card's "test split" is only reachable via a separate GitHub repo,
which isn't worth the integration complexity). We carve our own held-out
test set from `train` instead, the same way we already carve a validation
split out of CIFAKE's `train` folder.

**Final layout** (produced by `scripts/prepare_sid_set.py`, see below):

    data/raw/sid_set/
        train/
            REAL/
            FAKE/
        validation/
            REAL/
            FAKE/

This matches CIFAKE's `REAL`/`FAKE` folder convention exactly, so
`aigc_detect.data.list_samples`/`CIFAKEDataset`/`split_train_val` load it
directly - no separate dataset loader needed for training.

**How it's prepared:** raw Parquet (as downloaded from HuggingFace) mixes
all 3 labels together within each file and stores synthetic images as
lossless PNG, so it's not stored as-is. `scripts/prepare_sid_set.py`
(`aigc_detect/sid_data.py`) reads each Parquet file (local files first, then
network streaming for anything beyond what's local), keeps only real/fake
images, and saves them as JPEGs - raw Parquet is a transient intermediate
that gets deleted once its images are extracted, not something to expect to
find afterward. Already-JPEG images (real) are copied byte-for-byte, not
re-encoded, to avoid a double-JPEG-compression artifact that would
otherwise correlate with label (see the module docstring); only PNG
(synthetic) images get converted to JPEG, once.

Size/time warning: this is a large, photo-realistic dataset - a full
extraction run is a real time investment (tens of minutes, depending on how
much is already local vs. needs streaming), not a quick pipeline check.
`tests/test_sid_data.py` only exercises this logic against small synthetic
fixtures, not the real dataset, so the test suite stays fast and offline.

## Not yet added

- WildFake (ModelScope, needs the site's translation step before use): https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- Demo/validation-only set: COCO val2017 (4998 real) + DALL·E Advanced (8843 AIGC) —
  for tracking iterative progress only, never for training (per PROBLEM.md).
