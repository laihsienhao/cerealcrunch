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

Since `CIFAKEDataset` (shared by both datasets) now always applies
`strip_source_artifacts` (see below), CIFAKE images get upsized to a
randomized 256-800px range before training/eval too - any CIFAKE numbers
from before that fix landed aren't directly comparable to later runs.

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
        test/
            REAL/
            FAKE/

(`test/` is our own held-out set, carved from `train` - see above.)

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

## Known dataset confound: resolution/aspect-ratio/compression shortcut

Direct inspection (sampling 1000+ images per class per split) found that
every FAKE image in SID_Set is **exactly 1024x1024** and shares **one single,
identical JPEG quantization-table signature**, while REAL images have
diverse native resolutions (267-1024px), diverse aspect ratios, and a
genuinely mixed set of quantization signatures - identically across train,
validation, and test. This traces back to the source dataset: FAKE images
were very likely stored as lossless PNGs at a fixed diffusion-model-typical
1024x1024, while REAL images are natively JPEG from varied real-world
sources at native resolutions. A model can trivially separate the two
classes by native resolution/aspect-ratio/compression fingerprint alone,
with zero regard for actual image content - and since JPEG quantization
artifacts survive resizing, this shortcut also survives all 6 PROBLEM.md
robustness transforms, producing misleadingly uniform "robust" numbers.

**Fix:** `aigc_detect.transforms.strip_source_artifacts` runs on every image
before anything else, unconditionally, both classes, every split (train,
validation, test) and every robustness-eval condition including "clean" -
it randomizes aspect ratio (via a random-aspect crop), resolution (via a
resize to a randomized shorter-side length), and JPEG quality (via a
re-encode), so none of the three can carry label information. See
`CIFAKEDataset.__getitem__` (`aigc_detect/data.py`) and `apply_condition`
(`aigc_detect/evaluate.py`) for where it's wired in.

**Known, deliberately left unfixed:** direct inspection also found 130/300
sampled REAL images carry an embedded ICC color profile vs. 0/300 FAKE, and
3/300 REAL images are true grayscale vs. 0/300 FAKE. Neither is read as
metadata by the model (only pixel arrays are ever fed in), but both likely
mark genuine, systematic differences in real-camera vs. generator color/tone
processing. This is double-edged: it's simultaneously a legitimate forensic
signal used throughout the AIGC-detection literature, and a risk of being a
narrow single-source fingerprint that won't generalize to unseen generators
- fully resolving it needs multi-source training data, out of scope here.

**Why this matters beyond just fixing a bug:** an official organizer webinar (see `PROBLEM.md`'s "Webinar Information" section) explicitly frames this exact failure mode as the key thing to guard against - "do not just fine-tune a classifier; think about what your model is actually learning - is it a real artifact, or a dataset shortcut?" - and a cited NeurIPS 2025 finding (DDA) warns almost verbatim that "JPEG in your real images can become a spurious signal." This confound was found and fixed through that same discipline, not in response to the webinar - but it's a direct, concrete demonstration of exactly the risk the organizers flagged. Relatedly, the webinar specifies the scoring formula `0.50 x AUC_clean + 0.50 x AUC_robust` (ROC AUC, not accuracy, as the primary metric) - `aigc_detect.evaluate.compute_final_score` implements this directly, rather than us picking accuracy as a headline number arbitrarily.

## Not yet added

- WildFake (ModelScope, needs the site's translation step before use): https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- Demo/validation-only set: COCO val2017 (4998 real) + DALL·E Advanced (8843 AIGC) —
  for tracking iterative progress only, never for training (per PROBLEM.md).
