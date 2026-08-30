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

## Not yet added

- SID_Set (HuggingFace, higher resolution): https://huggingface.co/datasets/saberzl/SID_Set
- WildFake (ModelScope, needs the site's translation step before use): https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- Demo/validation-only set: COCO val2017 (4998 real) + DALL·E Advanced (8843 AIGC) —
  for tracking iterative progress only, never for training (per PROBLEM.md).
