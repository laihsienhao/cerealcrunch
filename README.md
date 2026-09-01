# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 submission — Problem Statement 5

Cereal Crunch is a binary AI-generated-image classifier (<2B params) that detects synthetic images on both clean images as well as images under 6 real-world post-processing transforms (JPEG compression, Gaussian blur, resize, Gaussian noise, color jitter, center crop).

## Project Overview

In recent years the internet has been bombarded with AI-generated content (AIGC) of all kinds, in particular countless synthetic images created with malicious intent to fuel misinformation, impersonation, and fraud at scale. Improvements in advanced AI models have also rendered them capable of generating increasingly convincing synthetic images, and as such detection of these images has become exponentially more difficult, especially when images have undergone transformations like cropping, blurring, compression, colour adjustment, and rescaling. This project builds a dual-branch classifier that combines a frozen CLIP ViT-B/32 vision encoder for high-level semantic features, with a small
trainable noise-residual branch (Sobel/Laplacian high-frequency filters) for low-level artifact detection. It is evaluated for clean accuracy alongside accuracy under all 6 transform families, using the single scoring system `0.5 x AUC_clean + 0.5 x AUC_robust`.

The classifier uses 87,948,305 parameters, but only 99,089 of them are trainable. The CLIP backbone is frozen and only the noise branch and a small fusion head are updated during training. The choice of using a frozen CLIP over fine-tuning a backbone was informed by literature (Ojha et al., 2023) showing that frozen features generalise better across different generators than fine-tuned ones, which is particularly important for the generalisability of the model's detection capabilities.

## Setup & Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Model weights download from Hugging Face on first
run (`openai/clip-vit-base-patch32`); set `HF_HUB_OFFLINE=1` once cached to
skip slow network metadata checks on subsequent runs. Developed and
trained locally on Apple Silicon (M4) - `torch` will use the MPS backend
automatically when available, falling back to CPU otherwise.

## Reproducing Results

Datasets are gitignored (large, rebuildable) - see `data/README.md` for full details on all three datasets. A summary is given as follows:

```
# 1. SID_Set (single-generator training set; where the confound was found)
python scripts/prepare_sid_set.py --target_real 50000 --target_fake 50000

# 2. ArtiFact (25-generator diversity; the primary submitted model's training set)
python scripts/extract_artifact.py --n_per_fake_generator 280 --n_total_real 7000
python scripts/prepare_artifact_split.py

# 3. Validate data integrity (opens + decodes every image, catches truncation)
python scripts/validate_dataset.py
```

Train the primary model from scratch (one epoch at a time; results were reviewed between epochs):

```
python scripts/train.py --data_root data/raw/artifact_full --val_split validation \
  --epochs 1 --batch_size 32 --num_workers 4 \
  --checkpoint_path models/checkpoints/aigc_classifier_artifact.pt \
  --latest_checkpoint_path models/checkpoints/latest_artifact.pt

# subsequent epochs: add --epochs N --resume_from models/checkpoints/latest_artifact.pt
```

Run the robustness evaluation (clean + all 14 image transforms), the per-generator
cross-generator breakdown, and the required prediction CLI:

```
python scripts/evaluate.py --checkpoint_path models/checkpoints/aigc_classifier_artifact.pt \
  --data_root data/raw/artifact_full --n_per_class 1000 --output_dir outputs/eval_final

python scripts/evaluate_crossgen_by_source.py --checkpoint_path models/checkpoints/aigc_classifier_artifact.pt \
  --data_root data/raw/artifact_full --split test --output_dir outputs/eval_final

python scripts/predict.py --input_dir path/to/images --output_json outputs/predictions.json
```

`predict.py`'s output is a JSON file of `{"image_path": ..., "pred": ...}` per image, `pred` a confidence score in [0, 1].

Calibration (temperature scaling applied by `predict.py`/`evaluate.py`):

```
python scripts/calibrate.py --checkpoint_path models/checkpoints/aigc_classifier_artifact.pt \
  --data_root data/raw/artifact_full --val_split validation
```

Error analysis (representative false positives/negatives):

```
python scripts/error_analysis.py
```

## Robustness Evaluation Summary

The model was evaluated on ArtiFact's held-out test split (1000 real + 1000 fake, stratified across all 25 generators / 8 real sources), after 3 epochs of from-scratch
training:

| condition | accuracy | f1 | roc_auc |
|---|---|---|---|
| clean | 0.8515 | 0.8478 | 0.9419 |
| jpeg q=90/70/50/30 | 0.854/0.855/0.853/0.844 | 0.850/0.853/0.853/0.845 | 0.941/0.938/0.938/0.933 |
| gaussian blur σ=0.5/1.0/2.0 | 0.856/0.851/0.844 | 0.852/0.850/0.850 | 0.942/0.941/0.931 |
| resize 0.5x/0.25x | 0.854/0.843 | 0.852/0.851 | 0.941/0.933 |
| gaussian noise σ=0.02/0.05/0.10 | 0.854/0.841/0.832 | 0.851/0.837/0.830 | 0.938/0.931/0.924 |
| color jitter ±20% | 0.853 | 0.849 | 0.941 |
| center crop 80% | 0.843 | 0.849 | 0.932 |

**AUC_clean = 0.9419, AUC_robust = 0.9360, Final score (0.5x clean +
0.5x robust) = 0.9390.** See `outputs/final/robustness_plot.png` for the
visual breakdown per transform family.

### Cross-generator Generalization

The model was also evaluated for each individual generator by pairing each of ArtiFact's 25 generators' fake images against the same pool of real images:

| generator | recall | roc_auc |
|---|---|---|
| generative_inpainting | 0.325 | 0.7575 |
| stylegan1 | 0.425 | 0.8121 |
| ddpm | 0.500 | 0.8462 |
| glide | 0.500 | 0.8712 |
| latent_diffusion | 0.750 | 0.9075 |
| gau_gan | 0.700 | 0.9174 |
| gansformer | 0.725 | 0.9211 |
| denoising_diffusion_gan | 0.800 | 0.9268 |
| lama | 0.825 | 0.9304 |
| mat | 0.850 | 0.9465 |
| taming_transformer | 0.950 | 0.9628 |
| sfhq | 0.925 | 0.9667 |
| vq_diffusion | 0.925 | 0.9736 |
| pro_gan | 0.950 | 0.9761 |
| palette | 0.975 | 0.9765 |
| big_gan | 0.950 | 0.9786 |
| projected_gan | 0.950 | 0.9831 |
| face_synthetics | 1.000 | 0.9947 |
| stable_diffusion | 1.000 | 0.9958 |
| stylegan2 | 1.000 | 0.9973 |
| star_gan | 1.000 | 0.9980 |
| diffusion_gan | 1.000 | 0.9986 |
| cips | 1.000 | 0.9990 |
| stylegan3 | 1.000 | 0.9998 |
| afhq | 1.000 | 0.9999 |

All 25 generators scored above chance, with the weakest (`generative_inpainting`, an inpainting method rather than a full-image generator, which may be harder as it only modifies part of the image) still scoring well above 0.5 AUC.

However, re-testing the ArtiFact-trained model back against SID_Set's original FLUX-generated images gave an AUC of ~0.48 again. Training on ArtiFact's diversity resulted in improved generalisation within the ArtiFact dataset, but it failed to produce unconditional generalisation to a generator family (FLUX) outside of the training distribution, which is somewhat expected.

## Error Analysis Note

On the same held-out test split: 15 false positives (FPR = 3.00%), 156 false negatives (FNR = 31.20%). The model is considerably more likely to miss a fake than to falsely flag a real image, which makes sense given that fake images can come from many different generator families, while real images are, intrisically, real. 10 representative examples of both false positives and false negatives have been provided in `outputs/final/error_analysis_examples/`, with per-image predicted probabilities in `errors.csv` there. This asymmetry, combined with the per-generator table above, suggests the false negatives concentrate in exactly the weaker generators (inpainting/older-GAN outputs) rather than being spread evenly.

False negatives concentrate overwhelmingly in inpainting-based generators; `generative_inpainting` alone accounts for 6 of the 18 lowest-confidence false negatives (the model's single worst generator in the per-generator table above, at 0.7575 AUC); `lama` (another inpainting method) also appears. A visual inspection of the most confidently-wrong example (`generative_inpainting__img000025.jpg`, predicted 97.6% real) shows that it is an ordinary photo of cows at a feeding trough, with no visually obvious synthetic region. Since inpainting only regenerates a small masked region of an otherwise real photo, the large majority of the image's actual pixel content may well be real. As such, a whole-image classifier may have little reason to classify the overall image as fake when most of it has been unmodified. The remaining false negatives cluster in older-generation, full-image generators (`stylegan1`, `pro_gan`, `ddpm`) rather than the modern ones (`stylegan2/3`, `stable_diffusion`) that score near-perfectly. These examples may lack the obvious synthetic tells (warped backgrounds, asymmetric features) that made earlier-generation GAN output easy to spot by eye.

False positives cluster disproportionately in real face photos; `celebahq`/`ffhq` account for roughly half of the 15, including the two most confident mistakes. A visual inspection of the top example (`celebahq__img000391.jpg`, predicted 81.6% fake) shows that it is a heavily-retouched, professionally-lit beauty photograph, with very smooth skin, soft studio lighting and symmetric framing. Its visual "closeness" to the characteristic polished and perfect aesthetic that face-generating models (`stylegan2/3`, `star_gan`, `face_synthetics` - all scoring 0.99+ AUC) produce may have interfered with the classifier's signals. The classifier has thus picked up on a visual similarity between heavily-processed real photography and generative smoothness, which may be a harder and more specific confusion than generic noise.

## Limitations & What I'd Improve Given More Time

- Generalization is bounded by training diversity. The ArtiFact-trained model fails to generalise to FLUX (SID_Set), a generator family outside its training distribution. More generator families in training could test this more rigorously and result in a more highly generalisable model.
- A frequency-domain (FFT/DCT) branch was deliberately deferred. While the current noise-residual branch uses local, spatial high-pass filters (Sobel/Laplacian), a global frequency-domain (FFT/DCT) branch could capture periodic upsampling artifacts that would be missed by current local filters.
- The per-generator numbers within the validation and test sets are based on 40 images per generator, which results in some sampling uncertainty, hence rankings between the closest generators should not be over-interpreted. Increased time and storage space would certainly allow for larger quantities of data for training, validation and testing.

## References

- Ojha, U., Li, Y., & Lee, Y. J. (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models*. CVPR 2023. [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)
- Bird, J. J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access. [arXiv:2303.14126](https://arxiv.org/abs/2303.14126) - source paper for the CIFAKE dataset (Kaggle: [birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)).
- *"SIDA"*, CVPR 2025. [arXiv:2412.04292](https://arxiv.org/abs/2412.04292) - source paper for SID_Set (Hugging Face: [saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)).
- *ArtiFact: A Large-Scale Dataset with Artificial and Factual Images for Generalizable and Robust Synthetic Image Detection*. ICIP 2023. [arXiv:2302.11970](https://arxiv.org/abs/2302.11970) - source paper for ArtiFact (Hugging Face mirror: [bitmind/ArtiFact](https://huggingface.co/datasets/bitmind/ArtiFact), original: [awsaf49/artifact-dataset](https://www.kaggle.com/datasets/awsaf49/artifact-dataset)).

See `data/README.md` for details on how each dataset was acquired and prepared.
