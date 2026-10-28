# MRI Brain Tumor Detection Using Image Processing and Deep Learning

An academic research prototype for four-class MRI image classification and
binary tumor-region segmentation. It includes leakage-audited data manifests,
preprocessing, PyTorch CNN and U-Net models, evaluation reports, visual
artifacts, and a local browser interface that runs the saved models.

> **Research use only.** This project is not clinically validated and is not
> intended for diagnosis, treatment decisions, or patient care.

## Objectives and features

- Classify one MRI as **Glioma**, **Meningioma**, **No Tumor**, or **Pituitary**.
- Predict a binary tumor-region mask and display it aligned to the original
  image. The configured probability threshold is `0.5`.
- Preserve source images and masks; preprocessing reads them without modifying
  source files.
- Keep classification and segmentation manifests, labels, preprocessing, and
  evaluation separate.
- Exclude conflicting MRI/mask pairs and keep exact duplicate MRI groups
  within one derived split.
- Run image inference locally with Apple MPS where supported, otherwise CPU.

The application returns model outputs, not medical diagnoses or calibrated
clinical confidence.

## Results at a glance

These are saved experimental results for the specific derived splits and
checkpoints documented in [the methodology and results report](docs/methodology_and_results.md).

| Task | Selected checkpoint | Validation result | Derived test result |
|---|---|---:|---:|
| Classification | `results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/best_model.pt` (epoch 2, selected by validation macro-F1) | Macro-F1 **0.70886**; loss **0.74713** | Accuracy **0.70377**; macro-F1 **0.67847**; weighted-F1 **0.67943** |
| Binary segmentation | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/best_model.pt` (epoch 15, selected by validation mean Dice) | Mean Dice **0.77197**; mean IoU **0.68485** | Mean Dice **0.75775**; mean IoU **0.66482** |

The classifier run was a two-epoch pipeline-validation baseline; its result
should not be interpreted as a fully tuned classifier. Segmentation validation
review identified 27/712 images with zero Dice at the fixed 0.5 threshold.
Neither task is externally validated or clinically evaluated.

## Model architecture

- **Classifier:** scratch-initialized compact CNN with three convolutional
  blocks (32/64/128 channels), batch normalization, ReLU, max pooling, adaptive
  average pooling, dropout, and a four-logit head. Cross-entropy and AdamW are
  used. The saved selected checkpoint uses dropout `0.25`.
- **Segmenter:** scratch-initialized four-level U-Net with skip connections,
  base width 16 and a single-channel logit output. Training combines
  BCE-with-logits and soft Dice loss. The probability threshold is `0.5`;
  sigmoid is used for probability inference, not on logits before the loss.
- No pretrained model weights are downloaded.

See [system architecture and workflow diagrams](docs/architecture_and_workflow.md).

## Dataset organization and derived splits

Place local datasets at:

```text
datasets/
├── classification/Brain Tumor MRI/
└── segmentation/brisc2025/
```

The saved leakage audit and manifests are under `results/dataset_inspection/`
and `results/data_manifests/`. The selected audit is
`leakage_audit_20261012T110042084868Z_4edcbdab`.

| Task | Training | Validation | Testing | Additional exclusions |
|---|---:|---:|---:|---|
| Classification images | 5,044 | 1,069 | 1,087 | No conflicting-label groups flagged |
| BRISC segmentation pairs | 3,323 | 712 | 712 | 46 conflicting-mask rows excluded |

Classification labels retain their saved encoding:
`glioma=0`, `meningioma=1`, `notumor=2`, `pituitary=3`. The UI presents
`notumor` as **No Tumor**. The BRISC segmentation manifest includes
image-mask paths; class-only classification images are never treated as mask
pairs. The 46 flagged rows refer to identical MRI content paired with
conflicting decoded masks.

Exact SHA-256 duplicate groups are indivisible across derived partitions.
There is no automatic fallback to official split folders. Exact hashes do not
prove patient-level independence and cannot identify re-encoded visual copies.
See the [saved audit report](results/dataset_inspection/leakage_audit_20261012T110042084868Z_4edcbdab.md).

### Inspect and audit

From the project root:

```bash
PYTHONPATH=src python scripts/inspect_datasets.py
PYTHONPATH=src python scripts/audit_dataset_leakage.py
PYTHONPATH=src python scripts/visualize_datasets.py --samples-per-class 1 --segmentation-examples 3
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/inspect_datasets.py
python scripts/audit_dataset_leakage.py
python scripts/visualize_datasets.py --samples-per-class 1 --segmentation-examples 3
```

Audit and visualization scripts create uniquely named or descriptive artifacts;
review them before training. The visualization script samples derived
classification training data and excludes flagged segmentation pairs.

## Methodology and evaluation

MRI images are converted to grayscale and resized to 256 × 256 with bilinear
resampling; uint8 intensities are scaled deterministically to `[0, 1]`.
Normalization does not estimate statistics from validation or testing data.
Masks are not normalized: decoded grayscale mask pixels at or above 128 are
tumor, and mask resizing uses nearest-neighbor interpolation. Training-only
random horizontal flips and rotations are applied identically to an MRI and
its paired mask. Validation and test preprocessing do not augment.

The classification CNN is selected using validation macro-F1. Classification
evaluation reports accuracy and per-class precision, recall and F1 as well as
macro/weighted F1 and a confusion matrix. Segmentation checkpoint selection
uses validation mean per-image Dice; testing is evaluated separately after
selection. Segmentation reports per-image-averaged and pixel-aggregated
(micro) Dice/IoU/precision/recall. The report defines the empty-mask scoring
convention. Full detail, formulas, saved metric values, and limitations are
in [Methodology and results](docs/methodology_and_results.md).

## Installation

Python 3.10 or newer is required. Create and activate an environment:

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch/torchvision using the command from the official selector for
the operating system and hardware. Select a macOS build that supports MPS when
available; choose CUDA only for a compatible NVIDIA GPU. CPU works as fallback.
Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

Training and preprocessing settings are recorded in `configs/` and in each
experiment's `training_config.json`. The current dataset locations can be
overridden with `MRI_CLASSIFICATION_DATASET_DIR` and
`MRI_SEGMENTATION_DATASET_DIR`.

## Training and evaluation commands

The commands below create new run directories; they do not overwrite the
historical checkpoints. Training needs the datasets and derived manifest.
Choose an available device (`mps`, `cuda`, or `cpu`); `auto` follows the
project's device selector.

### Classification

macOS / Linux:

```bash
PYTHONPATH=src python scripts/train_classifier.py \
  --manifest results/data_manifests/leakage_audit_20261012T110042084868Z_4edcbdab_classification.csv \
  --epochs 2 --batch-size 32 --seed 20261012 --device auto
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/train_classifier.py `
  --manifest results/data_manifests/leakage_audit_20261012T110042084868Z_4edcbdab_classification.csv `
  --epochs 2 --batch-size 32 --seed 20261012 --device auto
```

After training, evaluate a selected checkpoint once:

```bash
PYTHONPATH=src python scripts/evaluate_classifier.py --checkpoint path/to/run/best_model.pt
```

Use `--help` on the scripts for supported options. The documented historical
classifier used the derived manifest and ran two epochs; the default
`configs/classification.json` allows up to 30 epochs.

### Segmentation

macOS / Linux:

```bash
PYTHONPATH=src python scripts/train_segmenter.py \
  --manifest results/data_manifests/leakage_audit_20261012T110042084868Z_4edcbdab_segmentation.csv \
  --batch-size 2 --epochs 15 --patience 7 --seed 20261012 \
  --device mps --num-workers 0
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/train_segmenter.py `
  --manifest results/data_manifests/leakage_audit_20261012T110042084868Z_4edcbdab_segmentation.csv `
  --batch-size 2 --epochs 15 --patience 7 --seed 20261012 `
  --device auto --num-workers 0
```

After validation-based checkpoint selection, evaluate the derived test split
once using that run's checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate_segmenter.py \
  --checkpoint path/to/run/best_model.pt --device auto
```

Never select a checkpoint or tune a threshold using test metrics. Do not pass
the segmenter's `--max-train-batches` or `--max-validation-batches` options for
a full-data experiment.

## Single-image inference

The callable API is `mri_tumor.inference.predict_mri(image_path,
classification_checkpoint, segmentation_checkpoint)`. It returns the class,
class probabilities, segmentation probability map, fixed-threshold mask,
original image, nearest-neighbor-aligned mask, and overlay. Each model loads
its own checkpoint-saved preprocessing settings.

```python
from pathlib import Path
from mri_tumor.inference import predict_mri

result = predict_mri(
    Path("path/to/mri.png"),
    Path("results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/best_model.pt"),
    Path("results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/best_model.pt"),
)
print(result["predicted_class"], result["class_probabilities"])
```

## Launch the local web application

The static browser UI uses the standard-library server in `scripts/serve_app.py`;
it does not require a separate frontend build or web framework. It uses the
selected existing checkpoints by default, accepts PNG/JPEG/BMP/TIFF/WebP up to
10 MB, and binds to localhost.

macOS:

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_app.py
```

Windows PowerShell (with the project environment activated):

```powershell
$env:PYTHONPATH = "src"
python scripts/serve_app.py
```

Open <http://127.0.0.1:8000>. Override checkpoint locations with
`--classification-checkpoint` and `--segmentation-checkpoint`, or environment
variables `MRI_CLASSIFICATION_CHECKPOINT` and `MRI_SEGMENTATION_CHECKPOINT`.
The server refuses to start if a required checkpoint is missing.

## Saved visualizations and reports

- Classification class samples, preprocessing, and split distributions:
  `results/figures/`.
- Classification confusion matrix and per-class metrics:
  `results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/`.
- Segmentation learning curves, selected test comparisons, and per-image
  metrics: `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/`.
- Twelve-case **validation-only** review (MRI, ground truth, prediction,
  overlay): [validation review image](results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/validation_results_review_12_cases.png).
- Six-slide academic presentation outline:
  [presentation outline](docs/presentation_outline.md).

## Tests

macOS / Linux:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Limitations, ethics, and appropriate use

- The classification run is a short two-epoch baseline and is not a tuned or
  externally validated classifier. Its held-out derived test results show
  notable class-specific errors, particularly for meningioma.
- Segmentation review found 27 validation images with zero Dice at the fixed
  0.5 threshold and 35 images with Dice below 0.1. Aggregate overlap metrics
  do not remove these individual failures.
- The source data and derived image-level groups do not establish
  patient-level independence. Hash auditing detects exact byte duplicates, not
  re-encoded copies or subjects shared across partitions.
- Only the supplied dataset-derived splits have been evaluated. External,
  multi-site, scanner, demographic, and prospective validation are absent.
- A predicted class is not a diagnosis; probability outputs are not established
  as calibrated confidence. Segmentation masks can be incorrect or incomplete.
- MRI data can be sensitive. This local demo has no authentication and should
  not be exposed to a public network or used with identifiable data without
  appropriate authorization and safeguards.

This software must not be used to make medical decisions. Human review,
appropriate governance, external validation, and formal clinical evaluation
would be necessary before any medical application could be considered.
