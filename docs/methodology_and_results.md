# Methodology and Experimental Results

This report summarizes saved artifacts from the repository's derived
classification and BRISC 2025 segmentation experiments. It describes the
specific experiments and splits; it does not assert clinical validity or
superiority over another method.

## Dataset audit and splits

The selected leakage audit was generated with seed `20261012`. Source files
were hashed by SHA-256; exact duplicate image groups were assigned as
indivisible units to derived partitions. The derived manifests preserve image
labels and, for segmentation, manifest-linked image-mask pairs.

| Dataset/task | Training | Validation | Testing | Audit notes |
|---|---:|---:|---:|---|
| Brain Tumor MRI classification | 5,044 | 1,069 | 1,087 | 7,200 source images; 7,013 distinct hashes; 153 exact-duplicate groups (340 files); no conflicting-label groups |
| BRISC 2025 segmentation | 3,323 | 712 | 712 | 4,793 linked pairs in audit; 46 flagged conflict rows excluded; 21 duplicate groups with different decoded masks; zero exact image-hash overlap between eligible derived partitions |

The classification labels and indices are `glioma=0`, `meningioma=1`,
`notumor=2`, and `pituitary=3`. For segmentation, the 46 rows with identical
MRI content associated with different masks were excluded. Remaining
duplicate-image groups do not cross the derived partitions. Official split
assignments are retained as source metadata, not used as a fallback split.
Exact hashing cannot identify visual copies re-encoded with different bytes,
nor does it prove subject-level independence.

Audit report:
[`results/dataset_inspection/leakage_audit_20261012T110042084868Z_4edcbdab.md`](../results/dataset_inspection/leakage_audit_20261012T110042084868Z_4edcbdab.md)

## Preprocessing

The recorded preprocessing is 256 × 256 grayscale MRI input, bilinear image
resampling, and deterministic uint8 scaling to `[0, 1]`. No
validation/test-derived normalization statistics are used. Original files are
read only. BRISC masks preserve categorical semantics: grayscale mask values
are thresholded at 128 (values `>=128` mean tumor), not intensity-normalized,
and resized with nearest-neighbor interpolation. The same random horizontal
flip and rotation is applied to paired MRI and mask examples in training only;
validation and test data use deterministic preprocessing.

## Classification method

The classifier is a scratch-initialized compact CNN: convolutional blocks with
32, 64, and 128 channels, batch normalization, ReLU, max pooling, adaptive
average pooling, dropout, and a four-class linear output. It uses cross-entropy
loss and AdamW. The checkpoint was selected at epoch 2 by validation macro-F1.
The recorded run used MPS, seed `20261012`, batch size 32, learning rate
`0.001`, dropout `0.25`, and the full derived classification partitions.
Although the config permits 30 epochs, the saved experiment completed two
epochs as a pipeline-validation baseline.

### Classification validation

| Selected epoch | Validation loss | Validation macro-F1 | Selection rule |
|---:|---:|---:|---|
| 2 | 0.74713 | 0.70886 | Highest recorded validation macro-F1 |

Validation accuracy was not retained in the saved run artifacts and is not
reported here.

### Classification derived-test results

Evaluated once on 1,087 derived test images using the validation-selected
epoch-2 checkpoint.

| Metric | Saved value |
|---|---:|
| Accuracy | 0.70377 |
| Macro-F1 | 0.67847 |
| Weighted-F1 | 0.67943 |
| Test loss | 0.76988 |

Per-class test metrics:

| True class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| glioma | 0.62918 | 0.77239 | 0.69347 | 268 |
| meningioma | 0.73913 | 0.31365 | 0.44041 | 271 |
| notumor | 0.72414 | 0.97849 | 0.83232 | 279 |
| pituitary | 0.75188 | 0.74349 | 0.74766 | 269 |

Confusion matrix, rows = true labels and columns = predicted labels:

| True \ Predicted | glioma | meningioma | notumor | pituitary |
|---|---:|---:|---:|---:|
| glioma | 207 | 22 | 16 | 23 |
| meningioma | 94 | 85 | 52 | 40 |
| notumor | 3 | 0 | 273 | 3 |
| pituitary | 25 | 8 | 36 | 200 |

The saved rendered confusion matrix is
[`results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/confusion_matrix.png`](../results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/confusion_matrix.png);
raw values are in the same run's `confusion_matrix.csv`. The low meningioma
recall is a salient limitation of this short baseline.

## Segmentation method

The segmentation model is a four-level U-Net with base width 16 and
single-channel tumor-region logits. It was trained from scratch with
BCE-with-logits plus soft Dice loss, batch size 2, learning rate `0.001`, seed
`20261012`, and MPS. All 3,323 eligible training pairs and 712 eligible
validation pairs were used per epoch. The run completed its configured
15-epoch limit; early stopping did not occur. The epoch-15 checkpoint was
selected by validation mean per-image Dice. The 712 test pairs were reserved
for evaluation after selection.

### Segmentation validation

| Metric | Validation result |
|---|---:|
| Mean per-image Dice | 0.77197 |
| Mean per-image IoU | 0.68485 |
| Dice Q1 / median / Q3 | 0.69962 / 0.88434 / 0.94524 |
| IoU Q1 / median / Q3 | 0.53801 / 0.79266 / 0.89616 |
| Per-image Dice exactly zero | 27 / 712 |
| Per-image Dice below 0.1 | 35 / 712 |
| Empty ground-truth masks | 0 |
| Empty predicted masks at threshold 0.5 | 11 |

Failure is explicitly defined as **per-image Dice exactly zero**, meaning no
tumor pixels overlap at the fixed probability threshold `0.5`. The
representative 12-case validation review checked MRI/mask path association,
manifest order, dimensions, and availability of prediction maps. It includes
strong predictions, weak predictions, small foreground regions, and zero-Dice
failures:

![Twelve validation examples showing MRI, ground truth, prediction, and overlay](../results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/validation_results_review_12_cases.png)

This is a validation review only. Saved validation review metadata records no
missing images/masks, missing predictions, non-finite input images, mask
dimension mismatch, nonbinary preprocessed masks, or empty ground-truth masks.
It records 11 empty predictions and 27 zero-Dice cases; these cases should not
be obscured by the aggregate score.

### Segmentation derived-test results

The validation-selected epoch-15 checkpoint was evaluated once on all 712
derived test pairs. Mean scores average per-image metrics; micro scores
aggregate tumor-pixel counts over the full partition.

| Metric | Mean per image | Micro / pixel aggregate |
|---|---:|---:|
| Dice | 0.75775 | 0.74861 |
| IoU | 0.66482 | 0.59822 |
| Precision | 0.74717 | 0.69891 |
| Recall | 0.83242 | 0.80592 |

There were zero empty ground-truth test masks. The evaluator's empty-mask
convention is: if both prediction and target are empty, Dice/IoU/precision/
recall equal 1; if only the target is empty, all four equal 0. From the saved
712-row per-image CSV, test Dice quartiles are Q1 `0.67727`, median `0.85985`,
and Q3 `0.93728`; 30 images have exactly zero Dice and 37 have Dice below 0.1.
These descriptive counts were computed from the already saved CSV; test
inference was not repeated for this report.

The saved test comparison figure contains MRI, ground truth, prediction, and
overlay:

![Selected derived-test MRI, ground-truth mask, prediction, and overlay examples](../results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/test_visualizations.png)

## Checkpoint and artifact provenance

| Artifact | Project-relative path |
|---|---|
| Classification checkpoint | `results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/best_model.pt` |
| Classification test metrics | `results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/test_metrics.json` |
| Classification confusion matrix | `results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/confusion_matrix.csv` and `.png` |
| Segmentation selected checkpoint | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/best_model.pt` |
| Segmentation training history | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/epoch_metrics.csv` |
| Segmentation validation analysis | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/validation_analysis.json` |
| Segmentation validation per-image results | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/validation_per_image_metrics.csv` |
| Segmentation test metrics | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/test_metrics.json` |
| Segmentation test per-image results | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/per_image_metrics.csv` |
| Segmentation test figure | `results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/test_visualizations.png` |

The recorded validation probability archive contains 712 maps of shape
`(712, 1, 256, 256)`. Its manifest order and exact reproduction from the saved
epoch-15 checkpoint were verified (maximum absolute difference `0.0`). Saved
per-epoch probability summary fields in the original MPS epoch CSV were later
identified as invalid due to a device transfer/dtype conversion issue; those
historical fields are not used for any result reported here. Training was not
repeated to rewrite the run history.

## Metrics and interpretation

For a binary mask, Dice is `2TP / (2TP + FP + FN)`, IoU is
`TP / (TP + FP + FN)`, precision is `TP / (TP + FP)`, and recall is
`TP / (TP + FN)`. A mean-per-image score computes the metric for each image and
then averages images. A micro score sums pixel-level TP/FP/FN first and
computes the metric once. These aggregations answer different questions and
are reported separately.

The confusion matrix and segmentation examples are descriptive outcomes on
the recorded partitions. They do not establish clinical utility, calibrated
probabilities, generalization to other populations/scanners, or superiority
over other methods. No unrecorded validation result, external baseline, or
clinical claim is inferred.

## Limitations and ethics

1. The classification model was trained for only two epochs; the observed
   meningioma test recall is `0.31365`.
2. The segmentation model has 27 validation and 30 test examples with exactly
   zero per-image Dice at the configured threshold, plus additional low-overlap
   cases.
3. Hash-based split grouping handles exact-byte duplicate leakage, not
   re-encoded images or patient-level dependencies.
4. There is no external, prospective, multi-site, or clinical validation.
5. Model outputs may be wrong; class probabilities are not established as
   calibrated confidence. No user should use them for diagnosis or treatment.
6. MRI images may contain sensitive information. The demonstration server is
   local-only by default and has no authentication; network exposure requires
   security review and appropriate data governance.

Any future medical use would require appropriately governed data, patient-level
split design, external validation, calibration and robustness studies, expert
review, and formal regulatory/clinical evaluation. This report is an academic
summary, not clinical evidence.
