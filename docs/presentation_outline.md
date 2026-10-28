# Six-Slide Presentation Outline

Use this outline for a concise academic presentation. Keep the prototype
notice visible on results and conclusion slides. Treat all metrics as results
on the named derived splits, not as clinical performance.

## Slide 1 — Problem and motivation

- Brain MRI review involves identifying image-level categories and, where
  annotations are available, delineating tumor regions.
- Project question: can a compact CNN classify four labels and a U-Net predict
  binary tumor masks in a reproducible research workflow?
- State the scope: academic image-analysis prototype, not a diagnostic system.
- Visual: one of the existing samples in
  `results/figures/classification_samples_by_class.png`.

## Slide 2 — Objectives and data

- Classification categories: glioma, meningioma, no tumor (`notumor` in data),
  and pituitary; preserve indices 0–3.
- Segmentation: binary BRISC tumor-region masks, derived with grayscale
  threshold 128.
- Derived splits: classification 5,044 / 1,069 / 1,087; segmentation
  3,323 / 712 / 712 (training / validation / testing).
- Explain hash-group leakage audit and exclusion of 46 conflicting-mask pairs.
- Visual: audit split counts or class distributions under `results/figures/`.

## Slide 3 — Methodology

- MRI preprocessing: grayscale, 256 × 256, bilinear resize, fixed `[0, 1]`
  scaling; no statistics fitted on validation or test data.
- Mask preprocessing: preserve categorical meaning, threshold at 128, nearest
  resize; paired train-only geometry transforms.
- Compact CNN trained with cross-entropy; four-level U-Net with one logit map,
  BCE-with-logits plus soft Dice loss.
- Checkpoint selection: validation macro-F1 for classification; validation
  mean per-image Dice for segmentation.
- Visual: preprocessing comparison plus mask/overlay sample.

## Slide 4 — Architecture and workflow

- Show `docs/architecture_and_workflow.md` system architecture and workflow
  diagrams.
- Distinguish the classifier's class probabilities from segmentation
  probabilities and binary mask at threshold 0.5.
- Explain local browser -> Python inference -> saved models -> aligned mask
  and overlay; device used in saved experiments was MPS.
- Visual: architecture diagram; optionally screenshot of the running app.

## Slide 5 — Results

- Classification: epoch 2, validation macro-F1 0.70886; derived-test accuracy
  0.70377, macro-F1 0.67847. Note meningioma recall 0.31365.
- Segmentation: epoch 15, validation mean Dice 0.77197; derived-test mean Dice
  0.75775 and mean IoU 0.66482.
- Report mean-per-image and micro segmentation metrics separately.
- Explicit validation limitation: 27/712 cases had zero Dice; 35/712 below
  0.1 at threshold 0.5.
- Visual: saved classification confusion matrix and 12-case validation review.

## Slide 6 — Conclusion, limitations, and next steps

- The audited, reproducible pipelines and local integrated demo were completed.
- Results are experimental on these derived splits; classifier is a
  two-epoch baseline and segmentation retains case-level failures.
- No external/patient-level validation or clinical validation is available;
  probabilities are not established as calibrated confidence.
- Next research: review failure cases, assess patient-level leakage and
  external generalization, and conduct formal governance/clinical evaluation
  before considering any medical use.
- Close with the notice: **Not for diagnosis or treatment decisions.**

## Artifact references

- Full metrics and methodology: [`methodology_and_results.md`](methodology_and_results.md)
- Architecture diagrams: [`architecture_and_workflow.md`](architecture_and_workflow.md)
- Classification confusion matrix:
  `../results/classification/cnn_baseline_20261012T113201335170Z_8326fdd5/confusion_matrix.png`
- Segmentation validation review:
  `../results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/validation_results_review_12_cases.png`
- Segmentation test examples:
  `../results/segmentation/unet_baseline_20261012T120155132434Z_1cb36ccd/test_visualizations.png`
