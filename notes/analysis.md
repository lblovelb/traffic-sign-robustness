# Analysis of the full run

Run: 2026-11-11, commit e7fa3b4, 3 seeds. Numbers from `results/summary.md` and
`notebooks/analysis.ipynb`. See [[experiment-log]] for how we got here.

## Baseline (clean test images)
- CNN 95.1% ± 0.4, CNN augmented 95.7% ± 1.1, HOG+SVM 90.0%.
- The CNN is better than the classical baseline. Augmentation costs nothing on
  clean images: the difference is within the seed spread.

## Shift (clean-trained models on corrupted images)
- Average over all corruptions and severities: CNN 68.1%, HOG+SVM 60.4%.
- At severity 5 the clean-trained CNN loses about half or more of its accuracy
  on every corruption (relative drop 47–66%).
- HOG+SVM is almost unaffected by darkness (1% relative drop) but collapses
  under noise (85%) and rain (83%). HOG block-normalises gradients, so a global
  change in brightness barely changes its features.
- HOG+SVM beats the clean-trained CNN in darkness and fog from severity 2 upwards.

## Mitigation (augmentation with noise, darkness, fog)
- Seen corruptions: 69.4% → 85.1%. The gains on darkness (+17 pp), fog (+17 pp)
  and noise (+13 pp) are all well above the seed spread.
- Unseen corruptions: 66.1% → 69.2%. Blur +4.8 pp is borderline (seed std
  2.6 pp). Rain +1.3 pp is within the noise.
- Conclusion: augmentation makes the model robust to what it has seen, and
  barely to anything else.

## Feasibility
- Latency: CNN 0.29 ms per image, HOG+SVM 0.91 ms per image on a laptop CPU.
  Both are far below a ~33 ms per-frame budget at 30 fps, but a car runs many
  models at once on different hardware, so this would need measuring there.
- Size: CNN 2.4 MB, HOG+SVM 0.6 MB.

## Limitations
- Synthetic corruptions, not real weather.
- Cropped signs only (no detection step).
- Validation (~99%) is optimistic because GTSRB has tracks of near-identical
  frames; the test set is unaffected.
- One dataset, one resolution, 3 seeds.

## Ideas for next steps
- Evaluate on real adverse-weather images.
- Train HOG+SVM on augmented data too, for a complete 2×2 comparison.
- Combine the models (they fail in different conditions), or normalise
  brightness before the CNN.
- Split validation by track to get an honest validation number.
