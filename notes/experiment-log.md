# Experiment log

One entry per run, newest at the bottom. Template:

```
## YYYY-MM-DD — <quick|full> run
- Commit: <short sha>
- Changed: <what's different from the previous run>
- Clean accuracy: CNN … / HOG+SVM …
- Mean corrupted accuracy: CNN clean-trained … / CNN augmented … / HOG+SVM …
- Notes: <anything surprising>
```

## 2026-11-11 — quick run #1
- Commit: 1cf43d9 (initial scaffold)
- Changed: first run on the real GTSRB data, CNN 2 epochs
- Clean accuracy: CNN clean-trained 15.6% / CNN augmented 24.8% / HOG+SVM 86.2%
- Notes: CNN `val_accuracy` was 0.005, below chance (1/43 ≈ 0.023). Cause: data
  sorted by class plus Keras `validation_split` taking the last 10%, so the last
  classes were never trained on. → [[decisions]] "Shuffle before validation split".

## 2026-11-11 — quick run #2
- Commit: 9242977 (shuffle fix)
- Changed: training data shuffled once with a fixed seed
- Clean accuracy: CNN clean-trained 10.5% / CNN augmented 24.1% / HOG+SVM 86.2%
- Notes: `val_accuracy` now 0.13–0.24 and tracks training. Accuracy is still low
  because 2 epochs is far too few; training accuracy was still rising fast.

## 2026-11-11 — quick run #3
- Commit: efbb114 (quick run uses 10 epochs)
- Changed: `quick.cnn_epochs` 2 → 10
- Clean accuracy: CNN clean-trained 4.8% / CNN augmented 42.2% / HOG+SVM 86.2%
- Notes: training accuracy 98% but validation ~5%. Not normal overfitting:
  validation never rose first. Traced to BatchNorm running statistics lagging
  behind (momentum 0.99) in a short run. → [[decisions]] "BatchNorm momentum 0.9".

## 2026-11-11 — quick run #4
- Commit: e7fa3b4 (BatchNorm momentum 0.9)
- Changed: `BatchNormalization(momentum=0.9)` in both blocks
- Clean accuracy: CNN clean-trained 87.2% / CNN augmented 86.5% / HOG+SVM 86.2%
- Notes: validation tracks training (97.5% at epoch 10). Fix confirmed, ready for
  the full run.

## 2026-11-11 — full run
- Commit: e7fa3b4
- Changed: full dataset (26,640 train / 12,630 test), 3 seeds, severities 1–5
- Clean accuracy: CNN clean-trained 95.1% / CNN augmented 95.7% / HOG+SVM 90.0%
- Mean corrupted accuracy: CNN clean-trained 68.1% / CNN augmented 78.7% / HOG+SVM 60.4%
- Notes: ran ~1 hour on a laptop CPU. HOG+SVM gives identical results for every
  seed (deterministic). Interpretation in [[analysis]].
