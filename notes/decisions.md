# Decisions

Short records of choices and why they were made. Newest at the bottom.

## 2026-11-11 — Dataset: GTSRB
Public, well-known, 43 classes, small images (fast on a laptop CPU). It is
cropped signs only, so this project studies the *classification* step, not
detection. Stated as a limitation.

## 2026-11-11 — Baseline: HOG + linear SVM
The "existing solution" to compare against. Classical traffic-sign pipelines
used hand-crafted gradient features; this shows whether the CNN's advantage
holds when images degrade.

## 2026-11-11 — Synthetic corruptions, severities 1–5
Real bad-weather data with labels is scarce, so corruptions are simulated (in
the spirit of ImageNet-C). Each is seeded so every model sees the same images.

## 2026-11-11 — Seen vs unseen corruptions
Augmentation uses noise, darkness and fog only. Blur and rain are held out to
test whether robustness *generalises* to conditions never trained on — the
question that matters when a vehicle meets something new.

## 2026-11-11 — Three seeds, report mean ± std
One run can be lucky. Same practice as Love's bachelor thesis.
