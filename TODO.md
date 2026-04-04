# GuardAI Kids — Training TODO

Progress tracker for the full 7-model comparison run.

---

## In Progress

- [ ] Re-extract clip_nsfw_violence features (OCR length filter fix applied)

## Pending

- [ ] Train image-only model with clip_nsfw_violence
  - `python -m guardaikids.main train --mode image --image-analysis-model clip_nsfw_violence --artifact-dir artifacts/image_clip_nsfw_violence`

- [ ] Train multimodal model with clip_nsfw_violence
  - `python -m guardaikids.main train --mode multimodal --image-analysis-model clip_nsfw_violence --artifact-dir artifacts/multimodal_clip_nsfw_violence`

- [ ] Train multimodal model with clip_ocr
  - `python -m guardaikids.main train --mode multimodal --image-analysis-model clip_ocr --artifact-dir artifacts/multimodal_clip_ocr`

- [ ] Save results and generate graphs for clip_nsfw_violence and clip_ocr multimodal runs

- [ ] Take backup of trained models and results

- [ ] Train multimodal model with clip (baseline)
  - `python -m guardaikids.main train --mode multimodal --image-analysis-model clip --artifact-dir artifacts/multimodal_clip`

- [ ] Train text-only baseline model
  - `python -m guardaikids.main train --mode text --artifact-dir artifacts/text_v3`

- [ ] Generate full 7-way comparison graphs and results

## Completed

- [x] Fix OCR noise filter (OCR_MIN_TEXT_LENGTH = 15)
- [x] Fix class imbalance (pos_weight in BCEWithLogitsLoss)
- [x] Fix best model checkpoint selection (metric_for_best_model = eval_macro_f1)
- [x] Fix pos_weight device mismatch on GPU
- [x] Fix extract_image_features.py --image-analysis-model flag
- [x] Fix per-artifact inference thresholds
- [x] Train image-only clip (baseline)
- [x] Train image-only clip_ocr
- [x] Train image-only clip_nsfw_violence (pre OCR fix — to be redone)
