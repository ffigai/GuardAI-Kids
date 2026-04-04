# Changelog

All notable changes to GuardAI Kids are documented here.
Base version: [Release GuardAI Kids v2](https://github.com/ffigai/GuardAI-Kids) (commit `3e74621`)

---

## [Unreleased] — Post v2 development

### New: clip_nsfw_violence image analysis pipeline

A new image feature extraction pipeline that adds dedicated specialist classifiers on top of CLIP.

**Feature vector layout (537-dim):**
```
[0:512]    CLIP image embedding (openai/clip-vit-base-patch32)
[512:520]  CLIP harm similarity scores (8 prompts)
[520:525]  Image quality features (brightness, contrast, saturation, colorfulness, detail)
[525]      NSFW score — Marqo/nsfw-image-detection-384 classifier
[526]      Violence score — jaranohaal/vit-base-violence-detection classifier
[527]      has_ocr_text flag (1.0 if meaningful text detected, else 0.0)
[528:536]  OCR harm similarity scores (8, gated — zeroed when flag=0)
[536]      Missing thumbnail flag
```

**Files changed:**
- `src/guardaikids/image_features.py` — added `build_feature_vector_from_image_with_specialists()`, `_load_nsfw_classifier()`, `_load_violence_classifier()`, `_score_binary_classifier()`, `_score_violence_classifier()`
- `src/guardaikids/config.py` — replaced scalar `IMAGE_FEATURE_DIM` with `IMAGE_FEATURE_DIMS` dict keyed by model name; added `default_image_feature_dir_for_model()`; set `IMAGE_ANALYSIS_MODEL = "clip_nsfw_violence"`
- `src/guardaikids/main.py` — renamed `--image-model` to `--image-analysis-model`; added `clip_ocr` and `clip_nsfw_violence` as valid choices
- `src/guardaikids/modeling.py` — `image_feature_dim` now resolved at runtime from `IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL]`
- `src/guardaikids/workflow.py` — `run_training_workflow()` accepts `image_analysis_model` and `image_feature_dir` args
- `src/guardaikids/service.py` — `train_and_save_system()` resolves `image_feature_dir` from `default_image_feature_dir_for_model()`
- `src/guardaikids/data.py` — `load_image_features()` and `prepare_dataset_inputs()` accept optional `image_feature_dim`

**Violence classifier loading:** `jaranohaal/vit-base-violence-detection` lacks `model_type` and `architecture` metadata for both HuggingFace Auto classes and timm hub loading. Fixed by building `vit_base_patch16_224` architecture manually and downloading weights via `hf_hub_download`, loading with `load_state_dict` directly.

---

### Fix: violence score replaced from CLIP zero-shot to trained classifier

The original implementation used a CLIP cosine similarity proxy (`_clip_violence_score`) as a violence score. This was replaced with `jaranohaal/vit-base-violence-detection` (ViT-Base, 98.8% on Kaggle RLVS dataset, binary: violent/non-violent). The CLIP zero-shot approach and its associated `_get_violence_text_feature()` and `VIOLENCE_CLIP_PROMPT` were removed.

---

### Fix: OCR noise filtering in clip_ocr and clip_nsfw_violence

**Problem:** Short OCR hits (e.g. "HD", "4K", "SUBSCRIBE", channel handles) were being passed to CLIP text encoding and producing high cosine similarity scores against harm prompts by coincidence, introducing noise rather than signal.

**Fix:** Added `OCR_MIN_TEXT_LENGTH = 15` in `image_features.py`. `_extract_ocr_text()` now returns an empty string when the combined OCR text is shorter than 15 characters, causing the gating flag to be 0 and OCR scores to be zeroed.

**Impact:** `clip_ocr` macro F1 improved from 0.50 → 0.591 after re-extraction and retraining. Before the fix, `clip_ocr` performed worse than plain `clip`.

---

### Fix: training collapsed to all-negative predictions (class imbalance)

**Problem:** With `BCEWithLogitsLoss` and no class weighting, the model collapsed to predicting all-negative (zero recall on SXL, PH, HH) because the majority of samples are harmless.

**Fix:** Added per-label `pos_weight = (n_neg / n_pos).clip(max=10.0)` computed from the training set and injected into `BCEWithLogitsLoss` before training.

---

### Fix: image-only training hyperparameters

The image-only MLP head is randomly initialised and requires different training settings than fine-tuning a pretrained transformer. `build_training_args()` now accepts a `mode` parameter:

| Setting | image mode | text / multimodal mode |
|---|---|---|
| Learning rate | 1e-3 | 2e-5 |
| Epochs | 15 | 3 |

---

### Fix: best model checkpoint selection

`load_best_model_at_end` was selecting by `eval_loss`, which is dominated by the majority (harmless) class and does not reflect detection quality. Changed to `metric_for_best_model="eval_macro_f1"` with `greater_is_better=True`.

---

### Fix: per-artifact inference thresholds

**Problem:** All image models shared `IMAGE_THRESHOLDS` from `config.py`. Each model outputs probabilities in a different range, so a single threshold set is wrong for all but one model.

**Fix:** 
- `service.py` — added `_thresholds_from_artifact()` which converts per-artifact `f2_thresholds` (warn) and `f1_thresholds` (block) from `metadata.json` into the `{age_group: {label: {warn, block}}}` policy structure
- `explainability.py` — `summarize_risk_categories()` and `should_surface_supporting_cues()` now accept and propagate a `thresholds` parameter, falling back to `get_default_thresholds(mode)` when none is provided
- Each trained artifact now carries its own calibrated thresholds; inference automatically uses them without touching `config.py`

---

### Fix: extract_image_features.py always used config IMAGE_ANALYSIS_MODEL

**Problem:** Running `python scripts/extract_image_features.py --output-dir data/image_features_clip_ocr` still extracted `clip_nsfw_violence` features because `IMAGE_ANALYSIS_MODEL` in config was set to that value, ignoring the intent.

**Fix:** Added `--image-analysis-model` argument to `extract_image_features.py`. The output directory now also defaults automatically to `default_image_feature_dir_for_model(selected_model)` so `--output-dir` is rarely needed.

---

### Fix: modeling.py NameError on IMAGE_FEATURE_DIM

`MultimodalSequenceClassifier.__init__` used `IMAGE_FEATURE_DIM` (deleted scalar) as a default argument. Updated to `IMAGE_FEATURE_DIMS[IMAGE_ANALYSIS_MODEL]`.

---

### Fix: pos_weight device mismatch on GPU

**Problem:** `pos_weight` tensor was created on CPU but the model may be moved to GPU by the Trainer, causing a device mismatch in `BCEWithLogitsLoss` during training.

**Fix:** `modeling.py:304` — added `.to(model.device)` when constructing the `pos_weight` tensor so it always matches the model device.

---

### Re-training plan for fair 7-way comparison

The fixes below affect **all** training modes (text, image, multimodal):
- `pos_weight` class balancing
- `metric_for_best_model="eval_macro_f1"` checkpoint selection

As a result, all previously trained artifacts (text, image-clip, multimodal-clip) were produced without these fixes and are not directly comparable to the newly trained models. The following full re-training sequence was planned to produce a clean, fair comparison across all 7 configurations:

| Run | Command | Status |
|-----|---------|--------|
| text | `train --mode text --artifact-dir artifacts/text_v3` | pending |
| image - clip | `train --mode image --image-analysis-model clip --artifact-dir artifacts/image_clip` | pending |
| image - clip_ocr | `train --mode image --image-analysis-model clip_ocr --artifact-dir artifacts/image_clip_ocr` | ✅ done (fixes already applied) |
| image - clip_nsfw_violence | `train --mode image --image-analysis-model clip_nsfw_violence --artifact-dir artifacts/image_clip_nsfw_violence` | pending (re-extract first) |
| multimodal - clip | `train --mode multimodal --image-analysis-model clip --artifact-dir artifacts/multimodal_clip` | pending |
| multimodal - clip_ocr | `train --mode multimodal --image-analysis-model clip_ocr --artifact-dir artifacts/multimodal_clip_ocr` | pending |
| multimodal - clip_nsfw_violence | `train --mode multimodal --image-analysis-model clip_nsfw_violence --artifact-dir artifacts/multimodal_clip_nsfw_violence` | pending |

Re-extraction of `clip_nsfw_violence` features is required before its image and multimodal runs to apply the OCR length filter fix consistently.

---

### Experiment reporting improvements

`scripts/generate_experiment_report.py` extended with:
- ROC-AUC comparison plot across models
- `load_predictions()` helper
- `plot_roc_auc_comparison()` function

---

### Performance summary (image-only, validation set)

| Model | Mean AUC | Macro F1 |
|---|---|---|
| clip | 0.778 | 0.519 |
| clip_ocr | 0.845 | 0.591 |
| clip_nsfw_violence | 0.846 | 0.594 |
| text (reference) | 0.938 | — |
| multimodal - clip (reference) | 0.935 | — |

Multimodal training with `clip_ocr` and `clip_nsfw_violence` is pending.
