# Changelog

All notable changes to GuardAI Kids are documented here.

---

## [v4] — 2026-04-08

### Removed: age-aware policy layer

The three-age-group decision system (0–4, 5–8, 9–12) producing Allow / Warn / Block has been removed. The system now produces a single binary verdict — **Safe** or **Harmful** — with category tags indicating which harm categories fired (e.g. `Harmful · PH · ADD`).

**Design rationale:** Decision thresholds are now flat F2-optimized values loaded directly from each artifact's `metadata.json` at runtime. This removes hardcoded threshold dicts from `config.py` and ensures each model variant uses calibrated thresholds matched to its own probability distribution.

**Files changed:**
- `src/guardaikids/config.py` — removed `AGE_GROUPS`, `TEXT_THRESHOLDS`, `IMAGE_THRESHOLDS`, `MULTIMODAL_THRESHOLDS`; replaced with `DEFAULT_F2_THRESHOLDS` per mode; `get_default_thresholds()` now returns a flat `{label: threshold}` dict
- `src/guardaikids/policy.py` — fully rewritten; `rule_based_decision()` returns `(decision, categories)`; `get_policy_decision()` returns `{decision, categories, trigger_category, ...}`; `evaluate_policy()` and `evaluate_protection()` return flat metrics (no per-age-group loop)
- `src/guardaikids/explainability.py` — removed `age_group` parameter from `explain_video()`, `summarize_risk_categories()`, `should_surface_supporting_cues()`, `build_explanation_bullets()`; removed `format_age()`
- `src/guardaikids/service.py` — removed `AGE_GROUPS` import and age loop in `analyze_youtube_url()`; `_thresholds_from_artifact()` now returns flat F2 thresholds; result structure flattened to single decision + categories
- `src/guardaikids/workflow.py` — `build_decision_dataframe()` called with explicit `f2_thresholds`; removed `policy_thresholds` (age-grouped) from saved `metadata.json`
- `src/guardaikids/web_interface.py` — removed age selector; replaced 3-card age grid with single verdict banner and category tags; single threshold marker per label bar

---

### Removed: intermediate experiment artifacts

Only the three final artifact configurations are retained in the repository:

- `artifacts/text/`
- `artifacts/image_clip_nsfw_violence/`
- `artifacts/multimodal_clip_nsfw_violence/`

The following were removed: `artifacts/image/`, `artifacts/image_clip_ocr/`, `artifacts/multimodal/`, `artifacts/multimodal_clip_ocr/`, stale root-level report files, and `artifacts/metadata.json`.

---

### Updated: documentation

`README.md`, `PROJECT_SUMMARY.md`, and `SYSTEM_DIAGRAM.md` updated to reflect the current system architecture, the three final artifact configurations, and the 537-dim `clip_nsfw_violence` feature vector layout.

---

## [v3] — 2026-04-04

### New: clip_nsfw_violence image analysis pipeline

A new image feature extraction pipeline adding dedicated specialist classifiers on top of CLIP.

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

### Fix: violence score replaced with trained classifier

Replaced CLIP cosine similarity proxy with `jaranohaal/vit-base-violence-detection` (ViT-Base, binary: violent/non-violent).

### Fix: OCR noise filtering

Added `OCR_MIN_TEXT_LENGTH = 15`. Short OCR hits (e.g. "HD", "4K", channel handles) are discarded before CLIP encoding. `clip_ocr` macro F1 improved from 0.50 → 0.591 after fix.

### Fix: training collapsed to all-negative predictions

Added per-label `pos_weight = (n_neg / n_pos).clip(max=10.0)` in `BCEWithLogitsLoss` to address class imbalance.

### Fix: best model checkpoint selection

Changed `metric_for_best_model` from `eval_loss` to `eval_macro_f1`.

### Fix: per-artifact inference thresholds

Each artifact now carries calibrated F2/F1 thresholds in `metadata.json`. Inference loads these automatically instead of using shared config thresholds.

### Fix: image-only training hyperparameters

Image-only mode now trains with `lr=1e-3` and `epochs=15` instead of the transformer fine-tuning defaults.

### Performance summary (validation set, n = 2,767)

| Model | Mean AUC | Macro F1 |
|---|---|---|
| text only | 0.939 | 0.783 |
| image — clip | 0.843 | 0.588 |
| image — clip_ocr | 0.845 | 0.591 |
| image — clip_nsfw_violence | 0.846 | 0.586 |
| multimodal — clip | 0.937 | 0.779 |
| multimodal — clip_ocr | 0.937 | 0.781 |
| multimodal — clip_nsfw_violence | 0.935 | 0.782 |
