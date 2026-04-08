# GuardAI Kids Project Summary

## 1. Problem Definition

GuardAI Kids is a YouTube content safety system for children. It analyzes a YouTube video and classifies it as **Safe** or **Harmful**, surfacing which harm categories were detected and providing supporting evidence from text and thumbnail signals.

The system supports three input configurations:

- text-only: title, description, and transcript
- image-only: thumbnail-derived feature vector
- multimodal: text plus thumbnail-derived feature vector

The system produces two levels of output:

- model-level multi-label probabilities for four harm categories:
  - `ADD`: addictive or substance-related content
  - `SXL`: sexual or explicit material
  - `PH`: physical harm or self-injury references
  - `HH`: hate speech or harassment-related content
- decision-level binary verdict:
  - `Safe`
  - `Harmful` (with category tags indicating which labels fired)

## 2. Dataset Description

The training data is read from two Excel workbooks: `Harmful.xlsx` and `Harmless.xlsx`. The code requires the following source columns:

- `video_id`
- `harm_cat`
- `title`
- `description`
- `transcript`

`harm_cat` is the label field. It may contain comma-separated category codes. Labels are filtered against the fixed target set `["ADD", "SXL", "PH", "HH"]`.

`video_id` is used as the key that links a sample to its image features. Image features are stored as:

- `data/image_features/{video_id}.npy`

For image-focused experiments, the pipeline also supports an optional `thumbnail_harm_cat` column. In `image` mode, if this column is present and non-empty, it overrides `harm_cat` for that row. The code records whether labels came from the original video-level field or the thumbnail-specific field using a `label_source` column.

Missing thumbnails are handled indirectly through missing image-feature files. If `data/image_features/{video_id}.npy` does not exist, the system returns a synthetic missing vector:

- all zeros
- last element set to `1.0` as a missing flag

The same fallback is used when the `.npy` file exists but has the wrong shape or contains non-finite values.

## 3. Data Pipeline

### Text processing

The data pipeline concatenates text fields into one string:

- `title`
- `description`
- `transcript`

This concatenated string is stored as `text`.

Tokenization is performed with `AutoTokenizer.from_pretrained("distilroberta-base")`. For `text` and `multimodal` modes, tokenization uses:

- truncation enabled
- max length `512`
- padding to `max_length`

### Image feature loading

Image features are loaded from `.npy` files by `load_image_features()`. Each sample is mapped from `video_id` to a vector in `data/image_features/`.

The feature vector for the primary `clip_nsfw_violence` backend:

- `512` CLIP embedding dimensions
- `8` prompt-similarity dimensions
- `5` image-quality dimensions
- `1` NSFW score (`Marqo/nsfw-image-detection-384`)
- `1` violence score (`jaranohaal/vit-base-violence-detection`)
- `1` has_ocr_text flag
- `8` OCR harm-similarity scores
- `1` missing flag

Total: **537 dims**

### Dataset formatting by mode

The pipeline uses Hugging Face `Dataset` objects, not a custom PyTorch dataset class.

Mode-specific formatting:

- `text`
  - tokenized text
  - model columns: `input_ids`, `attention_mask`, `labels`
- `image`
  - no tokenization
  - model columns: `image_features`, `labels`
- `multimodal`
  - tokenized text plus image features
  - model columns: `input_ids`, `attention_mask`, `image_features`, `labels`

All formatted datasets are converted to torch tensors through `set_format(type="torch", columns=...)`.

## 4. Model Architecture

### Text branch

The text branch uses a transformer encoder loaded with:

- `AutoModel.from_pretrained("distilroberta-base")`

The text representation is extracted from:

- `outputs.last_hidden_state[:, 0, :]`

This is a CLS-style pooled representation from the first token position.

The text-only classifier head is:

- `Linear(text_hidden_size -> num_labels)`

### Image branch

The image branch does not train an end-to-end vision encoder. It consumes precomputed image features derived from CLIP-based thumbnail processing.

The image-only classifier is an MLP:

- `LayerNorm(image_feature_dim)`
- `Linear(image_feature_dim -> image_hidden_dim)`
- `ReLU`
- `Dropout(0.35)`
- `Linear(image_hidden_dim -> num_labels)`

`image_hidden_dim` is computed as:

- `max(128, min(512, image_feature_dim // 2))`

### Fusion

In `multimodal` mode, the model concatenates:

- text embedding (768-dim)
- image feature vector (537-dim)

The fusion head is:

- `Linear(768 + 537 -> 768)`
- `ReLU`
- `Dropout(0.3)`
- `Linear(768 -> num_labels)`

### Output

The task is multi-label classification over four categories. The model returns logits with optional loss.

Loss function: `BCEWithLogitsLoss`. Sigmoid is applied outside the loss path for evaluation and prediction.

## 5. Modes of Operation

### Text-only

Inputs: `input_ids`, `attention_mask`

Use case: baseline classification from title, description, and transcript.

### Image-only

Inputs: `image_features`

Use case: thumbnail-based classification without text.

### Multimodal

Inputs: `input_ids`, `attention_mask`, `image_features`

Use case: combined text and thumbnail analysis.

## 6. Training Setup

Training uses Hugging Face `Trainer`. Key settings:

- model: `distilroberta-base`
- learning rate: `2e-5`
- train/eval batch size: `16`
- epochs: `3`
- weight decay: `0.01`
- `load_best_model_at_end=True`

Loss: multi-label `BCEWithLogitsLoss`

## 7. Evaluation Methodology

### Model-level evaluation

Validation outputs are collected with `trainer.predict()`.

Metrics:

- macro F1, micro F1
- per-label ROC-AUC
- classification report

Threshold search from `0.1` to `0.85` in steps of `0.05` optimizes:

- F1 thresholds (precision-recall balance)
- F2 thresholds (recall-weighted — used as runtime decision thresholds)

### Policy-level evaluation

Decision logic: if any label probability exceeds its F2 threshold, the verdict is `Harmful` and the firing categories are returned as tags. Otherwise the verdict is `Safe`.

Policy metrics:

- `precision`: fraction of Harmful decisions that are correct
- `recall`: fraction of truly harmful videos that are flagged
- `false_positive_rate`
- `false_negative_rate`
- `protection_rate`: fraction of harmful videos receiving a Harmful verdict

## 8. Prediction Pipeline

Validation predictions are generated through `trainer.predict(val_dataset)`. Logits are converted to probabilities with `torch.sigmoid`.

Saved prediction payloads include: `mode`, `label_order`, `video_ids`, `texts`, `labels`, `logits`, `predictions`.

Prediction files are saved as `predictions_{mode}.json` under the artifact directory.

## 9. Experiment Setup

Three final artifact configurations are maintained:

- `artifacts/text` — text-only model
- `artifacts/image_clip_nsfw_violence` — image model with CLIP + NSFW + violence features
- `artifacts/multimodal_clip_nsfw_violence` — multimodal fusion model

Each artifact directory contains: `metadata.json`, `predictions_{mode}.json`, `model/` (weights), `tokenizer/`.

Comparison reports are generated by:

- `scripts/save_results_snapshot.py` → `artifacts/reports/`
- `scripts/generate_experiment_report.py` → `artifacts/reports/`

Decision thresholds can be reevaluated from saved predictions without retraining:

- `scripts/reevaluate_policy_from_predictions.py`

## 10. Explainability

The current explainability method is `gradient_tokens`.

### Text mode

Gradient-based token attributions:

- embeds the tokenized text
- retains gradients on input embeddings
- runs a forward pass, selects target category probability after sigmoid
- backpropagates
- ranks tokens by gradient norm

### Image mode

No true image saliency. The system summarizes the top prompt-similarity features as evidence cues.

### Multimodal mode

Text-token attribution with `image_features` held fixed during the forward pass. The explanation includes text cues, thumbnail cues, and a modality summary.

### Limitations

- no integrated gradients, SHAP, or LIME
- no end-to-end image saliency
- image explanations are heuristic summaries of feature prompts, not visual attribution maps

## 11. Key Design Decisions

- precomputed thumbnail features rather than end-to-end image training
- image features frozen during classifier training
- simple MLP for image-only classification
- concatenation plus MLP fusion for multimodal classification
- F2-optimized thresholds loaded from artifact metadata at runtime — no hardcoded thresholds
- predictions stored so threshold and decision changes can be reevaluated without retraining
- mode-specific artifact folders isolate text, image, and multimodal experiments

## 12. Limitations

- no end-to-end fine-tuning of a vision backbone
- reliance on precomputed image features stored as `.npy`
- image branch depends on CLIP-derived generic features rather than a dedicated safety vision model
- no true image explainability beyond prompt-based cue summaries
- multimodal fusion is simple concatenation MLP rather than gated or attention-based fusion
- decision is fully rule-based and threshold-based; it is not learned from data
- threshold calibration differs per mode because score distributions differ
