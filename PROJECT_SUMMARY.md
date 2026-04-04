# GuardAI Kids Project Summary

## 1. Problem Definition

GuardAI Kids is an age-aware YouTube content safety system. It analyzes a YouTube video and produces age-specific recommendations for whether the content should be allowed, warned, or blocked for children aged `0-4`, `5-8`, and `9-12`.

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
- policy-level age-aware decisions:
  - `Allow`
  - `Warn`
  - `Block`

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

The feature vector dimension is:

- `512` CLIP embedding dimensions
- `8` prompt-similarity dimensions
- `5` image-quality dimensions
- `1` missing flag

Total:

- `526`

The 8 prompt-similarity features correspond to fixed prompts such as:

- child-friendly harmless thumbnail
- addictive or substance-related content
- sexual or explicit content
- physical harm or self-injury
- hate speech or harassment
- weapons, threats, or intimidation
- disturbing or scary for children
- risky stunts, pranks, or dangerous challenges

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

Image features come from:

- `512`-dimensional CLIP embedding
- `8` prompt-similarity scores
- `5` image-quality features
- `1` missing flag

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

- text embedding
- image feature vector

The fusion head is:

- `Linear(text_hidden_size + image_feature_dim -> fusion_hidden_dim)`
- `ReLU`
- `Dropout(0.3)`
- `Linear(fusion_hidden_dim -> num_labels)`

`fusion_hidden_dim` defaults to the transformer hidden size unless explicitly overridden.

### Output

The task is multi-label classification over four categories. The model returns:

- logits
- optional loss

Loss function:

- `BCEWithLogitsLoss`

Sigmoid is applied outside the loss path for evaluation and prediction.

## 5. Modes of Operation

### Text-only

Inputs:

- `input_ids`
- `attention_mask`

Model path:

- transformer encoder
- text classifier head

Use case:

- baseline text classification from title, description, and transcript

### Image-only

Inputs:

- `image_features`

Model path:

- image MLP classifier only

Use case:

- thumbnail-based classification without text

### Multimodal

Inputs:

- `input_ids`
- `attention_mask`
- `image_features`

Model path:

- transformer encoder for text
- concatenation with image features
- fusion MLP

Use case:

- combined text and thumbnail analysis

## 6. Training Setup

Training uses Hugging Face `Trainer`.

Key components:

- tokenizer: `AutoTokenizer`
- model: `MultimodalSequenceClassifier`
- trainer: `transformers.Trainer`
- args: `TrainingArguments`

Training arguments are fixed in code:

- evaluation every epoch
- save every epoch
- learning rate `2e-5`
- train batch size `16`
- eval batch size `16`
- epochs `3`
- weight decay `0.01`
- `load_best_model_at_end=True`

Batching behavior depends on mode because the prepared dataset columns differ:

- `text`: text tensors only
- `image`: image feature tensors only
- `multimodal`: both text tensors and image tensors

The loss function remains the same across modes:

- multi-label `BCEWithLogitsLoss`

## 7. Evaluation Methodology

### Model-level evaluation

Validation outputs are collected with `trainer.predict()`.

Metrics:

- macro F1
- micro F1
- per-label ROC-AUC
- classification report

Default thresholding for `compute_metrics()` uses:

- sigmoid probabilities
- cutoff at `0.5`

In addition, the workflow searches per-label thresholds from `0.1` to `0.85` in steps of `0.05` to optimize:

- F1 thresholds
- F2 thresholds

`F2` thresholds are then applied to generate a tuned prediction summary.

### Policy-level evaluation

Age groups:

- `0_4`
- `5_8`
- `9_12`

Decision logic is rule-based and threshold-based:

1. if any label exceeds its age-specific `block` threshold, return `Block`
2. else if any label exceeds its age-specific `warn` threshold, return `Warn`
3. else return `Allow`

Policy metrics:

- `block_precision`
- `block_recall`
- `false_block_rate`
- `false_allow_rate`
- `protection_precision`

Protection is evaluated separately with:

- `protection_metrics`

which measures the rate at which harmful content receives either `Warn` or `Block`.

## 8. Prediction Pipeline

Validation predictions are generated through:

- `trainer.predict(val_dataset)`

The returned logits are converted to probabilities with:

- `torch.sigmoid`

Saved prediction payloads include:

- `mode`
- `label_order`
- `video_ids`
- `texts`
- `labels`
- `logits`
- `predictions`

Prediction files are saved as:

- `predictions_text.json`
- `predictions_image.json`
- `predictions_multimodal.json`

under the corresponding artifact directory.

## 9. Experiment Setup

The three modes are run separately:

- `python -m guardaikids train --mode text`
- `python -m guardaikids train --mode image`
- `python -m guardaikids train --mode multimodal`

Each mode writes to a separate artifact directory:

- `artifacts/text`
- `artifacts/image`
- `artifacts/multimodal`

This structure isolates:

- trained weights
- tokenizer
- metadata
- saved predictions

Comparison is performed by reading the saved `metadata.json` and prediction files, or through the reporting script:

- `scripts/generate_experiment_report.py`

Policy reevaluation is decoupled from retraining through:

- `scripts/reevaluate_policy_from_predictions.py`

## 10. Explainability

The current explainability method is:

- `gradient_tokens`

Supported XAI methods are constrained in code to:

- `{"gradient_tokens"}`

### Text mode

Text explanations are gradient-based token attributions. The implementation:

- embeds the tokenized text
- retains gradients on input embeddings
- runs a forward pass
- selects a target category probability after sigmoid
- backpropagates
- ranks tokens by gradient norm

### Image mode

There is no true image saliency method. Instead, the system summarizes the top image prompt-similarity features as parent-facing cues.

### Multimodal mode

Multimodal explanations support text-token attribution while keeping `image_features` fixed during the forward pass. The final explanation can include:

- text cues
- thumbnail cues
- modality summary

### Limitations in explainability

- no integrated gradients, SHAP, or LIME
- no end-to-end image saliency
- image explanations are heuristic summaries of feature prompts, not visual attribution maps

## 11. Key Design Decisions

Important implementation decisions include:

- use of precomputed thumbnail features rather than end-to-end image training
- keeping image features frozen during classifier training
- using a simple MLP for image-only classification
- using concatenation plus MLP fusion for multimodal classification
- separating model-level classification from age-aware policy logic
- storing predictions so policy and explanation changes can be reevaluated without retraining
- using mode-specific artifact folders to keep text, image, and multimodal experiments separate
- allowing CLI overrides for mode, image-analysis backend, and XAI method while still supporting config defaults

## 12. Limitations

Implementation-level limitations include:

- no end-to-end fine-tuning of a vision backbone
- reliance on precomputed image features stored as `.npy`
- image branch depends on CLIP-derived generic features rather than a dedicated safety vision model
- current live image/multimodal analysis still depends on runtime thumbnail feature extraction but not on a stronger specialized moderation model
- no true image explainability beyond prompt-based cue summaries
- multimodal fusion is a simple concatenation MLP rather than gated or attention-based fusion
- policy is fully rule-based and threshold-based; it is not learned from data
- threshold calibration must be tuned separately per mode because score distributions differ
