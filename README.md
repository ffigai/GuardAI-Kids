# GuardAI Kids

GuardAI Kids is an age-aware YouTube content safety analyzer. It reviews a YouTube video and recommends `ALLOW`, `WARN`, or `BLOCK` for three child age groups:

- `0-4`
- `5-8`
- `9-12`

The system supports three analysis modes:

- `text`
- `image`
- `multimodal`

It uses YouTube metadata, transcripts, and thumbnail-derived image features when available, then explains why each recommendation was made.

## What This Repo Contains

- `src/guardaikids/`: reusable modules for data loading, modeling, policy, explainability, workflow, YouTube integration, CLI, and web UI
- `scripts/`: dataset preparation, policy reevaluation, report generation, and other utilities
- `data/`: expected location for `Harmful.xlsx` and `Harmless.xlsx`
- `artifacts/`: saved trained model, tokenizer, metadata, and predictions
- `tests/`: lightweight regression tests

## Updates In This Version

Compared with the earlier text-focused version, this repo now includes:

- configurable `text`, `image`, and `multimodal` modes
- thumbnail download and image-feature extraction scripts
- separate artifact folders per mode:
  - `artifacts/text`
  - `artifacts/image`
  - `artifacts/multimodal`
- richer prediction artifacts for reevaluation without retraining
- policy reevaluation and experiment report scripts
- a more complete web UI with:
  - mode selection
  - thumbnail preview for image-capable modes
  - basic and detailed views
  - visual score bars and evidence cues

## System Requirements

### Required

- Python `3.12`
- A YouTube Data API key for live URL analysis
- The dataset files:
  - `data/Harmful.xlsx`
  - `data/Harmless.xlsx`

### Recommended

- NVIDIA GPU for training
- CUDA-capable PyTorch build if you want GPU acceleration

Training works on CPU, but it is much slower.

## Dataset Requirements

Your Excel files must contain these columns:

- `harm_cat`
- `title`
- `description`
- `transcript`

The project expects target labels from:

- `ADD`
- `SXL`
- `PH`
- `HH`

Useful additional columns for multimodal experiments:

- `video_id`
- `thumbnail_harm_cat`

See `data/README` for the dataset notes.

## Step-By-Step Setup

### 1. Install Python 3.12

Install Python `3.12` from:

`https://www.python.org/downloads/`

### 2. Open The Project Folder

```powershell
cd path\to\GuardAI-Kids
```

### 3. Create A Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

You should see Python `3.12.x`.

### 4. Install PyTorch

CPU only:

```powershell
python -m pip install --upgrade pip
python -m pip install torch
```

NVIDIA GPU example:

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Project Dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 6. Add Your Dataset Files

Place these in `data/`:

- `data/Harmful.xlsx`
- `data/Harmless.xlsx`

Or override them:

```powershell
$env:ETP_HARMFUL_XLSX="C:\path\to\Harmful.xlsx"
$env:ETP_HARMLESS_XLSX="C:\path\to\Harmless.xlsx"
```

### 7. Optional: Prepare Thumbnail Assets For Image And Multimodal Modes

Download thumbnails:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python scripts\fetch_dataset_thumbnails.py
```

This creates:

- `data/thumbnails/`
- `data/Harmful_with_thumbnails.xlsx`
- `data/Harmless_with_thumbnails.xlsx`

Extract image features:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python scripts\extract_image_features.py
```

This creates:

- `data/image_features/`
- `data/image_features/feature_manifest.json`

### 8. Set The YouTube API Key

```powershell
$env:YOUTUBE_API_KEY="your_api_key_here"
```

## How To Run The Project

There are three main stages:

1. train
2. evaluate
3. demo in the web UI

## Training

You can use `MODE` from `src/guardaikids/config.py`, or override it from the CLI.

### Text

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m guardaikids train --mode text
```

### Image

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m guardaikids train --mode image
```

### Multimodal

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m guardaikids train --mode multimodal
```

Each run writes to its own artifact folder:

- `artifacts/text`
- `artifacts/image`
- `artifacts/multimodal`

Each artifact folder contains:

- model files
- tokenizer files
- `metadata.json`
- `predictions_<mode>.json`

## Evaluation

### Compare the saved results

Look at:

- `artifacts/text/metadata.json`
- `artifacts/image/metadata.json`
- `artifacts/multimodal/metadata.json`

Important fields:

- `roc_auc`
- `policy_metrics`
- `protection_metrics`
- `f1_thresholds`
- `f2_thresholds`

### Reevaluate policy without retraining

If you only change thresholds or policy logic, you do not need to retrain.

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python scripts\reevaluate_policy_from_predictions.py --artifact-dir .\artifacts\text --mode text
python scripts\reevaluate_policy_from_predictions.py --artifact-dir .\artifacts\image --mode image
python scripts\reevaluate_policy_from_predictions.py --artifact-dir .\artifacts\multimodal --mode multimodal
```

### Generate experiment tables and plots

After all three modes are trained:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python scripts\generate_experiment_report.py
```

Outputs are written to `artifacts/reports/`.

## Analyze A YouTube URL From The CLI

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m guardaikids analyze --mode text --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional JSON output:

```powershell
python -m guardaikids analyze --mode multimodal --url "https://www.youtube.com/watch?v=VIDEO_ID" --json
```

## Run The Web UI

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m guardaikids.web_interface
```

Then open:

`http://127.0.0.1:5000`

The web UI includes:

- mode selection
- basic and detailed views
- thumbnail preview for image-capable modes
- age-group recommendations
- evidence summaries

## Troubleshooting

### `ModuleNotFoundError: No module named 'guardaikids'`

Use:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m guardaikids train --mode text
```

### Training Is Very Slow

Check whether PyTorch sees your GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### First Hugging Face / CLIP Run Downloads Models

That is expected on the first run of:

- training
- image feature extraction
- image or multimodal live analysis

### Web UI Says `YOUTUBE_API_KEY is required`

Set the key first:

```powershell
$env:YOUTUBE_API_KEY="your_api_key_here"
python -m guardaikids.web_interface
```

## Running Tests

```powershell
python -m unittest discover -s tests -v
```

If needed:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```

## Notes

- Training is the expensive build step.
- Policy and explanation changes can often be reevaluated from saved predictions without retraining.
- `artifacts/` should be kept if you want to reuse trained models.
- Thumbnail download and image feature extraction are dataset-preparation steps, so they live in `scripts/`.
