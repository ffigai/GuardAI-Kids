# GuardAI Kids
<img width="8192" height="2367" alt="hi level diagram-2026-03-23-004627" src="https://github.com/user-attachments/assets/df0fe83f-5c20-4390-8190-52e9b718bc36" />

GuardAI Kids is an age-aware YouTube content safety analyzer. Given a YouTube URL, it recommends `ALLOW`, `WARN`, or `BLOCK` for three child age groups — `0–4`, `5–8`, and `9–12` — across four harm categories: addictive content (ADD), sexual/explicit material (SXL), physical harm (PH), and hate/harassment (HH).

## How It Works

The system supports three analysis modes:

| Mode | What it uses |
|---|---|
| `text` | Video title, description, and transcript |
| `image` | Thumbnail image features (CLIP + NSFW + violence classifiers) |
| `multimodal` | Text and image combined via a fusion layer |

Each mode runs a trained multi-label classifier and applies an age-aware policy to produce per-age-group recommendations with supporting evidence cues.

## Model Performance (Validation Set)

| Model | Mean AUC | Macro F1 |
|---|---|---|
| text | 0.938 | — |
| image — clip | 0.778 | 0.519 |
| image — clip_ocr | 0.845 | 0.591 |
| image — clip_nsfw_violence | 0.846 | 0.594 |
| multimodal — clip | 0.935 | — |
| multimodal — clip_ocr | — | — |
| multimodal — clip_nsfw_violence | — | — |

Full comparison graphs are in `artifacts/reports/`.

## Repo Structure

- `src/guardaikids/` — all modules: data, modeling, policy, explainability, workflow, YouTube integration, CLI, web UI
- `scripts/` — dataset preparation, image feature extraction, report generation, results snapshot
- `data/` — place `Harmful.xlsx` and `Harmless.xlsx` here
- `artifacts/` — per-configuration metadata, predictions, and evaluation reports (model weights excluded — regenerate by training)
- `tests/` — lightweight regression tests

## Requirements

- Python 3.12
- A YouTube Data API key (for live URL analysis)
- `data/Harmful.xlsx` and `data/Harmless.xlsx` with columns: `harm_cat`, `title`, `description`, `transcript`, `video_id`
- NVIDIA GPU recommended for training (CPU works but is slow)

## Setup

### 1. Create environment and install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

For CPU-only, replace the torch line with:

```powershell
python -m pip install torch
```

### 2. Configure environment

Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_api_key_here
```

### 3. Place dataset files

```
data/Harmful.xlsx
data/Harmless.xlsx
```

## Training

### Text mode

```powershell
python -m guardaikids train --mode text --artifact-dir artifacts/text
```

### Image modes

```powershell
# Step 1 — extract features (choose one pipeline)
python scripts/extract_image_features.py --image-analysis-model clip
python scripts/extract_image_features.py --image-analysis-model clip_ocr
python scripts/extract_image_features.py --image-analysis-model clip_nsfw_violence

# Step 2 — train
python -m guardaikids train --mode image --image-analysis-model clip --artifact-dir artifacts/image
python -m guardaikids train --mode image --image-analysis-model clip_ocr --artifact-dir artifacts/image_clip_ocr
python -m guardaikids train --mode image --image-analysis-model clip_nsfw_violence --artifact-dir artifacts/image_clip_nsfw_violence
```

### Multimodal modes

```powershell
python -m guardaikids train --mode multimodal --image-analysis-model clip --artifact-dir artifacts/multimodal
python -m guardaikids train --mode multimodal --image-analysis-model clip_ocr --artifact-dir artifacts/multimodal_clip_ocr
python -m guardaikids train --mode multimodal --image-analysis-model clip_nsfw_violence --artifact-dir artifacts/multimodal_clip_nsfw_violence
```

## Evaluation

```powershell
# Generate 7-way comparison graphs and summary CSV
python scripts/save_results_snapshot.py

# Generate full experiment report
python scripts/generate_experiment_report.py
```

Outputs go to `artifacts/reports/`.

## Web UI

```powershell
python -m guardaikids.web_interface
```

Open `http://127.0.0.1:5000`. Select `text`, `image`, or `multimodal` — image and multimodal use the `clip_nsfw_violence` artifact by default.

## CLI Analysis

```powershell
python -m guardaikids analyze --mode text --url "https://www.youtube.com/watch?v=VIDEO_ID"
python -m guardaikids analyze --mode multimodal --url "https://www.youtube.com/watch?v=VIDEO_ID" --json
```

## Troubleshooting

**First run downloads models from HuggingFace** — expected for CLIP, NSFW classifier, and violence classifier on first image/multimodal run.

**GPU not detected:**

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

**Web UI says `YOUTUBE_API_KEY is required`** — check that your `.env` file is in the project root and contains `YOUTUBE_API_KEY=...`.

## Running Tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Model weights are not committed (too large). Run training to regenerate them.
- `artifacts/*/metadata.json` and `predictions_*.json` are committed and contain all evaluation metrics.
- Policy and threshold changes can be reevaluated from saved predictions without retraining using `scripts/reevaluate_policy_from_predictions.py`.
