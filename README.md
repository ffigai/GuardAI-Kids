# GuardAI Kids

GuardAI Kids is an age-aware YouTube content safety analyzer. It reviews a YouTube video and recommends `Allow`, `Warn`, or `Block` for three child age groups:

- `0-4`
- `5-8`
- `9-12`

The system uses YouTube metadata and transcripts when available, then explains why each recommendation was made.

## What This Repo Contains

- `main.py`: command-line entrypoint for training and analysis
- `web_interface.py`: local web UI for entering a YouTube URL
- `etp/`: reusable modules for data loading, modeling, policy, explainability, workflow, and YouTube integration
- `data/`: expected location for `Harmful.xlsx` and `Harmless.xlsx`
- `artifacts/`: saved trained model, tokenizer, and metadata after training
- `tests/`: lightweight regression tests

## System Requirements

### Required

- Windows with PowerShell
- Python `3.12`
- A YouTube Data API key for live URL analysis
- The dataset files:
  - `data/Harmful.xlsx`
  - `data/Harmless.xlsx`

### Recommended

- NVIDIA GPU for training
- CUDA-capable PyTorch build if you want GPU acceleration

Training works on CPU, but it is much slower. In your current setup, CPU training was measured in hours while GPU training dropped to minutes.

## Dataset Requirements

Your Excel files must contain these columns:

- `harm_cat`
- `title`
- `description`
- `transcript`

`harm_cat` contains the target labels. The project currently expects label values from:

- `ADD`
- `SXL`
- `PH`
- `HH`

## Step-By-Step Setup

### 1. Install Python 3.12

Use one of these:

```powershell
winget install Python.Python.3.12
```

Or download it from:

`https://www.python.org/downloads/windows/`

During installation, enable:

- `Add python.exe to PATH`

### 2. Open The Project Folder

```powershell
cd "c:\Users\ffigueroa\Desktop\AL_KSAIM_9\ETP\Github\ETP"
```

### 3. Create A Virtual Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

You should see Python `3.12.x`.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the virtual environment again.

### 4. Install PyTorch

Choose one path.

#### Option A: CPU Only

Use this if you do not have an NVIDIA GPU or do not want to use it.

```powershell
python -m pip install --upgrade pip
python -m pip install torch
```

#### Option B: NVIDIA GPU

Use this if your machine has an NVIDIA GPU and you want faster training.

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check whether PyTorch can see the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected GPU-style output:

- CUDA version is not `None`
- `torch.cuda.is_available()` prints `True`
- your NVIDIA GPU name is shown

### 5. Install Project Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 6. Add Your Dataset Files

Place these files in the `data/` folder:

- `data/Harmful.xlsx`
- `data/Harmless.xlsx`

Or override the paths with environment variables:

```powershell
$env:ETP_HARMFUL_XLSX="C:\path\to\Harmful.xlsx"
$env:ETP_HARMLESS_XLSX="C:\path\to\Harmless.xlsx"
```

### 7. Train The System

```powershell
python main.py train
```

This creates the `artifacts/` folder with:

- trained model files
- tokenizer files
- `metadata.json`

You only need to retrain when you want to rebuild the model.

### 8. Set The YouTube API Key

For the current PowerShell session:

```powershell
$env:YOUTUBE_API_KEY="your_api_key_here"
```

To save it for future sessions:

```powershell
[System.Environment]::SetEnvironmentVariable("YOUTUBE_API_KEY","your_api_key_here","User")
```

Then reopen PowerShell.

## How To Use The System

### Analyze A YouTube URL From The CLI

```powershell
python main.py analyze --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional JSON output:

```powershell
python main.py analyze --url "https://www.youtube.com/watch?v=VIDEO_ID" --json
```

### Run The Web Interface

```powershell
python web_interface.py
```

Then open:

`http://127.0.0.1:5000`

Paste a YouTube URL into the page to get:

- recommendation for ages `0-4`
- recommendation for ages `5-8`
- recommendation for ages `9-12`
- explanatory text for each age group

## Troubleshooting

### Python Not Found

If `python` does not resolve but `py` does, use `py -3.12` to create the environment and then use the virtual environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py train
```

### Training Is Very Slow

You are likely on CPU. Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, PyTorch is not using your GPU.

### Hugging Face Model Download Fails

The first training run needs internet access to download `distilroberta-base` from Hugging Face.

### Web UI Says `YOUTUBE_API_KEY is required`

Set the key first:

```powershell
$env:YOUTUBE_API_KEY="your_api_key_here"
python web_interface.py
```

## Running Tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Training is a build step.
- URL analysis is the finished product flow.
- `artifacts/` should be kept if you want to use the trained model without retraining.
