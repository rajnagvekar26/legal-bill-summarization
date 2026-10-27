# Project Setup and Operations

This guide covers a clean Windows setup, optional Linux/macOS equivalents, local verification, and running the API and dashboard. The project is a Python 3.11 application with a FastAPI backend, a Streamlit dashboard, and three Hugging Face transformer models.

## 1. Prerequisites

- Python 3.11 (recommended; this is the version used to verify this checkout). On Windows, install Python from python.org and enable the Python launcher (`py`).
- Git, if you are cloning the project or contributing changes.
- Internet access for Python packages and the base-model downloads described below.
- For real inference, enough disk space for the three base models and enough memory/VRAM for the model you select. CPU inference is supported but can be slow. NVIDIA CUDA is optional.

The fine-tuned adapters and tokenizers are in `artifacts/models/`. The larger base-model weights are not included. Model loading uses `local_files_only=True`, so the base models must be in the local Hugging Face cache before real inference; the application does not download them automatically.

## 2. Get the project

Clone the repository URL supplied by the project owner, then enter its directory:

```powershell
git clone <repository-url>
cd <repository-directory>
```

Or open the downloaded project directory in VS Code. Run all commands below from the directory containing `requirements.txt` and `startup.ps1`.

## 3. Create the Python environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for this terminal only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the same environment for tests and both services. `startup.ps1` creates `.venv` and installs `requirements.txt` if the environment does not exist; if it already exists, it does not reconcile or upgrade installed packages. The manual commands above are the reliable way to create or refresh the environment.

### PyTorch and GPU notes

`requirements.txt` installs PyTorch using pip's default wheel for the current platform. For CUDA inference, install a PyTorch wheel matching the installed NVIDIA driver and CUDA runtime using the official PyTorch install selector, then confirm `torch.cuda.is_available()` returns `True`. The Qwen model attempts 4-bit loading when CUDA and BitsAndBytes are available, and falls back to standard precision otherwise. `bitsandbytes` support varies by operating system and CUDA version; CPU inference does not use 4-bit quantization.

## 4. Download base models for real inference

The checked-in adapter weights are not full models. Download each base model into the Hugging Face cache before requesting real summaries. With the project environment activated, run:

```powershell
hf download facebook/bart-base
hf download allenai/led-base-16384
hf download Qwen/Qwen2.5-1.5B-Instruct
```

The equivalent commands work in a bash shell. These downloads require several gigabytes of free disk space. If `hf` is not recognized, verify that `.venv` is activated and run `python -m pip install --upgrade huggingface_hub`, then open a new terminal or use the installed `hf.exe` under `.venv\Scripts`.

The API's `dry_run` option is intended for tests and service checks; it does not load model weights and does not generate a model-quality summary. Set it to `false` (the default) for actual inference after downloading the base models.

## 5. Run the automated startup script (Windows)

From PowerShell in the project root:

```powershell
.\startup.ps1
```

The script starts the API on `http://localhost:8000`, waits for `/api/health`, starts the dashboard on `http://localhost:8501`, and opens the dashboard in a browser. Keep that terminal open. Press `Ctrl+C` there to stop the services.

Options:

```powershell
.\startup.ps1 -OnlyAPI
.\startup.ps1 -OnlyDashboard
.\startup.ps1 -NoBrowser
.\startup.ps1 -ApiPort 8080 -DashboardPort 8502 -NoBrowser
```

The health check confirms the API process is responding; it does not load the language models. If the script reports a health-check timeout, inspect its terminal output and try the manual API start below.

## 6. Run model inference on Colab with a local dashboard

The Streamlit UI can stay on Windows while the FastAPI inference service and all three models run
inside a Colab GPU runtime. The Colab workspace must contain this checkout, including the
`artifacts/models/` adapter files. For a private GitHub repository, use your approved authenticated
clone/upload workflow; do not put access tokens in notebook cells.

In Colab, first ensure a GPU runtime is selected, then run:

```python
# Clone or mount the current project checkout that contains src/ and artifacts/models/.
%cd /content
!git clone https://github.com/Ohmingx/TRIAL-DLE.git
%cd /content/TRIAL-DLE

# Keep Colab's preinstalled CUDA-enabled PyTorch; install the model and API dependencies.
%pip install transformers peft accelerate bitsandbytes pandas fastapi "uvicorn[standard]" huggingface_hub

import torch
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime in Colab before starting model inference.")
```

The model loaders intentionally use local-only Hugging Face loading. Download the three base
models into the Colab cache before starting the API:

```python
from huggingface_hub import snapshot_download

for model_id in (
    "facebook/bart-base",
    "allenai/led-base-16384",
    "Qwen/Qwen2.5-1.5B-Instruct",
):
    snapshot_download(repo_id=model_id)
```

Start the repository's real API (not a notebook with placeholder inference functions), then expose
it through LocalTunnel in a separate cell:

```python
import subprocess
import sys
import time
import requests

api_process = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "src.api.app:app",
        "--host", "0.0.0.0", "--port", "8000",
    ],
    cwd="/content/TRIAL-DLE",
)

for _ in range(60):
    if api_process.poll() is not None:
        raise RuntimeError("Colab API exited during startup.")
    try:
        health = requests.get("http://127.0.0.1:8000/api/health", timeout=2)
        if health.ok and health.json().get("status") == "healthy":
            print(health.json())
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Colab API did not become healthy.")
```

In another Colab cell:

```python
!npm install -g localtunnel
!lt --port 8000
```

Copy the `https://...loca.lt` URL printed by LocalTunnel and use it on Windows:

```powershell
.\startup.ps1 -OnlyDashboard -RemoteApiUrl https://your-current-tunnel.loca.lt
```

`startup.ps1` checks the remote `/api/health` endpoint and starts only the local Streamlit UI.
Alternatively, run Streamlit manually after setting the environment variable:

```powershell
$env:SUMMARIZATION_API_URL = "https://your-current-tunnel.loca.lt"
python -m streamlit run src/dashboard/app.py --server.port 8501
```

Colab tunnel URLs are temporary and typically change after a runtime restart, so copy the new URL
each time. The dashboard sends requests to `/api/summarize`; selected model keys are honored and
generation runs on the Colab host. Leave Dry-Run disabled. If model loading or inference fails,
the API returns an error rather than a fabricated dry-run summary. Keep the Colab notebook running
while using the local dashboard.

## 7. Run services manually

Activate `.venv` first. Start each service in its own terminal, from the project root.

API:

```powershell
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Dashboard:

```powershell
python -m streamlit run src/dashboard/app.py --server.port 8501
```

On Linux/macOS, use the same commands after activating `.venv`. The API docs are at `http://localhost:8000/docs`; the dashboard is at `http://localhost:8501`.

## 8. Verify the setup

Run the automated test suite from the project root:

```powershell
python -m pytest -v
```

The tests exercise API routes, model registry behavior, evaluation, entity extraction, and model-specific guardrails without downloading full model weights.

With the API running, check health in another PowerShell terminal:

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

A lightweight end-to-end API check (no base model download required):

```powershell
$body = @{ text = 'SECTION 1. This bill authorizes funding for water treatment.'; model_key = 'bart'; dry_run = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/summarize -Method Post -ContentType 'application/json' -Body $body
```

The `dry_run` response should include a non-empty placeholder summary and model metadata. To test real inference, set `dry_run` to `$false` after downloading the selected base model. For a direct model comparison, use `model_key = 'all'`; this requires all three base models for real inference.

## 9. API routes

- `GET /api/health`: service and PyTorch device status.
- `GET /api/models`: supported models and metadata.
- `GET /api/benchmarks/pareto`: checked-in benchmark summary.
- `POST /api/summarize`: generate a summary with one model or all models.
- `POST /api/evaluate`: compare a candidate and reference summary.
- `POST /api/extract-entities`: extract sections, U.S.C. citations, monetary values, and percentages.

The interactive request and response schemas are available at `/docs`.

## 10. Troubleshooting

- **`py` or `python` is not recognized:** install Python 3.11 and reopen PowerShell. Check with `py -3.11 --version`.
- **The wrong interpreter is active:** activate `.venv` and check `python -c "import sys; print(sys.executable)"`.
- **The API health check works but real summarization fails or falls back:** confirm the selected base model is cached under the same Windows user account running the API. The loaders use offline-only model loading.
- **CUDA is unavailable:** verify the installed PyTorch wheel and NVIDIA driver compatibility. The app will use CPU when device selection is `auto` and CUDA is unavailable.
- **Dashboard/API port is occupied:** choose unused ports with `-ApiPort` and `-DashboardPort`, or change the manual run command's `--port` value.
- **Dependency import errors after changing packages:** activate `.venv` and reinstall with `python -m pip install -r requirements.txt`.
- **Model inference is slow or runs out of memory:** test one model at a time, reduce generation token limits, or use a machine with more available memory/VRAM.

## 11. Repository layout

- `src/api/`: FastAPI application and request/response schemas.
- `src/dashboard/`: Streamlit user interface.
- `src/models/`: BART, LED, and Qwen summarizers, shared base class, and registry.
- `src/metrics/`: legal entity extraction and evaluation.
- `tests/`: automated tests.
- `artifacts/models/`: fine-tuned adapters and tokenizer files used by the summarizers.
- `artifacts/track_b_pareto_summary.csv`: benchmark data shown in the dashboard/API.
- `startup.ps1`: Windows launcher for both services.
- `requirements.txt`: Python dependencies.
