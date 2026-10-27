# Cross-Architecture Legal Bill Summarization Engine

An end-to-end inference engine, multi-axis evaluation service, REST API, and interactive dashboard for legal bill summarization across three transformer architectures fine-tuned on Congressional legislation:

1. **BART Base Dense** (`facebook/bart-base` + LoRA): Dense encoder-decoder, 1,024 context, ultra-fast inference latency (5.2 ms/token).
2. **LED Base Sparse 4K** (`allenai/led-base-16384` + LoRA): Sparse attention encoder-decoder, 4,096 context, linear scaling for long statutory texts.
3. **Qwen2.5-1.5B Causal** (`Qwen/Qwen2.5-1.5B-Instruct` + QLoRA): Autoregressive causal decoder, 2,048 context, state-of-the-art generation quality (ROUGE-L 25.76, BERTScore 80.05).

---

## 🏛️ Critical Architectural Guardrails

### 1. LED Global Attention Mask Guardrail (`src/models/led_summarizer.py`)
LED Longformer uses sparse local sliding-window attention for linear $O(N)$ memory efficiency. However, the summary generation decoder requires global visibility into the document start token:
- **Rule:** Explicitly generate a `global_attention_mask` tensor with the identical shape as `input_ids`.
- **Implementation:** Position `0` (the BOS `<s>` token) is explicitly set to `1` while all remaining positions are set to `0`.

### 2. Qwen Left-Padding & Prompt Slicing Guardrail (`src/models/qwen_summarizer.py`)
Autoregressive causal decoders will echo the input prompt text and fail batch alignment without strict left-padding:
- **Rule 1:** Enforce left-padding on the tokenizer: `tokenizer.padding_side = "left"`.
- **Rule 2:** `model.generate()` returns `[prompt_tokens + generated_tokens]`. We explicitly slice out the prompt tokens (`output_ids[:, prompt_len:]`) before decoding to ensure the model never echoes the bill text.

---

## 📊 Kaggle Track B Pareto Benchmark Summary

| Model Key | Architecture | Context | Params | Train Wall (s) | Peak VRAM | Latency / Sample | ROUGE-L | BERTScore F1 | Legal Entity Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`bart_base_dense`** | Seq2Seq Dense | 1,024 | 141M | 233s | 2.61 GB | **657.9 ms** | 19.90 | 78.30 | 10.76% |
| **`led_base_sparse_4k`** | Seq2Seq Sparse LED | 4,096 | 164M | 5,886s | **2.47 GB** | 1,250.9 ms | 18.79 | 77.97 | 7.77% |
| **`qwen25_1_5b_causal_qlora`**| Causal Decoder | 2,048 | 907M | 10,856s | 4.33 GB | 7,530.2 ms | **25.76** | **80.05** | **10.56%** |

---

## 🚀 Quickstart & Usage

For a complete clean-install guide, model downloads, CPU/GPU notes, verification steps, and troubleshooting, see [SETUP.md](SETUP.md).

### 1. One-Click Automated Startup (Recommended)
Run the automated PowerShell script to verify the environment, start both the FastAPI backend and Streamlit dashboard, wait for the health check, and open your browser:
```powershell
.\startup.ps1
```

Options:
- `.\startup.ps1 -OnlyAPI`: Launch only the FastAPI backend on port 8000.
- `.\startup.ps1 -OnlyDashboard`: Launch only the Streamlit dashboard on port 8501.
- `.\startup.ps1 -WithGraph` (or `-Graph`): Launch the interactive Knowledge Graph viewer (`graphify-out/graph.html`) on port 8088 alongside API and Dashboard.
- `.\startup.ps1 -OnlyGraph`: Launch only the Knowledge Graph viewer on port 8088.
- `.\startup.ps1 -UpdateGraph`: Re-extract and refresh the Graphify AST knowledge graph before launching.
- `.\startup.ps1 -RunTests`: Run the full pytest verification test suite before starting services.
- `.\startup.ps1 -NoBrowser`: Start services without automatically opening the browser.
- `.\startup.ps1 -OnlyDashboard -RemoteApiUrl https://<current-colab-tunnel>`: Run Streamlit locally and route model inference to the Colab API.
- `.\startup.ps1 -ApiPort 8080 -DashboardPort 8502 -GraphPort 8089`: Custom port mappings.

---

### 2. Manual Environment Setup & Tests
```powershell
# Create and activate a Python 3.11 virtual environment (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the test suite
python -m pytest -v
```
The tests cover model constraints, LED global attention, Qwen prompt slicing, regex entity extraction, evaluation pipelines, and FastAPI endpoints. Real inference also requires downloading the base models into the Hugging Face cache; see [SETUP.md](SETUP.md#4-download-base-models-for-real-inference).

### 3. Start the FastAPI Inference & Evaluation Backend
```bash
uvicorn src.api.app:app --reload --port 8000
```
- Swagger API Docs: `http://localhost:8000/docs`
- Endpoints:
  - `POST /api/summarize`: Summarize bill text with single model or comparative `"all"` models.
  - `POST /api/evaluate`: Compute ROUGE, BERTScore, and Legal Entity Recall.
  - `POST /api/extract-entities`: Extract sections, U.S.C. citations, monetary figures, and percentages.
  - `GET /api/benchmarks/pareto`: Retrieve empirical Track B Kaggle benchmark dataset.
  - `GET /api/models`: Model metadata and guardrail descriptions.

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```

For Colab GPU inference with a local dashboard, follow the Colab setup in
[SETUP.md](SETUP.md#6-run-model-inference-on-colab-with-a-local-dashboard). The tunnel URL changes
when the Colab runtime restarts; update `-RemoteApiUrl` each time. Keep Dry-Run disabled for
actual model-generated summaries.
- **🏛️ Model Arena:** Live side-by-side comparative inference across BART, LED, and Qwen2.5 with live latency and speed counters.
- **📊 Pareto Frontier:** Interactive Plotly curves (Quality vs. Latency, Memory vs. Quality, Green AI eco-efficiency).
- **🔬 Evaluation Lab:** Score candidate summaries against gold-standard references with colored legal entity tags.
- **📐 Architecture & Guardrails:** Code breakdown and technical specifications.

---

## 📁 Repository Structure
```
├── artifacts/                           # Extracted Track B adapters & Kaggle benchmark artifacts
│   ├── models/                          # Fine-tuned LoRA / QLoRA adapters
│   │   ├── bart_base_dense/             # facebook/bart-base LoRA adapter
│   │   ├── led_base_sparse_4k/          # allenai/led-base-16384 LoRA adapter
│   │   └── qwen25_1_5b_causal_qlora/    # Qwen2.5-1.5B QLoRA adapter
│   ├── track_b_pareto_summary.csv       # Benchmark metrics CSV
│   └── pareto_frontier_track_b.png      # Kaggle Pareto curve visualization
├── src/
│   ├── models/                          # Core Inference Engine
│   │   ├── base.py                      # BaseSummarizer interface & SummarizationResult
│   │   ├── bart_summarizer.py           # Dense BART summarizer
│   │   ├── led_summarizer.py            # Sparse LED with global_attention_mask guardrail
│   │   ├── qwen_summarizer.py           # Qwen with left-padding & prompt slicing guardrail
│   │   └── registry.py                  # Factory, caching, and model card registry
│   ├── metrics/                         # Multi-Axis Evaluation Service
│   │   ├── entity_extractor.py          # Regex legal entity extractor (Sections, USC, Money, %)
│   │   └── evaluator.py                 # LegalEvaluator (ROUGE, BERTScore, Entity Recall)
│   ├── api/                             # FastAPI Backend Service
│   │   ├── app.py                       # REST API implementation
│   │   └── schemas.py                   # Pydantic request & response schemas
│   └── dashboard/                       # Streamlit Interactive Dashboard
│       └── app.py                       # Side-by-side arena & Plotly Pareto frontier
├── tests/                               # Verification Test Suite
│   ├── test_api.py                      # API integration tests
│   ├── test_base_summarizer.py          # Base model interface tests
│   ├── test_evaluator.py                # Multi-axis evaluation tests
│   ├── test_led_guardrails.py           # LED global_attention_mask tests
│   ├── test_legal_entity_regex.py       # Legal entity regex extraction & recall tests
│   └── test_qwen_guardrails.py          # Qwen left-padding & prompt slicing tests
└── requirements.txt                     # Pinned project dependencies
```
