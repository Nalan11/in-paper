# AI Document Extractor 📄🚀

A robust, high-performance AI system for extracting structured JSON data from document images (invoices, receipts, etc.). This application leverages a sophisticated two-stage pipeline: **PaddleOCR-VL** for visual content understanding and **Qwen2.5-1.5B** for structured data extraction.

---

## 🏗️ Architecture & Component Isolation

To prevent dependency conflicts between **vLLM** and **PaddlePaddle**, this project uses two distinct environments:

1.  **Main Environment**: Runs the FastAPI backend, PaddleOCR-VL client, and pre/post-processing logic.
2.  **vLLM Isolation (`vllm_engine`)**: A dedicated virtual environment specifically for hosting the Large Language Model and Vision Language Model servers.

---

## ⚙️ Installation & Setup

### 1. Main Application Environment (FastAPI + Paddle Client)
Install the core application dependencies in your primary environment.
```bash
pip install -r requirements.txt
```

### 2. Isolated vLLM Environment (`vllm_engine`)
Create a separate environment using `uv` to host the model servers.
```bash
uv venv vllm_engine --python 3.10
source vllm_engine/bin/activate
uv pip install vllm
```

---

## 🚀 Running the System

### Phase 1: Start the Model Servers
You will need two separate terminals for the VLM and LLM servers. Ensure both are running from the `vllm_engine` environment.

#### **A. Start vLLM for OCR (VLM) - Port 8000**
```bash
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL \
  --trust-remote-code \
  --gpu-memory-utilization 0.4 \
  --max-model-len 8096 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --port 8000
```

#### **B. Start vLLM for JSON Extraction (LLM) - Port 8001**
```bash
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B \
  --trust-remote-code \
  --gpu-memory-utilization 0.25 \
  --max-num-batched-tokens 512 \
  --no-enable-prefix-caching \
  --max-model-len 1024 \
  --mm-processor-cache-gb 0 \
  --served-model-name Qwen2.5-1.5B \
  --port 8001
```

### Phase 2: Start the FastAPI Backend
Once the model servers are ready, run the main application from your primary environment.
```bash
python app.py
```
*Access the UI at: `http://localhost:8005` (or your assigned port).*

---

## 🛠️ Current Development Plan
The project is currently undergoing the following enhancements:
- [x] Collapsible UI tray for raw extracted text.
- [ ] Pre-processing logic (Markdown conversion, tag stripping).
- [ ] Deeply nested JSON schema for improved entity extraction.
- [ ] Validation gate for financial data integrity.
- [ ] Automated stress testing suite (VLM, LLM, and Pipeline evaluation).
