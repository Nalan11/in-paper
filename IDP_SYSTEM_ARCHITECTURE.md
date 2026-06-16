# Experimental IDP System: Architectural Blueprint & Handoff

## 1. System Overview & Project Context
This document serves as an exhaustive technical breakdown of an experimental Intelligent Document Processing (IDP) system. Built to operate within strict hardware constraints (a single 16GB VRAM Nvidia T4 GPU), the system evolved from a monolithic script into a highly parallelized, modular, and transparent AI platform. 

This blueprint is designed to act as the foundational context for migrating the IDP logic into a major subsystem of a larger, production-grade Document Archiving MVP.

---

## 2. Infrastructure & Serving Layer
*   **Environment**: Cloud-based Virtual Machine (Lightning AI Studio).
*   **Serving Engine**: **vLLM** is used exclusively for its high-throughput Continuous Batching and OpenAI-compatible API endpoints.
*   **Resource Management**: Strict VRAM partitioning is enforced using the `--gpu-memory-utilization` flag in vLLM to host multiple models concurrently (e.g., 0.40 for Vision, 0.30 for Reasoning) without triggering Out-Of-Memory (OOM) errors on the 16GB T4.
*   **Concurrency**: The backend utilizes Python's `ThreadPoolExecutor` to process multi-page PDFs concurrently, wrapping blocking I/O calls to prevent freezing the FastAPI `asyncio` event loop.

---

## 3. Core Processing Pipelines (Architectural Evolution)
The system's core extraction logic underwent a significant evolution to maximize accuracy and layout retention.

### Pipeline V1: The "Dual-Model" Approach (OCR + LLM)
*   **Step 1 (Vision):** `pdf2image` splits documents into images. **PaddleOCR-VL** (via vLLM) extracts raw text, preserving basic reading order.
*   **Step 2 (Context Engineering):** Python scripts strip noise (e.g., HTML `<img>` tags) and attempt to format bounding-box tables into Markdown to preserve structure.
*   **Step 3 (Reasoning):** A language model (Qwen 4B AWQ) receives the cleaned text and a strict System Prompt, outputting the final JSON.
*   *Limitation Discovered*: Complex tables and physically adjacent entities (like Vendor and Client names placed side-by-side) often lose their spatial relationship when flattened to text, leading to LLM confusion.

### Pipeline V2: The "Single-Model Vision" Approach (Native VLM)
*   **The Pivot**: Transitioning to native Vision-Language Models (e.g., **Qwen3-VL-4B-Instruct**).
*   **The Flow**: Images are encoded to base64 and sent directly to the VLM alongside the extraction prompt. 
*   **The Advantage**: The model processes the visual layout and text simultaneously in a single pass. This dramatically reduces Vendor/Client hallucination and retains structural integrity for complex tables, representing the optimal path forward for the MVP.

---

## 4. The "All-Rounder" Dynamic Routing System
To ensure the IDP can handle various document types within the archiving system, a modular routing architecture was implemented.

1.  **Decoupled Prompts**: Hardcoded rules were removed. Instructions and target JSON schemas are stored in external `.txt` files (e.g., `src/prompts/invoice.txt`, `resume.txt`, `receipt.txt`).
2.  **The Classifier Node**: A "Pre-Flight" request sends the first 1000 characters (or the first image) to the model with a classification prompt (*"Is this an INVOICE, RESUME, or RECEIPT?"*).
3.  **Dynamic Loading**: Based on the model's one-word classification, the backend dynamically loads the corresponding prompt schema from the file system and applies it to the extraction phase.

---

## 5. System Hardening & "Bulletproofing" (Defensive AI)
Production-grade IDP systems require resilience against unpredictable probabilistic outputs. Several defensive layers were engineered:

*   **The "Coercion" Pattern**: Standard Python validation expects a `dict`. If the LLM hallucinates and outputs an empty array `[]` or a raw string, it causes silent `TypeError` crashes. The system employs a generic safety net (`if not isinstance(data, dict): data = {}`) that forcefully coerces malformed outputs into a safe, empty dictionary, flagging it for human review rather than crashing the pipeline.
*   **Reasoning Tag Stripping**: Reasoning models (like Qwen3) inject "internal monologues" wrapped in `<think>...</think>` tags. A regex-based stripper (`re.sub(r'<think>.*?</think>', '', text)`) is applied to backend responses (like the Classifier node) to ensure internal thoughts do not break strict string-matching logic (e.g., file loading).
*   **Global Exception Middleware**: A FastAPI middleware catches unhandled exceptions globally, returning generic 500 errors gracefully while logging the stack trace.
*   **Background Janitor**: An asynchronous `background_cleanup_task` routinely purges the `temp/` directory to prevent disk bloat from concurrent PDF-to-Image splits.

---

## 6. UI/UX: The "Glass Pipeline" & Real-Time Monitoring
The experimental UI was designed for absolute transparency during the engineering phase, allowing developers to inspect every node of the AI pipeline.

*   **The 5-Stage Diagnostic View**: Instead of just showing the final result, the UI renders a vertical "Glass Pipeline" for every single page processed:
    1. VLM Raw Output
    2. Cleaned/Filtered Text
    3. Full Injected LLM Prompt Payload
    4. Raw LLM String Output (showing reasoning tags)
    5. Final Parsed JSON Object
*   **Event-Driven Progress**: Server-Sent Events (SSE) bridge the gap between backend concurrent threads and the frontend, driving real-time, multi-track progress bars (Vision, Reasoning, Storage) without polling.
*   **Infrastructure Management**: The dashboard includes controls to Start/Stop vLLM API servers and streams their internal `stdout` logs directly to the UI via WebSockets.
*   **Floating Chatbot**: An interactive widget utilizes `AsyncOpenAI` streaming to allow users to ask questions about the extracted document data. It includes a custom JavaScript parser to isolate and collapse `<think>` tags into a neat "Reasoning" block, maintaining a clean chat interface.

---

## 7. Accuracy Optimization Techniques (The Sandbox)
A dedicated `sandbox/` directory was established to push model accuracy to 99% without contaminating production code.

*   **Guided JSON Decoding**: Utilizing vLLM's `response_format={"type": "json_object"}` to force the model at the token-generation level to adhere to JSON syntax.
*   **Thermodynamic Restraints**: Operating at extreme low temperatures (`temperature=0.01` to `0.1`) to ensure deterministic, literal extraction over creative guessing.
*   **Few-Shot Patterning**: Injecting pairs of "Messy Input" vs "Perfect JSON Output" directly into the System Prompt to train the model via pattern recognition, which is highly effective for smaller 4B parameter models.
*   **Agentic Self-Correction**: Proof-of-concept Python logic demonstrating how an "Auditor Loop" can automatically evaluate extracted JSON (e.g., verifying `subtotal + tax == total`) and trigger a secondary "Fixer" LLM call if the math fails validation.

---

## 8. Data Persistence (Transition Plan)
*   **Current State**: Extracted data is normalized and stored in Relational CSVs (`invoices.csv`, `line_items.csv`), utilizing a unique `invoice_id` string.
*   **MVP Target**: The system is primed for a migration to SQLite/SQLAlchemy to support complex querying and robust data integrity required by the larger Document Archiving system.