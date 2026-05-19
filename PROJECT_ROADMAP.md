# Invoice Processing System - Final Architectural Roadmap

## Vision
To build a robust, production-grade document intelligence system that handles complex invoices with high accuracy, graceful error recovery, and visual transparency. The system is designed to run efficiently on a single 16GB T4 GPU using a modular "Foundation-First" approach.

---

## Phase 1: Modular Foundation & High-Availability Servers
*Goal: Move from monolithic scripts to a maintainable package structure with automated resource management.*
1.  **Directory Restructure:** Migrate logic into a `src/` package (e.g., `src.engines.vlm`, `src.engines.llm`, `src.utils.error_handler`).
2.  **Server Automation (`start_servers.sh`):** A robust shell script to boot VLM/LLM vLLM servers with `--gpu-memory-utilization` set to `0.45` each, ensuring they coexist on the 16GB T4 without OOM (Out of Memory) errors.
3.  **Graceful Init:** Implement health-check retries in the Python backend to wait until servers are fully "Ready" before accepting requests.

## Phase 1.5: Infrastructure Control Center & Live Logging
*Goal: Move infrastructure management from the terminal to the Web UI for better control and visibility.*
1.  **Process Manager:** Build a Python `ProcessManager` utility to start/stop vLLM subprocesses and monitor their PID/Health status.
2.  **Admin API:** Create FastAPI endpoints for remote server control (`/admin/start/{server_type}`, `/admin/stop/{server_type}`).
3.  **WebSocket Log Streamer:** Implement a WebSocket endpoint to stream real-time tail output from `logs/vlm.log` and `logs/llm.log` to the browser.
4.  **Log Management:** Implement a UI-driven "Clear Logs" feature that truncates log files on disk and circular buffering in the browser to prevent RAM bloat.
5.  **UI Control Panel:** A dedicated dashboard tab with live health badges (🔴/🟡/🟢), server toggles, and scrolling terminal windows.

## Phase 2: Multi-Page PDF Handling & Page-Level Extraction
*Goal: Solve the "Large Context" problem by processing documents one page at a time.*
1.  **PDF-to-Image Pipeline:** Use `pdf2image` to split documents into a temporary sequence of JPGs.
2.  **Sequential Processing:** Process each page individually through the VLM.
3.  **Context Management:** If a document is large, the system will extract data page-by-page and "compile" a final JSON by merging values, preventing context overflow.
4.  **Audit Logging:** Implement a `process.log` that records the success/failure of *every single page* extraction.

## Phase 2.5: Glass Pipeline Diagnostics & Strict Isolation
*Goal: Provide maximum transparency and solve LLM truncation on massive tables.*
1.  **Diagnostic Objects:** Update the backend to return an array of objects per page, containing VLM Raw Output, Cleaned Markdown, and Raw LLM JSON.
2.  **Strict Isolation:** Process every page as an independent unit to bypass `max_tokens` hard limits on large multi-page documents.
3.  **Diagnostics Dashboard:** Overhaul the UI into a three-column white-box view to allow deep inspection of every intermediate pipeline stage.
4.  **Lenient Validation:** Relax field-presence checks (like Vendor Name) for continuation pages to ensure raw data is always visible.

## Phase 2.6: Performance Optimization & Hardware Alignment
*Goal: Solve the "Quantization Bottleneck" and achieve production-grade throughput on T4 GPUs.*
1.  **FP16 Transition:** Identified that AWQ quantization on Tesla T4 creates a decompression bottleneck. Switched to FP16 (Half Precision) to achieve an 8x throughput increase (50-60 tok/s).
2.  **Repetition Control:** Implemented `frequency_penalty=1.1` to eliminate "Degeneration Loops" and ensure stable JSON generation for long tables.
3.  **Resource Calibration:** Finalized VRAM allocation (e.g., 0.40 VLM / 0.35 LLM) to allow high-speed models to coexist on a 16GB T4.

## Phase 3: Engineering-First Faceless Pipeline & Relational CSVs
*Goal: Decouple backend from UI for robust testing and establish a relational data trail.*
1.  **Developer Pipeline (Faceless):** Build a standalone Jupyter Notebook (`developer_pipeline.ipynb`) and/or CLI script to run documents through the pipeline without the UI. This provides bare-metal access to intermediate stages for easier debugging.
2.  **Relational CSV Storage:** Implement a `CSVDatabase` utility to save extractions into linked CSV files (`invoices.csv` for headers/totals and `line_items.csv` for row items), linked by a unique `invoice_id`. This mimics a relational database structure, preparing for SQLite migration.
3.  **Single-Pass Extraction:** Rely on the 4B model's capability for single-pass multi-page extraction, discarding the slower Segmented Extraction (Dual-Prompt) strategy.
4.  **Dynamic Prompt Foundation:** Prepare the architecture to accept dynamic prompts (e.g., Invoice vs. Resume) for future multi-purpose parsing capabilities.

## Phase 4: Robust Error Handling & Recovery
*Goal: Ensure the system is "Unstoppable" by handling failures at every step while maintaining transparency.*
1.  **Try-Except Middleware:** Global error handling in FastAPI to return structured error JSONs to the frontend instead of generic 500 errors, preventing UI freezes.
2.  **"Stuck" Process Cleanup:** Automated background cleanup utility (`src/utils/cleanup.py`) to periodically delete abandoned files in `/uploads` and `/temp` folders to prevent disk space issues after errors.
3.  **Retry Logic (Deferred):** LLM Auto-Retry logic has been intentionally deferred to future updates to ensure the pipeline "fails fast" and broken parts remain visible for developer debugging.

## Phase 5: The Visual Command Center (The "Face")
*Goal: Provide real-time transparency into the automation pipeline.*
1.  **Pipeline Dashboard:** A new UI view with "Nodes" representing Ingestion, VLM, LLM, and Storage.
2.  **Live State Monitoring:** Use WebSockets or Polling to show "Pulsing" animations on the active node.
3.  **Extraction Preview:** Side-by-side view of the "Cleaned OCR Text" vs. "Final JSON" to see exactly how the model thought.

## Phase 6: Agentic Expansion & SQL Persistence
*Goal: Graduate to a "Smart" system with self-correction and professional storage.*
1.  **SQLite Migration:** Move from CSV to a proper SQLite database for fast querying and historical tracking.
2.  **Self-Correction Agent (LangGraph/PydanticAI):** If math validation fails, an agent re-evaluates the page text and attempts a "Correction Loop."
3.  **Multi-Doc Router:** A classifier agent that identifies if the page is an Invoice, Resume, or Receipt and selects the optimized prompt.

---

## Technical Integrity & Compounding
*   **Phase 1-2** ensures we can handle any file size.
*   **Phase 3-4** ensures the data is accurate and the system doesn't crash.
*   **Phase 5-6** turns the tool into a professional "Product" for your portfolio.

Every step builds on the previous one. We will not move to Phase 6 until the Phase 3 persistence is 100% reliable.

---
**Architectural Note (Phase 2 Update):** Based on user review, the multi-page processing strategy was pivoted from "Python-based Merging" to a "Compile All Text" approach. The VLM still runs page-by-page to conserve VRAM, but the resulting Markdown is concatenated into a single, global context string before being sent to the LLM. This prevents "Page 2 Context Blindness," leverages Qwen's 8k token window, and makes the core Python logic completely schema-agnostic for future document types.

---
**Architectural Note (Phase 2.5 Update):** To solve LLM `max_tokens` truncation issues on massive tables and provide maximum transparency, the pipeline was updated to an "Unmerged Page-by-Page" strict isolation model. The `DocumentProcessor` now evaluates each page independently and returns a list of "Diagnostic Objects" (containing VLM Raw, Cleaned Text, and LLM JSON). The `index3.html` UI was overhauled into a "Glass Pipeline Diagnostics Dashboard" to allow the user to inspect every intermediate stage per page. Validation was relaxed to prevent missing continuation data (like Vendor names) from hiding the raw extracted JSON.
