import asyncio
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.document_loader import split_pdf_to_images
from src.utils.text_processing import clean_ocr_text

# Configure Audit Logging
logging.basicConfig(
    filename='process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentProcessor:
    def __init__(self, vlm_engine, llm_engine, prompts_dir="src/prompts"):
        self.vlm = vlm_engine
        self.llm = llm_engine
        self.prompts_dir = prompts_dir
        self.progress_lock = threading.Lock()

    def _safe_broadcast(self, broadcaster, message, loop):
        if broadcaster and loop:
            try:
                asyncio.run_coroutine_threadsafe(broadcaster.broadcast(message), loop)
            except Exception as e:
                logging.error(f"Broadcasting error: {e}")

    def _load_prompt(self, doc_type):
        """Loads prompt text from the prompts directory."""
        path = os.path.join(self.prompts_dir, f"{doc_type.lower()}.txt")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        # Fallback to invoice if not found
        with open(os.path.join(self.prompts_dir, "invoice.txt"), 'r') as f:
            return f.read()

    def classify_document(self, text):
        """Uses the LLM to classify the document type based on OCR text."""
        classification_prompt = """Analyze the following OCR text from a document. 
    Determine if this is an INVOICE, a RESUME, or a RECEIPT. 
    Return only the single word category name in uppercase. 
    If unsure, return INVOICE."""

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model_name,
                messages=[
                    {"role": "system", "content": classification_prompt},
                    {"role": "user", "content": f"Text to classify (first 1000 chars):\n{text[:1000]}"}
                ],
                temperature=0.0
            )
            raw_answer = response.choices[0].message.content

            # STRIP <think> tags: Removes reasoning monologue if model outputs it
            clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip().upper()

            # Extract just the first word in case it outputted more
            final_word = clean_answer.split()[0] if clean_answer else "INVOICE"
            return final_word
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            return "INVOICE"

    def process_vlm_page(self, page_num, img_path, total_pages, broadcaster, loop, counters):
        """Task 1: Run VLM on a single page."""
        try:
            raw_text, v_dur = self.vlm.predict(img_path)
            with self.progress_lock:
                counters['vlm'] += 1
                self._safe_broadcast(broadcaster, f"VLM_PROGRESS_{counters['vlm']}_{total_pages}", loop)
            return page_num, raw_text, v_dur
        except Exception as e:
            logging.error(f"VLM failed on page {page_num}: {e}")
            return page_num, None, 0.0

    def process_llm_page(self, page_num, cleaned_text, total_pages, system_prompt, broadcaster, loop, counters):
        """Task 2: Run LLM on a single page."""
        try:
            page_json, l_dur, diagnostics = self.llm.extract(cleaned_text, system_prompt)
            with self.progress_lock:
                counters['llm'] += 1
                self._safe_broadcast(broadcaster, f"LLM_PROGRESS_{counters['llm']}_{total_pages}", loop)
            return page_num, page_json, l_dur, diagnostics
        except Exception as e:
            logging.error(f"LLM failed on page {page_num}: {e}")
            return page_num, None, 0.0, {}

    def process(self, file_path, temp_dir, broadcaster=None, loop=None):
        """
        Processes a document using an asynchronous vision-first parallel pipeline.
        1. All VLM runs in parallel.
        2. Router classifies doc based on Page 1.
        3. All LLMs run in parallel using the dynamic prompt.
        """
        ext = os.path.splitext(file_path)[1].lower()
        image_paths = split_pdf_to_images(file_path, temp_dir) if ext == '.pdf' else [file_path]
        total_pages = len(image_paths)
        self._safe_broadcast(broadcaster, f"TOTAL_PAGES_{total_pages}", loop)

        # Shared stats
        counters = {'vlm': 0, 'llm': 0}
        vlm_texts = {}
        total_vlm_sec = 0.0
        total_llm_sec = 0.0

        # --- STAGE 1: PARALLEL VLM ---
        with ThreadPoolExecutor(max_workers=total_pages) as executor:
            futures = [executor.submit(self.process_vlm_page, i+1, path, total_pages, broadcaster, loop, counters) 
                       for i, path in enumerate(image_paths)]
            for future in as_completed(futures):
                p_num, text, dur = future.result()
                vlm_texts[p_num] = text
                total_vlm_sec += dur

        # --- STAGE 2: CLASSIFICATION & PROMPT LOADING ---
        # Classify based on the first available text (usually Page 1)
        first_page_text = vlm_texts.get(1, "")
        doc_type = self.classify_document(first_page_text)
        logging.info(f"Document classified as: {doc_type}")
        system_prompt = self._load_prompt(doc_type)

        # --- STAGE 3: PARALLEL LLM ---
        diagnostic_pages = []
        llm_results = {}
        with ThreadPoolExecutor(max_workers=total_pages) as executor:
            futures = []
            for p_num, raw_text in vlm_texts.items():
                if raw_text:
                    cleaned = clean_ocr_text(raw_text)
                    futures.append(executor.submit(self.process_llm_page, p_num, cleaned, total_pages, system_prompt, broadcaster, loop, counters))
                else:
                    # Handle VLM failure case
                    llm_results[p_num] = ({"requires_human_review": True}, 0.0, {})

            for future in as_completed(futures):
                p_num, res_json, dur, diag = future.result()
                llm_results[p_num] = (res_json, dur, diag)
                total_llm_sec += dur

        # --- STAGE 4: ASSEMBLE DIAGNOSTIC OBJECTS ---
        for i in range(total_pages):
            p_num = i + 1
            res_json, _, diag = llm_results.get(p_num, ({}, 0.0, {}))
            raw_vlm = vlm_texts.get(p_num, "")
            diagnostic_pages.append({
                "page_number": p_num,
                "stage_1_vlm_raw": raw_vlm,
                "stage_2_cleaned": clean_ocr_text(raw_vlm) if raw_vlm else "",
                "stage_3_llm_prompt": diag.get("prompt_messages", []),
                "stage_4_llm_raw_string": diag.get("raw_output_string", ""),
                "stage_5_llm_json": res_json or {"error": "Processing failed"}
            })

        return {
            "diagnostic_pages": diagnostic_pages,
            "total_vlm_sec": total_vlm_sec,
            "total_llm_sec": total_llm_sec,
            "doc_type": doc_type
        }
