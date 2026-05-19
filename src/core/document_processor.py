import asyncio
import os
import logging
from src.utils.document_loader import split_pdf_to_images
from src.utils.text_processing import clean_ocr_text

# Configure Audit Logging
logging.basicConfig(
    filename='process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentProcessor:
    def __init__(self, vlm_engine, llm_engine):
        self.vlm = vlm_engine
        self.llm = llm_engine

    def _safe_broadcast(self, broadcaster, message):
        if broadcaster:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcaster.broadcast(message), loop)
            else:
                # If running in developer_pipeline, no active event loop might be available
                pass

    def process(self, file_path, temp_dir, broadcaster=None):
        """
        Processes a document page-by-page. Returns an array of Diagnostic Objects
        for the "Glass Pipeline" UI to provide maximum transparency per page.
        """
        ext = os.path.splitext(file_path)[1].lower()
        image_paths = []
        
        if ext == '.pdf':
            logging.info(f"Processing PDF: {file_path}")
            image_paths = split_pdf_to_images(file_path, temp_dir)
        else:
            logging.info(f"Processing Image: {file_path}")
            image_paths = [file_path]

        diagnostic_pages = []
        total_vlm_duration = 0.0
        total_llm_duration = 0.0
        
        for i, img_path in enumerate(image_paths):
            page_num = i + 1
            logging.info(f"Processing page {page_num}/{len(image_paths)}")
            
            diagnostic_obj = {
                "page_number": page_num,
                "stage_1_vlm_raw": "",
                "stage_2_cleaned": "",
                "stage_3_llm_prompt": [],
                "stage_4_llm_raw_string": "",
                "stage_5_llm_json": {}
            }
            
            try:
                # 1. VLM Step
                self._safe_broadcast(broadcaster, f"VLM_PROCESSING_PAGE_{page_num}")
                raw_text, vlm_duration = self.vlm.predict(img_path)
                total_vlm_duration += vlm_duration
                diagnostic_obj["stage_1_vlm_raw"] = raw_text
                
                # 2. Pre-processing
                cleaned_text = clean_ocr_text(raw_text)
                diagnostic_obj["stage_2_cleaned"] = cleaned_text
                
                # 3. LLM Step (Per-page)
                self._safe_broadcast(broadcaster, f"LLM_EXTRACTION_PAGE_{page_num}")
                page_json, llm_duration, diagnostics = self.llm.extract(cleaned_text)
                total_llm_duration += llm_duration
                
                diagnostic_obj["stage_3_llm_prompt"] = diagnostics.get("prompt_messages", [])
                diagnostic_obj["stage_4_llm_raw_string"] = diagnostics.get("raw_output_string", "")
                diagnostic_obj["stage_5_llm_json"] = page_json
                
                logging.info(f"Page {page_num} processed successfully")
                
            except Exception as e:
                logging.error(f"Error processing page {page_num}: {e}")
                diagnostic_obj["stage_5_llm_json"] = {
                    "requires_human_review": True,
                    "validation_errors": [f"Pipeline failed on page {page_num}: {str(e)}"]
                }
            
            diagnostic_pages.append(diagnostic_obj)

        return {
            "diagnostic_pages": diagnostic_pages,
            "total_vlm_sec": total_vlm_duration,
            "total_llm_sec": total_llm_duration
        }
