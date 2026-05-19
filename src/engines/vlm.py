import os
import time
from paddleocr import PaddleOCRVL
from src.utils.text_processing import extract_and_combine_content

class VLMEngine:
    def __init__(self, server_url="http://localhost:8000/v1", model_name="PaddlePaddle/PaddleOCR-VL"):
        self.server_url = server_url
        self.model_name = model_name
        self.pipeline = None
        self._initialize()

    def _initialize(self):
        try:
            print(f"--- Initializing PaddleOCRVL (Backend: {self.server_url}) ---")
            self.pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=self.server_url,
                vl_rec_api_model_name=self.model_name
            )
            print("--- VLM Initialized ---")
        except Exception as e:
            print(f"Failed to initialize VLM pipeline: {e}")
            self.pipeline = None

    def predict(self, image_path):
        if not self.pipeline:
            raise RuntimeError("VLM pipeline not initialized")
        
        start_time = time.time()
        results = self.pipeline.predict(image_path)
        raw_text = extract_and_combine_content(results)
        duration = time.time() - start_time
        
        return raw_text, duration

    def is_ready(self):
        return self.pipeline is not None
