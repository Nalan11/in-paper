import os
import sys
from src.engines.vlm import VLMEngine

def test_vlm(image_path):
    print(f"--- Testing VLM Component ---")
    print(f"Loading PaddleOCR-VL...")
    
    # Initialize engine (assumes vLLM is running on port 8000)
    try:
        engine = VLMEngine(server_url="http://localhost:8000/v1")
    except Exception as e:
        print(f"Failed to connect to VLM server: {e}")
        print("Ensure 'start_servers.sh' is running or you have manually started vLLM on port 8000.")
        return
        
    print(f"Processing Image: {image_path}")
    try:
        raw_text, duration = engine.predict(image_path)
        print(f"\n--- VLM Raw Output (Duration: {duration:.2f}s) ---")
        print(raw_text)
    except Exception as e:
        print(f"Error during VLM prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_pipeline_vlm.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)
        
    test_vlm(img_path)