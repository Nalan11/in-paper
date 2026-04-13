import os
from paddleocr import PaddleOCRVL, PaddleOCR
import paddle

print("Paddle version:", paddle.__version__)

# Test simple PaddleOCR
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("PaddleOCR initialized successfully")
except Exception as e:
    print("PaddleOCR initialization failed:", e)

# Test PaddleOCRVL
try:
    # Use the model PaddleOCR-VL-1.5 as seen in the directories
    model_path = "/teamspace/studios/this_studio/.paddlex/official_models/PaddleOCR-VL-1.5"
    if os.path.exists(model_path):
        ocr_vl = PaddleOCRVL(model_name_or_path=model_path)
        print("PaddleOCRVL initialized successfully with local model")
    else:
        print(f"Model path {model_path} not found, trying default")
        ocr_vl = PaddleOCRVL(model_name='PaddleOCR-VL-1.5')
        print("PaddleOCRVL initialized successfully")
except Exception as e:
    print("PaddleOCRVL initialization failed:", e)
