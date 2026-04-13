import os
import uuid
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from paddleocr import PaddleOCRVL

app = FastAPI()

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Global Singletons ---
VLM_SERVER_URL = "http://localhost:8000/v1"
LLM_SERVER_URL = "http://localhost:8001/v1"
MODEL_NAME = "Qwen2.5-1.5B"
UPLOAD_DIR = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize models once at startup (Global)
# These are thread-safe as they primarily handle HTTP requests to the vLLM servers
try:
    print("--- Initializing PaddleOCRVL pipeline ---")
    pipeline_vlm = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=VLM_SERVER_URL,
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL"
    )
    print("--- VLM Initialized ---")
except Exception as e:
    print(f"Failed to initialize VLM pipeline: {e}")
    pipeline_vlm = None

client_llm = OpenAI(base_url=LLM_SERVER_URL, api_key="EMPTY")

SYSTEM_PROMPT = """You are a precise data extraction assistant. 
Extract the information from the user's text and return it strictly as a JSON object. 
If a field is not found, use null. 
Convert currency values (e.g., $100.00) to float numbers (e.g., 100.00). 
Dates should be in DD-MM-YYYY format.

You must output a valid JSON object using exactly this schema:
{
  "Person_name": "string",
  "Company_name": "string",
  "address": "string",
  "contact": "string",
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "subtotal": "float",
  "tax": "float",
  "total": "float"
}"""

# --- Helper Logic from your notebook ---
def extract_and_combine_content(data):
    combined_content = []
    if isinstance(data, list) and data:
        # Assuming the structure has a parsing_res_list within the first item
        if 'parsing_res_list' in data[0] and isinstance(data[0]['parsing_res_list'], list):
            for item in data[0]['parsing_res_list']:
                # Access 'content' as an attribute (as per notebook) or dict key
                content = None
                if hasattr(item, 'content'):
                    content = item.content
                elif isinstance(item, dict):
                    content = item.get('content')
                
                if content is not None:
                    combined_content.append(content)
    return '\n'.join(combined_content)

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    # We will serve the index.html from the static folder
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Frontend index.html not found in /static</h1>"

@app.post("/process")
def process_document(file: UploadFile = File(...)):
    """
    Handles document processing. Uses standard 'def' (synchronous) so FastAPI 
    automatically runs this in a thread pool for concurrent users.
    """
    if pipeline_vlm is None:
         raise HTTPException(status_code=503, detail="VLM pipeline is not initialized")

    # 1. Create unique path for this request
    request_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{request_id}{ext}")

    try:
        # 2. Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. VLM Processing (OCR & Layout)
        results = pipeline_vlm.predict(temp_path)
        extracted_text = extract_and_combine_content(results)

        if not extracted_text.strip():
             return {"error": "No text could be extracted from the document."}

        # 4. LLM Extraction (JSON Formatting)
        response = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text to process:\n{extracted_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        # 5. Parse and Return
        result_json = json.loads(response.choices[0].message.content)
        return {
            "raw_text": extracted_text,
            "structured_data": result_json
        }

    except Exception as e:
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup file for this specific request
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
