import os
import uuid
import json
import shutil
import re
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from paddleocr import PaddleOCRVL
from bs4 import BeautifulSoup

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

SYSTEM_PROMPT = """You are a precise data extraction assistant specialized in financial documents.
Extract information from the provided text and return it strictly as a JSON object.

RULES:
1. DATES: All dates MUST be converted to YYYY-MM-DD format.
2. NUMBERS: Convert currency and quantities to float numbers (e.g., $1,200.50 -> 1200.50).
3. NULLS: If a field is not present in the text, use null.
4. NESTING: Follow the exact nested structure provided below to separate Vendor vs Client details.
5. LINE ITEMS: Extract every row from tables into the line_items array.

TARGET JSON SCHEMA:
{
  "document_details": {
    "document_type": "string",
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD"
  },
  "vendor_details": {
    "company_name": "string",
    "person_name": "string",
    "address": "string",
    "contact_info": "string"
  },
  "client_details": {
    "company_name": "string",
    "person_name": "string",
    "address": "string",
    "contact_info": "string"
  },
  "line_items": [
    {
      "description": "string",
      "quantity": "float",
      "unit_price": "float",
      "line_total": "float"
    }
  ],
  "financials": {
    "subtotal": "float",
    "tax_amount": "float",
    "total_amount": "float"
  }
}"""

# --- Helper Logic ---

def html_table_to_markdown(html_content):
    """Converts HTML <table> to Markdown table using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    markdown_tables = []
    for table in tables:
        rows = table.find_all('tr')
        if not rows: continue
        
        md_rows = []
        for i, row in enumerate(rows):
            cols = row.find_all(['td', 'th'])
            cols_text = [c.get_text(strip=True) for c in cols]
            md_rows.append("| " + " | ".join(cols_text) + " |")
            
            # Add separator after header
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
        
        markdown_tables.append("\n".join(md_rows))
    
    return "\n\n".join(markdown_tables) if markdown_tables else html_content

def clean_ocr_text(text):
    """Cleans OCR text: removes img tags and converts tables."""
    # 1. Remove <img ...> tags
    text = re.sub(r'<img[^>]*>', '', text)
    
    # 2. Extract <table> contents and convert to markdown
    def table_replacer(match):
        return html_table_to_markdown(match.group(0))
    
    cleaned_text = re.sub(r'<table>.*?</table>', table_replacer, text, flags=re.DOTALL)
    
    # 3. Clean up excessive whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def extract_and_combine_content(data):
    combined_content = []
    if isinstance(data, list) and data:
        if 'parsing_res_list' in data[0] and isinstance(data[0]['parsing_res_list'], list):
            for item in data[0]['parsing_res_list']:
                content = None
                if hasattr(item, 'content'):
                    content = item.content
                elif isinstance(item, dict):
                    content = item.get('content')
                
                if content is not None:
                    combined_content.append(content)
    return '\n'.join(combined_content)

def attempt_json_recovery(truncated_json_str):
    """Attempts to close a truncated JSON string for partial extraction."""
    temp_str = truncated_json_str.strip()
    for _ in range(5):
        try:
            return json.loads(temp_str)
        except json.JSONDecodeError:
            if temp_str.endswith('"'): temp_str += ' }'
            elif temp_str.endswith(','): temp_str = temp_str[:-1] + ' }'
            else: temp_str += ' }'
    return {"requires_human_review": True, "error": "JSON Truncated"}

def ensure_structure(data):
    """Ensures the extracted JSON has all required top-level keys to avoid frontend crashes."""
    defaults = {
        "document_details": {},
        "vendor_details": {},
        "client_details": {},
        "line_items": [],
        "financials": {},
        "requires_human_review": False,
        "validation_errors": []
    }
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
    return data

def validate_extraction(data):
    """Validates the extracted JSON data for errors and mathematical consistency."""
    data = ensure_structure(data)
    issues = []
    
    financials = data.get("financials", {})
    subtotal = financials.get("subtotal") or 0.0
    tax = financials.get("tax_amount") or 0.0
    total = financials.get("total_amount") or 0.0
    
    # Math Check
    expected_total = subtotal + tax
    if abs(expected_total - total) > 0.02:
        issues.append(f"Math mismatch: Subtotal({subtotal}) + Tax({tax}) != Total({total})")
    
    # Critical Fields Check
    if not data.get("vendor_details", {}).get("company_name"):
        issues.append("Missing Vendor Name")
    
    if issues:
        data["requires_human_review"] = True
        data["validation_errors"] = issues
    else:
        # Preserve existing flag if recovery failed
        data["requires_human_review"] = data.get("requires_human_review", False)
        
    return data

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index2.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Frontend index2.html not found in /static</h1>"

@app.post("/process")
def process_document(file: UploadFile = File(...)):
    if pipeline_vlm is None:
         raise HTTPException(status_code=503, detail="VLM pipeline is not initialized")

    start_total = time.time()
    request_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{request_id}{ext}")

    try:
        # 1. Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. VLM Step
        vlm_start = time.time()
        results = pipeline_vlm.predict(temp_path)
        raw_text = extract_and_combine_content(results)
        vlm_duration = time.time() - vlm_start

        if not raw_text.strip():
             return {"error": "No text could be extracted from the document."}

        # 3. Pre-processing
        cleaned_text = clean_ocr_text(raw_text)

        # 4. LLM Step
        llm_start = time.time()
        response = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nBe as concise as possible to avoid truncation."},
                {"role": "user", "content": f"Text to process:\n{cleaned_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        try:
            result_json = json.loads(raw_content)
        except json.JSONDecodeError:
            result_json = attempt_json_recovery(raw_content)
            result_json["requires_human_review"] = True
            if "validation_errors" not in result_json:
                result_json["validation_errors"] = []
            result_json["validation_errors"].append("LLM output was truncated/incomplete")

        llm_duration = time.time() - llm_start

        # 5. Validation Gate
        result_json = validate_extraction(result_json)
        total_duration = time.time() - start_total

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "structured_data": result_json,
            "timings": {
                "vlm_sec": round(vlm_duration, 2),
                "llm_sec": round(llm_duration, 2),
                "total_sec": round(total_duration, 2)
            }
        }

    except Exception as e:
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
