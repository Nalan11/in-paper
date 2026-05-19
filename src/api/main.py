import os
import uuid
import time
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.engines.vlm import VLMEngine
from src.engines.llm import LLMEngine
from src.utils.text_processing import clean_ocr_text
from src.utils.process_manager import ProcessManager
from src.core.document_processor import DocumentProcessor
from src.utils.cleanup import background_cleanup_task

app = FastAPI(title="Invoice Intelligence API")

# --- Status Broadcaster for Pipeline UI ---
class StatusBroadcaster:
    def __init__(self):
        self.listeners = []

    async def broadcast(self, message: str):
        for q in self.listeners:
            await q.put(message)

    async def subscribe(self):
        q = asyncio.Queue()
        self.listeners.append(q)
        try:
            while True:
                msg = await q.get()
                yield dict(data=msg)
        finally:
            self.listeners.remove(q)

broadcaster = StatusBroadcaster()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Global Error: {str(exc)}")
    await broadcaster.broadcast("ERROR")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred during processing.",
            "detail": str(exc),
            "path": request.url.path
        }
    )

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_cleanup_task([UPLOAD_DIR, TEMP_DIR], interval_seconds=3600, max_age_seconds=3600))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
UPLOAD_DIR = "uploads"
LOG_DIR = "logs"
TEMP_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Singleton Engines & Managers ---
vlm_engine = VLMEngine()
llm_engine = LLMEngine()
process_manager = ProcessManager(log_dir=LOG_DIR)
doc_processor = DocumentProcessor(vlm_engine, llm_engine)

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index3.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Frontend index3.html not found in /static</h1>"

@app.get("/stream/pipeline")
async def stream_pipeline(request: Request):
    """SSE endpoint for real-time pipeline visualizer"""
    return EventSourceResponse(broadcaster.subscribe())

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    if not vlm_engine.is_ready():
         raise HTTPException(status_code=503, detail="VLM engine is not initialized or server is down")

    await broadcaster.broadcast("INGESTING")
    start_total = time.time()
    request_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{request_id}{ext}")
    
    # Create request-specific temp dir for PDF pages
    req_temp_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(req_temp_dir, exist_ok=True)

    try:
        # 1. Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        await broadcaster.broadcast("VLM_PROCESSING")
        await asyncio.sleep(0.1) # Yield control to let broadcast go through

        # 2. Process via Core Processor (handles PDF/Image + Merging)
        # We need a small hack to run synchronous doc_processor.process asynchronously or just let it block.
        # For true real-time SSE in FastAPI, long blocking calls freeze the event loop.
        # We use asyncio.to_thread to prevent blocking the SSE stream.
        result_payload = await asyncio.to_thread(doc_processor.process, temp_path, req_temp_dir, broadcaster)
        
        await broadcaster.broadcast("SAVING")
        total_duration = time.time() - start_total

        # Simulate quick saving step
        await asyncio.sleep(0.5)

        await broadcaster.broadcast("DONE")
        return {
            "structured_data": result_payload.get("diagnostic_pages", []),
            "timings": {
                "vlm_sec": round(result_payload.get("total_vlm_sec", 0), 2),
                "llm_sec": round(result_payload.get("total_llm_sec", 0), 2),
                "total_sec": round(total_duration, 2)
            }
        }
    except Exception as e:
        await broadcaster.broadcast("ERROR")
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(req_temp_dir):
            shutil.rmtree(req_temp_dir)

# --- Infrastructure Admin Endpoints ---

@app.get("/admin/status/{server_type}")
async def get_server_status(server_type: str):
    if server_type not in ["vlm", "llm"]:
        raise HTTPException(status_code=400, detail="Invalid server type")
    return {"status": process_manager.get_status(server_type)}

@app.post("/admin/start/{server_type}")
async def start_server(server_type: str):
    if server_type == "vlm":
        success, msg = process_manager.start_server(
            "vlm", "PaddlePaddle/PaddleOCR-VL", 8000, 0.40, 8192
        )
    elif server_type == "llm":
        success, msg = process_manager.start_server(
            "llm", "Qwen/Qwen2.5-1.5B", 8001, 0.30, 2048
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid server type")
    
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return {"message": msg}

@app.post("/admin/stop/{server_type}")
async def stop_server(server_type: str):
    if server_type not in ["vlm", "llm"]:
        raise HTTPException(status_code=400, detail="Invalid server type")
    success, msg = process_manager.stop_server(server_type)
    return {"message": msg, "success": success}

@app.post("/admin/clear_logs/{server_type}")
async def clear_logs(server_type: str):
    if server_type not in ["vlm", "llm"]:
        raise HTTPException(status_code=400, detail="Invalid server type")
    process_manager.clear_logs(server_type)
    return {"message": f"Logs cleared for {server_type}"}

# --- WebSocket Log Streaming ---

@app.websocket("/ws/logs/{server_type}")
async def websocket_logs(websocket: WebSocket, server_type: str):
    await websocket.accept()
    log_file_path = os.path.join(LOG_DIR, f"{server_type}.log")
    
    if not os.path.exists(log_file_path):
        open(log_file_path, 'a').close()

    try:
        # 1. Send History (Last 100 lines)
        with open(log_file_path, "r") as f:
            lines = f.readlines()
            for line in lines[-100:]:
                await websocket.send_text(line)
        
        # 2. Stream New Content
        with open(log_file_path, "r") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                await websocket.send_text(line)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
