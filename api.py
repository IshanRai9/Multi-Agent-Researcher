from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from main import build_graph
from agents.llm_config import check_ollama_health
from tools.vector_store import index_pdf_to_collection, delete_collection
import os
import json
import uuid
import tempfile
import shutil

app = FastAPI(title="Multi-Agent Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    pdf_collection: str = ""

# Temp directory for uploaded PDFs
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

graph = build_graph()

@app.on_event("startup")
async def startup_health_check():
    """Check Ollama connectivity on server startup."""
    if check_ollama_health():
        print("[OK] Ollama is running and reachable.")
    else:
        print("\n" + "=" * 60)
        print("[ERROR] Cannot connect to Ollama!")
        print("   Please start Ollama with: ollama serve")
        print("   Then restart this server.")
        print("=" * 60 + "\n")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it into a session-scoped ChromaDB collection."""
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    # Generate a unique collection name for this upload session
    collection_name = f"upload_{uuid.uuid4().hex[:12]}"

    # Save the uploaded file temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"{collection_name}.pdf")
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Index the PDF into a named collection
        success = index_pdf_to_collection(temp_path, collection_name)

        if success:
            return {"collection": collection_name, "filename": file.filename}
        else:
            return {"error": "Failed to extract or index the PDF content."}
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}
    finally:
        # Clean up the temp PDF file (data stays in ChromaDB)
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/research")
async def perform_research(request: QueryRequest):
    # Pre-flight: check Ollama before starting expensive graph execution
    if not check_ollama_health():
        error_payload = json.dumps({
            "error": "Cannot connect to Ollama. Please start or restart Ollama (`ollama serve`) and try again."
        })
        def error_stream():
            yield f"data: {error_payload}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    pdf_collection = request.pdf_collection

    initial_state = {
        "messages": [f"User query: {request.query}"],
        "query_type": "",
        "search_queries": [],
        "raw_context": "",
        "current_draft": "",
        "final_report": "",
        "errors": [],
        "retry_count": 0,
        "source_urls": [],
        "pdf_collection": pdf_collection
    }

    def event_generator():
        """Sync generator — Starlette runs this in a thread pool automatically."""
        final_state = None
        for event in graph.stream(initial_state):
            for key, state_snapshot in event.items():
                print(f"--- Node Executed: {key} ---")
                final_state = state_snapshot
                # Send SSE event for each completed node
                yield f"data: {json.dumps({'node': key})}\n\n"

        report = final_state.get("final_report", "Research failed.") if final_state else "Error compiling context."
        yield f"data: {json.dumps({'report': report})}\n\n"

        # Clean up the session collection after research completes
        if pdf_collection:
            delete_collection(pdf_collection)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Ensure public dir exists
os.makedirs("public", exist_ok=True)
app.mount("/", StaticFiles(directory="public", html=True), name="public")
