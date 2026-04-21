from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any
from main import build_graph
import os

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

graph = build_graph()

@app.post("/research")
async def perform_research(request: QueryRequest) -> Dict[str, Any]:
    initial_state = {
        "messages": [f"User query: {request.query}"],
        "query_type": "",
        "search_queries": [],
        "raw_context": "",
        "current_draft": "",
        "final_report": "",
        "errors": []
    }
    
    # Run graph
    final_state = None
    for event in graph.stream(initial_state):
        for key, state_snapshot in event.items():
            print(f"--- Node Executed: {key} ---")
            final_state = state_snapshot

    report = final_state.get("final_report", "Research failed.") if final_state else "Error compiling context."
    return {"report": report}

# Ensure public dir exists
os.makedirs("public", exist_ok=True)
app.mount("/", StaticFiles(directory="public", html=True), name="public")
