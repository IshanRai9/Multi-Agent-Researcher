# Multi-Agent Research Assistant 🤖🕵️‍♂️

A comprehensive Multi-Agent RAG (Retrieval-Augmented Generation) system built using **LangGraph**, **FastAPI**, and **Ollama**. It autonomously researches topics by searching the web and local documents, rigorously fact-checks its findings, and generates well-cited, academic-style reports.

## Features

- **Multi-Agent Orchestration**: Uses LangGraph to route tasks through specialized AI agents.
- **Local LLM Engine**: Powered by `qwen3-coder:30b` via Ollama for deep reasoning and strictly structured output (e.g. JSON), guaranteeing data privacy.
- **Robust Fact-Checking**: The "Ruthless Academic Auditor" agent performs structured logical audits and context verification, preventing hallucinations.
- **Hybrid Search Context**: Supplements web search data (Tavily) with a local knowledge base (HuggingFace Embeddings + ChromaDB).
- **Modern Web UI**: A beautiful frontend powered by FastAPI to visualize the research process and display the final report.

## Agents Layout

1. **Searcher Agent**: Receives a query, decomposes it into targeted sub-queries, and queries Tavily Search + Local DB.
2. **Summarizer Agent**: Extracts raw factual information with zero hallucination.
3. **Fact-Checker Agent**: Validates logical consistency and traces every claim back to the source data.
4. **Writer Agent**: Transforms verified facts into a polished, technical report with inline citations and presents auditor feedback if verification fails.

## Requirements

1. **Ollama**: Download and install [Ollama](https://ollama.com/), and pull the `qwen3-coder:30b` model:
   ```bash
   ollama run qwen3-coder:30b
   ```
   *(Note: This model requires around 20GB of VRAM to run smoothly. Adjust `agents/llm_config.py` to a smaller model like `llama3` if hardware is constrained.)*

2. **API Keys**: You'll need a [Tavily API Key](https://tavily.com/).
   - Copy `.env.example` to `.env` (or create a `.env` file) and add:
     ```env
     TAVILY_API_KEY=your_tavily_api_key_here
     ```

## Installation Setup

1. **Clone the repo** (if applicable) and navigate to the directory:
   ```bash
   cd Multi-Agent-Researcher
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the FastAPI application:

```bash
uvicorn api:app --reload
```

Open your browser to `http://localhost:8000` to interact with the web interface. 

Alternatively, if you want to run a query via command-line:
```bash
python main.py
```

## Structure
- `/agents` - Contains individual nodes for the LangGraph workflow (`searcher.py`, `summarizer.py`, `fact_checker.py`, `writer.py`, `llm_config.py`).
- `/tools` - Integrations for Tavily search (`search.py`) and Local ChromaDB (`vector_store.py`).
- `/public` - Frontend (HTML/JS/CSS) served by FastAPI.
- `/data` - Drop PDF files here to be indexed into local storage.
- `main.py` - Core logic for graph orchestration and local script runs.
- `api.py` - FastAPI application entry point.

## License

This project is open-source. Feel free to modify and expand upon it.
