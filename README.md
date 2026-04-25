# Multi-Agent Research Assistant 

A Multi-Agent Research system built using **LangGraph**, **FastAPI**, and **Ollama**. It autonomously researches topics by searching the web, rigorously fact-checks its findings, and generates well-cited, academic-style reports.

## Features

- **Multi-Agent Orchestration**: Uses LangGraph to route tasks through specialized AI agents with automatic retry and circuit-breaking.
- **Local LLM Engine**: Runs any Ollama-compatible model locally for deep reasoning and structured output, guaranteeing data privacy.
- **Robust Fact-Checking**: The "Ruthless Academic Auditor" agent performs structured logical audits with feedback-driven retries.
- **Web Search Context**: Advanced web search via Tavily API with synthesized answers and rich content extraction.
- **Source Citations**: Collects and preserves source URLs for inline citations and a references section.
- **Workflow Report**: Every research run generates a detailed `output/report.md` documenting every agent action from start to end.
- **Real-time Telemetry**: SSE-powered frontend shows agents activating step-by-step as the research progresses.

## Agents Layout

1. **Searcher Agent**: Decomposes the query into targeted sub-queries, queries Tavily Search (advanced depth), and collects source URLs.
2. **Summarizer Agent**: Extracts structured, factual information. On retries, incorporates auditor feedback to avoid repeating mistakes.
3. **Fact-Checker Agent**: Validates logical consistency and traces every claim back to source data. Rejects fabricated or contradicted claims.
4. **Writer Agent**: Transforms verified facts into a polished report with inline citations and a References section. Reports auditor concerns if verification fails after max retries.

## Requirements

1. **Ollama**: Download and install [Ollama](https://ollama.com/), and pull any model of your choice:
   ```bash
   ollama pull <model_name>
   ```
   **Any Ollama-compatible model works.** Change the model name in `agents/llm_config.py`. Popular options:

   | Model | Size | VRAM Needed | Best For |
   |-------|------|-------------|----------|
   | `llama3` | 8B | ~6GB | Fast, general-purpose |
   | `mistral` | 7B | ~6GB | Strong reasoning |
   | `gemma2` | 9B | ~7GB | Good balance |
   | `phi3` | 3.8B | ~3GB | Lightweight / low-end hardware |
   | `qwen2.5-coder` | 7B | ~6GB | Code-heavy queries |
   | `gemma4:e4b` | -- | ~16GB | Current default |
   | `qwen3-coder:30b` | 30B | ~20GB | Deep reasoning (high-end GPU) |

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
- `/tools` - Tavily web search integration (`search.py`).
- `/public` - Frontend (HTML/JS/CSS) served by FastAPI.
- `/output` - Generated workflow reports (`report.md`) documenting every agent action.
- `main.py` - Core logic for graph orchestration and local script runs.
- `api.py` - FastAPI application entry point.

## License

This project is open-source. Feel free to modify and expand upon it.
