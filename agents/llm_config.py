from langchain_community.chat_models import ChatOllama
import warnings
import requests
import re
warnings.filterwarnings("ignore", category=DeprecationWarning)

OLLAMA_BASE_URL = "http://localhost:11434"

def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model outputs (phi3, qwen3, etc.).
    Handles multiline content, multiple blocks, and unclosed tags."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also handle unclosed <think> tags (model cut off mid-thought)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

def check_ollama_health() -> bool:
    """Check if Ollama is running and reachable. Returns True if healthy."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False
    except Exception:
        return False

def get_llm():
    """Returns local ChatOllama model (0 temperature for reliable processing)."""
    return ChatOllama(model="phi4-mini-reasoning", temperature=0)
