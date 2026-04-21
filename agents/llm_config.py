from langchain_community.chat_models import ChatOllama
import warnings
import requests
warnings.filterwarnings("ignore", category=DeprecationWarning)

OLLAMA_BASE_URL = "http://localhost:11434"

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
    return ChatOllama(model="gemma4:e4b", temperature=0)
