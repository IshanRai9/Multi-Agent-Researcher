from langchain_community.chat_models import ChatOllama
import warnings
import requests
import re
warnings.filterwarnings("ignore", category=DeprecationWarning)

OLLAMA_BASE_URL = "http://localhost:11434"

def strip_thinking_tags(text: str) -> str:
    """Remove thinking/reasoning blocks from SLM outputs.
    
    Handles:
    - Standard <think>...</think> blocks
    - Misspelled variants: <thick>, <thinking>, <thinker>, etc.
    - Unclosed tags (model cut off mid-thought)
    - 'Final Answer:' preamble markers some models use after thinking
    """
    # Match any XML-like tag that looks like a thinking block:
    # <think>, <thick>, <thinking>, <thinker>, etc. (case-insensitive)
    cleaned = re.sub(r'<thi\w*>.*?</thi\w*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Handle unclosed thinking tags (model ran out of tokens mid-thought)
    cleaned = re.sub(r'<thi\w*>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Some models prefix with "Final Answer:" after their thinking block
    final_answer_match = re.search(r'\*{0,2}Final\s+Answer:?\*{0,2}\s*', cleaned, flags=re.IGNORECASE)
    if final_answer_match:
        cleaned = cleaned[final_answer_match.end():]
    
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
    """Returns local ChatOllama model (0 temperature for reliable processing).
    num_predict=4096 ensures SLMs have enough token budget for full reports.
    num_ctx=16384 ensures the large raw_context doesn't truncate the system prompt."""
    return ChatOllama(model="phi4-mini-reasoning", temperature=0, num_predict=4096, num_ctx=16384)
