from langchain_community.chat_models import ChatOllama
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_llm():
    """Returns local ChatOllama model (qwen3-coder:30b with 0 temperature for reliable processing)."""
    return ChatOllama(model="gemma4:e4b", temperature=0)
