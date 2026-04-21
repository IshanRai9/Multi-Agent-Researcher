import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def tavily_search(query: str) -> tuple[str, list[str]]:
    """
    Search the web using Tavily API and return a tuple of:
      (formatted_results_text, list_of_source_urls)
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        return "[Error] Tavily API key is missing or not configured correctly.", []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)
        
        results = response.get("results", [])
        if not results:
            return "No results found.", []
            
        formatted_results = []
        source_urls = []
        for i, res in enumerate(results):
            title = res.get("title", "No Title")
            content = res.get("content", "No Content")
            url = res.get("url", "")
            formatted_results.append(f"{i+1}. {title}: {content}")
            if url:
                source_urls.append(f"[{title}]({url})")
            
        return "\n".join(formatted_results), source_urls
    except Exception as e:
        return f"[Error] External search failed: {str(e)}", []

if __name__ == "__main__":
    text, urls = tavily_search("What is LangGraph?")
    print(text)
    print("\nSource URLs:", urls)
