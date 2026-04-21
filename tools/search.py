import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def tavily_search(query: str) -> str:
    """
    Search the web using Tavily API and return a concise string of top 3 results.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        return "[Error] Tavily API key is missing or not configured correctly."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)
        
        results = response.get("results", [])
        if not results:
            return "No results found."
            
        formatted_results = []
        for i, res in enumerate(results):
            title = res.get("title", "No Title")
            content = res.get("content", "No Content")
            formatted_results.append(f"{i+1}. {title}: {content}")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"[Error] External search failed: {str(e)}"

if __name__ == "__main__":
    print(tavily_search("What is LangGraph?"))
