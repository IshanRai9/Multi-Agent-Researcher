from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import PromptTemplate
from tools.search import tavily_search
from tools.vector_store import retrieve_from_collection
import ast
import re

def searcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Searcher Agent] Analyzing query & gathering info... ---")
    messages = state.get("messages", [])
    raw_msg = messages[-1] if messages else ""
    pdf_collection = state.get("pdf_collection", "")

    # Strip the "User query: " prefix to get the clean query
    user_query = re.sub(r"^User query:\s*", "", raw_msg, flags=re.IGNORECASE)

    # Connect to LLM to extract optimized queries
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert Search Strategist.\n"
        "Generate EXACTLY THREE targeted search queries for the user's question using these rules:\n\n"
        "1. Query Decomposition: Break multi-part queries into focused sub-queries.\n"
        "2. Query Rewriting: Use precise keywords and include academic modifiers (e.g., 'paper', 'survey', 'arXiv', 'review').\n"
        "3. Step-Back Prompting: Ensure ONE query is broad and conceptual.\n\n"
        "You MUST return ONLY a valid Python list of exactly 3 strings and nothing else.\n\n"
        "Query: {query}\n"
        "Search Phrases List:"
    )
    chain = prompt | llm
    raw_output = chain.invoke(user_query).content.strip()
    
    # Safely parse the LLM output into an array
    try:
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            optimized_queries = ast.literal_eval(match.group(0))
        else:
            optimized_queries = ast.literal_eval(raw_output)
            
        if not isinstance(optimized_queries, list):
            optimized_queries = [str(optimized_queries)]
    except Exception as e:
        print(f"--- [Searcher Agent] Parsing Failed: {e} ---")
        optimized_queries = [user_query] # graceful fallback
        
    print(f"--- [Searcher Agent] Formulated Sub-Queries: {optimized_queries} ---")
    
    # Gather context by routing across decomposed queries, collecting source URLs
    combined_context = ""
    all_urls = []
    for q in optimized_queries[:3]:
        tavily_context, urls = tavily_search(q)
        all_urls.extend(urls)

        # Only query local knowledge base if a PDF was uploaded for this session
        local_context = ""
        if pdf_collection:
            local_chunks = retrieve_from_collection(q, collection_name=pdf_collection, k=2)
            local_context = "\n".join(local_chunks)
            if local_context:
                combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n--- UPLOADED DOCUMENT ---\n{local_context}\n\n"
            else:
                combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n"
        else:
            combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n"
    
    if pdf_collection:
        print(f"--- [Searcher Agent] Using uploaded PDF collection: {pdf_collection} ---")
    else:
        print("--- [Searcher Agent] No PDF uploaded, using web search only ---")

    return {
        "search_queries": optimized_queries,
        "raw_context": combined_context,
        "source_urls": all_urls
    }
