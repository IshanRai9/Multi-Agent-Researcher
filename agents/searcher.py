from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import PromptTemplate
from tools.search import tavily_search
from tools.vector_store import retrieve_chunks
import ast

def searcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Searcher Agent] Analyzing query & gathering info... ---")
    messages = state.get("messages", [])
    user_query = messages[-1] if messages else ""
    
    import re
    # Connect to LLM to extract optimized queries
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert Search Strategist.\n"
        "First, classify the user's query into ONE of the following types:\n"
        "- literature_review\n"
        "- comparison\n"
        "- concept_explanation\n"
        "- implementation\n\n"
        "Then generate EXACTLY THREE targeted search queries using these rules:\n\n"
        "1. Query Decomposition: Break multi-part queries into focused sub-queries.\n"
        "2. Query Rewriting: Use precise keywords and include academic modifiers (e.g., 'paper', 'survey', 'arXiv', 'review').\n"
        "3. Step-Back Prompting: Ensure ONE query is broad and conceptual.\n\n"
        "Additional Rules:\n"
        "- If literature_review: include terms like 'survey', 'review', 'studies', 'urban solar placement LP', and optionally years (e.g., 2020 2024).\n"
        "- If comparison: include BOTH entities in at least TWO queries and use 'vs', 'difference', or 'comparison'.\n"
        "- If concept_explanation: include 'advantages', 'limitations', 'theory'.\n"
        "- If implementation: include 'example', 'code', or 'workflow'.\n\n"
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
    
    # Gather context by routing parallelly across decomposed queries
    combined_context = ""
    for q in optimized_queries[:3]:
        tavily_context = tavily_search(q)
        local_chunks = retrieve_chunks(q, k=2)
        local_context = "\n".join(local_chunks)
        combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n--- LOCAL KNOWLEDGE BASE ---\n{local_context}\n\n"
    
    return {
        "search_queries": optimized_queries,
        "raw_context": combined_context
    }
