from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import PromptTemplate
from tools.search import tavily_search
from tools.vector_store import retrieve_from_collection
from datetime import datetime
import ast
import re

def searcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Searcher Agent] Analyzing query & gathering info... ---")
    start_time = datetime.now()
    messages = state.get("messages", [])
    raw_msg = messages[-1] if messages else ""
    pdf_collection = state.get("pdf_collection", "")
    retry_count = state.get("retry_count", 0)

    # Strip the "User query: " prefix to get the clean query
    user_query = re.sub(r"^User query:\s*", "", raw_msg, flags=re.IGNORECASE)

    # --- Log header ---
    log_lines = []
    if retry_count > 0:
        log_lines.append(f"\n---\n\n## Searcher Agent (Retry #{retry_count})\n")
    else:
        log_lines.append(f"\n---\n\n## Searcher Agent\n")
    log_lines.append(f"**Timestamp:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_lines.append(f"**Input Query:** `{user_query}`\n")
    log_lines.append(f"**PDF Uploaded:** {'Yes (' + pdf_collection + ')' if pdf_collection else 'No'}\n")

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

    log_lines.append(f"\n### LLM Sub-Query Generation\n")
    log_lines.append(f"**Raw LLM Output:**\n```\n{raw_output}\n```\n")
    
    # Safely parse the LLM output into an array
    try:
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            optimized_queries = ast.literal_eval(match.group(0))
        else:
            optimized_queries = ast.literal_eval(raw_output)
            
        if not isinstance(optimized_queries, list):
            optimized_queries = [str(optimized_queries)]
        log_lines.append(f"**Parsed Sub-Queries:**\n")
        for i, q in enumerate(optimized_queries):
            log_lines.append(f"{i+1}. `{q}`\n")
    except Exception as e:
        print(f"--- [Searcher Agent] Parsing Failed: {e} ---")
        optimized_queries = [user_query]
        log_lines.append(f"**Parsing Failed:** `{e}` -- Falling back to original query.\n")
        
    print(f"--- [Searcher Agent] Formulated Sub-Queries: {optimized_queries} ---")
    
    # Gather context by routing across decomposed queries, collecting source URLs
    combined_context = ""
    all_urls = []
    for idx, q in enumerate(optimized_queries[:3]):
        log_lines.append(f"\n### Sub-Query {idx+1}: `{q}`\n")

        tavily_context, urls = tavily_search(q)
        all_urls.extend(urls)

        log_lines.append(f"**Tavily Web Results ({len(urls)} sources):**\n")
        log_lines.append(f"<details>\n<summary>Click to expand web results</summary>\n\n```\n{tavily_context[:2000]}{'...(truncated)' if len(tavily_context) > 2000 else ''}\n```\n</details>\n")

        if urls:
            log_lines.append(f"\n**Source URLs:**\n")
            for u in urls:
                log_lines.append(f"- {u}\n")

        # Only query local knowledge base if a PDF was uploaded for this session
        local_context = ""
        if pdf_collection:
            local_chunks = retrieve_from_collection(q, collection_name=pdf_collection, k=5)
            local_context = "\n".join(local_chunks)

            log_lines.append(f"\n**PDF Chunks Retrieved:** {len(local_chunks)} chunks\n")
            if local_context:
                log_lines.append(f"<details>\n<summary>Click to expand PDF context</summary>\n\n```\n{local_context[:2000]}{'...(truncated)' if len(local_context) > 2000 else ''}\n```\n</details>\n")
                combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n--- UPLOADED DOCUMENT ---\n{local_context}\n\n"
            else:
                combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n"
        else:
            combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n"
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log_lines.append(f"\n**Total Sources Collected:** {len(all_urls)}\n")
    log_lines.append(f"**Searcher Duration:** {elapsed:.1f}s\n")

    if pdf_collection:
        print(f"--- [Searcher Agent] Using uploaded PDF collection: {pdf_collection} ---")
    else:
        print("--- [Searcher Agent] No PDF uploaded, using web search only ---")

    return {
        "search_queries": optimized_queries,
        "raw_context": combined_context,
        "source_urls": all_urls,
        "workflow_log": log_lines
    }
