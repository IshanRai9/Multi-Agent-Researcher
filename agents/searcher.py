from typing import Any, Dict
from .llm_config import get_llm, strip_thinking_tags
from langchain_core.prompts import PromptTemplate
from tools.search import tavily_search
from datetime import datetime
import ast
import re

def searcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Searcher Agent] Analyzing query & gathering info... ---")
    start_time = datetime.now()
    messages = state.get("messages", [])
    raw_msg = messages[-1] if messages else ""
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

    # Connect to LLM to extract optimized queries
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert Academic Search Strategist. Your job is to take a complex user query "
        "and decompose it into EXACTLY THREE highly optimized search engine queries.\n\n"
        
        "CRITICAL RULES:\n"
        "1. Core Entity Preservation: Identify the main subject of the user's query. That subject MUST be present in all 3 sub-queries to prevent context drift.\n"
        "2. Query Decomposition: Break multi-part questions into isolated, focused searches.\n"
        "3. Academic Tone: Use technical keywords and append terms in the sub-queries like 'arXiv', 'research paper', 'case study', or 'benchmark' if it matches the user query, if not then don't add it.\n"
        
        "EXAMPLES:\n"
        "User Query: What are the main architectural differences between LangGraph and AutoGen for multi-agent LLM orchestration, and which one handles cyclic routing better?\n"
        "Output: [\"LangGraph vs AutoGen multi-agent architecture comparison\", \"LangGraph AutoGen cyclic routing\", \"Multi-agent LLM orchestration cyclic routing\"]\n\n"
        
        "User Query: Explain the use of MILP for optimizing solar panel placement in urban environments with high shading.\n"
        "Output: [\"MILP solar panel placement optimization\", \"Urban environment with high shading MILP\", \"MILP optimization solar panel placement urban environments\"]\n\n"
        
        "You MUST return ONLY a valid Python list of exactly 3 strings. Do not include markdown formatting, explanations, or introductory text.\n\n"
        
        "User Query: {query}\n"
        "Output:"
    )
    chain = prompt | llm
    raw_output = chain.invoke(user_query).content.strip()
    raw_output = strip_thinking_tags(raw_output)

    log_lines.append(f"\n### LLM Sub-Query Generation\n")
    log_lines.append(f"**Raw LLM Output:**\n```\n{raw_output}\n```\n")
    
    # Robust multi-stage parser for SLM outputs
    optimized_queries = None

    # Stage 1: Try standard ast.literal_eval on bracketed list
    try:
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            optimized_queries = ast.literal_eval(match.group(0))
            if not isinstance(optimized_queries, list):
                optimized_queries = None
    except Exception:
        pass

    # Stage 2: Extract individually quoted strings (handles broken brackets)
    if not optimized_queries:
        quoted = re.findall(r'["\']([^"\']{10,})["\']', raw_output)
        if len(quoted) >= 2:
            optimized_queries = quoted[:3]
            print(f"--- [Searcher Agent] Fallback: Extracted {len(optimized_queries)} queries from quoted strings ---")

    # Stage 3: Extract numbered list items (e.g. "1. query text")
    if not optimized_queries:
        numbered = re.findall(r'\d+\.\s*(.+)', raw_output)
        if len(numbered) >= 2:
            optimized_queries = [q.strip().strip('"\'') for q in numbered[:3]]
            print(f"--- [Searcher Agent] Fallback: Extracted {len(optimized_queries)} queries from numbered list ---")

    # Stage 4: Final fallback — generate 3 search variants from the original query
    if not optimized_queries:
        print(f"--- [Searcher Agent] Parsing Failed — using query-derived fallback ---")
        optimized_queries = [
            user_query,
            f"{user_query} research paper arXiv",
            f"{user_query} comparison analysis"
        ]
        log_lines.append(f"**Parsing Failed** -- Using query-derived search variants.\n")
    else:
        log_lines.append(f"**Parsed Sub-Queries:**\n")
        for i, q in enumerate(optimized_queries):
            log_lines.append(f"{i+1}. `{q}`\n")
        
    print(f"--- [Searcher Agent] Formulated Sub-Queries: {optimized_queries} ---")
    
    # Gather context from web search, collecting source URLs
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

        combined_context += f"### Sub-Query Context: {q}\n--- TAVILY WEB SEARCH ---\n{tavily_context}\n\n"
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log_lines.append(f"\n**Total Sources Collected:** {len(all_urls)}\n")
    log_lines.append(f"**Searcher Duration:** {elapsed:.1f}s\n")

    return {
        "search_queries": optimized_queries,
        "raw_context": combined_context,
        "source_urls": all_urls,
        "workflow_log": log_lines
    }
