from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate

def summarizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Summarizer Agent] Summarizing context... ---")
    raw_context = state.get("raw_context", "")
    messages = state.get("messages", [])
    user_query = messages[-1] if messages else ""
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical summarizer.\n"
        "Extract structured, high-value insights from the raw context to answer the user's query.\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT use prior knowledge.\n"
        "- If the context is insufficient, return EXACTLY: 'INSUFFICIENT_CONTEXT'\n\n"
        "You MUST organize output into these sections (as bullet points):\n"
        "- Key Findings\n"
        "- Methods / Approaches\n"
        "- Comparisons (if present)\n"
        "- Limitations / Gaps (if present)\n\n"
        "Special Instructions:\n"
        "- If the query is a literature review: group findings by different studies or approaches.\n"
        "- If the query is a comparison: clearly separate Entity A vs Entity B.\n\n"
        "Output ONLY bullet points under these headings."),
        ("user", "Query: {query}\n\nRaw Context:\n{context}")
    ])
    
    chain = prompt | llm
    summary = chain.invoke({"query": user_query, "context": raw_context}).content.strip()
    
    return {"current_draft": summary}
