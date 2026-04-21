from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def summarizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Summarizer Agent] Summarizing context... ---")
    start_time = datetime.now()
    raw_context = state.get("raw_context", "")
    messages = state.get("messages", [])
    user_query = messages[-1] if messages else ""
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)

    # --- Log header ---
    log_lines = []
    if retry_count > 0:
        log_lines.append(f"\n---\n\n## Summarizer Agent (Retry #{retry_count})\n")
    else:
        log_lines.append(f"\n---\n\n## Summarizer Agent\n")
    log_lines.append(f"**Timestamp:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_lines.append(f"**Input Context Length:** {len(raw_context)} characters\n")

    # Build the base system prompt
    system_prompt = (
        "You are an expert technical summarizer.\n"
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
        "Output ONLY bullet points under these headings."
    )

    # On retry: inject the auditor's feedback so the summarizer can course-correct
    audit_feedback = ""
    reject_errors = [e for e in errors if "REJECT" in e.upper() and e != "PASS"]
    if reject_errors:
        latest_feedback = reject_errors[-1]
        audit_feedback = (
            f"\n\n--- PREVIOUS AUDIT FEEDBACK (RETRY) ---\n"
            f"Your previous summary was REJECTED by the Academic Auditor for the following reasons:\n"
            f"{latest_feedback}\n\n"
            f"You MUST address these specific issues in your new summary:\n"
            f"- Remove or correct any contradicted claims.\n"
            f"- Only include claims that are directly supported by the raw context below.\n"
            f"- Do NOT repeat the same mistakes.\n"
            f"--- END AUDIT FEEDBACK ---"
        )
        print(f"--- [Summarizer Agent] Retry mode: incorporating auditor feedback ---")
        log_lines.append(f"\n### Retry Mode Active\n")
        log_lines.append(f"**Auditor Feedback Injected:**\n> {latest_feedback[:500]}{'...' if len(latest_feedback) > 500 else ''}\n")

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + audit_feedback),
        ("user", "Query: {query}\n\nRaw Context:\n{context}")
    ])
    
    chain = prompt | llm
    summary = chain.invoke({"query": user_query, "context": raw_context}).content.strip()

    elapsed = (datetime.now() - start_time).total_seconds()

    # --- Log output ---
    log_lines.append(f"\n### Generated Summary\n")
    log_lines.append(f"```\n{summary}\n```\n")
    log_lines.append(f"\n**Summary Length:** {len(summary)} characters\n")
    log_lines.append(f"**Summarizer Duration:** {elapsed:.1f}s\n")
    
    return {
        "current_draft": summary,
        "workflow_log": log_lines
    }
