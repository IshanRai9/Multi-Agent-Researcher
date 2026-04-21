from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate

def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Writer Agent] Writing final report... ---")
    current_draft = state.get("current_draft", "")
    errors = state.get("errors", [])
    source_urls = state.get("source_urls", [])
    retry_count = state.get("retry_count", 0)
    
    # Only check the LATEST error entry — errors accumulate via operator.add,
    # so old REJECT entries persist even after the fact-checker passes on retry.
    latest_error = errors[-1] if errors else "PASS"
    is_rejected = "REJECT" in latest_error.upper() and latest_error != "PASS"

    if is_rejected:
        report = (
            "## Research Report -- Verification Failed\n\n"
            f"The Academic Auditor could not verify the research findings after **{retry_count} attempts**.\n\n"
            "### Auditor's Final Feedback\n\n"
            f"> {latest_error}\n\n"
            "### What This Means\n\n"
            "The retrieved sources may contain conflicting information, or the query may be too broad "
            "for the available data to produce a fully verified report.\n\n"
            "### Suggested Next Steps\n\n"
            "1. **Refine your query** -- Try a more specific or narrowly scoped question.\n"
            "2. **Upload a PDF** -- Use the upload button to add relevant documents for RAG.\n"
            "3. **Try a different angle** -- Rephrase the question to focus on a specific aspect.\n"
        )
        return {"final_report": report}
        
    # Build the references section from collected source URLs
    references_context = ""
    if source_urls:
        unique_urls = list(dict.fromkeys(source_urls))  # Deduplicate while preserving order
        refs = "\n".join(f"  [{i+1}] {url}" for i, url in enumerate(unique_urls))
        references_context = (
            f"\n\nAvailable source references for inline citations:\n{refs}\n"
            "Use these as [1], [2], etc. in your report and include a References section at the end."
        )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical writer.\n\n"
        "First, identify the query type:\n"
        "- literature_review\n"
        "- comparison\n"
        "- concept_explanation\n\n"
        "Then write accordingly using the verified summary.\n\n"
        "Rules:\n"
        "- Do NOT hallucinate.\n"
        "- Use only the provided summary.\n"
        "- Include inline citations like [1], [2] using the provided references.\n\n"
        "If literature_review:\n"
        "- Structure:\n"
        "  1. Introduction\n"
        "  2. Review of Existing Work (group by approaches/studies)\n"
        "  3. Comparative Insights\n"
        "  4. Limitations and Research Gaps\n\n"
        "If comparison:\n"
        "- Compare entities point-by-point\n"
        "- End with a clear conclusion\n\n"
        "If concept_explanation:\n"
        "- Stay focused on the exact concept asked\n\n"
        "IMPORTANT:\n"
        "- Do NOT drift into generic explanations\n"
        "- Stay tightly aligned to the query\n"
        "- End with a '## References' section listing all cited sources\n\n"
        "If summary == 'INSUFFICIENT_CONTEXT':\n"
        "Output EXACTLY:\n"
        "'The provided documents and web search did not contain enough information to fully answer your query.'"
        + references_context),
        ("user", "Verified Summary:\n{summary}")
    ])
    
    chain = prompt | llm
    final_report = chain.invoke({"summary": current_draft}).content.strip()
    
    return {"final_report": final_report}
