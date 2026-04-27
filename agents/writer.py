from typing import Any, Dict
from .llm_config import get_llm, strip_thinking_tags
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Writer Agent] Writing final report... ---")
    start_time = datetime.now()
    current_draft = state.get("current_draft", "")
    errors = state.get("errors", [])
    source_urls = state.get("source_urls", [])
    retry_count = state.get("retry_count", 0)

    # --- Log header ---
    log_lines = []
    log_lines.append(f"\n---\n\n## Writer Agent\n")
    log_lines.append(f"**Timestamp:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_lines.append(f"**Total Retries Before Writing:** {retry_count}\n")
    log_lines.append(f"**Total Source URLs Available:** {len(source_urls)}\n")
    
    # Only check the LATEST error entry — errors accumulate via operator.add,
    # so old REJECT entries persist even after the fact-checker passes on retry.
    latest_error = errors[-1] if errors else "PASS"
    is_rejected = "REJECT" in latest_error.upper() and latest_error != "PASS"

    if is_rejected:
        log_lines.append(f"\n### Circuit Breaker Triggered\n")
        log_lines.append(f"**Reason:** Fact-checker could not verify after {retry_count} retries.\n")
        log_lines.append(f"**Last Error:** {latest_error[:500]}\n")

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
            "2. **Check your sources** -- Ensure relevant information is available online for this topic.\n"
            "3. **Try a different angle** -- Rephrase the question to focus on a specific aspect.\n"
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        log_lines.append(f"**Writer Duration:** {elapsed:.1f}s\n")
        log_lines.append(f"\n### Final Output\n")
        log_lines.append(f"Failure report generated (verification failed).\n")

        return {"final_report": report, "workflow_log": log_lines}
        
    # Build the references section from collected source URLs
    unique_urls = list(dict.fromkeys(source_urls)) if source_urls else []
    references_context = ""
    if unique_urls:
        refs = "\n".join(f"  [{i+1}] {url}" for i, url in enumerate(unique_urls))
        references_context = (
            f"\n\nAvailable source references for inline citations:\n{refs}\n"
            "Use these as [1], [2], etc. in your report and include a References section at the end."
        )

    log_lines.append(f"\n### References Provided to Writer\n")
    if unique_urls:
        for i, url in enumerate(unique_urls):
            log_lines.append(f"{i+1}. {url}\n")
    else:
        log_lines.append(f"No source URLs available.\n")

    # Map output_length selection to word count targets
    output_length = state.get("output_length", "standard")
    length_map = {
        "brief": ("150-200 words", "3-4 paragraphs", "a very short, high-level overview", 150),
        "concise": ("300-400 words", "5-6 paragraphs", "a focused summary hitting only key points", 300),
        "standard": ("500-700 words", "8-10 paragraphs", "a well-structured report with moderate detail", 500),
        "detailed": ("800-1200 words", "12-16 paragraphs", "a thorough, in-depth report covering all aspects", 800),
        "comprehensive": ("1500+ words", "20+ paragraphs", "an exhaustive, fully detailed academic-style report", 1500)
    }
    word_target, para_target, style_desc, min_words = length_map.get(output_length, length_map["standard"])
    length_instruction = (
        f"\n\nOUTPUT LENGTH REQUIREMENT (MANDATORY):\n"
        f"You MUST write {style_desc}.\n"
        f"Target: approximately {word_target} across {para_target}.\n"
        f"MINIMUM word count: {min_words} words. Your output MUST have at least {min_words} words.\n"
        f"Keep writing until you reach at least {min_words} words. Do NOT stop early.\n"
        f"Expand each section with details, examples, and explanations from the summary.\n"
        f"If a section seems short, add more context and analysis from the provided material."
    )

    log_lines.append(f"\n### Output Length Setting\n")
    log_lines.append(f"**Selected:** `{output_length}` ({word_target})\n")

    # Get the original user query to anchor the writer
    messages = state.get("messages", [])
    user_query = ""
    for msg in messages:
        if msg.startswith("User query:"):
            user_query = msg.replace("User query:", "").strip()
            break

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert **Technical Writer**. Your task is to structure the provided verified summary into a **clear, organised report** that directly answers the user's original research query.\n\n"
        "QUERY TO ANSWER: " + user_query + "\n\n"
        "CRITICAL RULES:\n"
        "- Your report MUST directly answer the QUERY above.\n"
        "- Use ONLY information from the verified summary below. Do NOT invent, fabricate, or extrapolate.\n"
        "- Do NOT drift to related but different topics.\n"
        "- Every claim in your report must come from the summary.\n"
        "- Include inline citations like [1], [2] using the provided references.\n\n"
        "First, identify the query type (DO NOT write the query type in the final report):\n"
        "- literature_review\n"
        "- comparison\n"
        "- concept_explanation\n\n"
        "Then write accordingly using the verified summary.\n\n"
        "Formatting Rules:\n"
        "- **Segment** content into appropriate sections with headings.\n"
        "- **Use** headings, subheadings, and lists where necessary.\n"
        "- **Ensure** logical progression and flow.\n"
        "- **Improve** readability without altering the core message.\n"
        "- Do NOT mention you are an AI or LLM.\n"
        "- Do NOT write any line as if you are talking to the user.\n\n"
        "If literature_review:\n"
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
        "- Do NOT drift into generic explanations or tangential topics\n"
        "- Stay tightly aligned to the QUERY above\n"
        "- End with a '## References' section listing all cited sources\n\n"
        "If summary == 'INSUFFICIENT_CONTEXT':\n"
        "Output EXACTLY:\n"
        "'The provided documents and web search did not contain enough information to fully answer your query.'"
        + length_instruction + references_context),
        ("user", "Original Research Query: {query}\n\nVerified Summary:\n{summary}")
    ])
    
    chain = prompt | llm
    raw_final_report = chain.invoke({"query": user_query, "summary": current_draft}).content.strip()
    final_report = strip_thinking_tags(raw_final_report)
    
    # Fallback in case stripping removed absolutely everything (e.g. model only output a think block or got cut off)
    if not final_report and raw_final_report:
        final_report = raw_final_report

    elapsed = (datetime.now() - start_time).total_seconds()

    log_lines.append(f"\n### Final Report Generated\n")
    log_lines.append(f"**Raw Report Length:** {len(raw_final_report)} characters\n")
    log_lines.append(f"**Cleaned Report Length:** {len(final_report)} characters\n")
    log_lines.append(f"**Writer Duration:** {elapsed:.1f}s\n")
    log_lines.append(f"\n### Full Report Content\n")
    log_lines.append(f"\n{final_report}\n")
    
    return {"final_report": final_report, "workflow_log": log_lines}
