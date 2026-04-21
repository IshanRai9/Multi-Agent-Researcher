from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate

def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Writer Agent] Writing final report... ---")
    current_draft = state.get("current_draft", "")
    errors = state.get("errors", [])
    
    if any("REJECTED by Auditor" in e for e in errors):
        # Find the specific auditor rejection message
        auditor_fail = next((e for e in errors if "REJECTED by Auditor" in e), "Audit failed.")
        return {"final_report": f"## Research Report Blocked\n\n**The Academic Auditor identified issues in the verified context:**\n\n> {auditor_fail}\n\n*Please refine your query or provided documents and try again.*"}
        
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
        "- Include inline citations like [1]. Do NOT include citations unless explicit references are provided.\n\n"
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
        "- Stay tightly aligned to the query\n\n"
        "If summary == 'INSUFFICIENT_CONTEXT':\n"
        "Output EXACTLY:\n"
        "'The provided documents and web search did not contain enough information to fully answer your query.'"),
        ("user", "Verified Summary:\n{summary}")
    ])
    
    chain = prompt | llm
    final_report = chain.invoke({"summary": current_draft}).content.strip()
    
    return {"final_report": final_report}
