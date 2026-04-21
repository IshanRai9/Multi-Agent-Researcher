import json
import re
from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate

def fact_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Fact-Checker Agent] Verifying draft... ---")
    current_draft = state.get("current_draft", "")
    raw_context = state.get("raw_context", "")
    retry_count = state.get("retry_count", 0)
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Role: You are a ruthless, highly critical Academic Auditor.\n\n"
            "Task: You will be given [Original Retrieved Context] and a [Generated Draft].\n"
            "Your job is to verify factual correctness while allowing reasonable synthesis.\n\n"
            "Instructions:\n"
            "1. CLAIM EXTRACTION: List all core technical claims.\n"
            "2. LOGICAL AUDIT: Check internal consistency.\n"
            "3. CONTEXT VERIFICATION: Validate claims against context.\n\n"
            "ALLOWABLE:\n"
            "- Logical synthesis or summarization of multiple points.\n"
            "- Generalization IF it does not introduce new factual claims.\n\n"
            "REJECT IF:\n"
            "- Any claim contradicts the context.\n"
            "- A claim is clearly unsupported or fabricated.\n\n"
            "Output ONLY a valid JSON:\n"
            "{{\n"
            "  \"status\": \"PASS\" | \"REJECT\",\n"
            "  \"contradictions_found\": [\"...\"],\n"
            "  \"feedback_for_writer\": \"...\"\n"
            "}}"),
        ("user", "[Original Retrieved Context]: {context}\n\n[Generated Draft]: {summary}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": raw_context, "summary": current_draft}).content.strip()
    
    try:
        # Robust JSON extraction block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            audit_result = json.loads(json_match.group(0))
        else:
            audit_result = json.loads(response)
            
        status = audit_result.get("status", "REJECT").upper()
        contradictions = audit_result.get("contradictions_found", [])
        feedback = audit_result.get("feedback_for_writer", "No feedback provided.")
        
        if status == "REJECT":
            error_msg = f"REJECTED by Auditor: {', '.join(contradictions)}. Fix: {feedback}"
            print(f"--- [Fact-Checker Agent] Audit Failed: {status} (attempt {retry_count + 1}) ---")
            return {"errors": [error_msg], "retry_count": retry_count + 1}
        else:
            print("--- [Fact-Checker Agent] Audit Passed ---")
            return {"errors": ["PASS"]}
            
    except Exception as e:
        print(f"--- [Fact-Checker Agent] Parsing Error: {e}. Raw Response: {response[:100]}... ---")
        # Fallback: conservative REJECT if parsing fails — garbled output should not silently pass
        if "PASS" in response.upper() and "REJECT" not in response.upper():
            return {"errors": ["PASS"]}
        return {"errors": [f"REJECT: Unstructured audit failure - {response[:200]}"], "retry_count": retry_count + 1}
