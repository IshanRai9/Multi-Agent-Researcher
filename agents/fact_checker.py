import json
import re
from typing import Any, Dict
from .llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# Semantic indicators used when the LLM fails to produce valid JSON
_PASS_INDICATORS = [
    "accurate", "correct", "verified", "no contradictions", "consistent",
    "comprehensive", "well-supported", "faithful", "aligns with", "supported by"
]
_REJECT_INDICATORS = [
    "contradict", "fabricat", "unsupported", "hallucin", "inaccurat",
    "inconsisten", "not supported", "false claim", "misleading"
]


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> blocks that some models (Gemma, Qwen) emit before the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _semantic_pass_check(text: str) -> bool:
    """
    When JSON parsing fails, infer PASS/REJECT from the prose.
    Returns True if the text reads like an approval, False if it reads like a rejection.
    """
    text_lower = text.lower()
    reject_score = sum(1 for kw in _REJECT_INDICATORS if kw in text_lower)
    pass_score = sum(1 for kw in _PASS_INDICATORS if kw in text_lower)
    if reject_score > 0 and reject_score >= pass_score:
        return False
    if pass_score > 0:
        return True
    return False


def fact_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- [Fact-Checker Agent] Verifying draft... ---")
    start_time = datetime.now()
    current_draft = state.get("current_draft", "")
    raw_context = state.get("raw_context", "")
    retry_count = state.get("retry_count", 0)

    # --- Log header ---
    log_lines = []
    log_lines.append(f"\n---\n\n## Fact-Checker Agent (Attempt #{retry_count + 1})\n")
    log_lines.append(f"**Timestamp:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_lines.append(f"**Draft Length Under Audit:** {len(current_draft)} characters\n")
    log_lines.append(f"**Context Length for Verification:** {len(raw_context)} characters\n")
    
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
            "CRITICAL: You MUST output ONLY a raw JSON object with NO other text, "
            "NO markdown fences, NO explanation before or after.\n"
            "The JSON schema is:\n"
            "{{\n"
            "  \"status\": \"PASS\" | \"REJECT\",\n"
            "  \"contradictions_found\": [\"...\"],\n"
            "  \"feedback_for_writer\": \"...\"\n"
            "}}"),
        ("user", "[Original Retrieved Context]: {context}\n\n[Generated Draft]: {summary}")
    ])
    
    chain = prompt | llm
    raw_response = chain.invoke({"context": raw_context, "summary": current_draft}).content.strip()

    # Strip <think> tags that some models emit before the actual answer
    response = _strip_thinking_tags(raw_response)

    log_lines.append(f"\n### Raw Auditor Response\n")
    log_lines.append(f"```\n{response[:3000]}{'...(truncated)' if len(response) > 3000 else ''}\n```\n")
    
    try:
        # Robust JSON extraction: find the first { ... } block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            audit_result = json.loads(json_match.group(0))
        else:
            audit_result = json.loads(response)
            
        status = audit_result.get("status", "REJECT").upper()
        contradictions = audit_result.get("contradictions_found", [])
        feedback = audit_result.get("feedback_for_writer", "No feedback provided.")

        log_lines.append(f"\n### Parsed Audit Result\n")
        log_lines.append(f"**Status:** `{status}`\n")
        if contradictions:
            log_lines.append(f"**Contradictions Found:**\n")
            for c in contradictions:
                log_lines.append(f"- {c}\n")
        else:
            log_lines.append(f"**Contradictions Found:** None\n")
        log_lines.append(f"**Feedback:** {feedback}\n")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log_lines.append(f"**Fact-Checker Duration:** {elapsed:.1f}s\n")

        if status == "REJECT":
            error_msg = f"REJECTED by Auditor: {', '.join(contradictions)}. Fix: {feedback}"
            print(f"--- [Fact-Checker Agent] Audit Failed: {status} (attempt {retry_count + 1}) ---")

            log_lines.append(f"\n### Routing Decision\n")
            log_lines.append(f"**Decision:** REJECT -- routing back to Searcher for retry #{retry_count + 1}\n")

            return {"errors": [error_msg], "retry_count": retry_count + 1, "workflow_log": log_lines}
        else:
            print("--- [Fact-Checker Agent] Audit Passed ---")

            log_lines.append(f"\n### Routing Decision\n")
            log_lines.append(f"**Decision:** PASS -- routing to Writer for final report\n")

            return {"errors": ["PASS"], "workflow_log": log_lines}
            
    except Exception as e:
        print(f"--- [Fact-Checker Agent] JSON Parsing Error: {e} ---")
        print(f"--- [Fact-Checker Agent] Raw response (first 200 chars): {response[:200]} ---")

        log_lines.append(f"\n### JSON Parsing Failed\n")
        log_lines.append(f"**Error:** `{e}`\n")

        # Semantic fallback: analyze the prose for pass/reject intent
        if _semantic_pass_check(response):
            print("--- [Fact-Checker Agent] Semantic fallback: detected PASS intent ---")

            elapsed = (datetime.now() - start_time).total_seconds()
            log_lines.append(f"**Semantic Fallback:** Detected PASS intent from prose analysis\n")
            log_lines.append(f"**Fact-Checker Duration:** {elapsed:.1f}s\n")
            log_lines.append(f"\n### Routing Decision\n")
            log_lines.append(f"**Decision:** PASS (semantic) -- routing to Writer\n")

            return {"errors": ["PASS"], "workflow_log": log_lines}
        else:
            print(f"--- [Fact-Checker Agent] Semantic fallback: detected REJECT intent (attempt {retry_count + 1}) ---")

            elapsed = (datetime.now() - start_time).total_seconds()
            log_lines.append(f"**Semantic Fallback:** Detected REJECT intent from prose analysis\n")
            log_lines.append(f"**Fact-Checker Duration:** {elapsed:.1f}s\n")
            log_lines.append(f"\n### Routing Decision\n")
            log_lines.append(f"**Decision:** REJECT (semantic) -- routing back to Searcher for retry #{retry_count + 1}\n")

            return {
                "errors": [f"REJECT: Unstructured audit failure - {response[:200]}"],
                "retry_count": retry_count + 1,
                "workflow_log": log_lines
            }
