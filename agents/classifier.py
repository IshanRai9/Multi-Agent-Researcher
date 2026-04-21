from typing import Any, Dict


def classify_query(query: str) -> str:
    """Classify a research query into a category to guide downstream agents."""
    q = query.lower()
    if "literature review" in q:
        return "literature_review"
    elif " vs " in q or "difference" in q:
        return "comparison"
    elif "how to" in q or "implement" in q:
        return "implementation"
    else:
        return "concept_explanation"


def classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-router node that classifies the user query before the Searcher runs."""
    messages = state.get("messages", [])
    user_query = messages[-1] if messages else ""

    query_type = classify_query(user_query)
    print(f"--- [Classifier] Query classified as: {query_type} ---")

    return {"query_type": query_type}
