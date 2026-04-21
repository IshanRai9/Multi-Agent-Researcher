from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, START, END

# Import nodes
from agents import classifier_node, searcher_node, summarizer_node, fact_checker_node, writer_node

# Define Graph State
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    query_type: str
    search_queries: List[str]
    raw_context: str
    current_draft: str
    final_report: str
    errors: Annotated[List[str], operator.add]

def route_fact_check(state: AgentState) -> str:
    errors = state.get("errors", [])
    if errors and "REJECT" in errors[-1]:
        print(">>> ROUTING: Hallucination flagged! Routing back to Searcher to retry.")
        return "Searcher"
    else:
        print(">>> ROUTING: Verification passed! Routing to Writer.")
        return "Writer"

def build_graph():
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Classifier", classifier_node)
    workflow.add_node("Searcher", searcher_node)
    workflow.add_node("Summarizer", summarizer_node)
    workflow.add_node("FactChecker", fact_checker_node)
    workflow.add_node("Writer", writer_node)
    
    # Define edges
    workflow.add_edge(START, "Classifier")
    workflow.add_edge("Classifier", "Searcher")
    workflow.add_edge("Searcher", "Summarizer")
    workflow.add_edge("Summarizer", "FactChecker")
    
    # Conditonal Edge
    workflow.add_conditional_edges("FactChecker", route_fact_check, {"Searcher": "Searcher", "Writer": "Writer"})
    
    workflow.add_edge("Writer", END)
    
    # Compile Graph
    return workflow.compile()

if __name__ == "__main__":
    import os
    print("Starting Multi-Agent Research Assistant...\n")
    graph = build_graph()
    
    query = "What are the latest advancements in solid-state batteries?"
    
    # Define initial state
    initial_state = {
        "messages": [f"User query: {query}"],
        "query_type": "",
        "search_queries": [],
        "raw_context": "",
        "current_draft": "",
        "final_report": "",
        "errors": []
    }
    
    # Run graph
    print("Invoking graph...\n")
    # Stream the events to trace steps easily
    final_state = None
    for event in graph.stream(initial_state):
        for key, state_snapshot in event.items():
            print(f"--- Node Executed: {key} ---")
            final_state = state_snapshot
            
    # Save the output formally
    os.makedirs("output", exist_ok=True)
    report_path = os.path.join("output", "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Research Query: {query}\n\n")
        if final_state:
            f.write(final_state.get("final_report", "No report generated."))
        
    print(f"\n--- Process Complete! Report saved to {report_path} ---")
