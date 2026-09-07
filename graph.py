import sqlite3
from typing import TypedDict, List, Optional, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents import (
    get_llm, 
    analyze_document, 
    classify_legal, 
    detect_risks, 
    explain_document,
    review_risks,
    compare_documents
)

# Define the state of our graph
class AgentState(TypedDict):
    text: str
    api_key: Optional[str]
    document_type: Optional[str]
    is_legal: Optional[bool]
    key_clauses: Annotated[List[str], operator.add]
    risks: Annotated[List[str], operator.add]
    summary: Optional[str]
    simplified_explanation: Optional[str]
    classification_reason: Optional[str]
    problematic_explanation: Optional[str]
    critic_reviews: Annotated[List[dict], operator.add]

# Node 1: Document Analyzer
def document_analyzer_node(state: AgentState):
    llm = get_llm(agent_name="document_analyzer", api_key=state["api_key"])
    result = analyze_document(state["text"], llm)
    return {
        "document_type": result.document_type,
        "key_clauses": result.key_clauses
    }

# Node 2: Legal Classifier
def legal_classifier_node(state: AgentState):
    llm = get_llm(agent_name="legal_classifier", api_key=state["api_key"])
    result = classify_legal(state["text"], llm)
    return {
        "is_legal": result.is_legal,
        "classification_reason": result.reason,
        "problematic_explanation": result.explanation
    }

# Node 3: Human Review Interrupt Gate Node
def human_review_node(state: AgentState):
    # Dummy node that acts as a pause barrier
    return {}

# Node 4: Risk Detector
def risk_detector_node(state: AgentState):
    llm = get_llm(agent_name="risk_detector", api_key=state["api_key"])
    result = detect_risks(state["text"], llm, api_key=state["api_key"])
    return {"risks": result.risks}

# Node 5: Explanation Agent
def explanation_agent_node(state: AgentState):
    llm = get_llm(agent_name="explanation_agent", api_key=state["api_key"])
    result = explain_document(state["text"], llm)
    return {
        "summary": result.summary,
        "simplified_explanation": result.simplified_explanation
    }

# Node 6: Critic/Verifier Node
def critic_node(state: AgentState):
    llm = get_llm(agent_name="critic", api_key=state.get("api_key"))
    result = review_risks(state["text"], state.get("risks", []), llm, api_key=state.get("api_key"))
    # Format review list as dictionaries
    reviews_data = [
        {"risk": r.risk, "verdict": r.verdict, "reason": r.reason}
        for r in result.reviews
    ]
    return {"critic_reviews": reviews_data}

# Conditional routing logic post-human review
def decide_post_classification(state: AgentState):
    if state.get("is_legal") is True:
        # Fan-out to Risk Detector and Explainer in parallel
        return ["risk_detector", "explainer"]
    # Bypass and route straight to END
    return "end"

# Define the graph
def create_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyzer", document_analyzer_node)
    workflow.add_node("classifier", legal_classifier_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("risk_detector", risk_detector_node)
    workflow.add_node("explainer", explanation_agent_node)
    workflow.add_node("critic", critic_node)

    # Parallel entry: run Analyzer and Classifier first
    workflow.add_edge(START, "analyzer")
    workflow.add_edge(START, "classifier")

    # Both paths merge into the human review gate
    workflow.add_edge("analyzer", "human_review")
    workflow.add_edge("classifier", "human_review")

    # From the review gate, branch conditionally
    workflow.add_conditional_edges(
        "human_review",
        decide_post_classification,
        {
            "risk_detector": "risk_detector",
            "explainer": "explainer",
            "end": END
        }
    )

    # Risk Detector goes to Critic Node sequentially
    workflow.add_edge("risk_detector", "critic")

    # Critic and Explainer both merge to END
    workflow.add_edge("critic", END)
    workflow.add_edge("explainer", END)

    # Setup memory checkpointer
    memory = MemorySaver()

    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"]
    )

# Compilation
legal_assistant_app = create_graph()

# --- COMPARISON GRAPH FLOW ---

# Define the state for comparison
class ComparisonState(TypedDict):
    text_a: str
    text_b: str
    api_key: Optional[str]
    comparisons: Annotated[List[dict], operator.add]

# Node: Compare Documents
def doc_comparison_node(state: ComparisonState):
    llm = get_llm(agent_name="comparator", api_key=state["api_key"])
    report = compare_documents(state["text_a"], state["text_b"], llm, api_key=state["api_key"])
    comp_list = [
        {
            "original_clause": c.original_clause,
            "modified_clause": c.modified_clause,
            "change_description": c.change_description,
            "risk_direction": c.risk_direction,
            "reason": c.reason
        }
        for c in report.comparisons
    ]
    return {"comparisons": comp_list}

def create_comparison_graph():
    workflow = StateGraph(ComparisonState)
    workflow.add_node("compare", doc_comparison_node)
    workflow.add_edge(START, "compare")
    workflow.add_edge("compare", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

comparison_app = create_comparison_graph()
