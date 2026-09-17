import sqlite3
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents import (
    get_llm,
    classify_legal,
    analyze_document,
    detect_risks,
    review_risks,
    explain_document,
    compare_documents,
    LegalClassification,
    DocumentAnalysis,
    RiskAssessment,
    CriticVerdict,
    DocumentExplanation,
    ComparisonReport
)


# =====================================================================
# AGENT STATE SCHEMAS
# =====================================================================

class AgentState(TypedDict):
    text: str
    api_key: Optional[str]
    file_metadata: Optional[Dict[str, Any]]
    is_legal: Optional[bool]
    document_type: Optional[str]
    classification_reason: Optional[str]
    classification_confidence: Optional[float]
    problematic_explanation: Optional[str]
    document_analysis: Optional[Dict[str, Any]]
    key_clauses: Annotated[List[str], operator.add]
    risks: Annotated[List[Dict[str, Any]], operator.add]
    critic_reviews: Annotated[List[Dict[str, Any]], operator.add]
    summary: Optional[str]
    simplified_explanation: Optional[str]
    risk_explanations: Annotated[List[Dict[str, Any]], operator.add]
    human_reviews: Annotated[List[Dict[str, Any]], operator.add]
    final_report: Optional[Dict[str, Any]]


# =====================================================================
# GRAPH NODES
# =====================================================================

def legal_classifier_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates document legally and determines whether to route to analysis or terminate."""
    llm = get_llm(agent_name="legal_classifier", api_key=state.get("api_key"))
    res = classify_legal(state["text"], llm=llm, api_key=state.get("api_key"))
    return {
        "is_legal": res.is_legal,
        "document_type": res.document_type,
        "classification_reason": res.reason,
        "classification_confidence": res.confidence,
        "problematic_explanation": res.explanation
    }


def non_legal_explainer_node(state: AgentState) -> Dict[str, Any]:
    """Generates a brief summary explaining why a non-legal document does not require contract risk analysis."""
    doc_type = state.get("document_type", "Non-Legal Document")
    reason = state.get("classification_reason", "Identified as non-contractual content.")
    explanation = state.get("problematic_explanation") or f"This document is classified as a {doc_type}. It does not establish legally binding contract obligations or legal liability requiring automated contract risk assessment."
    
    return {
        "summary": f"Document classified as {doc_type}: {reason}",
        "simplified_explanation": explanation,
        "risks": [],
        "critic_reviews": [],
        "key_clauses": []
    }


def document_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """Extracts document structure, parties, key dates, obligations, and omitted standard safeguards."""
    llm = get_llm(agent_name="document_analyzer", api_key=state.get("api_key"))
    analysis = analyze_document(state["text"], llm=llm, api_key=state.get("api_key"))
    
    analysis_dict = analysis.model_dump()
    resolved_type = analysis.document_type
    if not resolved_type or resolved_type in ["Contract / Document", "Legal Document", "Contract", "Document", "N/A"]:
        resolved_type = state.get("document_type") or resolved_type or "Contract / Document"
        
    return {
        "document_type": resolved_type,
        "document_analysis": analysis_dict,
        "key_clauses": analysis.key_clauses
    }


def risk_detector_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves standard legal baseline clauses via RAG and identifies grounded contract risks."""
    llm = get_llm(agent_name="risk_detector", api_key=state.get("api_key"))
    assessment = detect_risks(state["text"], llm=llm, api_key=state.get("api_key"))
    
    risks_list = [r.model_dump() for r in assessment.risks]
    return {"risks": risks_list}


def critic_node(state: AgentState) -> Dict[str, Any]:
    """Independently audits candidate risks, validates evidence support, and filters false positives."""
    llm = get_llm(agent_name="critic", api_key=state.get("api_key"))
    risks_data = state.get("risks", [])
    
    verdict = review_risks(state["text"], risks_data, llm=llm, api_key=state.get("api_key"))
    reviews_list = [r.model_dump() for r in verdict.reviews]
    return {"critic_reviews": reviews_list}


def explanation_agent_node(state: AgentState) -> Dict[str, Any]:
    """Generates plain-English summaries and explanations of validated risks for non-lawyers."""
    llm = get_llm(agent_name="explanation_agent", api_key=state.get("api_key"))
    risks_data = state.get("risks", [])
    
    explanation = explain_document(state["text"], risks=risks_data, llm=llm, api_key=state.get("api_key"))
    return {
        "summary": explanation.summary,
        "simplified_explanation": explanation.simplified_explanation,
        "risk_explanations": [item.model_dump() for item in explanation.risk_explanations]
    }


def human_review_node(state: AgentState) -> Dict[str, Any]:
    """Human-in-the-Loop breakpoint node where reviewer can accept, reject, or annotate findings."""
    return {}


def final_report_compiler_node(state: AgentState) -> Dict[str, Any]:
    """Compiles the verified AI findings, critic verdicts, and human decisions into a final report object."""
    report = {
        "document_type": state.get("document_type"),
        "is_legal": state.get("is_legal"),
        "classification_reason": state.get("classification_reason"),
        "confidence": state.get("classification_confidence"),
        "document_analysis": state.get("document_analysis"),
        "key_clauses": state.get("key_clauses"),
        "risks": state.get("risks"),
        "critic_reviews": state.get("critic_reviews"),
        "summary": state.get("summary"),
        "simplified_explanation": state.get("simplified_explanation"),
        "human_reviews": state.get("human_reviews", [])
    }
    return {"final_report": report}


# =====================================================================
# CONDITIONAL ROUTING
# =====================================================================

def route_after_classification(state: AgentState) -> str:
    """Routes valid legal documents to full multi-agent pipeline and non-legal items to fast exit."""
    if state.get("is_legal") is True:
        return "analyzer"
    return "non_legal_explainer"


# =====================================================================
# GRAPH COMPILATION
# =====================================================================

def create_graph():
    """Builds and compiles the LegalValidate AI Directed Acyclic Graph with HITL interruption."""
    workflow = StateGraph(AgentState)

    # 1. Add all functional nodes
    workflow.add_node("classifier", legal_classifier_node)
    workflow.add_node("non_legal_explainer", non_legal_explainer_node)
    workflow.add_node("analyzer", document_analyzer_node)
    workflow.add_node("risk_detector", risk_detector_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("explainer", explanation_agent_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("final_compiler", final_report_compiler_node)

    # 2. Graph Edges
    workflow.add_edge(START, "classifier")
    
    workflow.add_conditional_edges(
        "classifier",
        route_after_classification,
        {
            "analyzer": "analyzer",
            "non_legal_explainer": "non_legal_explainer"
        }
    )
    
    workflow.add_edge("non_legal_explainer", END)
    
    # Sequential Pipeline for Legal Documents
    workflow.add_edge("analyzer", "risk_detector")
    workflow.add_edge("risk_detector", "critic")
    workflow.add_edge("critic", "explainer")
    workflow.add_edge("explainer", "human_review")
    workflow.add_edge("human_review", "final_compiler")
    workflow.add_edge("final_compiler", END)

    # Compile with MemorySaver Checkpointer and HITL interrupt before human_review
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"]
    )


# Default compiled single-doc application
legal_assistant_app = create_graph()


# =====================================================================
# COMPARISON GRAPH WORKFLOW
# =====================================================================

class ComparisonState(TypedDict):
    text_a: str
    text_b: str
    api_key: Optional[str]
    comparisons: Annotated[List[Dict[str, Any]], operator.add]


def doc_comparison_node(state: ComparisonState) -> Dict[str, Any]:
    """Node executing semantic clause comparison and risk impact analysis."""
    llm = get_llm(agent_name="comparator", api_key=state.get("api_key"))
    report = compare_documents(state["text_a"], state["text_b"], llm=llm, api_key=state.get("api_key"))
    
    comp_list = [c.model_dump() for c in report.comparisons]
    return {"comparisons": comp_list}


def create_comparison_graph():
    workflow = StateGraph(ComparisonState)
    workflow.add_node("compare", doc_comparison_node)
    workflow.add_edge(START, "compare")
    workflow.add_edge("compare", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


comparison_app = create_comparison_graph()
