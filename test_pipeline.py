import os
import sys
import uuid
import time
from dotenv import load_dotenv

# Ensure utf-8 terminal output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import extract_document_text, clean_text, split_text_by_sections, generate_pdf_report
from retriever import get_retriever, retrieve_reference_clauses_with_metadata
from agents import (
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
from graph import create_graph, comparison_app
from db import init_db, save_analysis, get_all_analyses


def run_all_tests():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[FAIL] GROQ_API_KEY is not set.")
        return False

    print("=" * 75)
    print(" LegalValidate AI — Comprehensive Test Suite")
    print("=" * 75)

    # 1. RAG Retriever Grounding
    print("\n[Test 1/8] RAG Retriever Reference Grounding...", flush=True)
    query = "Landlord reserves the right to enter the premises at any time without prior notice."
    matches = retrieve_reference_clauses_with_metadata(query, k=2, min_score=0.10)
    assert len(matches) > 0, "No matches found"
    assert "clause_type" in matches[0], "Missing clause_type"
    print(f"  [OK] Grounded to: {matches[0]['clause_type']} (Score: {matches[0]['score']:.2f})")

    # 2. RAG Zero-Hallucination Fallback
    print("\n[Test 2/8] RAG Insufficient Evidence Fallback...", flush=True)
    unrelated = "Quantum chromodynamics gravitational cosmological constant."
    fb = retrieve_reference_clauses_with_metadata(unrelated, k=2, min_score=0.40)
    assert fb[0]["text"] == "Insufficient retrieved evidence", "Fallback failed"
    print("  [OK] Triggered 'Insufficient retrieved evidence' fallback correctly.")

    # 3. Legal Classifier
    print("\n[Test 3/8] Legal Classifier (Legal vs Non-Legal)...", flush=True)
    legal_doc = "MUTUAL NON-DISCLOSURE AGREEMENT. Acme Corp and Beta Inc agree to maintain confidentiality for 3 years."
    c1 = classify_legal(legal_doc, api_key=api_key)
    assert c1.is_legal is True, "Failed to classify legal document"
    print(f"  [OK] Legal Document: {c1.document_type} (Confidence: {c1.confidence:.2f})")

    recipe = "Spaghetti Carbonara: 400g pasta, 4 egg yolks, 150g guanciale, pecorino cheese."
    c2 = classify_legal(recipe, api_key=api_key)
    assert c2.is_legal is False, "Failed to reject non-legal content"
    print(f"  [OK] Non-Legal Document: {c2.document_type} (is_legal={c2.is_legal})")

    # 4. Document Structure Analyzer
    print("\n[Test 4/8] Document Structure Analyzer...", flush=True)
    nda_text = "MUTUAL NDA. Entered into between Alpha Corp and Beta Inc on Jan 1, 2026. Term: 2 years. Governing Law: Delaware."
    doc_meta = analyze_document(nda_text, api_key=api_key)
    assert isinstance(doc_meta, DocumentAnalysis), "Invalid analysis schema"
    print(f"  [OK] Extracted Type: {doc_meta.document_type} | Parties: {doc_meta.parties}")

    # 5. Risk Detector & Critic Verifier
    print("\n[Test 5/8] RAG Risk Detector & Critic Audit...", flush=True)
    lease = "RESIDENTIAL LEASE. Landlord may enter premises at any hour without prior notice. Security deposit is unconditionally retained upon exit."
    risks_res = detect_risks(lease, api_key=api_key)
    assert len(risks_res.risks) >= 1, "No risks detected"
    print(f"  [OK] Risk Detector flagged {len(risks_res.risks)} hazards (e.g. {risks_res.risks[0].risk_type})")

    critic_res = review_risks(lease, risks_res.risks, api_key=api_key)
    assert len(critic_res.reviews) >= 1, "Critic failed to review risks"
    print(f"  [OK] Critic audited {len(critic_res.reviews)} risks -> Verified Severity: {critic_res.reviews[0].verified_severity}")

    # 6. Explanation Agent
    print("\n[Test 6/8] Plain-Language Explanation Agent...", flush=True)
    expl = explain_document(lease, risks=risks_res.risks, api_key=api_key)
    assert isinstance(expl, DocumentExplanation), "Invalid explanation schema"
    assert "LegalValidate AI" in expl.disclaimer or "legal advice" in expl.disclaimer.lower(), "Missing disclaimer"
    print(f"  [OK] Explanation generated: {expl.summary[:80]}...")

    # 7. Semantic Redline Comparator
    print("\n[Test 7/8] Semantic Redline Comparator...", flush=True)
    v1 = "Payment is due within 30 days of invoice receipt."
    v2 = "Payment is due within 120 days of invoice receipt."
    comp = compare_documents(v1, v2, api_key=api_key)
    assert len(comp.comparisons) >= 1, "Comparison failed"
    print(f"  [OK] Redline Impact: {comp.comparisons[0].change_summary} ({comp.comparisons[0].risk_impact})")

    # 8. ReportLab PDF Generation & LangGraph HITL
    print("\n[Test 8/8] ReportLab PDF Generation & LangGraph DAG...", flush=True)
    test_state = {
        "document_type": "Residential Lease Agreement",
        "is_legal": True,
        "classification_reason": "Binding lease.",
        "summary": "Sample summary.",
        "simplified_explanation": "Sample explanation.",
        "risks": [risks_res.risks[0].model_dump()],
        "critic_reviews": [critic_res.reviews[0].model_dump()],
        "human_reviews": [{"risk_id": "risk_1", "decision": "Accept Risk", "note": "Negotiate with landlord"}]
    }
    pdf_bytes = generate_pdf_report(test_state)
    assert len(pdf_bytes) > 1000, "PDF generation failed"
    print(f"  [OK] Generated PDF report ({len(pdf_bytes)} bytes)")

    # LangGraph pipeline execution
    app = create_graph()
    t_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": t_id}}
    init_st = {
        "text": lease,
        "api_key": api_key,
        "file_metadata": {"filename": "test.txt"},
        "is_legal": None,
        "document_type": None,
        "classification_reason": None,
        "classification_confidence": None,
        "problematic_explanation": None,
        "document_analysis": None,
        "key_clauses": [],
        "risks": [],
        "critic_reviews": [],
        "summary": None,
        "simplified_explanation": None,
        "risk_explanations": [],
        "human_reviews": [],
        "final_report": None
    }
    
    for out in app.stream(init_st, cfg):
        pass
    snap = app.get_state(cfg)
    assert "human_review" in snap.next, "HITL breakpoint not triggered"
    
    for out in app.stream(None, cfg):
        pass
    fin = app.get_state(cfg).values
    assert fin.get("is_legal") is True, "Pipeline failed"
    print("  [OK] Full LangGraph DAG executed seamlessly through HITL breakpoint.")

    print("\n" + "=" * 75)
    print(" [PASS] All 8 Architectural & Functional Tests Passed Successfully!")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
