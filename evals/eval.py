import os
import json
import uuid
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Ensure utf-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to sys.path to import graph and agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import create_graph


def extract_risk_texts(risks_list: list) -> list:
    """Helper to extract searchable strings from list of risk dicts or strings."""
    extracted = []
    for r in risks_list:
        if isinstance(r, dict):
            extracted.append(r.get("risk_type", ""))
            extracted.append(r.get("clause", ""))
            extracted.append(r.get("explanation", ""))
            for ev in r.get("evidence", []):
                extracted.append(str(ev))
        elif isinstance(r, str):
            extracted.append(r)
    return [t for t in extracted if t]


def semantic_risk_match(expected_risk: str, candidate_items: list) -> bool:
    """
    Checks if an expected risk concept is semantically present in candidate items.
    """
    if not candidate_items:
        return False
        
    candidate_texts = extract_risk_texts(candidate_items)
    exp_lower = expected_risk.lower().strip()
    
    # 1. Direct substring check
    for cand in candidate_texts:
        cand_lower = cand.lower().strip()
        if exp_lower in cand_lower or cand_lower in exp_lower:
            return True

    # 2. Key concept mapping dictionary
    synonyms = {
        "confidentiality period": ["confidential", "5 years", "duration", "timeframe", "period", "2 years"],
        "assignment restriction": ["assign", "assignment", "transfer", "consent"],
        "indefinite confidentiality": ["indefinite", "perpetual", "forever", "strict confidence"],
        "unilateral indemnity": ["indemnify", "indemnification", "hold harmless", "unilateral"],
        "perpetual obligation": ["perpetual", "indefinite", "survive", "bound"],
        "non-compete": ["non-compete", "compete", "competitor", "competing"],
        "intellectual property assignment": ["intellectual property", "ip", "inventions", "ownership", "code", "creations"],
        "termination without severance": ["severance", "terminate at any time", "without cause", "without severance"],
        "at-will termination": ["at-will", "terminate at any time", "either party"],
        "unreasonable deposit forfeiture": ["deposit", "retain", "unconditionally", "forfeiture"],
        "entry without notice": ["enter", "notice", "inspection", "prior notice", "quiet enjoyment"],
        "automatic renewal": ["auto-renewal", "automatic renewal", "renew", "renews", "12-month", "2-year", "auto"],
        "unrestricted rent escalation": ["rent", "escalation", "increase", "15%"],
        "relocation clause": ["relocate", "suite", "smaller"],
        "delayed payment terms": ["120 days", "payment", "delay", "receipt", "delayed"],
        "unbalanced indemnification": ["indemnify", "indemnification", "capped at zero", "unlimited", "asymmetrical"],
        "unilateral pricing change": ["pricing", "price", "modify pricing", "rate"],
        "indemnification": ["indemnify", "indemnification", "third party", "claims"],
        "inadequate remedy for downtime": ["remedy", "credit", "5.00", "exclusive", "downtime"],
        "limitation of liability": ["liability", "disclaim", "damage", "consequential", "limit"],
        "unlimited scope revisions": ["revision", "modification", "unlimited", "extra compensation", "scope"],
        "delayed ip transfer": ["ip", "source code", "payment", "until final", "ownership"],
        "usurious interest rate": ["interest", "35%", "compounded", "usury"],
        "harsh acceleration clause": ["acceleration", "default fee", "immediately due", "balance", "default"],
        "uncapped consequential damages": ["consequential damages", "indirect", "lost profit", "without cap", "uncapped"],
        "obscure foreign jurisdiction": ["vanuatu", "governing law", "jurisdiction", "republic", "foreign"],
        "unilateral modification": ["modify", "unilateral", "change terms", "without notice"],
        "class action waiver": ["class action", "waiver", "participate"],
        "mandatory foreign arbitration": ["arbitrat", "vanuatu", "foreign"],
        "perpetual ip transfer": ["ip", "royalty-free", "user content", "derivative"],
        "missing default remedies": ["remedies", "default", "remedy", "notice", "cure"],
        "complete waiver of liability": ["zero liability", "waiver", "damage", "no liability", "assumes zero"],
        "unilateral risk release": ["release", "negligence", "own risk"],
        "hidden binding agreement": ["binding", "exclusive", "invoice", "acceptance", "fine print"],
        "excessive late fees": ["late fee", "late penalty", "25%", "overdue"],
        "exclusivity restriction": ["exclusiv", "competitor", "purchase"],
        "upfront payment demand": ["wire", "deposit", "payment", "600", "processing fee", "upfront"],
        "coercive legal threat": ["prosecution", "forfeiture", "legal action", "breach", "threat"],
        "fraudulent terms": ["scam", "crypto", "fraud", "directive"],
        "unilateral termination penalty": ["penalty", "liquidated", "50,000", "termination fee", "unilateral"],
        "prompt injection attempt": ["prompt", "injection", "instruction", "override", "unusual", "system instruction"]
    }

    key_terms = synonyms.get(exp_lower, [exp_lower])
    for cand in candidate_texts:
        cand_lower = cand.lower().strip()
        matches = sum(1 for term in key_terms if term in cand_lower)
        if matches >= 1:
            return True

    # 3. Token overlap fallback
    exp_tokens = set(exp_lower.split())
    for cand in candidate_texts:
        cand_tokens = set(cand.lower().split())
        overlap = exp_tokens.intersection(cand_tokens)
        if len(overlap) >= max(1, len(exp_tokens) // 2):
            return True

    return False


def match_document_type(predicted: str, expected: str) -> bool:
    """Evaluates whether the predicted document type matches expected document type."""
    if not predicted or not expected:
        return False
    pred = predicted.strip().lower()
    exp = expected.strip().lower()

    if pred == exp or exp in pred or pred in exp:
        return True

    mappings = {
        "non-disclosure agreement": ["nda", "confidentiality", "non-disclosure"],
        "employment agreement": ["employment", "job offer", "work agreement", "employee"],
        "residential lease agreement": ["lease", "rental", "residential"],
        "commercial lease agreement": ["lease", "commercial", "real estate"],
        "independent contractor agreement": ["contractor", "freelance", "service agreement"],
        "saas terms of service": ["saas", "terms of service", "terms", "cloud", "software as a service"],
        "service level agreement": ["sla", "service level"],
        "software development agreement": ["software", "development"],
        "promissory note": ["promissory", "loan", "note"],
        "master services agreement": ["msa", "master services"],
        "software license agreement": ["eula", "license", "software license"],
        "equipment loan agreement": ["loan", "equipment"],
        "liability waiver": ["waiver", "liability", "parking"],
        "invoice agreement": ["invoice", "purchase"],
        "recipe": ["recipe", "cooking", "food"],
        "shopping list": ["shopping", "list", "grocery"],
        "personal invitation": ["invitation", "party", "email"],
        "university admit card": ["admit card", "hall ticket", "exam"],
        "medical prescription": ["prescription", "medical", "rx"],
        "travel itinerary": ["itinerary", "flight", "booking"],
        "internal announcement": ["newsletter", "announcement"],
        "invoice": ["invoice", "receipt", "bill"],
        "non-binding letter of intent": ["letter of intent", "loi", "non-binding"],
        "academic exam question": ["exam", "case study", "question", "academic"],
        "fictional document": ["fictional", "treaty", "sci-fi", "story"],
        "educational textbook excerpt": ["textbook", "chapter", "educational", "excerpt"],
        "work schedule": ["schedule", "shift", "duty", "work schedule"],
        "recommendation letter": ["letter", "recommendation", "reference"],
        "terms of service": ["terms of service", "terms", "tos", "agreement"],
        "consulting agreement": ["consulting", "service", "advisory", "agreement"]
    }

    expected_kws = mappings.get(exp, [exp])
    if any(kw in pred for kw in expected_kws):
        return True

    return False


def run_evaluation():
    start_time = time.time()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(parent_dir, ".env"))

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in environment.")
        sys.exit(1)

    test_set_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_set.json")
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Metrics Tracking
    total_cases = len(test_cases)
    tp, tn, fp, fn = 0, 0, 0, 0
    matched_doc_types = 0

    total_expected_risks = 0
    total_recalled_risks = 0
    total_detected_risks_count = 0
    total_valid_detected_risks = 0

    critic_total_reviews = 0
    critic_verdicts = {"Confirmed": 0, "Downgraded": 0, "Removed": 0}
    critic_weak_flags_reviewed = 0
    critic_correct_downgrades = 0
    critic_valid_flags_reviewed = 0
    critic_correct_confirmations = 0

    rag_grounded_evidence_count = 0
    rag_total_queries = 0

    category_stats = {}
    detailed_results = []

    print("=" * 85, flush=True)
    print(" LegalValidate AI Multi-Agent Benchmark Evaluation Suite", flush=True)
    print(f" Test Suite Size: {total_cases} Labeled Contracts & Documents", flush=True)
    print("=" * 85, flush=True)

    for case in test_cases:
        doc_id = case["id"]
        name = case["name"]
        category = case.get("category", "uncategorized")
        text = case["text"]
        expected_is_legal = case["expected_is_legal"]
        expected_doc_type = case.get("expected_document_type", "Unknown")
        expected_risks = case.get("expected_risks", [])

        if category not in category_stats:
            category_stats[category] = {"total": 0, "class_correct": 0, "doc_type_match": 0, "exp_risks": 0, "caught_risks": 0}
        category_stats[category]["total"] += 1
        category_stats[category]["exp_risks"] += len(expected_risks)

        print(f"\n[Case #{doc_id:02d}] {name} ({category.upper()})", flush=True)
        print(f"  Length: {len(text)} chars | Expected Legal: {expected_is_legal} | Expected Type: '{expected_doc_type}'", flush=True)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        app = create_graph()

        state = {
            "text": text,
            "api_key": api_key,
            "file_metadata": {"case_id": doc_id, "name": name},
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

        try:
            # 1. Run Graph
            final_state = state.copy()
            for output in app.stream(state, config):
                if isinstance(output, dict):
                    for node_name, node_output in output.items():
                        print(f"    - Agent finished: {node_name}", flush=True)
                        if isinstance(node_output, dict):
                            final_state.update(node_output)

            # Check if paused at human_review
            snapshot = app.get_state(config)
            next_nodes = snapshot.next
            if next_nodes and "human_review" in next_nodes:
                for output in app.stream(None, config):
                    if isinstance(output, dict):
                        for node_name, node_output in output.items():
                            print(f"    - Agent finished: {node_name}", flush=True)
                            if isinstance(node_output, dict):
                                final_state.update(node_output)

            # Get final state from checkpointer
            full_values = app.get_state(config).values
            if full_values:
                final_state.update(full_values)

            predicted_is_legal = final_state.get("is_legal", False)
            predicted_doc_type = final_state.get("document_type", "Unknown")
            detected_risks = final_state.get("risks", []) or []
            critic_reviews = final_state.get("critic_reviews", []) or []

            # --- Agent 1: Legal Classifier Evaluation ---
            is_class_correct = (predicted_is_legal == expected_is_legal)
            if expected_is_legal and predicted_is_legal:
                tp += 1
            elif not expected_is_legal and not predicted_is_legal:
                tn += 1
            elif not expected_is_legal and predicted_is_legal:
                fp += 1
            elif expected_is_legal and not predicted_is_legal:
                fn += 1

            if is_class_correct:
                category_stats[category]["class_correct"] += 1
            class_status = "PASS" if is_class_correct else "FAIL"
            print(f"  [Classifier] Expected={expected_is_legal}, Predicted={predicted_is_legal} => {class_status}", flush=True)

            # --- Agent 2: Document Analyzer Evaluation ---
            doc_type_matched = match_document_type(predicted_doc_type, expected_doc_type)
            if doc_type_matched:
                matched_doc_types += 1
                category_stats[category]["doc_type_match"] += 1
            analyzer_status = "PASS" if doc_type_matched else "MISMATCH"
            print(f"  [Analyzer] Expected='{expected_doc_type}', Predicted='{predicted_doc_type}' => {analyzer_status}", flush=True)

            # --- Agent 3: Risk Detector & RAG Evaluation ---
            recalled_in_case = 0
            recalled_risk_list = []
            missed_risk_list = []

            if expected_is_legal:
                for exp_risk in expected_risks:
                    total_expected_risks += 1
                    matched = semantic_risk_match(exp_risk, detected_risks + critic_reviews)
                    if matched:
                        total_recalled_risks += 1
                        recalled_in_case += 1
                        recalled_risk_list.append(exp_risk)
                        category_stats[category]["caught_risks"] += 1
                    else:
                        missed_risk_list.append(exp_risk)

                case_recall_pct = (recalled_in_case / len(expected_risks) * 100) if expected_risks else 100.0
                print(f"  [Risk Detector] Recalled {recalled_in_case}/{len(expected_risks)} risks ({case_recall_pct:.1f}%)", flush=True)

                total_detected_risks_count += len(detected_risks)
                for det_r in detected_risks:
                    # Check RAG Grounding
                    if isinstance(det_r, dict):
                        rag_total_queries += 1
                        sources = det_r.get("source_reference", [])
                        if sources and sources[0] != "Insufficient retrieved evidence":
                            rag_grounded_evidence_count += 1
                            
                    if semantic_risk_match(str(det_r), expected_risks):
                        total_valid_detected_risks += 1
            else:
                print("  [Risk Detector] Early Exit (Non-legal document)", flush=True)

            # --- Agent 4: Critic Verifier Evaluation ---
            case_critic_summary = []
            if critic_reviews:
                for rev in critic_reviews:
                    if isinstance(rev, dict):
                        critic_total_reviews += 1
                        is_valid = rev.get("is_valid", True)
                        sev = rev.get("verified_severity", "MEDIUM")
                        verdict = "Confirmed" if is_valid else "Removed"
                        
                        critic_verdicts[verdict] = critic_verdicts.get(verdict, 0) + 1
                        case_critic_summary.append(f"{verdict} ({sev})")

                        is_valid_ground_truth = any(semantic_risk_match(exp_r, [rev.get("reason", "")]) for exp_r in expected_risks)
                        if not is_valid_ground_truth or len(expected_risks) == 0:
                            critic_weak_flags_reviewed += 1
                            if not is_valid or verdict in ["Downgraded", "Removed"]:
                                critic_correct_downgrades += 1
                        else:
                            critic_valid_flags_reviewed += 1
                            if is_valid:
                                critic_correct_confirmations += 1

                print(f"  [Critic Verifier] Reviewed {len(critic_reviews)} candidate risks -> {case_critic_summary[:3]}", flush=True)

            detailed_results.append({
                "id": doc_id,
                "name": name,
                "category": category,
                "expected_is_legal": expected_is_legal,
                "predicted_is_legal": predicted_is_legal,
                "classifier_pass": is_class_correct,
                "expected_doc_type": expected_doc_type,
                "predicted_doc_type": predicted_doc_type,
                "doc_type_pass": doc_type_matched,
                "expected_risks": expected_risks,
                "detected_risks": detected_risks,
                "recalled_risks": recalled_risk_list,
                "missed_risks": missed_risk_list,
                "critic_reviews": critic_reviews
            })

            time.sleep(1.0)

        except Exception as e:
            print(f"  [ERROR] Execution failed for case #{doc_id}: {e}", flush=True)

    elapsed_time = time.time() - start_time

    # Global Calculations
    accuracy = ((tp + tn) / total_cases) * 100
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall_legal = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1_legal = (2 * precision * recall_legal / (precision + recall_legal)) if (precision + recall_legal) > 0 else 0.0

    doc_type_match_rate = (matched_doc_types / total_cases) * 100
    risk_recall = (total_recalled_risks / total_expected_risks * 100) if total_expected_risks > 0 else 100.0
    risk_precision = (total_valid_detected_risks / total_detected_risks_count * 100) if total_detected_risks_count > 0 else 100.0

    critic_downgrade_rate = (critic_correct_downgrades / critic_weak_flags_reviewed * 100) if critic_weak_flags_reviewed > 0 else 100.0
    critic_confirmation_rate = (critic_correct_confirmations / critic_valid_flags_reviewed * 100) if critic_valid_flags_reviewed > 0 else 100.0
    rag_grounding_rate = (rag_grounded_evidence_count / rag_total_queries * 100) if rag_total_queries > 0 else 100.0

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{timestamp_str}.json"
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_filename)

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "total_test_cases": total_cases,
        "elapsed_seconds": round(elapsed_time, 2),
        "legal_classifier": {
            "accuracy_pct": round(accuracy, 2),
            "precision_pct": round(precision, 2),
            "recall_pct": round(recall_legal, 2),
            "f1_score_pct": round(f1_legal, 2),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        },
        "document_analyzer": {
            "match_rate_pct": round(doc_type_match_rate, 2),
            "matched_cases": matched_doc_types,
            "total_cases": total_cases
        },
        "risk_detector": {
            "recall_pct": round(risk_recall, 2),
            "recalled_risks": total_recalled_risks,
            "total_expected_risks": total_expected_risks,
            "precision_pct": round(risk_precision, 2),
            "total_detected_risks": total_detected_risks_count,
            "rag_grounding_rate_pct": round(rag_grounding_rate, 2)
        },
        "critic_verifier": {
            "total_reviews": critic_total_reviews,
            "verdicts": critic_verdicts,
            "downgrade_accuracy_pct": round(critic_downgrade_rate, 2),
            "confirmation_accuracy_pct": round(critic_confirmation_rate, 2)
        },
        "category_breakdown": category_stats,
        "case_details": detailed_results
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # PRINT SUMMARY
    print("\n" + "=" * 85)
    print("                        LEGALVALIDATE AI BENCHMARK REPORT")
    print("=" * 85)
    print(f" Evaluation Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Evaluation Duration  : {elapsed_time:.1f} seconds")
    print(f" Total Documents      : {total_cases}")
    print("-" * 85)
    print(f" {'AGENT / COMPONENT':<22} | {'PRIMARY METRIC':<25} | {'SCORE':<10} | {'DETAILS':<20}")
    print("-" * 85)
    print(f" {'Legal Classifier':<22} | {'Accuracy':<25} | {accuracy:6.1f}%   | TP:{tp} FP:{fp} TN:{tn} FN:{fn}")
    print(f" {'Legal Classifier':<22} | {'Precision / Recall / F1':<25} | {f1_legal:6.1f}%   | P:{precision:.1f}% R:{recall_legal:.1f}%")
    print(f" {'Document Analyzer':<22} | {'Doc Type Match Rate':<25} | {doc_type_match_rate:6.1f}%   | {matched_doc_types}/{total_cases} matched")
    print(f" {'Risk Detector':<22} | {'Risk Recall (Semantic)':<25} | {risk_recall:6.1f}%   | {total_recalled_risks}/{total_expected_risks} risks caught")
    print(f" {'RAG Retrieval':<22} | {'Evidence Grounding Rate':<25} | {rag_grounding_rate:6.1f}%   | {rag_grounded_evidence_count}/{rag_total_queries} queries grounded")
    print(f" {'Critic / Verifier':<22} | {'Confirmation Accuracy':<25} | {critic_confirmation_rate:6.1f}%   | {critic_correct_confirmations}/{critic_valid_flags_reviewed} verified")
    print("-" * 85)

    print("\n" + "-" * 85)
    print(" CATEGORY BREAKDOWN")
    print("-" * 85)
    print(f" {'CATEGORY':<15} | {'COUNT':<6} | {'CLASS ACCURACY':<16} | {'DOC TYPE MATCH':<16} | {'RISK RECALL':<15}")
    print("-" * 85)
    for cat, stats in category_stats.items():
        cat_count = stats["total"]
        cat_class_acc = (stats["class_correct"] / cat_count * 100) if cat_count > 0 else 0.0
        cat_doc_match = (stats["doc_type_match"] / cat_count * 100) if cat_count > 0 else 0.0
        cat_exp_r = stats["exp_risks"]
        cat_risk_rec = (stats["caught_risks"] / cat_exp_r * 100) if cat_exp_r > 0 else 100.0
        print(f" {cat.upper():<15} | {cat_count:<6} | {cat_class_acc:6.1f}% ({stats['class_correct']}/{cat_count})   | {cat_doc_match:6.1f}% ({stats['doc_type_match']}/{cat_count})   | {cat_risk_rec:6.1f}% ({stats['caught_risks']}/{cat_exp_r})")
    print("=" * 85)
    print(f"\n[INFO] Full results saved to: {results_path}\n")


if __name__ == "__main__":
    run_evaluation()
