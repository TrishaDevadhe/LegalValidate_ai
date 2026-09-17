import os
import json
import time
import re
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from utils import split_text_by_sections, clean_text
from retriever import retrieve_reference_clauses_with_metadata, retrieve_reference_clauses, LegalClauseRetriever
from config import MODEL_CONFIG, DEFAULT_MODEL


# =====================================================================
# LLM FACTORY
# =====================================================================

def get_llm(agent_name: str = None, model_name: str = None, api_key: str = None):
    """
    Returns a ChatGroq instance configured for a specific agent or model name.
    Preserves per-agent model routing defined in config.py.
    """
    if not model_name and agent_name:
        model_name = MODEL_CONFIG.get(agent_name, DEFAULT_MODEL)
    elif not model_name:
        model_name = DEFAULT_MODEL
        
    return ChatGroq(
        groq_api_key=api_key or os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=0
    )


# =====================================================================
# STRICT PYDANTIC STRUCTURED SCHEMAS
# =====================================================================

class LegalClassification(BaseModel):
    is_legal: bool = Field(description="True if the document is a legally binding contract, agreement, policy, or legal instrument. False if non-legal (e.g. recipe, invoice receipt, schedule, exam question, fictional document).")
    document_type: str = Field(description="The specific category of the document (e.g., 'Non-Disclosure Agreement', 'Residential Lease Agreement', 'Recipe', 'Personal Letter').")
    confidence: float = Field(default=0.95, description="Confidence score between 0.0 and 1.0.")
    reason: str = Field(description="A concise justification for the classification.")
    explanation: Optional[str] = Field(default=None, description="Detailed explanation if the document is non-legal or problematic.")


class DocumentAnalysis(BaseModel):
    document_type: str = Field(default="Contract / Document", description="The identified contract category or document type.")
    parties: List[str] = Field(default_factory=list, description="Identified contracting parties, entities, or individuals.")
    effective_date: Optional[str] = Field(default=None, description="Effective date or execution date if specified.")
    term: Optional[str] = Field(default=None, description="Contract duration or term length.")
    renewal: Optional[str] = Field(default=None, description="Renewal mechanisms (e.g. automatic renewal, month-to-month, mutual agreement).")
    payment_terms: Optional[str] = Field(default=None, description="Payment schedule, billing cadence, or fee arrangements.")
    termination: Optional[str] = Field(default=None, description="Termination rights (for cause, for convenience, notice requirements).")
    confidentiality: Optional[str] = Field(default=None, description="Confidentiality duration and scope summary.")
    intellectual_property: Optional[str] = Field(default=None, description="IP ownership, assignment, and license terms.")
    liability_and_indemnity: Optional[str] = Field(default=None, description="Summary of liability caps, indemnities, and consequential damages waivers.")
    governing_law_and_jurisdiction: Optional[str] = Field(default=None, description="Governing law state/country and dispute forum.")
    key_clauses: List[str] = Field(default_factory=list, description="List of primary clauses or section summaries identified.")
    missing_sections: List[str] = Field(default_factory=list, description="Important standard protections omitted from this document (e.g., 'Missing Liability Cap', 'Missing Data Protection Clause').")


class RiskItem(BaseModel):
    risk_id: str = Field(description="Unique risk identifier (e.g. 'risk_1', 'risk_2').")
    clause: str = Field(description="The exact text or excerpt from the contract clause containing the risk.")
    risk_type: str = Field(description="Categorical name of the risk (e.g. 'Unlimited Liability', 'Unilateral Indemnity', 'Usurious Interest Rate', 'Entry Without Notice').")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk severity level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.")
    explanation: str = Field(description="Clear explanation of why this clause creates legal or commercial hazard.")
    recommendation: str = Field(description="Actionable suggestion or redline recommendation to mitigate the risk.")
    evidence: List[str] = Field(default_factory=list, description="Supporting textual evidence or problematic phrases.")
    source_reference: List[str] = Field(default_factory=list, description="Retrieved baseline reference standard clause or 'Insufficient retrieved evidence'.")
    confidence: float = Field(default=0.85, description="Confidence score between 0.0 and 1.0.")


class RiskAssessment(BaseModel):
    risks: List[RiskItem] = Field(default_factory=list, description="List of structured risks identified in the document.")


class CriticReviewItem(BaseModel):
    risk_id: str = Field(description="The risk_id corresponding to the reviewed risk.")
    is_valid: bool = Field(description="True if the risk represents a genuine legal/commercial issue. False if standard boilerplate or false positive.")
    verified_severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="The audited and corrected risk severity.")
    reason: str = Field(description="Objective legal rationale for confirming, downgrading, or rejecting the candidate risk.")
    evidence_supported: bool = Field(default=True, description="True if the contract text and retrieved baseline evidence support the finding.")
    confidence: float = Field(default=0.90, description="Critic confidence score between 0.0 and 1.0.")


class CriticVerdict(BaseModel):
    reviews: List[CriticReviewItem] = Field(default_factory=list, description="List of independent audits for candidate risks.")


class ExplanationItem(BaseModel):
    risk_id: str = Field(description="Corresponding risk_id.")
    clause_summary: str = Field(description="What does the clause say in simple, jargon-free words?")
    business_impact: str = Field(description="Why could this matter from a practical/financial perspective?")
    severity: str = Field(description="Risk level (LOW, MEDIUM, HIGH, CRITICAL).")
    review_action: str = Field(description="What practical review point or conversation should be had with the counterparty?")


class DocumentExplanation(BaseModel):
    summary: str = Field(description="Brief executive summary of the overall document.")
    simplified_explanation: str = Field(description="Plain-English explanation of rights, obligations, and commercial terms for non-lawyers.")
    risk_explanations: List[ExplanationItem] = Field(default_factory=list, description="Plain-language breakdown of each detected risk.")
    disclaimer: str = Field(
        default="LegalValidate AI is an automated document analysis tool and does not constitute formal legal advice. Please consult qualified legal counsel for binding guidance.",
        description="Legal safety disclaimer."
    )


class ClauseComparisonItem(BaseModel):
    clause_type: str = Field(description="Category of the clause (e.g. 'Confidentiality Term', 'Indemnification', 'Payment Due Date').")
    status: Literal["ADDED", "REMOVED", "MODIFIED", "UNCHANGED"] = Field(description="Status of the clause in Version B relative to Version A.")
    old_text: Optional[str] = Field(default=None, description="Original clause text in Version A (or None if newly added).")
    new_text: Optional[str] = Field(default=None, description="Modified clause text in Version B (or None if removed).")
    change_summary: str = Field(description="Concise description of what changed between the two versions.")
    risk_impact: Optional[str] = Field(default=None, description="Risk impact: 'Increase', 'Decrease', or 'No Change' with brief explanation.")


class ComparisonReport(BaseModel):
    comparisons: List[ClauseComparisonItem] = Field(default_factory=list, description="Semantically aligned clause comparison records.")


# =====================================================================
# ROBUST PARSING, REPAIR & RETRY UTILITIES
# =====================================================================

def parse_dict_to_schema(schema, data: dict):
    """
    Sanitizes and maps raw dictionary fields to expected Pydantic schema types,
    handling aliasing, missing keys, and nested structures gracefully.
    """
    if not isinstance(data, dict):
        data = {}

    # Alias mapping for document classification
    if schema == LegalClassification:
        if "is_legal" not in data:
            data["is_legal"] = data.get("legal", True)
        if "document_type" not in data:
            data["document_type"] = data.get("doc_type", data.get("type", "Legal Document"))
        if "confidence" not in data:
            data["confidence"] = 0.90
        if "reason" not in data:
            data["reason"] = data.get("classification_reason", "Document evaluated.")

    # Alias mapping for DocumentAnalysis
    elif schema == DocumentAnalysis:
        if "document_type" not in data:
            for k in ["doc_type", "type", "category"]:
                if k in data and data[k]:
                    data["document_type"] = str(data[k])
                    break
            data.setdefault("document_type", "Contract / Document")
        data.setdefault("parties", [])
        data.setdefault("key_clauses", [])
        data.setdefault("missing_sections", [])

    # Alias mapping for RiskAssessment
    elif schema == RiskAssessment:
        raw_risks = data.get("risks", [])
        clean_risks = []
        for idx, r in enumerate(raw_risks, start=1):
            if isinstance(r, str):
                clean_risks.append({
                    "risk_id": f"risk_{idx}",
                    "clause": r,
                    "risk_type": "Contractual Risk",
                    "severity": "MEDIUM",
                    "explanation": r,
                    "recommendation": "Review clause with legal counsel.",
                    "evidence": [r],
                    "source_reference": ["Standard Commercial Practice"],
                    "confidence": 0.85
                })
            elif isinstance(r, dict):
                r.setdefault("risk_id", f"risk_{idx}")
                r.setdefault("clause", r.get("explanation", "Target clause"))
                r.setdefault("risk_type", "Identified Risk")
                r.setdefault("severity", "MEDIUM")
                if r["severity"] not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                    r["severity"] = "MEDIUM"
                r.setdefault("explanation", "Identified risk in clause.")
                r.setdefault("recommendation", "Review terms.")
                r.setdefault("evidence", [])
                r.setdefault("source_reference", ["Insufficient retrieved evidence"])
                r.setdefault("confidence", 0.85)
                clean_risks.append(r)
        data["risks"] = clean_risks

    # Alias mapping for CriticVerdict
    elif schema == CriticVerdict:
        raw_reviews = data.get("reviews", [])
        clean_reviews = []
        for idx, rev in enumerate(raw_reviews, start=1):
            if isinstance(rev, dict):
                rev.setdefault("risk_id", f"risk_{idx}")
                rev.setdefault("is_valid", rev.get("verdict", "Confirmed") != "Removed")
                sev = rev.get("verified_severity", rev.get("verdict", "MEDIUM"))
                if sev not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                    sev = "MEDIUM"
                rev["verified_severity"] = sev
                rev.setdefault("reason", rev.get("reason", "Audited by critic agent."))
                rev.setdefault("evidence_supported", True)
                rev.setdefault("confidence", 0.90)
                clean_reviews.append(rev)
        data["reviews"] = clean_reviews

    # Fallback default fill for all model fields
    for field_name, field_info in schema.model_fields.items():
        if field_name not in data or data[field_name] is None:
            field_type_str = str(field_info.annotation)
            if "List" in field_type_str or "list" in field_type_str:
                data[field_name] = []
            elif "bool" in field_type_str:
                data[field_name] = True
            elif "float" in field_type_str:
                data[field_name] = 0.85
            elif "str" in field_type_str:
                data[field_name] = "N/A"
            else:
                data[field_name] = None

    try:
        return schema(**data)
    except Exception:
        # Construct minimal instance
        min_args = {}
        for fname, finfo in schema.model_fields.items():
            if fname in data:
                min_args[fname] = data[fname]
        return schema.model_construct(**min_args)


def extract_json_from_text(raw_text: str) -> Optional[dict]:
    """Extracts and parses JSON object or array from LLM text responses or error strings."""
    if not raw_text:
        return None
    # Look for JSON markdown block ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', raw_text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Scan for first valid { ... }
    start = raw_text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw_text)):
            c = raw_text[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = raw_text[start:i+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            pass
                        break
        start = raw_text.find("{", start + 1)
    return None


def invoke_with_retry_and_repair(prompt_template, llm, schema, inputs: dict):
    """
    Executes a structured output chain with retry and automatic fallback repair,
    guaranteeing a valid Pydantic instance without application crashes.
    """
    time.sleep(0.5)
    try:
        chain = prompt_template | llm.with_structured_output(schema)
    except Exception:
        chain = prompt_template | llm

    for attempt in range(4):
        try:
            res = chain.invoke(inputs)
            if isinstance(res, schema):
                return res
            elif isinstance(res, dict):
                return parse_dict_to_schema(schema, res)
            elif hasattr(res, "content"):
                parsed = extract_json_from_text(res.content)
                if parsed:
                    return parse_dict_to_schema(schema, parsed)
            return res
        except Exception as e:
            err_msg = str(e)
            # Try extracting json from the exception message
            parsed = extract_json_from_text(err_msg)
            if parsed:
                return parse_dict_to_schema(schema, parsed)
                
            if "429" in err_msg or "rate_limit" in err_msg:
                time.sleep(8)
                continue
            time.sleep(1.5)

    # Fallback to schema JSON repair prompt
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    try:
        content_repr = prompt_template.format(**inputs)
    except Exception:
        content_repr = str(inputs)

    repair_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert JSON output engine. Format your entire answer strictly as valid JSON matching the provided JSON Schema. Do not wrap in markdown or conversational text."),
        ("human", "SCHEMA:\n{schema_json}\n\nCONTEXT:\n{content_repr}\n\nOutput valid JSON:")
    ])
    
    try:
        repair_res = (repair_prompt | llm).invoke({"schema_json": schema_json, "content_repr": content_repr[:3000]})
        parsed = extract_json_from_text(repair_res.content if hasattr(repair_res, "content") else str(repair_res))
        if parsed:
            return parse_dict_to_schema(schema, parsed)
    except Exception:
        pass

    return parse_dict_to_schema(schema, {})


# =====================================================================
# AGENT 1: LEGAL CLASSIFIER
# =====================================================================

def classify_legal(text: str, llm=None, api_key: str = None) -> LegalClassification:
    """
    Evaluates whether an uploaded document is a legally binding document or a non-legal item.
    Returns a validated LegalClassification schema.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="legal_classifier", api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a senior legal classification expert. You strictly output tool schema JSON and never output conversational text.\n\n"
         "Your task is to classify whether the provided document is a binding legal contract or non-legal content.\n\n"
         "1. LEGAL DOCUMENTS (is_legal=True):\n"
         "   - Non-Disclosure Agreements (NDAs), Employment Contracts, Residential/Commercial Leases, Independent Contractor Agreements, Master Service Agreements (MSAs), Service Level Agreements (SLAs), Software Licenses/EULAs, Promissory Notes, Liability Waivers, Terms of Service, Privacy Policies.\n\n"
         "2. NON-LEGAL DOCUMENTS (is_legal=False):\n"
         "   - Recipes, Shopping Lists, Personal Letters/Emails, Travel Itineraries, Medical Prescriptions, Work/Shift Schedules, University Admit Cards, Resumes, News Articles, Technical Manuals.\n"
         "   - Academic Exam Questions / Hypotheticals (e.g., 'LAW SCHOOL FINAL EXAM', 'QUESTION 1:', 'Analyze whether Party A breached').\n"
         "   - Fictional / Sci-Fi Treaties (e.g., 'Intergalactic Treaty of Alpha Centauri', 'Starfleet Accord').\n"
         "   - Explicitly Non-Binding Letters of Intent (LOIs).\n\n"
         "Always assign a confidence score between 0.0 and 1.0 and specify the document_type."),
        ("human", "Document to classify:\n\n{text}")
    ])

    chunks = split_text_by_sections(text, max_chars=12000)
    # Evaluate header and initial chunk
    sample_text = chunks[0] if chunks else text[:4000]
    result = invoke_with_retry_and_repair(prompt, llm, LegalClassification, {"text": sample_text})
    
    if not isinstance(result, LegalClassification):
        result = parse_dict_to_schema(LegalClassification, result if isinstance(result, dict) else {})
    return result


# =====================================================================
# AGENT 2: DOCUMENT ANALYZER
# =====================================================================

def analyze_document(text: str, llm=None, api_key: str = None) -> DocumentAnalysis:
    """
    Extracts structural components: parties, dates, term, renewal, payment, termination,
    confidentiality, IP, liability/indemnity, governing law, and missing standard safeguards.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="document_analyzer", api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a cautious and highly accurate document analyst.\n"
         "1. Identify the EXACT specific category/type of the document provided (e.g. 'Non-Disclosure Agreement', 'Employment Agreement', 'Residential Lease Agreement', 'Commercial Lease Agreement', 'Independent Contractor Agreement', 'SaaS Terms of Service', 'Service Level Agreement', 'Software Development Agreement', 'Promissory Note', 'Master Services Agreement', 'Software License Agreement', 'Equipment Loan Agreement', 'Liability Waiver', 'Consulting Agreement').\n"
         "2. Extract parties, effective dates, term, obligations, and identify any critical missing contractual protections.\n"
         "Strictly output the schema tool call."),
        ("human", "Analyze this contract:\n\n{text}")
    ])

    chunks = split_text_by_sections(text, max_chars=16000)
    sample_text = chunks[0] if len(chunks) == 1 else text[:20000]
    
    result = invoke_with_retry_and_repair(prompt, llm, DocumentAnalysis, {"text": sample_text})
    if not isinstance(result, DocumentAnalysis):
        result = parse_dict_to_schema(DocumentAnalysis, result if isinstance(result, dict) else {})
    return result


# =====================================================================
# AGENT 3: RAG-GROUNDED RISK DETECTOR
# =====================================================================

def detect_risks(text: str, llm=None, api_key: str = None) -> RiskAssessment:
    """
    Identifies contract clauses, retrieves baseline reference standards from the RAG store,
    compares the contract against baseline standards, and outputs grounded RiskItem records.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="risk_detector", api_key=api_key)

    chunks = split_text_by_sections(text, max_chars=4000)
    all_risks: List[RiskItem] = []
    seen_risks = set()
    risk_counter = 1

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior RAG-grounded legal risk detection agent. You compare target contract clauses against retrieved standard commercial legal baselines.\n\n"
         "INSTRUCTIONS:\n"
         "1. Identify hazardous, asymmetric, non-compliant, or overreaching terms in the target clause.\n"
         "2. Compare the clause directly with the provided RETRIEVED REFERENCE BASELINES.\n"
         "3. If a baseline is available, use it as 'source_reference' to support your finding.\n"
         "4. If no relevant baseline was retrieved, set source_reference to ['Insufficient retrieved evidence'] and do NOT invent legal citations.\n"
         "5. Assign severity strictly as 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.\n"
         "6. Provide concrete recommendation and cite verbatim evidence from the clause.\n\n"
         "OUTPUT SCHEMA: Return a RiskAssessment with a list of RiskItem objects."),
        ("human", 
         "RETRIEVED REFERENCE BASELINES:\n{reference_context}\n\n"
         "TARGET CONTRACT CLAUSE/SECTION:\n{chunk_text}")
    ])

    for chunk in chunks:
        if len(chunk.strip()) < 40:
            continue
            
        # 1. RAG Retrieval: Query reference baseline store
        ref_matches = retrieve_reference_clauses_with_metadata(chunk, k=2, min_score=0.10)
        ref_context_lines = []
        source_refs = []
        for m in ref_matches:
            if m["text"] != "Insufficient retrieved evidence":
                ref_context_lines.append(f"- [{m['clause_type']}]: {m['text']} (Baseline Standard: {m['baseline_standard']})")
                source_refs.append(f"{m['clause_type']}: {m['text']}")
            else:
                ref_context_lines.append("- Insufficient retrieved evidence in reference vector store.")
                source_refs.append("Insufficient retrieved evidence")
                
        ref_context = "\n".join(ref_context_lines)

        # 2. Invoke Risk Detector
        res = invoke_with_retry_and_repair(
            prompt,
            llm,
            RiskAssessment,
            {"reference_context": ref_context, "chunk_text": chunk}
        )

        if isinstance(res, RiskAssessment) and res.risks:
            for item in res.risks:
                # Deduplicate by risk_type or clause snippet
                key = f"{item.risk_type.lower()}_{item.clause[:40].lower()}"
                if key not in seen_risks:
                    seen_risks.add(key)
                    item.risk_id = f"risk_{risk_counter}"
                    risk_counter += 1
                    if not item.source_reference:
                        item.source_reference = source_refs[:2]
                    all_risks.append(item)

    return RiskAssessment(risks=all_risks)


# =====================================================================
# AGENT 4: CRITIC / VERIFICATION AGENT
# =====================================================================

def review_risks(text: str, risks: List[Any], llm=None, api_key: str = None) -> CriticVerdict:
    """
    Independently audits candidate risks flagged by the Risk Detector.
    Verifies existence of clause, evidence support, justifies severity, and filters false positives.
    """
    if not risks:
        return CriticVerdict(reviews=[])
        
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="critic", api_key=api_key)

    # Format risk items for review
    formatted_risks = []
    for idx, r in enumerate(risks, start=1):
        if isinstance(r, RiskItem):
            formatted_risks.append(
                f"Risk ID: {r.risk_id}\n"
                f"Type: {r.risk_type}\n"
                f"Severity: {r.severity}\n"
                f"Clause: {r.clause}\n"
                f"Explanation: {r.explanation}\n"
                f"RAG Sources: {', '.join(r.source_reference)}"
            )
        elif isinstance(r, dict):
            formatted_risks.append(
                f"Risk ID: {r.get('risk_id', f'risk_{idx}')}\n"
                f"Type: {r.get('risk_type', 'Identified Risk')}\n"
                f"Severity: {r.get('severity', 'MEDIUM')}\n"
                f"Clause: {r.get('clause', '')}\n"
                f"Explanation: {r.get('explanation', '')}"
            )
        else:
            formatted_risks.append(f"Risk ID: risk_{idx}\nDescription: {str(r)}")

    risks_input_text = "\n\n---\n\n".join(formatted_risks)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict, senior legal auditor and Critic Agent. Your sole purpose is to audit candidate risks flagged by an automated detector.\n\n"
         "AUDIT CRITERIA:\n"
         "1. FALSE POSITIVES / STANDARD TERMS (is_valid=False):\n"
         "   - Reject standard, customary commercial terms (e.g. mutual 2-3 year confidentiality, standard at-will employment, standard Net-30 payment, Delaware governing law).\n"
         "   - Reject mere clause restatements that describe provisions without any actual hazard.\n"
         "2. VALID RISKS (is_valid=True):\n"
         "   - Confirm asymmetric, unconscionable, or overreaching terms (e.g. unilateral indemnity, unlimited liability, entry without notice, usurious interest >18%, perpetual non-competes, delayed 120-day payments).\n"
         "3. SEVERITY ADJUSTMENT:\n"
         "   - Assign verified_severity strictly as 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.\n"
         "4. EVIDENCE VERIFICATION:\n"
         "   - Set evidence_supported=True if the target document clause actually substantiates the risk.\n\n"
         "Return a CriticVerdict containing CriticReviewItem for each candidate risk."),
        ("human",
         "DOCUMENT CONTEXT:\n{doc_text}\n\n"
         "CANDIDATE RISKS TO AUDIT:\n{risks_input_text}")
    ])

    result = invoke_with_retry_and_repair(
        prompt,
        llm,
        CriticVerdict,
        {"doc_text": text[:15000], "risks_input_text": risks_input_text}
    )

    if not isinstance(result, CriticVerdict):
        result = parse_dict_to_schema(CriticVerdict, result if isinstance(result, dict) else {})
    return result


# =====================================================================
# AGENT 5: EXPLANATION AGENT
# =====================================================================

def explain_document(text: str, risks: List[Any] = None, llm=None, api_key: str = None) -> DocumentExplanation:
    """
    Translates legal clauses and validated risks into clear, jargon-free plain English for non-lawyers.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="explanation_agent", api_key=api_key)

    risks_summary_list = []
    if risks:
        for r in risks:
            if isinstance(r, RiskItem):
                risks_summary_list.append(f"[{r.risk_id}] ({r.severity}) {r.risk_type}: {r.explanation}")
            elif isinstance(r, dict):
                risks_summary_list.append(f"[{r.get('risk_id', 'risk')}] ({r.get('severity', 'MEDIUM')}) {r.get('risk_type', '')}: {r.get('explanation', '')}")
            else:
                risks_summary_list.append(str(r))

    risks_context = "\n".join(risks_summary_list) if risks_summary_list else "No major risks flagged."

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a plain-language legal explainer. Your job is to translate complex legal terms into simple, actionable summaries for business founders, tenants, employees, and non-lawyers.\n\n"
         "INSTRUCTIONS:\n"
         "1. Provide a concise 'summary' of the document.\n"
         "2. Provide a 'simplified_explanation' of what rights and duties are being created.\n"
         "3. For each flagged risk, explain: what does the clause say, why could this matter financially/operationally, and what should be reviewed/discussed.\n"
         "4. Include the formal disclaimer stating this is an AI analytical tool and not legal advice.\n"
         "Strictly output valid tool schema."),
        ("human", 
         "DOCUMENT TEXT:\n{text}\n\n"
         "FLAGGED RISKS:\n{risks_context}")
    ])

    result = invoke_with_retry_and_repair(
        prompt,
        llm,
        DocumentExplanation,
        {"text": text[:15000], "risks_context": risks_context}
    )

    if not isinstance(result, DocumentExplanation):
        result = parse_dict_to_schema(DocumentExplanation, result if isinstance(result, dict) else {})
    return result


# =====================================================================
# AGENT 6: SEMANTIC CLAUSE COMPARATOR
# =====================================================================

def compare_documents(text_a: str, text_b: str, llm=None, api_key: str = None) -> ComparisonReport:
    """
    Semantically aligns clauses between Contract Version A and Version B.
    Identifies ADDED, REMOVED, MODIFIED, and UNCHANGED clauses and determines risk impact.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="comparator", api_key=api_key)

    clauses_a = [c.strip() for c in split_text_by_sections(text_a, max_chars=1800) if c.strip()]
    clauses_b = [c.strip() for c in split_text_by_sections(text_b, max_chars=1800) if c.strip()]

    if not clauses_a and not clauses_b:
        return ComparisonReport(comparisons=[])

    # Align clauses using TF-IDF / N-gram cosine similarity
    retriever_a = LegalClauseRetriever([{"text": c, "metadata": {"idx": i}} for i, c in enumerate(clauses_a)])
    
    pairs = []
    matched_b = set()
    matched_a = set()

    for idx_b, cb in enumerate(clauses_b):
        matches = retriever_a.similarity_search_with_score(cb, k=1, min_score=0.18)
        if matches:
            best_idx_a = matches[0].get("metadata", {}).get("idx")
            if best_idx_a is not None and best_idx_a not in matched_a:
                pairs.append((clauses_a[best_idx_a], cb, "MATCH"))
                matched_a.add(best_idx_a)
                matched_b.add(idx_b)
                continue
        # Unmatched in A -> ADDED in B
        pairs.append((None, cb, "ADDED"))

    # Unmatched in A -> REMOVED in B
    for idx_a, ca in enumerate(clauses_a):
        if idx_a not in matched_a:
            pairs.append((ca, None, "REMOVED"))

    # Batch and send to Comparator LLM
    batch_size = 4
    comparison_items: List[ClauseComparisonItem] = []

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert contract comparison and redlining analyst.\n"
         "For each pair of clauses provided, determine:\n"
         "1. 'clause_type': General category (e.g., 'Indemnity', 'Liability Cap', 'Payment Terms', 'Termination', 'Confidentiality').\n"
         "2. 'status': Strictly 'ADDED', 'REMOVED', 'MODIFIED', or 'UNCHANGED'.\n"
         "3. 'change_summary': Concise summary of what was altered, deleted, or inserted.\n"
         "4. 'risk_impact': Strictly 'Increase', 'Decrease', or 'No Change', followed by brief risk rationale.\n\n"
         "Output ComparisonReport schema with list of ClauseComparisonItem."),
        ("human", "Analyze these contract revisions:\n\n{clauses_input}")
    ])

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        input_text = ""
        for j, (orig, mod, hint) in enumerate(batch, start=1):
            orig_str = orig if orig else "[None - Newly Added in Version B]"
            mod_str = mod if mod else "[None - Deleted in Version B]"
            input_text += f"=== Revision Pair #{j} ({hint}) ===\nORIGINAL (Version A):\n{orig_str}\n\nMODIFIED (Version B):\n{mod_str}\n\n"

        res = invoke_with_retry_and_repair(
            prompt,
            llm,
            ComparisonReport,
            {"clauses_input": input_text}
        )

        if isinstance(res, ComparisonReport) and res.comparisons:
            comparison_items.extend(res.comparisons)

    return ComparisonReport(comparisons=comparison_items)
