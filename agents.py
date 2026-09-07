from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import split_text_by_sections
from retriever import retrieve_reference_clauses
from config import MODEL_CONFIG, DEFAULT_MODEL

# Define the models
def get_llm(agent_name: str = None, model_name: str = None, api_key: str = None):
    """
    Returns a ChatGroq instance configured for a specific agent or model name.
    If agent_name is specified, pulls the model mapping from MODEL_CONFIG in config.py.
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

class DocumentAnalysis(BaseModel):
    document_type: str = Field(description="The specific category/type of the document (e.g., 'Non-Disclosure Agreement', 'Employment Contract', 'Privacy Policy', 'Invoice', 'Personal Letter', etc.). Be as specific as possible.")
    key_clauses: List[str] = Field(description="A JSON array of strings, each representing a key section or clause identified. If no clauses are found, return an empty list [].")

class LegalClassification(BaseModel):
    is_legal: bool = Field(description="Whether the document is a legal document")
    classification: str = Field(description="Strictly 'Legal' or 'Non-Legal'")
    reason: str = Field(description="A one-line reason for the classification")
    explanation: Optional[str] = Field(description="Detailed explanation of why it's problematic if non-legal")

class RiskAssessment(BaseModel):
    risks: List[str] = Field(description="A JSON array of strings, each representing a potential risk, unfair clause, or legal issue. If no risks are found, return an empty list [].")

class SimplifiedExplanation(BaseModel):
    summary: str = Field(description="Brief summary of the document")
    simplified_explanation: str = Field(description="Simple terms explanation for a non-legal person")

def parse_dict_to_schema(schema, data: dict):
    """Fills missing required schema fields with sensible defaults and alias mapping when parsing LLM output."""
    # Map common key aliases to standard schema fields
    if "document_type" not in data or not data["document_type"]:
        for alt_key in ["doc_type", "type_of_document", "category", "document_category", "type", "document_name", "title", "name", "kind", "document", "document_title", "agreement_type", "contract_type"]:
            if alt_key in data and data[alt_key]:
                data["document_type"] = str(data[alt_key])
                break
        if "document_type" not in data or not data["document_type"]:
            for k, v in data.items():
                if ("type" in k.lower() or "category" in k.lower() or "title" in k.lower()) and v and isinstance(v, str):
                    data["document_type"] = str(v)
                    break
        if "document_type" not in data or not data["document_type"]:
            data["document_type"] = "Contract / Document"

    if "risks" not in data or not data["risks"]:
        for alt_key in ["identified_risks", "potential_risks", "flagged_risks", "risk_factors", "clause_risks", "unfair_clauses"]:
            if alt_key in data and isinstance(data[alt_key], list):
                data["risks"] = [str(r) for r in data[alt_key]]
                break

    if "key_clauses" not in data or not data["key_clauses"]:
        for alt_key in ["clauses", "sections", "key_sections", "identified_clauses", "findings"]:
            if alt_key in data and isinstance(data[alt_key], list):
                data["key_clauses"] = [str(c) for c in data[alt_key]]
                break

    fields = schema.model_fields
    for field_name, field_info in fields.items():
        if field_name not in data or data[field_name] is None:
            field_type_str = str(field_info.annotation)
            if "List" in field_type_str or "list" in field_type_str:
                data[field_name] = []
            elif field_name == "is_legal":
                data[field_name] = True
            elif field_name == "classification":
                data[field_name] = "Legal"
            elif field_name == "reason":
                data[field_name] = "Analyzed document context"
            elif field_name == "explanation":
                data[field_name] = None
            elif field_name == "summary":
                data[field_name] = data.get("simplified_explanation", "Summary of document")
            elif field_name == "simplified_explanation":
                data[field_name] = data.get("summary", "Simplified explanation of terms")
            elif field_name == "risks":
                data[field_name] = []
            else:
                data[field_name] = ""

    # Sanitize nested list items (e.g. RiskReview dicts inside reviews)
    for field_name, val in list(data.items()):
        if isinstance(val, list):
            new_list = []
            for item in val:
                if isinstance(item, dict):
                    if "verdict" in item and "risk" not in item:
                        item["risk"] = "Identified risk"
                    if "risk" in item and "verdict" not in item:
                        item["verdict"] = "Confirmed"
                    if "risk" in item and "reason" not in item:
                        item["reason"] = "Reviewed by critic verifier"
                    new_list.append(item)
                else:
                    new_list.append(item)
            data[field_name] = new_list

    return schema(**data)

def extract_json_from_error(error_msg: str) -> Optional[str]:
    """Scans error message for valid JSON object strings, ignoring outer error wrappers."""
    # Prioritize starting from failed_generation if present
    idx = error_msg.find("failed_generation")
    search_positions = [idx] if idx != -1 else []
    search_positions.append(0)

    for pos in search_positions:
        start = error_msg.find("{", pos)
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(error_msg)):
                char = error_msg[i]
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            raw = error_msg[start:i+1]
                            for candidate in [raw, raw.replace('\\n', '\n').replace('\\"', '"')]:
                                try:
                                    parsed = json.loads(candidate)
                                    if isinstance(parsed, dict) and "error" not in parsed:
                                        return candidate
                                except Exception:
                                    pass
                            try:
                                decoded = raw.encode('utf-8').decode('unicode_escape')
                                parsed = json.loads(decoded)
                                if isinstance(parsed, dict) and "error" not in parsed:
                                    return decoded
                            except Exception:
                                pass
                            break
            start = error_msg.find("{", start + 1)
    return None

def invoke_with_retry_and_repair(prompt_template, llm, schema, inputs):
    """Invokes the LLM with structured output, retrying on failure and using a repair prompt if needed."""
    time.sleep(1.5)
    chain = prompt_template | llm.with_structured_output(schema)
    
    for attempt in range(8):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_msg = str(e)
            
            # Attempt instant JSON extraction from error message
            json_str = extract_json_from_error(error_msg)
            if json_str:
                try:
                    data = json.loads(json_str)
                    return parse_dict_to_schema(schema, data)
                except Exception:
                    pass

            # If rate limit 429 error, sleep 12s for TPM window reset and retry
            if "429" in error_msg or "rate_limit" in error_msg:
                time.sleep(12)
                continue

            # Break to fallback repair
            break

    # Fallback to repair prompt
    schema_schema = json.dumps(schema.model_json_schema(), indent=2)
    try:
        original_content = prompt_template.format(**inputs)
    except Exception:
        original_content = f"Inputs: {inputs}"
        
    repair_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert JSON repair assistant. A previous attempt to parse a structured output failed.\n"
                   "Your job is to generate the correct structured JSON output matching the requested schema.\n"
                   "Do not output conversational text."),
        ("human", "Here is the schema we need to match:\n{schema_schema}\n\n"
                  "Context:\n{original_content}\n\n"
                  "Please generate the corrected JSON object matching the schema.")
    ])
    
    repair_chain = repair_prompt | llm.with_structured_output(schema)
    for attempt in range(3):
        try:
            return repair_chain.invoke({
                "schema_schema": schema_schema,
                "original_content": original_content
            })
        except Exception as rep_e:
            rep_msg = str(rep_e)
            json_str = extract_json_from_error(rep_msg)
            if json_str:
                data = json.loads(json_str)
                return parse_dict_to_schema(schema, data)
            if attempt == 2:
                return parse_dict_to_schema(schema, {})
    return parse_dict_to_schema(schema, {})

# Agent 1: Document Analyzer
def analyze_document(text: str, llm=None, api_key: str = None):
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="document_analyzer", api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cautious and highly accurate document analyst. You must format your output using the designated tool schema and NEVER output conversational text."),
        ("human", "Your task is to identify the EXACT specific category/type of the document provided.\n\n"
        "Instructions:\n"
        "1. Identify the specific document category (e.g. 'Non-Disclosure Agreement', 'Employment Agreement', 'Residential Lease Agreement', 'Commercial Lease Agreement', 'Independent Contractor Agreement', 'SaaS Terms of Service', 'Service Level Agreement', 'Software Development Agreement', 'Promissory Note', 'Master Services Agreement', 'Software License Agreement', 'Equipment Loan Agreement', 'Liability Waiver', 'Consulting Agreement', 'Invoice', 'University Admit Card', 'Recipe', 'Shopping List', 'Travel Itinerary', 'Medical Prescription', 'Work Schedule', 'Recommendation Letter', etc.).\n"
        "2. If it is a legal contract, state its specific contract classification (e.g. 'Non-Disclosure Agreement', 'Employment Agreement').\n"
        "3. Return 'key_clauses' as a list of strings summarizing the main clauses or data points identified.\n\n"
        "Document: {text}")
    ])
    
    chunks = split_text_by_sections(text, max_chars=30000)
    results = []
    for chunk in chunks:
        res = invoke_with_retry_and_repair(prompt, llm, DocumentAnalysis, {"text": chunk})
        results.append(res)
        
    valid_results = [r for r in results if r and getattr(r, "document_type", "").strip()]
    if len(valid_results) == 1:
        return valid_results[0]
    if valid_results:
        types = [r.document_type for r in valid_results if r.document_type.strip()]
        doc_type = max(set(types), key=types.count) if types else "Contract / Document"
        key_clauses = []
        seen = set()
        for r in valid_results:
            for clause in (getattr(r, "key_clauses", []) or []):
                if clause not in seen:
                    seen.add(clause)
                    key_clauses.append(clause)
        return DocumentAnalysis(document_type=doc_type, key_clauses=key_clauses)
    return DocumentAnalysis(document_type="Contract / Document", key_clauses=[])

# Agent 2: Legal Classifier
def classify_legal(text: str, llm=None, api_key: str = None):
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="legal_classifier", api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal classifier. You must format your output using the designated tool schema and NEVER output conversational text."),
        ("human", "Determine whether the following document is a legal document (contract, agreement, policy, terms of service, liability waiver, or binding terms).\n\n"
        "CLASSIFICATION RULES:\n\n"
        "1. LEGAL DOCUMENTS ('Legal' -> is_legal=True):\n"
        "   - Standard commercial contracts and agreements, including templates with standard placeholders (NDAs, Leases, Employment Contracts, SLAs, Service Agreements, Software Licenses, Promissory Notes, Liability Waivers).\n"
        "   - Online Terms of Service, Privacy Policies, and EULAs (including documents containing prompt injections formatted as terms).\n"
        "   - Invoices that explicitly include formal binding contract terms and fine print purchase agreements.\n\n"
        "2. NON-LEGAL DOCUMENTS ('Non-Legal' -> is_legal=False):\n"
        "   - Standard retail invoices, receipts, shopping lists, recipes, personal letters, duty schedules, travel itineraries, admit cards, medical prescriptions, and recommendation letters.\n"
        "   - Academic & Educational Framing: Law school exam questions, textbook excerpts, case study exercises (e.g., 'QUESTION 1:', 'Law School Exam', 'For the purposes of this problem').\n"
        "   - Fictional / Sci-Fi Framing: Fictional space treaties or sci-fi accords (e.g., 'Intergalactic Treaty of Alpha Centauri', 'Starfleet Accord') with no real-world legal context.\n"
        "   - Non-Binding Agreements: Explicitly non-binding Letters of Intent (LOIs).\n\n"
        "FEW-SHOT EXAMPLES:\n\n"
        "Example 1 (Real Contract):\n"
        "Document: 'RESIDENTIAL LEASE AGREEMENT. Landlord agrees to rent the premises to Tenant for $1,500/month. Security deposit of $1,500 required...'\n"
        "Classification: Legal (is_legal=True)\n"
        "Reason: Binding residential lease contract establishing rental obligations.\n\n"
        "Example 2 (Academic Exam Question):\n"
        "Document: 'LAW SCHOOL FINAL EXAM - QUESTION 4: Party A enters into a contract with Party B to purchase 100 widgets. Analyze whether Party A can claim breach...'\n"
        "Classification: Non-Legal (is_legal=False)\n"
        "Reason: Academic law school examination question framing a hypothetical exercise.\n\n"
        "Example 3 (Fictional Treaty):\n"
        "Document: 'INTERGALACTIC PEACE TREATY OF ALPHA CENTAURI (3042). The United Federation of Planets hereby agrees with the Empire of Romulus to cede Starbase 9...'\n"
        "Classification: Non-Legal (is_legal=False)\n"
        "Reason: Fictional sci-fi treaty with no real-world legal enforceability.\n\n"
        "Document to classify:\n{text}")
    ])
    
    chunks = split_text_by_sections(text, max_chars=30000)
    results = []
    for chunk in chunks:
        res = invoke_with_retry_and_repair(prompt, llm, LegalClassification, {"text": chunk})
        results.append(res)
        
    valid_results = [r for r in results if r and getattr(r, "is_legal", None) is not None]
    if len(valid_results) == 1:
        return valid_results[0]
    if valid_results:
        is_legal = any(r.is_legal for r in valid_results)
        classification = "Legal" if is_legal else "Non-Legal"
        reasons = [r.reason for r in valid_results if getattr(r, "reason", None)]
        reason = "; ".join(dict.fromkeys(reasons)) if reasons else "Document classification completed."
        explanations = [r.explanation for r in valid_results if getattr(r, "explanation", None)]
        explanation = "\n".join(dict.fromkeys(explanations)) if explanations else None
        return LegalClassification(
            is_legal=is_legal,
            classification=classification,
            reason=reason,
            explanation=explanation
        )
    return LegalClassification(
        is_legal=True,
        classification="Legal",
        reason="Default legal classification fallback.",
        explanation=None
    )

# Agent 3: Risk Detector
def detect_risks(text: str, llm=None, api_key: str = None):
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if llm is None:
        llm = get_llm(agent_name="risk_detector", api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert risk assessor. You strictly output tool calls without extra prefixes or suffixes."),
        ("human", "Identify any potential risks, unfair clauses, or legal issues in the following document.\n"
        "Focus on: Ambiguous terms, one-sided clauses, missing protections, and legal risks.\n\n"
        "IMPORTANT: You must return the 'risks' as a list of strings (an array). If no risks are found, return [].\n\n"
        "{reference_context}"
        "Document: {text}")
    ])
    
    chunks = split_text_by_sections(text, max_chars=30000)
    results = []
    for chunk in chunks:
        # Retrieve references if api_key is available
        reference_clauses = []
        if api_key:
            reference_clauses = retrieve_reference_clauses(chunk, api_key, k=3)
        
        reference_context = ""
        if reference_clauses:
            ref_str = "\n".join(f"- {c}" for c in reference_clauses)
            reference_context = f"STANDARD REFERENCE CLAUSES FOR COMPARISON:\n{ref_str}\n\n"
            
        res = invoke_with_retry_and_repair(
            prompt, 
            llm, 
            RiskAssessment, 
            {"text": chunk, "reference_context": reference_context}
        )
        results.append(res)
        
    if len(results) == 1:
        return results[0]
        
    # Merge results
    risks = []
    seen = set()
    for r in results:
        for risk in r.risks:
            if risk not in seen:
                seen.add(risk)
                risks.append(risk)
                
    return RiskAssessment(risks=risks)

# Agent 4: Explanation Agent
def explain_document(text: str, llm=None, api_key: str = None):
    if llm is None:
        llm = get_llm(agent_name="explanation_agent", api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a legal summarizer. Always respond via the provided tool schema. Never hallucinate raw text like '<function='."),
        ("human", "Explain the following document in simple terms so a non-legal person can understand it.\n"
        "Avoid legal jargon. Use simple language and short sentences.\n"
        "Provide a brief summary and then the simplified explanation.\n\n"
        "Document: {text}")
    ])
    
    chunks = split_text_by_sections(text, max_chars=30000)
    results = []
    for chunk in chunks:
        res = invoke_with_retry_and_repair(prompt, llm, SimplifiedExplanation, {"text": chunk})
        results.append(res)
        
    if len(results) == 1:
        return results[0]
        
    # Merge results
    summaries = [r.summary for r in results if r.summary]
    explanations = [r.simplified_explanation for r in results if r.simplified_explanation]
    
    summary = "\n\n".join(summaries)
    simplified_explanation = "\n\n".join(explanations)
    
    return SimplifiedExplanation(
        summary=summary,
        simplified_explanation=simplified_explanation
    )

class RiskReview(BaseModel):
    risk: str = Field(description="The original risk description being reviewed")
    verdict: str = Field(description="Strictly 'Confirmed', 'Downgraded', or 'Removed'")
    reason: str = Field(description="Detailed reason or explanation for this verdict")

class CriticVerdict(BaseModel):
    reviews: List[RiskReview] = Field(description="A list of reviews, one for each flagged risk.")

def review_risks(text: str, risks: List[str], llm=None, api_key: str = None) -> CriticVerdict:
    """Reviews the list of identified risks against the document and returns a structured verdict/reasoning for each."""
    if not risks:
        return CriticVerdict(reviews=[])
    if llm is None:
        llm = get_llm(agent_name="critic", api_key=api_key)
        
    reference_clauses = []
    if api_key:
        reference_clauses = retrieve_reference_clauses(text[:3000], api_key, k=3)
    
    reference_context = ""
    if reference_clauses:
        ref_str = "\n".join(f"- {c}" for c in reference_clauses)
        reference_context = f"STANDARD REFERENCE CLAUSES FOR COMPARISON:\n{ref_str}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict, senior legal auditor. Your sole purpose is to audit and filter a list of candidate risks flagged by an automated detector.\n\n"
         "You must evaluate each flagged risk against these STRICT AUDITING CRITERIA:\n\n"
         "1. DOWNGRADE OR REMOVE (VERDICT: 'Downgraded' or 'Removed'):\n"
         "   - Standard / Customary Terms: The clause is standard boilerplate for this document type (e.g. an NDA having a 2-3 year confidentiality period, or an employment contract stating at-will employment) and contains no unusual or harsh burdens. DO NOT flag standard boilerplate as a risk.\n"
         "   - Mere Clause Restatements: The flag simply restates what a clause says without identifying an actual legal hazard, one-sided liability, or missing protection.\n"
         "   - Exaggerated Severity: The flag claims a catastrophic risk for a standard commercial practice (e.g. calling a 30-day notice period an 'unreasonable delay').\n"
         "   - Generic Non-Issues: The flag uses vague language like 'may cause confusion' without specifying a real legal liability.\n\n"
         "2. CONFIRM (VERDICT: 'Confirmed'):\n"
         "   - Genuine Asymmetry or Overreach: One party bears severe, uncapped, or one-sided obligations (e.g. unilateral indemnity, unlimited consequential damages, 120-day payment delay, 3-year global non-compete, landlord entry without notice).\n"
         "   - Material Missing Protection: Omission of critical statutory or commercial safeguards (e.g. missing cap on damages, missing return of materials clause, usurious 35% interest rate).\n"
         "   - Severe Ambiguity: Terms so vague they create legal exposure or litigation risk.\n\n"
         "FEW-SHOT EXAMPLES:\n"
         "Example 1 (Standard Terms -> DOWNGRADED):\n"
         "Flagged Risk: 'The agreement contains a standard confidentiality clause requiring recipient to protect confidential information for 3 years.'\n"
         "Verdict: 'Downgraded'\n"
         "Reason: 'A 3-year confidentiality term is customary in commercial NDAs and does not constitute a legal risk.'\n\n"
         "Example 2 (Mere Restatement / Non-Issue -> REMOVED):\n"
         "Flagged Risk: 'The contract specifies that payment is due within 30 days of invoice receipt.'\n"
         "Verdict: 'Removed'\n"
         "Reason: 'Net-30 payment terms are standard commercial practice and present no legal risk or asymmetry.'\n\n"
         "Example 3 (Genuine Legal Risk -> CONFIRMED):\n"
         "Flagged Risk: 'The landlord may enter the leased premises at any time without prior notice.'\n"
         "Verdict: 'Confirmed'\n"
         "Reason: 'Unannounced entry violates statutory tenant privacy rights and quiet enjoyment standards.'\n\n"
         "You MUST provide a verdict ('Confirmed', 'Downgraded', or 'Removed') and a concise, objective legal reason for every flagged risk."),
        ("human", "DOCUMENT CONTEXT:\n{text}\n\n"
                  "{reference_context}"
                  "FLAGGED RISKS TO AUDIT:\n{risks_list}\n\n"
                  "Output the structured list of reviews.")
    ])
    
    risks_list = "\n".join(f"- {r}" for r in risks)
    
    return invoke_with_retry_and_repair(
        prompt, 
        llm, 
        CriticVerdict, 
        {"text": text, "reference_context": reference_context, "risks_list": risks_list}
    )

class ClauseComparison(BaseModel):
    original_clause: str = Field(description="The text of the clause in the original document, or 'N/A' if newly added")
    modified_clause: str = Field(description="The text of the matching clause in the modified document, or 'N/A' if deleted")
    change_description: str = Field(description="Description of what was changed/modified/added/removed")
    risk_direction: str = Field(description="Strictly 'Increase', 'Decrease', or 'No Change'")
    reason: str = Field(description="A one-line reason explaining the risk impact")

class ComparisonReport(BaseModel):
    comparisons: List[ClauseComparison] = Field(description="A list of semantically aligned clause comparisons.")

def compare_documents(text_a: str, text_b: str, llm=None, api_key: str = None) -> ComparisonReport:
    """Semantically aligns clauses between two documents and compares changes for risk impact."""
    if llm is None:
        llm = get_llm(agent_name="comparator", api_key=api_key)
    from retriever import GroqEmbeddings
    import math
    
    # 1. Split both documents into paragraphs/clauses
    clauses_a = [c.strip() for c in split_text_by_sections(text_a, max_chars=2000) if c.strip()]
    clauses_b = [c.strip() for c in split_text_by_sections(text_b, max_chars=2000) if c.strip()]
    
    pairs = []
    
    if api_key and clauses_a and clauses_b:
        try:
            embeddings_model = GroqEmbeddings(api_key=api_key)
            embeds_a = embeddings_model.embed_documents(clauses_a)
            embeds_b = embeddings_model.embed_documents(clauses_b)
            
            def cosine_similarity(v1, v2):
                dot_product = sum(x*y for x, y in zip(v1, v2))
                magnitude1 = math.sqrt(sum(x*x for x in v1))
                magnitude2 = math.sqrt(sum(x*x for x in v2))
                if magnitude1 == 0 or magnitude2 == 0:
                    return 0.0
                return dot_product / (magnitude1 * magnitude2)
                
            matched_b_indices = set()
            for idx_a, v_a in enumerate(embeds_a):
                best_sim = -1.0
                best_idx_b = -1
                for idx_b, v_b in enumerate(embeds_b):
                    sim = cosine_similarity(v_a, v_b)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx_b = idx_b
                
                # If similarity threshold is met, align them
                if best_sim >= 0.55:
                    pairs.append((clauses_a[idx_a], clauses_b[best_idx_b]))
                    matched_b_indices.add(best_idx_b)
                else:
                    # Clause was removed
                    pairs.append((clauses_a[idx_a], None))
                    
            # Any unmatched clauses in B are additions
            for idx_b, clause_b in enumerate(clauses_b):
                if idx_b not in matched_b_indices:
                    pairs.append((None, clause_b))
        except Exception as e:
            # Fallback to simple matching (pairwise zip) if embedding fails
            print(f"Embedding alignment failed, falling back to sequential alignment: {e}")
            min_len = min(len(clauses_a), len(clauses_b))
            for i in range(min_len):
                pairs.append((clauses_a[i], clauses_b[i]))
            for i in range(min_len, len(clauses_a)):
                pairs.append((clauses_a[i], None))
            for i in range(min_len, len(clauses_b)):
                pairs.append((None, clauses_b[i]))
    else:
        # Zip fallback
        min_len = min(len(clauses_a), len(clauses_b))
        for i in range(min_len):
            pairs.append((clauses_a[i], clauses_b[i]))
        for i in range(min_len, len(clauses_a)):
            pairs.append((clauses_a[i], None))
        for i in range(min_len, len(clauses_b)):
            pairs.append((None, clauses_b[i]))

    # If no pairs are found, return empty report
    if not pairs:
        return ComparisonReport(comparisons=[])
        
    # Group pairs into batches to query the LLM efficiently
    batch_size = 5
    comparison_results = []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert contract comparison analyst. Your task is to review proposed changes between an original and a modified version of a contract.\n"
                   "For each pair of clauses provided, determine:\n"
                   "1. What changed (describe the modification, deletion, or addition).\n"
                   "2. The direction of risk change (strictly 'Increase', 'Decrease', or 'No Change').\n"
                   "3. A concise one-line reason explaining the risk impact.\n\n"
                   "Do not output conversational text."),
        ("human", "Analyze these contract clauses:\n\n{clauses_input}")
    ])
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        clauses_input = ""
        for j, (orig, mod) in enumerate(batch):
            orig_txt = orig if orig else "[NO ORIGINAL CLAUSE - NEWLY ADDED]"
            mod_txt = mod if mod else "[DELETED IN MODIFIED VERSION]"
            clauses_input += f"--- Clause Pair {j+1} ---\nORIGINAL:\n{orig_txt}\n\nMODIFIED:\n{mod_txt}\n\n"
            
        res = invoke_with_retry_and_repair(
            prompt,
            llm,
            ComparisonReport,
            {"clauses_input": clauses_input}
        )
        if res and res.comparisons:
            comparison_results.extend(res.comparisons)
            
    return ComparisonReport(comparisons=comparison_results)
