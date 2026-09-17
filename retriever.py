import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional


class LegalClauseRetriever:
    """
    In-memory TF-IDF and N-gram cosine vector retrieval engine specialized for standard legal clauses.
    Provides fast, deterministic, zero-hallucination reference retrieval with similarity score thresholds.
    """
    def __init__(self, corpus: List[Dict[str, Any]] = None):
        self.corpus = corpus or STANDARD_CLAUSE_TEMPLATES
        self.documents = [item["text"] for item in self.corpus]
        self.metadatas = [item.get("metadata", {}) for item in self.corpus]
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize words + 2-grams for legal phrase matching
        words = re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _build_index(self):
        doc_tokens_list = [self._tokenize(doc) for doc in self.documents]
        n_docs = len(self.documents)
        
        # Build vocabulary & Document Frequencies
        df = {}
        for tokens in doc_tokens_list:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1

        self.vocab = {t: idx for idx, t in enumerate(df.keys())}
        self.idf = {t: math.log((n_docs + 1) / (df[t] + 1)) + 1.0 for t in df}

        # Build TF-IDF vectors for documents
        self.doc_vectors = []
        for tokens in doc_tokens_list:
            vec = self._compute_tfidf_vector(tokens)
            self.doc_vectors.append(vec)

    def _compute_tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        
        vec = {}
        norm_sq = 0.0
        for t, count in tf.items():
            if t in self.idf:
                weight = (1 + math.log(count)) * self.idf[t]
                vec[t] = weight
                norm_sq += weight * weight
        
        # L2 normalize
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        for t in vec:
            vec[t] /= norm
        return vec

    def similarity_search_with_score(self, query: str, k: int = 3, min_score: float = 0.12) -> List[Dict[str, Any]]:
        """
        Searches the standard legal corpus for clauses matching the query.
        Returns top k items exceeding min_score.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
            
        q_vec = self._compute_tfidf_vector(q_tokens)
        if not q_vec:
            return []

        scored_results = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            # Compute cosine similarity (dot product of normalized vectors)
            score = 0.0
            for t, w in q_vec.items():
                if t in doc_vec:
                    score += w * doc_vec[t]
                    
            if score >= min_score:
                meta = self.metadatas[idx].copy()
                scored_results.append({
                    "text": self.documents[idx],
                    "score": round(score, 4),
                    "clause_type": meta.get("clause_type", "Standard Provision"),
                    "category": meta.get("category", "General Commercial"),
                    "baseline_standard": meta.get("standard", "Standard commercial baseline provision.")
                })

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:k]


# Standard Legal Clause Baseline Knowledge Corpus (25 Comprehensive Standards)
STANDARD_CLAUSE_TEMPLATES = [
    # 1. NDA Confidentiality Definition
    {
        "text": "Confidential Information refers to all non-public, proprietary information disclosed by one party to the other, marked as confidential or that reasonably should be understood to be confidential given the context.",
        "metadata": {"clause_type": "Confidentiality Definition", "category": "NDA", "standard": "Standard bilateral definition requiring explicit marking or clear context."}
    },
    # 2. NDA Exclusions
    {
        "text": "Confidential Information does not include information that: (a) is or becomes publicly known through no breach; (b) was already in recipient's possession without restriction; (c) is independently developed without reference to discloser's confidential information.",
        "metadata": {"clause_type": "Confidentiality Exclusions", "category": "NDA", "standard": "Essential 3-prong standard exclusions protecting recipient from unbounded liability."}
    },
    # 3. NDA Reasonable Term
    {
        "text": "The obligations of confidentiality under this Agreement shall survive termination and continue for a period of two (2) to three (3) years from the date of disclosure, except for trade secrets which remain protected for so long as they qualify as trade secrets under applicable law.",
        "metadata": {"clause_type": "Confidentiality Duration", "category": "NDA", "standard": "Standard commercial confidentiality duration is 2-3 years, not perpetual."}
    },
    # 4. Return or Destruction of Materials
    {
        "text": "Upon written request, the receiving party shall promptly return or certify the destruction of all Confidential Information, provided that recipient may retain one archival copy solely for compliance, regulatory, or audit purposes.",
        "metadata": {"clause_type": "Return of Materials", "category": "NDA", "standard": "Standard return/destruction with customary compliance archive carve-out."}
    },
    # 5. Mutual Indemnification
    {
        "text": "Each party shall defend, indemnify, and hold harmless the other party, its officers, and employees from and against any third-party claims, liabilities, or damages arising directly from material breach of this Agreement, gross negligence, or willful misconduct.",
        "metadata": {"clause_type": "Indemnification", "category": "Indemnity", "standard": "Mutual, bilateral indemnification limited to third-party claims and material breach/negligence."}
    },
    # 6. Consequential Damages Waiver
    {
        "text": "Neither party shall be liable to the other for any indirect, incidental, special, consequential, or punitive damages, including loss of profits, data, or business interruption, arising out of this Agreement.",
        "metadata": {"clause_type": "Consequential Damages Waiver", "category": "Limitation of Liability", "standard": "Mutual waiver of indirect/consequential damages is standard market practice."}
    },
    # 7. Liability Cap (Fee-Based)
    {
        "text": "Each party's maximum aggregate liability arising out of or related to this Agreement shall be strictly capped at the total amount of fees paid or payable by Customer in the twelve (12) month period preceding the incident giving rise to liability.",
        "metadata": {"clause_type": "Liability Cap", "category": "Limitation of Liability", "standard": "Standard 12-month fees paid aggregate cap protects both parties from unbounded financial exposure."}
    },
    # 8. Termination for Cause (Cure Period)
    {
        "text": "Either party may terminate this Agreement immediately upon written notice if the other party materially breaches any provision and fails to cure such breach within thirty (30) calendar days of receiving written notice.",
        "metadata": {"clause_type": "Termination for Cause", "category": "Termination", "standard": "Standard 30-day cure period before unilateral contract cancellation."}
    },
    # 9. Termination for Convenience
    {
        "text": "Either party may terminate this Agreement for convenience and without cause by providing at least thirty (30) to sixty (60) days prior written notice to the other party.",
        "metadata": {"clause_type": "Termination for Convenience", "category": "Termination", "standard": "Bilateral termination for convenience with reasonable 30-60 days advance notice."}
    },
    # 10. Commercial Payment Terms (Net-30)
    {
        "text": "Invoices are payable within thirty (30) days of receipt (Net-30). Overdue balances shall accrue interest at the lesser of 1.0% to 1.5% per month or the maximum rate permitted by law.",
        "metadata": {"clause_type": "Payment Terms", "category": "Payment", "standard": "Standard Net-30 payment terms and statutory non-usurious interest caps (max 1.5%/month)."}
    },
    # 11. Fair Residential Security Deposit
    {
        "text": "Landlord shall hold the Security Deposit in an interest-bearing escrow account and return the full deposit within twenty-one (21) to thirty (30) days of lease termination, minus documented deductions for damage exceeding normal wear and tear.",
        "metadata": {"clause_type": "Security Deposit", "category": "Lease", "standard": "Statutory tenant protection requiring deposit return and banning unconditional forfeiture."}
    },
    # 12. Landlord Entry with 24h Notice
    {
        "text": "Landlord may enter the leased premises only during reasonable business hours upon providing at least twenty-four (24) hours advance written notice, except in genuine emergencies threatening life or structural integrity.",
        "metadata": {"clause_type": "Landlord Inspection", "category": "Lease", "standard": "Standard covenant of quiet enjoyment requiring mandatory 24-hour advance entry notice."}
    },
    # 13. Balanced Lease Renewal Terms
    {
        "text": "Upon expiration of the initial term, this Lease may be renewed upon mutual written agreement, or continue on a month-to-month basis with either party having the right to terminate upon thirty (30) days advance written notice.",
        "metadata": {"clause_type": "Lease Renewal", "category": "Lease", "standard": "Month-to-month or bilateral renewal, preventing multi-year lock-in automatic traps."}
    },
    # 14. Employment At-Will and Separation
    {
        "text": "Employment is at-will, allowing either party to terminate the employment relationship at any time, with or without cause, upon reasonable advance notice. Company shall pay all accrued wages and benefits upon departure.",
        "metadata": {"clause_type": "Employment Termination", "category": "Employment", "standard": "Standard at-will employment with prompt settlement of earned wages."}
    },
    # 15. Reasonable Non-Compete & Non-Solicit
    {
        "text": "Employee agrees not to solicit company employees or customers for twelve (12) months following separation. Any restrictive covenant must be narrowly tailored in geographic scope and duration to protect legitimate business interests.",
        "metadata": {"clause_type": "Restrictive Covenants", "category": "Employment", "standard": "Restrictive covenants must be limited to 12 months and reasonable geographic scope."}
    },
    # 16. Work for Hire & IP Assignment
    {
        "text": "Employee assigns to Company all right, title, and interest in inventions and works created within the scope of employment and using company resources. Inventions developed on personal time without company resources remain employee's property.",
        "metadata": {"clause_type": "IP Ownership", "category": "Intellectual Property", "standard": "Standard carve-out for personal time inventions under California/statutory labor standards."}
    },
    # 17. Software / SaaS SLA and Remedies
    {
        "text": "Provider will maintain 99.9% service availability. In the event of unscheduled downtime exceeding permitted SLA thresholds, Customer shall be eligible for proportional service credits, and the right to terminate if downtime persists.",
        "metadata": {"clause_type": "SLA Remedy", "category": "SaaS", "standard": "Standard proportional fee credits and termination rights for chronic service outages."}
    },
    # 18. Data Protection & Privacy (GDPR / CCPA)
    {
        "text": "Vendor agrees to implement appropriate administrative, technical, and physical safeguards to protect Personal Data, comply with applicable privacy laws (GDPR/CCPA), and notify Customer of any confirmed data breach within forty-eight (48) hours.",
        "metadata": {"clause_type": "Data Protection", "category": "Compliance", "standard": "Mandatory 48-72h security breach notice and statutory privacy compliance."}
    },
    # 19. Governing Law & Jurisdiction
    {
        "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware (or mutual domicile), without regard to conflict of laws principles. The parties consent to jurisdiction in state or federal courts located therein.",
        "metadata": {"clause_type": "Governing Law", "category": "Dispute Resolution", "standard": "Standard domestic governing law (e.g. Delaware, New York, California) in mutual commercial courts."}
    },
    # 20. Dispute Resolution (Arbitration)
    {
        "text": "Any dispute arising under this Agreement shall first be submitted to good-faith mediation. If unresolved, the dispute shall be settled by binding arbitration administered by the American Arbitration Association (AAA) under its Commercial Arbitration Rules.",
        "metadata": {"clause_type": "Dispute Resolution", "category": "Dispute Resolution", "standard": "Standard multi-tier mediation followed by neutral AAA/JAMS binding arbitration."}
    },
    # 21. Severability
    {
        "text": "If any provision of this Agreement is held invalid or unenforceable, such provision shall be modified to the minimum extent necessary, and the remaining provisions shall remain in full force and effect.",
        "metadata": {"clause_type": "Severability", "category": "Boilerplate", "standard": "Standard severability provision preserving remainder of contract."}
    },
    # 22. Entire Agreement & Amendments
    {
        "text": "This Agreement constitutes the entire agreement between the parties with respect to its subject matter. No amendment or modification shall be effective unless executed in writing by authorized representatives of both parties.",
        "metadata": {"clause_type": "Entire Agreement", "category": "Boilerplate", "standard": "Integration clause preventing oral modifications or conflicting informal emails."}
    },
    # 23. Force Majeure
    {
        "text": "Neither party will be liable for delay or failure in performance due to causes beyond its reasonable control, such as acts of God, strikes, natural disasters, or governmental embargoes, provided prompt written notice is given.",
        "metadata": {"clause_type": "Force Majeure", "category": "Boilerplate", "standard": "Standard mutual force majeure excusing performance during unforeseeable catastrophes."}
    },
    # 24. Assignment Restriction
    {
        "text": "Neither party may assign this Agreement without the prior written consent of the other, except in connection with a merger, acquisition, or sale of substantially all assets, provided the assignee assumes all obligations.",
        "metadata": {"clause_type": "Assignment", "category": "Boilerplate", "standard": "Mutual assignment restriction with standard M&A / corporate restructuring exception."}
    },
    # 25. Promissory Note Reasonable Interest & Acceleration
    {
        "text": "Borrower promises to pay Principal with simple interest at a rate not exceeding 8.0% per annum. In the event of default, Lender may accelerate the remaining balance only after providing thirty (30) days written notice and opportunity to cure.",
        "metadata": {"clause_type": "Loan Terms", "category": "Finance", "standard": "Reasonable commercial loan interest rate and mandatory 30-day default notice before acceleration."}
    }
]

# Singleton Global Retriever Store Instance
_GLOBAL_RETRIEVER: Optional[LegalClauseRetriever] = None

def get_retriever() -> LegalClauseRetriever:
    """Returns the singleton instance of the LegalClauseRetriever."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = LegalClauseRetriever(STANDARD_CLAUSE_TEMPLATES)
    return _GLOBAL_RETRIEVER


def retrieve_reference_clauses_with_metadata(query: str, k: int = 3, min_score: float = 0.12) -> List[Dict[str, Any]]:
    """
    Retrieves standard legal baseline reference clauses matching the query clause with metadata and similarity scores.
    If no relevant references are found, returns a structured indicator to prevent LLM hallucinations.
    """
    retriever = get_retriever()
    matches = retriever.similarity_search_with_score(query, k=k, min_score=min_score)
    if not matches:
        return [{
            "text": "Insufficient retrieved evidence",
            "score": 0.0,
            "clause_type": "None",
            "category": "None",
            "baseline_standard": "No sufficiently relevant standard baseline clause was found in the reference knowledge base."
        }]
    return matches


def retrieve_reference_clauses(text: str, api_key: str = None, k: int = 3) -> List[str]:
    """
    Backward-compatible helper returning a list of matched reference text strings.
    """
    matches = retrieve_reference_clauses_with_metadata(text, k=k)
    return [m["text"] for m in matches if m["text"] != "Insufficient retrieved evidence"]
