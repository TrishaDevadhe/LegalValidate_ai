import os
import requests
from typing import List, Tuple
from langchain_core.embeddings import Embeddings

# Groq-based embedding model using nomic-embed-text-v1.5
class GroqEmbeddings(Embeddings):
    def __init__(self, api_key=None, model="nomic-embed-text-v1.5"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.url = "https://api.groq.com/openai/v1/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": texts,
            "model": self.model
        }
        res = requests.post(self.url, json=payload, headers=headers, timeout=2)
        res.raise_for_status()
        data = res.json()
        return [item["embedding"] for item in data["data"]]
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Simple In-Memory Vector Store Fallback
class SimpleVectorStore:
    def __init__(self, embeddings_model=None):
        self.embeddings_model = embeddings_model
        self.documents = []
        self.metadatas = []
        
    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        self.documents.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])
            
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[float, str, dict]]:
        import re
        def tokenize(txt):
            return set(re.findall(r'\w+', str(txt).lower()))
            
        q_tokens = tokenize(query)
        if not q_tokens:
            return [(0.0, doc, meta) for doc, meta in zip(self.documents[:k], self.metadatas[:k])]
            
        scores = []
        for i, doc in enumerate(self.documents):
            d_tokens = tokenize(doc)
            intersection = q_tokens.intersection(d_tokens)
            union = q_tokens.union(d_tokens)
            sim = len(intersection) / len(union) if union else 0.0
            scores.append((sim, doc, self.metadatas[i]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

# Standard Clause Templates Seed Corpus (18 example reference clauses)
# (Included in full below)

def get_retriever_store(api_key: str = None):
    """
    Initializes and returns an in-memory vector store seeded with standard legal clause templates.
    Uses singleton caching for efficiency.
    """
    global _RETRIEVER_STORE
    if _RETRIEVER_STORE is not None:
        return _RETRIEVER_STORE
        
    try:
        store = SimpleVectorStore()
        texts = [c["text"] for c in STANDARD_CLAUSE_TEMPLATES]
        metadatas = [c["metadata"] for c in STANDARD_CLAUSE_TEMPLATES]
        store.add_texts(texts, metadatas=metadatas)
        _RETRIEVER_STORE = store
    except Exception as e:
        print(f"Retriever initialization failed: {e}")
        _RETRIEVER_STORE = None
        
    return _RETRIEVER_STORE

# Standard Clause Templates Seed Corpus (18 example reference clauses)
STANDARD_CLAUSE_TEMPLATES = [
    # 1. NDA Confidentiality
    {
        "text": "Confidential Information refers to all proprietary and non-public information disclosed by one party to the other, whether orally or in writing, that is designated as confidential or that reasonably should be understood to be confidential given the nature of the information.",
        "metadata": {"clause_type": "Confidentiality Definition", "category": "NDA"}
    },
    # 2. NDA Exclusions
    {
        "text": "Confidential Information shall not include information that: (a) is or becomes publicly known through no breach of this Agreement; (b) was already in the receiving party's possession prior to disclosure; (c) is independently developed without use of or reference to the disclosing party's Confidential Information.",
        "metadata": {"clause_type": "Confidentiality Exclusions", "category": "NDA"}
    },
    # 3. NDA Term
    {
        "text": "The obligations of confidentiality under this Agreement shall survive the termination of this Agreement and remain in effect for a period of three (3) years from the date of disclosure.",
        "metadata": {"clause_type": "Confidentiality Duration", "category": "NDA"}
    },
    # 4. NDA Return of Materials
    {
        "text": "Upon written request of the disclosing party, the receiving party shall promptly return or destroy all documents, materials, and copies containing Confidential Information, and certify such destruction in writing.",
        "metadata": {"clause_type": "Return of Materials", "category": "NDA"}
    },
    # 5. Indemnification
    {
        "text": "Each party agrees to indemnify, defend, and hold harmless the other party from and against any and all claims, liabilities, losses, damages, or costs (including reasonable attorneys' fees) arising out of or relating to its material breach of this Agreement or its gross negligence.",
        "metadata": {"clause_type": "Indemnification Obligations", "category": "Indemnification"}
    },
    # 6. Limitation of Liability
    {
        "text": "In no event shall either party be liable to the other for any indirect, incidental, consequential, special, or exemplary damages, including lost profits, arising out of this Agreement, even if advised of the possibility of such damages.",
        "metadata": {"clause_type": "Consequential Damages Waiver", "category": "Limitation of Liability"}
    },
    # 7. Liability Cap
    {
        "text": "Each party's total aggregate liability under this Agreement, whether in contract, tort, or otherwise, shall be strictly limited to the total fees paid or payable by the customer to the provider in the twelve (12) month period preceding the event giving rise to liability.",
        "metadata": {"clause_type": "Liability Cap", "category": "Limitation of Liability"}
    },
    # 8. Termination for Cause
    {
        "text": "Either party may terminate this Agreement immediately upon written notice if the other party fails to cure a material breach within thirty (30) days after receipt of written notice describing such breach in detail.",
        "metadata": {"clause_type": "Termination for Cause", "category": "Termination"}
    },
    # 9. Termination for Convenience
    {
        "text": "Either party may terminate this Agreement without cause upon providing at least thirty (30) days prior written notice to the other party.",
        "metadata": {"clause_type": "Termination for Convenience", "category": "Termination"}
    },
    # 10. Governing Law
    {
        "text": "This Agreement shall be governed by, construed, and enforced in accordance with the laws of the State of Delaware, without giving effect to any conflict of laws principles.",
        "metadata": {"clause_type": "Governing Law", "category": "Governing Law"}
    },
    # 11. Severability
    {
        "text": "If any provision of this Agreement is held to be invalid, illegal, or unenforceable, the remaining provisions of this Agreement shall remain in full force and effect to the maximum extent permitted by law.",
        "metadata": {"clause_type": "Severability", "category": "Boilerplate"}
    },
    # 12. Entire Agreement
    {
        "text": "This Agreement constitutes the entire agreement between the parties regarding its subject matter and supersedes all prior or contemporaneous understandings, discussions, or agreements, whether written or oral.",
        "metadata": {"clause_type": "Entire Agreement", "category": "Boilerplate"}
    },
    # 13. Amendments
    {
        "text": "No amendment, modification, or waiver of any provision of this Agreement shall be valid or binding unless it is made in writing and signed by duly authorized representatives of both parties.",
        "metadata": {"clause_type": "Amendments", "category": "Boilerplate"}
    },
    # 14. Assignment
    {
        "text": "Neither party may assign or transfer any of its rights or obligations under this Agreement without the prior written consent of the other party, which consent shall not be unreasonably withheld.",
        "metadata": {"clause_type": "Assignment Restriction", "category": "Boilerplate"}
    },
    # 15. Force Majeure
    {
        "text": "Neither party shall be liable for any delay or failure in performance under this Agreement due to causes beyond its reasonable control, including natural disasters, acts of war or terrorism, labor strikes, or government regulations.",
        "metadata": {"clause_type": "Force Majeure", "category": "Boilerplate"}
    },
    # 16. Intellectual Property
    {
        "text": "Except as expressly stated herein, each party retains all rights, title, and interest in and to its pre-existing intellectual property. Any intellectual property created or developed solely by a party under this Agreement shall belong exclusively to that party.",
        "metadata": {"clause_type": "IP Ownership", "category": "IP"}
    },
    # 17. Dispute Resolution
    {
        "text": "Any dispute, controversy, or claim arising out of or relating to this Agreement shall be resolved through confidential, binding arbitration in accordance with the rules of the American Arbitration Association (AAA) or JAMS.",
        "metadata": {"clause_type": "Dispute Resolution", "category": "Boilerplate"}
    },
    # 18. Non-Solicitation
    {
        "text": "During the term of this Agreement and for a period of twelve (12) months thereafter, neither party shall, directly or indirectly, solicit for employment any employee of the other party involved in the performance of this Agreement.",
        "metadata": {"clause_type": "Non-Solicitation", "category": "Human Resources"}
    }
]

# Global cache for the initialized vector store
_RETRIEVER_STORE = None
_RETRIEVER_CHECKED = False

def get_retriever_store(api_key=None):
    global _RETRIEVER_STORE, _RETRIEVER_CHECKED
    
    # Check config toggle
    enable_retrieval = os.getenv("ENABLE_RETRIEVAL", "true").lower() == "true"
    if not enable_retrieval:
        return None
        
    if _RETRIEVER_CHECKED:
        return _RETRIEVER_STORE
        
    _RETRIEVER_CHECKED = True
    try:
        store = SimpleVectorStore()
        texts = [c["text"] for c in STANDARD_CLAUSE_TEMPLATES]
        metadatas = [c["metadata"] for c in STANDARD_CLAUSE_TEMPLATES]
        store.add_texts(texts, metadatas=metadatas)
        _RETRIEVER_STORE = store
    except Exception as e:
        print(f"Retriever initialization failed: {e}")
        _RETRIEVER_STORE = None
        
    return _RETRIEVER_STORE


def retrieve_reference_clauses(text: str, api_key: str = None, k: int = 3) -> List[str]:
    """
    Given a clause or section text, retrieves up to k most similar standard legal clause templates
    from the reference vector store.
    """
    store = get_retriever_store(api_key)
    if store is None:
        return []
        
    try:
        results = store.similarity_search(text, k=k)
        retrieved = []
        for item in results:
            if hasattr(item, "page_content"):
                retrieved.append(item.page_content)
            elif isinstance(item, tuple) and len(item) >= 2:
                retrieved.append(str(item[1]))
            else:
                retrieved.append(str(item))
        return retrieved
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
