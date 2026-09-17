# LegalValidate AI

**LegalValidate AI** is a production-grade multi-agent legal document validation, risk intelligence, and redline comparison system powered by **LangGraph**, **LangChain**, and **Groq LLMs**. The platform classifies documents, analyzes contractual structure, detects grounded risks with RAG retrieval against standard baseline templates, verifies risk validity and severity via an independent Critic agent, provides plain-language explanations for non-lawyers, supports human-in-the-loop audit decisions, and performs semantic clause comparisons between contract revisions.

---

## 🏗️ Multi-Agent Architecture & LangGraph DAG

```
                                  [Input Document]
                                         │
                                         ▼
                               [Agent 1: Classifier]
                                  (gpt-oss-20b)
                                         │
                        ┌────────────────┴────────────────┐
                 [is_legal = False]                [is_legal = True]
                        │                                 │
                        ▼                                 ▼
              [Non-Legal Explainer]             [Agent 2: Analyzer]
                        │                          (gpt-oss-20b)
                        │                                 │
                        │                                 ▼
                        │                      [Agent 3: Risk Detector]
                        │                        (qwen3.8-27b + RAG)
                        │                                 │
                        │                                 ▼
                        │                       [Agent 4: Critic Node]
                        │                          (qwen3.8-27b)
                        │                                 │
                        │                                 ▼
                        │                      [Agent 5: Explainer]
                        │                          (gpt-oss-20b)
                        │                                 │
                        │                                 ▼
                        │                    [Human-in-the-Loop Checkpoint]
                        │                     (Accept / Reject / Note)
                        │                                 │
                        │                                 ▼
                        │                     [Final Report Compiler]
                        │                                 │
                        └────────────────┬────────────────┘
                                         ▼
                                 [Final Dashboard]
```

### Agent Roles & Model Configuration
The system uses a **per-agent model configuration** (`config.py`):

| Agent Name | Configured Model | Complexity & Responsibility |
| :--- | :--- | :--- |
| `legal_classifier` | `openai/gpt-oss-20b` | High-speed binary legal vs non-legal classification with confidence scoring |
| `document_analyzer` | `openai/gpt-oss-20b` | Structured extraction of parties, effective dates, terms, payment, governing law, and missing safeguards |
| `risk_detector` | `qwen/qwen3.8-27b` | RAG retrieval grounding against 25 standard legal clause templates; structured risk detection |
| `critic` | `qwen/qwen3.8-27b` | Independent legal audit; filters false positives, justifies severity (CRITICAL, HIGH, MEDIUM, LOW) |
| `explanation_agent` | `openai/gpt-oss-20b` | Jargon-free translations, business impact, and review action points for non-lawyers |
| `comparator` | `qwen/qwen3.8-27b` | Semantic clause alignment between Version A and B, detecting ADDED, REMOVED, MODIFIED clauses and risk direction |

---

## 🛡️ RAG Grounding & Zero-Hallucination Policy

1. **25 Gold-Standard Legal Reference Templates**: Built-in reference corpus across NDAs, Leases, Employment, SaaS SLAs, MSAs, IP Assignment, Indemnity, Liability Caps, and Compliance.
2. **Deterministic Similarity Retrieval**: Vector search calculates cosine similarity against baseline standards.
3. **Evidence Requirement**: Every detected risk cites verbatim contract evidence and links to a baseline reference standard.
4. **No-Hallucination Guardrail**: If similarity is below the threshold, the system returns `"Insufficient retrieved evidence"` and does not invent statutes or citations.

---

## 🧑‍⚖️ Human-in-the-Loop (HITL) Checkpoints

After the Risk Detector and Critic verify findings, LangGraph pauses execution at an interactive breakpoint (`interrupt_before=["human_review"]`):
- Reviewers inspect each flagged risk alongside Critic justifications.
- Reviewers can **Accept Risk**, **Reject Finding**, or **Mark for Review**, and attach custom legal notes.
- The system preserves original AI findings while recording the human decision trail and timestamp.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10+ (or Docker)
- Groq API Key ([https://console.groq.com](https://console.groq.com))

### 2. Installation
```bash
git clone https://github.com/your-username/legalvalidate_ai.git
cd legalvalidate_ai
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create `.env` in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
ENABLE_RETRIEVAL=true
```

### 4. Run Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker

```bash
# Build and run containerized application
docker compose up --build
```
Access the application at `http://localhost:8501`.

---

## 🧪 Automated Testing & Evaluation

### Run Integration Test Suite
```bash
python test_pipeline.py
```

### Run Benchmark Suite across 34 Labeled Test Cases
```bash
python evals/eval.py
```

---

## ⚖️ Legal Safety Disclaimer

> **IMPORTANT:** LegalValidate AI is an automated machine-learning document analysis assistant. It does **NOT** provide legal advice, establish an attorney-client relationship, or replace review by a licensed attorney. Always consult qualified legal counsel before executing binding agreements.
