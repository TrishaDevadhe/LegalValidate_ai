# LegalValidate AI — Technical Architecture & Engineering Case Study

This document details the engineering architecture, system design decisions, RAG strategy, human-in-the-loop orchestration, model selection evolution, and empirical benchmark evaluation for **LegalValidate AI**.

---

## 1. Multi-Agent Architecture & LangGraph DAG Orchestration

LegalValidate AI processes legal contracts through a non-linear Directed Acyclic Graph (DAG) built with **LangGraph**. Rather than relying on a monolithic prompt or rigid sequential chain, the pipeline decomposes document validation into modular, specialized agent nodes:

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

### Architectural Highlights
- **Early Exit on Non-Legal Documents**: If an uploaded file is classified as non-legal (recipe, invoice receipt, flight itinerary, academic exam, sci-fi treaty), the graph skips heavy downstream reasoning and routes immediately to the Non-Legal Explainer, saving token costs and eliminating false positives.
- **RAG-Grounded Risk Detection**: Identifies contractual clauses and queries an in-memory TF-IDF vector store populated with 25 standard legal baseline clauses.
- **Independent Critic Verification**: Audits candidate risks against contract evidence, filters out standard boilerplate, and corrects severity levels (CRITICAL, HIGH, MEDIUM, LOW).
- **Human-in-the-Loop Breakpoint**: State pauses before final compilation, allowing attorneys and business reviewers to Accept, Reject, Mark for Review, and annotate each finding.

---

## 2. Model Selection Story & Per-Agent Architecture

### The Problem: Lightweight 8B Models
During early development, unified 8B models (such as `llama-3.1-8b-instant`) were evaluated across all nodes. However, smaller 8B models on Groq encountered:
1. Strict `BadRequestError` exceptions (`tool calling is not supported with this model`).
2. Malformed JSON output and missing schema fields when using Pydantic structured output chains (`with_structured_output`).

### The Solution: Tiered Per-Agent Model Routing (`config.py`)
To maximize accuracy, throughput, and schema adherence:

| Agent Name | Configured Model | Rationale & Task Complexity |
| :--- | :--- | :--- |
| `legal_classifier` | `openai/gpt-oss-20b` | High-speed binary legal classification & confidence scoring |
| `document_analyzer` | `openai/gpt-oss-20b` | Structured JSON document category & key section extraction |
| `risk_detector` | `qwen/qwen3.8-27b` | RAG retrieval grounding & multi-clause legal reasoning |
| `critic` | `qwen/qwen3.8-27b` | Objective risk auditing & severity grading |
| `explanation_agent` | `openai/gpt-oss-20b` | Plain-language contract summarization |
| `comparator` | `qwen/qwen3.8-27b` | Semantic alignment & revision redlining |

---

## 3. RAG Grounding Strategy & Zero-Hallucination Policy

1. **Standard Legal Baseline Corpus**: 25 comprehensive legal templates across NDA, Leases, Employment, SaaS SLAs, MSAs, IP Assignment, Indemnity, Liability Caps, and Compliance.
2. **Cosine Similarity Filtering**: Vector search calculates cosine similarity against baseline standards.
3. **Traceable Evidence**: Every detected risk cites verbatim contract evidence and links to a baseline reference standard.
4. **Zero-Hallucination Guardrail**: When no reference standard is retrieved, the system outputs `"Insufficient retrieved evidence"` and does not invent legal citations.

---

## 4. Human-in-the-Loop (HITL) Checkpoint Orchestration

Legal technology requires human oversight to ensure compliance and accountability.

### LangGraph `MemorySaver` Checkpointer
- **State Interruption**: After initial risk detection, critic auditing, and plain-language explanation, the graph halts at `interrupt_before=["human_review"]`.
- **Interactive UI Form**: The reviewer inspects each risk, assigns a human decision (**Accept Risk**, **Reject Finding**, **Mark for Review**), and adds reviewer notes.
- **Audit Trail Persistence**: Decisions, notes, and timestamps are recorded in the SQLite database and compiled into the exportable PDF report without altering the underlying AI evidence trail.

---

## 5. Document Ingestion & Redline Comparison

- **Multi-Format Support**: Native PDF extraction with page number preservation, `.docx` XML parsing, and clean `.txt` decoding.
- **OCR Fallback**: Automatic image rendering and OCR via PyPDFium2 and Tesseract for scanned PDFs.
- **Semantic Redline Comparison**: Uses TF-IDF cosine alignment to match clauses between Contract Version A and Version B, categorizing changes into `ADDED`, `REMOVED`, `MODIFIED`, and `UNCHANGED` with risk impact analysis.

---

## 6. Empirical Evaluation Benchmark Results

The pipeline is benchmarked across a 34-document labeled evaluation suite (`evals/test_set.json` & `evals/eval.py`) covering Standard Contracts, Non-Legal Documents, Edge Cases, and Adversarial Examples:

| Pipeline Component | Primary Metric | Target / Observed Performance |
| :--- | :--- | :--- |
| **Legal Classifier** | Accuracy / Recall | **97.1% / 100.0%** (Zero false negatives on true contracts) |
| **Document Analyzer** | Doc Type Match Rate | **100.0%** across labeled categories |
| **Risk Detector** | Recall (Semantic) | **100.0%** of human-flagged contract risks identified |
| **RAG Retrieval** | Grounding Relevance | **90.0%+** reference template alignment |
| **Critic Verifier** | Audit Confirmation | **85.0%+** objective severity verification |

---

## 7. Legal Safety Notice

LegalValidate AI is an automated AI document analysis tool and does not constitute formal legal advice, an attorney-client relationship, or a substitute for review by qualified legal counsel.
