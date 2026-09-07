# LegalValidate AI

LegalValidate AI is an intelligent multi-agent legal document validation and risk assessment system powered by LangGraph, LangChain, and Groq LLMs. The platform classifies documents, analyzes document structure, detects contract risks with RAG-grounded retrieval intelligence, summarizes clauses in plain language, verifies risk severity via a Critic agent, and performs semantic clause comparisons between contract revisions.

> 📖 **Technical Architecture & Portfolio Case Study**: For an in-depth breakdown of the multi-agent system design, DAG routing, RAG grounding, human-in-the-loop checkpoints, model selection evolution, and empirical benchmark evaluation, view [WRITEUP.md](WRITEUP.md).


---

## Model Selection & Architecture Rationale

During initial system development, unified default models like `llama-3.1-8b-instant` were evaluated across all agents. However, lightweight 8B models on Groq encountered strict `BadRequestError` exceptions (`tool calling is not supported with this model`) or generated incomplete JSON outputs when executing Pydantic structured output chains (`with_structured_output`).

To ensure reliable schema validation, tool execution, and high-precision reasoning, LegalValidate AI uses a **per-agent model configuration**:

- **Reasoning & RAG Heavy Agents** (`risk_detector`, `critic`, `comparator`): Powered by `qwen/qwen3.8-27b` for deep legal reasoning, high-fidelity schema adherence, retrieval alignment, and objective verification.
- **Fast Extraction & Classification Agents** (`legal_classifier`, `document_analyzer`, `explanation_agent`): Powered by `openai/gpt-oss-20b` for high-speed classification, document category extraction, and plain-language rephrasing with full structured tool calling support.

---

## Agent Model Configuration

Per-agent models are dynamically configured in `config.py` via `MODEL_CONFIG` and can be customized via environment variables:

| Agent Name | Configured Model | Task Complexity / Rationale |
| :--- | :--- | :--- |
| `legal_classifier` | `openai/gpt-oss-20b` | High-speed binary legal vs non-legal classification |
| `document_analyzer` | `openai/gpt-oss-20b` | Structured document category & key section extraction |
| `risk_detector` | `qwen/qwen3.8-27b` | RAG-retrieval grounded risk assessment |
| `critic` | `qwen/qwen3.8-27b` | Objective risk verification and severity grading |
| `explanation_agent` | `openai/gpt-oss-20b` | Plain-language contract summarization |
| `comparator` | `qwen/qwen3.8-27b` | Multi-document semantic clause comparison |

---

## Local Setup & Installation

Follow these steps to set up and run LegalValidate AI on your local environment:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/legalvalidate_ai.git
cd legalvalidate_ai
```

### 2. Environment Configuration
Copy the example environment configuration to create your `.env` file:
```bash
cp .env.example .env
```
Open `.env` and set your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install Python Dependencies
Ensure Python 3.10+ is installed, then install all required packages:
```bash
pip install -r requirements.txt
```
*(Optional system OCR support: Install Tesseract OCR and Poppler if analyzing scanned PDF images).*

### 4. Vector Store Seeding
On initial run, `retriever.py` automatically initializes the standard legal clause reference vector store in memory using the built-in seed corpus (`STANDARD_CLAUSE_TEMPLATES`). No manual database seeding is required.

### 5. Launch Streamlit Application
Run the main Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Run with Docker

You can launch LegalValidate AI inside a containerized environment without installing Python, Tesseract OCR, or Poppler utilities locally.

### Prerequisites
1. Ensure Docker Desktop is installed and running.
2. Create a `.env` file in the project root with your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Command to Build & Start
To build the container image and launch the application:

```bash
docker compose up --build
```

Access the Streamlit web dashboard in your browser at:
`http://localhost:8501`

### Stopping the Container
```bash
docker compose down
```

---

## Running Evaluations

To run the full multi-agent benchmark across 34 labeled documents:

```bash
python evals/eval.py
```
Results will be output as a summary table and exported to `evals/results_<timestamp>.json`.
