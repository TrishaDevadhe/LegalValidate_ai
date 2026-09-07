import os

# Per-Agent Model Configuration
# Defines which model powers each specific agent in the LegalValidate AI multi-agent workflow.
MODEL_CONFIG = {
    "document_analyzer": os.getenv("MODEL_DOCUMENT_ANALYZER", "qwen/qwen3.8-27b"),
    "legal_classifier": os.getenv("MODEL_LEGAL_CLASSIFIER", "openai/gpt-oss-20b"),
    "risk_detector": os.getenv("MODEL_RISK_DETECTOR", "qwen/qwen3.8-27b"),
    "critic": os.getenv("MODEL_CRITIC", "qwen/qwen3.8-27b"),
    "explanation_agent": os.getenv("MODEL_EXPLANATION_AGENT", "openai/gpt-oss-20b"),
    "comparator": os.getenv("MODEL_COMPARATOR", "qwen/qwen3.8-27b")
}

DEFAULT_MODEL = "qwen/qwen3.8-27b"
