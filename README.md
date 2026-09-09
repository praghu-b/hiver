# AppleSupport AI Agent: Production-Grade Customer Support Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()
[![Hiver SDE Intern](https://img.shields.io/badge/Hiver-SDE%20Intern%20Assignment-purple.svg)]()

An AI customer support agent for **AppleSupport** built on real-world multi-turn Twitter support interactions from the Kaggle Customer Support dataset (`thoughtvector/customer-support-on-twitter`).

The system implements:
1. **Intent Classification**: Classifies customer queries into 5 empirical intents derived from data.
2. **Historically Grounded Reply Drafting**: Drafts empathetic, technically accurate replies grounded in real historical AppleSupport resolutions.
3. **Calibrated Escalation Triaging**: Distinguishes between automated self-service resolution and human escalation with explicit, explainable reasoning.
4. **Golden Evaluation Set**: 200 hand-labelled examples with sampling notes.
5. **Evaluation Harness & LLM-as-Judge**: Automated metrics + multi-dimensional LLM-as-judge rubric with human agreement validation (Cohen's $\kappa$).
6. **Baselines Comparison**: Evaluated against both Trivial and Simple baselines.
7. **Comprehensive Engineering Report & Decision Log**: Full failure analysis, headline number critique, and 12 non-obvious engineering decisions.

---

## Quickstart: Reproduce Results in Under 15 Minutes

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/apple-support-ai-agent.git
cd apple-support-ai-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory (copy from `.env.example`):
```bash
cp .env.example .env
```
Add your Google Gemini API key to `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

### 3. Run Evaluation Harness (<2 minutes)
Reproduce all headline benchmark results across the Golden Evaluation Set:
```bash
python evaluate.py
```

### 4. Run Interactive Demo
Test customer support queries interactively:
```bash
python run_pipeline.py
```

### 5. Run Test Suite
```bash
pytest tests/ -v
```

---

## System Architecture

```
                                 [ Incoming Customer Tweet ]
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Intent Classifier (5-Class)  │
                             └─────────────────────────────────┘
                                              │
                                ┌─────────────┴─────────────┐
                                ▼                           ▼
                   ┌────────────────────────┐  ┌─────────────────────────┐
                   │  Historical Retriever  │  │   Escalation Engine     │
                   │    (BM25 + Semantic)   │  │   (Triage + Reasoning)  │
                   └────────────────────────┘  └─────────────────────────┘
                                │                           │
                                └─────────────┬─────────────┘
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Response Generator (LLM)     │
                             │  (Apple Tone + Grounding RAG)   │
                             └─────────────────────────────────┘
                                              │
                                              ▼
                                   [ SupportResponse JSON ]
                                  - Intent & Confidence
                                  - Grounded Draft Reply
                                  - Should Escalate (True/False)
                                  - Escalation Stated Reason
```

---

## Deliverables Index

- **Full Report**: [`REPORT.md`](REPORT.md)
  - Problem framing & what we chose not to build
  - Results vs. Trivial & Simple baselines
  - Top 5 failure modes with real examples and hypotheses
  - Mandatory critique: *"What is misleading about my headline number?"*
  - Next steps with one more week
- **Decision Log**: [`DECISION_LOG.md`](DECISION_LOG.md) (12 non-obvious engineering decisions)
- **Golden Evaluation Set**: [`data/golden_set.json`](data/golden_set.json) (200 hand-labelled queries)
- **Golden Set Methodology**: [`data/golden_set_notes.md`](data/golden_set_notes.md)
- **Human-Judge Benchmark**: [`data/human_judge_benchmark.json`](data/human_judge_benchmark.json)
