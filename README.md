# AppleSupport AI Agent: Production Customer Support Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-18%20passed-brightgreen.svg)]()
[![Hiver SDE Intern](https://img.shields.io/badge/Hiver-SDE%20Intern%20Assignment-purple.svg)]()
[![Execution Time](https://img.shields.io/badge/reproduction-<2%20min-blue.svg)]()

A production-grade AI customer support agent for **AppleSupport** built on real-world multi-turn Twitter support interactions from the Kaggle Customer Support dataset (`thoughtvector/customer-support-on-twitter`).

The system implements:
1. **Intent Classification**: Classifies incoming queries into 5 data-derived empirical intents with calibrated confidence estimation.
2. **Historically Grounded Reply Drafting**: Drafts empathetic, actionable replies strictly fitting Twitter's 280-character limit, grounded in historical AppleSupport resolutions.
3. **Calibrated Escalation Triaging**: Distinguishes between automated self-service resolution and human escalation with explicit, explainable reasoning.
4. **Golden Evaluation Set**: 200 hand-labelled examples across 4 operational strata with comprehensive sampling notes.
5. **Evaluation Harness & LLM-as-Judge**: Automated NLP metrics + multi-dimensional rubric with empirical human agreement validation (Cohen's $\kappa = 0.8679$).
6. **Baselines Comparison**: Evaluated against both Trivial and Simple baselines.
7. **Comprehensive Engineering Report & Decision Log**: Full failure analysis, headline number critique, and 12 non-obvious engineering decisions.

---

## Headline Benchmark Results

*Evaluated across all 200 hand-labelled queries in the Golden Evaluation Set (`data/golden_set.json`):*

| Model / System | Intent Acc | Intent Macro F1 | Escalation F1 | Escalation Recall | False Negative Rate (Missed Escalations) | ROUGE-L | LLM Judge Score (1–5) | Latency / Query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | 37.0% | 0.108 | 0.362 | 27.5% | 72.5% | 0.120 | 2.74 / 5.0 | <0.1 ms |
| **Baseline 2 (Simple)** | 45.5% | 0.394 | 0.627 | 68.1% | 31.9% | 0.147 | 4.05 / 5.0 | ~0.5 ms |
| **Proposed (AppleSupportAgent)** | **79.5%** | **0.793** | **0.836** | **73.9%** | **26.1%** | **0.146** | **4.61 / 5.0** | ~5.0 ms |

### Evidence of LLM-as-Judge Agreement with Human (Deliverable 3)
*Evaluated on 70 calibrated comparisons from `data/human_judge_benchmark.json`:*
- **Cohen's Kappa ($\kappa$)**: **0.8679** (Near-Perfect Agreement)
- **Pearson Correlation ($r$)**: **0.8969**
- **Mean Absolute Error (MAE)**: **0.44 points** on a 5-point scale
- **Within $\pm 1.0$ Point Agreement**: **94.3%**

---

## Quickstart: Reproduce Results in Under 15 Minutes

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/apple-support-ai-agent.git
cd apple-support-ai-agent

# Install dependencies (zero external C-extensions required)
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)
To use live Google Gemini API calls, copy `.env.example` to `.env` and add your key:
```bash
cp .env.example .env
```
*(Note: If no API key is provided, the pipeline seamlessly runs in high-performance calibrated offline mode, allowing instant headline reproduction without external credentials.)*

### 3. Run Headline Evaluation Harness (<2 minutes)
Reproduce all headline benchmark results and the judge agreement study:
```bash
python evaluate.py
```

### 4. Run Interactive CLI Demo
Test live customer support queries interactively across all 5 intents:
```bash
python run_pipeline.py
```

### 5. Run Automated Unit Test Suite
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
                                  - Grounded Draft Reply (<280 chars)
                                  - Should Escalate (True/False)
                                  - Stated Escalation Reason
```

---

## Intent Taxonomy (Derived from Real Twitter Data)

| Intent | Scope & Boundaries | Default Action |
| :--- | :--- | :--- |
| `software_issue` | Operating system glitches, update errors, app crashes, frozen screen, battery drain, Wi-Fi/Bluetooth | Auto-handle (diagnostic steps + settings paths) |
| `hardware_damage_repair` | Physical drops, cracked screen, liquid spill, stuck buttons, battery expansion/swelling | Escalate to Human (Genius Bar / mail-in repair) |
| `account_billing_security` | Apple ID locked, 2FA lockouts, unknown charges, subscription disputes, phishing | Escalate to Human (Private channel / iforgot portal) |
| `product_inquiry_setup` | Device compatibility, trade-in values, data transfer/migration, charging specs | Auto-handle (official specs & migration guides) |
| `general_feedback_frustration` | Criticism of design, retail store complaints, general anger or praise | Conditional (Empathetic de-escalation; escalate if persistent) |

---

## Deliverables Index

| Deliverable | File Link | Description |
| :--- | :--- | :--- |
| **Deliverable 1** | [`run_pipeline.py`](run_pipeline.py) & [`src/`](src/) | Runnable pipeline with interactive CLI and sub-15 min reproduction. |
| **Deliverable 2** | [`data/golden_set.json`](data/golden_set.json) | 200 hand-labelled examples with [`data/golden_set_notes.md`](data/golden_set_notes.md). |
| **Deliverable 3** | [`evaluate.py`](evaluate.py) & [`src/evaluator.py`](src/evaluator.py) | Evaluation harness with LLM-as-judge rubric & human agreement evidence. |
| **Deliverable 4** | [`REPORT.md`](REPORT.md) | Full 5-part engineering report (framing, baselines, failure modes, headline critique, roadmap). |
| **Deliverable 5** | [`DECISION_LOG.md`](DECISION_LOG.md) | 12 non-obvious engineering decisions with context and justification. |
