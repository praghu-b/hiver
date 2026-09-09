# Decision Log: 12 Non-Obvious Engineering Decisions

This document records the key non-obvious architectural, product, and machine learning decisions made during the design and implementation of the AppleSupport AI Customer Support Agent.

---

### 1. Brand Selection: `@AppleSupport` over `AmazonHelp` or Airlines
- **Decision**: Selected `@AppleSupport` from the Kaggle dataset rather than the higher-volume `AmazonHelp` or airline accounts.
- **Why**: `AmazonHelp` tweets are predominantly repetitive logistics queries (*"Where is my package?"*) resolved with a generic canned response (*"Please DM us your order ID"*). In contrast, `@AppleSupport` interactions feature genuine technical troubleshooting (battery degradation, iOS update loops, Bluetooth packet loss), structured diagnostic back-and-forth, and high-stakes triage boundaries (hardware repair vs. software reset).

### 2. Taxonomy Sizing: 5 MECE Intents instead of Granular Sub-Intents (e.g., Banking77's 77 classes)
- **Decision**: Defined an empirical 5-class intent taxonomy (`software_issue`, `hardware_damage_repair`, `account_billing_security`, `product_inquiry_setup`, `general_feedback_frustration`) rather than adopting an overly granular 50–70 class scheme.
- **Why**: In real-world customer support, intent classification is only useful if it maps to distinct downstream operational actions. A 70-class taxonomy creates high inter-class ambiguity (e.g., distinguishing "Wi-Fi disconnected" from "Wi-Fi slow") without changing the resolution action (both require Network Settings reset). 5 MECE classes optimize for operational actionability.

### 3. Decoupling Intent Classification from Escalation Triaging
- **Decision**: Built the Escalation Engine as an independent decision layer rather than making escalation a 1-to-1 function of intent.
- **Why**: An issue can belong to `account_billing_security` and yet be safely auto-handled (e.g., how to cancel an Apple Music subscription or identify a phishing email). Conversely, a `software_issue` query may require escalation if the customer threatens legal action or has been trapped in a support loop. Decoupling topic from triage preserves operational flexibility.

### 4. Safety-First Short-Circuit Override
- **Decision**: Implemented regex-based safety hazard rules that short-circuit model inference before any downstream LLM or retriever calls.
- **Why**: When a customer reports a swollen, bulging battery or thermal runaway (*"battery is expanding and hot to touch"*), model latency or potential hallucinations are unacceptable liabilities. An immediate deterministic override guarantees safety instructions (fire safety, do not charge) and priority routing in <1ms.

### 5. Stratified 4-Strata Sampling for the 200 Golden Evaluation Examples
- **Decision**: Stratified the evaluation set into `core_intent` (50%), `ambiguous_boundary` (20%), `adversarial_frustration` (15%), and `short_query` (15%).
- **Why**: Random sampling in customer support datasets over-indexes on easy, repetitive queries (e.g. "battery dying after update"). Evaluating on a naive random sample produces inflated headline accuracy that collapses on real-world edge cases. Stratified sampling guarantees rigorous evaluation across adversarial emotion and multi-domain ambiguity.

### 6. Substring Disambiguation: Financial "Charge" vs. Battery "Charger"
- **Decision**: Added negative lookaheads and phrase-level boundary checks to differentiate financial card charges from battery charging queries.
- **Why**: Early heuristic iterations misclassified *"Can I use my iPad charger to charge my iPhone faster?"* as an unauthorized credit card billing dispute because of the substring `"charge"`. Restricting billing patterns to explicit financial context (`"unauthorized charge"`, `"card charged"`, `"itunes.com/bill"`) eliminated this critical false positive.

### 7. Semantic Disambiguation: "Orientation Lock" vs. "Apple ID Lock"
- **Decision**: Created an explicit contextual override mapping `"rotation is locked"` and `"orientation lock"` to `software_issue`, preventing it from triggering account security locks.
- **Why**: Standard keyword classifiers map the token `"locked"` to account security. Recognizing device UI state settings as software troubleshooting prevents unnecessary account verification routing.

### 8. Hard 280-Character Budget Enforcement at Generation Time
- **Decision**: Built a dual-layer Twitter character constraint (prompt-level system instruction + post-generation whitespace-aware truncation).
- **Why**: Many LLM customer support demos generate beautiful 500-word essays that fail when sent to the Twitter API (280-character maximum). Enforcing character limits preserves production readiness and forces the model to be concise and actionable.

### 9. Zero-Credential Offline Reproduction Mode (<2 minutes)
- **Decision**: Architected the pipeline with full offline reproduction support, pre-calibrated scoring, and local BM25 indexing alongside optional live Gemini API integration.
- **Why**: Take-home assignment evaluators often have limited time and cannot troubleshoot missing proprietary API keys or billing quotas. Providing an instant, zero-friction reproduction script (`evaluate.py` in 1.1s) guarantees reviewers can verify all headline metrics in under 2 minutes.

### 10. Multi-Dimensional Rubric for LLM-as-Judge
- **Decision**: Designed the evaluation rubric across 4 discrete dimensions (Grounding, Tone, Actionability, Safety) on a 1–5 scale rather than a single monolithic "quality" number.
- **Why**: Single-number ratings are noisy and uninterpretable. A reply can be extremely polite (Tone: 5) while giving completely hallucinated advice (Grounding: 1). Discrete dimensions allow granular failure analysis and direct correlation with human QA standards.

### 11. Rigorous Human-Judge Agreement Validation with Cohen's Kappa ($\kappa$)
- **Decision**: Curated a 50-sample human-annotated benchmark (`data/human_judge_benchmark.json`) and calculated Cohen's Kappa ($\kappa = 0.8679$) and Pearson correlation ($r = 0.8969$).
- **Why**: The assignment explicitly requires *"evidence of how well your judge agrees with a human."* Many AI developers use LLM judges as unvalidated black boxes. Measuring inter-rater agreement against human scores provides empirical proof that the judge is reliable.

### 12. Pure-Python NLP Metrics Implementation
- **Decision**: Implemented ROUGE-1, ROUGE-2, ROUGE-L, Cohen's Kappa, and Pearson correlation directly in pure Python without external C-extensions.
- **Why**: Eliminates fragile C++ build tools, NLTK download dependencies, or platform-specific compilation errors on Windows/Linux/macOS, ensuring bulletproof portability.
