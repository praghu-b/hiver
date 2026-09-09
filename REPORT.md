# Engineering Report: AI Customer Support Agent for AppleSupport

**Candidate**: SDE Intern Candidate  
**Role**: SDE Intern Take-Home Assignment — Hiver  
**Brand Selected**: `@AppleSupport` (from Kaggle `thoughtvector/customer-support-on-twitter`)  
**Evaluation Set**: 200 hand-labelled queries across 4 strata (`data/golden_set.json`)  
**Calibration Set**: 50 annotated interactions (`data/human_judge_benchmark.json`)  

---

## 1. Problem Framing

### What "Good" Means for `@AppleSupport`
On Twitter (X), customer support is inherently public, fast-moving, and high-visibility. For Apple, customer support is not just ticket resolution—it is brand protection. An incoming tweet often comes from a stressed user whose device has frozen, whose battery is draining after an update, or who was unexpectedly charged. 

"Good" support for `@AppleSupport` is defined by four non-negotiable operational principles:
1. **Empathetic, Non-Defensive De-Escalation**: Acknowledging customer frustration without corporate defensiveness or assigning premature blame to the user.
2. **Actionable Diagnostic Probing**: Rather than guessing solutions or hallucinating settings, the agent asks crisp diagnostic questions (e.g., specific iOS version, exact device model, or whether the symptom occurs in safe mode/reboot).
3. **Strict Privacy & Channel Discipline**: Public Twitter is an insecure broadcast channel. "Good" support strictly forbids soliciting Apple ID credentials, passwords, 2FA codes, serial numbers, or payment details publicly. Private inquiries must be guided to secure direct channels (`iforgot.apple.com`, `reportaproblem.apple.com`, or private Twitter DM).
4. **Safety-First Triage**: Physical hazards (such as battery swelling, severe overheating, or liquid ingress) must be identified immediately and escalated to human emergency service protocols with explicit fire safety warnings, bypassing routine software reboot advice.

### What We Chose NOT to Build
Engineering discipline requires knowing what *not* to build. In this pipeline, we explicitly scoped out:
- **No Autonomous Account Mutation**: The agent does not execute account unlocks, password resets, or refund approvals autonomously over public Twitter. Doing so would violate Apple's zero-trust security architecture and expose customers to social engineering attacks.
- **No Speculative Hardware Diagnostics**: The agent will never attempt to diagnose logic board failure or internal solder cracks over text. If a symptom is physical, it immediately routes the user to an Apple Store Genius Bar or Authorized Service Provider.
- **No Unconstrained Open-Domain Conversational Chatbot**: The system will not engage in chit-chat, political debates, or non-Apple technical inquiries. It strictly bounds inputs to Apple products, services, and policies.

---

## 2. Results vs. Baselines

We evaluated three complete systems on our 200-sample hand-labelled Golden Evaluation Set (`data/golden_set.json`), spanning 5 empirical intents (`software_issue`, `hardware_damage_repair`, `account_billing_security`, `product_inquiry_setup`, `general_feedback_frustration`) and 4 sampling strata (`core_intent`, `ambiguous_boundary`, `adversarial_frustration`, `short_query`).

### Comparative Benchmark Results

| Model / System | Intent Accuracy | Intent Macro F1 | Escalation F1 | Escalation Recall | False Negative Rate (Missed Escalations) | ROUGE-L | LLM Judge Score (1–5) | Latency / Query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | 37.0% | 0.108 | 0.362 | 27.5% | 72.5% | 0.120 | 2.74 / 5.0 | <0.1 ms |
| **Baseline 2 (Simple)** | 45.5% | 0.394 | 0.627 | 68.1% | 31.9% | 0.147 | 4.05 / 5.0 | ~0.5 ms |
| **Proposed (AppleSupportAgent)** | **79.5%** | **0.793** | **0.836** | **73.9%** | **26.1%** | **0.146** | **4.61 / 5.0** | ~5.0 ms |

*(Benchmark run on 200 golden samples via `evaluate.py`. Automated reproduction time: 1.1 seconds.)*

### In-Depth Comparative Analysis

#### 1. Baseline 1 (Trivial Baseline)
- **Architecture**: Majority-class intent classifier (`software_issue`), canned static reply (*"Hello! We are here to help. Please restart your device or visit support.apple.com."*), and naive keyword escalation (`["refund", "lawyer", "broken", "cancel"]`).
- **Performance & Flaws**: 
  - Intent accuracy sits at 37.0% (the prevalence of `software_issue` in the dataset), with a near-zero Macro F1 (0.108) because it fails to identify any of the other 4 intents.
  - Crucially, its **False Negative Rate on Escalations is 72.5%**: it fails to escalate almost three-quarters of customer queries requiring human intervention (such as locked Apple IDs, water ingress, and swollen batteries). In a real support environment, this would cause severe customer abandonment and safety liabilities.
  - The judge scores it at 2.74/5.0 due to complete lack of diagnostic relevance.

#### 2. Baseline 2 (Simple Baseline)
- **Architecture**: TF-IDF token-overlap retrieval over historical resolutions, returning the nearest historical Apple reply verbatim, paired with rule-based intent mapping.
- **Performance & Flaws**:
  - Intent accuracy improves to 45.5% and Escalation F1 to 0.627.
  - Because it copies historical AppleSupport tweets verbatim, its text quality is superficially coherent (Judge score 4.05), but it frequently outputs mismatched details (e.g. telling a customer who asked about iPhone X display lines to "check their Apple Watch charger").
  - It suffers from a 31.9% False Negative Rate on critical escalations.

#### 3. Proposed Agent (AppleSupportAgent)
- **Architecture**: Contextual compound-pattern intent classifier + BM25 RAG retriever over curated historical resolutions + safety/security triage escalation engine + Apple tone-constrained response generator.
- **Performance & Gains**:
  - **Intent Accuracy jumps to 79.5%** and **Macro F1 to 0.793** (+101% relative improvement over Simple Baseline).
  - **Escalation F1 reaches 0.836**, with Escalation Recall at 73.9% and FNR dropping to 26.1%.
  - **Judge Quality Score reaches 4.61 / 5.0**, consistently delivering empathetic, actionable responses under Twitter's 280-character limit with accurate navigation paths (`Settings > General > About`) and official Apple portal links (`iforgot.apple.com`, `reportaproblem.apple.com`).

---

## 3. Failure Analysis: Top 5 Failure Modes

Through systematic inspection of the 41 intent misclassifications and 20 escalation mismatches on the Golden Set (`scripts/analyze_errors.py`), we identified the following top 5 failure modes:

### Failure Mode 1: Intermittent Hardware Failure Described as Software Glitch
- **Real Example** (`eval_141`):  
  *Customer Tweet*: `"My iPhone won't charge unless I wiggle the cable at a specific angle. Is this iOS or the port?"`  
  *Ground Truth*: Intent: `hardware_damage_repair` | Escalate: `True`  
  *Agent Output*: Intent: `software_issue` | Escalate: `False`  
- **Root-Cause Hypothesis**: The customer explicitly included the phrase *"Is this iOS or the port?"*, injecting strong software tokens into a mechanical contact failure query. Because the user asked a diagnostic question about iOS, the classifier prioritized the software framing over the physical cable wiggling symptom.
- **Mitigation**: Add a mechanical contact rule: any symptom requiring physical manipulation (wiggling, pressing hard, tapping) must be classified as hardware diagnostic triage.

### Failure Mode 2: Degraded Battery Capacity at Threshold Boundary
- **Real Example** (`eval_142`):  
  *Customer Tweet*: `"Battery health says 79% and my phone keeps shutting down at 30% battery."`  
  *Ground Truth*: Intent: `hardware_damage_repair` | Escalate: `True` (Apple policy: battery capacity <80% requires physical replacement)  
  *Agent Output*: Intent: `software_issue` | Escalate: `False`  
- **Root-Cause Hypothesis**: The token `"battery health"` and `"shutting down"` strongly trigger the software battery diagnostic rule (`Settings > Battery`). The model lacked a numerical parsing sub-routine to evaluate whether the stated percentage was below the official 80% AppleCare replacement threshold.
- **Mitigation**: Introduce a numerical extractor for battery health percentages (`r"(\d+)%"`): if percentage $\le 80\%$, escalate directly to battery hardware service.

### Failure Mode 3: Sarcastic & Subtle Frustration without Explicit Keywords
- **Real Example** (`eval_171`):  
  *Customer Tweet*: `"Love how my brand new $1200 iPhone doubles as a pocket hand warmer that dies in 2 hours."`  
  *Ground Truth*: Intent: `general_feedback_frustration` | Escalate: `False` (De-escalate sarcasm, offer battery diagnostics)  
  *Agent Output*: Intent: `software_issue` | Escalate: `False`  
- **Root-Cause Hypothesis**: Sarcasm inversion. The user says *"Love how my brand new..."*, which lexical classifiers misread as positive/neutral, while the secondary terms (*"dies in 2 hours"*) trigger the literal battery drain software pattern.
- **Mitigation**: Deploy a dedicated sentiment dissonance detector: when positive praise verbs (*"love"*, *"fantastic"*, *"great"*) are paired with negative physical symptoms (*"dies"*, *"hand warmer"*, *"useless"*), route to frustration de-escalation.

### Failure Mode 4: Multi-Intent Queries (App Crash + Demand for Refund)
- **Real Example** (`eval_149`):  
  *Customer Tweet*: `"Can I get a refund for an app that constantly crashes every time I open it?"`  
  *Ground Truth*: Intent: `account_billing_security` | Escalate: `True`  
  *Agent Output*: Intent: `account_billing_security` | Escalate: `True` *(Correctly triaged, but draft reply focused on troubleshooting rather than refund procedure)*  
- **Root-Cause Hypothesis**: Multi-intent overlap. The query contains both a software symptom (*"constantly crashes"*) and an explicit financial demand (*"get a refund"*). While the classifier chose billing, the generator's retrieved context pulled app troubleshooting rather than the `reportaproblem.apple.com` refund portal.
- **Mitigation**: Multi-label intent extraction: when both technical failure and billing terms co-occur, enforce financial resolution precedence in the generator prompt.

### Failure Mode 5: Generic Refurbished Unit Complaints with Low Context
- **Real Example** (`eval_151`):  
  *Customer Tweet*: `"My iPhone 8 has intermittent issues after buying a refurbished unit. Should I reboot or visit the store?"`  
  *Ground Truth*: Intent: `hardware_damage_repair` | Escalate: `True`  
  *Agent Output*: Intent: `software_issue` | Escalate: `False`  
- **Root-Cause Hypothesis**: The inquiry is intentionally ambiguous (*"Should I reboot or visit the store?"*). In customer support operations, conservative self-service models will always attempt software rebooting first before booking store appointments to prevent store overcrowding.
- **Mitigation**: Incorporate warranty provenance: queries mentioning 3rd-party refurbished or used purchases have higher hardware defect probability and should prompt for purchase origin before escalating.

---

## 4. "What is Misleading About My Headline Number?" (Mandatory Section)

Our headline numbers—**79.5% Intent Accuracy**, **0.836 Escalation F1**, and **4.61/5.0 Judge Score**—demonstrate strong performance. However, evaluating AI systems honestly requires confronting the ways headline metrics can be deceptive:

### 1. High ROUGE-L Does Not Guarantee Technical Soundness
In natural language generation, ROUGE-L measures the longest common subsequence of tokens between the candidate and the reference text. A model that generates:  
*"Please restart your phone and check Settings > General > Software Update."*  
achieves high ROUGE overlap with an Apple Support reply. However, if the user was reporting a physically swollen battery, this response is not just wrong—it is dangerously negligent. ROUGE cannot penalize catastrophic safety omissions.

### 2. High Escalation Recall Can Mask Lazy Over-Escalation
An agent can easily achieve **100% Escalation Recall** by simply escalating every incoming tweet to a human agent. Doing so would completely defeat the purpose of automation and overwhelm human support queues with trivial questions. In our evaluation, the proposed agent achieved 73.9% recall with a low False Positive Rate (10.7%), but in real production environments, support managers must continuously tune the precision-recall trade-off based on live agent headcount and queue backlog.

### 3. Sampling Bias in the Golden Evaluation Set
Our Golden Set was constructed using stratified sampling across 4 strata (`core_intent`, `ambiguous_boundary`, `adversarial_frustration`, `short_query`). While this guarantees coverage of edge cases, real-world Twitter distribution is not stratified—it is heavily skewed toward short, noisy, repetitive complaints during iOS release weeks (e.g. millions of identical battery drain complaints). An agent tuned to perform well across balanced strata may perform differently when faced with real-world distribution shifts or major product launch spikes.

### 4. LLM-as-Judge Inherent Biases
While our LLM-as-Judge demonstrated substantial agreement with human annotators ($\kappa = 0.8679, r = 0.8969$), LLM judges suffer from well-documented systematic biases:
- **Verbosity Bias**: LLMs naturally score longer, well-formatted replies higher, even when a shorter, blunt diagnostic question is more operationally effective on Twitter.
- **Self-Enhancement Bias**: LLM judges score responses generated by LLMs higher than human agent responses because they match the model's preferred syntactic distributions.

---

## 5. What You'd Do Next with One More Week

With an additional week of engineering time, we would prioritize the following production enhancements:

1. **Multi-Turn Conversation State Tracking**:  
   Real Twitter customer support is rarely single-turn. Customers reply with screenshots, answers to diagnostic questions, or follow-up complaints. We would implement a lightweight session state tracker (using Redis or in-memory LRU) to maintain conversation history across tweet thread IDs (`in_reply_to_status_id`).
2. **Real-Time Apple System Status API Integration**:  
   Integrate automated health checks against the public Apple System Status API (`https://www.apple.com/support/systemstatus/`). When thousands of users tweet that iCloud or Apple Music is down, the agent should automatically recognize global outages and broadcast status alerts rather than diagnosing individual devices.
3. **Multimodal Screenshot OCR & Inspection**:  
   Over 20% of inbound Twitter support queries contain only an image or video attachment (e.g., error alert screenshot, cracked screen photo). We would integrate a vision model (e.g. Gemini Vision API) to extract on-screen error codes and detect physical display fractures.
4. **Active Learning & Human-in-the-Loop Triage Dashboard**:  
   Deploy a lightweight reviewer queue where human agents review automated drafts in real time. Disagreements and manual edits would be logged to continuously fine-tune intent classification boundaries and knowledge retrieval rankings.
5. **Brand Tone Fine-Tuning (LoRA)**:  
   Fine-tune an open 7B/8B model (e.g., Gemma 2 or Llama 3) via LoRA specifically on verified AppleSupport Twitter interaction pairs, eliminating LLM prompt overhead and achieving sub-20ms inference latency at zero API cost.
