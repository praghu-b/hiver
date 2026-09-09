# Golden Evaluation Set Methodology & Sampling Notes

## Overview
The Golden Evaluation Set comprises **200 hand-labelled and curated customer queries** directed to `@AppleSupport`, derived from the Kaggle Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`).

## Sampling Strategy & Strata Breakdown
To avoid naive evaluation on trivially easy queries, the dataset is stratified into four distinct operational strata:

| Stratum | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `core_intent` | 100 | 50% | Clear, typical customer inquiries representing standard operational volume across all 5 intents. |
| `ambiguous_boundary` | 40 | 20% | Inquiries spanning multiple problem domains (e.g. software symptom stemming from water damage, or billing block affecting app downloads). |
| `adversarial_frustration` | 30 | 15% | Emotionally charged, sarcastic, or demanding messages testing the agent's de-escalation, empathy, and tone stability. |
| `short_query` | 30 | 15% | Terse, low-context tweets (e.g. "won't charge", "screen black") testing diagnostic probing and clarifying questions. |

## Intent Class Distribution
The dataset is balanced across the 5 empirical intents:
- `software_issue`: 74 examples
- `hardware_damage_repair`: 39 examples
- `account_billing_security`: 32 examples
- `product_inquiry_setup`: 25 examples
- `general_feedback_frustration`: 30 examples

## Escalation Class Distribution
- **Auto-Handled (`escalate = False`)**: 131 examples (65.5%)
  - Resolved via diagnostic self-service steps, official knowledge-base links, feature explanations, or empathetic de-escalation.
- **Escalate to Human (`escalate = True`)**: 69 examples (34.5%)
  - Requires in-person Genius Bar appointment, mail-in repair depot inspection, private Apple ID account recovery, financial transaction review, or urgent safety handling (e.g. battery swelling).

## Labelling Guidelines & Quality Control
1. **Sanitization**: Removed raw Twitter anonymized IDs (e.g., `@115854`) while preserving genuine technical entities (iOS versions, error codes, device models).
2. **Deterministic Escalation Rules**:
   - Physical structural damage, liquid ingress, or hardware component failure -> **Must Escalate** (`True`).
   - Account security, 2FA lockout, disputed credit card charges -> **Must Escalate** (`True`).
   - Software bugs, configuration issues, how-to setup, general feedback -> **Auto-Handle** (`False`).
3. **Reference Resolutions**: Grounded directly in verified AppleSupport communication standards:
   - Twitter character limit (<280 characters)
   - Professional empathy without unwarranted blame
   - Explicit navigation paths (e.g., *Settings > General > About*)
   - Direct official links (`support.apple.com`, `iforgot.apple.com`, `reportaproblem.apple.com`)
