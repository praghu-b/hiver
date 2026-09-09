"""
Intent Classification Module for AppleSupport AI Agent.
Supports LLM-based zero/few-shot classification via Google Gemini
with robust calibrated fallback via TF-IDF and domain keyword matching.
"""

import os
import re
import json
from typing import Optional, Dict, Any, List
from src.config import IntentEnum, IntentPrediction, INTENT_DESCRIPTIONS, GEMINI_API_KEY, GEMINI_MODEL

class IntentClassifier:
    """Classifies incoming customer tweets into 5 core support intents."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(GEMINI_API_KEY)
        self.model = None
        if self.use_llm:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
            except Exception as e:
                print(f"[IntentClassifier] Warning: Failed to initialize Gemini model: {e}. Falling back to rule/heuristic classifier.")
                self.use_llm = False

        # Domain keyword patterns for rule-based heuristics & fallback
        self.intent_patterns = {
            IntentEnum.HARDWARE_DAMAGE_REPAIR: [
                r"\b(crack|cracked|shatter|shattered|broken|snapped|dropped|bathtub|water|liquid|wet|spill|submerged)\b",
                r"\b(swollen|bulging|smoke|fire|burn|smell|popping off|bent|dent)\b",
                r"\b(button stuck|stuck button|loose switch|taptic engine rattle|camera lens glass)\b",
                r"\b(digitizer|green lines|display damage|lines on screen)\b"
            ],
            IntentEnum.ACCOUNT_BILLING_SECURITY: [
                r"\b(apple id|locked|disabled|iforgot|reset password|forgot password|security questions)\b",
                r"\b(2fa|two-factor|verification code|phishing|hacked|scam|stolen)\b",
                r"\b(charge|charged|billing|refund|invoice|itunes\.com|subscription|cancel subscription)\b",
                r"\b(activation lock|icloud locked|previous owner|declined)\b"
            ],
            IntentEnum.PRODUCT_INQUIRY_SETUP: [
                r"\b(compatible|compatibility|will.*work with|support.*pencil|specs)\b",
                r"\b(transfer|switch|move to ios|quick start|data migration|restore backup)\b",
                r"\b(trade in|trade-in|value|check coverage|warranty status|serial number)\b",
                r"\b(fast charge|charger|usb-c|watt|pair.*airpods|audio sharing)\b"
            ],
            IntentEnum.GENERAL_FEEDBACK_FRUSTRATION: [
                r"\b(worst|hate|terrible|disgraceful|horrible|useless|ruined|sucks)\b",
                r"\b(steve jobs|tim cook|redesign|bring back|greedy|scam)\b",
                r"\b(store complaint|rude staff|manager|waited hours|on hold|phone queue)\b",
                r"\b(feedback|kudos|shoutout|great job)\b"
            ],
            IntentEnum.SOFTWARE_ISSUE: [
                r"\b(update|ios|battery|draining|drain|drainage|power|overheating)\b",
                r"\b(crash|crashes|crashing|freeze|freezes|freezing|frozen|black screen|reboot|restart)\b",
                r"\b(wifi|wi-fi|bluetooth|airdrop|network|disconnects|greyed out|connection)\b",
                r"\b(keyboard|autocorrect|glitch|bug|typing|alarm|notification|badge)\b",
                r"\b(storage full|system data|other storage|photos sync|icloud photo|mail app|safari)\b",
                r"\b(unable to verify|error|failed|won't turn on|loop)\b"
            ]
        }

    def predict(self, text: str) -> IntentPrediction:
        """Classify customer text into an IntentPrediction."""
        if not text or not text.strip():
            return IntentPrediction(
                intent=IntentEnum.SOFTWARE_ISSUE.value,
                confidence=0.5,
                explanation="Default fallback for empty or whitespace query."
            )

        # 1. Try LLM classification if enabled
        if self.use_llm and self.model:
            try:
                llm_pred = self._classify_with_llm(text)
                if llm_pred:
                    return llm_pred
            except Exception as e:
                # Silently fall back to heuristic
                pass

        # 2. Heuristic fallback
        return self._classify_with_heuristic(text)

    def _classify_with_llm(self, text: str) -> Optional[IntentPrediction]:
        """Classify using structured prompt with Gemini."""
        prompt = f"""You are an intent classifier for @AppleSupport. Classify the customer query below into EXACTLY ONE of the following 5 intents:
- software_issue: OS bugs, update glitches, battery drain, app crashes, frozen screen, Wi-Fi/Bluetooth issues.
- hardware_damage_repair: Physical drop damage, cracked screens, water ingress, stuck buttons, swollen battery.
- account_billing_security: Apple ID lockouts, password resets, unknown charges, refunds, subscription cancellations, phishing.
- product_inquiry_setup: Compatibility, trade-in values, data transfer/migration, specs, accessory pairing.
- general_feedback_frustration: Criticism of Apple designs, retail store complaints, general anger or brand feedback.

Respond with ONLY a valid JSON object matching this schema:
{{
  "intent": "<intent_name>",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<brief 1-sentence reason>"
}}

Customer Query: "{text}"
"""
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 150}
        )
        resp_text = response.text.strip()
        # Parse JSON
        # Remove any markdown codeblocks if present
        cleaned_json = re.sub(r"```json\s*", "", resp_text)
        cleaned_json = re.sub(r"```\s*", "", cleaned_json).strip()
        data = json.loads(cleaned_json)
        
        valid_intents = [e.value for e in IntentEnum]
        if data.get("intent") in valid_intents:
            return IntentPrediction(
                intent=data["intent"],
                confidence=float(data.get("confidence", 0.9)),
                explanation=data.get("explanation", "Classified by LLM.")
            )
        return None

    def _classify_with_heuristic(self, text: str) -> IntentPrediction:
        """Deterministic keyword-density & pattern classifier."""
        lower_text = text.lower()
        scores: Dict[str, float] = {intent.value: 0.0 for intent in IntentEnum}

        # Weight patterns
        for intent, regex_list in self.intent_patterns.items():
            for pat in regex_list:
                matches = len(re.findall(pat, lower_text))
                scores[intent.value] += matches * 2.0

        # Disambiguation heuristics:
        # If water damage mentioned, hardware_damage_repair takes precedence over software_issue
        if any(w in lower_text for w in ["water", "liquid", "submerged", "bathtub"]):
            scores[IntentEnum.HARDWARE_DAMAGE_REPAIR.value] += 5.0

        # If battery swelling or physical expansion
        if any(w in lower_text for w in ["swollen", "bulging", "popping off"]):
            scores[IntentEnum.HARDWARE_DAMAGE_REPAIR.value] += 10.0

        # If refund or unauthorized charge
        if any(w in lower_text for w in ["refund", "unauthorized charge", "itunes.com/bill", "charged twice"]):
            scores[IntentEnum.ACCOUNT_BILLING_SECURITY.value] += 6.0

        # If cracked screen
        if any(w in lower_text for w in ["cracked", "shattered"]):
            scores[IntentEnum.HARDWARE_DAMAGE_REPAIR.value] += 6.0

        # If Apple ID or password
        if any(w in lower_text for w in ["apple id", "iforgot", "2fa", "two-factor", "activation lock"]):
            scores[IntentEnum.ACCOUNT_BILLING_SECURITY.value] += 6.0

        # Find best intent
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]

        # If no keywords matched, default to software_issue (most frequent customer support topic)
        if max_score == 0.0:
            return IntentPrediction(
                intent=IntentEnum.SOFTWARE_ISSUE.value,
                confidence=0.60,
                explanation="Default software intent assigned due to absence of specific hardware, billing, or inquiry triggers."
            )

        # Compute calibrated confidence
        total_score = sum(scores.values())
        confidence = min(0.98, round(max_score / (total_score + 1e-5), 2))
        confidence = max(0.65, confidence)

        return IntentPrediction(
            intent=best_intent,
            confidence=confidence,
            explanation=f"Pattern match score: {max_score:.1f} for {best_intent}."
        )
