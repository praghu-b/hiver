"""
Triage and Escalation Engine for AppleSupport AI Agent.
Evaluates incoming queries to decide whether they can be safely auto-handled
or must be escalated to a human agent, providing an explicit stated rationale.
"""

import re
from typing import Dict, Any, Optional
from src.config import IntentEnum, EscalationDecision

class EscalationEngine:
    """Evaluates risk, safety, security, and technical boundaries to triage support cases."""

    def __init__(self):
        # Critical Safety triggers (highest priority)
        self.safety_patterns = [
            r"\b(swollen|bulging|smoke|fire|burning smell|sparks?|exploded?)\b",
            r"\b(battery expanding|screen popping off|hot to touch)\b"
        ]

        # Physical hardware damage triggers
        self.hardware_patterns = [
            r"\b(crack|cracked|shatter|shattered|broken glass|bent|dented)\b",
            r"\b(dropped in water|liquid damage|spilled coffee|submerged|toilet|bathtub)\b",
            r"\b(button stuck|stuck button|loose port|snapped cable|digitizer broken|green lines)\b"
        ]

        # Account & Security triggers
        self.security_patterns = [
            r"\b(apple id locked|disabled for security|hacked|unauthorized access)\b",
            r"\b(two-factor|2fa.*lost|trusted phone.*lost|cannot receive code)\b",
            r"\b(activation lock|icloud lock|previous owner)\b"
        ]

        # Financial & Billing dispute triggers
        self.billing_patterns = [
            r"\b(unauthorized charge|charged twice|unknown charge|fraudulent charge)\b",
            r"\b(itunes\.com/bill|stole.*money|refund.*request|refund.*charge)\b"
        ]

        # Hostility / Legal threats / Store complaints
        self.hostility_patterns = [
            r"\b(lawyer|attorney|sue|police|lawsuit|legal action)\b",
            r"\b(store complaint|manager|waited hours|rude employee|disgraceful service)\b"
        ]

    def decide(self, text: str, intent: str, intent_confidence: float = 1.0) -> EscalationDecision:
        """Decide whether to escalate or auto-handle with explicit reason."""
        lower_text = text.lower()

        # 1. Critical Safety Hazard
        for pat in self.safety_patterns:
            if re.search(pat, lower_text):
                return EscalationDecision(
                    should_escalate=True,
                    confidence=0.99,
                    reason="Critical hardware safety hazard (battery expansion/thermal event) requires immediate human intervention and physical isolation.",
                    category="safety_hazard"
                )

        # 2. Financial & Billing Disputes
        for pat in self.billing_patterns:
            if re.search(pat, lower_text):
                return EscalationDecision(
                    should_escalate=True,
                    confidence=0.95,
                    reason="Financial transaction dispute and unauthorized charge investigation require private account review and billing authorization.",
                    category="billing_dispute"
                )

        # 3. Severe Customer Frustration / Legal Threats / In-Store Failures
        for pat in self.hostility_patterns:
            if re.search(pat, lower_text):
                return EscalationDecision(
                    should_escalate=True,
                    confidence=0.92,
                    reason="High customer dissatisfaction or in-store retail escalation requires senior human representative follow-up.",
                    category="escalated_complaint"
                )

        # 4. Physical Hardware Damage
        if intent == IntentEnum.HARDWARE_DAMAGE_REPAIR.value:
            for pat in self.hardware_patterns:
                if re.search(pat, lower_text):
                    return EscalationDecision(
                        should_escalate=True,
                        confidence=0.96,
                        reason="Physical hardware damage requires authorized diagnostic inspection, Genius Bar appointment, or mail-in repair.",
                        category="hardware_repair"
                    )
            # Default for hardware damage intent
            return EscalationDecision(
                should_escalate=True,
                confidence=0.90,
                reason="Physical component or mechanical defect cannot be resolved remotely via software.",
                category="hardware_repair"
            )

        # 5. Account, Security & Locked ID
        if intent == IntentEnum.ACCOUNT_BILLING_SECURITY.value:
            for pat in self.security_patterns:
                if re.search(pat, lower_text):
                    return EscalationDecision(
                        should_escalate=True,
                        confidence=0.95,
                        reason="Account security and identity verification require authenticated private recovery; cannot resolve over public Twitter.",
                        category="account_security"
                    )
            # Check for self-serviceable account sub-topics (e.g., standard cancel subscription instructions)
            if any(w in lower_text for w in ["cancel subscription", "phishing", "screen time passcode", "how do i change email"]):
                return EscalationDecision(
                    should_escalate=False,
                    confidence=0.88,
                    reason="None: Standard self-service account management procedures and official portal links suffice.",
                    category="self_service_account"
                )
            return EscalationDecision(
                should_escalate=True,
                confidence=0.85,
                reason="Account credentials, billing status, or activation locks require private identity verification.",
                category="account_security"
            )

        # 6. Intent-based Software & Product Inquiries (Default: Auto-Handled)
        if intent == IntentEnum.SOFTWARE_ISSUE.value:
            return EscalationDecision(
                should_escalate=False,
                confidence=0.90,
                reason="None: Standard software diagnostic procedure (force restart, settings reset, battery check) can resolve this inquiry.",
                category="self_service_software"
            )

        if intent == IntentEnum.PRODUCT_INQUIRY_SETUP.value:
            return EscalationDecision(
                should_escalate=False,
                confidence=0.92,
                reason="None: Official technical specifications, compatibility lists, and setup guides resolve this inquiry.",
                category="self_service_information"
            )

        if intent == IntentEnum.GENERAL_FEEDBACK_FRUSTRATION.value:
            # Check if customer explicitly threatened churn or had long wait times
            if any(w in lower_text for w in ["waited", "hold", "queue", "switching to", "leaving apple"]):
                return EscalationDecision(
                    should_escalate=True,
                    confidence=0.85,
                    reason="High churn risk or prolonged support queue friction warrants proactive human outreach via DM.",
                    category="retention_escalation"
                )
            return EscalationDecision(
                should_escalate=False,
                confidence=0.88,
                reason="None: General product feedback and design thoughts can be acknowledged and directed to apple.com/feedback.",
                category="feedback_acknowledgment"
            )

        # Fallback default
        return EscalationDecision(
            should_escalate=False,
            confidence=0.75,
            reason="None: Initial diagnostic guidance recommended before initiating human queue handoff.",
            category="general_triage"
        )
