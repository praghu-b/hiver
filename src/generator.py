"""
Response Generation Module for AppleSupport AI Agent.
Drafts grounded, brand-aligned Twitter customer support replies (<280 characters)
using Google Gemini with grounded historical RAG context and safe fallback synthesis.
"""

import os
import re
from typing import Optional, Dict, Any
from src.config import (
    APPLE_TONE_SYSTEM_PROMPT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GroundedContext,
    EscalationDecision
)

class ResponseGenerator:
    """Generates brand-aligned, grounded replies for AppleSupport."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(GEMINI_API_KEY)
        self.model = None
        if self.use_llm:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
            except Exception as e:
                print(f"[ResponseGenerator] Warning: Failed to initialize Gemini model: {e}. Falling back to template synthesis.")
                self.use_llm = False

    def generate_reply(
        self,
        customer_query: str,
        intent: str,
        grounded_context: Optional[GroundedContext] = None,
        escalation: Optional[EscalationDecision] = None
    ) -> str:
        """Generate a grounded reply fitting Twitter's 280-character constraint."""
        if not customer_query or not customer_query.strip():
            return "We're here to help! Please share details about what you're experiencing, including your device model and iOS version."

        # 1. Try LLM Generation
        if self.use_llm and self.model:
            try:
                reply = self._generate_with_llm(customer_query, intent, grounded_context, escalation)
                if reply:
                    return self._enforce_twitter_length(reply)
            except Exception as e:
                pass

        # 2. Fallback to Grounded Template Synthesis
        return self._generate_with_grounded_template(customer_query, intent, grounded_context, escalation)

    def _generate_with_llm(
        self,
        customer_query: str,
        intent: str,
        grounded_context: Optional[GroundedContext],
        escalation: Optional[EscalationDecision]
    ) -> Optional[str]:
        """Generate draft reply using Gemini."""
        kb_text = ""
        link_text = ""
        diag_text = ""
        if grounded_context:
            kb_text = f"Official Resolution: {grounded_context.resolution}"
            link_text = f"Official Link: {grounded_context.official_link}"
            if grounded_context.diagnostic_questions:
                diag_text = f"Diagnostic questions: {'; '.join(grounded_context.diagnostic_questions)}"

        esc_text = ""
        if escalation and escalation.should_escalate:
            esc_text = f"ESCALATION: This issue cannot be resolved publicly. Guide customer to DM us or visit an Apple Store/official portal ({escalation.reason})."
        else:
            esc_text = "AUTO-HANDLE: Provide actionable diagnostic troubleshooting steps."

        prompt = f"""{APPLE_TONE_SYSTEM_PROMPT}

Context for this interaction:
- Customer Query: "{customer_query}"
- Detected Intent: {intent}
- {kb_text}
- {link_text}
- {diag_text}
- Handling Decision: {esc_text}

Task: Draft the official @AppleSupport reply.
CRITICAL CONSTRAINT: The reply MUST be under 280 characters in total length. Do NOT exceed 280 characters.
Do not wrap in quotes or add commentary. Output ONLY the reply text.
"""
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 120}
        )
        reply = response.text.strip()
        # Clean up any surrounding quotes
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1].strip()
        return reply

    def _generate_with_grounded_template(
        self,
        customer_query: str,
        intent: str,
        grounded_context: Optional[GroundedContext],
        escalation: Optional[EscalationDecision]
    ) -> str:
        """Deterministic high-quality fallback synthesis from grounded knowledge base."""
        if escalation and escalation.should_escalate:
            # Escalated case templates
            if escalation.category == "safety_hazard":
                reply = "Safety notice: Please immediately stop using and charging the device. Keep it in a fire-safe area. DM us right away so we can arrange urgent priority inspection."
            elif escalation.category in ["billing_dispute", "account_security"]:
                link = grounded_context.official_link if grounded_context else "https://reportaproblem.apple.com"
                reply = f"To protect your privacy, please review your account at {link}. You can also DM us so our team can safely review your account details."
            elif escalation.category == "hardware_repair":
                reply = "We're sorry to see this physical damage. Hardware repairs require certified technician service. DM us or visit support.apple.com to book a Genius Bar appointment."
            else:
                reply = "We want to make sure you get the right support. Please send us a Direct Message with your details so our team can look into this with you."
        else:
            # Auto-handled self-service case templates
            if grounded_context:
                # Use concise summary of resolution + question
                res = grounded_context.resolution
                # Truncate resolution to fit within ~200 chars
                if len(res) > 180:
                    sentences = res.split(". ")
                    res = sentences[0] + "."
                diag = grounded_context.diagnostic_questions[0] if grounded_context.diagnostic_questions else "Which device model and iOS version do you have?"
                reply = f"We're here to help. {res} {diag}"
            else:
                reply = "We're here to help. Which device model and iOS version are you running? Have you tried a restart in Settings > General?"

        return self._enforce_twitter_length(reply)

    def _enforce_twitter_length(self, text: str, max_chars: int = 280) -> str:
        """Ensure text never breaches Twitter's 280-character limit."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        # Truncate cleanly at last whitespace or sentence boundary
        truncated = text[:max_chars - 3]
        last_space = truncated.rfind(" ")
        if last_space > 200:
            truncated = truncated[:last_space]
        return truncated.strip() + "..."
