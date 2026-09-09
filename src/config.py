"""
Configuration, constants, and data schemas for AppleSupport AI Agent.
"""

import os
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

class IntentEnum(str, Enum):
    SOFTWARE_ISSUE = "software_issue"
    HARDWARE_DAMAGE_REPAIR = "hardware_damage_repair"
    ACCOUNT_BILLING_SECURITY = "account_billing_security"
    PRODUCT_INQUIRY_SETUP = "product_inquiry_setup"
    GENERAL_FEEDBACK_FRUSTRATION = "general_feedback_frustration"

INTENT_DESCRIPTIONS = {
    IntentEnum.SOFTWARE_ISSUE: "Operating system glitches, update errors, app crashes, frozen screen, battery drain, Wi-Fi/Bluetooth issues.",
    IntentEnum.HARDWARE_DAMAGE_REPAIR: "Physical damage, cracked screen, liquid spill, broken buttons, swollen battery, physical hardware failures.",
    IntentEnum.ACCOUNT_BILLING_SECURITY: "Apple ID locked, 2FA issues, unknown App Store/iTunes charges, subscription cancellation, phishing.",
    IntentEnum.PRODUCT_INQUIRY_SETUP: "Device compatibility, specs, trade-in values, data transfer/migration, new device setup.",
    IntentEnum.GENERAL_FEEDBACK_FRUSTRATION: "Criticism of updates/design, retail store complaints, general brand dissatisfaction without explicit technical defect."
}

# Escalation Policy defaults
INTENT_DEFAULT_ESCALATION = {
    IntentEnum.SOFTWARE_ISSUE: False,
    IntentEnum.HARDWARE_DAMAGE_REPAIR: True,
    IntentEnum.ACCOUNT_BILLING_SECURITY: True,
    IntentEnum.PRODUCT_INQUIRY_SETUP: False,
    IntentEnum.GENERAL_FEEDBACK_FRUSTRATION: False
}

class IntentPrediction(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

class EscalationDecision(BaseModel):
    should_escalate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    category: str

class GroundedContext(BaseModel):
    kb_id: str
    topic: str
    resolution: str
    official_link: str
    diagnostic_questions: List[str]
    score: float

class SupportResponse(BaseModel):
    customer_query: str
    intent: str
    intent_confidence: float
    draft_reply: str
    should_escalate: bool
    escalation_confidence: float
    escalation_reason: str
    grounded_kb_id: Optional[str] = None
    grounded_topic: Optional[str] = None
    character_count: int = 0
    processing_time_ms: float = 0.0

# Apple Support Brand Tone Guidelines
APPLE_TONE_SYSTEM_PROMPT = """You are an official Apple Support representative on Twitter (@AppleSupport).
Your goal is to draft a helpful, professional, and empathetic reply to the customer's tweet.

Brand Communication Guidelines:
1. Tone: Empathetic, polite, professional, and concise. Acknowledge customer frustration constructively.
2. Length: Strictly under 280 characters (standard Twitter post limit). Be direct and actionable.
3. Diagnostic questions: When symptoms are unclear, ask concise diagnostic questions (e.g., specific iOS version, device model, or error message).
4. Troubleshooting: Provide concrete navigation paths (e.g., "Settings > General > About" or "Settings > Battery").
5. Grounding: Rely strictly on Apple's official support practices. Never hallucinate unofficial third-party tools or non-existent settings.
6. Safety & Privacy: NEVER ask for sensitive passwords, 2FA codes, or credit cards publicly. If account lookup or hardware repair booking is required, direct them to DM (Direct Message) or official portals (support.apple.com, iforgot.apple.com, reportaproblem.apple.com).
"""
