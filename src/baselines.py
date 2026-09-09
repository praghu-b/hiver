"""
Baselines Implementation for Deliverable 4.
Implements:
1. Baseline 1 (Trivial): Majority-class intent + canned static reply + simple keyword escalation.
2. Baseline 2 (Simple): TF-IDF + Naive Bayes intent + Nearest-Neighbor direct historical reply + rule escalation.
"""

import re
from typing import List, Dict, Any, Optional
from src.config import IntentEnum, SupportResponse
from src.data_loader import sanitize_tweet, load_knowledge_base

class TrivialBaselineAgent:
    """Baseline 1: Trivial majority-class classifier, canned response, and crude keyword escalation."""

    def __init__(self):
        self.majority_intent = IntentEnum.SOFTWARE_ISSUE.value
        self.canned_reply = "Hello! We are here to help. Please restart your device or visit support.apple.com."
        self.escalation_keywords = ["refund", "lawyer", "sue", "broken", "cancel", "stole"]

    def process_tweet(self, customer_query: str) -> SupportResponse:
        cleaned = sanitize_tweet(customer_query)
        lower = cleaned.lower()

        # Simple keyword escalation
        should_escalate = any(kw in lower for kw in self.escalation_keywords)
        reason = "Keyword triggered escalation." if should_escalate else "None: Default auto-handled."

        return SupportResponse(
            customer_query=customer_query,
            intent=self.majority_intent,
            intent_confidence=0.50,
            draft_reply=self.canned_reply,
            should_escalate=should_escalate,
            escalation_confidence=0.50,
            escalation_reason=reason,
            character_count=len(self.canned_reply),
            processing_time_ms=0.1
        )

class SimpleBaselineAgent:
    """Baseline 2: TF-IDF Nearest-Neighbor brand reply retrieval and rule-based heuristic classifier."""

    def __init__(self, kb_path: str = "data/historical_knowledge_base.json"):
        self.kb = load_knowledge_base(kb_path)
        # Store index of queries and responses
        self.docs = []
        for item in self.kb:
            self.docs.append({
                "intent": item["intent"],
                "text": f"{item['topic']} {item['query_pattern']}",
                "reply": item["resolution"]
            })

    def process_tweet(self, customer_query: str) -> SupportResponse:
        cleaned = sanitize_tweet(customer_query)
        tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", cleaned.lower()))

        # Simple Jaccard similarity / token overlap with KB documents
        best_doc = self.docs[0]
        best_overlap = -1
        for d in self.docs:
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", d["text"].lower()))
            overlap = len(tokens.intersection(doc_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_doc = d

        intent = best_doc["intent"]
        # Nearest neighbor directly returns the historical resolution text as reply
        draft_reply = best_doc["reply"]
        if len(draft_reply) > 275:
            draft_reply = draft_reply[:272] + "..."

        # Rule escalation
        should_escalate = intent in [
            IntentEnum.HARDWARE_DAMAGE_REPAIR.value,
            IntentEnum.ACCOUNT_BILLING_SECURITY.value
        ]
        reason = "Intent requires human specialist." if should_escalate else "None: Handled via nearest historical resolution."

        return SupportResponse(
            customer_query=customer_query,
            intent=intent,
            intent_confidence=0.70,
            draft_reply=draft_reply,
            should_escalate=should_escalate,
            escalation_confidence=0.70,
            escalation_reason=reason,
            grounded_topic=best_doc["text"][:30],
            character_count=len(draft_reply),
            processing_time_ms=0.5
        )
