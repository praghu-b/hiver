"""
End-to-End Customer Support Agent Pipeline for AppleSupport.
Orchestrates:
  Incoming Tweet -> Intent Classification -> Grounding Retrieval -> Escalation Triage -> Response Drafting
"""

import time
from typing import Optional, Dict, Any
from src.config import SupportResponse
from src.data_loader import sanitize_tweet
from src.classifier import IntentClassifier
from src.retriever import HistoricalRetriever
from src.generator import ResponseGenerator
from src.escalator import EscalationEngine

class AppleSupportAgent:
    """Production customer support pipeline for @AppleSupport."""

    def __init__(self, use_llm: bool = True):
        self.classifier = IntentClassifier(use_llm=use_llm)
        self.retriever = HistoricalRetriever()
        self.escalator = EscalationEngine()
        self.generator = ResponseGenerator(use_llm=use_llm)

    def process_tweet(self, customer_query: str) -> SupportResponse:
        """Process a customer tweet through the full agent pipeline."""
        start_time = time.time()
        cleaned_text = sanitize_tweet(customer_query)

        # 1. Intent Classification
        intent_pred = self.classifier.predict(cleaned_text)

        # 2. Historical Knowledge Retrieval (RAG Grounding)
        retrieved_contexts = self.retriever.retrieve(cleaned_text, intent_filter=intent_pred.intent, top_k=1)
        grounded_context = retrieved_contexts[0] if retrieved_contexts else None

        # 3. Escalation Decision Engine
        escalation = self.escalator.decide(
            text=cleaned_text,
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence
        )

        # 4. Draft Reply Generation
        draft_reply = self.generator.generate_reply(
            customer_query=cleaned_text,
            intent=intent_pred.intent,
            grounded_context=grounded_context,
            escalation=escalation
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        return SupportResponse(
            customer_query=customer_query,
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            draft_reply=draft_reply,
            should_escalate=escalation.should_escalate,
            escalation_confidence=escalation.confidence,
            escalation_reason=escalation.reason,
            grounded_kb_id=grounded_context.kb_id if grounded_context else None,
            grounded_topic=grounded_context.topic if grounded_context else None,
            character_count=len(draft_reply),
            processing_time_ms=round(elapsed_ms, 2)
        )
