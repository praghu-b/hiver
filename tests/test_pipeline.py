"""
Unit tests for AppleSupportAgent end-to-end pipeline.
"""

import pytest
from src.pipeline import AppleSupportAgent

@pytest.fixture
def agent():
    return AppleSupportAgent(use_llm=False)

def test_pipeline_software_query(agent):
    query = "My iPhone 8 battery drops 30% in an hour after installing iOS 11. What should I do?"
    resp = agent.process_tweet(query)

    assert resp.customer_query == query
    assert resp.intent == "software_issue"
    assert resp.should_escalate is False
    assert len(resp.draft_reply) > 0
    assert resp.character_count <= 280
    assert resp.character_count == len(resp.draft_reply)
    assert resp.grounded_kb_id is not None
    assert resp.processing_time_ms > 0

def test_pipeline_hardware_escalation(agent):
    query = "Dropped my phone on concrete and the back glass is completely shattered."
    resp = agent.process_tweet(query)

    assert resp.intent == "hardware_damage_repair"
    assert resp.should_escalate is True
    assert "hardware" in resp.escalation_reason.lower() or "repair" in resp.escalation_reason.lower()
    assert resp.character_count <= 280

def test_pipeline_empty_query(agent):
    resp = agent.process_tweet("")
    assert resp.character_count <= 280
    assert "help" in resp.draft_reply.lower()
