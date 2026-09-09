"""
Unit tests for HistoricalRetriever.
"""

import pytest
from src.retriever import HistoricalRetriever
from src.config import IntentEnum

@pytest.fixture
def retriever():
    return HistoricalRetriever()

def test_retrieve_battery_issue(retriever):
    query = "iPhone battery draining quickly"
    results = retriever.retrieve(query, intent_filter=IntentEnum.SOFTWARE_ISSUE.value, top_k=1)
    assert len(results) == 1
    ctx = results[0]
    assert ctx.topic == "battery_drain"
    assert ctx.official_link.startswith("https://")
    assert len(ctx.diagnostic_questions) > 0

def test_retrieve_cracked_screen(retriever):
    query = "Shattered cracked screen display"
    results = retriever.retrieve(query, intent_filter=IntentEnum.HARDWARE_DAMAGE_REPAIR.value, top_k=1)
    assert len(results) == 1
    ctx = results[0]
    assert ctx.topic == "cracked_screen_display_damage"
    assert "screen" in ctx.resolution.lower()

def test_retrieve_empty_query(retriever):
    results = retriever.retrieve("")
    assert results == []
