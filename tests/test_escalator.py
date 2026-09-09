"""
Unit tests for EscalationEngine.
"""

import pytest
from src.escalator import EscalationEngine
from src.config import IntentEnum

@pytest.fixture
def escalator():
    return EscalationEngine()

def test_escalate_swollen_battery_safety_hazard(escalator):
    query = "My iPhone battery is physically expanding and the screen is popping off the frame!"
    dec = escalator.decide(query, intent=IntentEnum.HARDWARE_DAMAGE_REPAIR.value)
    assert dec.should_escalate is True
    assert dec.confidence >= 0.95
    assert "safety" in dec.category.lower() or "hazard" in dec.reason.lower()

def test_escalate_physical_hardware_damage(escalator):
    query = "Dropped my iPhone and screen is shattered into pieces."
    dec = escalator.decide(query, intent=IntentEnum.HARDWARE_DAMAGE_REPAIR.value)
    assert dec.should_escalate is True
    assert "hardware" in dec.category.lower() or "repair" in dec.reason.lower()

def test_escalate_unauthorized_billing_charge(escalator):
    query = "I see three unauthorized charges of $29.99 from ITUNES.COM/BILL on my bank account!"
    dec = escalator.decide(query, intent=IntentEnum.ACCOUNT_BILLING_SECURITY.value)
    assert dec.should_escalate is True
    assert dec.category == "billing_dispute"

def test_auto_handle_software_troubleshooting(escalator):
    query = "My battery is draining fast after the iOS 11 update. How do I fix it?"
    dec = escalator.decide(query, intent=IntentEnum.SOFTWARE_ISSUE.value)
    assert dec.should_escalate is False
    assert "None:" in dec.reason

def test_auto_handle_subscription_cancellation(escalator):
    query = "How do I cancel my Apple Music family subscription before it renews?"
    dec = escalator.decide(query, intent=IntentEnum.ACCOUNT_BILLING_SECURITY.value)
    assert dec.should_escalate is False
    assert "self-service" in dec.reason.lower()

def test_auto_handle_product_compatibility(escalator):
    query = "Will the Apple Pencil work with my new iPad?"
    dec = escalator.decide(query, intent=IntentEnum.PRODUCT_INQUIRY_SETUP.value)
    assert dec.should_escalate is False
