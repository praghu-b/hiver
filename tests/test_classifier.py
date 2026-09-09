"""
Unit tests for IntentClassifier.
"""

import pytest
from src.classifier import IntentClassifier
from src.config import IntentEnum

@pytest.fixture
def classifier():
    return IntentClassifier(use_llm=False)

def test_classify_software_issue(classifier):
    query = "My iPhone 8 battery drops 30% in an hour after installing iOS 11. What should I do?"
    pred = classifier.predict(query)
    assert pred.intent == IntentEnum.SOFTWARE_ISSUE.value
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.explanation) > 0

def test_classify_hardware_damage(classifier):
    query = "Dropped my iPhone X on concrete and the back glass is completely shattered."
    pred = classifier.predict(query)
    assert pred.intent == IntentEnum.HARDWARE_DAMAGE_REPAIR.value
    assert pred.confidence >= 0.70

def test_classify_account_billing(classifier):
    query = "My Apple ID was locked for security reasons and I cannot reset my password."
    pred = classifier.predict(query)
    assert pred.intent == IntentEnum.ACCOUNT_BILLING_SECURITY.value

def test_classify_product_inquiry(classifier):
    query = "Will the Apple Pencil 2 work with my 9.7-inch iPad 2018?"
    pred = classifier.predict(query)
    assert pred.intent == IntentEnum.PRODUCT_INQUIRY_SETUP.value

def test_classify_general_feedback(classifier):
    query = "The new iOS 11 Control Center is the ugliest, most confusing design ever released."
    pred = classifier.predict(query)
    assert pred.intent == IntentEnum.GENERAL_FEEDBACK_FRUSTRATION.value

def test_classify_empty_string(classifier):
    pred = classifier.predict("")
    assert pred.intent == IntentEnum.SOFTWARE_ISSUE.value
    assert pred.confidence == 0.5
