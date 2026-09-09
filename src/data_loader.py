"""
Data loading and preprocessing utilities for AppleSupport AI Agent.
"""

import json
import os
import re
import html
from typing import List, Dict, Any

def sanitize_tweet(text: str) -> str:
    """Clean and normalize raw customer tweet."""
    if not text:
        return ""
    # Decode HTML entities (&amp; -> &, &gt; -> >)
    text = html.unescape(text)
    # Remove user mentions (@AppleSupport, @115854)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove raw t.co links if embedded with other text
    text = re.sub(r'https?://t\.co/[A-Za-z0-9]+', '', text)
    # Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_golden_set(filepath: str = "data/golden_set.json") -> List[Dict[str, Any]]:
    """Load the 200 hand-labelled golden evaluation examples."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden evaluation set not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_knowledge_base(filepath: str = "data/historical_knowledge_base.json") -> List[Dict[str, Any]]:
    """Load curated historical knowledge base for grounding."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Knowledge base not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_human_benchmark(filepath: str = "data/human_judge_benchmark.json") -> List[Dict[str, Any]]:
    """Load 50-example human ratings benchmark for LLM-as-judge alignment."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Human benchmark not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
