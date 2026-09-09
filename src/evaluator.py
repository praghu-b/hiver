"""
Comprehensive Evaluation Harness for AppleSupport AI Agent.
Computes:
1. Classification Metrics (Accuracy, Macro/Weighted F1, Confusion Matrix, Per-class stats)
2. Escalation Triage Metrics (Precision, Recall, F1, False Negative Rate, False Positive Rate)
3. Text Generation Metrics (ROUGE-1, ROUGE-2, ROUGE-L, Length compliance)
4. Multi-dimensional LLM-as-Judge Rubric (Grounding, Tone, Actionability, Safety)
5. Human-Judge Agreement Study (Cohen's Kappa, Pearson Correlation, MAE)
"""

import math
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from src.config import GEMINI_API_KEY, GEMINI_MODEL, IntentEnum

# --- Automated NLP Metrics ---

def compute_rouge(candidate: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    cand_tokens = re.findall(r"\w+", candidate.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())

    if not cand_tokens or not ref_tokens:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ROUGE-1
    c1 = get_ngrams(cand_tokens, 1)
    r1 = get_ngrams(ref_tokens, 1)
    overlap1 = sum((c1 & r1).values())
    p1 = overlap1 / len(cand_tokens) if cand_tokens else 0
    rec1 = overlap1 / len(ref_tokens) if ref_tokens else 0
    f1_1 = (2 * p1 * rec1) / (p1 + rec1) if (p1 + rec1) > 0 else 0.0

    # ROUGE-2
    c2 = get_ngrams(cand_tokens, 2)
    r2 = get_ngrams(ref_tokens, 2)
    overlap2 = sum((c2 & r2).values())
    p2 = overlap2 / (len(cand_tokens) - 1) if len(cand_tokens) > 1 else 0
    rec2 = overlap2 / (len(ref_tokens) - 1) if len(ref_tokens) > 1 else 0
    f1_2 = (2 * p2 * rec2) / (p2 + rec2) if (p2 + rec2) > 0 else 0.0

    # ROUGE-L (Longest Common Subsequence)
    m, n = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if cand_tokens[i] == ref_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    lcs = dp[m][n]
    pL = lcs / m if m > 0 else 0
    recL = lcs / n if n > 0 else 0
    f1_L = (2 * pL * recL) / (pL + recL) if (pL + recL) > 0 else 0.0

    return {
        "rouge1": round(f1_1, 4),
        "rouge2": round(f1_2, 4),
        "rougeL": round(f1_L, 4)
    }

def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """Compute Accuracy, Macro F1, Weighted F1, Per-class stats, and Confusion Matrix."""
    total = len(y_true)
    if total == 0:
        return {}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    confusion_matrix = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in confusion_matrix and yp in confusion_matrix[yt]:
            confusion_matrix[yt][yp] += 1

    per_class = {}
    f1_list = []
    weights = []

    for label in labels:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[l][label] for l in labels if l != label)
        fn = sum(confusion_matrix[label][l] for l in labels if l != label)
        support = sum(confusion_matrix[label].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        f1_list.append(f1)
        weights.append(support)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    weighted_f1 = sum(f * w for f, w in zip(f1_list, weights)) / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix
    }

def compute_escalation_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """Compute Precision, Recall, F1, False Negative Rate, and False Positive Rate for Escalation."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Critical error rates:
    # FNR = fn / (tp + fn) -> human escalations mistakenly auto-handled (critical safety/account risk)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    # FPR = fp / (fp + tn) -> auto-handled queries needlessly escalated (human queue overloading)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def compute_cohens_kappa(rater1: List[Any], rater2: List[Any]) -> float:
    """Compute Cohen's Kappa coefficient for inter-rater agreement."""
    n = len(rater1)
    if n == 0 or len(rater2) != n:
        return 0.0

    categories = list(set(rater1).union(set(rater2)))
    po = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == r2) / n

    c1 = Counter(rater1)
    c2 = Counter(rater2)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in categories)

    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)

def compute_correlation_and_mae(s1: List[float], s2: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation and Mean Absolute Error between continuous rating lists."""
    n = len(s1)
    if n == 0 or len(s2) != n:
        return 0.0, 0.0

    mae = sum(abs(a - b) for a, b in zip(s1, s2)) / n

    mean1 = sum(s1) / n
    mean2 = sum(s2) / n
    cov = sum((a - mean1) * (b - mean2) for a, b in zip(s1, s2))
    var1 = sum((a - mean1) ** 2 for a in s1)
    var2 = sum((b - mean2) ** 2 for b in s2)

    denom = math.sqrt(var1 * var2)
    corr = (cov / denom) if denom > 0 else 0.0
    return round(corr, 4), round(mae, 4)

# --- LLM-as-Judge Evaluator ---

class SupportJudge:
    """Evaluates customer support reply quality using a structured rubric."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(GEMINI_API_KEY)
        self.model = None
        if self.use_llm:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
            except Exception as e:
                self.use_llm = False

    def evaluate_reply(
        self,
        customer_query: str,
        candidate_reply: str,
        reference_reply: str,
        should_escalate: bool
    ) -> Dict[str, Any]:
        """Score candidate reply on 4 rubric dimensions (1 to 5 scale)."""
        if self.use_llm and self.model:
            try:
                score_dict = self._evaluate_with_llm(customer_query, candidate_reply, reference_reply, should_escalate)
                if score_dict:
                    return score_dict
            except Exception:
                pass

        return self._evaluate_with_heuristic(customer_query, candidate_reply, reference_reply, should_escalate)

    def _evaluate_with_llm(
        self,
        customer_query: str,
        candidate_reply: str,
        reference_reply: str,
        should_escalate: bool
    ) -> Optional[Dict[str, Any]]:
        """Run LLM-as-Judge evaluation using Gemini."""
        prompt = f"""You are an expert QA auditor evaluating customer support replies from @AppleSupport on Twitter.
Evaluate the candidate reply based on the customer query and the reference resolution.

Score each dimension on an integer scale from 1 (poor) to 5 (excellent):
1. grounding_accuracy: Is the technical advice accurate, grounded in official Apple procedures, with no hallucinated features?
2. brand_tone: Is the tone polite, professional, empathetic, and appropriate for Twitter?
3. actionability: Does it provide concrete steps, diagnostic questions, or official links?
4. safety_privacy: Does it protect user privacy (directing to DM for sensitive details, avoiding public credential collection)?

Customer Query: "{customer_query}"
Reference Resolution: "{reference_reply}"
Candidate Reply: "{candidate_reply}"
Expected Handling: {"Escalate to Human/DM" if should_escalate else "Auto-handled Self-Service"}

Respond with ONLY valid JSON:
{{
  "grounding_accuracy": <int 1-5>,
  "brand_tone": <int 1-5>,
  "actionability": <int 1-5>,
  "safety_privacy": <int 1-5>,
  "overall_score": <float 1.0-5.0>,
  "brief_feedback": "<1 sentence>"
}}
"""
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 120}
        )
        cleaned = re.sub(r"```json\s*", "", resp.text.strip())
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        data = json.loads(cleaned)
        return data

    def _evaluate_with_heuristic(
        self,
        customer_query: str,
        candidate_reply: str,
        reference_reply: str,
        should_escalate: bool
    ) -> Dict[str, Any]:
        """Calibrated fallback judge for offline reproducibility."""
        lower_cand = candidate_reply.lower()
        lower_ref = reference_reply.lower()

        is_canned_generic = "restart your device or visit support.apple.com" in lower_cand

        # 1. Grounding (token overlap with reference & domain relevance)
        cand_words = set(re.findall(r"\w+", lower_cand))
        ref_words = set(re.findall(r"\w+", lower_ref))
        overlap = len(cand_words.intersection(ref_words)) / (len(ref_words) or 1)
        
        if is_canned_generic:
            grounding = 2
        elif overlap >= 0.30:
            grounding = 5
        elif overlap >= 0.18:
            grounding = 4
        elif overlap >= 0.08:
            grounding = 3
        else:
            grounding = 2

        # 2. Tone (empathy, personalized address vs robotic template)
        if is_canned_generic:
            tone = 3
        else:
            polite_markers = ["help", "we're here", "sorry", "understand", "let's", "pleased", "assist"]
            has_polite = any(m in lower_cand for m in polite_markers)
            tone = 5 if has_polite else 4

        # 3. Actionability (specific navigation path, diagnostic question, or specific portal)
        if is_canned_generic:
            actionability = 2
        else:
            has_specific_path = ">" in candidate_reply or "?" in candidate_reply or any(p in lower_cand for p in ["iforgot", "reportaproblem", "checkcoverage", "appointment"])
            actionability = 5 if has_specific_path else 3

        # 4. Safety & Privacy
        if should_escalate:
            has_safe_escalation = any(m in lower_cand for m in ["dm", "private", "direct message", "appointment", "store", "iforgot", "reportaproblem", "safe"])
            safety = 5 if has_safe_escalation else 2
        else:
            safety = 5

        overall = round((grounding + tone + actionability + safety) / 4.0, 2)

        return {
            "grounding_accuracy": grounding,
            "brand_tone": tone,
            "actionability": actionability,
            "safety_privacy": safety,
            "overall_score": overall,
            "brief_feedback": "Automated calibrated heuristic assessment."
        }
