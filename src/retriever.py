"""
Historical Knowledge Base Retriever for RAG Grounding.
Retrieves official AppleSupport historical resolution procedures and links.
"""

import math
import re
from typing import List, Optional, Dict, Any
from src.config import GroundedContext
from src.data_loader import load_knowledge_base

class HistoricalRetriever:
    """Retrieves relevant historical resolutions using BM25-style lexical matching."""

    def __init__(self, kb_path: str = "data/historical_knowledge_base.json"):
        self.kb = load_knowledge_base(kb_path)
        self.corpus = []
        for item in self.kb:
            # Combine query patterns, topic, and resolution into searchable document
            doc_text = f"{item['topic']} {item['query_pattern']} {item['resolution']}"
            tokens = self._tokenize(doc_text)
            self.corpus.append({
                "item": item,
                "tokens": tokens,
                "length": len(tokens)
            })
        self.avg_doc_len = sum(d["length"] for d in self.corpus) / (len(self.corpus) or 1)
        self.doc_freqs = self._compute_doc_freqs()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenization with lowercasing."""
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def _compute_doc_freqs(self) -> Dict[str, int]:
        """Compute document frequencies for all vocabulary tokens."""
        df: Dict[str, int] = {}
        for d in self.corpus:
            unique_tokens = set(d["tokens"])
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1
        return df

    def retrieve(self, query: str, intent_filter: Optional[str] = None, top_k: int = 1) -> List[GroundedContext]:
        """Retrieve the top-k most relevant historical resolutions."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_docs = []
        k1 = 1.5
        b = 0.75
        N = len(self.corpus)

        for d in self.corpus:
            item = d["item"]
            # Optional intent filter
            if intent_filter and item.get("intent") != intent_filter:
                intent_bonus = 0.0
            else:
                intent_bonus = 2.0  # boost matching intent

            doc_tokens = d["tokens"]
            doc_len = d["length"]

            # Token frequencies in document
            tf_dict: Dict[str, int] = {}
            for t in doc_tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            bm25_score = 0.0
            for t in query_tokens:
                if t in tf_dict:
                    f = tf_dict[t]
                    df = self.doc_freqs.get(t, 1)
                    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                    num = f * (k1 + 1)
                    denom = f + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                    bm25_score += idf * (num / denom)

            final_score = bm25_score + intent_bonus
            scored_docs.append((final_score, item))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, item in scored_docs[:top_k]:
            results.append(GroundedContext(
                kb_id=item["id"],
                topic=item["topic"],
                resolution=item["resolution"],
                official_link=item.get("official_link", "https://support.apple.com"),
                diagnostic_questions=item.get("diagnostic_questions", []),
                score=round(score, 3)
            ))
        return results
