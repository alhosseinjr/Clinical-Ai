"""
Semantic local retriever for the Evidence Retrieval (RAG) agent.

Uses sentence-transformers (all-MiniLM-L6-v2) to encode guidelines and
queries into dense vectors, then uses cosine similarity to find the most
relevant snippets. This understands semantic meaning (e.g., "heart attack"
matches "myocardial infarction") much better than keyword-based TF-IDF.

The model (~80MB) is downloaded automatically on first run and cached locally.
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util


class GuidelineRetriever:
    def __init__(self, guidelines_dir: str):
        self.guidelines_dir = guidelines_dir
        self.doc_names: List[str] = []
        self.doc_texts: List[str] = []
        self._load_documents()

        # Load the embedding model (cached locally after first download)
        # all-MiniLM-L6-v2 is fast, small (~80MB), and high-quality for semantic search
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode all documents once during initialization
        if self.doc_texts:
            print(f"Encoding {len(self.doc_texts)} guideline documents...")
            self.doc_embeddings = self.model.encode(self.doc_texts, convert_to_tensor=True)
        else:
            self.doc_embeddings = None

    def _load_documents(self):
        if not os.path.isdir(self.guidelines_dir):
            return
        for fname in sorted(os.listdir(self.guidelines_dir)):
            if fname.endswith(".txt"):
                path = os.path.join(self.guidelines_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                self.doc_names.append(fname)
                self.doc_texts.append(text)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return the top_k most relevant guideline snippets for a query string."""
        if not self.doc_texts or self.doc_embeddings is None:
            return []

        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between query and all documents
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]

        # Get the top-k highest scoring documents
        top_results = cos_scores.topk(top_k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            score_val = score.item()
            # Only include results with positive similarity
            if score_val <= 0:
                continue
            
            name = self.doc_names[idx.item()]
            text = self.doc_texts[idx.item()]
            
            snippet = text.strip()
            results.append({
                "source": name,
                "snippet": snippet[:600] + ("..." if len(snippet) > 600 else ""),
                "score": round(float(score_val), 4),
            })
        return results