"""
Lightweight local retriever for the Evidence Retrieval (RAG) agent.

Uses TF-IDF + cosine similarity over the guideline .txt files in
data/guidelines/. This avoids any dependency on downloading embedding
models, so the project runs fully offline once pip dependencies are
installed.
"""

import os
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GuidelineRetriever:
    def __init__(self, guidelines_dir: str):
        self.guidelines_dir = guidelines_dir
        self.doc_names: List[str] = []
        self.doc_texts: List[str] = []
        self._load_documents()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts) if self.doc_texts else None

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
        if not self.doc_texts or self.doc_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]

        ranked = sorted(zip(self.doc_names, self.doc_texts, scores), key=lambda x: x[2], reverse=True)

        results = []
        for name, text, score in ranked[:top_k]:
            if score <= 0:
                continue
            snippet = text.strip()
            results.append({
                "source": name,
                "snippet": snippet[:600] + ("..." if len(snippet) > 600 else ""),
                "score": round(float(score), 4),
            })
        return results
