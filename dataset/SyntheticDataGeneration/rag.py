# app/rag.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.config import EMBEDDING_MODEL

class RAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.docs = []
    
    def build(self, docs: list[str]):
        self.docs = docs
        embeddings = self.embedder.encode(docs, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def query(self, q: str, k=5):
        query_vec = self.embedder.encode([q])
        D, I = self.index.search(query_vec, k)
        return [self.docs[i] for i in I[0]]

if __name__ == "__main__":
    rag = RAG()
    rag.build(["Doc 1", "Doc 2", "Doc 3"])
    print(rag.query("Doc?"))
