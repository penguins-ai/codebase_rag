import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

THRESHOLD = 0.45
EXPANSIONS = {
    "garbage collection": "GC GC_and_WL_Unit",
    "wear leveling":      "WL GC_and_WL_Unit",
    "flash translation":  "FTL Address_Mapping_Unit",
    "host interface":     "Host_Interface_NVMe Host_Interface_SATA",
    "block erase":        "GC_and_WL_Unit Flash_Block_Manager",
    "address mapping":    "Address_Mapping_Unit LBA physical",
}


class Retriever:
    

    # Only expand natural language → code symbols
# The index is full of abbreviations/class names, not prose

    def expand_query(self, q: str) -> str:
        q_lower = q.lower()
        additions = []

        for phrase, symbols in EXPANSIONS.items():
            if phrase in q_lower:
                additions.append(symbols)

        return q + " " + " ".join(additions) if additions else q
    
    def __init__(self, index_path: str, metadata_path: str, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        corpus = [
                (c["name"] + " " + c.get("content", "")).split()
                for c in self.metadata
            ]
        self.bm25 = BM25Okapi(corpus)

    # def search(self, question, top_k):
    #     # dense
    #     emb = self.model.encode([self.expand_query(question)], normalize_embeddings=True)
    #     scores, indices = self.index.search(np.array(emb, dtype=np.float32), top_k)
        
    #     # bm25
    #     bm25_scores = self.bm25.get_scores(question.split())
    #     bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

    #     # merge by index, take union
    #     seen = set()
    #     results = []
    #     for score, idx in zip(scores[0], indices[0]):
    #         if idx != -1 and float(score) >= THRESHOLD:
    #             seen.add(idx)
    #             chunk = self.metadata[idx].copy()
    #             chunk["score"] = round(float(score), 3)
    #             results.append(chunk)
        
    #     for idx in bm25_top:
    #         if idx not in seen and bm25_scores[idx] > 0:
    #             chunk = self.metadata[idx].copy()
    #             chunk["score"] = round(float(bm25_scores[idx] / 10), 3)  # normalize roughly
    #             results.append(chunk)

    #     return results[:top_k], int(self.index.ntotal)

    def search(self, question: str, top_k: int):
        expanded_query = self.expand_query(question)

        emb = self.model.encode(
            [expanded_query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            np.array(emb, dtype=np.float32),
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if float(score) < THRESHOLD:
                continue

            chunk = self.metadata[idx].copy()
            chunk["score"] = round(float(score), 3)

            results.append(chunk)

        return results, int(self.index.ntotal)