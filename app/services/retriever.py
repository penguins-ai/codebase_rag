import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.50
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

    def expand_query(q: str) -> str:
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

    def search(self, question: str, top_k: int) -> tuple[list[dict], int]:
        emb = self.model.encode([self.expand_query(question)], normalize_embeddings=True)#creates embedding for query using same model
        emb = np.array(emb, dtype=np.float32)
        scores, indices = self.index.search(emb, top_k)#find top k closest chunks to search

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and float(score) >= THRESHOLD:
                chunk = self.metadata[idx].copy()
                chunk["score"] = round(float(score), 3)
                results.append(chunk)

        return results, int(self.index.ntotal)  