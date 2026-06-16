import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.45

EXPANSIONS = {
    "garbage collection": "GC GC_and_WL_Unit Check_gc_required Run_GC",
    "wear leveling": "WL GC_and_WL_Unit",
    "flash translation": "FTL Address_Mapping_Unit",
    "host interface": "Host_Interface_NVMe Host_Interface_SATA",
    "block erase": "GC_and_WL_Unit Flash_Block_Manager",
    "address mapping": "Address_Mapping_Unit LBA physical",
}


def chunk_key(c: dict):
    return (c["file"], c["line_start"], c["name"])


class Retriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

    def expand_query(self, q: str) -> str:
        q_lower = q.lower()
        additions = []

        for phrase, symbols in EXPANSIONS.items():
            if phrase in q_lower:
                additions.append(symbols)

        return q + " " + " ".join(additions) if additions else q

    def expand_context(self, results: list[dict], query: str, max_extra: int = 4) -> list[dict]:#expanding symbols
        seen = {chunk_key(c) for c in results}
        extra = []
        q_lower = query.lower()

        for chunk in results:
            if chunk.get("class_name"):
                siblings = [
                    c.copy() for c in self.metadata
                    if c.get("class_name") == chunk["class_name"]
                    and chunk_key(c) not in seen
                ][:2]

                for c in siblings:
                    c["score"] = chunk.get("score", 0)
                    c["context_reason"] = "same_class"
                    extra.append(c)
                    seen.add(chunk_key(c))

            if any(kw in q_lower for kw in ["gc", "garbage collection", "wear level"]):
                symbol_matches = [
                    c.copy() for c in self.metadata
                    if any(tok in c["name"].lower() for tok in ["gc", "wl", "run_gc", "check_gc"])
                    and chunk_key(c) not in seen
                ][:2]

                for c in symbol_matches:
                    c["score"] = chunk.get("score", 0)
                    c["context_reason"] = "symbol_match"
                    extra.append(c)
                    seen.add(chunk_key(c))

            neighbors = [
                c.copy() for c in self.metadata
                if c["file"] == chunk["file"]
                and chunk_key(c) not in seen
                and abs(c["line_start"] - chunk["line_start"]) < 50
            ][:1]

            for c in neighbors:
                c["score"] = chunk.get("score", 0)
                c["context_reason"] = "nearby"
                extra.append(c)
                seen.add(chunk_key(c))

        return results + extra[:max_extra]

    def search(self, question: str, top_k: int) -> tuple[list[dict], int]:
        expanded = self.expand_query(question)

        emb = self.model.encode(
            [expanded],
            normalize_embeddings=True,
        )
        emb = np.array(emb, dtype=np.float32)

        retrieve_k = max(top_k * 3, 20)
        scores, indices = self.index.search(emb, retrieve_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if float(score) < THRESHOLD:
                continue

            chunk = self.metadata[idx].copy()
            chunk["score"] = round(float(score), 3)
            chunk["context_reason"] = "semantic"
            results.append(chunk)

        results = results[:top_k]
        results = self.expand_context(results, question)

        return results, int(self.index.ntotal)