# embed_and_store.py
import json
import os
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/chunks/mqsim_chunks.json"
INDEX_PATH  = ROOT / "data/mqsim.index"
META_PATH   = ROOT / "data/mqsim_metadata.json"

token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

model = SentenceTransformer("BAAI/bge-m3", token=token)

def build_embed_text(chunk: dict) -> str:
    """Richer text = better retrieval. Combine metadata + content."""
    header = f"[{chunk['type'].upper()}] {chunk['name']}"
    if chunk["class_name"]:
        header += f" in class {chunk['class_name']}"
    header += f"\nFile: {chunk['file'].split('/')[-1]}"
    if chunk["namespace"]:
        header += f"\nNamespace: {chunk['namespace']}"
    return f"{header}\n\n{chunk['content']}"

def main():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks...")
    texts = [build_embed_text(c) for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=16,        # bge-m3 is large, keep batch small
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save("data/mqsim_embeddings.npy", embeddings)   # <-- add this

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via inner product (works after L2 norm)
    index.add(embeddings)

    Path("data").mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Done. Index: {INDEX_PATH} | Metadata: {META_PATH}")
    print(f"Embedding dim: {dim} | Vectors stored: {index.ntotal}")

if __name__ == "__main__":
    main()