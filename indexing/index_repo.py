from pathlib import Path
from collections import Counter
import json
from dataclasses import asdict
from app.chunker import extract_chunks, CodeChunk


IGNORE_DIRS = {
    ".git",
    "build",
    "cmake-build-debug",
    "cmake-build-release",
    "gnn_training",
    "traces",
    "fast18",
}
IGNORE_FILES = {
    "rapidxml.hpp",
    "rapidxml_iterators.hpp",
    "rapidxml_print.hpp",
    "rapidxml_utils.hpp",
}


def should_skip(filepath: Path) -> bool:
    return any(part in IGNORE_DIRS for part in filepath.parts)


def collect_all_chunks(repo_path: str) -> list[CodeChunk]:
    repo = Path(repo_path).resolve()
    all_chunks: list[CodeChunk] = []

    for filepath in repo.rglob("*"):
        if should_skip(filepath):
            continue
        if filepath.name in IGNORE_FILES:
            continue

        if filepath.suffix in (".cpp", ".h", ".cc", ".hpp"):
            try:
                chunks = extract_chunks(str(filepath))
                all_chunks.extend(chunks)
                print(f"{filepath.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"SKIP {filepath.name}: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(Counter(chunk.type for chunk in all_chunks))

    return all_chunks


if __name__ == "__main__":
    MQSIM_PATH = "/home/av/Projects/try_new/MQSim"  # change this
    all_chunks = collect_all_chunks(MQSIM_PATH)
    out_path = Path("data/chunks/mqsim_chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump([asdict(chunk) for chunk in all_chunks], f, indent=2)

    print(f"Saved to {out_path}")