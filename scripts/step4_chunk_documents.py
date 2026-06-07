"""
Step 4: Chunk Documents
=======================
Splits each loaded Document on the "===============================" delimiter
that separates individual synthetic documents within each knowledge base file.
Produces one chunk per document (90 total), preserving category metadata.

Requires:
    docs.pkl  (produced by step3_load_documents.py)

Run:
    python step4_chunk_documents.py
"""

import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT
from collections import Counter
from langchain.schema import Document

SEPARATOR = "==============================="
MIN_CHUNK_LENGTH = 100   # characters — filters empty separator lines

# Load docs from Step 3
with open(CACHE_DIR / "docs.pkl", "rb") as f:
    docs = pickle.load(f)

print(f"Loaded {len(docs)} documents from docs.pkl\n")

chunks = []
for doc in docs:
    raw_chunks = doc.page_content.split(SEPARATOR)
    for raw in raw_chunks:
        text = raw.strip()
        if len(text) < MIN_CHUNK_LENGTH:
            continue
        chunks.append(
            Document(
                page_content=text,
                metadata=doc.metadata.copy()   # carries category + source forward
            )
        )

counts = Counter(c.metadata["category"] for c in chunks)

print(f"Total chunks: {len(chunks)}\n")
for category, count in sorted(counts.items()):
    print(f"  {category:>12} : {count} chunks")

# Save for next step
with open(CACHE_DIR / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("\nSaved: chunks.pkl")
