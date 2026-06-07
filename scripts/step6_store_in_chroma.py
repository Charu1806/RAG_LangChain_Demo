"""
Step 6: Store Embeddings in ChromaDB
======================================
Indexes all 90 document chunks into a persistent ChromaDB vector store using
the same all-MiniLM-L6-v2 embedding function.

What ChromaDB does:
  - Stores the raw text, metadata (category, source), and embedding vectors together
  - Persists everything to ./vector_db so you do not need to re-embed on each run
  - Exposes similarity_search() for retrieval in the RAG step

Import note:
  Use  langchain_community.embeddings  NOT  langchain.embeddings
  The langchain.embeddings path is deprecated and will raise a warning/error
  in langchain >= 0.2.

Requires:
    chunks.pkl   (produced by step4_chunk_documents.py)

Run:
    python step6_store_in_chroma.py
"""

import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Load chunks ────────────────────────────────────────────────────────────────
with open(CACHE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks\n")

# ── Embedding function ─────────────────────────────────────────────────────────
# Must use the same model as Step 5 so vectors are in the same space.
print("Loading HuggingFaceEmbeddings wrapper (all-MiniLM-L6-v2) ...")
embedding_fn = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},      # change to "cuda" if you have a GPU
    encode_kwargs={"normalize_embeddings": True},  # cosine-friendly unit vectors
)
print("Embedding function ready.\n")

# ── Build ChromaDB vector store ────────────────────────────────────────────────
PERSIST_DIR = str(PROJECT_ROOT / "vector_db")

# Remove old DB if re-running, to avoid duplicate documents
if os.path.exists(PERSIST_DIR):
    import shutil
    shutil.rmtree(PERSIST_DIR)
    print(f"Removed existing vector store at {PERSIST_DIR}")

print(f"Embedding and indexing {len(chunks)} chunks into ChromaDB ...")
print("(This re-embeds via the wrapper — takes ~10–30 seconds on CPU)\n")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_fn,
    persist_directory=PERSIST_DIR,
)

# Note: db.persist() is no longer needed in chromadb >= 0.4
# The vector store auto-persists to disk on creation.

# ── Verify ────────────────────────────────────────────────────────────────────
collection = db._collection
count = collection.count()
print(f"\n✅ ChromaDB collection contains {count} documents")
print(f"   Persisted to: {os.path.abspath(PERSIST_DIR)}")

# ── Quick similarity search smoke test ────────────────────────────────────────
print("\n── Smoke test: similarity search ──────────────────────────────")
test_queries = [
    ("What is the maternity leave policy?",  "hr"),
    ("How does Kubernetes deployment work?",  "engineering"),
    ("How do I reset my password?",           "support"),
]

for query, expected_category in test_queries:
    results = db.similarity_search(query, k=1)
    top = results[0]
    actual_category = top.metadata["category"]
    match = "✅" if actual_category == expected_category else "⚠️ "
    print(f"\n  Query    : {query}")
    print(f"  Expected : {expected_category}")
    print(f"  Got      : {actual_category}  {match}")
    print(f"  Preview  : {top.page_content[:100].replace(chr(10), ' ')}...")

print("\nSaved: ./vector_db/")
print("Next step: python step7_umap_visualisation.py")
