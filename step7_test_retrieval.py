"""
Step 7: Test Retrieval
=======================
Queries the ChromaDB vector store with natural language questions and prints
the top-k most semantically similar chunks, along with their metadata and
similarity scores.

similarity_search_with_score() returns (Document, score) tuples where:
  - Score is L2 distance (lower = more similar) when using HuggingFaceEmbeddings
  - Score is cosine distance (lower = more similar) with normalize_embeddings=True
  - Range is roughly 0.0 (identical) → 2.0 (completely dissimilar)
  - A score below 0.5 is a strong match for this dataset

Requires:
    ./vector_db/   (produced by step6_store_in_chroma.py)

Run:
    python step7_test_retrieval.py
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Reload ChromaDB from disk ──────────────────────────────────────────────────
PERSIST_DIR = "./vector_db"

if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(
        f"Vector store not found at '{PERSIST_DIR}'.\n"
        "Run step6_store_in_chroma.py first."
    )

embedding_fn = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_fn,
)
print(f"Loaded ChromaDB — {db._collection.count()} documents\n")

# ── Helper: pretty-print results ──────────────────────────────────────────────
def run_query(query: str, k: int = 3) -> None:
    print("=" * 70)
    print(f"  QUERY : {query}")
    print("=" * 70)

    results = db.similarity_search_with_score(query, k=k)

    for rank, (doc, score) in enumerate(results, start=1):
        category = doc.metadata.get("category", "unknown")
        source   = doc.metadata.get("source",   "unknown")
        preview  = doc.page_content.strip()[:400].replace("\n", " ")

        # Interpret the score
        if score < 0.30:
            confidence = "🟢 Strong match"
        elif score < 0.60:
            confidence = "🟡 Good match"
        else:
            confidence = "🔴 Weak match"

        print(f"\n  Result #{rank}")
        print(f"  Category   : {category}")
        print(f"  Source     : {source}")
        print(f"  Score      : {score:.4f}  ({confidence})")
        print(f"  Content    :\n")
        print(f"    {preview}...")
        print()

    print("-" * 70 + "\n")


# ── Run a variety of queries across all 6 categories ─────────────────────────
queries = [
    # HR
    "How many maternity leave days are allowed?",
    "What is the work from home policy?",

    # Engineering
    "How are Kubernetes deployments configured?",
    "What is the incident management process?",

    # Finance
    "How is the annual bonus calculated?",
    "What expenses can I claim for reimbursement?",

    # Support
    "How do I reset my account password?",
    "How do I set up Single Sign-On?",

    # Product
    "What is the product roadmap for next year?",
    "How are features prioritised?",

    # Employee
    "Who manages the engineering team?",
    "What are Sarah Mitchell's current projects?",
]

for query in queries:
    run_query(query, k=3)

print("✅ Retrieval test complete.")
print("Next step: python step8_umap_visualisation.py")
