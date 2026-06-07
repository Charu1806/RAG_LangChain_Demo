"""
Step 5: Generate Embeddings
============================
Converts each chunk's text into a 384-dimensional dense vector using the
sentence-transformers model  all-MiniLM-L6-v2.

Why this model?
  - Lightweight (~80 MB), runs on CPU in seconds
  - 384-dimensional output — compact but expressive
  - Strong performance on semantic similarity benchmarks
  - Perfect for learning / local demos without a GPU or API key

Expected output shape:  (90, 384)
  90  = number of chunks (one per synthetic document)
  384 = embedding dimensions produced by all-MiniLM-L6-v2

Requires:
    chunks.pkl   (produced by step4_chunk_documents.py)

Run:
    python step5_generate_embeddings.py
"""

import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load chunks ────────────────────────────────────────────────────────────────
with open(CACHE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks\n")

# ── Load embedding model ───────────────────────────────────────────────────────
print("Loading sentence-transformers model: all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

# ── Generate embeddings ────────────────────────────────────────────────────────
texts = [chunk.page_content for chunk in chunks]

print(f"Embedding {len(texts)} chunks — this takes ~10–30 seconds on CPU ...")
embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True,   # shows a tqdm progress bar per batch
    batch_size=32,            # process 32 chunks at a time
)

# ── Inspect shape ──────────────────────────────────────────────────────────────
print(f"\nembeddings.shape : {embeddings.shape}")
print(f"  └─ {embeddings.shape[0]} chunks  ×  {embeddings.shape[1]} dimensions")
print(f"\nSample embedding (first 8 values of chunk 0):")
print(f"  {embeddings[0][:8].round(4)}")

# ── Sanity check ───────────────────────────────────────────────────────────────
assert embeddings.shape == (len(chunks), 384), (
    f"Unexpected shape {embeddings.shape}. "
    f"Expected ({len(chunks)}, 384)."
)
print("\n✅ Shape check passed.")

# ── Save for next step ─────────────────────────────────────────────────────────
with open(CACHE_DIR / "embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Saved: embeddings.pkl")
print("Next step: python step6_store_in_chroma.py")
