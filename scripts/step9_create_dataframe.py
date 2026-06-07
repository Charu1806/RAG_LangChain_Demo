"""
Step 9: Create DataFrame
=========================
Assembles the final plot-ready DataFrame by combining:
  - 2D UMAP coordinates  (x, y)       from Step 8
  - 3D UMAP coordinates  (x3, y3, z3) from Step 8
  - Category label                     from chunk metadata
  - Source filename                    from chunk metadata
  - Document ID          (e.g. HR-001) parsed from chunk text
  - Document title                     parsed from chunk text
  - Word count                         computed from chunk text
  - Short preview                      first 120 characters

Fix vs original snippet:
  - Variable name:  embedding_2d  →  coords_2d  (matches Step 8 output)
  - Added 3D cols, doc_id, title, word_count, preview for richer hover tooltips
  - Added validation and printed summary before saving

Requires:
    chunks.pkl       (produced by step4_chunk_documents.py)
    embeddings.pkl   (produced by step5_generate_embeddings.py)

Run:
    python step9_create_dataframe.py
    (or run after step8_umap_visualisation.py which already has coords in memory)
"""

import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT
import re
import numpy as np
import pandas as pd
from umap import UMAP

# ── Load chunks & embeddings ───────────────────────────────────────────────────
with open(CACHE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open(CACHE_DIR / "embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(f"Chunks     : {len(chunks)}")
print(f"Embeddings : {embeddings.shape}\n")

# ── Run UMAP 2D & 3D (same params as Step 8 for reproducibility) ──────────────
UMAP_PARAMS = dict(n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42)

print("Running UMAP 384 → 2D ...")
coords_2d = UMAP(n_components=2, **UMAP_PARAMS).fit_transform(embeddings)

print("Running UMAP 384 → 3D ...")
coords_3d = UMAP(n_components=3, **UMAP_PARAMS).fit_transform(embeddings)
print()

# ── Helper: parse doc_id and title from chunk text ────────────────────────────
def parse_doc_id(text: str) -> str:
    """Extract Document ID like EMP-001, HR-003 from the chunk header."""
    match = re.search(r"(EMP|HR|FIN|ENG|SUP|PM)-\d+", text)
    return match.group(0) if match else "UNKNOWN"

def parse_title(text: str) -> str:
    """
    Extract the document title — the line that starts with 'Title:'.
    Falls back to the first non-empty, non-header line.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("title:"):
            return line.split(":", 1)[-1].strip()
    # fallback: first meaningful line
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("Document") and len(line) > 8:
            return line[:80]
    return "Untitled"

# ── Build the DataFrame ────────────────────────────────────────────────────────
rows = []
for i, chunk in enumerate(chunks):
    text     = chunk.page_content.strip()
    doc_id   = parse_doc_id(text)
    title    = parse_title(text)
    preview  = text[:120].replace("\n", " ")

    rows.append({
        # UMAP coordinates
        "x":          coords_2d[i, 0],
        "y":          coords_2d[i, 1],
        "x3":         coords_3d[i, 0],
        "y3":         coords_3d[i, 1],
        "z3":         coords_3d[i, 2],
        # Metadata
        "category":   chunk.metadata["category"],
        "source":     chunk.metadata["source"],
        "doc_id":     doc_id,
        "title":      title,
        # Content stats
        "word_count": len(text.split()),
        "char_count": len(text),
        "preview":    preview,
    })

df = pd.DataFrame(rows)

# ── Validate ───────────────────────────────────────────────────────────────────
assert len(df) == len(chunks),      f"Row count mismatch: {len(df)} vs {len(chunks)}"
assert df["x"].isna().sum() == 0,   "NaN values found in x column"
assert df["y"].isna().sum() == 0,   "NaN values found in y column"
assert df["category"].nunique() == 6, f"Expected 6 categories, got {df['category'].nunique()}"

# ── Print summary ──────────────────────────────────────────────────────────────
print("=" * 60)
print(f"DataFrame shape : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"Columns         : {list(df.columns)}")
print()
print(df.groupby("category").agg(
    count=("doc_id", "count"),
    avg_words=("word_count", "mean"),
    sample_doc=("doc_id", "first"),
).to_string())

print("\nSample rows:")
print(df[["doc_id", "category", "title", "word_count", "x", "y"]].head(8).to_string(index=False))

# ── Save for next step ─────────────────────────────────────────────────────────
with open(CACHE_DIR / "plot_df.pkl", "wb") as f:
    pickle.dump(df, f)

print("\n✅ DataFrame saved to plot_df.pkl")
print("Next step: python step10_plotly_cluster_plot.py")
