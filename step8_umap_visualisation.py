"""
Step 8: Visualise Clusters — UMAP 2D & 3D
==========================================
Reduces 384-dimensional embeddings → 2D and 3D using UMAP,
then plots interactive Plotly scatter charts coloured by category.

What UMAP does:
  - Preserves the LOCAL neighbourhood structure of high-dimensional vectors
  - Documents that are semantically similar end up close together
  - Documents from different categories separate into visible clusters
  - Each point = one synthetic AcmeTech document (90 points total)

UMAP key parameters:
  n_neighbors  : how many nearby points to consider when learning structure
                 lower  → tighter local clusters
                 higher → more global shape preserved
  min_dist     : minimum spacing between points in the 2D/3D layout
                 lower  → denser, more packed clusters
                 higher → points spread out more
  metric       : distance measure — cosine fits normalised embeddings best

Requires:
    chunks.pkl      (produced by step4_chunk_documents.py)
    embeddings.pkl  (produced by step5_generate_embeddings.py)

Run:
    python step8_umap_visualisation.py
"""

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP

# ── Load data ──────────────────────────────────────────────────────────────────
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(f"Chunks    : {len(chunks)}")
print(f"Embeddings: {embeddings.shape}  (384-dim vectors)\n")

# ── Build metadata DataFrame ───────────────────────────────────────────────────
df = pd.DataFrame([
    {
        "category": c.metadata["category"],
        "source":   c.metadata["source"],
        "preview":  c.page_content.strip()[:120].replace("\n", " "),
        # Extract document title from the first non-empty line
        "title":    next(
            (line.strip() for line in c.page_content.splitlines()
             if line.strip() and not line.startswith("Document") and len(line.strip()) > 5),
            c.metadata["category"]
        ),
    }
    for c in chunks
])

CATEGORY_COLORS = {
    "employee":    "#636EFA",
    "hr":          "#EF553B",
    "finance":     "#00CC96",
    "engineering": "#AB63FA",
    "support":     "#FFA15A",
    "product":     "#19D3F3",
}

# ── UMAP shared settings ───────────────────────────────────────────────────────
UMAP_PARAMS = dict(
    n_neighbors=10,    # small dataset (90 pts) — use lower n_neighbors
    min_dist=0.1,      # tight clusters for clear category separation
    metric="cosine",   # cosine distance matches normalised embeddings
    random_state=42,   # reproducible layout
)

# ══════════════════════════════════════════════════════════════════════════════
# 2D UMAP
# ══════════════════════════════════════════════════════════════════════════════
print("Running UMAP 2D reduction  (384 → 2 dimensions) ...")
umap_2d = UMAP(n_components=2, **UMAP_PARAMS)
coords_2d = umap_2d.fit_transform(embeddings)
print(f"  Output shape: {coords_2d.shape}\n")

df["x"] = coords_2d[:, 0]
df["y"] = coords_2d[:, 1]

# ── Chart 1: Basic 2D scatter ──────────────────────────────────────────────────
fig_2d = px.scatter(
    df,
    x="x", y="y",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_data={"x": False, "y": False, "category": True, "preview": True},
    title="🗺️  UMAP 2D — AcmeTech Document Clusters (384 → 2 dimensions)",
    labels={"category": "Category"},
    width=1000,
    height=650,
)
fig_2d.update_traces(
    marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white"))
)
fig_2d.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0f0f1a",
    font_color="white",
    title_font_size=18,
    legend=dict(
        bgcolor="rgba(255,255,255,0.08)",
        bordercolor="rgba(255,255,255,0.2)",
        borderwidth=1,
    ),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
)
fig_2d.show()

# ── Chart 2: 2D scatter with category centroid labels ─────────────────────────
fig_2d_labels = px.scatter(
    df,
    x="x", y="y",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_data={"x": False, "y": False, "category": True, "preview": True},
    title="🏷️  UMAP 2D — With Category Labels",
    width=1000,
    height=650,
)
fig_2d_labels.update_traces(
    marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white"))
)

# Add a text annotation at each category's centroid
for category, color in CATEGORY_COLORS.items():
    subset = df[df["category"] == category]
    cx = subset["x"].mean()
    cy = subset["y"].mean()
    fig_2d_labels.add_annotation(
        x=cx, y=cy,
        text=f"<b>{category.upper()}</b>",
        showarrow=False,
        font=dict(size=13, color=color),
        bgcolor="rgba(15,15,26,0.6)",
        borderpad=4,
    )

fig_2d_labels.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0f0f1a",
    font_color="white",
    title_font_size=18,
    legend=dict(bgcolor="rgba(255,255,255,0.08)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
)
fig_2d_labels.show()

# ══════════════════════════════════════════════════════════════════════════════
# 3D UMAP
# ══════════════════════════════════════════════════════════════════════════════
print("Running UMAP 3D reduction  (384 → 3 dimensions) ...")
umap_3d = UMAP(n_components=3, **UMAP_PARAMS)
coords_3d = umap_3d.fit_transform(embeddings)
print(f"  Output shape: {coords_3d.shape}\n")

df["x3"] = coords_3d[:, 0]
df["y3"] = coords_3d[:, 1]
df["z3"] = coords_3d[:, 2]

# ── Chart 3: Interactive 3D scatter ───────────────────────────────────────────
fig_3d = px.scatter_3d(
    df,
    x="x3", y="y3", z="z3",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_data={"x3": False, "y3": False, "z3": False,
                "category": True, "preview": True},
    title="🌐  UMAP 3D — AcmeTech Document Clusters (384 → 3 dimensions)",
    labels={"category": "Category"},
    width=1000,
    height=750,
)
fig_3d.update_traces(
    marker=dict(size=6, opacity=0.90, line=dict(width=0.3, color="white"))
)
fig_3d.update_layout(
    paper_bgcolor="#0f0f1a",
    font_color="white",
    title_font_size=18,
    scene=dict(
        bgcolor="#0f0f1a",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="", backgroundcolor="#0f0f1a"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="", backgroundcolor="#0f0f1a"),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="", backgroundcolor="#0f0f1a"),
    ),
    legend=dict(bgcolor="rgba(255,255,255,0.08)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
)
fig_3d.show()

# ── Chart 4: Side-by-side 2D vs 3D (static summary) ──────────────────────────
print("\nCluster separation summary:")
print("-" * 45)
for category in sorted(df["category"].unique()):
    subset = df[df["category"] == category]
    cx = subset["x"].mean()
    cy = subset["y"].mean()
    spread = np.sqrt(subset["x"].var() + subset["y"].var())
    print(f"  {category:>12} | centroid ({cx:+.2f}, {cy:+.2f}) | spread {spread:.3f}")

print("\n✅ UMAP visualisation complete.")
print("Tip: hover over any point to see the document preview.")
print("Tip: in the 3D chart, click and drag to rotate the cluster cloud.")
print("\nNext step: python step9_rag_query.py")
