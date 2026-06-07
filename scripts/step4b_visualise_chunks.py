"""
Step 4b: Visualise Chunks (Before Embedding)
=============================================
Diagnostic visualisations on the raw chunks — no embeddings needed.
Useful to verify chunk quality and category balance before committing
to the (slower) embedding step.

Charts produced:
  1. Bar chart     — chunk count per category
  2. Box plot      — word count distribution per category
  3. Histogram     — overall word count spread
  4. Console table — sample chunk preview per category

Requires:
    chunks.pkl  (produced by step4_chunk_documents.py)

Run:
    python step4b_visualise_chunks.py
"""

import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load chunks ────────────────────────────────────────────────────────────────
with open(CACHE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks\n")

# ── Build a DataFrame for easy analysis ───────────────────────────────────────
df = pd.DataFrame([
    {
        "category": c.metadata["category"],
        "source":   c.metadata["source"],
        "word_count":  len(c.page_content.split()),
        "char_count":  len(c.page_content),
        "preview":     c.page_content[:120].replace("\n", " "),
    }
    for c in chunks
])

# ── Console summary ───────────────────────────────────────────────────────────
print("=" * 60)
print("CHUNK SUMMARY BY CATEGORY")
print("=" * 60)
summary = (
    df.groupby("category")["word_count"]
    .agg(chunks="count", min_words="min", max_words="max", avg_words="mean")
    .reset_index()
)
print(summary.to_string(index=False))
print()

# ── Sample preview per category ───────────────────────────────────────────────
print("=" * 60)
print("SAMPLE CHUNK PREVIEW (first chunk per category)")
print("=" * 60)
for category in sorted(df["category"].unique()):
    sample = df[df["category"] == category].iloc[0]
    print(f"\n[{category.upper()}]")
    print(f"  Words   : {sample['word_count']}")
    print(f"  Preview : {sample['preview']}...")

print()

# ── Category colour palette ───────────────────────────────────────────────────
CATEGORY_COLORS = {
    "employee":    "#636EFA",
    "hr":          "#EF553B",
    "finance":     "#00CC96",
    "engineering": "#AB63FA",
    "support":     "#FFA15A",
    "product":     "#19D3F3",
}

# ── Chart 1: Chunk count per category (bar chart) ────────────────────────────
count_df = df.groupby("category").size().reset_index(name="chunk_count")

fig1 = px.bar(
    count_df,
    x="category",
    y="chunk_count",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    text="chunk_count",
    title="📊 Chunk Count per Category",
    labels={"category": "Category", "chunk_count": "Number of Chunks"},
)
fig1.update_traces(textposition="outside")
fig1.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#eeeeee"),
    title_font_size=18,
)
fig1.show()

# ── Chart 2: Word count distribution per category (box plot) ─────────────────
fig2 = px.box(
    df,
    x="category",
    y="word_count",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    points="all",                   # show individual chunk dots
    title="📦 Word Count Distribution per Category",
    labels={"category": "Category", "word_count": "Words per Chunk"},
)
fig2.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#eeeeee"),
    title_font_size=18,
)
fig2.show()

# ── Chart 3: Overall word count histogram ────────────────────────────────────
fig3 = px.histogram(
    df,
    x="word_count",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    nbins=30,
    barmode="overlay",
    opacity=0.7,
    title="📈 Word Count Spread Across All Chunks",
    labels={"word_count": "Words per Chunk", "count": "Number of Chunks"},
)
fig3.update_layout(
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#eeeeee"),
    title_font_size=18,
    legend_title_text="Category",
)
fig3.show()

# ── Chart 4: Combined dashboard (subplots) ────────────────────────────────────
fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Chunk Count per Category", "Avg Word Count per Category"),
)

avg_df = df.groupby("category")["word_count"].mean().reset_index()
avg_df.columns = ["category", "avg_words"]

for _, row in count_df.iterrows():
    fig4.add_trace(
        go.Bar(
            name=row["category"],
            x=[row["category"]],
            y=[row["chunk_count"]],
            marker_color=CATEGORY_COLORS[row["category"]],
            showlegend=True,
        ),
        row=1, col=1,
    )

for _, row in avg_df.iterrows():
    fig4.add_trace(
        go.Bar(
            name=row["category"],
            x=[row["category"]],
            y=[round(row["avg_words"], 1)],
            marker_color=CATEGORY_COLORS[row["category"]],
            showlegend=False,
        ),
        row=1, col=2,
    )

fig4.update_layout(
    title_text="📋 Knowledge Base Dashboard — Pre-Embedding Diagnostics",
    title_font_size=18,
    plot_bgcolor="white",
    barmode="group",
)
fig4.show()

print("✅ All charts rendered.")
print("Next step: python step5_embed_and_store.py")
