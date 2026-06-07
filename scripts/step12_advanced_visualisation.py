"""
Step 12: Advanced Visualisation
=================================
Three genuinely advanced charts that go beyond the basic cluster plot.
Each one is specifically designed to help explain RAG and embeddings to stakeholders.

Chart A — Full Text Hover
  Every point shows the FULL document text in the tooltip (not just 120 chars).
  Wrap long text into readable lines so it fits the tooltip.

Chart B — Category Filter Buttons
  Dropdown buttons let you isolate a single category at a time.
  Great for live demos: "Let me show you just the Engineering documents."

Chart C — Similarity Spotlight
  Pick a query → find its position in embedding space → draw lines
  from the query point to its top-5 nearest neighbours.
  Makes retrieval spatial and visual: "The database found these 5 documents
  because they live closest to your question in vector space."

Requires:
    plot_df.pkl    (produced by step9_create_dataframe.py)
    chunks.pkl     (produced by step4_chunk_documents.py)
    embeddings.pkl (produced by step5_generate_embeddings.py)

Run:
    python step12_advanced_visualisation.py
"""

import pickle
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

# ── Load data ──────────────────────────────────────────────────────────────────
with open(CACHE_DIR / "plot_df.pkl", "rb") as f:
    df = pickle.load(f)

with open(CACHE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open(CACHE_DIR / "embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(f"Loaded df: {df.shape}, chunks: {len(chunks)}, embeddings: {embeddings.shape}\n")

CATEGORY_COLORS = {
    "employee":    "#636EFA",
    "hr":          "#EF553B",
    "finance":     "#00CC96",
    "engineering": "#AB63FA",
    "support":     "#FFA15A",
    "product":     "#19D3F3",
}
DARK_BG = "#0f0f1a"

def dark_2d(fig):
    ax = dict(showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        font_color="white", title_font_size=19, xaxis=ax, yaxis=ax,
        legend=dict(title="Category", bgcolor="rgba(255,255,255,0.08)",
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Chart A: Full Text Hover
# Shows the COMPLETE document text in the tooltip — not just 120 chars.
# ══════════════════════════════════════════════════════════════════════════════
print("Building Chart A: Full Text Hover ...")

def wrap_text(text: str, width: int = 80, max_lines: int = 20) -> str:
    """Wrap long text into tooltip-friendly lines."""
    lines = []
    for para in text.strip().splitlines():
        para = para.strip()
        if not para:
            continue
        wrapped = textwrap.wrap(para, width=width)
        lines.extend(wrapped)
        if len(lines) >= max_lines:
            break
    clipped = lines[:max_lines]
    if len(lines) > max_lines:
        clipped.append("… [truncated]")
    return "<br>".join(clipped)

df["full_text_wrapped"] = [
    wrap_text(c.page_content) for c in chunks
]

figA = px.scatter(
    df, x="x", y="y",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_name="title",
    hover_data={
        "doc_id":           True,
        "category":         True,
        "word_count":       True,
        "full_text_wrapped": True,
        "x": False, "y": False,
    },
    title="📄  Full Document Text on Hover — Explain RAG to Stakeholders",
    width=1100, height=720,
)
figA.update_traces(
    marker=dict(size=12, opacity=0.88, line=dict(width=0.6, color="white")),
    hoverlabel=dict(bgcolor="#1a1a2e", font_size=12, namelength=-1),
)
for cat, color in CATEGORY_COLORS.items():
    sub = df[df["category"] == cat]
    figA.add_annotation(x=sub["x"].mean(), y=sub["y"].mean(),
        text=f"<b>{cat.upper()}</b>", showarrow=False,
        font=dict(size=13, color=color), bgcolor="rgba(15,15,26,0.70)", borderpad=4)
dark_2d(figA)
figA.show()

# ══════════════════════════════════════════════════════════════════════════════
# Chart B: Category Filter Buttons
# Dropdown lets you isolate one category at a time — great for live demos.
# ══════════════════════════════════════════════════════════════════════════════
print("Building Chart B: Category Filter Buttons ...")

figB = go.Figure()

categories = sorted(df["category"].unique())

for cat in categories:
    sub = df[df["category"] == cat]
    figB.add_trace(go.Scatter(
        x=sub["x"], y=sub["y"],
        mode="markers",
        name=cat,
        marker=dict(size=11, color=CATEGORY_COLORS[cat],
                    opacity=0.88, line=dict(width=0.6, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ID: %{customdata[1]}<br>"
            "Words: %{customdata[2]}<br>"
            "<i>%{customdata[3]}</i>"
            "<extra></extra>"
        ),
        customdata=sub[["title", "doc_id", "word_count", "preview"]].values,
    ))

# Build dropdown buttons — "All" + one per category
buttons = [dict(label="All Categories", method="update",
                args=[{"visible": [True] * len(categories)},
                      {"title": "🏷️  All Categories — UMAP 2D Clusters"}])]

for i, cat in enumerate(categories):
    visibility = [j == i for j in range(len(categories))]
    buttons.append(dict(
        label=cat.capitalize(),
        method="update",
        args=[{"visible": visibility},
              {"title": f"🔍  Filtered: {cat.upper()} documents only"}],
    ))

figB.update_layout(
    title="🏷️  Category Filter — Use Dropdown to Isolate a Cluster",
    title_font_size=19,
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.01, xanchor="left",
        y=1.12, yanchor="top",
        bgcolor="#1e1e3a",
        bordercolor="rgba(255,255,255,0.3)",
        font=dict(color="white", size=13),
    )],
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color="white",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    legend=dict(title="Category", bgcolor="rgba(255,255,255,0.08)",
                bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
    width=1100, height=720,
)
figB.show()

# ══════════════════════════════════════════════════════════════════════════════
# Chart C: Similarity Spotlight
# Embeds a query → finds nearest neighbours → draws connection lines.
# Makes retrieval spatial: "These 5 documents live closest to your question."
# ══════════════════════════════════════════════════════════════════════════════
print("Building Chart C: Similarity Spotlight ...")

SPOTLIGHT_QUERY = "How many days of maternity leave are employees entitled to?"
TOP_K           = 5

# Embed the query using the same model
model      = SentenceTransformer("all-MiniLM-L6-v2")
query_vec  = model.encode([SPOTLIGHT_QUERY])[0]          # shape (384,)

# Cosine similarity against all chunk embeddings
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed     = embeddings / (norms + 1e-10)
query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
sims       = normed @ query_norm                         # shape (90,)
top_k_idx  = np.argsort(sims)[::-1][:TOP_K]

# Project query into 2D using the average of its top-k neighbours' coords
# (query itself has no 2D coordinate — we place it at the centroid of top-k)
qx = df.iloc[top_k_idx]["x"].mean()
qy = df.iloc[top_k_idx]["y"].mean()

figC = go.Figure()

# Background: all documents (faded)
figC.add_trace(go.Scatter(
    x=df["x"], y=df["y"],
    mode="markers",
    name="All documents",
    marker=dict(size=8, color=[CATEGORY_COLORS[c] for c in df["category"]],
                opacity=0.25, line=dict(width=0)),
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    customdata=df[["title", "preview"]].values,
))

# Highlight top-k retrieved documents
top_df = df.iloc[top_k_idx].copy()
top_df["rank"] = range(1, TOP_K + 1)
top_df["sim"]  = sims[top_k_idx].round(4)

figC.add_trace(go.Scatter(
    x=top_df["x"], y=top_df["y"],
    mode="markers+text",
    name=f"Top {TOP_K} retrieved",
    marker=dict(size=16, color=[CATEGORY_COLORS[c] for c in top_df["category"]],
                opacity=1.0, line=dict(width=2, color="white"),
                symbol="circle"),
    text=[f"#{r}" for r in top_df["rank"]],
    textposition="top center",
    textfont=dict(size=11, color="white"),
    hovertemplate=(
        "<b>#%{customdata[0]} — %{customdata[1]}</b><br>"
        "Category: %{customdata[2]}<br>"
        "Similarity: %{customdata[3]}<br>"
        "<i>%{customdata[4]}</i>"
        "<extra></extra>"
    ),
    customdata=top_df[["rank", "title", "category", "sim", "preview"]].values,
))

# Draw lines from query point to each retrieved document
for _, row in top_df.iterrows():
    figC.add_shape(type="line",
        x0=qx, y0=qy, x1=row["x"], y1=row["y"],
        line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"))

# Query star marker
figC.add_trace(go.Scatter(
    x=[qx], y=[qy],
    mode="markers+text",
    name="Query",
    marker=dict(size=20, color="#FFD700", symbol="star",
                line=dict(width=2, color="white")),
    text=["QUERY"],
    textposition="bottom center",
    textfont=dict(size=12, color="#FFD700"),
    hovertemplate=f"<b>Query</b><br>{SPOTLIGHT_QUERY}<extra></extra>",
))

figC.update_layout(
    title=f"🔍  Similarity Spotlight — Query: \"{SPOTLIGHT_QUERY[:60]}...\"",
    title_font_size=17,
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color="white",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    legend=dict(bgcolor="rgba(255,255,255,0.08)", bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1),
    width=1100, height=740,
)
figC.show()

# Print the retrieved docs to console as well
print(f"\nQuery: \"{SPOTLIGHT_QUERY}\"")
print(f"Top {TOP_K} retrieved documents:\n")
for rank, idx in enumerate(top_k_idx, 1):
    row = df.iloc[idx]
    print(f"  #{rank}  [{row['category']:>12}]  {row['doc_id']}  "
          f"sim={sims[idx]:.4f}  {row['title']}")

print("\n✅ Step 12 complete — 3 advanced charts rendered.")
print("Next step: python step13_rag_query.py  (Gemini RAG pipeline)")
