"""
Step 10: Interactive Plot — 2D & 3D
=====================================
Produces four publication-quality interactive Plotly charts using the
enriched DataFrame from Step 9.

Charts:
  1. 🗺️  2D scatter       — clean cluster view, hover shows doc_id + title
  2. 🏷️  2D labelled      — centroid labels + point sized by word count
  3. 🌐  3D scatter       — rotatable, full depth separation visible
  4. 🔍  3D with titles   — each point labelled with doc_id on hover

Requires:
    plot_df.pkl   (produced by step9_create_dataframe.py)

Run:
    python step10_interactive_plot.py
"""

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Load enriched DataFrame ────────────────────────────────────────────────────
with open("plot_df.pkl", "rb") as f:
    df = pickle.load(f)

print(f"DataFrame loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Categories: {sorted(df['category'].unique())}\n")

CATEGORY_COLORS = {
    "employee":    "#636EFA",
    "hr":          "#EF553B",
    "finance":     "#00CC96",
    "engineering": "#AB63FA",
    "support":     "#FFA15A",
    "product":     "#19D3F3",
}

DARK_BG   = "#0f0f1a"
LEGEND_BG = "rgba(255,255,255,0.08)"
LEGEND_BC = "rgba(255,255,255,0.20)"

# ══════════════════════════════════════════════════════════════════════════════
# Chart 1: 2D Scatter — clean, hover shows doc_id + title + preview
# ══════════════════════════════════════════════════════════════════════════════
fig1 = px.scatter(
    df,
    x="x", y="y",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_name="title",                           # bold header in tooltip
    hover_data={
        "doc_id":     True,
        "word_count": True,
        "preview":    True,
        "x":          False,
        "y":          False,
        "category":   False,
    },
    title="🗺️  Vector Embedding Clusters — UMAP 2D  (384 → 2 dimensions)",
    width=1050, height=680,
)
fig1.update_traces(
    marker=dict(size=11, opacity=0.88, line=dict(width=0.6, color="white")),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "ID: %{customdata[0]}<br>"
        "Words: %{customdata[1]}<br>"
        "<i>%{customdata[2]}</i>"
        "<extra></extra>"
    ),
)
fig1.update_layout(
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color="white",
    title_font_size=19,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               title="UMAP Dimension 1"),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               title="UMAP Dimension 2"),
    legend=dict(title="Category", bgcolor=LEGEND_BG,
                bordercolor=LEGEND_BC, borderwidth=1),
)
fig1.show()

# ══════════════════════════════════════════════════════════════════════════════
# Chart 2: 2D Labelled — centroid annotations + points sized by word count
# ══════════════════════════════════════════════════════════════════════════════
fig2 = px.scatter(
    df,
    x="x", y="y",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    size="word_count",                            # bigger doc = bigger dot
    size_max=22,
    hover_name="title",
    hover_data={
        "doc_id": True, "word_count": True,
        "preview": True, "x": False, "y": False,
        "category": False, "size": False,
    },
    title="🏷️  UMAP 2D — Category Labels + Point Size = Word Count",
    width=1050, height=680,
)
fig2.update_traces(
    marker=dict(opacity=0.80, line=dict(width=0.5, color="white")),
)

# Bold category label at each cluster centroid
for cat, color in CATEGORY_COLORS.items():
    sub = df[df["category"] == cat]
    fig2.add_annotation(
        x=sub["x"].mean(), y=sub["y"].mean(),
        text=f"<b>{cat.upper()}</b>",
        showarrow=False,
        font=dict(size=14, color=color),
        bgcolor="rgba(15,15,26,0.70)",
        borderpad=5,
    )

fig2.update_layout(
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color="white",
    title_font_size=19,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               title="UMAP Dimension 1"),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               title="UMAP Dimension 2"),
    legend=dict(title="Category", bgcolor=LEGEND_BG,
                bordercolor=LEGEND_BC, borderwidth=1),
)
fig2.show()

# ══════════════════════════════════════════════════════════════════════════════
# Chart 3: 3D Scatter — rotatable, full spatial separation
# ══════════════════════════════════════════════════════════════════════════════
fig3 = px.scatter_3d(
    df,
    x="x3", y="y3", z="z3",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_name="title",
    hover_data={
        "doc_id":     True,
        "word_count": True,
        "preview":    True,
        "x3": False, "y3": False, "z3": False,
        "category": False,
    },
    title="🌐  Vector Embedding Clusters — UMAP 3D  (384 → 3 dimensions)",
    width=1050, height=780,
)
fig3.update_traces(
    marker=dict(size=6, opacity=0.90, line=dict(width=0.3, color="white")),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "ID: %{customdata[0]}<br>"
        "Words: %{customdata[1]}<br>"
        "<i>%{customdata[2]}</i>"
        "<extra></extra>"
    ),
)
fig3.update_layout(
    paper_bgcolor=DARK_BG, font_color="white", title_font_size=19,
    scene=dict(
        bgcolor=DARK_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="Dim 1", backgroundcolor=DARK_BG),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="Dim 2", backgroundcolor=DARK_BG),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="Dim 3", backgroundcolor=DARK_BG),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),  # angled starting view
    ),
    legend=dict(title="Category", bgcolor=LEGEND_BG,
                bordercolor=LEGEND_BC, borderwidth=1),
)
fig3.show()

# ══════════════════════════════════════════════════════════════════════════════
# Chart 4: 3D — sized by word count, doc_id visible on hover
# ══════════════════════════════════════════════════════════════════════════════
fig4 = px.scatter_3d(
    df,
    x="x3", y="y3", z="z3",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    size="word_count",
    size_max=14,
    hover_name="title",
    hover_data={
        "doc_id": True, "word_count": True, "category": True,
        "preview": True, "x3": False, "y3": False, "z3": False,
    },
    title="🔍  UMAP 3D — Point Size = Word Count, Hover = Document Detail",
    width=1050, height=780,
)
fig4.update_traces(
    marker=dict(opacity=0.85, line=dict(width=0.2, color="white")),
)
fig4.update_layout(
    paper_bgcolor=DARK_BG, font_color="white", title_font_size=19,
    scene=dict(
        bgcolor=DARK_BG,
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title="Dim 1", backgroundcolor=DARK_BG),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title="Dim 2", backgroundcolor=DARK_BG),
        zaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title="Dim 3", backgroundcolor=DARK_BG),
        camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)),
    ),
    legend=dict(title="Category", bgcolor=LEGEND_BG,
                bordercolor=LEGEND_BC, borderwidth=1),
)
fig4.show()

# ── Cluster separation table ───────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  CLUSTER SEPARATION SUMMARY")
print("=" * 62)
print(f"  {'Category':>12}  {'Centroid (x,y)':>18}  {'Spread':>7}  Docs")
print("-" * 62)
for cat in sorted(df["category"].unique()):
    sub    = df[df["category"] == cat]
    cx, cy = sub["x"].mean(), sub["y"].mean()
    spread = np.sqrt(sub["x"].var() + sub["y"].var())
    n      = len(sub)
    print(f"  {cat:>12}  ({cx:+.2f}, {cy:+.2f})       {spread:.3f}   {n}")

print("\n💡 Hover over any point → doc title, ID, word count, preview")
print("💡 3D charts: click & drag to rotate | scroll to zoom")
print("💡 Click a legend item to hide/show that category")
print("\n✅ Step 10 complete.")
print("Next step: python step11_rag_query.py")
