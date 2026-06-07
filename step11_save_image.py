"""
Step 11: Save Charts as Images & HTML
=======================================
Exports all four Step 10 charts in two formats:

  PNG  — static, presentation-ready (requires kaleido)
  HTML — fully interactive, works in any browser, no Python needed

Install kaleido before running:
    pip install kaleido

Requires:
    plot_df.pkl   (produced by step9_create_dataframe.py)

Run:
    python step11_save_image.py
"""

import pickle
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from sentence_transformers import SentenceTransformer

# ── Recreate everything needed ─────────────────────────────────────────────────
with open("plot_df.pkl", "rb") as f:
    df = pickle.load(f)

# ── Constants ──────────────────────────────────────────────────────────────────
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

def dark_layout(fig, is_3d=False):
    axis = dict(showgrid=False, zeroline=False, showticklabels=False, title="")
    base = dict(paper_bgcolor=DARK_BG, font_color="white", title_font_size=19,
                legend=dict(title="Category", bgcolor=LEGEND_BG,
                            bordercolor=LEGEND_BC, borderwidth=1))
    if is_3d:
        base["scene"] = dict(bgcolor=DARK_BG,
            xaxis={**axis, "backgroundcolor": DARK_BG},
            yaxis={**axis, "backgroundcolor": DARK_BG},
            zaxis={**axis, "backgroundcolor": DARK_BG})
    else:
        base["plot_bgcolor"] = DARK_BG
        base["xaxis"] = axis
        base["yaxis"] = axis
    return fig.update_layout(**base)

# ── Rebuild the 4 figures ──────────────────────────────────────────────────────
# Chart 1: 2D clean
fig1 = px.scatter(df, x="x", y="y", color="category",
    color_discrete_map=CATEGORY_COLORS, hover_name="title",
    hover_data={"doc_id": True, "word_count": True, "preview": True,
                "x": False, "y": False, "category": False},
    title="🗺️  Vector Embedding Clusters — UMAP 2D  (384 → 2 dimensions)",
    width=1200, height=800)
fig1.update_traces(marker=dict(size=11, opacity=0.88, line=dict(width=0.6, color="white")))
dark_layout(fig1)

# Chart 2: 2D labelled
fig2 = px.scatter(df, x="x", y="y", color="category",
    color_discrete_map=CATEGORY_COLORS, size="word_count", size_max=22,
    hover_name="title",
    hover_data={"doc_id": True, "word_count": True, "preview": True,
                "x": False, "y": False, "category": False},
    title="🏷️  UMAP 2D — Category Labels + Point Size = Word Count",
    width=1200, height=800)
fig2.update_traces(marker=dict(opacity=0.82, line=dict(width=0.5, color="white")))
for cat, color in CATEGORY_COLORS.items():
    sub = df[df["category"] == cat]
    fig2.add_annotation(x=sub["x"].mean(), y=sub["y"].mean(),
        text=f"<b>{cat.upper()}</b>", showarrow=False,
        font=dict(size=14, color=color), bgcolor="rgba(15,15,26,0.70)", borderpad=5)
dark_layout(fig2)

# Chart 3: 3D clean
fig3 = px.scatter_3d(df, x="x3", y="y3", z="z3", color="category",
    color_discrete_map=CATEGORY_COLORS, hover_name="title",
    hover_data={"doc_id": True, "word_count": True, "preview": True,
                "x3": False, "y3": False, "z3": False, "category": False},
    title="🌐  Vector Embedding Clusters — UMAP 3D  (384 → 3 dimensions)",
    width=1200, height=850)
fig3.update_traces(marker=dict(size=6, opacity=0.90, line=dict(width=0.3, color="white")))
dark_layout(fig3, is_3d=True)
fig3.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)))

# Chart 4: 3D sized
fig4 = px.scatter_3d(df, x="x3", y="y3", z="z3", color="category",
    color_discrete_map=CATEGORY_COLORS, size="word_count", size_max=14,
    hover_name="title",
    hover_data={"doc_id": True, "word_count": True, "category": True,
                "preview": True, "x3": False, "y3": False, "z3": False},
    title="🔍  UMAP 3D — Point Size = Word Count  |  Hover = Document Detail",
    width=1200, height=850)
fig4.update_traces(marker=dict(opacity=0.85, line=dict(width=0.2, color="white")))
dark_layout(fig4, is_3d=True)
fig4.update_layout(scene_camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)))

figures = {
    "umap_2d_clusters":         fig1,
    "umap_2d_labelled":         fig2,
    "umap_3d_clusters":         fig3,
    "umap_3d_sized":            fig4,
}

# ── Export: PNG (static, for slides) ──────────────────────────────────────────
print("Saving PNG images (requires kaleido) ...")
png_errors = []
for name, fig in figures.items():
    path = f"{name}.png"
    try:
        fig.write_image(path, width=1200, height=800, scale=2)  # scale=2 → 2x resolution
        size_kb = os.path.getsize(path) // 1024
        print(f"  ✅  {path}  ({size_kb} KB)")
    except Exception as e:
        print(f"  ⚠️  {path}  FAILED: {e}")
        print(f"      → Install kaleido:  pip install kaleido")
        png_errors.append(name)

# ── Export: HTML (interactive, shareable) ─────────────────────────────────────
print("\nSaving interactive HTML files ...")
for name, fig in figures.items():
    path = f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")  # cdn = small file, needs internet
    size_kb = os.path.getsize(path) // 1024
    print(f"  ✅  {path}  ({size_kb} KB)  — open in any browser")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EXPORT SUMMARY")
print("=" * 55)
print(f"  PNG  files : {len(figures) - len(png_errors)} saved  "
      f"({len(png_errors)} failed — install kaleido)")
print(f"  HTML files : {len(figures)} saved")
print()
print("  PNG  → paste into PowerPoint / Google Slides")
print("  HTML → open in browser / share with stakeholders")
print("         (3D charts remain fully rotatable in HTML)")
