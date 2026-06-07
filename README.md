# RAG-Based Company Document Visualisation
### Built with LangChain · ChromaDB · UMAP · Plotly · Mistral AI

A complete, end-to-end **Retrieval Augmented Generation (RAG)** demo using a synthetic enterprise knowledge base. Transforms 90 company documents across 6 categories into searchable vector embeddings, visualises them as interactive 2D and 3D cluster plots, and answers natural-language questions using Mistral AI — all grounded in the actual documents with source citations.

---

## What This Project Demonstrates

| Concept | How it's shown |
|---|---|
| **Document embeddings** | 90 documents → 384-dimensional vectors via `all-MiniLM-L6-v2` |
| **Semantic similarity** | Documents with similar meaning cluster together in vector space |
| **Vector database** | ChromaDB stores and retrieves embeddings by cosine similarity |
| **Dimensionality reduction** | UMAP compresses 384 dims → 2D and 3D for human-readable plots |
| **RAG pipeline** | Question → retrieve top-4 chunks → Mistral AI → cited answer |
| **Grounded generation** | LLM answers only from retrieved context — no hallucination |

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                           │
│                                                                 │
│  6 × .txt files  →  90 chunks  →  384-dim embeddings           │
│  (knowledge base)   (splitter)    (sentence-transformers)       │
│                                          │                      │
│                                     ChromaDB                    │
│                                   (vector store)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                              │
│                                                                 │
│  User question  →  embed query  →  ChromaDB similarity search   │
│                                          │                      │
│                                    top-4 chunks                 │
│                                          │                      │
│                                   Mistral AI LLM                │
│                                          │                      │
│                             Grounded answer + sources           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  VISUALISATION PIPELINE                         │
│                                                                 │
│  384-dim embeddings  →  UMAP reduction  →  Plotly charts        │
│                         (2D and 3D)       (interactive)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Knowledge Base — AcmeTech Solutions (Synthetic)

90 synthetic documents for a fictional company. No real people or companies.

| Category | Documents | Example content |
|---|---|---|
| 👤 **Employee Directory** | 20 | Employee profiles with skills, projects, career goals |
| 📋 **HR Policies** | 15 | Leave policies, code of conduct, performance reviews |
| 💰 **Finance & Tax** | 15 | Bonus framework, payroll, procurement approvals |
| ⚙️ **Engineering Docs** | 15 | Kubernetes guide, CI/CD, incident management |
| 🎧 **Customer Support KB** | 15 | Password reset, SSO setup, billing issues |
| 🗺️ **Product Management** | 10 | Roadmap, OKRs, competitive analysis |

**Vocabulary is intentionally distinct per category** so embedding clusters separate cleanly in UMAP — ideal for teaching RAG concepts visually.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB (persistent, local) |
| Dimensionality reduction | UMAP (`umap-learn`) |
| Visualisation | Plotly (interactive 2D + 3D) |
| LLM | Mistral AI (`mistral-small-latest`) |
| Orchestration | LangChain (`create_retrieval_chain`) |
| Notebook | Jupyter / Google Colab |

---

## Project Structure

```
RAG-Based-Company-Document-Visualisation/
│
├── Knowledge Base (source documents)
│   ├── employee_directory.txt          # 20 employee profiles
│   ├── hr_policies.txt                 # 15 HR policy documents
│   ├── finance_tax.txt                 # 15 finance documents
│   ├── engineering_documentation.txt   # 15 engineering docs
│   ├── customer_support_kb.txt         # 15 support articles
│   └── product_management.txt          # 10 product strategy docs
│
├── Pipeline Scripts (run in order)
│   ├── step3_load_documents.py         # Load .txt files → Documents
│   ├── step4_chunk_documents.py        # Split into 90 chunks
│   ├── step4b_visualise_chunks.py      # Pre-embedding diagnostics
│   ├── step5_generate_embeddings.py    # Embed with sentence-transformers
│   ├── step6_store_in_chroma.py        # Index into ChromaDB
│   ├── step7_test_retrieval.py         # Test similarity search
│   ├── step8_umap_visualisation.py     # UMAP 2D + 3D plots
│   ├── step9_create_dataframe.py       # Build enriched plot DataFrame
│   ├── step10_interactive_plot.py      # 4 interactive Plotly charts
│   ├── step11_save_image.py            # Export PNG + HTML
│   ├── step12_advanced_visualisation.py# Full-text hover, filter, spotlight
│   └── step13_rag_query.py             # RAG: ChromaDB + Mistral AI
│
├── Notebook
│   └── rag_visualisation.ipynb         # Complete Colab notebook
│
├── Config
│   ├── requirements.txt                # All Python dependencies
│   ├── master_prompt.txt               # Prompt used to generate knowledge base
│   └── .gitignore                      # Excludes venv, vector_db, pkl, keys
```

---

## Visualisations Produced

### Step 4b — Pre-Embedding Diagnostics
- Bar chart: chunk count per category
- Box plot: word count distribution per category
- Histogram: overall word count spread

### Step 10 — Interactive Cluster Plots
| Chart | What it shows |
|---|---|
| 🗺️ **2D clean** | Six colour-coded document clusters on a dark background |
| 🏷️ **2D labelled** | Category names at cluster centroids, point size = word count |
| 🌐 **3D rotatable** | Full spatial separation — drag to orbit the cluster cloud |
| 🔍 **3D sized** | 3D view with word-count sizing |

### Step 12 — Advanced Visualisations
| Chart | Purpose |
|---|---|
| **Full Text Hover** | Hover over any point to read the complete document |
| **Category Filter** | Dropdown to isolate one category at a time — great for live demos |
| **Similarity Spotlight** | Shows a query ⭐ with dotted lines to the top-5 retrieved documents — makes RAG retrieval *visible* |

---

## RAG Query Examples

```
❓ How many weeks of maternity leave does AcmeTech provide?
💬 AcmeTech provides up to 26 weeks. First 16 weeks at 100% pay,
   weeks 17–26 at 60% pay.  Sources: HR-003

❓ How is the annual performance bonus calculated?
💬 Bonus = Target Bonus × Company Performance Multiplier ×
   Individual Performance Multiplier. Level 4–5 target is 10%
   of base salary.  Sources: FIN-005

❓ What must every Kubernetes service implement before production?
💬 Every service must implement liveness and readiness probes.
   Readiness probes validate DB connectivity and downstream
   service availability.  Sources: ENG-001
```

---

## Quick Start

### 1. Clone and set up
```bash
git clone https://github.com/Charu1806/RAG_LangChain_Demo.git
cd RAG_LangChain_Demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your Mistral API key
```bash
# Get a free key at: https://console.mistral.ai
export MISTRAL_API_KEY="your-key-here"
```

### 3. Run the full pipeline
```bash
python3 step3_load_documents.py
python3 step4_chunk_documents.py
python3 step5_generate_embeddings.py
python3 step6_store_in_chroma.py
python3 step8_umap_visualisation.py
python3 step13_rag_query.py
```

### Or use the Colab notebook
Upload `rag_visualisation.ipynb` to [Google Colab](https://colab.research.google.com) and run all cells.

---

## API Keys Required

| Service | Where to get | Free tier |
|---|---|---|
| **Mistral AI** | https://console.mistral.ai | Generous — no daily cap on `mistral-small` |

No OpenAI key needed. Embeddings run locally via `sentence-transformers`.

---

## Key Learning Concepts

**Why do the clusters appear?**
Documents in the same category share vocabulary. The embedding model learns that *"Kubernetes deployment"* and *"microservice architecture"* are semantically close — closer than *"Kubernetes"* and *"maternity leave"*. UMAP then arranges these similarity relationships in 2D/3D space, making the clusters visible.

**Why does RAG work better than pure LLM?**
A plain LLM can hallucinate policy details it was never trained on. RAG retrieves the *exact* AcmeTech policy document first, then the LLM summarises only that retrieved content — making the answer auditable and verifiable via the source citation.

**What is cosine similarity?**
The score shown during retrieval measures the angle between two embedding vectors. A score near 0 means the documents point in nearly the same direction in 384-dimensional space — i.e., they mean similar things.

---

## Author

Built by Charu Gupta — [@Charu1806](https://github.com/Charu1806)
