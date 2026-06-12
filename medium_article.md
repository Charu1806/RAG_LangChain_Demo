# How I Built a RAG Pipeline from Scratch — and What It Taught Me About Modern AI

*A hands-on journey through embeddings, vector databases, semantic search, and why retrieval matters more than you might think.*

---

## Why I Built This

I had been reading about Retrieval Augmented Generation for months. The papers made sense on paper. The diagrams looked reasonable. But I didn't really *understand* it until I built it myself.

So I created a complete RAG pipeline from scratch: 90 synthetic enterprise documents spanning HR policies, engineering documentation, finance procedures, product strategy, and employee records — all made searchable through natural language.

The project is open source: [RAG_LangChain_Demo on GitHub](https://github.com/Charu1806/RAG_LangChain_Demo).

This is what I learned.

---

## What I Built

The pipeline covers the full journey from raw text to grounded answers:

1. **90 synthetic company documents** across 6 categories, generated to mimic real enterprise knowledge bases
2. **Chunking and embedding** using `sentence-transformers/all-MiniLM-L6-v2` — a model that runs entirely locally
3. **Persistent vector storage** in ChromaDB
4. **Dimensionality reduction and visualization** using UMAP, collapsing 384-dimensional vectors into interactive 2D and 3D plots with Plotly
5. **Natural-language question answering** via Mistral AI, orchestrated with LangChain

The entire pipeline is broken into 13 sequential scripts so each step is independently understandable — nothing is hidden inside a magic abstraction.

---

## The Surprising Similarity to How Modern AI Systems Work

One of the biggest "aha" moments from this project was realizing that the architecture I built is surprisingly similar to how many production AI systems operate today.

Before this project, terms like *embeddings*, *vector databases*, *semantic search*, and *retrieval* felt abstract. After building the pipeline myself, I realized that the workflow is conceptually straightforward:

```
Documents
    ↓
Embeddings
    ↓
Vector Database
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
LLM
    ↓
Answer
```

This is not the same as how a foundation model is trained internally. Large language models learn statistical patterns from enormous datasets during training and store that knowledge within billions of parameters.

However, modern AI applications often add a retrieval layer on top of the model. This retrieval layer follows a remarkably similar pattern to what I built:

1. Convert information into embeddings.
2. Store those embeddings in a vector database.
3. Convert the user's question into an embedding.
4. Find semantically similar information.
5. Add the retrieved information to the prompt.
6. Ask the LLM to generate a response.

In other words, many enterprise AI assistants are not relying solely on what the model memorized during training. They first search for relevant information and then provide that information as context to the model. This approach is known as **Retrieval Augmented Generation (RAG)**.

What fascinated me most was seeing this process *visually*.

When I plotted the embeddings generated from HR policies, engineering documentation, finance procedures, product strategy documents, and employee records, the documents naturally formed clusters.

*[IMAGE: UMAP visualization showing document clusters by category]*

This made semantic search feel much less magical. The system isn't searching for exact keywords. Instead, it is searching for **nearby concepts in a high-dimensional meaning space**.

The same intuition powers many enterprise AI assistants today. Vector databases are commonly used to retrieve relevant documents based on semantic similarity before the information is passed to an LLM.

Perhaps the biggest lesson for me was this:

> **Vector databases don't store documents. They store representations of meaning.**

Once I understood that idea, many modern AI architectures became much easier to reason about.

---

## The Technical Stack — and Why Each Piece Exists

**`sentence-transformers/all-MiniLM-L6-v2`** — A small, fast embedding model that runs locally with no API calls. Each piece of text becomes a 384-dimensional vector. Small enough to iterate quickly; good enough to produce meaningful semantic clusters.

**ChromaDB** — A lightweight vector database that persists embeddings to disk. It handles the similarity search so you don't have to implement nearest-neighbor lookup yourself.

**UMAP** — Uniform Manifold Approximation and Projection. Takes those 384-dimensional vectors and compresses them to 2D or 3D while trying to preserve neighborhood structure. This is what makes the clusters visible.

**LangChain** — Orchestration. Handles the retrieve-then-generate loop so the plumbing between ChromaDB and Mistral doesn't need to be written by hand.

**Mistral AI** — The generation step. Given retrieved context, it produces a grounded answer with citations back to source documents.

---

## What Surprised Me

**Chunking matters more than I expected.** How you split documents affects what gets retrieved. A chunk that cuts across a meaningful boundary can cause the system to retrieve the wrong half of an answer.

**The clusters were interpretable without labeling them.** When I visualized the UMAP output, the six document categories separated clearly in 2D space — without telling the model what the categories were. The embedding model had never seen these specific documents. It grouped them by meaning anyway.

**Hallucination dropped noticeably with retrieval.** Asking Mistral a question cold (no context) versus asking it with retrieved passages produced measurably different answers. With retrieval, responses cited specific policy numbers, employee names, and figures from the source documents. Without it, responses were confident but generic.

---

## What I'd Do Differently

- **Evaluate retrieval quality separately from generation quality.** I spent too long tuning the LLM prompt before realizing the retrieval itself was the bottleneck.
- **Use a hybrid search** (keyword + semantic) for documents with specific identifiers like policy numbers or employee IDs. Pure semantic search struggles with exact lookups.
- **Add a reranker.** Retrieving the top-k chunks by cosine similarity is a good first pass, but a cross-encoder reranker reading the query alongside each candidate would improve precision.

---

## Where to Go From Here

If this post resonated, the full code is at [github.com/Charu1806/RAG_LangChain_Demo](https://github.com/Charu1806/RAG_LangChain_Demo). The 13 scripts are designed to be read in order — each one does one thing.

The concepts here scale directly to production systems: swap ChromaDB for Pinecone or Weaviate, swap Mistral for GPT-4o or Claude, and the architecture is essentially unchanged. The core insight stays the same.

*Embeddings are compressed meaning. Retrieval is meaning-based search. RAG is putting those two ideas together.*
