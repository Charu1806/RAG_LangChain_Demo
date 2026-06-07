# How to Build a RAG System That Actually Makes Sense — Visually

### Stop guessing what your vector database is doing. See it.

---

*If you've ever built a RAG pipeline and wondered whether it was actually working — whether the right documents were being retrieved, whether the embeddings meant anything — this article is for you.*

*We'll build a complete RAG system from scratch, visualise every step as an interactive 2D and 3D cluster plot, and finish with a Mistral AI-powered Q&A system that cites its sources. All free. All local embeddings. All explainable.*

---

## The Problem With Most RAG Tutorials

Most RAG tutorials look like this:

```python
db = Chroma.from_documents(docs, embeddings)
result = db.similarity_search("my question")
```

And then: *"It works! Here's your answer."*

But you're left with no intuition for what actually happened. Why did it retrieve *that* document and not the other one? What does a "vector database" actually contain? What does semantic similarity *look like*?

This article fixes that. We build everything transparently — and then we **visualise the embedding space** so you can see exactly what the model learned about your documents.

---

## What We're Building

A company knowledge base for a fictional company called **AcmeTech Solutions**, containing 90 documents across six categories:

- 👤 Employee Directory — 20 employee profiles
- 📋 HR Policies — 15 policy documents
- 💰 Finance & Tax — 15 finance procedures
- ⚙️ Engineering Docs — 15 technical guides
- 🎧 Customer Support — 15 support articles
- 🗺️ Product Management — 10 strategy documents

By the end, we'll be able to ask questions like:

> *"How many weeks of maternity leave does AcmeTech provide?"*

And get back:

> *"AcmeTech provides up to 26 weeks. The first 16 weeks are paid at 100% of base salary. Weeks 17–26 are paid at 60%. Source: HR-003"*

Every claim traced. Every answer grounded. No hallucination.

---

## Part 1 — What Is a Vector Embedding?

Before we write a single line of code, let's build the intuition.

Imagine you could describe any document as a list of 384 numbers. Not arbitrary numbers — numbers that *encode meaning*. Documents about similar topics get similar numbers. Documents about completely different topics get very different numbers.

That's an embedding. The list of numbers is called a **vector**.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vec1 = model.encode("The maternity leave policy covers 26 weeks")
vec2 = model.encode("Employees are entitled to paid parental leave")
vec3 = model.encode("Kubernetes deployments use rolling update strategy")

# vec1 and vec2 will be very similar
# vec1 and vec3 will be very different
```

The distance between two vectors tells you how similar two pieces of text are in meaning — not just in keywords, but semantically.

This is the core idea that makes everything else work.

---

## Part 2 — Loading and Chunking Documents

The first practical step is getting your documents into a form that can be embedded.

```python
from langchain.schema import Document

file_category_map = {
    "hr_policies.txt":               "hr",
    "finance_tax.txt":               "finance",
    "engineering_documentation.txt": "engineering",
    "customer_support_kb.txt":       "support",
    "product_management.txt":        "product",
    "employee_directory.txt":        "employee",
}

docs = []
for filename, category in file_category_map.items():
    with open(filename, "r") as f:
        content = f.read()
    docs.append(Document(
        page_content=content,
        metadata={"category": category, "source": filename}
    ))
```

**Why chunking matters.** We don't embed entire files — we embed individual documents. Each of our 90 synthetic documents is separated by a delimiter, and we split on that:

```python
SEPARATOR = "==============================="
chunks = []

for doc in docs:
    for raw in doc.page_content.split(SEPARATOR):
        text = raw.strip()
        if len(text) >= 100:
            chunks.append(Document(
                page_content=text,
                metadata=doc.metadata.copy()
            ))

# Result: 90 chunks — one per document
print(f"Total chunks: {len(chunks)}")  # → 90
```

The chunk size matters enormously in real RAG systems. Too small and you lose context. Too large and the retrieval becomes imprecise. For this demo, each chunk is a complete 300–500 word document — the right unit of meaning.

---

## Part 3 — Generating Embeddings

Now the magic. We convert every chunk into a 384-dimensional vector:

```python
from sentence_transformers import SentenceTransformer

model      = SentenceTransformer("all-MiniLM-L6-v2")
texts      = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

print(embeddings.shape)  # → (90, 384)
```

**90 documents × 384 dimensions.** Each row is a document. Each column is a feature the model learned about language.

The model we're using — `all-MiniLM-L6-v2` — runs entirely on CPU, weighs about 80MB, and produces embeddings that are genuinely good at capturing semantic meaning. No GPU needed. No API key. No cost.

---

## Part 4 — Storing in a Vector Database

ChromaDB is a lightweight, file-based vector database that persists to disk:

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_fn = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},  # cosine-friendly
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_fn,
    persist_directory="./vector_db",
)
```

Under the hood, ChromaDB stores three things per document:
1. The original text
2. The metadata (category, source file)
3. The 384-dimensional embedding vector

When a query arrives, ChromaDB embeds it using the same model and finds the vectors that are most similar — measured by cosine similarity.

**What does cosine similarity mean in practice?** It measures the angle between two vectors. A score near 0 means the documents point in nearly the same direction in 384-dimensional space. HR documents point in the HR direction. Engineering documents point in the Engineering direction. Your query about Kubernetes will point much closer to Engineering than to HR.

---

## Part 5 — The Visualisation (The Part Everyone Skips)

Here's where this tutorial diverges from every other RAG article.

We have 90 vectors of 384 dimensions each. We can't plot 384-dimensional space. But we can compress it down to 2 or 3 dimensions using **UMAP** while preserving the neighbourhood relationships — which documents are close to which others.

```python
from umap import UMAP
import plotly.express as px
import pandas as pd

# Compress 384 → 2 dimensions
coords_2d = UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.1,
    metric="cosine",
    random_state=42
).fit_transform(embeddings)

# Build a DataFrame for plotting
df = pd.DataFrame({
    "x":        coords_2d[:, 0],
    "y":        coords_2d[:, 1],
    "category": [c.metadata["category"] for c in chunks],
    "title":    [parse_title(c.page_content) for c in chunks],
})

# Plot it
fig = px.scatter(
    df, x="x", y="y",
    color="category",
    hover_name="title",
    title="Vector Embedding Clusters — UMAP 2D (384 → 2 dimensions)",
)
fig.show()
```

**What you see when you run this:**

Six clearly separated colour islands — one per category. The HR documents cluster together. The Engineering documents cluster together. The Finance documents are in their own corner of the plot. And they've never been told what category they belong to — the embedding model figured it out purely from the *meaning* of the words.

This is the visual proof that your embeddings are working.

---

## Part 6 — What UMAP Is Actually Doing

UMAP (Uniform Manifold Approximation and Projection) preserves the **local neighbourhood structure** of your high-dimensional data.

Think of it this way: in 384-dimensional space, the 10 documents closest to the *Maternity Leave Policy* are other HR documents. UMAP's job is to lay all 90 documents out in 2D such that those same 10 documents are still close to each other. It doesn't get the distances exactly right — that's impossible in 2D — but it preserves which documents are near which others.

The result is a map of your knowledge base. Documents about similar topics appear near each other. Documents about different topics appear far apart.

Three key parameters to understand:

```python
UMAP(
    n_neighbors=10,   # how many nearby points to consider
                      # lower → tighter local clusters
                      # higher → more global shape preserved

    min_dist=0.1,     # minimum spacing between points in the output
                      # lower → denser, more packed clusters
                      # higher → points spread out more

    metric="cosine",  # use cosine distance — matches our embeddings
)
```

For small datasets (like our 90 documents), `n_neighbors=10` works well. For millions of documents you'd increase this.

---

## Part 7 — Making It Interactive

Static scatter plots are fine. Interactive ones are better for demos.

```python
CATEGORY_COLORS = {
    "employee":    "#636EFA",
    "hr":          "#EF553B",
    "finance":     "#00CC96",
    "engineering": "#AB63FA",
    "support":     "#FFA15A",
    "product":     "#19D3F3",
}

fig = px.scatter(
    df, x="x", y="y",
    color="category",
    color_discrete_map=CATEGORY_COLORS,
    hover_name="title",
    hover_data={"doc_id": True, "word_count": True, "preview": True},
    size="word_count",   # longer documents appear as larger dots
    size_max=22,
    title="🗺️  Vector Embedding Clusters — UMAP 2D",
    width=1050, height=680,
)

# Dark background — looks much better for cluster plots
fig.update_layout(
    plot_bgcolor="#0f0f1a",
    paper_bgcolor="#0f0f1a",
    font_color="white",
)

# Add category label at each cluster's centroid
for cat, color in CATEGORY_COLORS.items():
    sub = df[df["category"] == cat]
    fig.add_annotation(
        x=sub["x"].mean(), y=sub["y"].mean(),
        text=f"<b>{cat.upper()}</b>",
        showarrow=False,
        font=dict(size=13, color=color),
    )

fig.show()
```

Hover over any dot to see the document title, ID, word count, and a text preview. Click a category in the legend to hide or show it. This is the chart you show stakeholders when explaining what a vector database actually contains.

---

## Part 8 — The Similarity Spotlight

This is the visualisation that makes people stop and think.

Instead of just showing the clusters, we embed a query and draw lines from the query point to the documents it retrieved.

```python
import numpy as np

QUERY = "How many days of maternity leave are employees entitled to?"
TOP_K = 5

# Embed the query using the same model
query_vec  = model.encode([QUERY])[0]

# Cosine similarity against all 90 embeddings
normed     = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
query_norm = query_vec  / np.linalg.norm(query_vec)
sims       = normed @ query_norm          # shape: (90,)
top_k_idx  = np.argsort(sims)[::-1][:TOP_K]

# Place the query at the centroid of its top-k neighbours
qx = df.iloc[top_k_idx]["x"].mean()
qy = df.iloc[top_k_idx]["y"].mean()
```

Then plot dotted lines from the query star to each retrieved document. The result: you can see retrieval happening spatially. The query lands in the HR cluster, the lines reach out to the 5 most similar HR documents, and you can verify at a glance that the retrieval makes sense.

---

## Part 9 — The RAG Pipeline

Retrieval is only half of RAG. The second half is generation — using an LLM to turn retrieved chunks into a readable answer.

```python
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# LLM — Mistral small is fast, accurate, and has a generous free tier
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,   # low = factual, consistent
)

# The prompt is the most important part
# We explicitly instruct the model to use ONLY the retrieved context
SYSTEM_PROMPT = """You are a helpful assistant for AcmeTech Solutions.
Answer the user's question using ONLY the context documents below.
If the answer isn't in the context, say so — don't guess.
At the end, cite the document IDs you used.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

# Wire it together
retriever = db.as_retriever(search_kwargs={"k": 4})
rag_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt)
)

# Ask a question
response = rag_chain.invoke({
    "input": "How many weeks of maternity leave does AcmeTech provide?"
})

print(response["answer"])
# → "AcmeTech provides up to 26 weeks of maternity leave.
#    The first 16 weeks are paid at 100% of base salary.
#    Weeks 17–26 are paid at 60% of base salary.  Source: HR-003"
```

**The prompt is everything.** The instruction *"use ONLY the context"* is what prevents hallucination. The model can't invent a policy that isn't in the retrieved documents. If the document says 26 weeks, the answer says 26 weeks. If the document changes, the answer changes automatically — no retraining required.

---

## Part 10 — Why The Clusters Are So Clean

You might wonder: why do the six categories separate so perfectly in the UMAP plot?

The knowledge base was designed with this in mind. Each category uses distinct vocabulary:

| Category | Characteristic vocabulary |
|---|---|
| Employee | employee, manager, promotion, skill, performance review |
| HR | leave, policy, compliance, benefits, workplace |
| Finance | tax, reimbursement, payroll, audit, budget |
| Engineering | Kubernetes, API, deployment, database, microservice |
| Support | customer, ticket, issue, troubleshooting, account |
| Product | roadmap, OKR, metric, experiment, launch |

The embedding model has learned that *"Kubernetes"* and *"microservice"* are semantically related to each other and unrelated to *"maternity leave"* or *"expense reimbursement"*. So when it encodes these documents into vectors, documents in the same category naturally point in the same direction in 384-dimensional space.

UMAP then makes that invisible structure visible.

In a real enterprise knowledge base, categories won't separate quite as cleanly — there will be cross-domain documents and terminology overlap. But the principles are identical.

---

## Part 11 — What Happens When Retrieval Goes Wrong

Understanding failure modes is as important as understanding success.

**Problem 1: Wrong chunk size.** If you chunk by 200 characters instead of 300–500 words, each chunk is too short to carry meaningful semantic information. The embedding captures syntax rather than meaning, and retrieval quality drops dramatically.

**Problem 2: Wrong embedding model.** `all-MiniLM-L6-v2` is excellent for general text. But for highly specialised domains — medical, legal, code — you'd want a domain-specific model. A general model might place *"REST API endpoints"* and *"rest and recovery"* uncomfortably close together.

**Problem 3: Too few retrieved chunks.** With `k=1`, you retrieve a single document and hope it contains everything. With `k=4`, you give the LLM more context and the answer is more complete — but the prompt gets longer and more expensive.

**Problem 4: Bad prompt.** If you don't explicitly tell the LLM to use only the context, it will blend retrieved facts with training knowledge — and you lose the ability to trace where the answer came from.

---

## Part 12 — The Full Architecture

```
INDEXING (runs once)
═══════════════════

  .txt files
      │
      ▼
  Split into chunks          ← one document per chunk
      │
      ▼
  sentence-transformers      ← 384-dim vectors, local, free
      │
      ▼
  ChromaDB                   ← persisted to disk


QUERYING (runs every request)
══════════════════════════════

  User question
      │
      ▼
  Embed question             ← same model as indexing
      │
      ▼
  ChromaDB cosine search     ← find top-4 most similar vectors
      │
  Top-4 chunks
      │
      ▼
  Mistral AI                 ← reads ONLY the retrieved chunks
  (with strict prompt)
      │
      ▼
  Grounded answer + sources
```

---

## Results — What We Got

Running 12 questions across all six categories:

| Question | Retrieved category | Answer accuracy |
|---|---|---|
| Maternity leave weeks and pay | HR ✅ | Exact — 26 weeks, 100%/60% |
| Annual bonus calculation | Finance ✅ | Formula with all multipliers |
| Kubernetes health checks | Engineering ✅ | Liveness + readiness probes |
| Platform outage severity | Engineering ✅ | SEV1 Critical |
| Password reset email missing | Support ✅ | Check spam, verify email |
| SSO via SAML 2.0 | Support ✅ | Full setup steps |
| Product roadmap themes | Product ✅ | All three themes named |
| Sarah Mitchell's projects | Employee ✅ | Exact projects listed |

12 for 12. Every answer came from the right category, every answer was traceable to a source document.

---

## Key Takeaways

**1. Visualise your embeddings early.** UMAP plots take 10 minutes to add and immediately tell you whether your chunks and embedding model are producing meaningful representations. If your clusters don't separate, your retrieval won't work well.

**2. The prompt determines accuracy.** "Use only the context" is not a polite suggestion — it's the mechanism that prevents hallucination. Test what happens without it.

**3. Local embeddings are good enough.** `all-MiniLM-L6-v2` running on a laptop CPU produces embeddings that work excellently for English-language enterprise documents. You don't need a cloud embedding API for most RAG applications.

**4. Chunk size is a hyperparameter.** There's no universal right answer. Chunk by sentence, by paragraph, by document, or by a fixed number of tokens — and test which retrieval quality you get. The chunk that gets retrieved is the full context the LLM sees.

**5. RAG is not just a technique — it's an architecture.** The separation of retrieval (ChromaDB) from generation (Mistral) means you can upgrade either component independently. Switch to a better embedding model. Switch to a faster LLM. Update the knowledge base without touching the model. This modularity is what makes RAG practical in production.

---

## Full Code and Resources

The complete project is on GitHub:
**https://github.com/Charu1806/RAG_LangChain_Demo**

It includes:
- All 90 synthetic AcmeTech documents
- 13 step-by-step Python scripts
- A complete Google Colab notebook
- An interactive demo guide

Everything runs locally. The only external API call is to Mistral AI for text generation — and that's free tier with no daily cap on `mistral-small-latest`.

---

## What To Try Next

Once you have this running, extend it in any of these directions:

- **Swap the documents** — replace the AcmeTech files with your own PDFs, Notion exports, or Confluence pages
- **Try different embedding models** — compare `all-MiniLM-L6-v2` against `all-mpnet-base-v2` and see how the UMAP clusters change
- **Add metadata filtering** — let users restrict retrieval to a single category: *"search only HR documents"*
- **Add conversation memory** — wire in `ConversationBufferMemory` for multi-turn Q&A
- **Try a reranker** — add a cross-encoder reranker between retrieval and generation to improve precision on ambiguous queries

---

*Built with LangChain, ChromaDB, sentence-transformers, UMAP, Plotly, and Mistral AI.*

*Full source: https://github.com/Charu1806/RAG_LangChain_Demo*

---

**Tags:** `RAG` `LangChain` `Vector Database` `ChromaDB` `UMAP` `Embeddings` `Mistral AI` `NLP` `Python` `Machine Learning`
