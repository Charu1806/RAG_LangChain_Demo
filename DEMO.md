# Demo Guide — RAG-Based Company Document Visualisation

A step-by-step script for presenting this project to stakeholders, interviewers,
or a technical audience. Estimated time: **20–30 minutes**.

---

## Before the Demo — Setup Checklist

Run these the night before or at least 10 minutes before the demo:

```bash
cd ~/RAG_LangChain_Demo
source venv/bin/activate
export MISTRAL_API_KEY="your-key-here"

# Pre-run the embedding and indexing steps so they don't run live
python3 scripts/step3_load_documents.py
python3 scripts/step4_chunk_documents.py
python3 scripts/step5_generate_embeddings.py
python3 scripts/step6_store_in_chroma.py
```

Keep two windows open:
- **Terminal** — for running scripts
- **Browser** — for showing Plotly charts (they open automatically)

---

## Part 1 — The Problem (2 min)

**Say:**
> "Imagine a company with thousands of documents — HR policies, engineering guides,
> finance procedures. An employee asks: *'How many weeks of maternity leave do I get?'*
> They either search through PDFs manually, or ask a chatbot that might just make up
> an answer. RAG solves this — it finds the right document first, then generates
> a grounded answer from it."

**Show:** The `hr_policies.txt` file briefly — point out it has 15 real-looking documents.

---

## Part 2 — The Knowledge Base (3 min)

**Say:**
> "We have 90 synthetic company documents across 6 categories — HR, Finance,
> Engineering, Customer Support, Product Management, and Employee Directory.
> Each document is 300–500 words of professional corporate content."

**Show:**
```bash
python3 scripts/step3_load_documents.py
python3 scripts/step4_chunk_documents.py
```

**Point out the output:**
```
Loaded 6 category files
  [    employee]  employee_directory.txt  (9,842 words)
  [          hr]  hr_policies.txt         (8,731 words)
  ...
Total chunks: 90
```

> "Each document becomes one chunk — one unit of meaning that can be retrieved
> independently."

---

## Part 3 — Embeddings (3 min)

**Say:**
> "Now we convert each document into a list of 384 numbers — a vector.
> Documents that mean similar things get similar vectors.
> The model runs entirely locally — no API call needed for this step."

**Show:**
```bash
python3 scripts/step5_generate_embeddings.py
```

**Point out:**
```
embeddings.shape : (90, 384)
  └─ 90 chunks  ×  384 dimensions
```

> "Each of these 90 documents is now a point in 384-dimensional space.
> We can't visualise 384 dimensions — but we can compress them down to 2 or 3."

---

## Part 4 — The Cluster Plot (5 min) ⭐ Main Visual

**Say:**
> "This is where it gets visual. We use UMAP to compress 384 dimensions into 2.
> Watch what happens."

**Run:**
```bash
python3 scripts/step8_umap_visualisation.py
```

**When the 2D chart opens:**
> "Look — six distinct colour islands, one per category. The HR documents
> cluster together. The Engineering documents cluster together. They've never
> been told they belong to the same category — the embedding model figured it
> out purely from the meaning of the words."

**Hover over a point:**
> "Hover over any dot to see which document it is."

**When the labelled 2D chart opens:**
> "The larger the dot, the longer the document. Notice HR and Engineering
> documents tend to be longer — more detailed policies and technical guides."

**When the 3D chart opens:**
> "In 3D you can see even more separation. Click and drag to rotate."
> *(Rotate the chart slowly)*
> "Clusters that seem to touch in 2D are actually separated in the third dimension."

---

## Part 5 — The Advanced Visualisations (5 min)

**Run:**
```bash
python3 scripts/step12_advanced_visualisation.py
```

### Chart A — Full Text Hover
> "Hover over any dot. You can read the entire document without leaving the chart.
> This is how you'd explore what's in your vector database."

*(Hover over an Engineering dot, then an HR dot — show the contrast)*

### Chart B — Category Filter
> "For a live demo, this is powerful. Use the dropdown to show just one category."

*(Select "Engineering" from the dropdown)*
> "Now I'm looking at only the 15 engineering documents.
> Notice how tight this cluster is — they all talk about Kubernetes,
> deployments, APIs. The model has grouped them because they share meaning."

*(Select "HR")*
> "HR documents — leave policies, code of conduct, performance reviews.
> A completely different region of vector space."

### Chart C — Similarity Spotlight ⭐ The 'Aha' Moment
> "This is the one that makes people stop and think.
> I've asked: *'How many days of maternity leave are employees entitled to?'*
> The gold star is where that question lands in vector space.
> The dotted lines show the five documents the database retrieved as most similar."

*(Point to the lines)*
> "Every line connects the question to a document it retrieved.
> They're all in the HR cluster — exactly where you'd expect.
> This is retrieval made *spatial*. You can see the database making its decision."

---

## Part 6 — RAG in Action (5 min) ⭐ The Payoff

**Say:**
> "Now we wire the retrieval to an LLM. ChromaDB finds the relevant documents,
> injects them into a prompt, and Mistral reads only those documents to answer.
> It cannot make things up — it can only use what was retrieved."

**Run:**
```bash
python3 scripts/step13_rag_query.py
```

**While it runs, explain:**
> "For each question, you'll see the answer and the source documents it came from.
> Every claim is traceable."

**Highlight these answers as they appear:**

**Maternity leave question:**
> "26 weeks — 16 at full pay, then 60%. Exact policy detail. Source: HR-003.
> We can open HR-003 and verify every word."

**Kubernetes question:**
> "Liveness and readiness probes — straight from the engineering runbook.
> An engineer who joined yesterday could ask this and get an accurate answer."

**Bonus calculation question:**
> "Target Bonus × Company Multiplier × Individual Multiplier.
> The formula, the weights, the rating levels — all pulled from FIN-005."

**Say:**
> "Notice what's happening: the LLM is not answering from its training data.
> It's reading four specific AcmeTech documents and summarising them.
> If the policy changes, you update the document, re-index, and the answer
> changes automatically. No retraining required."

---

## Part 7 — Architecture Summary (2 min)

**Draw or show this flow:**

```
User Question
      │
      ▼
  Embed query          ← same model as the documents
      │
      ▼
  ChromaDB             ← cosine similarity search
  (vector search)
      │
  Top 4 chunks
      │
      ▼
  Mistral AI           ← reads ONLY the retrieved chunks
      │
      ▼
  Answer + Sources     ← grounded, auditable, cited
```

> "The key insight is that the embedding model is used twice:
> once to index the documents, once to embed the question.
> Because they use the same model, similar meanings land in the same region —
> and that's how retrieval works."

---

## Anticipated Questions

**"Could you use OpenAI instead of Mistral?"**
> "Yes — swap `ChatMistralAI` for `ChatOpenAI` in one line. The rest of the
> pipeline is identical. We chose Mistral because it's free-tier generous
> and performs excellently on factual retrieval tasks."

**"What happens if the answer isn't in the documents?"**
> "The prompt instructs Mistral to say *'I couldn't find this in the AcmeTech
> knowledge base'* rather than guessing. Try asking about something not covered —
> like the company's stock price — and it will decline to answer."

**"How does this scale to millions of documents?"**
> "ChromaDB scales to millions of vectors. For very large corpora you'd switch
> to a managed vector database like Pinecone or Weaviate, but the LangChain
> interface stays identical — one line change."

**"Why UMAP instead of t-SNE?"**
> "UMAP preserves global structure better and is significantly faster.
> t-SNE is great for tight local clusters but distorts the distances between
> clusters — which matters when you want to show that HR and Engineering
> are far apart in meaning, not just that they cluster internally."

**"How were the documents created?"**
> "A structured prompt generated them using an LLM — the prompt is in
> `master_prompt.txt`. The vocabulary was deliberately designed to be
> category-distinct so the clusters separate cleanly in UMAP."

---

## Closing Line

> "What you've seen today is the full RAG stack — from raw text files to
> a searchable knowledge base to a question-answering system with cited answers.
> Every component is open-source or free-tier. The same architecture powers
> enterprise document Q&A systems at scale — the difference is just the size
> of the vector database and the quality of the source documents."

---

## Quick Reference — Commands

```bash
# Full pipeline from scratch
source venv/bin/activate
export MISTRAL_API_KEY="your-key"
python3 scripts/step3_load_documents.py       # Load docs
python3 scripts/step4_chunk_documents.py      # Chunk into 90 pieces
python3 scripts/step5_generate_embeddings.py  # Embed (384-dim vectors)
python3 scripts/step6_store_in_chroma.py      # Index into ChromaDB
python3 scripts/step8_umap_visualisation.py   # 2D + 3D cluster plots
python3 scripts/step12_advanced_visualisation.py  # Advanced charts
python3 scripts/step13_rag_query.py           # RAG answers

# Just the visuals (if ChromaDB already built)
python3 scripts/step8_umap_visualisation.py
python3 scripts/step12_advanced_visualisation.py

# Just the RAG (if ChromaDB already built)
python3 scripts/step13_rag_query.py
```
