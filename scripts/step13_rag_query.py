"""
Step 13: RAG Query with Google Gemini
=======================================
Wires ChromaDB retrieval to Google Gemini to answer natural-language questions
about the AcmeTech knowledge base.

How RAG works here:
  1. User asks a question in plain English
  2. ChromaDB finds the top-k most semantically similar document chunks
  3. Those chunks are injected into a prompt as "context"
  4. Gemini reads the context and generates a grounded, cited answer
  5. Source documents are printed so you can verify every claim

Setup:
  Get a free Gemini API key at https://aistudio.google.com/app/apikey
  Then set it one of two ways:

  Option A — environment variable (recommended for local use):
      export GOOGLE_API_KEY="your-key-here"

  Option B — hard-code below (never commit to git):
      GOOGLE_API_KEY = "your-key-here"

  Option C — Colab secrets (recommended for Colab):
      Left panel → 🔑 Secrets → add GOOGLE_API_KEY

Requires:
    ./vector_db/   (produced by step6_store_in_chroma.py)

Run:
    python step13_rag_query.py
"""

import os
import time
import textwrap
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
KB_DIR       = PROJECT_ROOT / "knowledge_base"
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ── API Key ────────────────────────────────────────────────────────────────────
# Get a free key at: https://console.mistral.ai → API Keys → Create new key
# Then run: export MISTRAL_API_KEY="your-key-here"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "\n\n  ❌ MISTRAL_API_KEY not set.\n"
        "  Get a free key at: https://console.mistral.ai\n"
        "  Then run:  export MISTRAL_API_KEY='your-key-here'\n"
    )

print(f"✅ Mistral API key found  (ends with ...{MISTRAL_API_KEY[-4:]})\n")

# ── Embedding function ─────────────────────────────────────────────────────────
embedding_fn = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Load or build ChromaDB ─────────────────────────────────────────────────────
PERSIST_DIR = str(PROJECT_ROOT / "vector_db")

def clean_and_build_vector_db():
    """
    Wipe any existing vector_db completely, then rebuild from knowledge base files.
    Always wipes first to avoid read-only / corrupted SQLite errors.
    """
    import shutil
    from langchain.schema import Document

    # ── 1. Full wipe — removes partial / corrupted / empty state ──────────────
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    os.makedirs(PERSIST_DIR, mode=0o755)   # fresh dir with write permissions

    # ── 2. Chunk the knowledge base files ─────────────────────────────────────
    SEPARATOR = "==============================="
    file_category_map = {
        "employee_directory.txt":        "employee",
        "hr_policies.txt":               "hr",
        "finance_tax.txt":               "finance",
        "engineering_documentation.txt": "engineering",
        "customer_support_kb.txt":       "support",
        "product_management.txt":        "product",
    }
    chunks = []
    for filename, category in file_category_map.items():
        filepath = KB_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"\n  Missing file: {filepath}\n"
                "  Run from project root:\n"
                "    python3 scripts/step13_rag_query.py"
            )
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        for raw in content.split(SEPARATOR):
            text = raw.strip()
            if len(text) >= 100:
                chunks.append(Document(
                    page_content=text,
                    metadata={"category": category, "source": filename}
                ))

    print(f"   Chunked {len(chunks)} documents — embedding now (~30 sec) ...")

    # ── 3. Embed in batches to avoid memory pressure on Python 3.9 ────────────
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR,
    )
    return db


# ── Decide whether to load or build ───────────────────────────────────────────
needs_build = (
    not os.path.exists(PERSIST_DIR)
    or not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3"))
)

if needs_build:
    print("⚠️  vector_db missing or incomplete — building now ...\n")
    db = clean_and_build_vector_db()
    print(f"   ✅ Built — {db._collection.count()} documents indexed\n")
else:
    db    = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_fn)
    count = db._collection.count()
    if count == 0:
        print("⚠️  vector_db existed but was empty — rebuilding ...\n")
        db = clean_and_build_vector_db()
        print(f"   ✅ Rebuilt — {db._collection.count()} documents indexed\n")
    else:
        print(f"✅ ChromaDB loaded — {count} documents\n")

# ── Mistral LLM ────────────────────────────────────────────────────────────────
# mistral-small-latest: fast, accurate, generous free tier — ideal for RAG demos
# Other options: mistral-medium-latest, open-mistral-7b, open-mixtral-8x7b
MODEL = "mistral-small-latest"
print(f"Using model: {MODEL}\n")

llm = ChatMistralAI(
    model=MODEL,
    api_key=MISTRAL_API_KEY,
    temperature=0.2,
    max_tokens=1024,
)

# ── RAG Prompt ─────────────────────────────────────────────────────────────────
# The prompt instructs Gemini to answer ONLY from the retrieved context
# and to clearly say when it doesn't know — preventing hallucination.
SYSTEM_PROMPT = """You are a helpful assistant for AcmeTech Solutions employees.
Answer the user's question using ONLY the context documents provided below.

Rules:
- Be concise and specific — answer the question directly.
- If the answer is a number, date, or policy detail, state it clearly.
- If the context does not contain enough information, say:
  "I couldn't find a specific answer in the AcmeTech knowledge base."
- Never make up information not present in the context.
- At the end of your answer, list the document IDs you used as sources.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

# ── RAG Chain ──────────────────────────────────────────────────────────────────
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},            # retrieve top 4 most relevant chunks
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain          = create_retrieval_chain(retriever, combine_docs_chain)

# ── Rate limit config ──────────────────────────────────────────────────────────
# Mistral free tier: 1 req/sec — a 1.5s pause is enough
RATE_LIMIT_PAUSE = 1.5  # seconds
_last_call_time  = 0.0

# ── Helper: pretty-print a RAG response ───────────────────────────────────────
def ask(question: str) -> None:
    global _last_call_time

    # Respect free-tier rate limit — pause if needed
    elapsed = time.time() - _last_call_time
    if elapsed < RATE_LIMIT_PAUSE and _last_call_time > 0:
        wait = RATE_LIMIT_PAUSE - elapsed
        print(f"  ⏳  Rate limit pause: {wait:.1f}s ...\n")
        time.sleep(wait)

    print("=" * 70)
    print(f"  ❓  {question}")
    print("=" * 70)

    _last_call_time = time.time()
    response = rag_chain.invoke({"input": question})

    answer   = response["answer"]
    sources  = response["context"]      # list of retrieved Document objects

    # Wrap answer for readable terminal output
    wrapped = textwrap.fill(answer, width=68, subsequent_indent="  ")
    print(f"\n  💬  Answer:\n")
    for line in wrapped.splitlines():
        print(f"  {line}")

    # Source attribution
    print(f"\n  📄  Retrieved from ({len(sources)} chunks):\n")
    seen = set()
    for doc in sources:
        cat    = doc.metadata.get("category", "unknown")
        source = doc.metadata.get("source",   "unknown")
        key    = (cat, source)
        if key not in seen:
            seen.add(key)
            preview = doc.page_content.strip()[:90].replace("\n", " ")
            print(f"    [{cat:>12}]  {source}")
            print(f"               \"{preview}...\"")

    print()

# ── Example queries — one per category ────────────────────────────────────────
print("Running RAG queries across all 6 categories...\n")

queries = [

    # HR
    "How many weeks of maternity leave does AcmeTech provide, and what is the pay structure?",
    "What are the rules for carrying over unused annual leave?",

    # Finance
    "How is the annual performance bonus calculated for a mid-level employee?",
    "What is the procurement approval limit for a Director-level employee?",

    # Engineering
    "What health checks must every Kubernetes service implement before going to production?",
    "What is the incident severity classification for a complete platform outage?",

    # Support
    "What should a customer do if they do not receive a password reset email?",
    "How do I configure Single Sign-On using SAML 2.0 in AcmeTech?",

    # Product
    "What are AcmeTech's three strategic product themes for FY2026?",
    "How does AcmeTech calculate an OKR score and what does 0.7 mean?",

    # Employee
    "What are Sarah Mitchell's current projects and performance rating?",
    "Who is the Engineering Director and how many engineers do they manage?",
]

for q in queries:
    ask(q)

print("✅ All RAG queries complete.")
