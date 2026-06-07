"""
Step 3: Load Documents
======================
Loads all 6 AcmeTech knowledge base text files into LangChain Document objects,
each tagged with a category and source filename in its metadata.

Run from project root:
    python scripts/step3_load_documents.py
"""

import pickle
from pathlib import Path
from langchain.schema import Document

# ── Resolve paths relative to project root ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
KB_DIR       = PROJECT_ROOT / "knowledge_base"
OUTPUT_DIR   = PROJECT_ROOT                        # pkl files saved at root

file_category_map = {
    "employee_directory.txt":        "employee",
    "hr_policies.txt":               "hr",
    "finance_tax.txt":               "finance",
    "engineering_documentation.txt": "engineering",
    "customer_support_kb.txt":       "support",
    "product_management.txt":        "product",
}

docs = []
for filename, category in file_category_map.items():
    filepath = KB_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    docs.append(
        Document(
            page_content=content,
            metadata={"category": category, "source": filename}
        )
    )

print(f"Loaded {len(docs)} category files\n")
for doc in docs:
    word_count = len(doc.page_content.split())
    print(f"  [{doc.metadata['category']:>12}]  {doc.metadata['source']}  ({word_count:,} words)")

# Save for next step
with open(OUTPUT_DIR / "docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("\nSaved: docs.pkl")
