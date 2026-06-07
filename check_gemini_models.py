"""
check_gemini_models.py
Lists every Gemini model your API key can access for generateContent.
Run this to find the correct model name to use in step13.

Usage:
    python3 check_gemini_models.py
"""
import os
import google.generativeai as genai

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY first:  export GOOGLE_API_KEY='your-key'")

genai.configure(api_key=GOOGLE_API_KEY)

print("Models available for generateContent:\n")
usable = []
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")
        usable.append(m.name)

print(f"\nTotal: {len(usable)} models")
print("\nRecommended — paste one of these into step13_rag_query.py:")
for name in usable:
    short = name.replace("models/", "")
    if any(x in short for x in ["flash", "pro", "gemini"]):
        print(f"  model=\"{short}\"")
