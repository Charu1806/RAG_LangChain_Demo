#!/bin/bash
# install_requirements.sh
# Run this once inside your activated venv to install all project dependencies.
#
# Usage:
#   source venv/bin/activate          # activate your venv first
#   bash install_requirements.sh      # then run this script

echo "Installing RAG-Based-Company-Document-Visualisation dependencies..."
echo ""

pip install --upgrade pip

pip install \
    langchain \
    langchain-community \
    langchain-google-genai \
    langchain-core \
    faiss-cpu \
    sentence-transformers \
    chromadb \
    plotly \
    umap-learn \
    pandas \
    numpy \
    kaleido

echo ""
echo "✅ All packages installed."
echo ""
echo "Verify with:"
echo "   python3 -c \"from langchain_google_genai import ChatGoogleGenerativeAI; print('OK')\""
