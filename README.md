# LegalVerifiRAG

Claim-level verification system for legal AI. Detects hallucinations in AI-generated legal analysis by decomposing responses into atomic claims and verifying each against authoritative legal sources.

## Overview

Commercial legal AI tools hallucinate at 17-33% rates (Stanford Law, 2024). They cite documents but don't verify that claims about those documents are accurate. LegalVerifiRAG adds a verification layer that:

1. Retrieves relevant legal passages
2. Generates a response grounded in those passages
3. Decomposes the response into typed legal claims
4. Verifies each claim against the retrieved sources using NLI
5. Outputs the response with inline verification badges

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Ollama and pull model (for local deployment)
ollama pull llama3.1:8b
```

## Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

## Usage

```bash
# Build corpus
python scripts/download_corpus.py

# Build indices
python scripts/build_index.py

# Run the app
streamlit run src/app.py
```

## Project Structure

```
legalverifirag/
├── src/
│   ├── ingestion/      # Document downloading and processing
│   ├── indexing/       # Vector stores and search indices
│   ├── retrieval/      # Hybrid search
│   ├── generation/     # LLM backends and prompts
│   ├── verification/   # Claim verification algorithms
│   ├── pipeline.py     # End-to-end orchestration
│   └── app.py          # Streamlit UI
├── data/
│   ├── raw/            # Downloaded cases and statutes
│   ├── processed/      # Chunked documents
│   ├── index/          # FAISS and BM25 indices
│   └── eval/           # Test datasets
├── tests/              # Unit tests
└── scripts/            # Utility scripts
```

## License

MIT
