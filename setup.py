"""Setup script for LegalVerifiRAG."""

from setuptools import setup, find_packages

setup(
    name="legalverifirag",
    version="0.1.0",
    description="Claim-level verification system for legal AI",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.2.2",
        "spacy>=3.7.0",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.0",
        "rank-bm25>=0.2.2",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "google-generativeai>=0.4.0",
        "eyecite>=2.6.0",
        "rapidfuzz>=3.0.0",
        "pymupdf>=1.23.0",
        "python-docx>=0.8.11",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.9.0",
        "tenacity>=8.0.0",
        "streamlit>=1.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "cloud": [
            "pinecone-client>=3.0.0",
        ],
    },
)
