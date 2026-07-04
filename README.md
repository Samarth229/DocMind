# DocMind

A RAG (Retrieval-Augmented Generation) assistant that ingests PDFs and text documents and answers questions over them with cited sources.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed

# Run the app
streamlit run app.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with your chosen model pulled (`ollama pull llama3`)
