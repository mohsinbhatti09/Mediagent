# MediAgent AI

MediAgent is a document-grounded PDF research assistant built with Streamlit, LangChain, LangGraph, ChromaDB, Hugging Face embeddings, a cross-encoder reranker, and Groq.

## Features

- Upload text-based PDF documents and index them into a persistent Chroma collection.
- Preserve source filename and page number in every chunk.
- Retrieve a broad candidate set, rerank query/passage pairs, and use an LLM relevance grader before answer generation.
- Generate answers grounded in selected excerpts, with source IDs and page references.
- Display only the excerpts used for the answer.
- Replace the current collection when re-indexing, preventing stale chunks from accumulating.
- Cache embedding and reranker models for reuse during the process lifetime.
- Handle empty indexes, scanned PDFs without extractable text, missing API keys, and model-loading failures.
- Run locally or with Docker Compose.

## Requirements

- Python 3.12 recommended.
- A Groq API key for answer generation.
- Internet access on first run to download the embedding and cross-encoder models.

The first model download can take a few minutes, especially on CPU. The reranker falls back to vector-search ranking if its model cannot load. The app still needs the Groq API key to answer questions.

## Local setup (Windows PowerShell)

Run commands from the project root — the folder containing `requirements.txt`:

```powershell
# Create a virtual environment if you do not already have one
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

Add your real Groq API key to `.env`:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Do not commit or share `.env`. Then run:

```powershell
streamlit run frontend/streamlit_app.py
```

Open the local URL shown in the terminal, normally `http://localhost:8501`.

## Docker setup

1. Create `.env` from `.env.example` and add `GROQ_API_KEY`.
2. From the project root, run:

```powershell
docker compose up --build
```

3. Open `http://localhost:8501`.

Stop with `Ctrl+C`, or run `docker compose down`. Chroma data is stored under `data/chroma`; the Hugging Face model cache is stored in a named Docker volume.

## Indexing behavior

Each time you click **Index Documents**, MediAgent replaces the active Chroma collection with the PDFs selected in that upload action. Upload all documents you want searchable together. The persistent database remains available across app restarts.

PDFs that consist only of scanned images need OCR; OCR is not built into this version. PDF text extraction quality depends on the source PDF layout.

## Retrieval workflow

```text
PDF upload
   -> page-aware text extraction
   -> chunking with overlap
   -> Hugging Face embeddings
   -> persistent Chroma similarity retrieval
   -> cross-encoder reranking
   -> LLM relevance grading
   -> grounded answer with source/page references
```

The reranker and LLM grader improve retrieval quality but do not guarantee perfect relevance or factual accuracy. Test against representative questions and inspect the source excerpts. Similarity and reranker scores are ranking signals, not calibrated probabilities.

## Safety and limitations

- Intended for educational and document-research use, not medical diagnosis or treatment.
- Do not upload identifiable patient data unless you have appropriate authorization and safeguards.
- The model may miss relevant passages or produce errors; verify citations against the original PDF.
- AutoGen is not part of the active request-answering graph in this version; the live workflow is orchestrated with LangGraph.
