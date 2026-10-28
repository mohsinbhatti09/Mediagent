# MediAgent upgrade notes

This bundle updates the source/configuration files only. It intentionally excludes
`.git`, `.env`, local ChromaDB data, uploaded documents, and Python environments.

## Main changes
- Added a cross-encoder reranking stage to the LangGraph pipeline.
- Increased initial vector retrieval to 12 candidates, reranks candidates, then lets
  the LLM relevance grader select evidence from at most 8 passages.
- If the grader returns invalid JSON, the app falls back to the best two ranked
  passages instead of treating every candidate as relevant.
- Keeps source IDs consistent between answer context and displayed evidence.
- Adds citation-ID validation so the model cannot cite an ID that was not supplied.
- Improves PDF text normalization and preserves page/source/chunk metadata.
- Fixes repeated indexing so stale/duplicate chunks are not accumulated.
- Removes caching from the vector-store loader so new sessions can detect an
  existing persistent index.
- Adds `.env.example`, `.gitignore`, `.dockerignore`, Dockerfile, Compose setup,
  and updated documentation.
- Removes unused AutoGen packages from the default requirements. AutoGen was
  listed in the previous README/dependencies but was not connected to the active
  question-answering workflow.

## Important
The first run downloads the embedding model and cross-encoder model. The app has
not been tested against your Groq account from this environment. After applying
the files, run the smoke checks in the chat instructions and test the three known
CareLock questions.
