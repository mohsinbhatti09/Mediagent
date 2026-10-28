# Local data

MediAgent creates `data/chroma/` at runtime for the persistent ChromaDB index.
Do not commit the index or uploaded documents to Git. Re-index PDFs through the UI
if the local index is missing or has been intentionally removed.
