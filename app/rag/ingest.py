import os

from pathlib import Path


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from langchain_chroma import Chroma


# ---------------------------------
# Embedding model
# ---------------------------------

embedding_model = (
    FastEmbedEmbeddings()
)

# ---------------------------------
# Vector DB path
# ---------------------------------

CHROMA_PATH = "chroma_db"

# ---------------------------------
# KB folder
# ---------------------------------

KB_PATH = "data/knowledge_base"

# ---------------------------------
# Chunking strategy
# ---------------------------------

splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
)

all_chunks = []

# ---------------------------------
# Load PDFs
# ---------------------------------

for pdf_path in Path(
    KB_PATH
).glob("*.pdf"):

    print(
        f"\nProcessing: {pdf_path.name}"
    )

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    chunks = splitter.split_documents(
        documents
    )

    # ---------------------------------
    # Metadata enrichment
    # ---------------------------------

    for chunk in chunks:

        chunk.metadata[
            "source_file"
        ] = pdf_path.name

        chunk.metadata[
            "category"
        ] = pdf_path.stem

        chunk.metadata[
            "document_type"
        ] = "policy"

    all_chunks.extend(chunks)

# ---------------------------------
# Final stats
# ---------------------------------

print(
    f"\nTotal chunks created: "
    f"{len(all_chunks)}"
)

# ---------------------------------
# Create vector DB
# ---------------------------------

vector_store = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

print(
    "\nChromaDB created successfully."
)