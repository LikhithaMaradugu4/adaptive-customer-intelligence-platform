import os

from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.document_loaders import (
    PyMuPDFLoader
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
# Knowledge base folder
# ---------------------------------

KB_PATH = "data/knowledge_base"

# ---------------------------------
# Chunking strategy
# ---------------------------------

splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
)

all_chunks = []

# ---------------------------------
# Load PDFs
# ---------------------------------

pdf_files = list(
    Path(KB_PATH).glob("*.pdf")
)

if not pdf_files:

    print(
        "\nNo PDF files found."
    )

    exit()

for pdf_path in pdf_files:

    print(
        f"\nProcessing: {pdf_path.name}"
    )

    try:

        # ---------------------------------
        # Load PDF
        # ---------------------------------

        loader = PyMuPDFLoader(
            str(pdf_path)
        )

        documents = loader.load()

        print(
            f"Pages loaded: {len(documents)}"
        )

        # ---------------------------------
        # Split into chunks
        # ---------------------------------

        chunks = splitter.split_documents(
            documents
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

        # ---------------------------------
        # Metadata enrichment
        # ---------------------------------

        for index, chunk in enumerate(chunks):

            chunk.metadata[
                "source_file"
            ] = pdf_path.name

            chunk.metadata[
                "category"
            ] = pdf_path.stem

            chunk.metadata[
                "document_type"
            ] = "policy"

            chunk.metadata[
                "chunk_id"
            ] = index

        all_chunks.extend(chunks)

    except Exception as error:

        print(
            f"Error processing "
            f"{pdf_path.name}: {error}"
        )

# ---------------------------------
# Final stats
# ---------------------------------

print(
    f"\nTotal chunks created: "
    f"{len(all_chunks)}"
)

# ---------------------------------
# Create ChromaDB
# ---------------------------------

if all_chunks:

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print(
        "\nChromaDB created successfully."
    )

else:

    print(
        "\nNo chunks available."
    )