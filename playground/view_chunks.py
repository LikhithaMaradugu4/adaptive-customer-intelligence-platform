from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from langchain_chroma import Chroma


# ---------------------------------
# Vector DB path
# ---------------------------------

CHROMA_PATH = "chroma_db"

# ---------------------------------
# Embedding model
# ---------------------------------

embedding_model = (
    FastEmbedEmbeddings()
)

# ---------------------------------
# Load vector DB
# ---------------------------------

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

# ---------------------------------
# Fetch stored chunks
# ---------------------------------

data = vector_store.get()

documents = data["documents"]

metadatas = data["metadatas"]

print(
    f"\nTotal Chunks: {len(documents)}"
)

# ---------------------------------
# Display chunks
# ---------------------------------

for idx, (doc, metadata) in enumerate(
    zip(documents, metadatas),
    start=1
):

    print("\n" + "=" * 80)

    print(f"\nCHUNK {idx}")

    print("\nMETADATA:")
    print(metadata)

    print("\nCONTENT:\n")

    print(doc[:1200])

    print("\n" + "=" * 80)

    input(
        "\nPress Enter for next chunk..."
    )