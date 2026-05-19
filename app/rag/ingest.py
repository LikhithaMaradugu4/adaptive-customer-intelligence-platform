import os
import json

from langchain_core.documents import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from langchain_community.vectorstores import (
    Chroma
)


# ---------------------------------
# Embedding model
# ---------------------------------

embedding_model = FastEmbedEmbeddings()

# ---------------------------------
# Vector DB path
# ---------------------------------

CHROMA_PATH = "chroma_db"

# ---------------------------------
# Load markdown documents
# ---------------------------------

documents = []

KB_PATH = "data/knowledge_base"

for filename in os.listdir(KB_PATH):

    if filename.endswith(".md"):

        file_path = os.path.join(
            KB_PATH,
            filename
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": filename
                }
            )
        )

# ---------------------------------
# Load FAQs
# ---------------------------------

FAQ_PATH = "data/faqs/faqs.json"

with open(
    FAQ_PATH,
    "r",
    encoding="utf-8"
) as f:

    faq_data = json.load(f)

for item in faq_data:

    faq_text = (
        f"Question: {item['question']}\n"
        f"Answer: {item['answer']}"
    )

    documents.append(
        Document(
            page_content=faq_text,
            metadata={
                "source": "faq"
            }
        )
    )

# ---------------------------------
# Chunk documents
# ---------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    documents
)

print(f"\nTotal chunks created: {len(chunks)}")

# ---------------------------------
# Create Chroma vector DB
# ---------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

print("\nChromaDB created successfully.")