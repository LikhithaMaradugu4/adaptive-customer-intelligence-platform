from app.rag.retriever import (
    retrieve_documents
)


query = "How long does refund processing take?"

results = retrieve_documents(query)

print("\nRetrieved Docs:\n")

for idx, doc in enumerate(results, start=1):

    print(f"\n--- Document {idx} ---")
    print("Source:", doc["source"])
    print(doc["content"])