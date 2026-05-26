from pathlib import Path

import json

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_community.embeddings import (
    FastEmbedEmbeddings
)

from app.graph import graph

from app.state import CustomerState

from app.services.llm_service import (
    llm_service
)

# ---------------------------------
# Base directory
# ---------------------------------

BASE_DIR = (
    Path(__file__).resolve().parent
)

# ---------------------------------
# Dataset path
# ---------------------------------

DATASET_PATH = (
    BASE_DIR /
    "dataset_for_rag_evaluation.json"
)

# ---------------------------------
# Load evaluation dataset
# ---------------------------------

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    evaluation_data = json.load(f)

# ---------------------------------
# Evaluator LLM
# ---------------------------------

evaluator_llm = (
    llm_service._create_llm(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )
)

# ---------------------------------
# Evaluator embeddings
# ---------------------------------

evaluator_embeddings = (
    FastEmbedEmbeddings()
)

# ---------------------------------
# Build evaluation lists
# ---------------------------------

questions = []

answers = []

contexts = []

ground_truths = []

# ---------------------------------
# Run evaluation pipeline
# ---------------------------------

for idx, item in enumerate(
    evaluation_data,
    start=1
):

    print("\n" + "=" * 80)

    print(
        f"\nRunning Evaluation "
        f"{idx}/{len(evaluation_data)}"
    )

    question = item[
        "user_question"
    ]

    ground_truth = item[
        "ground_truth_answer"
    ]

    print("\nQUESTION:")
    print(question)

    # ---------------------------------
    # Create fresh state
    # ---------------------------------

    state = CustomerState(

        customer_id="CUST_001",

        session_id="rag_eval_session",

        query=question,

        conversation_history=[]
    )

    # ---------------------------------
    # Run graph
    # ---------------------------------

    result = graph.invoke(
        state.model_dump()
    )

    # ---------------------------------
    # Extract answer
    # ---------------------------------

    answer = result.get(
        "response",
        ""
    )

    print("\nANSWER:")
    print(answer)

    # ---------------------------------
    # Extract retrieved docs
    # ---------------------------------

    retrieved_docs = result.get(
        "retrieved_docs",
        []
    )

    retrieved_contexts = []

    print("\nRETRIEVED DOCUMENTS:")

    for doc_idx, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        source = doc.get(
            "source",
            "unknown"
        )

        page = doc.get(
            "page",
            "unknown"
        )

        category = doc.get(
            "category",
            "unknown"
        )

        content = doc.get(
            "content",
            ""
        )

        print("\n" + "-" * 50)

        print(
            f"\nChunk {doc_idx}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Category: {category}"
        )

        print("\nCONTENT PREVIEW:\n")

        print(content[:500])

        retrieved_contexts.append(
            content
        )

    # ---------------------------------
    # Append evaluation data
    # ---------------------------------

    questions.append(
        question
    )

    answers.append(
        answer
    )

    contexts.append(
        retrieved_contexts
    )

    ground_truths.append(
        ground_truth
    )

# ---------------------------------
# Create HuggingFace dataset
# ---------------------------------

dataset = Dataset.from_dict({

    "question": questions,

    "answer": answers,

    "contexts": contexts,

    "ground_truth": ground_truths
})

# ---------------------------------
# Run RAGAS evaluation
# ---------------------------------

print("\n" + "=" * 80)

print("\nRunning RAGAS Evaluation...\n")

result = evaluate(

    dataset=dataset,

    metrics=[

        faithfulness,

        answer_relevancy,

        context_precision,

        context_recall
    ],

    llm=evaluator_llm,

    embeddings=evaluator_embeddings
)

# ---------------------------------
# Final results
# ---------------------------------

print("\n" + "=" * 80)

print("\nRAGAS RESULTS:\n")

print(result)

print("\n" + "=" * 80)