# Adaptive Customer Intelligence Platform

> AI-powered multi-agent customer support system with hybrid ML + LLM orchestration, RAG retrieval, customer profiling, and business-rule decisioning.

---

## Overview

Adaptive Customer Intelligence Platform is an AI-driven customer support system designed for e-commerce environments. The project simulates a production-style intelligent support assistant for a fictional platform called **ShopSphere**.

The system combines:
- Traditional machine learning
- LLM reasoning
- Retrieval-Augmented Generation (RAG)
- Business-rule orchestration
- Tool-calling agents

to generate grounded, context-aware customer support responses.

The platform analyzes customer intent and emotion, retrieves relevant policies and FAQs, accesses customer/order/product information, applies business rules, and generates intelligent responses through a multi-agent workflow powered by LangGraph.

---

## Features

- Hybrid intent detection using TF-IDF + Logistic Regression with LLM fallback
- Hybrid emotion analysis using VADER with LLM fallback
- Multi-agent orchestration using LangGraph
- Retrieval-Augmented Generation (RAG) using ChromaDB
- Business-rule decisioning and escalation handling
- Customer profiling using historical behavior
- Tool-calling support for orders, products, customers, and policies
- Session-aware conversations with persisted history
- Grounded response generation using retrieved knowledge
- Interactive Streamlit chat interface

---

## Why Hybrid AI?

The platform combines multiple AI paradigms to improve reliability and reduce hallucinations.

### Traditional ML
Used for:
- Fast intent classification
- Cost-efficient routing
- Deterministic predictions

### LLM Reasoning
Used for:
- Fallback classification
- Contextual understanding
- Natural response generation

### RAG (Retrieval-Augmented Generation)
Used for:
- Policy grounding
- FAQ retrieval
- Reducing hallucinated responses

### Rule-Based Decisioning
Used for:
- Escalation handling
- Risk detection
- Clarification workflows

This hybrid architecture creates a more realistic enterprise AI support pipeline compared to standalone chatbot systems.

---

## System Workflow

```text
User Query
    ↓
Intent Detection
    ↓
Emotion Detection
    ↓
Memory & Profile Retrieval
    ↓
Business Rule Decisioning
    ↓
RAG Knowledge Retrieval
    ↓
Response Generation
    ↓
Escalation (if required)
```

---

## Architecture

The application workflow is orchestrated using LangGraph inside `app/graph.py`.

### Agent Pipeline

1. **IntentAgent**
   - Detects customer intent using ML + LLM fallback

2. **EmotionAgent**
   - Identifies customer emotion using VADER sentiment analysis

3. **MemoryAgent**
   - Retrieves conversation history from SQLite

4. **ProfileAgent**
   - Builds customer profile using MongoDB data

5. **DecisionAgent**
   - Applies business rules
   - Decides tool usage
   - Handles clarification and escalation logic

6. **RAGAgent**
   - Retrieves grounded policy and FAQ information from ChromaDB

7. **EscalationAgent**
   - Creates escalation workflows for high-risk scenarios

8. **ResponseAgent**
   - Generates the final customer response using:
     - conversation context
     - retrieved knowledge
     - customer profile
     - tool outputs

---

## Tech Stack

### Core Technologies
- Python
- Streamlit
- LangGraph
- LangChain
- Groq LLM API

### Machine Learning
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- VADER Sentiment Analysis

### Retrieval & Vector Search
- ChromaDB
- FastEmbed

### Databases
- MongoDB
- SQLite

### Additional Tools
- Pydantic
- dotenv
- Pandas
- NumPy

---

## Project Structure

```bash
app/
├── agents/                 # Core AI agents
├── database/               # MongoDB connection setup
├── models/                 # Trained ML artifacts
├── rag/                    # RAG ingestion and retrieval
├── services/               # LLM, DB, and business services
├── tools/                  # Tool-calling functions
├── utils/                  # Utility/helper functions
├── graph.py                # LangGraph workflow
├── schemas.py              # Structured output schemas
├── state.py                # Shared workflow state
└── supervisor.py           # Validation and retry logic

data/
├── knowledge_base/         # Policy documents
├── faqs/                   # FAQ datasets
└── training/               # Training datasets

frontend/
└── streamlit_app.py        # Streamlit chat interface

playground/
└── test_graph.py           # Local workflow testing

training/
└── train_intent_model.py   # ML training script

chroma_db/                  # Persisted vector database

customer_support.db         # SQLite memory database

requirements.txt            # Python dependencies
```

---

## Current Interface

The platform currently operates through a Streamlit-based interactive UI that directly invokes the LangGraph workflow.

No dedicated HTTP backend layer has been implemented yet. The orchestration pipeline currently runs locally within the application runtime.

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd adaptive-customer-intelligence-platform
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
MONGO_URI=your_mongodb_connection_string
```

### Variable Descriptions

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq LLM access |
| `MONGO_URI` | MongoDB connection string |

---

## Running the Project

### Start the Streamlit Application

```bash
streamlit run frontend/streamlit_app.py
```

### Run Workflow Test Script

```bash
python -m playground.test_graph
```

### Train Intent Classification Model

```bash
python -m training.train_intent_model
```

### Build the Chroma Vector Store

```bash
python -m app.rag.ingest
```

---

## AI/LLM Workflow

### IntentAgent
Uses:
- TF-IDF vectorization
- Logistic Regression classifier
- Groq LLM fallback for uncertain predictions

### EmotionAgent
Uses:
- VADER sentiment analysis
- LLM fallback for contextual emotion understanding

### DecisionAgent
Handles:
- business rules
- escalation logic
- clarification handling
- tool selection

### RAGAgent
Retrieves:
- policies
- FAQs
- grounded support information

using ChromaDB and FastEmbed embeddings.

### ResponseAgent
Generates grounded responses using:
- retrieved documents
- conversation history
- customer profile
- tool outputs
- workflow state

---

## Database Design

### SQLite (`customer_support.db`)

Used for conversation memory.

#### Table: `conversations`

| Column | Description |
|---|---|
| customer_id | Customer identifier |
| query | Customer message |
| intent | Predicted intent |
| emotion | Predicted emotion |
| escalated | Escalation status |
| timestamp | Message timestamp |

---

### MongoDB Collections

| Collection | Purpose |
|---|---|
| customers | Customer profiles |
| orders | Order information |
| products | Product catalog |
| sessions | Chat sessions |
| messages | Conversation messages |

---

## Tool-Calling Capabilities

The DecisionAgent can invoke tools for:
- customer lookup
- order retrieval
- product retrieval
- policy retrieval
- session access

This enables dynamic response generation based on real-time contextual data.

---

## Sample Queries

- “Where is my order?”
- “I want to return my Samsung TV.”
- “My refund has not arrived yet.”
- “Can I replace a damaged product?”
- “What is your return policy?”
- “I am frustrated with the delivery delay.”

---

## Example Business Rules

- Refund-related complaints are marked as high-risk
- Angry or frustrated customers trigger escalation workflows
- Missing order details trigger clarification requests
- Policy-sensitive queries invoke RAG retrieval

---

## Example Usage

1. Launch the Streamlit application
2. Enter a customer ID (example: `CUST_001`)
3. Ask customer support questions
4. The workflow:
   - classifies intent
   - analyzes emotion
   - retrieves context
   - applies business rules
   - generates a grounded response

---

## Current Limitations

- No authentication or authorization layer
- No dedicated HTTP backend/API service
- Product retrieval logic is still limited
- Workflow currently runs locally
- No async execution support
- Limited observability and tracing
- Automated testing coverage is minimal

---

## Future Improvements

- Add FastAPI backend for API-based access
- Add Docker-based deployment support
- Implement authentication and RBAC
- Add streaming LLM responses
- Introduce async agent execution
- Add LangSmith/OpenTelemetry observability
- Improve product and order retrieval accuracy
- Add multi-tenant support
- Add production-grade caching and rate limiting

---

## Troubleshooting

### Missing Environment Variables
The application will fail if:
- `GROQ_API_KEY`
- `MONGO_URI`

are not configured.

---

### Empty Vector Store
If RAG retrieval is not working:

```bash
python -m app.rag.ingest
```

---

### MongoDB Connection Errors
Ensure:
- MongoDB is running
- credentials are valid
- URI is correct

---

### Model Not Found Errors
Retrain the intent classifier:

```bash
python -m training.train_intent_model
```

---

## Author

**Likhitha Maradugu**
---

## License

License not specified.

Add a `LICENSE` file if the project is intended for public distribution.