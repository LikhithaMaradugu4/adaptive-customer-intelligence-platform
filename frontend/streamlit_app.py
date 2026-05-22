import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from app.graph import graph
from app.state import CustomerState


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="Adaptive Customer Intelligence Platform",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------
# Minimal styling
# ---------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
    }

    .title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9CA3AF;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:

    st.markdown("## Customer Session")

    customer_id = st.text_input(
        "Customer ID",
        value="CUST_001"
    )

    st.markdown("---")

    st.markdown(
        """
        ### System Features

        - Intent Detection
        - Emotion Analysis
        - RAG Retrieval
        - Customer Profiling
        - Decision Engine
        - Escalation Workflow
        - Conversational Memory
        """
    )


# ---------------------------------
# Session state initialization
# ---------------------------------

if "customer_state" not in st.session_state:

    st.session_state.customer_state = CustomerState(
        customer_id=customer_id,
        query=""
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------
# Header
# ---------------------------------

st.markdown(
    '<div class="title">Adaptive Customer Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered multi-agent customer support assistant</div>',
    unsafe_allow_html=True
)


# ---------------------------------
# Display chat history
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------
# Chat input
# ---------------------------------

user_query = st.chat_input(
    "Type your message here..."
)


# ---------------------------------
# Handle user input
# ---------------------------------

if user_query:

    # ---------------------------------
    # Update customer ID dynamically
    # ---------------------------------

    st.session_state.customer_state.customer_id = (
        customer_id
    )

    # ---------------------------------
    # Add user message
    # ---------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):

        st.markdown(user_query)

    # ---------------------------------
    # Update query in state
    # ---------------------------------

    st.session_state.customer_state.query = (
        user_query
    )

    # ---------------------------------
    # Store conversation memory
    # ---------------------------------

    st.session_state.customer_state.conversation_history.append({

        "role": "customer",

        "message": user_query
    })

    # ---------------------------------
    # Run graph
    # ---------------------------------

    with st.spinner("Processing..."):

        result = graph.invoke(
            st.session_state
            .customer_state
            .model_dump()
        )

        updated_state = CustomerState(
            **result
        )

        st.session_state.customer_state = (
            updated_state
        )

        assistant_response = (
            updated_state.response
        )

    # ---------------------------------
    # Display assistant response
    # ---------------------------------

    with st.chat_message("assistant"):

        st.markdown(assistant_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })

    # ---------------------------------
    # Store assistant response
    # ---------------------------------

    st.session_state.customer_state.conversation_history.append({

        "role": "assistant",

        "message": assistant_response
    })


# ---------------------------------
# Footer info
# ---------------------------------

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Current Decision",
        st.session_state
        .customer_state
        .decision
        or "N/A"
    )

with col2:

    st.metric(
        "Detected Emotion",
        st.session_state
        .customer_state
        .emotion
        or "N/A"
    )

with col3:

    st.metric(
        "Escalated",
        str(
            st.session_state
            .customer_state
            .escalated
        )
    )