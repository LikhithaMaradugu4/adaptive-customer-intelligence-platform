import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from app.graph import graph

from app.state import CustomerState

from app.services.session_service import (
    session_service
)

from app.services.customer_service import (
    customer_service
)

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
# Session state initialization
# ---------------------------------

if "customer_state" not in st.session_state:

    st.session_state.customer_state = None

if "messages" not in st.session_state:

    st.session_state.messages = []

if "current_session_id" not in st.session_state:

    st.session_state.current_session_id = None

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

    # ---------------------------------
    # New Chat
    # ---------------------------------

    if st.button("➕ New Chat"):

        # ---------------------------------
        # Fetch customer profile
        # ---------------------------------

        customer_profile = (
            customer_service
            .get_customer_by_id(
                customer_id
            )
        )

        # ---------------------------------
        # Auto-create customer
        # ---------------------------------

        if not customer_profile:

            customer_profile = (
                customer_service
                .create_customer(
                    customer_id
                )
            )

            st.success(
                f"New customer created: "
                f"{customer_id}"
            )

        # ---------------------------------
        # Create new session
        # ---------------------------------

        new_session = (
            session_service.create_session(
                customer_id=customer_id,
                title="New Conversation"
            )
        )

        st.session_state.current_session_id = (
            new_session["session_id"]
        )

        st.session_state.messages = []

        st.session_state.customer_state = (
            CustomerState(
                customer_id=customer_id,
                session_id=(
                    new_session["session_id"]
                ),
                customer_profile=(
                    customer_profile
                ),
                conversation_history=[],
                query=""
            )
        )

        st.rerun()

    st.markdown("---")

    st.markdown("### Previous Chats")

    customer_sessions = (
        session_service.get_customer_sessions(
            customer_id
        )
    )

    # ---------------------------------
    # Session List
    # ---------------------------------

    for session in customer_sessions:

        session_title = session.get(
            "title",
            "Untitled Chat"
        )

        session_id = session[
            "session_id"
        ]

        if st.button(
            session_title,
            key=session_id
        ):

            # ---------------------------------
            # Load messages
            # ---------------------------------

            messages = (
                session_service
                .get_session_messages(
                    session_id
                )
            )

            # ---------------------------------
            # Fetch customer profile
            # ---------------------------------

            customer_profile = (
                customer_service
                .get_customer_by_id(
                    customer_id
                )
            )

            st.session_state.messages = []

            conversation_history = []

            for msg in messages:

                st.session_state.messages.append({

                    "role": msg["role"],

                    "content": msg["message"]
                })

                conversation_history.append({

                    "role": msg["role"],

                    "message": msg["message"]
                })

            # ---------------------------------
            # Restore fully hydrated state
            # ---------------------------------

            st.session_state.current_session_id = (
                session_id
            )

            st.session_state.customer_state = (
                CustomerState(
                    customer_id=customer_id,
                    session_id=session_id,
                    customer_profile=(
                        customer_profile
                    ),
                    conversation_history=(
                        conversation_history
                    ),
                    query=""
                )
            )

            st.rerun()

    st.markdown("---")

    st.markdown(
        """
        ### System Features

        - Intent Detection
        - Emotion Analysis
        - Conditional RAG
        - Tool Calling
        - Decision Engine
        - Escalation Workflow
        - Persistent Memory
        - MongoDB Sessions
        """
    )

# ---------------------------------
# Create default session
# ---------------------------------

if (
    st.session_state.current_session_id
    is None
):

    # ---------------------------------
    # Fetch customer profile
    # ---------------------------------

    customer_profile = (
        customer_service
        .get_customer_by_id(
            customer_id
        )
    )

    # ---------------------------------
    # Auto-create customer
    # ---------------------------------

    if not customer_profile:

        customer_profile = (
            customer_service
            .create_customer(
                customer_id
            )
        )

        st.success(
            f"New customer created: "
            f"{customer_id}"
        )

    # ---------------------------------
    # Create session
    # ---------------------------------

    default_session = (
        session_service.create_session(
            customer_id=customer_id,
            title="New Conversation"
        )
    )

    st.session_state.current_session_id = (
        default_session["session_id"]
    )

    st.session_state.customer_state = (
        CustomerState(
            customer_id=customer_id,
            session_id=(
                default_session["session_id"]
            ),
            customer_profile=(
                customer_profile
            ),
            conversation_history=[],
            query=""
        )
    )

# ---------------------------------
# Header
# ---------------------------------

st.markdown(
    '<div class="title">'
    'Adaptive Customer Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered multi-agent customer support assistant'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# Display chat history
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

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

    current_session_id = (
        st.session_state
        .current_session_id
    )

    # ---------------------------------
    # Display user message
    # ---------------------------------

    with st.chat_message("user"):

        st.markdown(user_query)

    # ---------------------------------
    # Add user message to UI
    # ---------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_query
    })

    # ---------------------------------
    # Save user message
    # ---------------------------------

    session_service.save_message(
        session_id=current_session_id,
        customer_id=customer_id,
        role="user",
        message=user_query
    )

    # ---------------------------------
    # Generate session title
    # ---------------------------------

    current_messages = (
        st.session_state.messages
    )

    if len(current_messages) == 1:

        from app.utils.title_generator import (
            generate_session_title
        )

        title = generate_session_title(
            user_query
        )

        session_service.update_session_title(
            session_id=current_session_id,
            title=title
        )

    # ---------------------------------
    # Update state
    # ---------------------------------

    st.session_state.customer_state.query = (
        user_query
    )

    # ---------------------------------
    # Update conversation history
    # ---------------------------------

    st.session_state.customer_state.conversation_history.append({

        "role": "user",

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

        updated_state = (
            CustomerState(
                **result
            )
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

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            assistant_response
        )

    # ---------------------------------
    # Add assistant message to UI
    # ---------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": assistant_response
    })

    # ---------------------------------
    # Save assistant message
    # ---------------------------------

    session_service.save_message(
        session_id=current_session_id,
        customer_id=customer_id,
        role="assistant",
        message=assistant_response
    )

    # ---------------------------------
    # Update conversation history
    # ---------------------------------

    st.session_state.customer_state.conversation_history.append({

        "role": "assistant",

        "message": assistant_response
    })