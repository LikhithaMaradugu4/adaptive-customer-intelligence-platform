import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from dotenv import load_dotenv

from app.graph import graph

from app.state import CustomerState

from app.services.session_service import (
    session_service
)

from app.services.customer_service import (
    customer_service
)

from app.utils.title_generator import (
    generate_session_title
)

load_dotenv()

# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(

    page_title=(
        "Adaptive Customer "
        "Intelligence Platform"
    ),

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
        font-size: 34px;
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

if "selected_customer_id" not in st.session_state:

    st.session_state.selected_customer_id = ""

if "active_customer_id" not in st.session_state:

    st.session_state.active_customer_id = None

# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:

    st.markdown(
        "## Customer Session"
    )

    customer_id = st.text_input(

        "Customer ID",
        key="selected_customer_id",
        placeholder="e.g., CUST_123"
    )

    customer_id = (
        st.session_state
        .selected_customer_id
        .strip()
    )


    if not customer_id:

        st.session_state.active_customer_id = None

    elif (

        st.session_state
        .active_customer_id

        != customer_id
    ):

        st.session_state.active_customer_id = (
            customer_id
        )

        st.session_state.current_session_id = None

        st.session_state.customer_state = None

        st.session_state.messages = []

    st.markdown("---")

    # ---------------------------------
    # New Chat
    # ---------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        if not customer_id:

            st.warning(
                "Please enter a valid "
                "customer ID."
            )

            st.stop()

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
                f"New customer created:"
                f" {customer_id}"
            )

        # ---------------------------------
        # Create session
        # ---------------------------------

        new_session = (
            session_service
            .create_session(

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

    st.markdown(
        "### Previous Chats"
    )

    customer_sessions = (
        session_service
        .get_customer_sessions(
            customer_id
        )
        if customer_id
        else []
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

        col1, col2 = st.columns(
            [5, 1]
        )

        # ---------------------------------
        # Open chat
        # ---------------------------------

        with col1:

            is_active = (

                st.session_state
                .current_session_id

                == session_id
            )

            button_label = (

                f"🟢 {session_title}"

                if is_active

                else session_title
            )

            if st.button(

                button_label,

                key=f"open_{session_id}",

                use_container_width=True
            ):

                messages = (
                    session_service
                    .get_session_messages(
                        session_id
                    )
                )

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

        # ---------------------------------
        # Delete chat
        # ---------------------------------

        with col2:

            if st.button(

                "🗑️",

                key=f"delete_{session_id}"
            ):

                confirm_key = (
                    f"confirm_delete_"
                    f"{session_id}"
                )

                st.session_state[
                    confirm_key
                ] = True

        # ---------------------------------
        # Delete confirmation
        # ---------------------------------

        confirm_key = (
            f"confirm_delete_{session_id}"
        )

        if st.session_state.get(
            confirm_key,
            False
        ):

            st.warning(
                f"Delete '{session_title}'?"
            )

            confirm_col1, confirm_col2 = (
                st.columns([1, 1])
            )

            with confirm_col1:

                if st.button(

                    "Confirm",

                    key=f"confirm_yes_"
                    f"{session_id}"
                ):

                    success = (
                        session_service
                        .delete_session(
                            session_id
                        )
                    )

                    if success:

                        if (

                            st.session_state
                            .current_session_id

                            == session_id
                        ):

                            st.session_state.messages = []

                            st.session_state.customer_state = None

                            st.session_state.current_session_id = None

                        st.success(
                            "Chat deleted."
                        )

                    else:

                        st.error(
                            "Failed to delete chat."
                        )

                    st.session_state[
                        confirm_key
                    ] = False

                    st.rerun()

            with confirm_col2:

                if st.button(

                    "Cancel",

                    key=f"confirm_no_"
                    f"{session_id}"
                ):

                    st.session_state[
                        confirm_key
                    ] = False

                    st.rerun()

    st.markdown(
        """
        ### System Features

        - Intent Detection
        - Emotion Analysis
        - Tool Calling
        - Query Rewriting
        - RAG Retrieval
        - Reranking
        - Escalation Workflow
        - Follow-Up Agent
        - MongoDB Memory
        """
    )

# ---------------------------------
# Create default session
# ---------------------------------

if (
    st.session_state.current_session_id
    is None
    and customer_id
):

    customer_profile = (
        customer_service
        .get_customer_by_id(
            customer_id
        )
    )

    if not customer_profile:

        customer_profile = (
            customer_service
            .create_customer(
                customer_id
            )
        )

    default_session = (
        session_service
        .create_session(

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
    """
    <div class="title">
    Adaptive Customer Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered multi-agent customer support assistant
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Global warnings
# ---------------------------------

if not customer_id:

    st.warning(
        "Please enter a valid "
        "customer ID to start."
    )

if st.session_state.customer_state:

    retry_count = (
        st.session_state
        .customer_state
        .retry_count
    )

    errors = (
        st.session_state
        .customer_state
        .errors
    )

    if retry_count >= 2:

        st.warning(
            " System experienced "
            "multiple retries."
        )

    if errors:

        latest_error = errors[-1]

        st.warning(
            f"System Notice: "
            f"{latest_error}"
        )

# ---------------------------------
# Empty chat placeholder
# ---------------------------------

if not st.session_state.messages:

    st.info(
        "👋 Start a conversation "
        "with the AI assistant."
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

    if not customer_id:

        st.warning(
            "Please enter a "
            "customer ID to continue."
        )

        st.stop()

    if not st.session_state.customer_state:

        st.error(
            "No active customer session."
        )

        st.stop()

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
    # Update UI memory
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
    # Generate title
    # ---------------------------------

    current_messages = (
        st.session_state.messages
    )

    if len(current_messages) == 1:

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

    st.session_state.customer_state.conversation_history.append({

        "role": "user",

        "message": user_query
    })

    # ---------------------------------
    # Execute graph
    # ---------------------------------

    try:

        with st.spinner(
            "⚡ Running multi-agent workflow..."
        ):

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

    except Exception as e:

        error_text = str(e)

        if (
            "rate_limit"
            in error_text.lower()
        ):

            st.error(
                "API rate limit reached."
            )

        else:

            st.error(
                "Unexpected system error."
            )

        assistant_response = (
            "We are currently facing "
            "technical difficulties."
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
    # Follow-up rendering
    # ---------------------------------

    if (
        st.session_state
        .customer_state
        .follow_up_required
    ):

        followup_message = (
            st.session_state
            .customer_state
            .follow_up_message
        )

        if followup_message:

            st.info(
                f"💡 "
                f"{followup_message}"
            )

    # ---------------------------------
    # Add assistant response to UI
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

# ---------------------------------
# Footer
# ---------------------------------

st.markdown("---")

st.caption(
    "Adaptive Customer Intelligence Platform • "
    "Enterprise Multi-Agent Customer Support System"
)