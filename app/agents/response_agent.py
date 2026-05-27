from app.state import CustomerState

from app.schemas import ResponseOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.context_manager import (
    get_agent_context
)


class ResponseAgent:
    """
    Enterprise-grade response generation agent.

    Responsibilities:
    - grounded response generation
    - conversational continuity
    - escalation messaging
    - clarification handling
    - emotionally adaptive responses
    """

    def __init__(self):

        self.max_context_docs = 3

    # ---------------------------------
    # Build retrieved context
    # ---------------------------------

    def _build_retrieved_context(
        self,
        state: CustomerState
    ) -> str:

        if not state.retrieved_docs:

            return "No retrieved documents."

        formatted_context = ""

        for idx, doc in enumerate(
            state.retrieved_docs[
                :self.max_context_docs
            ],
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
            )[:800]

            formatted_context += f"""

Document {idx}

Source: {source}
Page: {page}
Category: {category}

Content:
{content}
"""

        return formatted_context

    # ---------------------------------
    # Build tool context
    # ---------------------------------

    def _build_tool_context(
        self,
        state: CustomerState
    ) -> str:

        if not getattr(
            state,
            "tool_outputs",
            None
        ):

            return "No tool outputs."

        formatted_tools = ""

        for idx, tool_output in enumerate(
            state.tool_outputs,
            start=1
        ):

            tool_name = tool_output.get(
                "tool_name",
                "unknown"
            )

            tool_result = tool_output.get(
                "tool_output",
                ""
            )

            formatted_tools += f"""

Tool {idx}

Name:
{tool_name}

Output:
{tool_result}
"""

        return formatted_tools

    # ---------------------------------
    # Optimized system rules
    # ---------------------------------

    def _build_system_rules(self):

        return """
You are ShopSphere's AI support assistant.

Goals:
- provide accurate support
- maintain conversational continuity
- use operational context naturally
- avoid hallucinations

Rules:
1. Use conversation context naturally.
2. Resolve references like:
   - it
   - that order
   - previous issue
   - this product

3. Never ask again for known information.
4. Use retrieved docs and tool outputs when available.
5. Never invent policies, timelines, or workflows.
6. Ask clarification only if required.
7. Be:
   - concise
   - professional
   - conversational
   - emotionally aware

8. For frustrated users:
   acknowledge first, then solve.

9. Use customer/profile details naturally.
   Never mention fetching or lookup actions.

10. For casual or unrelated conversation:
   respond briefly and naturally,
   then gently redirect to ShopSphere support.

Formatting:
- short paragraphs
- bullets when useful
- avoid large text blocks
"""

    # ---------------------------------
    # Clarification response
    # ---------------------------------

    def _clarification_flow(
        self,
        state: CustomerState,
        conversation_context: str,
        tool_context: str
    ):

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate a natural clarification response.

Conversation:
{conversation_context}

Tool Context:
{tool_context}

Customer Query:
{state.query}

Clarification Needed:
{state.clarification_question}

Requirements:
- concise
- natural
- non-repetitive
- conversational
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.3,

                structured_output=ResponseOutput
            )
        )

        return result.response

    # ---------------------------------
    # Escalation response
    # ---------------------------------

    def _escalation_flow(
        self,
        state: CustomerState,
        conversation_context: str,
        tool_context: str
    ):

        escalation_details = (
            state.escalation_details
            or {}
        )

        assigned_team = (
            escalation_details.get(
                "assigned_team",
                "Support Team"
            )
        )

        ticket_id = (
            escalation_details.get(
                "ticket_id",
                "N/A"
            )
        )

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate a professional escalation response.

Conversation:
{conversation_context}

Tool Context:
{tool_context}

Emotion:
{state.emotion}

Decision:
{state.decision}

Assigned Team:
{assigned_team}

Ticket ID:
{ticket_id}

Requirements:
- reassure customer
- acknowledge issue seriously
- mention escalation naturally
- include ticket ID
- concise and calm
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.4,

                structured_output=ResponseOutput
            )
        )

        return result.response

    # ---------------------------------
    # Normal grounded response
    # ---------------------------------

    def _normal_response_flow(
        self,
        state: CustomerState,
        conversation_context: str,
        tool_context: str,
        retrieved_context: str
    ):

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate the best grounded customer-support response.

Conversation:
{conversation_context}

Tool Outputs:
{tool_context}

Customer Query:
{state.query}

Emotion:
{state.emotion}

Intent:
{state.intent}

Retrieved Context:
{retrieved_context}

Requirements:
- answer directly
- maintain continuity
- use retrieved knowledge accurately
- avoid unnecessary clarification
- explain steps clearly when needed
- avoid repetition or robotic wording

If unrelated to ShopSphere:
- respond briefly
- gently redirect to ShopSphere support
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.4,

                structured_output=ResponseOutput
            )
        )

        return result.response

    # ---------------------------------
    # Main execution
    # ---------------------------------

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Lightweight conversation context
            # ---------------------------------

            conversation_context = (
                get_agent_context(
                    state
                )
            )

            tool_context = (
                self._build_tool_context(
                    state
                )
            )

            retrieved_context = (
                self._build_retrieved_context(
                    state
                )
            )

            # ---------------------------------
            # Clarification flow
            # ---------------------------------

            if (
                state.clarification_needed
                and state.clarification_question
            ):

                state.response = (
                    self._clarification_flow(

                        state,

                        conversation_context,

                        tool_context
                    )
                )

                state.metadata[
                    "response_source"
                ] = "clarification"

                return state

            # ---------------------------------
            # Out-of-scope flow
            # ---------------------------------

            if state.decision == "OUT_OF_SCOPE":

                state.response = (
                    "I can help with ShopSphere "
                    "orders, account, payments, "
                    "delivery, or support."
                )

                return state

            # ---------------------------------
            # Escalation flow
            # ---------------------------------

            if state.decision in [

                "ESCALATE",

                "HUMAN_APPROVAL",

                "FRAUD_REVIEW"
            ]:

                state.response = (
                    self._escalation_flow(

                        state,

                        conversation_context,

                        tool_context
                    )
                )

                state.metadata[
                    "response_source"
                ] = "escalation"

                return state

            # ---------------------------------
            # Normal grounded response
            # ---------------------------------

            state.response = (
                self._normal_response_flow(

                    state,

                    conversation_context,

                    tool_context,

                    retrieved_context
                )
            )

            state.metadata[
                "response_source"
            ] = "grounded_llm"

            return state

        except Exception as e:

            print("\nResponseAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"ResponseAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.response = (
                "We're currently facing "
                "technical difficulties. "
                "Please try again shortly."
            )

            return state