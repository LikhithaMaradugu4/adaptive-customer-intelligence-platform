from app.state import CustomerState

from app.schemas import ResponseOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.conversation import (
    build_conversation_context
)


class ResponseAgent:
    """
    Generates final customer-facing responses.
    """

    def _build_context(
        self,
        state: CustomerState
    ) -> str:
        """
        Build grounded RAG context.
        """

        if not state.retrieved_docs:

            return "No retrieved documents."

        retrieved_context = ""

        for idx, doc in enumerate(
            state.retrieved_docs,
            start=1
        ):

            retrieved_context += (
                f"\nDocument {idx}:\n"
                f"{doc['content']}\n"
            )

        return retrieved_context

    def _build_tool_context(
        self,
        state: CustomerState
    ) -> str:
        """
        Build operational tool context.
        """

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

            formatted_tools += (
                f"\nTool {idx}:\n"
                f"Tool Name: "
                f"{tool_output.get('tool_name')}\n"
                f"Tool Output: "
                f"{tool_output.get('tool_output')}\n"
            )

        return formatted_tools

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Conversation context
            # ---------------------------------

            conversation_context = (
                build_conversation_context(
                    state.conversation_history
                )
            )

            # ---------------------------------
            # Tool context
            # ---------------------------------

            tool_context = (
                self._build_tool_context(
                    state
                )
            )

            # ---------------------------------
            # Clarification Flow
            # ---------------------------------

            if (
                state.clarification_needed
                and state.clarification_question
            ):

                prompt = f"""
You are a professional customer support assistant.

IMPORTANT:
- Use conversation history
- Use operational memory/tool outputs
- Maintain conversational continuity
- Avoid repetitive clarification
- Ask clarification naturally

Conversation Context:
{conversation_context}

Operational Tool Context:
{tool_context}

Current Customer Query:
{state.query}

Clarification Question:
{state.clarification_question}

Requirements:
- Be conversational
- Be concise
- Be natural
"""

                llm = llm_service._create_llm(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3
                )

                structured_llm = (
                    llm.with_structured_output(
                        ResponseOutput
                    )
                )

                result = structured_llm.invoke(
                    prompt
                )

                state.response = (
                    result.response
                )

                state.metadata[
                    "response_source"
                ] = "clarification"

                return state

            # ---------------------------------
            # Out-of-scope Flow
            # ---------------------------------

            if state.decision == "OUT_OF_SCOPE":

                state.response = (
                    "I can only assist with "
                    "ShopSphere-related customer support queries."
                )

                return state

            # ---------------------------------
            # Escalation Flow
            # ---------------------------------

            if state.decision in [
                "ESCALATE",
                "HUMAN_APPROVAL",
                "FRAUD_REVIEW"
            ]:

                escalation_details = (
                    state.escalation_details
                    or {}
                )

                assigned_team = (
                    escalation_details.get(
                        "assigned_team",
                        "support team"
                    )
                )

                ticket_id = (
                    escalation_details.get(
                        "ticket_id",
                        "N/A"
                    )
                )

                prompt = f"""
You are a professional customer support assistant.

Conversation Context:
{conversation_context}

Operational Tool Context:
{tool_context}

Customer Emotion:
{state.emotion}

Decision:
{state.decision}

Assigned Team:
{assigned_team}

Ticket ID:
{ticket_id}

IMPORTANT:
- Use conversation history
- Use operational memory
- Be empathetic
- Maintain conversational continuity

Requirements:
- Mention escalation
- Mention ticket ID
- Keep response concise

If the latest user message appears to answer
a previous clarification question,
continue the previous workflow naturally.

IMPORTANT:
Customer ID is already available in system state.
Do not ask customer again for customer ID
unless absolutely necessary.
"""

            else:

                # ---------------------------------
                # Normal Response Flow
                # ---------------------------------

                retrieved_context = (
                    self._build_context(
                        state
                    )
                )

                prompt = f"""
You are a professional customer support assistant.

IMPORTANT BEHAVIOR RULES:

1. Use conversation history heavily.
2. Use operational tool outputs heavily.
3. Maintain conversational continuity.
4. Resolve references like:
   - it
   - that
   - earlier product
   - previous order
5. If customer asks general policy/process questions:
   - answer directly
   - do NOT ask unnecessary clarification
6. Use retrieved company knowledge whenever available.
7. If tool outputs contain customer/profile/order/session/product data:
   use it naturally in response.
8. Never ignore known conversational context.
9. Do not repeatedly ask for order IDs
   unless operational processing is required.

Conversation History:
{conversation_context}

Operational Tool Outputs:
{tool_context}

Customer Query:
{state.query}

Customer Emotion:
{state.emotion}

Customer Intent:
{state.intent}

Relevant Knowledge Base Context:
{retrieved_context}

Requirements:
- Do not hallucinate
- Be concise
- Be professional
- Be conversational
- Maintain continuity
- Use memory naturally
- Be empathetic if customer is frustrated or angry
"""

            # ---------------------------------
            # Create LLM
            # ---------------------------------

            llm = llm_service._create_llm(
                model_name="llama-3.3-70b-versatile",
                temperature=0.4
            )

            structured_llm = (
                llm.with_structured_output(
                    ResponseOutput
                )
            )

            result = structured_llm.invoke(
                prompt
            )

            state.response = (
                result.response
            )

            state.metadata[
                "response_source"
            ] = "llm"

            return state

        except Exception as e:

            print("\nResponseAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"ResponseAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.response = (
                "We are currently facing "
                "technical difficulties. "
                "Please try again later."
            )

            return state