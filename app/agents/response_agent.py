from app.state import CustomerState
from app.schemas import ResponseOutput
from app.services.llm_service import llm_service

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

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Build conversation context
            # ---------------------------------

            conversation_context = (
                build_conversation_context(
                    state.conversation_history
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

Conversation Context:
{conversation_context}

Current Customer Query:
{state.query}

Clarification Question:
{state.clarification_question}

Requirements:
- Ask the clarification naturally
- Be conversational
- Use conversation history for continuity
- Be concise
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
Generate a professional customer support response.

Conversation Context:
{conversation_context}

Customer Emotion:
{state.emotion}

Decision:
{state.decision}

Assigned Team:
{assigned_team}

Ticket ID:
{ticket_id}

Requirements:
- Be empathetic
- Mention escalation
- Mention ticket ID
- Use conversation context
- Keep response concise
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
Generate a professional customer support response.

Conversation History:
{conversation_context}

Customer Query:
{state.query}

Customer Emotion:
{state.emotion}

Customer Intent:
{state.intent}

Relevant Knowledge Base Context:
{retrieved_context}

IMPORTANT:
- Use conversation history
- Resolve references like:
  "it", "that product", "earlier"
- Maintain conversational continuity
- Use retrieved knowledge when relevant

Requirements:
- Do not hallucinate
- Be concise
- Be professional
- Be conversational
- Be empathetic if customer is frustrated or angry
"""

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