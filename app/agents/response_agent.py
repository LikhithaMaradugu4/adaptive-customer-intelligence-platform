from app.state import CustomerState
from app.schemas import ResponseOutput
from app.services.llm_service import llm_service


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
            # Clarification Flow
            # ---------------------------------

            if (
                state.clarification_needed
                and state.clarification_question
            ):

                state.response = (
                    state.clarification_question
                )

                state.metadata[
                    "response_source"
                ] = "clarification"

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

Customer Query:
{state.query}

Customer Emotion:
{state.emotion}

Customer Intent:
{state.intent}

Relevant Knowledge Base Context:
{retrieved_context}

Requirements:
- Use the retrieved context
- Do not hallucinate
- Be concise
- Be professional
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