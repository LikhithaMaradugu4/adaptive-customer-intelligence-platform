from typing import Callable

from app import state
from app.state import CustomerState


class Supervisor:
    """
    Central workflow orchestrator.

    Responsibilities:
    - Run agents
    - Validate outputs
    - Handle retries
    - Log execution
    """

    def __init__(self):

        self.max_retries = 2

    def validate_intent(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate Intent Agent output.
        """

        if not state.intent:
            return False

        if state.intent_confidence is None:
            return False

        return True
    def validate_emotion(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate Emotion Agent output.
        """

        if not state.emotion:
            return False

        if state.emotion_confidence is None:
            return False

        return True
    def validate_memory(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate Memory Agent output.
        """

        if state.customer_history is None:
            return False
        return True 
    def validate_rag(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate RAG retrieval output.
        """

        if state.retrieved_docs is None:
            return False

        return True
    def validate_profile(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate profile output.
        """

        if not state.customer_profile:
            return False
        return True 
    def validate_decision( 
            self,
            state: CustomerState
    ) -> bool:
        """
        Validate decision agent output.
        """
        if not state.decision:
            return False
        return True 
    
    def validate_escalation(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate if escalation is needed.
        """
        if state.decision in [
            "FRAUD_REVIEW",
            "ESCALATE",
            "HUMAN_APPROVAL"
        ]:
            if not state.escalation_details:
                return False
        return True
    
    def validate_response(
        self,
        state: CustomerState
    ) -> bool:
        """
        Validate generated response.
        """

        if not state.response:
            return False

        return True

    def run_agent(
        self,
        agent_callable: Callable,
        state: CustomerState,
        validator: Callable = None
    ) -> CustomerState:
        """
        Execute an agent with validation + retry support.
        """

        for attempt in range(
            self.max_retries
        ):

            try:

                updated_state = agent_callable(
                    state
                )

                # ---------------------------------
                # Validation
                # ---------------------------------

                if validator:

                    is_valid = validator(
                        updated_state
                    )

                    if not is_valid:

                        updated_state.errors.append(
                            f"Validation failed on attempt {attempt + 1}"
                        )

                        continue

                updated_state.metadata[
                    "last_successful_agent"
                ] = agent_callable.__self__.__class__.__name__

                return updated_state

            except Exception as e:

                state.errors.append(
                    f"Supervisor Error: {str(e)}"
                )

                state.retry_count += 1

        # ---------------------------------
        # Escalation after failure
        # ---------------------------------

        state.escalated = True

        state.escalation_reason = (
            "agent_failure"
        )

        return state


# Singleton instance
supervisor = Supervisor()