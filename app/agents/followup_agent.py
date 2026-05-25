from app.state import CustomerState


class FollowUpAgent:
    """
    Intelligent follow-up orchestration agent.
    """

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Default
            # ---------------------------------

            state.follow_up_required = False

            state.follow_up_type = None

            state.follow_up_message = None

            # ---------------------------------
            # Clarification follow-up
            # ---------------------------------

            if state.clarification_needed:

                state.follow_up_required = True

                state.follow_up_type = (
                    "clarification"
                )

                state.follow_up_message = (
                    state.clarification_question
                )

                return state

            # ---------------------------------
            # Escalation reassurance
            # ---------------------------------

            if state.decision in [
                "ESCALATE",
                "HUMAN_APPROVAL",
                "FRAUD_REVIEW"
            ]:

                state.follow_up_required = True

                state.follow_up_type = (
                    "escalation_reassurance"
                )

                state.follow_up_message = (
                    "Our support team is "
                    "actively reviewing your issue."
                )

                return state

            # ---------------------------------
            # Emotional recovery
            # ---------------------------------

            if state.emotion in [
                "ANGRY",
                "FRUSTRATED"
            ]:

                state.follow_up_required = True

                state.follow_up_type = (
                    "emotional_recovery"
                )

                state.follow_up_message = (
                    "I'm here to help resolve "
                    "this as quickly as possible."
                )

                return state

            # ---------------------------------
            # Workflow continuation
            # ---------------------------------

            if (
                state.intent
                and (
                    "RETURN_ISSUE"
                    in state.intent
                    or "REFUND_ISSUE"
                    in state.intent
                )
            ):

                state.follow_up_required = True

                state.follow_up_type = (
                    "workflow_continuation"
                )

                state.follow_up_message = (
                    "Would you like guidance "
                    "for the next steps?"
                )

                return state

            # ---------------------------------
            # Smart recommendations
            # ---------------------------------

            if (
                state.intent
                and "PRODUCT_ISSUE"
                in state.intent
            ):

                state.follow_up_required = True

                state.follow_up_type = (
                    "smart_recommendation"
                )

                state.follow_up_message = (
                    "I can also help you "
                    "with replacement options "
                    "or warranty support."
                )

            return state

        except Exception as e:

            print("\nFollowUpAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"FollowUpAgent Error: {str(e)}"
            )

            return state