from app.state import CustomerState

from app.services.llm_service import (
    llm_service
)

from app.utils.conversation import (
    build_conversation_context
)


class FollowUpAgent:
    """
    Enterprise-grade intelligent follow-up agent.

    Responsibilities:
    - clarification continuity
    - escalation reassurance
    - emotional recovery
    - workflow continuation
    - proactive support
    - conversational retention
    """

    def __init__(self):

        # ---------------------------------
        # Follow-up enabled intents
        # ---------------------------------

        self.workflow_intents = [

            "RETURN_ISSUE",

            "REFUND_ISSUE",

            "DELIVERY_ISSUE",

            "PAYMENT_ISSUE"
        ]

    # ---------------------------------
    # Build follow-up system rules
    # ---------------------------------

    def _build_system_rules(self):

        return """
You are ShopSphere's intelligent follow-up orchestration engine.

Your role:
- maintain conversational continuity
- improve customer satisfaction
- reduce frustration
- guide workflow completion
- provide proactive support

CRITICAL BEHAVIOR RULES:

1. Follow-ups must feel:
   - natural
   - conversational
   - non-robotic
   - context-aware

2. NEVER repeat the main response.

3. Follow-ups should:
   - guide next steps
   - reassure customers
   - reduce confusion
   - encourage completion

4. If customer is frustrated:
   prioritize emotional reassurance.

5. If escalation happened:
   reassure operational continuity.

6. If workflow is incomplete:
   encourage next-step continuation.

7. Keep follow-ups concise.

8. Avoid sounding pushy.

9. Follow-ups should feel premium
and human-like.

10. Never hallucinate policies
or timelines.
"""

    # ---------------------------------
    # Generate intelligent follow-up
    # ---------------------------------

    def _generate_followup(
        self,
        state: CustomerState,
        followup_type: str
    ):

        conversation_context = (
            build_conversation_context(
                state.conversation_history
            )
        )

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate a concise intelligent follow-up.

Follow-Up Type:
{followup_type}

Customer Emotion:
{state.emotion}

Customer Intent:
{state.intent}

Decision:
{state.decision}

Latest Assistant Response:
{state.response}

Conversation Context:
{conversation_context}

IMPORTANT:
- keep it short
- conversational
- supportive
- context-aware
- do not repeat previous response
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.4
            )
        )

        # ---------------------------------
        # Handle raw string OR AIMessage
        # ---------------------------------

        if hasattr(
            result,
            "content"
        ):

            return result.content

        return str(result)

    # ---------------------------------
    # Main execution
    # ---------------------------------

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Reset follow-up state
            # ---------------------------------

            state.follow_up_required = False

            state.follow_up_type = None

            state.follow_up_message = None

            # ---------------------------------
            # Clarification continuity
            # ---------------------------------

            if state.clarification_needed:

                state.follow_up_required = True

                state.follow_up_type = (
                    "clarification_continuity"
                )

                state.follow_up_message = (
                    self._generate_followup(

                        state,

                        "clarification_continuity"
                    )
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
                    self._generate_followup(

                        state,

                        "escalation_reassurance"
                    )
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
                    self._generate_followup(

                        state,

                        "emotional_recovery"
                    )
                )

                return state

            # ---------------------------------
            # Workflow continuation
            # ---------------------------------

            if any(

                intent in state.intent

                for intent in self.workflow_intents
            ):

                state.follow_up_required = True

                state.follow_up_type = (
                    "workflow_continuation"
                )

                state.follow_up_message = (
                    self._generate_followup(

                        state,

                        "workflow_continuation"
                    )
                )

                return state

            # ---------------------------------
            # Product recommendation follow-up
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
                    self._generate_followup(

                        state,

                        "smart_recommendation"
                    )
                )

                return state

            # ---------------------------------
            # Satisfaction reinforcement
            # ---------------------------------

            if state.emotion == "SATISFIED":

                state.follow_up_required = True

                state.follow_up_type = (
                    "satisfaction_reinforcement"
                )

                state.follow_up_message = (
                    self._generate_followup(

                        state,

                        "satisfaction_reinforcement"
                    )
                )

                return state

            return state

        except Exception as e:

            print("\nFollowUpAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"FollowUpAgent Error: {str(e)}"
            )

            # ---------------------------------
            # Safe fallback
            # ---------------------------------

            state.follow_up_required = False

            state.follow_up_type = None

            state.follow_up_message = None

            return state