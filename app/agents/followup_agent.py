from app.state import CustomerState

from app.services.llm_service import (
    llm_service
)

from app.utils.context_manager import (
    get_agent_context
)


class FollowUpAgent:
    """
    Enterprise-grade intelligent follow-up agent.

    Responsibilities:
    - clarification continuity
    - escalation reassurance
    - emotional recovery
    - workflow continuation
    """

    def __init__(self):

        # ---------------------------------
        # Workflow follow-up intents
        # ---------------------------------

        self.workflow_intents = [

            "RETURN_ISSUE",

            "REFUND_ISSUE",

            "DELIVERY_ISSUE",

            "PAYMENT_ISSUE"
        ]

        # ---------------------------------
        # High-priority intents
        # ---------------------------------

        self.high_priority_intents = {

            "PAYMENT_ISSUE",

            "REFUND_ISSUE",

            "DELIVERY_ISSUE",

            "RETURN_ISSUE",

            "ACCOUNT_ISSUE",

            "PRODUCT_ISSUE",

            "COMPLAINT"
        }

        # ---------------------------------
        # Low-priority intents
        # ---------------------------------

        self.low_priority_intents = {

            "GENERAL_QUERY",

            "UNKNOWN_INTENT",

            "NON_SUPPORT"
        }

    # ---------------------------------
    # Normalize intents
    # ---------------------------------

    def _normalize_intents(
        self,
        intents
    ):

        if not intents:

            return set()

        return {

            intent

            for intent in intents

            if isinstance(intent, str)
        }

    # ---------------------------------
    # Greeting / casual detection
    # ---------------------------------

    def _is_greeting_or_small_talk(
        self,
        query: str
    ) -> bool:

        if not query:

            return False

        lowered = query.lower().strip()

        greeting_phrases = [

            "hi",

            "hello",

            "hey",

            "good morning",

            "good afternoon",

            "good evening",

            "how are you",

            "what's up",

            "thanks",

            "thank you",

            "bye",

            "goodbye"
        ]

        return any(

            lowered == phrase

            or lowered.startswith(
                f"{phrase} "
            )

            for phrase in greeting_phrases
        )

    # ---------------------------------
    # Response action detection
    # ---------------------------------

    def _response_requests_action(
        self,
        response: str
    ) -> bool:

        if not response:

            return False

        lowered = response.lower()

        action_markers = [

            "please provide",

            "please share",

            "can you confirm",

            "could you confirm",

            "share your",

            "provide your",

            "confirm your",

            "verify your",

            "order id",

            "order number",

            "email address",

            "phone number",

            "verification",

            "upload",

            "send"
        ]

        if "?" in lowered:

            return True

        return any(

            marker in lowered

            for marker in action_markers
        )

    # ---------------------------------
    # Response completion check
    # ---------------------------------

    def _response_is_complete(
        self,
        response: str
    ) -> bool:

        if not response:

            return False

        if self._response_requests_action(
            response
        ):

            return False

        return True

    # ---------------------------------
    # Account lock signal
    # ---------------------------------

    def _has_account_lock_signal(
        self,
        query: str
    ) -> bool:

        if not query:

            return False

        lowered = query.lower()

        return any(

            marker in lowered

            for marker in [

                "account locked",

                "locked out",

                "account suspended",

                "account blocked",

                "disabled account"
            ]
        )

    # ---------------------------------
    # Resolution detection
    # ---------------------------------

    def _issue_resolved(
        self,
        state: CustomerState
    ) -> bool:

        if state.clarification_needed:

            return False

        if state.decision in [

            "ESCALATE",

            "HUMAN_APPROVAL",

            "FRAUD_REVIEW",

            "CLARIFY"
        ]:

            return False

        if state.emotion in [

            "ANGRY",

            "FRUSTRATED"
        ]:

            return False

        if state.workflow_status and (

            state.workflow_status.upper()
            in [

                "RESOLVED",

                "COMPLETED",

                "CLOSED",

                "DONE"
            ]
        ):

            return True

        return self._response_is_complete(
            state.response
        )

    # ---------------------------------
    # Follow-up eligibility gating
    # ---------------------------------

    def _should_generate_followup(
        self,
        state: CustomerState
    ) -> bool:

        if not state.response:

            return False

        if self._is_greeting_or_small_talk(
            state.query
        ):

            return False

        intents = self._normalize_intents(
            state.intent
        )

        if intents & self.low_priority_intents:

            return False

        if state.metadata.get(
            "response_source"
        ) == "clarification":

            return False

        if self._issue_resolved(state):

            return False

        if self._response_requests_action(
            state.response
        ):

            return False

        if state.clarification_needed:

            return True

        if state.decision in [

            "ESCALATE",

            "HUMAN_APPROVAL",

            "FRAUD_REVIEW"
        ]:

            return True

        if state.emotion in [

            "ANGRY",

            "FRUSTRATED"
        ]:

            return True

        if self._has_account_lock_signal(
            state.query
        ):

            return True

        if intents & self.high_priority_intents:

            return True

        return False

    # ---------------------------------
    # Optimized system rules
    # ---------------------------------

    def _build_system_rules(self):

        return """
You are ShopSphere's follow-up engine.

Goal:
Generate concise follow-ups only when they improve customer outcome.

Rules:
1. Follow-ups are rare and intentional.
2. Never add filler or continue conversation unnecessarily.
3. Never generate follow-ups for:
   - greetings
   - small talk
   - FAQs
   - resolved queries
   - simple answered questions

4. Generate follow-ups only for:
   - clarification
   - escalation reassurance
   - emotional recovery
   - incomplete workflows
   - pending customer action
   - critical refund/payment/order flows

5. Never repeat the main response.

6. Keep follow-ups:
   - short
   - natural
   - supportive
   - operationally useful

7. Max 1 sentence.

8. Never hallucinate policies, timelines, or actions.
"""

    # ---------------------------------
    # Generate follow-up
    # ---------------------------------

    def _generate_followup(
        self,
        state: CustomerState,
        followup_type: str
    ):

        # ---------------------------------
        # Lightweight context
        # ---------------------------------

        conversation_context = (
            get_agent_context(
                state
            )
        )

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate a concise follow-up.

Type:
{followup_type}

Emotion:
{state.emotion}

Intent:
{state.intent}

Decision:
{state.decision}

Assistant Response:
{state.response}

Conversation:
{conversation_context}

Requirements:
- max 1 sentence
- no filler
- no repetition
- conversational
- context-aware
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.3
            )
        )

        # ---------------------------------
        # Handle AIMessage or raw string
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

            intents = self._normalize_intents(
                state.intent
            )

            # ---------------------------------
            # Strict gating
            # ---------------------------------

            if not self._should_generate_followup(
                state
            ):

                return state

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

            if (
                state.decision in [

                    "ESCALATE",

                    "HUMAN_APPROVAL",

                    "FRAUD_REVIEW"
                ]
                and (
                    state.emotion in [

                        "ANGRY",

                        "FRUSTRATED"
                    ]
                    or state.escalated
                    or state.decision == "FRAUD_REVIEW"
                )
            ):

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

            workflow_incomplete = (

                not state.workflow_status
                or state.workflow_status.upper()
                not in [

                    "RESOLVED",

                    "COMPLETED",

                    "CLOSED",

                    "DONE"
                ]
            )

            if (
                workflow_incomplete
                and (
                    any(

                        intent in intents

                        for intent in self.workflow_intents
                    )
                    or self._has_account_lock_signal(
                        state.query
                    )
                )
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
            # Product workflow support
            # ---------------------------------

            if (
                "PRODUCT_ISSUE" in intents
                and workflow_incomplete
                and (
                    state.clarification_needed
                    or state.emotion in [

                        "ANGRY",

                        "FRUSTRATED"
                    ]
                    or state.decision in [

                        "ESCALATE",

                        "HUMAN_APPROVAL",

                        "FRAUD_REVIEW"
                    ]
                )
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