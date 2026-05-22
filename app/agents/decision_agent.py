from app import state
from app.state import CustomerState
from app.schemas import DecisionOutput
from app.services.llm_service import llm_service

from app.utils.conversation import (
    build_conversation_context
)


class DecisionAgent:
    """
    Central operational reasoning agent.
    """

    def __init__(self):

        self.high_risk_intents = [
            "PAYMENT_ISSUE",
            "REFUND_ISSUE"
        ]

    def _apply_business_rules(
        self,
        state: CustomerState
    ):
        """
        Deterministic business rules.
        """

        profile = (
            state.customer_profile or {}
        )

        profile_type = profile.get(
            "profile_type",
            "REGULAR"
        )

        emotion = state.emotion

        intent = (
            state.intent[0]
            if state.intent
            else "UNKNOWN_INTENT"
        )

        # ---------------------------------
        # Angry premium customer
        # ---------------------------------

        if (
            profile_type == "PREMIUM"
            and emotion == "ANGRY"
        ):

            return {
                "decision": "ESCALATE",
                "priority": "HIGH",
                "clarification_needed": False,
                "human_approval_required": False
            }

        # ---------------------------------
        # High-risk customer
        # ---------------------------------

        if profile_type == "HIGH_RISK":

            return {
                "decision": "FRAUD_REVIEW",
                "priority": "HIGH",
                "clarification_needed": False,
                "human_approval_required": True,
                "approval_reason": (
                    "high_risk_customer"
                )
            }

        # ---------------------------------
        # Repeat issue escalation
        # ---------------------------------

        same_intent_count = 0

        for history in state.customer_history:

            if history["intent"] in state.intent:

                same_intent_count += 1

        if (
            same_intent_count >= 2
            and state.emotion in [
                "ANGRY",
                "FRUSTRATED"
            ]
        ):

            return {
                "decision": "ESCALATE",
                "priority": "HIGH",
                "clarification_needed": False,
                "human_approval_required": False
            }

        # ---------------------------------
        # Unknown intent
        # ---------------------------------

        if intent == "UNKNOWN_INTENT":

            return {
                "decision": "CLARIFY",
                "priority": "NORMAL",
                "clarification_needed": True,
                "clarification_question":
                    (
                        "Could you please explain your issue in more detail?"
                    ),
                "human_approval_required": False
            }

        return None

    def _llm_decision(
        self,
        state: CustomerState
    ):
        """
        LLM reasoning layer.
        """

        # ---------------------------------
        # Build conversation context
        # ---------------------------------

        conversation_context = (
            build_conversation_context(
                state.conversation_history
            )
        )

        prompt = f"""
You are a customer support decision engine.

Your job is to determine the correct operational action.
You must also determine whether external company knowledge retrieval is required based on query.

If retrieved company knowledge is required to answer,
set requires_rag=True instead of clarification.

IMPORTANT:
You must also see the conversation context
to understand the customer's journey
and past interactions.

You must evaluate whether the retrieved
documents contain sufficient informationinformation
to answer the customer query.

Decision Rules:

1. RESPOND
- Retrieved documents are relevant
- Enough information is available

2. ESCALATE
- Customer issue requires human intervention
- High-risk or repeated unresolved issue

3. CLARIFY
- More customer information is required
- Missing order ID, payment details, etc.
- Query is ambiguous

4. OUT_OF_SCOPE
- Query is unrelated to ShopSphere support

5. HUMAN_APPROVAL
- Sensitive/high-risk operation

Conversation Context:
{conversation_context}

Customer Query:
{state.query}

Customer Intent:
{state.intent}

Customer Emotion:
{state.emotion}

Retrieved Documents:
{state.retrieved_docs}

Customer Profile:
{state.customer_profile}

Customer History:
{state.customer_history}
"""

        llm = llm_service._create_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )

        structured_llm = (
            llm.with_structured_output(
                DecisionOutput
            )
        )

        response = structured_llm.invoke(
            prompt
        )

        return response

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Step 1: Business rules
            # ---------------------------------

            rule_result = (
                self._apply_business_rules(
                    state
                )
            )

            # ---------------------------------
            # Rule-based decision
            # ---------------------------------

            if rule_result:

                state.decision = (
                    rule_result["decision"]
                )

                state.priority = (
                    rule_result.get(
                        "priority",
                        "NORMAL"
                    )
                )

                state.clarification_needed = (
                    rule_result.get(
                        "clarification_needed",
                        False
                    )
                )

                state.clarification_question = (
                    rule_result.get(
                        "clarification_question"
                    )
                )

                state.human_approval_required = (
                    rule_result.get(
                        "human_approval_required",
                        False
                    )
                )

                state.approval_reason = (
                    rule_result.get(
                        "approval_reason"
                    )
                )

                state.metadata[
                    "decision_source"
                ] = "business_rules"

                return state

            # ---------------------------------
            # Step 2: LLM reasoning
            # ---------------------------------

            decision_output = (
                self._llm_decision(
                    state
                )
            )

            # ---------------------------------
            # Clarification overrides escalation
            # ---------------------------------

            if (
                decision_output
                .clarification_needed
            ):

                decision_output.decision = (
                    "CLARIFY"
                )

            # ---------------------------------
            # Update state
            # ---------------------------------

            state.decision = (
                decision_output.decision
            )

            state.priority = (
                decision_output.priority
            )

            state.clarification_needed = (
                decision_output
                .clarification_needed
            )

            state.clarification_question = (
                decision_output
                .clarification_question
            )

            state.human_approval_required = (
                decision_output
                .human_approval_required
            )

            state.approval_reason = (
                decision_output
                .approval_reason
            )
            state.requires_rag = (
                decision_output
                .requires_rag
            )
            state.metadata[
                "decision_source"
            ] = "llm_reasoning"
            print("\nRequires RAG:")
            print(state.requires_rag)

            return state

        except Exception as e:

            print("\nDecisionAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"DecisionAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.decision = "ESCALATE"

            state.priority = "HIGH"

            state.clarification_needed = False

            return state