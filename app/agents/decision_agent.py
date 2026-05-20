from app.state import CustomerState
from app.schemas import DecisionOutput
from app.services.llm_service import llm_service


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
                "human_approval_required": False
            }

        # ---------------------------------
        # High-risk customer
        # ---------------------------------

        if profile_type == "HIGH_RISK":

            return {
                "decision": "FRAUD_REVIEW",
                "priority": "HIGH",
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

        prompt = f"""
You are a customer support decision engine.

Based on the customer situation,
determine the next operational action.

Possible decisions:

- RESPOND
- CLARIFY
- ESCALATE
- HUMAN_APPROVAL
- FRAUD_REVIEW

Customer Intent:
{state.intent}

Customer Emotion:
{state.emotion}

Customer Profile:
{state.customer_profile}

Customer History:
{state.customer_history}

Retrieved Knowledge:
{state.retrieved_docs}
"""

        llm = llm_service._create_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )

        structured_llm = llm.with_structured_output(
            DecisionOutput
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

            state.metadata[
                "decision_source"
            ] = "llm_reasoning"

            return state

        except Exception as e:

            print("\nDecisionAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"DecisionAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.decision = "ESCALATE"

            return state