from app.state import CustomerState

from app.schemas import DecisionOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.conversation import (
    build_conversation_context
)

# ---------------------------------
# Tools
# ---------------------------------

from app.tools.memory_tools import (
    get_recent_messages,
    get_customer_sessions,
)

from app.tools.commerce_tools import (
    get_customer_orders,
    get_order_by_id,
    get_product_details,
    get_all_products
)

from app.tools.customer_tools import (
    get_customer_profile
)

from app.tools.retrieval_tools import (
    retrieve_policy_documents
)


class DecisionAgent:
    """
    Central operational reasoning agent.

    Tool-calling orchestration layer.
    """

    def __init__(self):

        self.high_risk_intents = [
            "PAYMENT_ISSUE",
            "REFUND_ISSUE"
        ]

        # ---------------------------------
        # Register tools
        # ---------------------------------

        self.tools = [

            get_recent_messages,

            get_customer_sessions,

            get_customer_profile,

            retrieve_policy_documents,

            get_customer_orders,

            get_order_by_id,

            get_product_details,
            get_all_products
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
        # Premium angry customer
        # ---------------------------------

        if (
            profile_type == "PREMIUM"
            and emotion == "ANGRY"
        ):

            return {
                "decision": "ESCALATE",
                "priority": "HIGH",
                "clarification_needed": False,
                "human_approval_required": False,
                "requires_rag": False
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
                "approval_reason":
                    "high_risk_customer",
                "requires_rag": False
            }

        # ---------------------------------
        # Repeated unresolved issue
        # ---------------------------------

        same_intent_count = 0

        for history in state.customer_history:

            if history.get(
                "intent"
            ) in state.intent:

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
                "human_approval_required": False,
                "requires_rag": False
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
                "human_approval_required": False,
                "requires_rag": False
            }

        return None

    def _execute_tool_calls(
        self,
        tool_calls
    ):
        """
        Execute tool calls dynamically.
        """

        tool_outputs = []

        tool_map = {
            tool.name: tool
            for tool in self.tools
        }

        for tool_call in tool_calls:

            tool_name = (
                tool_call["name"]
            )

            tool_args = (
                tool_call["args"]
            )

            print(
                f"\nExecuting Tool: {tool_name}"
            )

            try:

                tool = tool_map.get(
                    tool_name
                )

                if not tool:

                    continue

                output = tool.invoke(
                    tool_args
                )

                tool_outputs.append({

                    "tool_name": tool_name,

                    "tool_output": output
                })

            except Exception as e:

                print(
                    f"\nTool Error: {str(e)}"
                )

        return tool_outputs

    def _llm_decision(
        self,
        state: CustomerState
    ):
        """
        Tool-calling decision layer.
        """

        # ---------------------------------
        # Conversation context
        # ---------------------------------

        conversation_context = (
            build_conversation_context(
                state.conversation_history
            )
        )

        # ---------------------------------
        # Initial orchestration prompt
        # ---------------------------------

        prompt = f"""
You are an intelligent customer support orchestration agent.

You have access to operational tools.

Your responsibilities:
- understand customer intent
- determine operational action
- decide whether retrieval is needed
- decide whether escalation is needed
- use tools whenever operational memory
  or customer data is required

  If customer asks about:
- orders
- purchased products
- returns
- deliveries
- refunds
- product availability at inventory

use commerce tools.

MANDATORY TOOL RULES:

1. If customer asks about:
   - their identity
   - their name
   - their account
   - previous orders
   - previous chats
   - previous purchases

   MUST use:
   - get_customer_profile
   - get_recent_messages
   - get_customer_sessions

2. If customer asks about:
   - return policy
   - refund policy
   - shipping
   - cancellation
   - warranty
   - company process

   MUST use:
   retrieve_policy_documents

3. Do NOT ask clarification unnecessarily.

4. Prefer answering using:
   - retrieval
   - operational memory
   - tools

5. Clarification should ONLY happen if:
   - information is truly missing
   - query is ambiguous
   - action cannot proceed
IMPORTANT:

Current Customer ID:
{state.customer_id}

If customer_id already exists,
DO NOT ask customer again for identity.
Use profile tools directly.
If they ask about products:
use get_product_details tool.
If they ask about orders:
use get_customer_orders and get_order_by_id (if they specify order id) tools.

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
"""

        # ---------------------------------
        # Create LLM
        # ---------------------------------

        llm = llm_service.get_llm_for_agent(
            agent_name="decision",
            temperature=0.0
        )

        # ---------------------------------
        # Bind tools
        # ---------------------------------

        tool_llm = llm.bind_tools(
            self.tools
        )

        # ---------------------------------
        # First reasoning pass
        # ---------------------------------

        response = tool_llm.invoke(
            prompt
        )

        # ---------------------------------
        # Execute tool calls
        # ---------------------------------

        tool_outputs = []

        if response.tool_calls:

            tool_outputs = (
                self._execute_tool_calls(
                    response.tool_calls
                )
            )

        # ---------------------------------
        # Persist tool outputs
        # ---------------------------------

        state.tool_outputs = (
            tool_outputs
        )

        # ---------------------------------
        # Final reasoning prompt
        # ---------------------------------

        final_prompt = f"""
You are a customer support decision engine.

Generate the FINAL structured operational decision.

IMPORTANT RULES:

1. Prefer RESPOND over CLARIFY
   whenever sufficient knowledge exists.

2. If tool outputs contain:
   - policies
   - operational memory
   - customer context

   then avoid clarification.

3. General informational questions
   should usually become:
   RESPOND

4. Escalation should happen ONLY for:
   - unresolved operational issues
   - high-risk situations
   - repeated failures
   - sensitive actions

5. If retrieval/policy lookup is needed:
   set requires_rag=True
IMPORTANT:

Current Customer ID:
{state.customer_id}

If customer_id already exists,
DO NOT ask customer again for identity.
Use profile tools directly.
If customer asks about:
- orders
- purchased products
- returns
- deliveries
- refunds
- product availability at inventory

use commerce tools.

Decision Types:
- RESPOND
- ESCALATE
- CLARIFY
- OUT_OF_SCOPE
- HUMAN_APPROVAL

Conversation Context:
{conversation_context}

Customer Query:
{state.query}

Customer Intent:
{state.intent}

Customer Emotion:
{state.emotion}

Tool Outputs:
{tool_outputs}

Retrieved Documents:
{state.retrieved_docs}
"""

        structured_llm = (
            llm.with_structured_output(
                DecisionOutput
            )
        )

        final_response = (
            structured_llm.invoke(
                final_prompt
            )
        )

        return final_response

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Step 1: Deterministic rules
            # ---------------------------------

            rule_result = (
                self._apply_business_rules(
                    state
                )
            )

            # ---------------------------------
            # Apply rule-based decision
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

                state.requires_rag = (
                    rule_result.get(
                        "requires_rag",
                        False
                    )
                )

                state.metadata[
                    "decision_source"
                ] = "business_rules"

                return state

            # ---------------------------------
            # Step 2: Tool-calling reasoning
            # ---------------------------------

            decision_output = (
                self._llm_decision(
                    state
                )
            )

            # ---------------------------------
            # Clarification override
            # ---------------------------------

            if (
                decision_output
                .clarification_needed
            ):

                decision_output.decision = (
                    "CLARIFY"
                )

            # ---------------------------------
            # Update runtime state
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
            ] = "tool_calling_llm"

            print("\nDecision:")
            print(state.decision)

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

            state.requires_rag = False

            return state