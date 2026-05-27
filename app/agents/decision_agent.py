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
    Enterprise-grade orchestration agent.

    Responsibilities:
    - operational reasoning
    - tool orchestration
    - escalation detection
    - retrieval triggering
    - clarification minimization
    """

    def __init__(self):

        # ---------------------------------
        # High-risk intents
        # ---------------------------------

        self.high_risk_intents = [

            "PAYMENT_ISSUE",

            "REFUND_ISSUE"
        ]

        # ---------------------------------
        # Registered tools
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

    # ---------------------------------
    # Business rules
    # ---------------------------------

    def _apply_business_rules(
        self,
        state: CustomerState
    ):

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
        # Premium angry escalation
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
        # High-risk profile
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
        # Repeat unresolved issue
        # ---------------------------------

        same_intent_count = 0

        for history in state.customer_history:

            if history.get(
                "intent"
            ) in state.intent:

                same_intent_count += 1

        if (
            same_intent_count >= 2
            and emotion in [

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
                        "Could you explain "
                        "your issue in a bit "
                        "more detail?"
                    ),

                "human_approval_required": False,

                "requires_rag": False
            }

        return None

    # ---------------------------------
    # Execute tools
    # ---------------------------------

    def _execute_tool_calls(
        self,
        tool_calls
    ):

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
                f"\nExecuting Tool:"
                f" {tool_name}"
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
                    f"\nTool ERROR:"
                    f"\n{str(e)}"
                )

        return tool_outputs

    # ---------------------------------
    # Build orchestration rules
    # ---------------------------------

    def _build_system_rules(self):

        return """
You are ShopSphere's intelligent orchestration engine.

Your responsibilities:
- determine operational action
- orchestrate tool usage
- reduce unnecessary clarification
- intelligently retrieve customer context
- intelligently retrieve company policy context

CRITICAL BEHAVIOR RULES:

1. NEVER ask for customer identity
if customer_id already exists.

2. Use tools aggressively whenever:
   - customer references past orders
   - customer references previous chats
   - customer references purchases
   - customer references products
   - operational context is required

3. General informational questions
should usually become RESPOND.

4. Clarification should ONLY happen if:
   - information is truly missing
   - operational action cannot proceed
   - query is genuinely ambiguous

5. If customer asks about:
   - returns
   - refunds
   - cancellation
   - shipping
   - warranty
   - company policy

   use retrieve_policy_documents.

6. If customer asks about:
   - orders
   - purchased products
   - deliveries
   - previous purchases

   use commerce tools.

7. Prefer operational grounding
over asking repetitive questions.

8. Avoid over-escalation.

9. Escalation should happen ONLY for:
   - sensitive operations
   - fraud risk
   - unresolved operational failures
   - repeated frustration

10. Use conversation history heavily.

11. If enough information exists:
    choose RESPOND.

12. requires_rag should be TRUE
ONLY if additional external knowledge
retrieval is required.

13. Never hallucinate operational actions.
"""

    # ---------------------------------
    # Tool orchestration layer
    # ---------------------------------

    def _llm_decision(
        self,
        state: CustomerState
    ):

        conversation_context = (
            build_conversation_context(
                state.conversation_history
            )
        )

        # ---------------------------------
        # Tool orchestration prompt
        # ---------------------------------

        orchestration_prompt = f"""
{self._build_system_rules()}

TASK:
Determine whether operational tools
must be used before final decision making.

Current Customer ID:
{state.customer_id}

Conversation Context:
{conversation_context}

Customer Query:
{state.query}

Customer Intent:
{state.intent}

Customer Emotion:
{state.emotion}

Current Retrieved Documents:
{state.retrieved_docs}

IMPORTANT:
Use tools whenever customer context,
order context,
policy context,
or memory context is needed.
"""

        # ---------------------------------
        # Tool-calling inference
        # ---------------------------------

        orchestration_response = (
            llm_service
            .invoke_with_fallback(

                agent_name="decision",

                prompt=orchestration_prompt,

                temperature=0.0,

                tools=self.tools
            )
        )

        # ---------------------------------
        # Execute tool calls
        # ---------------------------------

        tool_outputs = []

        if getattr(
            orchestration_response,
            "tool_calls",
            None
        ):

            tool_outputs = (
                self._execute_tool_calls(
                    orchestration_response.tool_calls
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
{self._build_system_rules()}

TASK:
Generate the FINAL structured decision.

AVAILABLE DECISIONS:
- RESPOND
- ESCALATE
- CLARIFY
- OUT_OF_SCOPE
- HUMAN_APPROVAL

DECISION GUIDELINES:

1. Prefer RESPOND whenever possible.

2. Avoid clarification if:
   - tools already provide context
   - policies already provide answers
   - operational context exists

3. Escalate ONLY if truly needed.

4. If additional company knowledge
must still be retrieved:
set requires_rag=True

5. If retrieved information already
appears sufficient:
set requires_rag=False

Current Customer ID:
{state.customer_id}

Conversation Context:
{conversation_context}

Customer Query:
{state.query}

Customer Intent:
{state.intent}

Customer Emotion:
{state.emotion}

Operational Tool Outputs:
{tool_outputs}

Retrieved Documents:
{state.retrieved_docs}
"""

        # ---------------------------------
        # Structured decision inference
        # ---------------------------------

        final_response = (
            llm_service
            .invoke_with_fallback(

                agent_name="decision",

                prompt=final_prompt,

                temperature=0.0,

                structured_output=DecisionOutput
            )
        )

        return final_response

    # ---------------------------------
    # Main execution
    # ---------------------------------

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Step 1
            # Business rules
            # ---------------------------------

            rule_result = (
                self._apply_business_rules(
                    state
                )
            )

            # ---------------------------------
            # Apply deterministic rules
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
            # Step 2
            # LLM orchestration
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
            ] = "llm_orchestration"

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

            # ---------------------------------
            # Safe fallback behavior
            # ---------------------------------

            state.decision = "ESCALATE"

            state.priority = "HIGH"

            state.clarification_needed = False

            state.requires_rag = False

            return state