from app.state import CustomerState

from app.schemas import DecisionOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.context_manager import (
    get_agent_context
)

from app.utils.helpers import (
    get_rag_reason,
    should_use_rag
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
        tool_calls,
        state: CustomerState
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

            # ---------------------------------
            # Strict RAG gating
            # ---------------------------------

            if (
                tool_name == "retrieve_policy_documents"
                and not should_use_rag(
                    state.query,
                    state.intent,
                    state.decision,
                    state.intent_confidence
                )
            ):

                state.metadata[
                    "rag_reason"
                ] = get_rag_reason(
                    state.query,
                    state.intent,
                    state.decision,
                    state.intent_confidence,
                    False
                )

                print(
                    f"\nSkipping Tool: "
                    f"{tool_name}"
                )

                continue

            print(
                f"\nExecuting Tool: "
                f"{tool_name}"
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
                    f"\nTool ERROR:\n{str(e)}"
                )

        return tool_outputs

    # ---------------------------------
    # Optimized orchestration rules
    # ---------------------------------

    def _build_system_rules(self):

        return """
You are ShopSphere's orchestration engine.

Responsibilities:
- choose operational action
- orchestrate tools
- minimize unnecessary clarification
- retrieve customer/company context when needed

RULES:

1. Never ask for identity if customer_id exists.

2. Use tools only when operational context is required:
   - orders
   - products
   - deliveries
   - purchases
   - previous chats
   - policies
   - customer history

3. Never use tools for:
   - greetings
   - small talk
   - jokes
   - general knowledge
   - unrelated queries

4. Use retrieve_policy_documents only for:
   - refunds
   - returns
   - cancellations
   - shipping
   - warranty
   - company policy

5. Use commerce tools for:
   - orders
   - deliveries
   - purchased products
   - product lookup

6. Prefer RESPOND whenever enough context exists.

7. Use CLARIFY only if critical information is missing.

8. Escalate only for:
   - fraud risk
   - sensitive operations
   - repeated unresolved frustration
   - operational failure

9. Set requires_rag=True only if external retrieval is still needed.

10. If query is unrelated to ShopSphere:
   - decision=OUT_OF_SCOPE
   - requires_rag=False

11. Never hallucinate policies or operational actions.
"""

    # ---------------------------------
    # Tool orchestration layer
    # ---------------------------------

    def _llm_decision(
        self,
        state: CustomerState
    ):

        # ---------------------------------
        # Lightweight context
        # ---------------------------------

        conversation_context = (
            get_agent_context(
                state
            )
        )

        # ---------------------------------
        # Tool orchestration prompt
        # ---------------------------------

        orchestration_prompt = f"""
{self._build_system_rules()}

TASK:
Determine whether tools are needed before final decision-making.

Customer ID:
{state.customer_id}

Query:
{state.query}

Intent:
{state.intent}

Emotion:
{state.emotion}

Conversation:
{conversation_context}

Retrieved Docs:
{state.retrieved_docs[:2]}

Use tools only if operational/customer/policy context is required.
Skip tools for general or unrelated queries.
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
                    orchestration_response.tool_calls,
                    state
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
Generate the final structured decision.

Allowed Decisions:
- RESPOND
- ESCALATE
- CLARIFY
- OUT_OF_SCOPE
- HUMAN_APPROVAL

Guidelines:
- Prefer RESPOND when sufficient context exists
- Avoid unnecessary clarification
- Escalate only when necessary
- requires_rag=True only if more retrieval is needed

Customer ID:
{state.customer_id}

Query:
{state.query}

Intent:
{state.intent}

Emotion:
{state.emotion}

Tool Outputs:
{tool_outputs}

Retrieved Docs:
{state.retrieved_docs[:2]}
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

                state.metadata[
                    "rag_reason"
                ] = get_rag_reason(
                    state.query,
                    state.intent,
                    state.decision,
                    state.intent_confidence,
                    state.requires_rag
                )

                print("\nRAG Routing Debug:")
                print(f"query: {state.query}")
                print(f"intent: {state.intent}")
                print(f"confidence: {state.intent_confidence}")
                print(f"requires_rag: {state.requires_rag}")
                print(
                    f"rag_reason: "
                    f"{state.metadata.get('rag_reason')}"
                )

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

            rag_allowed = should_use_rag(
                state.query,
                state.intent,
                decision_output.decision,
                state.intent_confidence
            )

            state.requires_rag = (
                decision_output.requires_rag
                and rag_allowed
            )

            state.metadata[
                "rag_reason"
            ] = get_rag_reason(
                state.query,
                state.intent,
                decision_output.decision,
                state.intent_confidence,
                state.requires_rag
            )

            state.metadata[
                "decision_source"
            ] = "llm_orchestration"

            print("\nDecision:")
            print(state.decision)

            print("\nRequires RAG:")
            print(state.requires_rag)

            print("\nRAG Routing Debug:")
            print(f"query: {state.query}")
            print(f"intent: {state.intent}")
            print(f"confidence: {state.intent_confidence}")
            print(f"requires_rag: {state.requires_rag}")

            print(
                f"rag_reason: "
                f"{state.metadata.get('rag_reason')}"
            )

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