from app.state import CustomerState

from app.schemas import ResponseOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.conversation import (
    build_conversation_context
)


class ResponseAgent:
    """
    Enterprise-grade response generation agent.

    Responsibilities:
    - grounded response generation
    - conversational continuity
    - escalation messaging
    - clarification handling
    - emotionally adaptive responses
    """

    def __init__(self):

        self.max_context_docs = 3

    # ---------------------------------
    # Build retrieved context
    # ---------------------------------

    def _build_retrieved_context(
        self,
        state: CustomerState
    ) -> str:

        if not state.retrieved_docs:

            return "No retrieved documents."

        formatted_context = ""

        for idx, doc in enumerate(
            state.retrieved_docs[
                :self.max_context_docs
            ],
            start=1
        ):

            source = doc.get(
                "source",
                "unknown"
            )

            page = doc.get(
                "page",
                "unknown"
            )

            category = doc.get(
                "category",
                "unknown"
            )

            content = doc.get(
                "content",
                ""
            )

            # ---------------------------------
            # Truncate huge chunks
            # ---------------------------------

            content = content[:1200]

            formatted_context += f"""

Document {idx}

Source: {source}
Page: {page}
Category: {category}

Content:
{content}

"""

        return formatted_context

    # ---------------------------------
    # Build tool context
    # ---------------------------------

    def _build_tool_context(
        self,
        state: CustomerState
    ) -> str:

        if not getattr(
            state,
            "tool_outputs",
            None
        ):

            return "No tool outputs."

        formatted_tools = ""

        for idx, tool_output in enumerate(
            state.tool_outputs,
            start=1
        ):

            tool_name = tool_output.get(
                "tool_name",
                "unknown"
            )

            tool_result = tool_output.get(
                "tool_output",
                ""
            )

            formatted_tools += f"""

Tool {idx}

Tool Name:
{tool_name}

Tool Output:
{tool_result}

"""

        return formatted_tools

    # ---------------------------------
    # Build system instructions
    # ---------------------------------

    def _build_system_rules(self):

        return """
You are ShopSphere's enterprise AI customer support assistant.

CORE RESPONSIBILITIES:
- help customers professionally
- maintain conversational continuity
- provide grounded policy-based responses
- avoid hallucinations
- use operational context intelligently

CRITICAL BEHAVIOR RULES:

1. NEVER ignore previous conversation context.

2. Resolve references naturally:
   - it
   - that order
   - earlier product
   - previous issue
   - this item

3. NEVER repeatedly ask for:
   - customer ID
   - order ID
   - already known information

4. If customer asks general policy questions:
   answer directly using knowledge base.

5. Use retrieved documents heavily when available.

6. Use operational tool outputs naturally.

7. Maintain emotional intelligence:
   - empathetic when frustrated
   - reassuring during escalation
   - concise during clarification

8. NEVER hallucinate policies,
timelines, or operational workflows.

9. If retrieved knowledge is insufficient:
   politely ask clarification.

10. Responses should feel:
   - premium
   - conversational
   - intelligent
   - operationally accurate

RESPONSE STYLE:
- concise but informative
- clean formatting
- professional tone
- natural conversational flow
- avoid robotic wording

FORMATTING RULES:
- use bullet points when helpful
- use short paragraphs
- avoid giant text blocks
- make operational steps easy to follow
"""

    # ---------------------------------
    # Generate clarification response
    # ---------------------------------

    def _clarification_flow(
        self,
        state: CustomerState,
        conversation_context: str,
        tool_context: str
    ):

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate a natural clarification response.

Conversation Context:
{conversation_context}

Operational Tool Context:
{tool_context}

Customer Query:
{state.query}

Clarification Needed:
{state.clarification_question}

IMPORTANT:
- avoid repetitive clarification
- maintain continuity
- ask naturally
- sound human
- do not over-explain
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.3,

                structured_output=ResponseOutput
            )
        )

        return result.response

    # ---------------------------------
    # Generate escalation response
    # ---------------------------------

    def _escalation_flow(
        self,
        state: CustomerState,
        conversation_context: str,
        tool_context: str
    ):

        escalation_details = (
            state.escalation_details
            or {}
        )

        assigned_team = (
            escalation_details.get(
                "assigned_team",
                "Support Team"
            )
        )

        ticket_id = (
            escalation_details.get(
                "ticket_id",
                "N/A"
            )
        )

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate a professional escalation response.

Conversation Context:
{conversation_context}

Operational Tool Context:
{tool_context}

Customer Emotion:
{state.emotion}

Decision:
{state.decision}

Assigned Team:
{assigned_team}

Ticket ID:
{ticket_id}

IMPORTANT:
- reassure customer
- acknowledge issue seriously
- maintain trust
- mention escalation clearly
- include ticket ID naturally
- avoid sounding alarming
- keep response concise
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.4,

                structured_output=ResponseOutput
            )
        )

        return result.response

    # ---------------------------------
    # Generate normal grounded response
    # ---------------------------------

    def _normal_response_flow(
        self,
        state: CustomerState,
        conversation_context: str,
        tool_context: str,
        retrieved_context: str
    ):

        prompt = f"""
{self._build_system_rules()}

TASK:
Generate the best possible grounded customer-support response.

Conversation Context:
{conversation_context}

Operational Tool Outputs:
{tool_context}

Customer Query:
{state.query}

Customer Emotion:
{state.emotion}

Customer Intent:
{state.intent}

Retrieved Knowledge Base Context:
{retrieved_context}

IMPORTANT:
- use retrieved documents heavily
- maintain conversational continuity
- answer directly when possible
- avoid unnecessary clarification
- preserve operational correctness
- use retrieved policy timelines accurately
- if retrieved docs contain procedural steps:
  explain them clearly

IF MULTIPLE STEPS EXIST:
format them cleanly using bullets.

IF CUSTOMER IS FRUSTRATED:
be empathetic first before solution.

DO NOT:
- invent policies
- invent timelines
- invent workflows
- repeat same sentence
- sound robotic

If the users asks something  that is not realted to customer support like "What is the weather today?" or "Who won the game last night?", or How to cook pasta?, or general queries out of the Shopshere respond with:
"I'm here to assist with ShopSphere-related support queries.
"""

        result = (
            llm_service
            .invoke_with_fallback(

                agent_name="response",

                prompt=prompt,

                temperature=0.4,

                structured_output=ResponseOutput
            )
        )

        return result.response

    # ---------------------------------
    # Main execution
    # ---------------------------------

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Build contexts
            # ---------------------------------

            conversation_context = (
                build_conversation_context(
                    state.conversation_history
                )
            )

            tool_context = (
                self._build_tool_context(
                    state
                )
            )

            retrieved_context = (
                self._build_retrieved_context(
                    state
                )
            )

            # ---------------------------------
            # Clarification flow
            # ---------------------------------

            if (
                state.clarification_needed
                and state.clarification_question
            ):

                state.response = (
                    self._clarification_flow(

                        state,

                        conversation_context,

                        tool_context
                    )
                )

                state.metadata[
                    "response_source"
                ] = "clarification"

                return state

            # ---------------------------------
            # Out of scope
            # ---------------------------------

            if state.decision == "OUT_OF_SCOPE":

                state.response = (
                    "I can assist only with "
                    "ShopSphere-related support "
                    "queries."
                )

                return state

            # ---------------------------------
            # Escalation flow
            # ---------------------------------

            if state.decision in [

                "ESCALATE",

                "HUMAN_APPROVAL",

                "FRAUD_REVIEW"
            ]:

                state.response = (
                    self._escalation_flow(

                        state,

                        conversation_context,

                        tool_context
                    )
                )

                state.metadata[
                    "response_source"
                ] = "escalation"

                return state

            # ---------------------------------
            # Normal grounded response
            # ---------------------------------

            state.response = (
                self._normal_response_flow(

                    state,

                    conversation_context,

                    tool_context,

                    retrieved_context
                )
            )

            state.metadata[
                "response_source"
            ] = "grounded_llm"

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
                "Please try again shortly."
            )

            return state