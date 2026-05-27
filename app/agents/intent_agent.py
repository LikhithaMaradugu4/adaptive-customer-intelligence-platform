import os

import joblib

import numpy as np

from app.state import CustomerState

from app.schemas import IntentOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.context_manager import (
    get_agent_context
)

from app.utils.helpers import (
    is_domain_relevant,
    is_non_support_query
)


class IntentAgent:
    """
    Enterprise-grade hybrid intent agent.

    Pipeline:
    1. Fast ML inference
    2. LLM fallback
    3. Conversational continuity reasoning
    """

    def __init__(self):

        # ---------------------------------
        # ML confidence threshold
        # ---------------------------------

        self.confidence_threshold = 0.72

        # ---------------------------------
        # Resolve model path
        # ---------------------------------

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        )

        MODEL_PATH = os.path.join(

            BASE_DIR,

            "app",

            "models",

            "intent_classifier.pkl"
        )

        # ---------------------------------
        # Load ML pipeline
        # ---------------------------------

        self.pipeline = joblib.load(
            MODEL_PATH
        )

        # ---------------------------------
        # Keyword overrides
        # ---------------------------------

        self.intent_keywords = {

            "REFUND_ISSUE": [

                "refund",

                "money back",

                "refund status"
            ],

            "RETURN_ISSUE": [

                "return",

                "replace",

                "send back"
            ],

            "DELIVERY_ISSUE": [

                "delivery",

                "shipping",

                "late",

                "not delivered"
            ],

            "PAYMENT_ISSUE": [

                "payment",

                "charged",

                "upi",

                "transaction failed"
            ],

            "PRODUCT_ISSUE": [

                "broken",

                "damaged",

                "defective",

                "not working"
            ],

            "ACCOUNT_ISSUE": [

                "login",

                "password",

                "account",

                "sign in"
            ]
        }

    # ---------------------------------
    # Keyword override
    # ---------------------------------

    def _keyword_override(
        self,
        query: str
    ):

        lowered_query = query.lower()

        matched_intents = []

        for intent, keywords in (
            self.intent_keywords.items()
        ):

            if any(

                keyword in lowered_query

                for keyword in keywords
            ):

                matched_intents.append(
                    intent
                )

        if len(matched_intents) == 1:

            return matched_intents[0], 0.95

        if len(matched_intents) > 1:

            return "MULTI_INTENT", 0.90

        return None, None

    # ---------------------------------
    # ML prediction
    # ---------------------------------

    def _predict_with_ml(
        self,
        query: str
    ):

        probabilities = (
            self.pipeline.predict_proba(
                [query]
            )[0]
        )

        predicted_index = np.argmax(
            probabilities
        )

        predicted_intent = (
            self.pipeline.classes_[
                predicted_index
            ]
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        return predicted_intent, confidence

    # ---------------------------------
    # LLM reasoning
    # ---------------------------------

    def _predict_with_llm(
        self,
        query: str,
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

        prompt = f"""
You are ShopSphere's intent engine.

TASK:
Classify customer intent.

Allowed intents:
- PAYMENT_ISSUE
- DELIVERY_ISSUE
- REFUND_ISSUE
- RETURN_ISSUE
- PRODUCT_ISSUE
- ACCOUNT_ISSUE
- NON_SUPPORT
- GENERAL_QUERY
- COMPLAINT
- UNKNOWN_INTENT
- MULTI_INTENT

Rules:
1. Use both current query and conversation context.
2. Resolve references like:
   - it
   - that
   - previous order
   - earlier product

3. MULTI_INTENT only if multiple unrelated issues exist.
4. UNKNOWN_INTENT only if query is unclear.
5. COMPLAINT = dissatisfaction without clear operational request.
6. NON_SUPPORT = greetings, jokes, small talk, unrelated queries.
7. GENERAL_QUERY = informational but non-operational questions.
8. Do not force operational intents for casual queries.
9. Prioritize operational meaning over emotional wording.

Conversation:
{conversation_context}

Current Query:
{query}

Return:
- intent
- confidence (0.0 to 1.0)
"""

        response = (
            llm_service
            .invoke_with_fallback(

                agent_name="intent",

                prompt=prompt,

                temperature=0.0,

                structured_output=IntentOutput
            )
        )

        return (

            response.intent,

            response.confidence
        )

    # ---------------------------------
    # Conversational continuity
    # ---------------------------------

    def _adjust_using_history(
        self,
        current_intent,
        conversation_history
    ):

        if not conversation_history:

            return current_intent

        recent_messages = (
            conversation_history[-6:]
        )

        previous_text = " ".join([

            item.get(
                "message",
                ""
            ).lower()

            for item in recent_messages
        ])

        # ---------------------------------
        # Return continuity
        # ---------------------------------

        if (
            current_intent in [

                "GENERAL_QUERY",

                "NON_SUPPORT"
            ]
            and any(

                keyword in previous_text

                for keyword in [

                    "refund",

                    "return",

                    "replacement"
                ]
            )
        ):

            return "RETURN_ISSUE"

        return current_intent

    # ---------------------------------
    # Main execution
    # ---------------------------------

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            query = state.query

            # ---------------------------------
            # Stage 0
            # Keyword override
            # ---------------------------------

            keyword_intent, keyword_confidence = (
                self._keyword_override(
                    query
                )
            )

            if keyword_intent:

                print(
                    "\nKeyword Override:"
                )

                print(keyword_intent)

                state.intent = [
                    keyword_intent
                ]

                state.intent_confidence = (
                    keyword_confidence
                )

                state.metadata[
                    "intent_source"
                ] = "keyword_override"

                return state

            # ---------------------------------
            # Stage 0.5
            # Non-support short-circuit
            # ---------------------------------

            if (
                is_non_support_query(query)
                and not is_domain_relevant(
                    query,
                    state.intent
                )
            ):

                state.intent = [

                    "NON_SUPPORT"
                ]

                state.intent_confidence = 0.85

                state.metadata[
                    "intent_source"
                ] = "non_support_short_circuit"

                return state

            # ---------------------------------
            # Stage 1
            # ML prediction
            # ---------------------------------

            predicted_intent, confidence = (
                self._predict_with_ml(
                    query
                )
            )

            print("\nML Prediction:")
            print(predicted_intent)

            print("\nConfidence:")
            print(confidence)

            # ---------------------------------
            # High-confidence ML path
            # ---------------------------------

            if (
                confidence >=
                self.confidence_threshold
            ):

                predicted_intent = (
                    self._adjust_using_history(

                        predicted_intent,

                        state.conversation_history
                    )
                )

                state.intent = [
                    predicted_intent
                ]

                state.intent_confidence = (
                    confidence
                )

                state.metadata[
                    "intent_source"
                ] = "ml"

                return state

            # ---------------------------------
            # Stage 2
            # LLM fallback
            # ---------------------------------

            if (
                is_non_support_query(query)
                and not is_domain_relevant(
                    query,
                    state.intent
                )
            ):

                state.intent = [

                    "NON_SUPPORT"
                ]

                state.intent_confidence = 0.75

                state.metadata[
                    "intent_source"
                ] = "non_support_low_confidence"

                return state

            print(
                "\nUsing LLM fallback..."
            )

            intents, llm_confidence = (
                self._predict_with_llm(

                    query,

                    state
                )
            )

            # ---------------------------------
            # Normalize response
            # ---------------------------------

            if isinstance(
                intents,
                str
            ):

                intents = [intents]

            # ---------------------------------
            # Conversational continuity
            # ---------------------------------

            adjusted_intents = []

            for intent in intents:

                adjusted_intents.append(

                    self._adjust_using_history(

                        intent,

                        state.conversation_history
                    )
                )

            state.intent = adjusted_intents

            state.intent_confidence = (
                llm_confidence
            )

            state.metadata[
                "intent_source"
            ] = "llm_fallback"

            return state

        except Exception as e:

            print("\nIntentAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"IntentAgent Error: {str(e)}"
            )

            state.retry_count += 1

            # ---------------------------------
            # Safe fallback
            # ---------------------------------

            state.intent = [
                "UNKNOWN_INTENT"
            ]

            state.intent_confidence = 0.0

            state.metadata[
                "intent_source"
            ] = "safe_fallback"

            return state