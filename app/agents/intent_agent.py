import os

import joblib

import numpy as np

from app.state import CustomerState

from app.schemas import IntentOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.conversation import (
    build_conversation_context
)


class IntentAgent:
    """
    Enterprise-grade hybrid intent agent.

    Pipeline:
    1. Fast ML inference
    2. LLM fallback
    3. Conversational continuity reasoning

    Goals:
    - low latency
    - high accuracy
    - multi-intent awareness
    - contextual continuity
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
        # Strong keyword overrides
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
        conversation_history
    ):

        conversation_context = (
            build_conversation_context(
                conversation_history
            )
        )

        prompt = f"""
You are ShopSphere's intelligent intent orchestration engine.

TASK:
Classify the customer's operational intent.

AVAILABLE INTENTS:
- PAYMENT_ISSUE
- DELIVERY_ISSUE
- REFUND_ISSUE
- RETURN_ISSUE
- PRODUCT_ISSUE
- ACCOUNT_ISSUE
- GENERAL_QUERY
- COMPLAINT
- UNKNOWN_INTENT
- MULTI_INTENT

CRITICAL CLASSIFICATION RULES:

1. Use BOTH:
   - current query
   - conversation history

2. Resolve conversational references:
   - it
   - that
   - previous order
   - earlier product

3. Detect workflow continuation:
   Example:
   "Can I return that?"
   should inherit previous product context.

4. MULTI_INTENT:
   use ONLY if customer clearly discusses
   multiple unrelated operational issues.

5. UNKNOWN_INTENT:
   use ONLY if query is genuinely unclear.

6. COMPLAINT:
   use when customer primarily expresses
   dissatisfaction rather than operational action.

7. GENERAL_QUERY:
   informational questions without operational issue.

8. Prioritize operational intent
over emotional wording.

Conversation Context:
{conversation_context}

Current Customer Query:
{query}

Return:
- intent
- confidence

Confidence must be:
0.0 to 1.0
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
            current_intent == "GENERAL_QUERY"
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

            print(
                "\nUsing LLM fallback..."
            )

            intents, llm_confidence = (
                self._predict_with_llm(

                    query,

                    state.conversation_history
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