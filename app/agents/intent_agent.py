import os
import joblib
import numpy as np

from app.state import CustomerState
from app.services.llm_service import llm_service
from app.schemas import IntentOutput


class IntentAgent:
    """
    Hybrid Intent Detection Agent

    Stage 1:
    TF-IDF + Logistic Regression

    Stage 2:
    LLM fallback using structured outputs
    """

    def __init__(self):

        self.confidence_threshold = 0.75

        # ---------------------------------
        # Resolve model path safely
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

    def _predict_with_ml(
        self,
        query: str
    ):
        """
        Predict using TF-IDF + Logistic Regression.
        """

        probabilities = self.pipeline.predict_proba(
            [query]
        )[0]

        predicted_index = np.argmax(
            probabilities
        )

        predicted_intent = self.pipeline.classes_[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        return predicted_intent, confidence

    def _predict_with_llm(
        self,
        query: str
    ):
        """
        Structured LLM fallback.
        """

        prompt = f"""
You are an intent classification system for an e-commerce customer support platform.

Classify the customer query into one or more intents.

Possible intents:

- PAYMENT_ISSUE
- DELIVERY_ISSUE
- REFUND_ISSUE
- PRODUCT_ISSUE
- ACCOUNT_ISSUE
- GENERAL_QUERY
- COMPLAINT
- UNKNOWN_INTENT
- MULTI_INTENT

Rules:
- Return MULTI_INTENT if multiple issues are present.
- Return UNKNOWN_INTENT if unclear.
"""

        llm = llm_service._create_llm(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0
        )

        structured_llm = llm.with_structured_output(
            IntentOutput
        )

        response = structured_llm.invoke(
            prompt + f'\nCustomer Query: "{query}"'
        )

        return response.intent, response.confidence

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:
        """
        Main execution workflow.
        """

        try:

            query = state.query

            # ---------------------------------
            # Stage 1: ML Prediction
            # ---------------------------------

            predicted_intent, confidence = (
                self._predict_with_ml(query)
            )

            print("\nML Prediction:")
            print(predicted_intent)
            print(confidence)

            # ---------------------------------
            # High confidence → use ML
            # ---------------------------------

            if confidence >= self.confidence_threshold:

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
            # Low confidence → use LLM fallback
            # ---------------------------------

            print("\nUsing LLM fallback...")

            intents, llm_confidence = (
                self._predict_with_llm(
                    query
                )
            )

            state.intent = intents

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

            state.intent = [
                "UNKNOWN_INTENT"
            ]

            state.intent_confidence = 0.0

            return state