from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

from app.state import CustomerState
from app.services.llm_service import llm_service
from app.schemas import EmotionOutput

from app.utils.conversation import (
    build_conversation_context
)


class EmotionAgent:
    """
    Hybrid Emotion Detection Agent

    Stage 1:
    VADER sentiment analysis

    Stage 2:
    LLM fallback for ambiguous emotions
    """

    def __init__(self):

        self.analyzer = (
            SentimentIntensityAnalyzer()
        )

        self.confidence_threshold = 0.75

    def _predict_with_vader(
        self,
        query: str
    ):
        """
        Predict emotion using VADER.
        """

        scores = self.analyzer.polarity_scores(
            query
        )

        compound = scores["compound"]

        confidence = abs(compound)

        # ---------------------------------
        # Emotion mapping
        # ---------------------------------

        if compound >= 0.5:

            emotion = "SATISFIED"

        elif compound >= 0:

            emotion = "NEUTRAL"

        elif compound >= -0.5:

            emotion = "FRUSTRATED"

        else:

            emotion = "ANGRY"

        return emotion, confidence

    def _predict_with_llm(
        self,
        query: str,
        conversation_history
    ):
        """
        LLM fallback for ambiguous emotion.
        """

        # ---------------------------------
        # Build conversation context
        # ---------------------------------

        conversation_context = (
            build_conversation_context(
                conversation_history
            )
        )

        prompt = f"""
You are an emotion classification system
for customer support.

Classify the customer's emotional state.

Possible emotions:

- ANGRY
- FRUSTRATED
- NEUTRAL
- SATISFIED

IMPORTANT:
You must also analyze the customer's
past conversation context to understand
emotion progression and emotional continuity.

Conversation Context:
{conversation_context}

Current Customer Query:
"{query}"

Return:
- emotion
- confidence
"""

        llm = llm_service.get_llm_for_agent(
            agent_name="emotion",
            temperature=0.0
        )

        structured_llm = (
            llm.with_structured_output(
                EmotionOutput
            )
        )

        response = structured_llm.invoke(
            prompt
        )

        return (
            response.emotion,
            response.confidence
        )

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
            # Stage 1: VADER prediction
            # ---------------------------------

            emotion, confidence = (
                self._predict_with_vader(
                    query
                )
            )

            print("\nEmotion Prediction:")
            print(emotion)
            print(confidence)

            # ---------------------------------
            # High confidence → use VADER
            # ---------------------------------

            if confidence >= self.confidence_threshold:

                state.emotion = emotion

                state.emotion_confidence = (
                    confidence
                )

                state.metadata[
                    "emotion_source"
                ] = "vader"

                return state

            # ---------------------------------
            # Low confidence → LLM fallback
            # ---------------------------------

            print(
                "\nUsing Emotion LLM fallback..."
            )

            emotion, llm_confidence = (
                self._predict_with_llm(
                    query,
                    state.conversation_history
                )
            )

            state.emotion = emotion

            state.emotion_confidence = (
                llm_confidence
            )

            state.metadata[
                "emotion_source"
            ] = "llm_fallback"

            return state

        except Exception as e:

            print("\nEmotionAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"EmotionAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.emotion = "NEUTRAL"

            state.emotion_confidence = 0.0

            return state