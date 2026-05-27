from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

from app.state import CustomerState

from app.schemas import EmotionOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.conversation import (
    build_conversation_context
)


class EmotionAgent:
    """
    Enterprise-grade hybrid emotion agent.

    Pipeline:
    1. Fast VADER inference
    2. LLM fallback for ambiguity
    3. Conversational emotional continuity

    Goals:
    - low latency
    - emotionally aware support
    - escalation awareness
    - frustration progression tracking
    """

    def __init__(self):

        self.analyzer = (
            SentimentIntensityAnalyzer()
        )

        # ---------------------------------
        # Confidence threshold
        # ---------------------------------

        self.confidence_threshold = 0.72

        # ---------------------------------
        # Strong escalation phrases
        # ---------------------------------

        self.high_risk_keywords = [

            "worst",

            "terrible",

            "horrible",

            "fraud",

            "angry",

            "useless",

            "disappointed",

            "not happy",

            "cancel everything",

            "ridiculous",

            "pathetic",

            "lawsuit",

            "complaint",

            "never again",

            "very bad service"
        ]

    # ---------------------------------
    # Strong keyword override
    # ---------------------------------

    def _contains_high_risk_keywords(
        self,
        query: str
    ):

        lowered_query = query.lower()

        return any(

            keyword in lowered_query

            for keyword in self.high_risk_keywords
        )

    # ---------------------------------
    # VADER prediction
    # ---------------------------------

    def _predict_with_vader(
        self,
        query: str
    ):

        scores = (
            self.analyzer
            .polarity_scores(query)
        )

        compound = scores["compound"]

        confidence = abs(compound)

        # ---------------------------------
        # Emotion mapping
        # ---------------------------------

        if compound >= 0.55:

            emotion = "SATISFIED"

        elif compound >= 0.10:

            emotion = "NEUTRAL"

        elif compound >= -0.45:

            emotion = "FRUSTRATED"

        else:

            emotion = "ANGRY"

        return emotion, confidence

    # ---------------------------------
    # LLM emotional reasoning
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
You are ShopSphere's emotion intelligence engine.

TASK:
Analyze the customer's emotional state.

AVAILABLE EMOTIONS:
- ANGRY
- FRUSTRATED
- NEUTRAL
- SATISFIED

IMPORTANT ANALYSIS RULES:

1. Analyze emotional progression
across conversation history.

2. Detect escalation patterns:
   - repeated complaints
   - disappointment
   - frustration buildup
   - passive aggression

3. Customer may sound polite
while still emotionally frustrated.

4. Use BOTH:
   - current query
   - previous conversational tone

5. If customer expresses:
   - appreciation
   - gratitude
   - satisfaction
   classify as SATISFIED.

6. If customer expresses:
   - irritation
   - repeated dissatisfaction
   - complaint repetition
   classify as FRUSTRATED.

7. If customer expresses:
   - anger
   - threats
   - strong negative wording
   classify as ANGRY.

Conversation Context:
{conversation_context}

Current Customer Query:
{query}

Return:
- emotion
- confidence

Confidence must be between:
0.0 and 1.0
"""

        response = (
            llm_service
            .invoke_with_fallback(

                agent_name="emotion",

                prompt=prompt,

                temperature=0.0,

                structured_output=EmotionOutput
            )
        )

        return (

            response.emotion,

            response.confidence
        )

    # ---------------------------------
    # Emotional continuity upgrade
    # ---------------------------------

    def _adjust_using_history(
        self,
        current_emotion: str,
        conversation_history
    ):

        if not conversation_history:

            return current_emotion

        recent_messages = (
            conversation_history[-6:]
        )

        frustration_count = 0

        for item in recent_messages:

            message = (
                item.get(
                    "message",
                    ""
                ).lower()
            )

            if any(

                keyword in message

                for keyword in [

                    "refund",

                    "again",

                    "still",

                    "not working",

                    "issue",

                    "problem",

                    "bad",

                    "angry",

                    "frustrated"
                ]
            ):

                frustration_count += 1

        # ---------------------------------
        # Escalate neutral → frustrated
        # ---------------------------------

        if (
            frustration_count >= 3
            and current_emotion == "NEUTRAL"
        ):

            return "FRUSTRATED"

        return current_emotion

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
            # Strong keyword override
            # ---------------------------------

            if self._contains_high_risk_keywords(
                query
            ):

                state.emotion = "ANGRY"

                state.emotion_confidence = 0.95

                state.metadata[
                    "emotion_source"
                ] = "keyword_override"

                return state

            # ---------------------------------
            # Fast VADER stage
            # ---------------------------------

            emotion, confidence = (
                self._predict_with_vader(
                    query
                )
            )

            print("\nEmotion Prediction:")
            print(emotion)

            print("\nConfidence:")
            print(confidence)

            # ---------------------------------
            # High-confidence fast path
            # ---------------------------------

            if (
                confidence >=
                self.confidence_threshold
            ):

                emotion = (
                    self._adjust_using_history(

                        emotion,

                        state.conversation_history
                    )
                )

                state.emotion = emotion

                state.emotion_confidence = (
                    confidence
                )

                state.metadata[
                    "emotion_source"
                ] = "vader"

                return state

            # ---------------------------------
            # LLM fallback
            # ---------------------------------

            print(
                "\nUsing LLM emotion fallback..."
            )

            emotion, llm_confidence = (
                self._predict_with_llm(

                    query,

                    state.conversation_history
                )
            )

            # ---------------------------------
            # Historical emotional continuity
            # ---------------------------------

            emotion = (
                self._adjust_using_history(

                    emotion,

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

            # ---------------------------------
            # Safe fallback
            # ---------------------------------

            state.emotion = "NEUTRAL"

            state.emotion_confidence = 0.0

            state.metadata[
                "emotion_source"
            ] = "safe_fallback"

            return state