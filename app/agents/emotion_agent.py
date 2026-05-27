from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

from app.state import CustomerState

from app.schemas import EmotionOutput

from app.services.llm_service import (
    llm_service
)

from app.utils.context_manager import (
    get_agent_context
)


class EmotionAgent:
    """
    Enterprise-grade hybrid emotion agent.

    Pipeline:
    1. Fast VADER inference
    2. LLM fallback for ambiguity
    3. Emotional continuity tracking
    """

    def __init__(self):

        self.analyzer = (
            SentimentIntensityAnalyzer()
        )

        # ---------------------------------
        # Higher confidence threshold
        # ---------------------------------

        self.confidence_threshold = 0.80

        # ---------------------------------
        # High-risk escalation phrases
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

            "very bad service",

            "scam",

            "fake",

            "cheated"
        ]

        # ---------------------------------
        # Lightweight conversational queries
        # ---------------------------------

        self.casual_queries = [

            "hi",

            "hello",

            "hey",

            "thanks",

            "thank you",

            "okay",

            "ok",

            "yes",

            "no",

            "hmm",

            "good morning",

            "good evening",

            "good afternoon",

            "who are you",

            "do you know my name",

            "can you help",

            "help me"
        ]

    # ---------------------------------
    # High-risk keyword detection
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
    # Casual query detection
    # ---------------------------------

    def _is_casual_query(
        self,
        query: str
    ):

        query = query.strip().lower()

        if query in self.casual_queries:

            return True

        if len(query.split()) <= 3:

            return True

        return False

    # ---------------------------------
    # Fast VADER prediction
    # ---------------------------------

    def _predict_with_vader(
        self,
        query: str
    ):

        # ---------------------------------
        # Casual shortcut
        # ---------------------------------

        if self._is_casual_query(query):

            return "NEUTRAL", 0.95

        scores = (
            self.analyzer
            .polarity_scores(query)
        )

        compound = scores["compound"]

        confidence = abs(compound)

        # ---------------------------------
        # Emotion boundaries
        # ---------------------------------

        if compound >= 0.60:

            emotion = "SATISFIED"

        elif compound >= -0.25:

            emotion = "NEUTRAL"

        elif compound >= -0.65:

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
        state: CustomerState
    ):

        conversation_context = (
            get_agent_context(
                state
            )
        )

        prompt = f"""
You are ShopSphere's emotion engine.

TASK:
Classify customer emotion.

Allowed emotions:
- ANGRY
- FRUSTRATED
- NEUTRAL
- SATISFIED

Rules:
1. Greetings and casual queries are usually NEUTRAL.
2. Use both current query and recent conversation tone.
3. Detect repeated frustration, disappointment, or escalation patterns.
4. Polite wording may still indicate frustration.
5. Appreciation or gratitude -> SATISFIED.
6. Complaints or repeated dissatisfaction -> FRUSTRATED.
7. Aggression, threats, or strong negativity -> ANGRY.
8. Avoid overpredicting frustration.

Conversation:
{conversation_context}

Current Query:
{query}

Return:
- emotion
- confidence (0.0 to 1.0)
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
    # Emotional continuity adjustment
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

        strong_frustration_keywords = [

            "refund",

            "cancel",

            "angry",

            "frustrated",

            "worst",

            "bad service",

            "still not working",

            "again and again",

            "very disappointed",

            "terrible",

            "useless"
        ]

        for item in recent_messages:

            message = (
                item.get(
                    "message",
                    ""
                ).lower()
            )

            if any(

                keyword in message

                for keyword in strong_frustration_keywords
            ):

                frustration_count += 1

        # ---------------------------------
        # Escalate repeated frustration
        # ---------------------------------

        if (
            frustration_count >= 4
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

            query = (
                state.query
                .strip()
            )

            # ---------------------------------
            # High-risk keyword override
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

                    state
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