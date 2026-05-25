import os
from typing import Dict, List

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


load_dotenv()


class LLMService:
    """
    Production-grade centralized LLM service.

    Features:
    - Agent-specific model routing
    - Model fallbacks
    - Retry handling
    - Output validation
    """

    def __init__(self):

        self.api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not self.api_key:

            raise ValueError(
                "GROQ_API_KEY missing in .env file"
            )

        # ---------------------------------
        # Agent → ordered model fallbacks
        # ---------------------------------

        self.agent_model_map: Dict[
            str,
            List[str]
        ] = {

            "test": [

                "llama-3.1-8b-instant",

                "qwen/qwen3-32b"
            ],

            "intent": [

                "llama-3.3-70b-versatile",

                "qwen/qwen3-32b"
            ],

            "emotion": [

                "qwen/qwen3-32b",

                "llama-3.3-70b-versatile"
            ],

            "decision": [

                "llama-3.3-70b-versatile",

                "qwen/qwen3-32b"
            ],

            "response": [

                "llama-3.3-70b-versatile",

                "qwen/qwen3-32b"
            ]
        }

        self.max_retries_per_model = 2

    # ---------------------------------
    # Create raw LLM
    # ---------------------------------

    def _create_llm(
        self,
        model_name: str,
        temperature: float
    ) -> ChatGroq:
        """
        Create model instance.
        """

        return ChatGroq(

            api_key=self.api_key,

            model=model_name,

            temperature=temperature
        )

    # ---------------------------------
    # Get LLM for agent
    # ---------------------------------

    def get_llm_for_agent(
        self,
        agent_name: str,
        temperature: float = 0.0
    ) -> ChatGroq:
        """
        Return primary configured model
        for an agent.
        """

        models = self.agent_model_map.get(
            agent_name
        )

        if not models:

            raise ValueError(
                f"No models configured for agent: {agent_name}"
            )

        primary_model = models[0]

        return self._create_llm(

            model_name=primary_model,

            temperature=temperature
        )

    # ---------------------------------
    # Output validation
    # ---------------------------------

    def _validate_output(
        self,
        output: str
    ) -> bool:
        """
        Basic output validation.
        """

        if not output:

            return False

        if not output.strip():

            return False

        return True

    # ---------------------------------
    # Fallback execution
    # ---------------------------------

    def _run_with_fallback(
        self,
        prompt: str,
        agent_name: str,
        temperature: float
    ) -> str:
        """
        Try models in order.
        Retry each model before fallback.
        """

        models = self.agent_model_map.get(
            agent_name
        )

        if not models:

            raise ValueError(
                f"No models configured for agent: {agent_name}"
            )

        all_errors = []

        for model_name in models:

            for retry in range(
                self.max_retries_per_model
            ):

                try:

                    llm = self._create_llm(

                        model_name=model_name,

                        temperature=temperature
                    )

                    response = llm.invoke([

                        HumanMessage(
                            content=prompt
                        )
                    ])

                    output = response.content

                    if self._validate_output(
                        output
                    ):

                        return output

                    all_errors.append(
                        f"{model_name}: Empty output"
                    )

                except Exception as e:

                    all_errors.append(
                        f"{model_name}: {str(e)}"
                    )

        raise RuntimeError(

            f"All models failed for {agent_name}. "
            f"Errors: {all_errors}"
        )

    # ---------------------------------
    # General generation
    # ---------------------------------

    def generate(
        self,
        prompt: str,
        agent_name: str,
        temperature: float = 0.2
    ) -> str:

        return self._run_with_fallback(

            prompt=prompt,

            agent_name=agent_name,

            temperature=temperature
        )

    # ---------------------------------
    # Deterministic classification
    # ---------------------------------

    def classify(
        self,
        prompt: str,
        agent_name: str
    ) -> str:

        return self._run_with_fallback(

            prompt=prompt,

            agent_name=agent_name,

            temperature=0.0
        )


# ---------------------------------
# Singleton
# ---------------------------------

llm_service = LLMService()