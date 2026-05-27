import os

from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()


class LLMService:
    """
    Enterprise-grade centralized LLM orchestration.

    Features:
    - model fallback
    - retries
    - structured outputs
    - tool binding
    - prompt truncation
    """

    def __init__(self):

        self.api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not self.api_key:

            raise ValueError(
                "GROQ_API_KEY missing."
            )

        # ---------------------------------
        # Agent routing
        # ---------------------------------

        self.agent_model_map: Dict[
            str,
            List[str]
        ] = {

            "intent": [

                "llama-3.1-8b-instant",

                "qwen/qwen3-32b"
            ],

            "emotion": [

                "qwen/qwen3-32b",

                "llama-3.1-8b-instant"
            ],

            "decision": [

                "llama-3.3-70b-versatile",

                "qwen/qwen3-32b"
            ],

            "response": [

                "llama-3.3-70b-versatile",

                "qwen/qwen3-32b",

                "llama-3.1-8b-instant"
            ],

            "rag": [

                "llama-3.1-8b-instant",

                "qwen/qwen3-32b"
            ]
        }

        self.max_retries_per_model = 2

        self.max_prompt_chars = 45000

    # ---------------------------------
    # Raw LLM
    # ---------------------------------

    def _create_llm(
        self,
        model_name: str,
        temperature: float
    ):

        return ChatGroq(

            api_key=self.api_key,

            model=model_name,

            temperature=temperature
        )

    # ---------------------------------
    # Prompt truncation
    # ---------------------------------

    def _truncate_prompt(
        self,
        prompt: str
    ) -> str:

        if len(prompt) <= self.max_prompt_chars:

            return prompt

        print(
            "\nPrompt too large. "
            "Applying truncation..."
        )

        return prompt[
            -self.max_prompt_chars:
        ]

    # ---------------------------------
    # Universal invoke pipeline
    # ---------------------------------

    def invoke_with_fallback(

        self,

        agent_name: str,

        prompt: str,

        temperature: float = 0.0,

        structured_output: Optional[
            Type
        ] = None,

        tools: Optional[list] = None
    ):

        models = self.agent_model_map.get(
            agent_name
        )

        if not models:

            raise ValueError(
                f"No models configured "
                f"for {agent_name}"
            )

        prompt = self._truncate_prompt(
            prompt
        )

        all_errors = []

        # ---------------------------------
        # Model fallback loop
        # ---------------------------------

        for model_name in models:

            print(
                f"\nTrying model: "
                f"{model_name}"
            )

            for retry in range(
                self.max_retries_per_model
            ):

                try:

                    llm = self._create_llm(

                        model_name=model_name,

                        temperature=temperature
                    )

                    # ---------------------------------
                    # Bind tools
                    # ---------------------------------

                    if tools:

                        llm = llm.bind_tools(
                            tools
                        )

                    # ---------------------------------
                    # Structured output
                    # ---------------------------------

                    if structured_output:

                        llm = (
                            llm
                            .with_structured_output(
                                structured_output
                            )
                        )

                    # ---------------------------------
                    # Invoke
                    # ---------------------------------

                    response = llm.invoke(
                        prompt
                    )

                    return response

                except Exception as e:

                    error_message = (
                        f"{model_name} "
                        f"(retry {retry+1}): "
                        f"{str(e)}"
                    )

                    print(
                        f"\nLLM ERROR:\n"
                        f"{error_message}"
                    )

                    all_errors.append(
                        error_message
                    )

                    # ---------------------------------
                    # Immediate fallback for
                    # token/context errors
                    # ---------------------------------

                    if any(

                        keyword in str(e).lower()

                        for keyword in [

                            "token",

                            "context",

                            "rate limit",

                            "too large"
                        ]
                    ):

                        print(
                            "\nSwitching "
                            "to fallback model..."
                        )

                        break

        raise RuntimeError(

            f"\nAll fallback models failed.\n"
            f"Errors:\n{all_errors}"
        )


# ---------------------------------
# Singleton
# ---------------------------------

llm_service = LLMService()