import os
from types import SimpleNamespace

import pytest

# Ensure required env vars exist before app imports
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

from app.state import CustomerState
from app.schemas import ResponseOutput


class FakeStructuredLLM:
    def __init__(self, response_obj, capture=None):
        self.response_obj = response_obj
        self.capture = capture

    def invoke(self, prompt):
        if self.capture is not None:
            self.capture["prompt"] = prompt
        return self.response_obj


class FakeLLM:
    def __init__(self, response_obj=None, tool_calls=None, capture=None):
        self.response_obj = response_obj
        self.tool_calls = tool_calls or []
        self.capture = capture

    def with_structured_output(self, _schema):
        return FakeStructuredLLM(
            self.response_obj,
            capture=self.capture,
        )

    def bind_tools(self, _tools):
        return self

    def invoke(self, prompt):
        if self.capture is not None:
            self.capture["prompt"] = prompt
        return SimpleNamespace(tool_calls=self.tool_calls)


@pytest.fixture
def base_state():
    return CustomerState(query="I need help with my order")


@pytest.fixture
def prompt_capture():
    return {}


@pytest.fixture
def fake_llm_response():
    return ResponseOutput(response="Test response")


@pytest.fixture
def fake_llm(prompt_capture, fake_llm_response):
    return FakeLLM(response_obj=fake_llm_response, capture=prompt_capture)
