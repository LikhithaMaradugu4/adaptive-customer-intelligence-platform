import pytest
from pydantic import ValidationError

from app.state import CustomerState


def test_customer_state_requires_query():
    with pytest.raises(ValidationError):
        CustomerState()


def test_customer_state_rejects_invalid_conversation_history():
    with pytest.raises(ValidationError):
        CustomerState(query="hi", conversation_history="not-a-list")
