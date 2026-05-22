from app.services.session_service import (
    session_service
)

# ---------------------------------
# Create session
# ---------------------------------

session = session_service.create_session(
    customer_id="CUST_001",
    title="Refund Issue"
)

print("\nSESSION:")
print(session)

# ---------------------------------
# Save messages
# ---------------------------------

session_service.save_message(
    session_id=session["session_id"],
    customer_id="CUST_001",
    role="user",
    message="I need refund help"
)

session_service.save_message(
    session_id=session["session_id"],
    customer_id="CUST_001",
    role="assistant",
    message="Sure, please provide order ID."
)

# ---------------------------------
# Load messages
# ---------------------------------

messages = (
    session_service.get_session_messages(
        session["session_id"]
    )
)

print("\nMESSAGES:")
print(messages)