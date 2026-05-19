from app.services.database import (
    database_service
)


database_service.save_conversation(
    customer_id="CUST_001",
    query="Refund not received",
    intent="REFUND_ISSUE",
    emotion="ANGRY",
    escalated=True
)

database_service.save_conversation(
    customer_id="CUST_001",
    query="Order delayed again",
    intent="DELIVERY_ISSUE",
    emotion="FRUSTRATED",
    escalated=False
)

print("Memory seeded successfully.")