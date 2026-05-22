class MemoryService:
    """
    Customer memory retrieval layer.
    """

    def get_customer_history(
        self,
        customer_id: str
    ):

        # Temporary mock memory

        return [
            {
                "query": "Refund not received",
                "intent": "REFUND_ISSUE",
                "emotion": "ANGRY",
                "escalated": True
            }
        ]


memory_service = MemoryService()