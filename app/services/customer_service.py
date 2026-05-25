from datetime import datetime

from app.database.mongo import mongodb


class CustomerService:
    """
    Customer data access layer.
    """

    def __init__(self):

        self.collection = mongodb.get_collection(
            "customers"
        )

    # ---------------------------------
    # Get customer by ID
    # ---------------------------------

    def get_customer_by_id(
        self,
        customer_id: str
    ):

        customer = self.collection.find_one(
            {
                "customer_id": customer_id
            },
            {
                "_id": 0
            }
        )

        return customer

    # ---------------------------------
    # Create customer
    # ---------------------------------

    def create_customer(
        self,
        customer_id: str
    ):

        customer_data = {
            "customer_id": customer_id,
            "full_name": "New Customer",
            "email": "",
            "phone": "",
            "customer_type": "NEW",
            "account_status": "ACTIVE",
            "total_orders": 0,
            "total_spent": 0,
            "complaint_count": 0,
            "average_satisfaction_score": 0,
            "preferred_category": "",
            "payment_preference": "",
            "language_preference": "English",
            "created_at": datetime.utcnow()
        }

        self.collection.insert_one(
            customer_data
        )

        return customer_data


# ---------------------------------
# Singleton instance
# ---------------------------------

customer_service = CustomerService()