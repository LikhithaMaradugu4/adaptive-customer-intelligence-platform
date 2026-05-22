from app.database.mongo import (
    mongodb
)


class CustomerService:
    """
    Customer data access layer.
    """

    def __init__(self):

        self.collection = (
            mongodb.get_collection(
                "customers"
            )
        )

    def get_customer_by_id(
        self,
        customer_id: str
    ):

        customer = (
            self.collection.find_one(
                {
                    "customer_id": customer_id
                },
                {
                    "_id": 0
                }
            )
        )

        return customer


customer_service = CustomerService()