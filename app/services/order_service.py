from app.database.mongo import (
    mongodb
)


class OrderService:

    def __init__(self):

        self.collection = (
            mongodb.get_collection(
                "orders"
            )
        )

    def get_customer_orders(
        self,
        customer_id: str
    ):

        orders = list(
            self.collection.find(
                {
                    "customer_id": customer_id
                },
                {
                    "_id": 0
                }
            )
        )

        return orders

    def get_order_by_id(
        self,
        order_id: str
    ):

        order = self.collection.find_one(
            {
                "order_id": order_id
            },
            {
                "_id": 0
            }
        )

        return order


order_service = OrderService()