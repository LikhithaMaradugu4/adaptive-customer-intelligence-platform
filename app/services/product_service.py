from app.database.mongo import (
    mongodb
)


class ProductService:

    def __init__(self):

        self.collection = (
            mongodb.get_collection(
                "products"
            )
        )

    def get_product_by_id(
        self,
        product_id: str
    ):

        product = self.collection.find_one(
            {
                "product_id": product_id
            },
            {
                "_id": 0
            }
        )

        return product

    def get_product_by_name(
        self,
        product_name: str
    ):

        product = self.collection.find_one(
            {
                "name": {
                    "$regex": product_name,
                    "$options": "i"
                }
            },
            {
                "_id": 0
            }
        )

        return 
    def get_all_products(
        self
    ):

        products = list(
            self.collection.find(
                {},
                {
                    "_id": 0
                }
            )
        )

        return products


product_service = ProductService()