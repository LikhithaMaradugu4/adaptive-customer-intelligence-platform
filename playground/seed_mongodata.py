from app.database.mongo import (
    mongodb
)

collection = mongodb.get_collection(
    "customers"
)

collection.insert_one(
    {
        "customer_id": "CUST_001",
        "name": "Likhitha"
    }
)

print("MongoDB connected successfully.")