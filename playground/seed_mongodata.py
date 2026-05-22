from app.database.mongo import mongodb

collection = mongodb.get_collection("orders")

collection.insert_many(
    [
  {
    "order_id": "ORD_1001",
    "customer_id": "CUST_001",
    "product_id": "PROD_001",
    "product_name": "Samsung Smart TV 55 Inch",
    "order_status": "DELIVERED",
    "payment_status": "PAID",
    "delivery_date": "2026-05-14",
    "amount": 54999,
    "return_eligible": True
  },
  {
    "order_id": "ORD_1002",
    "customer_id": "CUST_001",
    "product_id": "PROD_004",
    "product_name": "Sony Noise Cancelling Headphones",
    "order_status": "SHIPPED",
    "payment_status": "PAID",
    "delivery_date": None,
    "amount": 19999,
    "return_eligible": False
  },
  {
    "order_id": "ORD_1003",
    "customer_id": "CUST_002",
    "product_id": "PROD_002",
    "product_name": "LG Washing Machine 7kg",
    "order_status": "DELIVERED",
    "payment_status": "PAID",
    "delivery_date": "2026-05-10",
    "amount": 28999,
    "return_eligible": False
  },
  {
    "order_id": "ORD_1004",
    "customer_id": "CUST_003",
    "product_id": "PROD_003",
    "product_name": "Apple iPhone 15",
    "order_status": "PROCESSING",
    "payment_status": "PAID",
    "delivery_date": None,
    "amount": 79999,
    "return_eligible": False
  },
  {
    "order_id": "ORD_1005",
    "customer_id": "CUST_004",
    "product_id": "PROD_005",
    "product_name": "Dell Inspiron Laptop",
    "order_status": "RETURN_REQUESTED",
    "payment_status": "PAID",
    "delivery_date": "2026-04-28",
    "amount": 68999,
    "return_eligible": False
  }
]
)

print("MongoDB connected successfully.")