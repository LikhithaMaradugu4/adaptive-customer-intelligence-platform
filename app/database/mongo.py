import os

from dotenv import load_dotenv

from pymongo import MongoClient

# ---------------------------------
# Load environment variables
# ---------------------------------

load_dotenv()


class MongoDB:

    def __init__(self):

        mongo_uri = os.getenv(
            "MONGO_URI"
        )

        self.client = MongoClient(
            mongo_uri
        )

        self.db = self.client[
            "adaptive_customer_platform"
        ]

    def get_collection(
        self,
        collection_name: str
    ):

        return self.db[
            collection_name
        ]


# ---------------------------------
# Singleton instance
# ---------------------------------

mongodb = MongoDB()