from langchain.tools import tool

from app.services.customer_service import (
    customer_service
)


@tool
def get_customer_profile(
    customer_id: str
):
    """
    Fetch customer profile
    information.
    """

    customer = (
        customer_service
        .get_customer_by_id(
            customer_id
        )
    )

    return customer