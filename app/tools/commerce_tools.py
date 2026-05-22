from langchain.tools import tool

from app.services.order_service import (
    order_service
)

from app.services.product_service import (
    product_service
)


@tool
def get_customer_orders(
    customer_id: str
):
    """
    Fetch all customer orders.
    """

    orders = (
        order_service
        .get_customer_orders(
            customer_id
        )
    )

    return orders


@tool
def get_order_by_id(
    order_id: str
):
    """
    Fetch specific order details.
    """

    order = (
        order_service
        .get_order_by_id(
            order_id
        )
    )

    return order


@tool
def get_product_details(
    product_name: str
):
    """
    Fetch product details.
    """

    product = (
        product_service
        .get_product_by_name(
            product_name
        )
    )

    return product
@tool 
def get_all_products(
):
    """
    Fetch all products.
    """

    products = (
        product_service
        .get_all_products()
    )

    return products