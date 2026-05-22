from app.graph import graph
from app.state import CustomerState


def main():

    print("\n================================")
    print(" ShopSphere Support Assistant ")
    print("================================\n")

    # ---------------------------------
    # Persistent session state
    # ---------------------------------

    state = CustomerState(
        customer_id="CUST_001",
        query=""
    )

    while True:

        user_query = input(
            "\nCustomer: "
        )

        # ---------------------------------
        # Exit condition
        # ---------------------------------

        if user_query.lower() in [
            "exit",
            "quit"
        ]:

            print("\nSession ended.\n")
            break

        # ---------------------------------
        # Update query
        # ---------------------------------

        state.query = user_query

        # ---------------------------------
        # Add to conversation history
        # ---------------------------------

        state.conversation_history.append({

            "role": "customer",

            "message": user_query
        })

        # ---------------------------------
        # Run graph
        # ---------------------------------

        result = graph.invoke(
            state.model_dump()
        )

        # ---------------------------------
        # Convert back to state object
        # ---------------------------------

        state = CustomerState(
            **result
        )

        # ---------------------------------
        # Print assistant response
        # ---------------------------------

        print(
            f"\nAssistant: "
            f"{state.response}"
        )

        # ---------------------------------
        # Store assistant response
        # ---------------------------------

        state.conversation_history.append({

            "role": "assistant",

            "message": state.response
        })


if __name__ == "__main__":

    main()