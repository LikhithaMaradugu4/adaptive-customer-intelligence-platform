from app.state import CustomerState
from app.services.database import (
    database_service
)


class MemoryAgent:
    """
    Customer memory retrieval agent.
    """

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:
            #-----------------------------
            # Skip if history already exists
            #-----------------------------
            if len(state.customer_history) > 0:
                print("\nCustomer history already exists, skipping retrieval.")
                return state

            customer_id = (
                state.customer_id
            )

            history = (
                database_service
                .get_customer_history(
                    customer_id
                )
            )

            state.customer_history = (
                history
            )

            state.metadata[
                "memory_records_found"
            ] = len(history)

            return state

        except Exception as e:

            print("\nMemoryAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"MemoryAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.customer_history = []

            return state