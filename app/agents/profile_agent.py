from app.state import CustomerState

from app.services.customer_service import (
    customer_service
)


class ProfileAgent:
    """
    Customer profiling agent.

    Applies adaptive business rules
    to determine customer category.
    """

    def __init__(self):

        pass

    def _classify_customer(
        self,
        customer_data
    ):
        """
        Adaptive business profiling logic.
        """

        total_spent = float(
            customer_data.get(
                "total_spent",
                0
            )
        )

        total_orders = int(
            customer_data.get(
                "total_orders",
                0
            )
        )

        complaint_count = int(
            customer_data.get(
                "complaint_count",
                0
            )
        )

        satisfaction_score = float(
            customer_data.get(
                "average_satisfaction_score",
                0
            )
        )

        account_status = str(
            customer_data.get(
                "account_status",
                "ACTIVE"
            )
        )

        existing_type = str(
            customer_data.get(
                "customer_type",
                "REGULAR"
            )
        )

        # ---------------------------------
        # Adaptive business rules
        # ---------------------------------

        if (
            account_status.lower()
            != "active"
        ):

            profile_type = "HIGH_RISK"

        elif complaint_count >= 5:

            profile_type = "HIGH_RISK"

        elif (
            total_spent >= 10000
            and satisfaction_score >= 4
        ):

            profile_type = "PREMIUM"

        elif total_orders <= 3:

            profile_type = "NEW"

        else:

            profile_type = "REGULAR"

        return {
            "profile_type": profile_type,
            "existing_profile": existing_type,
            "total_spent": total_spent,
            "total_orders": total_orders,
            "complaint_count": complaint_count,
            "satisfaction_score": satisfaction_score,
            "account_status": account_status,
            "preferred_category": customer_data.get(
                "preferred_category"
            ),
            "payment_preference": customer_data.get(
                "payment_preference"
            ),
            "language_preference": customer_data.get(
                "language_preference"
            ),
            "profile_loaded": True
        }

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Skip if profile already loaded
            # ---------------------------------

            if (
                state.customer_profile
                and state.customer_profile.get(
                    "profile_loaded"
                )
            ):

                print(
                    "\nProfile already exists, skipping profiling."
                )

                return state

            customer_id = (
                state.customer_id
            )

            # ---------------------------------
            # Fetch customer data
            # ---------------------------------

            customer_data = (
                customer_service
                .get_customer_by_id(
                    customer_id
                )
            )

            # ---------------------------------
            # Customer not found
            # ---------------------------------

            if not customer_data:

                state.customer_profile = {
                    "profile_type": "NEW",
                    "profile_loaded": True
                }

                state.metadata[
                    "profile_source"
                ] = "default"

                return state

            # ---------------------------------
            # Generate profile
            # ---------------------------------

            profile = (
                self._classify_customer(
                    customer_data
                )
            )

            state.customer_profile = (
                profile
            )

            state.metadata[
                "profile_source"
            ] = "business_rules"

            return state

        except Exception as e:

            print("\nProfileAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"ProfileAgent Error: {str(e)}"
            )

            state.retry_count += 1

            state.customer_profile = {
                "profile_type": "UNKNOWN",
                "profile_loaded": False
            }

            return state