import pandas as pd

from app.state import CustomerState


class ProfileAgent:
    """
    Customer profiling agent.

    Applies adaptive business rules
    to determine customer category.
    """

    def __init__(self):

        self.df = pd.read_csv(
            "data/customers/customer_profiles.csv"
        )

    def _classify_customer(
        self,
        customer_data
    ):
        """
        Adaptive business profiling logic.
        """

        total_spent = float(
            customer_data["total_spent"]
        )

        total_orders = int(
            customer_data["total_orders"]
        )

        complaint_count = int(
            customer_data["complaint_count"]
        )

        satisfaction_score = float(
            customer_data[
                "average_satisfaction_score"
            ]
        )

        account_status = str(
            customer_data["account_status"]
        )

        existing_type = str(
            customer_data["customer_type"]
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
            "preferred_category": customer_data[
                "preferred_category"
            ],
            "payment_preference": customer_data[
                "payment_preference"
            ],
            "language_preference": customer_data[
                "language_preference"
            ]
        }

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            customer_id = (
                state.customer_id
            )

            customer_rows = self.df[
                self.df["customer_id"]
                == customer_id
            ]

            # ---------------------------------
            # Customer not found
            # ---------------------------------

            if customer_rows.empty:

                state.customer_profile = {
                    "profile_type": "NEW"
                }

                state.metadata[
                    "profile_source"
                ] = "default"

                return state

            customer_data = (
                customer_rows.iloc[0]
            )

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
                "profile_type": "UNKNOWN"
            }

            return state