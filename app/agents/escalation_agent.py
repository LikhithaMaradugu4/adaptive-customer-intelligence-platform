# app/agents/escalation_agent.py

import uuid

from app.state import CustomerState


class EscalationAgent:
    """
    Handles escalation workflows.

    Creates escalation metadata,
    assigns priority,
    and prepares human handoff.
    """

    def run(
        self,
        state: CustomerState
    ) -> CustomerState:

        try:

            # ---------------------------------
            # Only process escalations
            # ---------------------------------

            if state.decision not in [
                "ESCALATE",
                "HUMAN_APPROVAL",
                "FRAUD_REVIEW"
            ]:

                return state

            # ---------------------------------
            # Create escalation ticket
            # ---------------------------------

            ticket_id = (
                f"TKT-{uuid.uuid4().hex[:8].upper()}"
            )

            escalation_type = (
                state.decision
            )

            # ---------------------------------
            # Escalation routing
            # ---------------------------------

            if escalation_type == "FRAUD_REVIEW":

                assigned_team = (
                    "Fraud Investigation Team"
                )

            elif escalation_type == "HUMAN_APPROVAL":

                assigned_team = (
                    "Manager Approval Queue"
                )

            else:

                assigned_team = (
                    "Priority Support Team"
                )

            # ---------------------------------
            # Update state
            # ---------------------------------

            state.escalated = True

            state.escalation_details = {

                "ticket_id": ticket_id,

                "escalation_type": escalation_type,

                "assigned_team": assigned_team,

                "priority": state.priority,

                "approval_reason":
                    state.approval_reason
            }

            state.metadata[
                "escalation_created"
            ] = True

            return state

        except Exception as e:

            print("\nEscalationAgent ERROR:")
            print(str(e))

            state.errors.append(
                f"EscalationAgent Error: {str(e)}"
            )

            state.retry_count += 1

            return state