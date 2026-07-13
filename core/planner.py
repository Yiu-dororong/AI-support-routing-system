from core.types import RoutingDecision
from llm import prompts


class ExecutionPlanner:
    """
    Orchestrates query reasoning and path classification using
    local structured LLM outputs. Contains no retrieval or database access.
    """

    def __init__(self, structured_planner, system_prompt: str | None = None):
        self.structured_planner = structured_planner
        self.system_prompt = system_prompt or prompts.EXECUTION_PLANNER_SYSTEM_PROMPT

    def plan(
        self, query: str, intent: str, callbacks=None, metadata: dict = None
    ) -> tuple[RoutingDecision, str | None]:
        """
        Invokes the structured planner LLM to decide on a RoutingDecision.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=prompts.EXECUTION_PLANNER_USER_TEMPLATE.format(
                    intent=intent, query=query
                )
            ),
        ]

        config = {"callbacks": callbacks}
        if metadata:
            config["metadata"] = metadata
            config["run_name"] = "support_router_query"

        try:
            decision = self.structured_planner.invoke(messages, config=config)

            # Coerce path to 'rag_llm' defensively if tools were emitted
            if (
                hasattr(decision, "tools")
                and decision.tools
                and decision.path != "rag_llm"
            ):
                decision.path = "rag_llm"

            raw_output = decision.model_dump_json(indent=2)
            return decision, raw_output
        except Exception as e:
            # Fallback on failure
            return RoutingDecision(
                path="rag",
                reason=(
                    f"Local execution planning failed with error: {str(e)}. "
                    "Defaulting to standard RAG lookup."
                ),
            ), f"Error details: {str(e)}"
