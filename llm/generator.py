import json
import os

from llm import prompts


class ResponseGenerator:
    """
    Handles prompt engineering and final LLM response generation for RAG.
    """

    def __init__(self, synthesis_llm, server_exe, local_model_path):
        self.synthesis_llm = synthesis_llm
        self.server_exe = server_exe
        self.local_model_path = local_model_path

    def generate(
        self,
        query: str,
        retrieved_docs: list[dict],
        tool_results: dict | None = None,
        callbacks=None,
        metadata: dict = None,
    ) -> tuple[str, str]:
        if not self.local_model_path or not os.path.exists(self.server_exe):
            # Fallback if local model/binary is missing
            fallback_resp = prompts.LLM_UNAVAILABLE_FALLBACK_HEADER
            for doc in retrieved_docs:
                fallback_resp += (
                    f"**{doc['metadata']['title']}** "
                    f"(Confidence: {doc['similarity']:.2f})\n"
                    f"{doc['content']}\n\n"
                )
            return (
                fallback_resp,
                "Local model not initialized. Surfaced retrieved documents directly.",
            )

        # Prepare context from retrieved documents
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += (
                f"--- Document {i + 1}: {doc['metadata']['title']} ---\n"
                f"{doc['content']}\n\n"
            )

        # Dynamically append tool results as structured blocks
        friendly_tool_names = {
            "get_order_details": "Order Details",
            "get_customer_profile": "Customer Profile",
            "search_events": "Event Search Results",
            "get_event_details": "Event Details",
        }
        tool_context_str = ""
        if tool_results:
            for tool_name, result in tool_results.items():
                display_name = friendly_tool_names.get(
                    tool_name, tool_name.replace("_", " ").title()
                )
                tool_context_str += f"=== {display_name} ===\n"
                if isinstance(result, dict) and "error" in result:
                    tool_context_str += f"Error: {result['error']}\n\n"
                else:
                    tool_context_str += f"{json.dumps(result, indent=2)}\n\n"

        from langchain_core.messages import HumanMessage, SystemMessage

        user_part = f"Retrieved Documents:\n{context_str}\n"
        if tool_context_str:
            user_part += f"External Tool Context:\n{tool_context_str}\n"
        user_part += f"User Query: \"{query}\""

        messages = [
            SystemMessage(content=prompts.RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_part),
        ]

        prompt = (
            f"System:\n{prompts.RESPONSE_SYNTHESIS_SYSTEM_PROMPT}\n\nUser:\n{user_part}"
        )

        config = {"callbacks": callbacks}
        if metadata:
            config["metadata"] = metadata
            config["run_name"] = "support_router_query"

        try:
            response = self.synthesis_llm.invoke(messages, config=config)

            content = response.content
            reasoning = ""
            if hasattr(response, "additional_kwargs"):
                reasoning = response.additional_kwargs.get("reasoning_content", "")
            if not reasoning and response.response_metadata:
                reasoning = response.response_metadata.get("reasoning_content", "")

            if reasoning:
                raw_output = (
                    f"[Start thinking]\n{reasoning}\n[End thinking]\n\n{content}"
                )
            else:
                raw_output = content
            return raw_output, prompt
        except Exception as e:
            # Fallback on failure: Surface supporting articles directly
            fallback_resp = prompts.LLM_SYNTHESIS_FAILED_FALLBACK_HEADER
            for doc in retrieved_docs:
                fallback_resp += (
                    f"**{doc['metadata']['title']}** "
                    f"(Confidence: {doc['similarity']:.2f})\n"
                    f"{doc['content']}\n\n"
                )
            return (
                fallback_resp,
                (
                    "Local model synthesis failed with error: "
                    f"{str(e)}. Surfaced retrieved documents directly."
                ),
            )
