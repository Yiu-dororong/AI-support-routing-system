import json
import os

from pydantic import BaseModel, Field

from llm import prompts


class SynthesisResponse(BaseModel):
    cacheable_answer: str = Field(
        description="General knowledge/policy answer reusable across queries (e.g. 30-day policy window)."
    )
    specific_answer: str = Field(
        default="",
        description="Query-specific calculation or context (e.g. 'Your iPhone 11 was bought 20 days ago')."
    )


class ResponseGenerator:
    """
    Handles prompt engineering and final LLM response generation for RAG.
    """

    def __init__(self, synthesis_llm, server_exe, local_model_path):
        self.synthesis_llm = synthesis_llm
        self.server_exe = server_exe
        self.local_model_path = local_model_path
        self.last_synthesis_response: SynthesisResponse | None = None

    def generate(
        self,
        query: str,
        retrieved_docs: list[dict],
        tool_results: dict | None = None,
        callbacks=None,
        metadata: dict = None,
    ) -> tuple[str, str]:
        if not self.synthesis_llm:
            # Fallback if synthesis model is unavailable
            fallback_resp = prompts.LLM_UNAVAILABLE_FALLBACK_HEADER
            for doc in retrieved_docs:
                fallback_resp += (
                    f"**{doc['metadata']['title']}** "
                    f"(Confidence: {doc['similarity']:.2f})\n"
                    f"{doc['content']}\n\n"
                )
            self.last_synthesis_response = SynthesisResponse(
                cacheable_answer=fallback_resp,
                specific_answer=""
            )
            return (
                fallback_resp,
                "LLM model not initialized. Surfaced retrieved documents directly.",
            )

        # Prepare context from retrieved documents
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += (
                f"--- Document {i + 1}: {doc['metadata']['title']} ---\n"
                f"{doc['content']}\n\n"
            )

        # Dynamically append tool results as structured blocks
        tool_context_str = ""
        if tool_results:
            for tool_name, result in tool_results.items():
                display_name = tool_name.replace("_", " ").strip().title()
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
            # Try structured output first if supported by model
            try:
                structured_llm = self.synthesis_llm.with_structured_output(SynthesisResponse)
                synth_obj: SynthesisResponse = structured_llm.invoke(messages, config=config)
                self.last_synthesis_response = synth_obj
                combined = synth_obj.cacheable_answer
                if synth_obj.specific_answer and synth_obj.specific_answer.strip():
                    combined += f"\n\n{synth_obj.specific_answer.strip()}"
                return combined, prompt
            except Exception:
                # Fallback to plain completion
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

                # cacheable_answer stores only the final answer, never the
                # reasoning block, which is query-specific chain-of-thought.
                self.last_synthesis_response = SynthesisResponse(
                    cacheable_answer=content,
                    specific_answer=""
                )
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
            self.last_synthesis_response = SynthesisResponse(
                cacheable_answer=fallback_resp,
                specific_answer=""
            )
            return (
                fallback_resp,
                (
                    "Local model synthesis failed with error: "
                    f"{str(e)}. Surfaced retrieved documents directly."
                ),
            )

