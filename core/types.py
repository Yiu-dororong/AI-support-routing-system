from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    # PostgreSQL tools
    get_order_details = "get_order_details"
    get_customer_profile = "get_customer_profile"

    # Notion tools
    search_events = "search_events"
    get_event_details = "get_event_details"


class ToolCall(BaseModel):
    tool: ToolName = Field(
        description=(
            "The name of the tool to invoke, chosen from the available tools "
            "listed in the system prompt."
        )
    )
    query: str | None = Field(
        default=None,
        description=(
            "The search argument for the tool, as described in the tool's own entry "
            "in the available tools list. Set to null if not applicable."
        ),
    )


class RoutingDecision(BaseModel):
    path: Literal["refuse", "clarify", "rag", "rag_llm", "escalate"] = Field(
        description=(
            "The routing path for the query. 'refuse' for unsafe/abusive "
            "content, 'clarify' for underspecified/vague queries, 'rag' for "
            "simple factual lookups, 'rag_llm' for queries requiring "
            "reasoning/synthesis, and 'escalate' for human handoff."
        )
    )
    reason: str = Field(description="A brief explanation for this routing decision.")
    clarification_question: str | None = Field(
        default=None,
        description=(
            "If path is 'clarify', the clarification question to ask the user. "
            "Otherwise null/None."
        ),
    )
    refusal_message: str | None = Field(
        default=None,
        description=(
            "If path is 'refuse', the refusal message to show. Otherwise null/None."
        ),
    )
    tools: list[ToolCall] = Field(
        default_factory=list,
        description=(
            "A list of tool calls to run. Only populated and valid when path is "
            "'rag_llm'."
        ),
    )
