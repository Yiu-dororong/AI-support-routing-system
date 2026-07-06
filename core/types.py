from typing import Literal

from pydantic import BaseModel, Field


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
