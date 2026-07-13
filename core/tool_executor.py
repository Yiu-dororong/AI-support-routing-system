import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.types import ToolCall, ToolName


logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    name: str
    description: str


@dataclass
class ToolDefinition:
    name: ToolName
    server: MCPServer
    description: str  # Why to call this tool
    query_description: str  # Format/content expected in 'query'
    handler: Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[ToolName, ToolDefinition] = {}
        self._servers: dict[str, MCPServer] = {}

    def register(
        self,
        name: ToolName,
        server: MCPServer,
        description: str,
        query_description: str,
    ):
        """Decorator to register a tool associated with a specific server."""

        def decorator(func: Callable) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                server=server,
                description=description,
                query_description=query_description,
                handler=func,
            )
            if server.name not in self._servers:
                self._servers[server.name] = server
            return func

        return decorator

    def get_tool_descriptions(self) -> str:
        """
        Renders registered tools grouped by their owning MCPServer.
        Bakes into the Execution Planner system prompt at startup.
        """
        lines = []
        # Group tools by server for cleaner LLM context
        grouped: dict[str, list[ToolDefinition]] = {}
        for tool in self._tools.values():
            grouped.setdefault(tool.server.name, []).append(tool)

        for server_name, tools in grouped.items():
            server = self._servers[server_name]
            lines.append(f"Server: {server.name} ({server.description})")
            for tool in tools:
                lines.append(f"  - tool='{tool.name.value}': {tool.description}")
                lines.append(f"    query: {tool.query_description}")
            lines.append("")
        return "\n".join(lines).strip()

    async def dispatch(
        self, name: ToolName, query: str | None, context: dict[str, Any]
    ) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"No handler registered for tool: {name}")
        return await tool.handler(query, context)


registry = ToolRegistry()

# ----------------- Server Definitions -----------------
postgres_server = MCPServer(
    "PostgreSQL",
    "Database containing live customer transactions, orders, and profiles.",
)
notion_server = MCPServer(
    "Notion", "CMS containing dynamic marketing promotions and operational schedules."
)

# ----------------- Mock Fallback Datasets -----------------
MOCK_ORDERS = {
    "4471": {
        "id": "4471",
        "customer_id": 1,
        "order_date": "2026-07-08",
        "status": "Shipped",
        "total_amount": 129.99,
        "items": ["UltraCharge 100W Adapter", "Braided USB-C Cable 2m"],
    },
    "4472": {
        "id": "4472",
        "customer_id": 1,
        "order_date": "2026-07-12",
        "status": "Processing",
        "total_amount": 45.00,
        "items": ["Premium Leather Key Organiser"],
    },
}

MOCK_CUSTOMERS = {
    1: {
        "id": 1,
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "loyalty_points": 150,
    }
}

MOCK_EVENTS = [
    {
        "id": "39b29cc2-7e57-80d8-944d-e023ac8f1e65",
        "title": "Black Friday",
        "properties": {
            "Active": True,
            "Promo Code": "BFCM2026",
            "Discount Rate": "20% off",
            "Start Date": {"start": "2026-11-27", "end": "2026-12-01"},
        },
        "content": (
            "# Black Friday Sale 2026\n\nWelcome to our Black Friday event. "
            "Promo Code: BFCM2026. Discount: 20% off. "
            "Extended return policy: return until January 31, 2027."
        ),
    },
    {
        "id": "39b29cc2-7e57-8039-8f4b-f797c2a1c29b",
        "title": "Summer Referral",
        "properties": {
            "Active": False,
            "Promo Code": "SUMMER50",
            "Discount Rate": "$50 off",
            "Start Date": {"start": "2026-06-01", "end": "2026-08-31"},
        },
        "content": (
            "# Summer Referral Program 2026\n\nInvite your friends and save. "
            "Promo Code: SUMMER50. Referral credit: $50 off. Status: Inactive."
        ),
    },
]

# ----------------- Tool Handlers -----------------


@registry.register(
    ToolName.get_order_details,
    server=postgres_server,
    description=(
        "Retrieve details for a specific customer order, including order date, "
        "current status, total amount, and purchased items. "
        "Use for order-specific questions."
    ),
    query_description="The order ID digits (e.g. '4471'). Required.",
)
async def execute_get_order_details(
    query: str | None, context: dict[str, Any]
) -> dict[str, Any]:
    if not query:
        return {"error": "Missing order ID parameter"}
    customer_id = context.get("customer_id")
    order_service = context.get("order_service")

    # Try live service
    if order_service and customer_id is not None:
        try:
            res = await order_service.get_order_details(
                order_id=query, customer_id=customer_id
            )
            if "error" not in res:
                return res
        except Exception:
            pass

    # Fallback to mock data
    order = MOCK_ORDERS.get(query)
    if order and order["customer_id"] == customer_id:
        return order
    return {"error": f"Order {query} not found or unauthorized."}


@registry.register(
    ToolName.get_customer_profile,
    server=postgres_server,
    description=(
        "Fetch authenticated customer account information, including profile details, "
        "loyalty points, and purchase history. "
        "Use for account-related questions and purchase history, not order inquiries."
    ),
    query_description=(
        "No query needed; pass null. Profile is fetched via "
        "authenticated session ID."
    ),
)
async def execute_get_customer_profile(
    query: str | None, context: dict[str, Any]
) -> dict[str, Any]:
    customer_id = context.get("customer_id")
    order_service = context.get("order_service")

    # Try live service
    if order_service and customer_id is not None:
        try:
            res = await order_service.get_customer_profile(customer_id=customer_id)
            if "error" not in res:
                return res
        except Exception:
            pass

    # Fallback to mock data
    profile = MOCK_CUSTOMERS.get(customer_id)
    if profile:
        return profile
    return {"error": "Unauthorized session or customer profile not found."}


@registry.register(
    ToolName.search_events,
    server=notion_server,
    description=(
        "Search promotions and operational events by keyword. "
        "Use to discover whether an event exists or find a matching event name. "
        "For full event terms and policies, use get_event_details."
    ),
    query_description=(
        "Keywords to search for "
        "(e.g. 'Black Friday', 'referral', 'double points')."
    ),
)


def _mock_search_events(query: str | None) -> list[dict]:
    """Fallback search against local mock events."""

    if not query:
        return [
            {
                "id": event["id"],
                "title": event["title"],
                "properties": event["properties"],
            }
            for event in MOCK_EVENTS
        ]

    keywords = query.lower().split()
    matches = []

    for event in MOCK_EVENTS:
        searchable_text = (
            f"{event['title']} {event['content']}"
        ).lower()

        if any(keyword in searchable_text for keyword in keywords):
            matches.append(
                {
                    "id": event["id"],
                    "title": event["title"],
                    "properties": event["properties"],
                }
            )

    return matches


async def execute_search_events(
    query: str | None,
    context: dict[str, Any],
) -> Any:
    """
    Search operational events.

    Production:
        Delegate keyword search to EventService (backed by the Notion MCP server).

    Development:
        Fall back to the in-memory mock dataset when no EventService is available.
    """

    event_service = context.get("event_service")

    if event_service:
        try:
            return await event_service.search_events(keyword=query or "")
        except Exception:
            logger.exception(
                "EventService search failed. Falling back to mock events."
            )

    return _mock_search_events(query)

@registry.register(
    ToolName.get_event_details,
    server=notion_server,
    description=(
        "Retrieve complete mechanics of a known event, including discount terms, "
        "applicable categories, promo codes, and return policies."
    ),
    query_description=(
        "The exact event title (e.g. 'Spring Promo Sale 2026', "
        "'Double Points Weekend'). Required — exact match."
    ),
)
async def execute_get_event_details(
    query: str | None, context: dict[str, Any]
) -> dict[str, Any]:
    if not query:
        return {"error": "Missing event title parameter"}
    event_service = context.get("event_service")

    # Try live service
    if event_service:
        try:
            res = await event_service.get_event_details(title=query)
            if "error" not in res:
                return res
        except Exception:
            pass

    # Fallback to mock data
    for ev in MOCK_EVENTS:
        if query.lower() in ev["title"].lower():
            return {
                "title": ev["title"],
                "properties": ev["properties"],
                "content": ev["content"],
            }
    return {"error": f"Event {query} not found."}


# ----------------- Orchestration & Error Handling -----------------


class ToolExecutor:
    """
    Orchestrates parallel execution of selected MCP tools with timeouts
    and partial failure safety.
    Accepts registry as a constructor argument for testability
    (avoids coupling to module-level global).
    """

    def __init__(self, registry: ToolRegistry, timeout_seconds: float = 3.0):
        self.registry = registry
        self.timeout_seconds = timeout_seconds

    async def run_calls(
        self, tool_calls: list[ToolCall], session_context: dict[str, Any]
    ) -> dict[str, Any]:
        if not tool_calls:
            return {}

        tasks = []
        for call in tool_calls:
            task = asyncio.create_task(
                self.registry.dispatch(call.tool, call.query, session_context)
            )
            tasks.append((call.tool.value, task))

        try:
            results_list = await asyncio.wait_for(
                asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error("Overall tool execution timed out! Cancelling pending tasks.")
            for _, task in tasks:
                task.cancel()
            await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            return {t[0]: {"error": "Timeout during execution"} for t in tasks}

        formatted_results = {}
        for (tool_name, _), result in zip(tasks, results_list):
            if isinstance(result, Exception):
                logger.error(f"Error executing tool {tool_name}: {result}")
                formatted_results[tool_name] = {
                    "error": f"Failed to retrieve data: {str(result)}"
                }
            else:
                formatted_results[tool_name] = result

        return formatted_results
