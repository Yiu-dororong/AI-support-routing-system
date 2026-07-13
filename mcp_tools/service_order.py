import json
import logging

from client_manager import MCPClientManager


logger = logging.getLogger("service_order")


class OrderService:
    """
    Lightweight service layer wrapping PostgreSQL MCP client calls for orders
    and profiles.
    """

    def __init__(self, mcp_client: MCPClientManager):
        self.mcp_client = mcp_client

    async def get_order_details(self, order_id: str, customer_id: int) -> dict:
        try:
            content_list = await self.mcp_client.call_tool(
                "get_order_details",
                {"order_id": order_id, "customer_id": int(customer_id)},
            )
            # Find TextContent in response list
            for content in content_list:
                if hasattr(content, "text"):
                    return json.loads(content.text)
                elif isinstance(content, dict) and "text" in content:
                    return json.loads(content["text"])
                elif hasattr(content, "type") and content.type == "text":
                    return json.loads(content.text)
            return {"error": "Invalid response format from PostgreSQL MCP server"}
        except Exception as e:
            logger.error(f"Error fetching order details via PostgreSQL MCP: {e}")
            return {"error": f"Failed to retrieve order details: {str(e)}"}

    async def get_customer_profile(self, customer_id: int) -> dict:
        try:
            content_list = await self.mcp_client.call_tool(
                "get_customer_profile", {"customer_id": int(customer_id)}
            )
            # Find TextContent in response list
            for content in content_list:
                if hasattr(content, "text"):
                    return json.loads(content.text)
                elif isinstance(content, dict) and "text" in content:
                    return json.loads(content["text"])
                elif hasattr(content, "type") and content.type == "text":
                    return json.loads(content.text)
            return {"error": "Invalid response format from PostgreSQL MCP server"}
        except Exception as e:
            logger.error(f"Error fetching customer profile via PostgreSQL MCP: {e}")
            return {"error": f"Failed to retrieve customer profile: {str(e)}"}
