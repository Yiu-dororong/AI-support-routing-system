import logging
import os
import platform
import sys

from client_manager import MCPClientManager
from service_event import EventService
from service_order import OrderService


logger = logging.getLogger("mcp_manager")


class MCPServicesContainer:
    """
    Persistent container initialized at server boot time.
    Manages client sessions and exposes order and event services.
    """

    def __init__(self):
        # 1. PostgreSQL MCP Client Server parameters
        # Runs our custom local postgres server in Python using current sys.executable
        postgres_server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mcp_tools",
            "server_postgres.py",
        )
        self.postgres_client = MCPClientManager(
            command=sys.executable, args=[postgres_server_path]
        )

        # 2. Notion MCP Client Server parameters
        # Runs the official Notion MCP server using npx
        npx_cmd = "npx.cmd" if platform.system() == "Windows" else "npx"
        self.notion_client = MCPClientManager(
            command=npx_cmd, args=["-y", "@notionhq/notion-mcp-server"]
        )

        # Initialize Services wrapping the clients
        self.order_service = OrderService(self.postgres_client)
        self.event_service = EventService(self.notion_client)

    async def start(self):
        """Start both MCP client sessions persistently."""
        logger.info("Initializing persistent PostgreSQL MCP connection...")
        try:
            await self.postgres_client.start()
        except Exception as e:
            logger.error(f"PostgreSQL MCP client start failure: {e}")

        # Start Notion MCP connection only if NOTION_TOKEN / NOTION_API_KEY
        # is configured
        if "NOTION_TOKEN" in os.environ or "NOTION_API_KEY" in os.environ:
            # Map NOTION_TOKEN to NOTION_API_KEY if needed by the official Notion server
            if "NOTION_TOKEN" in os.environ and "NOTION_API_KEY" not in os.environ:
                os.environ["NOTION_API_KEY"] = os.environ["NOTION_TOKEN"]

            logger.info("Initializing persistent Notion MCP connection...")
            try:
                await self.notion_client.start()
            except Exception as e:
                logger.error(f"Notion MCP client start failure: {e}")
        else:
            logger.warning(
                "Notion API credentials (NOTION_TOKEN/NOTION_API_KEY) not "
                "detected in environment. Notion MCP client bypass active."
            )

    async def stop(self):
        """Clean up both sessions."""
        logger.info("Cleaning up MCP sessions...")
        await self.postgres_client.stop()
        await self.notion_client.stop()
