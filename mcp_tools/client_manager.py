import asyncio
import logging
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


logger = logging.getLogger("mcp_client_manager")


class MCPClientManager:
    """
    Manages a persistent stdio connection and ClientSession with an MCP server.
    Uses AsyncExitStack to keep the connection and session alive.
    """

    def __init__(self, command: str, args: list[str], env: dict = None):
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()
        self.session = None
        self._exit_stack = None
        self._lock = asyncio.Lock()

    async def start(self):
        async with self._lock:
            if self.session is not None:
                return

            logger.info(f"Starting MCP server: {self.command} {' '.join(self.args)}")
            server_params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env
            )

            self._exit_stack = AsyncExitStack()
            try:
                # Enter stdio_client context
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )

                # Enter ClientSession context
                self.session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                # Initialize session
                await self.session.initialize()
                logger.info("MCP client session successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to start/initialize MCP client: {e}")
                await self._exit_stack.aclose()
                self._exit_stack = None
                self.session = None
                raise

    async def call_tool(self, tool_name: str, arguments: dict = None) -> list:
        if not self.session:
            await self.start()

        logger.info(f"Calling MCP tool '{tool_name}' with arguments: {arguments}")
        response = await self.session.call_tool(tool_name, arguments or {})
        return response.content

    async def stop(self):
        async with self._lock:
            if self._exit_stack:
                logger.info("Stopping MCP client session.")
                try:
                    await self._exit_stack.aclose()
                except BaseException as e:
                    logger.debug(f"MCP client shutdown warning (ignored): {e}")
                finally:
                    self._exit_stack = None
                    self.session = None
