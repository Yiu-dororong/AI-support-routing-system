import asyncio
import logging
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


import sys

logger = logging.getLogger("mcp_client_manager")


def _safe_asyncgen_finalizer(gen):
    # Intentional no-op: attempting to close the MCP stdio_client asyncgen
    # from outside its originating task causes anyio to raise:
    #   RuntimeError: Attempted to exit cancel scope in a different task
    # We discard the generator without touching it — the process exit will
    # clean up all file descriptors anyway.
    pass


try:
    sys.set_asyncgen_hooks(finalizer=_safe_asyncgen_finalizer)
except Exception:
    pass


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
                except BaseExceptionGroup:
                    # anyio task group errors during stdio_client teardown — safe to ignore
                    pass
                except (RuntimeError, Exception) as e:
                    logger.debug(f"MCP client shutdown warning (ignored): {e}")
                finally:
                    self._exit_stack = None
                    self.session = None
