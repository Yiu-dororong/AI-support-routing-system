import json
import logging

from client_manager import MCPClientManager


logger = logging.getLogger("service_event")


class EventService:
    """
    Service layer wrapping Notion MCP client calls.
    Translates business capabilities into official Notion MCP server tool invocations.
    """

    def __init__(self, mcp_client: MCPClientManager):
        self.mcp_client = mcp_client

    def _parse_mcp_response(self, content_list) -> dict | list:
        for content in content_list:
            if hasattr(content, "text"):
                return json.loads(content.text)
            elif isinstance(content, dict) and "text" in content:
                return json.loads(content["text"])
            elif hasattr(content, "type") and content.type == "text":
                return json.loads(content.text)
        raise ValueError("Invalid response format from Notion MCP server")

    def _get_page_title(self, page_data: dict) -> str:
        # Extract title from Notion page properties
        properties = page_data.get("properties", {})
        for prop_name, prop_val in properties.items():
            if prop_val.get("type") == "title":
                title_list = prop_val.get("title", [])
                if title_list:
                    return "".join(t.get("plain_text", "") for t in title_list)
        return "Untitled"

    def _serialize_properties(self, properties: dict) -> dict:
        serialized = {}
        for name, val in properties.items():
            if not isinstance(val, dict):
                continue
            p_type = val.get("type")
            if p_type == "checkbox":
                serialized[name] = val.get("checkbox")
            elif p_type == "select":
                sel = val.get("select")
                serialized[name] = sel.get("name") if sel else None
            elif p_type == "multi_select":
                ms = val.get("multi_select", [])
                serialized[name] = [
                    item.get("name") for item in ms if isinstance(item, dict)
                ]
            elif p_type == "date":
                dt = val.get("date")
                serialized[name] = dt if dt else None
            elif p_type == "number":
                serialized[name] = val.get("number")
            elif p_type == "rich_text":
                rt = val.get("rich_text", [])
                serialized[name] = "".join(
                    t.get("plain_text", "") for t in rt if isinstance(t, dict)
                )
            elif p_type == "title":
                t_list = val.get("title", [])
                serialized[name] = "".join(
                    t.get("plain_text", "") for t in t_list if isinstance(t, dict)
                )
            elif p_type == "url":
                serialized[name] = val.get("url")
            elif p_type == "email":
                serialized[name] = val.get("email")
            elif p_type == "phone_number":
                serialized[name] = val.get("phone_number")
        return serialized

    async def search_events(self, keyword: str) -> list[dict]:
        try:
            logger.info(f"Searching events in Notion with keyword='{keyword}'")
            # Call the official Notion search tool (API-post-search)
            content_list = await self.mcp_client.call_tool(
                "API-post-search", {"query": keyword}
            )
            response = self._parse_mcp_response(content_list)

            # Extract results
            results = (
                response if isinstance(response, list) else response.get("results", [])
            )
            events = []
            for item in results:
                if item.get("object") == "page":
                    title = self._get_page_title(item)
                    serialized_props = self._serialize_properties(
                        item.get("properties", {})
                    )
                    events.append(
                        {
                            "id": item.get("id"),
                            "title": title,
                            "properties": serialized_props,
                        }
                    )
            return events
        except Exception as e:
            logger.error(f"Error in search_events via Notion MCP: {e}")
            return [{"error": f"Failed to search Notion events: {str(e)}"}]

    async def get_event_details(self, title: str) -> dict:
        try:
            logger.info(f"Fetching Notion event details for title='{title}'")
            # 1. Search for the page with the title to get its ID
            content_list = await self.mcp_client.call_tool(
                "API-post-search", {"query": title}
            )
            search_response = self._parse_mcp_response(content_list)
            results = (
                search_response
                if isinstance(search_response, list)
                else search_response.get("results", [])
            )

            target_page_id = None
            for item in results:
                if item.get("object") == "page":
                    p_title = self._get_page_title(item)
                    if (
                        p_title.strip().lower() in title.strip().lower()
                        or title.strip().lower() in p_title.strip().lower()
                    ):
                        target_page_id = item.get("id")
                        break

            if not target_page_id:
                # Fallback: take the first page if search matches closely
                for item in results:
                    if item.get("object") == "page":
                        target_page_id = item.get("id")
                        break

            if not target_page_id:
                return {"error": f"Event '{title}' not found in Notion."}

            # 2. Retrieve the page properties
            page_content = await self.mcp_client.call_tool(
                "API-retrieve-a-page", {"page_id": target_page_id}
            )
            page_data = self._parse_mcp_response(page_content)

            # 3. Retrieve page content as Markdown
            markdown_content = await self.mcp_client.call_tool(
                "API-retrieve-page-markdown", {"page_id": target_page_id}
            )
            markdown_data = self._parse_mcp_response(markdown_content)
            body_text = markdown_data.get("markdown", "")

            serialized_props = self._serialize_properties(
                page_data.get("properties", {})
            )

            details = {
                "title": self._get_page_title(page_data),
                "properties": serialized_props,
                "content": body_text,
            }
            return details
        except Exception as e:
            logger.error(f"Error in get_event_details via Notion MCP: {e}")
            return {"error": f"Failed to retrieve Notion event details: {str(e)}"}
