# Custom Domain Setup Guide

This folder contains a starter template to adapt **AI-support-routing-system** to your own custom domain (e.g. IT Helpdesk, SaaS Customer Success, Healthcare, Finance).

---

## 📁 Template Structure

* `domain_config.json`: Defines domain metadata, system persona, and out-of-scope refusal messaging.
* `intents.json`: Defines intent categories and training/centroid sentence examples.
* `faqs.json`: Curated Q&A dataset for exact/fuzzy semantic matching.
* `documents/`: (Optional) Directory containing `.pdf`, `.md`, `.txt`, or `.docx` policy files for vector RAG retrieval.

---

## 🛠️ How to Deploy a New Domain

1. **Copy the Template:**
   Copy the `examples/template/` directory to `data/` (or your custom data directory path):
   ```bash
   cp examples/template/domain_config.json data/domain_config.json
   cp examples/template/intents.json data/intents.json
   cp examples/template/faqs.json data/faqs.json
   ```

2. **Configure Environment Variables in `.env`:**
   Point environment variables to your new domain files:
   ```env
   DOMAIN_CONFIG_FILE="data/domain_config.json"
   INTENTS_FILE="data/intents.json"
   FAQS_FILE="data/faqs.json"
   ```

3. **(Optional) Register Custom Tools:**
   If your domain requires external tools (e.g., querying a CRM or internal DB), register them using the decorator in Python:
   ```python
   from core.tool_executor import registry, MCPServer

   my_server = MCPServer("my_service", "My custom backend API")

   @registry.register(
       name="lookup_user_tier",
       server=my_server,
       description="Look up user subscription tier and status",
       query_description="User email or account ID"
   )
   async def lookup_user_tier(query: str, context: dict):
       # Your custom API logic here
       return {"tier": "Enterprise", "status": "Active"}
   ```

4. **Launch the Router:**
   ```bash
   python app.py
   ```
