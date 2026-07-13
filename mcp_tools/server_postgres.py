import json
import logging
import os

import psycopg
import psycopg.rows
from mcp.server.fastmcp import FastMCP


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_postgres_server")

mcp = FastMCP("PostgreSQL")


def get_db_connection():
    host = os.environ.get("DB_HOST", "127.0.0.1")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "postgres")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")

    logger.info(f"Connecting to database {dbname} on {host}:{port} as {user}")
    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        row_factory=psycopg.rows.dict_row,
        connect_timeout=1,
    )


@mcp.tool()
def get_order_details(order_id: str, customer_id: int) -> str:
    """
    Retrieve details for a specific customer order, including order date,
    current status, total amount, and purchased items.
    Use for order-specific questions.
    """
    logger.info(
        "Tool get_order_details called for order_id=%s, customer_id=%s",
        order_id,
        customer_id,
    )
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, order_date, status, total_amount, items "
                    "FROM orders WHERE id = %s AND customer_id = %s",
                    (order_id, customer_id),
                )
                row = cur.fetchone()
                if not row:
                    return json.dumps(
                        {"error": "Order not found or unauthorized for this customer."}
                    )

                # Format non-serializable fields
                row["order_date"] = str(row["order_date"])
                row["total_amount"] = float(row["total_amount"])
                return json.dumps(row)
    except Exception as e:
        logger.error(f"Error in get_order_details: {e}")
        return json.dumps({"error": f"Database error: {str(e)}"})


@mcp.tool()
def get_customer_profile(customer_id: int) -> str:
    """
    Fetch authenticated customer account information, including profile details
    and loyalty points.
    Use for account-related questions, not order inquiries.
    """
    logger.info(f"Tool get_customer_profile called for customer_id={customer_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, email, loyalty_points "
                    "FROM customers WHERE id = %s",
                    (customer_id,),
                )
                row = cur.fetchone()
                if not row:
                    return json.dumps({"error": "Customer profile not found."})
                return json.dumps(row)
    except Exception as e:
        logger.error(f"Error in get_customer_profile: {e}")
        return json.dumps({"error": f"Database error: {str(e)}"})


if __name__ == "__main__":
    mcp.run()
