import logging
import os
import sys

import psycopg
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("seed_db")

# Add project root to path to read dotenv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

load_dotenv(dotenv_path=os.path.join(project_root, ".env"))


def seed_database():
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "postgres")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")

    logger.info(f"Connecting to database {dbname} on {host}:{port} as {user}...")

    try:
        with psycopg.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        ) as conn:
            with conn.cursor() as cur:
                # 1. Create tables
                logger.info("Creating 'customers' table if it doesn't exist...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS customers (
                        id INT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        loyalty_points INT DEFAULT 0
                    );
                """)

                logger.info("Creating 'orders' table if it doesn't exist...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id VARCHAR(50) PRIMARY KEY,
                        customer_id INT NOT NULL
                            REFERENCES customers(id) ON DELETE CASCADE,
                        order_date DATE NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        total_amount NUMERIC(10, 2) NOT NULL,
                        items JSONB NOT NULL
                    );
                """)

                # 2. Insert seed customer
                logger.info("Inserting seed customer (Jane Doe)...")
                cur.execute("""
                    INSERT INTO customers (id, name, email, loyalty_points)
                    VALUES (1, 'Jane Doe', 'jane.doe@example.com', 150)
                    ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        email = EXCLUDED.email,
                        loyalty_points = EXCLUDED.loyalty_points;
                """)

                # 3. Insert seed orders
                logger.info("Inserting seed orders (#4471, #4472)...")
                cur.execute("""
                    INSERT INTO orders (
                        id, customer_id, order_date, status, total_amount, items
                    )
                    VALUES
                      (
                          '4471', 1, '2026-07-08', 'Shipped', 129.99,
                          '["UltraCharge 100W Adapter", "Braided USB-C Cable 2m"]'
                      ),
                      (
                          '4472', 1, '2026-07-12', 'Processing', 45.00,
                          '["Premium Leather Key Organiser"]'
                      )
                    ON CONFLICT (id) DO UPDATE
                    SET customer_id = EXCLUDED.customer_id,
                        order_date = EXCLUDED.order_date,
                        status = EXCLUDED.status,
                        total_amount = EXCLUDED.total_amount,
                        items = EXCLUDED.items;
                """)

                logger.info("Database successfully seeded!")

    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        logger.error(
            "Please ensure your PostgreSQL container/service is running and "
            "credentials in .env are correct."
        )


if __name__ == "__main__":
    seed_database()
