from __future__ import annotations

#!/usr/bin/env python3
"""Database initialization script.

Creates the PostgreSQL database and applies the schema.
"""

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables
load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SCHEMA_FILE = PROJECT_ROOT / "scripts" / "schema.sql"


def parse_database_url(url: str) -> dict[str, str]:
    """Parse DATABASE_URL into components."""
    # Format: postgresql://user:password@host:port/database
    url = url.replace("postgresql://", "")

    # Split credentials and host
    if "@" in url:
        creds, host_part = url.split("@")
        if ":" in creds:
            user, password = creds.split(":", 1)
        else:
            user, password = creds, ""
    else:
        user, password = "", ""
        host_part = url

    # Split host and database
    if "/" in host_part:
        host_port, database = host_part.rsplit("/", 1)
    else:
        host_port, database = host_part, "sports_betting"

    # Split host and port
    if ":" in host_port:
        host, port = host_port.split(":")
    else:
        host, port = host_port, "5432"

    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "database": database,
    }


def create_database(db_config: dict[str, str]) -> bool:
    """Create the database if it doesn't exist."""
    database = db_config["database"]

    # Connect to default postgres database
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database,)
        )
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
            )
            logger.info(f"Created database: {database}")
        else:
            logger.info(f"Database already exists: {database}")

        cursor.close()
        conn.close()
        return True

    except psycopg2.Error as e:
        logger.error(f"Failed to create database: {e}")
        return False


def apply_schema(db_config: dict[str, str]) -> bool:
    """Apply the SQL schema to the database."""
    if not SCHEMA_FILE.exists():
        logger.error(f"Schema file not found: {SCHEMA_FILE}")
        return False

    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
        )
        cursor = conn.cursor()

        # Read and execute schema
        schema_sql = SCHEMA_FILE.read_text()
        cursor.execute(schema_sql)
        conn.commit()

        logger.info("Schema applied successfully")

        # List created tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        logger.info(f"Created {len(tables)} tables:")
        for (table_name,) in tables:
            logger.info(f"  - {table_name}")

        cursor.close()
        conn.close()
        return True

    except psycopg2.Error as e:
        logger.error(f"Failed to apply schema: {e}")
        return False


def verify_setup(db_config: dict[str, str]) -> bool:
    """Verify the database setup is correct."""
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
        )
        cursor = conn.cursor()

        # Check expected tables exist
        expected_tables = [
            "teams", "games", "kalshi_markets", "odds_history",
            "injuries", "weather_conditions", "predictions",
            "bets", "portfolio_snapshots", "api_requests"
        ]

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        actual_tables = {row[0] for row in cursor.fetchall()}

        missing = set(expected_tables) - actual_tables
        if missing:
            logger.warning(f"Missing tables: {missing}")
            return False

        # Check indexes exist
        cursor.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)
        indexes = {row[0] for row in cursor.fetchall()}
        logger.info(f"Found {len(indexes)} indexes")

        cursor.close()
        conn.close()

        logger.info("Database verification passed")
        return True

    except psycopg2.Error as e:
        logger.error(f"Verification failed: {e}")
        return False


@click.command()
@click.option("--drop", is_flag=True, help="Drop existing database before creating")
@click.option("--verify-only", is_flag=True, help="Only verify existing setup")
def main(drop: bool, verify_only: bool) -> None:
    """Initialize the sports betting database."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    db_config = parse_database_url(database_url)
    logger.info(f"Target database: {db_config['database']} on {db_config['host']}:{db_config['port']}")

    if verify_only:
        success = verify_setup(db_config)
        sys.exit(0 if success else 1)

    if drop:
        try:
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                user=db_config["user"],
                password=db_config["password"],
                database="postgres",
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Terminate existing connections
            cursor.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
            """, (db_config["database"],))

            cursor.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(db_config["database"])
                )
            )
            logger.info(f"Dropped database: {db_config['database']}")

            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            logger.error(f"Failed to drop database: {e}")
            sys.exit(1)

    # Create database
    if not create_database(db_config):
        sys.exit(1)

    # Apply schema
    if not apply_schema(db_config):
        sys.exit(1)

    # Verify setup
    if not verify_setup(db_config):
        sys.exit(1)

    logger.info("Database setup complete!")


if __name__ == "__main__":
    main()
