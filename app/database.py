import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from psycopg import AsyncConnection

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


@asynccontextmanager
async def get_async_connection():
    """Async context manager for database connections."""
    conn = await AsyncConnection.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()
