
import asyncio

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def debug_checkpointer():
    db_path = "debug_warden.db"
    print(f"Connecting to {db_path}...")
    try:
        conn = await aiosqlite.connect(db_path)
        print("Connected.")

        memory = AsyncSqliteSaver(conn)
        print("AsyncSqliteSaver instantiated.")

        # Test 1: Check if create_tables exists
        if hasattr(memory, "create_tables"):
            print("Method create_tables exists.")
            await memory.create_tables(memory.conn)
            print("create_tables called successfully.")
        else:
            print("Method create_tables DOES NOT exist.")

        # Test 2: Check if setup exists
        if hasattr(memory, "setup"):
            print("Method setup exists.")
            await memory.setup()
            print("setup called successfully.")
        else:
            print("Method setup DOES NOT exist.")

        await conn.close()
        print("Connection closed.")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(debug_checkpointer())
