
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph


async def debug_context_manager():
    db_path = "debug_warden_2.db"
    print(f"Connecting to {db_path} via context manager...")
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
            print("Entered context manager successfully.")
            # Verify we can use it
            config = {"configurable": {"thread_id": "1"}}
            print("Checkpointer ready.")

            # Optionally compile a dummy graph
            print("Compiling dummy graph...")
            builder = StateGraph(dict)
            builder.add_node("node", lambda s: s)
            builder.set_entry_point("node")
            builder.add_edge("node", "node") # infinite loop but just compiling
            app = builder.compile(checkpointer=memory)
            print("Graph compiled.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_context_manager())
