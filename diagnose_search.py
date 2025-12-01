"""Diagnose SearchResult structure"""

from arc_saga.storage.sqlite import SQLiteStorage
from arc_saga.models import Message, MessageRole, Provider
import asyncio
from pathlib import Path

async def test_search():
    db_path = Path.home() / ".arc_saga" / "memory.db"
    storage = SQLiteStorage(str(db_path))
    
    # Search with a real query
    results = await storage.search_messages(query="test", limit=5)
    
    print(f"Found {len(results)} results")
    print()
    
    if results:
        first = results[0]
        print("SearchResult attributes:")
        for attr in dir(first):
            if not attr.startswith('_'):
                print(f"   - {attr}: {type(getattr(first, attr))}")
        print()
        
        print("SearchResult values:")
        for attr in ['score', 'snippet']:
            if hasattr(first, attr):
                print(f"   {attr} = {getattr(first, attr)}")
        
        # Check if it has message or is the message itself
        if hasattr(first, 'message'):
            print("   Has 'message' attribute")
            print(f"   message type: {type(first.message)}")
        else:
            print("   No 'message' attribute - might BE the message")
            print(f"   Has 'content': {hasattr(first, 'content')}")
            print(f"   Has 'role': {hasattr(first, 'role')}")

asyncio.run(test_search())