"""
ARC Saga Diagnostic Tool
Inspects the current system state and identifies issues
"""

import sys
import inspect
from pathlib import Path

print("=" * 70)
print("🔍 ARC SAGA DIAGNOSTIC REPORT")
print("=" * 70)
print()

# ============================================================================
# 1. CHECK IMPORTS
# ============================================================================
print("1. CHECKING IMPORTS...")
print("-" * 70)

try:
    from arc_saga.models import Message, MessageRole, Provider
    print("✅ Models imported successfully")
    
    # Inspect Message class
    print("\n📋 Message class attributes:")
    for attr in dir(Message):
        if not attr.startswith('_'):
            print(f"   - {attr}")
    
    # Check if it's a dataclass
    import dataclasses
    if dataclasses.is_dataclass(Message):
        print("\n   Message is a dataclass")
        print("   Fields:")
        for field in dataclasses.fields(Message):
            print(f"      - {field.name}: {field.type}")
    
    # Provider enum values
    print("\n📋 Provider enum values:")
    for provider in Provider:
        print(f"   - {provider.name} = {provider.value}")
    
    # MessageRole enum values
    print("\n📋 MessageRole enum values:")
    for role in MessageRole:
        print(f"   - {role.name} = {role.value}")
    
except Exception as e:
    print(f"❌ Models import failed: {e}")
    import traceback
    traceback.print_exc()

print()

try:
    from arc_saga.storage.sqlite import SQLiteStorage
    print("✅ SQLiteStorage imported successfully")
    
    # Inspect SQLiteStorage methods
    print("\n📋 SQLiteStorage public methods:")
    for name, method in inspect.getmembers(SQLiteStorage, predicate=inspect.isfunction):
        if not name.startswith('_'):
            sig = inspect.signature(method)
            print(f"   - {name}{sig}")
    
    # Check if methods are async
    print("\n📋 Async methods:")
    for name, method in inspect.getmembers(SQLiteStorage, predicate=inspect.isfunction):
        if not name.startswith('_') and inspect.iscoroutinefunction(method):
            print(f"   - {name} (async)")
    
except Exception as e:
    print(f"❌ SQLiteStorage import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 2. CHECK DATABASE
# ============================================================================
print("2. CHECKING DATABASE...")
print("-" * 70)

db_path = Path.home() / ".arc_saga" / "memory.db"
print(f"Database path: {db_path}")
print(f"Exists: {db_path.exists()}")

if db_path.exists():
    print(f"Size: {db_path.stat().st_size} bytes")
    
    # Try to connect and inspect
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\n📋 Tables in database:")
        for table in tables:
            print(f"   - {table[0]}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table[0]})")
            columns = cursor.fetchall()
            print(f"     Columns:")
            for col in columns:
                print(f"       - {col[1]} ({col[2]})")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database inspection failed: {e}")

else:
    print("⚠️  Database does not exist yet")

print()

# ============================================================================
# 3. TEST STORAGE INITIALIZATION
# ============================================================================
print("3. TESTING STORAGE INITIALIZATION...")
print("-" * 70)

try:
    from arc_saga.storage.sqlite import SQLiteStorage
    import asyncio
    
    storage = SQLiteStorage(str(db_path))
    print("✅ SQLiteStorage instantiated")
    
    # Check for initialization method
    if hasattr(storage, 'initialize'):
        print("   Found: initialize() method")
        if inspect.iscoroutinefunction(storage.initialize):
            print("   → It's async, attempting to run...")
            try:
                asyncio.run(storage.initialize())
                print("   ✅ initialize() completed successfully")
            except Exception as e:
                print(f"   ❌ initialize() failed: {e}")
        else:
            print("   → It's sync, attempting to run...")
            try:
                storage.initialize()
                print("   ✅ initialize() completed successfully")
            except Exception as e:
                print(f"   ❌ initialize() failed: {e}")
    else:
        print("   ⚠️  No initialize() method found")
        
        # Check for other init methods
        init_methods = [m for m in dir(storage) if 'init' in m.lower() and not m.startswith('_')]
        if init_methods:
            print(f"   Found other init methods: {init_methods}")
    
except Exception as e:
    print(f"❌ Storage initialization test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 4. TEST MESSAGE CREATION
# ============================================================================
print("4. TESTING MESSAGE CREATION...")
print("-" * 70)

try:
    from arc_saga.models import Message, MessageRole, Provider
    from datetime import datetime
    
    # Try to create a Message
    test_message = Message(
        provider=Provider.OPENAI,
        role=MessageRole.USER,
        content="Test message for diagnostic"
    )
    
    print("✅ Message created successfully")
    print(f"   ID: {test_message.id}")
    print(f"   Provider: {test_message.provider}")
    print(f"   Role: {test_message.role}")
    print(f"   Content: {test_message.content[:50]}...")
    
    # Try to convert to dict
    if hasattr(test_message, 'to_dict'):
        msg_dict = test_message.to_dict()
        print(f"   ✅ to_dict() works")
    
except Exception as e:
    print(f"❌ Message creation failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 5. TEST SAVE MESSAGE
# ============================================================================
print("5. TESTING SAVE MESSAGE...")
print("-" * 70)

try:
    from arc_saga.storage.sqlite import SQLiteStorage
    from arc_saga.models import Message, MessageRole, Provider
    import asyncio
    
    storage = SQLiteStorage(str(db_path))
    
    # Initialize if method exists
    if hasattr(storage, 'initialize'):
        if inspect.iscoroutinefunction(storage.initialize):
            asyncio.run(storage.initialize())
        else:
            storage.initialize()
    
    # Create test message
    test_message = Message(
        provider=Provider.OPENAI,
        role=MessageRole.USER,
        content="Diagnostic test message"
    )
    
    # Try to save
    if hasattr(storage, 'save_message'):
        print("   Found: save_message() method")
        
        if inspect.iscoroutinefunction(storage.save_message):
            print("   → It's async, attempting save...")
            message_id = asyncio.run(storage.save_message(test_message))
            print(f"   ✅ Message saved with ID: {message_id}")
        else:
            print("   → It's sync, attempting save...")
            message_id = storage.save_message(test_message)
            print(f"   ✅ Message saved with ID: {message_id}")
    else:
        print("   ⚠️  No save_message() method found")
        save_methods = [m for m in dir(storage) if 'save' in m.lower() and not m.startswith('_')]
        print(f"   Found other save methods: {save_methods}")
    
except Exception as e:
    print(f"❌ Save message test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 6. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("=" * 70)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 70)
print()
print("Next steps based on findings:")
print("1. Review the method signatures above")
print("2. Check if database tables exist")
print("3. Verify async/sync method usage")
print("4. Ensure Message model matches storage expectations")
print()
print("=" * 70)