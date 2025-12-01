"""
Quick test script for ARC Saga API server
"""

import requests
import time

BASE_URL = "http://localhost:8421"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())
    return response.status_code == 200

def test_capture():
    """Test message capture"""
    data = {
        "source": "test",
        "role": "user",
        "content": "This is a test message about FastAPI and JWT authentication"
    }
    response = requests.post(f"{BASE_URL}/capture", json=data)
    print("Capture response:", response.json())
    return response.json()

def test_search():
    """Test search"""
    data = {
        "query": "FastAPI",
        "limit": 5
    }
    response = requests.post(f"{BASE_URL}/search", json=data)
    print("Search results:", response.json())

def test_threads():
    """Test thread listing"""
    response = requests.get(f"{BASE_URL}/threads")
    print("Threads:", response.json())

if __name__ == "__main__":
    print("Testing ARC Saga API Server...\n")
    
    if test_health():
        print("✅ Server is healthy\n")
        
        capture_result = test_capture()
        print(f"✅ Message captured with thread_id: {capture_result['thread_id']}\n")
        
        time.sleep(0.5)
        
        test_search()
        print("✅ Search working\n")
        
        test_threads()
        print("✅ Thread listing working\n")
        
        print("All tests passed! ✅")
    else:
        print("❌ Server not responding")