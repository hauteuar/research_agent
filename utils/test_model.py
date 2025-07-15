#!/usr/bin/env python3
"""
Test script to verify model server is working correctly
"""

import asyncio
import aiohttp
import json
import time

async def test_model_server_direct():
    """Test the model server directly to isolate issues"""
    
    server_url = "http://171.201.3.164:8101"  # Your server URL
    
    print("🧪 Testing Model Server Direct Connection")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health Check
        print("🔍 Test 1: Health Check")
        try:
            async with session.get(f"{server_url}/health") as response:
                print(f"   Status: {response.status}")
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   Response: {json.dumps(health_data, indent=2)}")
                    print("   ✅ Health check passed")
                else:
                    print(f"   ❌ Health check failed with status {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
        
        print()
        
        # Test 2: Server Status
        print("🔍 Test 2: Server Status")
        try:
            async with session.get(f"{server_url}/status") as response:
                print(f"   Status: {response.status}")
                if response.status == 200:
                    status_data = await response.json()
                    print(f"   Model: {status_data.get('model', 'Unknown')}")
                    print(f"   Server ID: {status_data.get('server_id', 'Unknown')}")
                    print(f"   Active Requests: {status_data.get('active_requests', 0)}")
                    print("   ✅ Status check passed")
                else:
                    print(f"   ❌ Status check failed")
        except Exception as e:
            print(f"   ❌ Status check error: {e}")
        
        print()
        
        # Test 3: Simple Generation Request
        print("🔍 Test 3: Simple Generation Request")
        test_data = {
            "prompt": "Hello, world!",
            "max_tokens": 10,
            "temperature": 0.1,
            "stream": False
        }
        
        print(f"   Request: {json.dumps(test_data, indent=2)}")
        
        try:
            start_time = time.time()
            
            async with session.post(
                f"{server_url}/generate",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                end_time = time.time()
                duration = end_time - start_time
                
                print(f"   Status: {response.status}")
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    response_data = await response.json()
                    print(f"   Response keys: {list(response_data.keys())}")
                    print(f"   Generated text: '{response_data.get('text', 'None')}'")
                    print(f"   Server ID: {response_data.get('server_id', 'Unknown')}")
                    print(f"   Finish reason: {response_data.get('finish_reason', 'Unknown')}")
                    print("   ✅ Generation request successful!")
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ Generation failed with status {response.status}")
                    print(f"   Error: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            print("   ❌ Generation request timed out")
            return False
        except Exception as e:
            print(f"   ❌ Generation request error: {type(e).__name__}: {e}")
            return False

async def test_multiple_requests():
    """Test multiple requests to check for consistency"""
    
    server_url = "http://171.201.3.164:8101"
    
    print("\n🔄 Testing Multiple Requests")
    print("=" * 50)
    
    success_count = 0
    total_requests = 3
    
    async with aiohttp.ClientSession() as session:
        for i in range(total_requests):
            print(f"📝 Request {i+1}/{total_requests}")
            
            test_data = {
                "prompt": f"Test prompt number {i+1}:",
                "max_tokens": 5,
                "temperature": 0.1,
                "stream": False
            }
            
            try:
                async with session.post(
                    f"{server_url}/generate",
                    json=test_data,
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"   ✅ Success: '{result.get('text', '')[:50]}...'")
                        success_count += 1
                    else:
                        error = await response.text()
                        print(f"   ❌ Failed: HTTP {response.status} - {error[:100]}...")
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
            # Small delay between requests
            await asyncio.sleep(1)
    
    print(f"\n📊 Results: {success_count}/{total_requests} requests successful")
    return success_count == total_requests

async def main():
    """Run all tests"""
    print("🚀 Model Server Test Suite")
    print("=" * 60)
    
    # Test basic connectivity
    basic_success = await test_model_server_direct()
    
    if basic_success:
        # Test multiple requests
        multi_success = await test_multiple_requests()
        
        if multi_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Your model server is working correctly")
            print("🔧 The issue is likely in the coordinator's request handling")
        else:
            print("\n⚠️  Basic test passed but multiple requests failed")
            print("🔧 Check server stability and resource usage")
    else:
        print("\n❌ BASIC TEST FAILED")
        print("🔧 Fix the model server before testing coordinator")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())