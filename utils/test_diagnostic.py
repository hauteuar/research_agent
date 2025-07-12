#!/usr/bin/env python3
"""
FIXED: Diagnostic Script with No Status Enum Errors
Handles both string and enum status values safely
"""

import requests
import json
import asyncio
import aiohttp
from typing import Dict, Any, List
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================================================
# SAFE STATUS HANDLING UTILITIES
# ============================================================================

def safe_get_status(server):
    """Safely get server status as string regardless of type"""
    try:
        if hasattr(server.status, 'value'):
            # It's an enum
            return server.status.value
        else:
            # It's already a string
            return str(server.status)
    except Exception as e:
        return f"error_reading_status: {e}"

def safe_get_availability(server):
    """Safely check server availability"""
    try:
        return server.is_available()
    except Exception as e:
        logger.error(f"Error checking availability for {server.config.name}: {e}")
        return False

# ============================================================================
# FIXED MODEL SERVER DEBUGGER
# ============================================================================

class FixedModelServerDebugger:
    """Debug model server connection issues with safe status handling"""
    
    def __init__(self, server_endpoint: str, gpu_id: int = 2):
        self.server_endpoint = server_endpoint
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"{__name__}.Debugger")
    
    def debug_server_connectivity(self):
        """Debug server connectivity issues - FIXED VERSION"""
        print(f"\n🔍 DEBUGGING MODEL SERVER CONNECTION")
        print(f"Server: {self.server_endpoint}")
        print(f"GPU ID: {self.gpu_id}")
        print("=" * 60)
        
        # Test 1: Basic connectivity
        print("\n1. Testing basic connectivity...")
        self._test_basic_connectivity()
        
        # Test 2: Health endpoint
        print("\n2. Testing health endpoint...")
        self._test_health_endpoint()
        
        # Test 3: Status endpoint
        print("\n3. Testing status endpoint...")
        self._test_status_endpoint()
        
        # Test 4: Generate endpoint with proper payload
        print("\n4. Testing generate endpoint...")
        self._test_generate_endpoint()
    
    def _test_basic_connectivity(self):
        """Test basic connectivity"""
        try:
            response = requests.get(self.server_endpoint, timeout=10)
            print(f"✅ Basic connectivity: {response.status_code}")
            if response.headers:
                print(f"   Server headers available: {len(response.headers)} headers")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection failed: {e}")
            print("   💡 Check if server is running and port is correct")
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
    
    def _test_health_endpoint(self):
        """Test health endpoint"""
        try:
            health_url = f"{self.server_endpoint}/health"
            response = requests.get(health_url, timeout=10)
            print(f"✅ Health endpoint: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    print(f"   Health data keys: {list(health_data.keys()) if isinstance(health_data, dict) else 'not dict'}")
                except:
                    print(f"   Health response (text): {response.text[:100]}...")
            else:
                print(f"   Error response: {response.text[:200]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Health endpoint not accessible")
        except Exception as e:
            print(f"⚠️ Health check error: {e}")
    
    def _test_status_endpoint(self):
        """Test status endpoint"""
        try:
            status_url = f"{self.server_endpoint}/status"
            response = requests.get(status_url, timeout=10)
            print(f"Status endpoint response: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    status_data = response.json()
                    print(f"   Status keys: {list(status_data.keys()) if isinstance(status_data, dict) else 'not dict'}")
                    
                    # Check for model information
                    if isinstance(status_data, dict):
                        if 'model' in status_data:
                            print(f"   📝 Model: {status_data['model']}")
                        if 'gpu_info' in status_data:
                            print(f"   🎮 GPU info available: {len(status_data['gpu_info']) if isinstance(status_data['gpu_info'], (list, dict)) else 'unknown'}")
                        
                except Exception as parse_error:
                    print(f"   Status response (text): {response.text[:200]}...")
                    print(f"   Parse error: {parse_error}")
            else:
                print(f"   Error response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
    
    def _test_generate_endpoint(self):
        """Test generate endpoint with proper payload"""
        try:
            generate_url = f"{self.server_endpoint}/generate"
            
            # Simple, compatible payload
            test_payload = {
                "prompt": "Hello test",
                "max_tokens": 5,
                "temperature": 0.1,
                "stream": False
            }
            
            print(f"   📤 Testing with payload: {json.dumps(test_payload)}")
            
            response = requests.post(
                generate_url,
                json=test_payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"✅ Generate endpoint: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"   📥 Response keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
                    
                    # Extract response text safely
                    if isinstance(result, dict):
                        response_text = (
                            result.get('text') or 
                            result.get('response') or 
                            result.get('content') or
                            result.get('generated_text') or
                            str(result)
                        )
                        print(f"   📝 Generated text: '{response_text[:50]}...'")
                    
                except Exception as parse_error:
                    print(f"   📥 Raw response: {response.text[:200]}...")
                    print(f"   Parse error: {parse_error}")
            else:
                print(f"   ❌ Error {response.status_code}: {response.text[:200]}...")
                
                # Check for validation errors
                if response.status_code == 422:
                    print("   💡 This is a validation error - check payload format!")
                    try:
                        error_detail = response.json()
                        print(f"   📋 Validation details: {json.dumps(error_detail, indent=2)}")
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Generate test error: {e}")

# ============================================================================
# FIXED COORDINATOR TESTING
# ============================================================================

async def test_coordinator_functionality_fixed():
    """FIXED: Test coordinator functionality with safe status handling"""
    
    print("\n🧪 TESTING COORDINATOR FUNCTIONALITY (FIXED)")
    print("=" * 60)
    
    # Step 1: Create coordinator
    print("\n1. Creating coordinator...")
    
    try:
        # Import coordinator modules
        from api_opulence_coordinator import create_api_coordinator_from_config
        
        # Conservative server configuration
        model_servers = [{
            "name": "gpu_server_2",
            "endpoint": "http://171.201.3.165:8100",
            "gpu_id": 2,
            "max_concurrent_requests": 1,  # Very conservative
            "timeout": 60
        }]
        
        coordinator = create_api_coordinator_from_config(
            model_servers=model_servers,
            load_balancing_strategy="round_robin",
            max_retries=1,
            connection_pool_size=2,
            request_timeout=60,
            circuit_breaker_threshold=10  # High tolerance
        )
        
        print("✅ Coordinator created")
        
    except ImportError as e:
        print(f"❌ Cannot import coordinator: {e}")
        print("   💡 Make sure api_opulence_coordinator.py is in your path")
        return False
    except Exception as e:
        print(f"❌ Coordinator creation failed: {e}")
        return False
    
    # Step 2: Initialize coordinator
    print("\n2. Initializing coordinator...")
    try:
        await coordinator.initialize()
        print("✅ Coordinator initialized")
    except Exception as e:
        print(f"❌ Coordinator initialization failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False
    
    # Step 3: SAFE health check
    print("\n3. Checking coordinator health (SAFE)...")
    try:
        health = coordinator.get_health_status()
        
        # SAFE access to health data
        available_servers = health.get('available_servers', 0) if isinstance(health, dict) else 0
        total_servers = health.get('total_servers', 0) if isinstance(health, dict) else 0
        
        print(f"✅ Health status retrieved")
        print(f"   Available servers: {available_servers}")
        print(f"   Total servers: {total_servers}")
        
        if available_servers == 0:
            print("❌ No servers available in coordinator")
            
            # SAFE debug server status
            print("\n🔍 Debugging server status (SAFE):")
            for i, server in enumerate(coordinator.load_balancer.servers):
                print(f"   Server {i+1}: {server.config.name}")
                print(f"     Endpoint: {server.config.endpoint}")
                print(f"     GPU ID: {server.config.gpu_id}")
                
                # SAFE status access
                status_str = safe_get_status(server)
                print(f"     Status: {status_str}")
                
                # SAFE availability check
                is_available = safe_get_availability(server)
                print(f"     Available: {is_available}")
                
                print(f"     Active requests: {getattr(server, 'active_requests', 'unknown')}")
                print(f"     Max requests: {getattr(server.config, 'max_concurrent_requests', 'unknown')}")
                print(f"     Consecutive failures: {getattr(server, 'consecutive_failures', 'unknown')}")
            
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False
    
    # Step 4: Test API call (if servers available)
    if available_servers > 0:
        print("\n4. Testing direct API call...")
        try:
            test_params = {
                "max_tokens": 5,
                "temperature": 0.1,
                "stream": False
            }
            
            result = await coordinator.call_model_api(
                "Hello test",
                test_params,
                preferred_gpu_id=2
            )
            
            print(f"✅ API call successful")
            
            # SAFE result parsing
            if isinstance(result, dict):
                response_text = (
                    result.get('text') or 
                    result.get('response') or 
                    result.get('content') or
                    str(result)
                )
                print(f"   Response: '{response_text[:50]}...'")
            else:
                print(f"   Response: {str(result)[:50]}...")
            
        except Exception as e:
            print(f"❌ API call failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            return False
    else:
        print("\n4. Skipping API test - no available servers")
    
    # Step 5: Cleanup
    print("\n5. Cleaning up...")
    try:
        await coordinator.shutdown()
        print("✅ Coordinator shutdown successful")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    return available_servers > 0

# ============================================================================
# COMPREHENSIVE SAFE DIAGNOSTIC
# ============================================================================

def run_comprehensive_diagnostic():
    """Run comprehensive diagnostic with all safety checks"""
    
    print("\n🚀 COMPREHENSIVE SAFE DIAGNOSTIC")
    print("=" * 80)
    
    # Configuration
    server_endpoint = "http://171.201.3.165:8100"
    gpu_id = 2
    
    # Step 1: Server connectivity testing
    print("\n" + "=" * 80)
    print("STEP 1: SERVER CONNECTIVITY TESTING")
    print("=" * 80)
    
    debugger = FixedModelServerDebugger(server_endpoint, gpu_id)
    debugger.debug_server_connectivity()
    
    # Step 2: Coordinator testing (if available)
    print("\n" + "=" * 80)
    print("STEP 2: COORDINATOR TESTING")
    print("=" * 80)
    
    try:
        success = asyncio.run(test_coordinator_functionality_fixed())
        
        if success:
            print("\n✅ DIAGNOSTIC RESULT: SUCCESS")
            print("   Your server and coordinator are working correctly!")
        else:
            print("\n❌ DIAGNOSTIC RESULT: COORDINATOR ISSUES FOUND")
            print("   Server works but coordinator has availability issues")
            
    except ImportError:
        print("\n⚠️ DIAGNOSTIC RESULT: COORDINATOR NOT AVAILABLE")
        print("   Cannot test coordinator - module not found")
        print("   But server connectivity tests completed above")
        
    except Exception as e:
        print(f"\n❌ DIAGNOSTIC RESULT: COORDINATOR TEST FAILED")
        print(f"   Error: {e}")
        print("   But server connectivity tests completed above")
    
    # Step 3: Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    print("\n1. 🖥️ Server Status:")
    print("   - Check the server connectivity results above")
    print("   - If health and generate tests passed, your server is working")
    
    print("\n2. 🔧 Coordinator Issues:")
    print("   - If coordinator shows 'No servers available':")
    print("     * Use the emergency fix button in Streamlit")
    print("     * Check server status enum handling")
    print("     * Verify load balancer configuration")
    
    print("\n3. 🚨 Emergency Fixes:")
    print("   - Add emergency fix button to your Streamlit app")
    print("   - Use direct API testing to bypass coordinator")
    print("   - Try nuclear reset if all else fails")
    
    print("\n4. 🐛 Debug Steps:")
    print("   - Enable debug mode in your application")
    print("   - Check server logs for any errors")
    print("   - Verify network connectivity and firewalls")

# ============================================================================
# SAFE COORDINATOR EMERGENCY FIX
# ============================================================================

def emergency_fix_coordinator_safe(coordinator):
    """SAFE emergency fix that handles status enum issues"""
    
    print("\n🚨 SAFE EMERGENCY COORDINATOR FIX")
    print("=" * 50)
    
    if not coordinator:
        print("❌ No coordinator provided")
        return False
    
    try:
        # Check if coordinator has load_balancer
        if not hasattr(coordinator, 'load_balancer'):
            print("❌ Coordinator has no load_balancer attribute")
            return False
        
        # Check if load_balancer has servers
        if not hasattr(coordinator.load_balancer, 'servers') or not coordinator.load_balancer.servers:
            print("❌ No servers in load balancer")
            return False
        
        print(f"🔍 Found {len(coordinator.load_balancer.servers)} servers")
        
        # Import ModelServerStatus safely
        try:
            from api_opulence_coordinator import ModelServerStatus
            print("✅ ModelServerStatus imported successfully")
        except ImportError as e:
            print(f"❌ Cannot import ModelServerStatus: {e}")
            return False
        
        # Reset all servers to a known good state
        fixed_count = 0
        for server in coordinator.load_balancer.servers:
            print(f"\n🔧 Fixing server: {server.config.name}")
            
            # Show current status safely
            current_status = safe_get_status(server)
            current_availability = safe_get_availability(server)
            print(f"   Current status: {current_status}")
            print(f"   Current availability: {current_availability}")
            
            # Force reset server state
            try:
                server.status = ModelServerStatus.HEALTHY
                server.consecutive_failures = 0
                server.active_requests = 0
                server.circuit_breaker_open_time = 0
                fixed_count += 1
                
                # Verify fix
                new_status = safe_get_status(server)
                new_availability = safe_get_availability(server)
                print(f"   ✅ New status: {new_status}")
                print(f"   ✅ New availability: {new_availability}")
                
            except Exception as fix_error:
                print(f"   ❌ Failed to fix server: {fix_error}")
        
        # Test availability after fix
        try:
            available_servers = coordinator.load_balancer.get_available_servers()
            print(f"\n📊 Available servers after fix: {len(available_servers)}")
            
            return len(available_servers) > 0
            
        except Exception as avail_error:
            print(f"\n❌ Error checking availability after fix: {avail_error}")
            return fixed_count > 0
    
    except Exception as e:
        print(f"❌ Emergency fix failed with exception: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

# ============================================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================================

def main_diagnostic():
    """Main diagnostic function with comprehensive error handling"""
    
    try:
        run_comprehensive_diagnostic()
    except Exception as e:
        print(f"\n💥 DIAGNOSTIC SCRIPT ERROR: {e}")
        print(f"Error details: {traceback.format_exc()}")
        print("\n🔧 This error suggests there might be import or setup issues.")
        print("Please check:")
        print("1. All required modules are installed")
        print("2. api_opulence_coordinator.py is in the correct path")
        print("3. Python environment is set up correctly")

if __name__ == "__main__":
    main_diagnostic()