#!/usr/bin/env python3
"""
Debug and Fix Model Server Connection Issues
Identifies and resolves the "No available model servers" error
"""

import requests
import json
import asyncio
import aiohttp
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================================================
# ISSUE IDENTIFICATION AND FIXES
# ============================================================================

class ModelServerDebugger:
    """Debug and fix model server connection issues"""
    
    def __init__(self, server_endpoint: str, gpu_id: int = 2):
        self.server_endpoint = server_endpoint
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"{__name__}.Debugger")
    
    def debug_server_connectivity(self):
        """Debug server connectivity issues"""
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
        
        # Test 5: Coordinator configuration
        print("\n5. Testing coordinator configuration...")
        self._test_coordinator_config()
    
    def _test_basic_connectivity(self):
        """Test basic connectivity"""
        try:
            response = requests.get(self.server_endpoint, timeout=10)
            print(f"✅ Basic connectivity: {response.status_code}")
            if response.headers:
                print(f"   Headers: {dict(response.headers)}")
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
                    print(f"   Health data: {health_data}")
                except:
                    print(f"   Health response: {response.text}")
            else:
                print(f"   Error response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Health endpoint not accessible")
        except Exception as e:
            print(f"⚠️ Health check error: {e}")
    
    def _test_status_endpoint(self):
        """Test status endpoint"""
        try:
            status_url = f"{self.server_endpoint}/status"
            response = requests.get(status_url, timeout=10)
            print(f"✅ Status endpoint: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    status_data = response.json()
                    print(f"   Status data: {json.dumps(status_data, indent=2)}")
                    
                    # Check for model information
                    if 'model' in status_data:
                        print(f"   📝 Model loaded: {status_data['model']}")
                    if 'gpu_info' in status_data:
                        print(f"   🎮 GPU info: {status_data['gpu_info']}")
                        
                except:
                    print(f"   Status response: {response.text}")
            else:
                print(f"   Error response: {response.text}")
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
    
    def _test_generate_endpoint(self):
        """Test generate endpoint with proper payload"""
        try:
            generate_url = f"{self.server_endpoint}/generate"
            
            # FIXED: Proper payload matching your model server's expected format
            test_payload = {
                "prompt": "Hello, this is a test prompt",
                "max_tokens": 10,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
            
            print(f"   📤 Sending payload: {json.dumps(test_payload, indent=2)}")
            
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
                    print(f"   📥 Response: {json.dumps(result, indent=2)}")
                except:
                    print(f"   📥 Raw response: {response.text}")
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
                # IMPORTANT: Check for validation errors
                if response.status_code == 422:
                    print("   💡 This is a validation error - check payload format!")
                    try:
                        error_detail = response.json()
                        print(f"   📋 Validation details: {json.dumps(error_detail, indent=2)}")
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Generate test error: {e}")
    
    def _test_coordinator_config(self):
        """Test coordinator configuration issues"""
        print("   🔍 Checking coordinator configuration issues...")
        
        # Issue 1: GPU ID mismatch
        print(f"   1. Your GPU ID in config: {self.gpu_id}")
        print(f"   2. Server endpoint: {self.server_endpoint}")
        print("   💡 Make sure GPU ID matches your server configuration")
        
        # Issue 2: Load balancer configuration
        print("   3. Load balancer will mark server unavailable if:")
        print("      - Health check fails")
        print("      - Too many consecutive failures")
        print("      - Circuit breaker is open")
        
        # Issue 3: Server configuration
        print("   4. Check ModelServerConfig settings:")
        print("      - max_concurrent_requests")
        print("      - timeout values")
        print("      - endpoint URL format")

# ============================================================================
# FIXED COORDINATOR CONFIGURATION
# ============================================================================

def create_fixed_coordinator_config():
    """Create a properly configured coordinator"""
    
    # FIXED: Proper server configuration
    model_servers = [{
        "name": "gpu_server_2",
        "endpoint": "http://171.201.3.165:8100",  # Your actual server
        "gpu_id": 2,  # Your GPU ID
        "max_concurrent_requests": 3,  # Reduced for single GPU
        "timeout": 120  # Increased timeout
    }]
    
    print("🔧 FIXED COORDINATOR CONFIGURATION:")
    print(json.dumps(model_servers, indent=2))
    
    return model_servers

# ============================================================================
# FIXED API COORDINATOR INITIALIZATION
# ============================================================================

async def create_fixed_api_coordinator():
    """Create properly configured API coordinator"""
    
    # Import your coordinator (adjust import path as needed)
    try:
        from api_opulence_coordinator import create_api_coordinator_from_config
    except ImportError:
        print("❌ Cannot import coordinator - check your imports")
        return None
    
    # Use fixed configuration
    model_servers = create_fixed_coordinator_config()
    
    # FIXED: Create coordinator with proper settings
    coordinator = create_api_coordinator_from_config(
        model_servers=model_servers,
        load_balancing_strategy="round_robin",  # Simple strategy for single server
        max_retries=2,  # Reduced retries
        connection_pool_size=5,  # Smaller pool for single server
        request_timeout=120,  # Longer timeout
        circuit_breaker_threshold=3,  # Fewer failures before circuit opens
        retry_delay=1.0  # Shorter retry delay
    )
    
    return coordinator

# ============================================================================
# FIXED BASE AGENT API CALLS
# ============================================================================

class FixedBaseOpulenceAgent:
    """Fixed base agent with proper API call handling"""
    
    def __init__(self, coordinator, agent_type: str, gpu_id: int = 2):
        self.coordinator = coordinator
        self.agent_type = agent_type
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # FIXED: Simplified API parameters that match server expectations
        self.api_params = {
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
    
    async def call_api(self, prompt: str, params: Dict[str, Any] = None) -> str:
        """FIXED: API call with better error handling"""
        
        try:
            # FIXED: Merge and validate parameters
            final_params = self.api_params.copy()
            if params:
                final_params.update(params)
            
            # FIXED: Remove None values and validate ranges
            cleaned_params = {}
            for key, value in final_params.items():
                if value is not None:
                    if key == "max_tokens":
                        cleaned_params[key] = max(1, min(value, 2048))
                    elif key == "temperature":
                        cleaned_params[key] = max(0.0, min(value, 1.0))
                    elif key == "top_p":
                        cleaned_params[key] = max(0.0, min(value, 1.0))
                    else:
                        cleaned_params[key] = value
            
            self.logger.debug(f"Making API call with params: {cleaned_params}")
            
            # FIXED: Check coordinator and load balancer
            if not self.coordinator:
                raise RuntimeError("No coordinator available")
            
            if not hasattr(self.coordinator, 'load_balancer'):
                raise RuntimeError("Coordinator has no load_balancer")
            
            # FIXED: Check available servers before making call
            available_servers = self.coordinator.load_balancer.get_available_servers()
            self.logger.debug(f"Available servers: {len(available_servers)}")
            
            if not available_servers:
                # Try to get server status for debugging
                all_servers = self.coordinator.load_balancer.servers
                self.logger.error(f"No available servers. All servers status:")
                for server in all_servers:
                    self.logger.error(f"  {server.config.name}: status={server.status.value}, available={server.is_available()}")
                raise RuntimeError("No available model servers")
            
            # Make API call
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=cleaned_params,
                preferred_gpu_id=self.gpu_id
            )
            
            # FIXED: Extract response text properly
            if isinstance(result, dict):
                response_text = (
                    result.get('text') or 
                    result.get('response') or 
                    result.get('content') or
                    result.get('generated_text') or
                    str(result.get('choices', [{}])[0].get('text', '')) or
                    str(result)
                )
            else:
                response_text = str(result)
            
            self.logger.debug(f"API call successful: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            
            # FIXED: Better error categorization
            error_msg = str(e).lower()
            if "no available" in error_msg:
                raise RuntimeError(f"No model servers available - check server status and configuration")
            elif "422" in error_msg or "validation" in error_msg:
                raise RuntimeError(f"API validation error - check parameters: {cleaned_params}")
            elif "timeout" in error_msg:
                raise RuntimeError(f"API timeout - server may be overloaded")
            elif "connection" in error_msg:
                raise RuntimeError(f"Connection error - check server endpoint: {self.coordinator.load_balancer.servers[0].config.endpoint if self.coordinator.load_balancer.servers else 'None'}")
            else:
                raise RuntimeError(f"API call failed: {str(e)}")

# ============================================================================
# DIAGNOSTIC AND TESTING FUNCTIONS
# ============================================================================

async def test_coordinator_functionality():
    """Test coordinator functionality step by step"""
    
    print("\n🧪 TESTING COORDINATOR FUNCTIONALITY")
    print("=" * 60)
    
    # Step 1: Create coordinator
    print("\n1. Creating coordinator...")
    coordinator = await create_fixed_api_coordinator()
    
    if not coordinator:
        print("❌ Failed to create coordinator")
        return False
    
    # Step 2: Initialize coordinator
    print("\n2. Initializing coordinator...")
    try:
        await coordinator.initialize()
        print("✅ Coordinator initialized")
    except Exception as e:
        print(f"❌ Coordinator initialization failed: {e}")
        return False
    
    # Step 3: Check health
    print("\n3. Checking coordinator health...")
    try:
        health = coordinator.get_health_status()
        print(f"✅ Health status: {health.get('status')}")
        print(f"   Available servers: {health.get('available_servers')}")
        print(f"   Total servers: {health.get('total_servers')}")
        
        if health.get('available_servers', 0) == 0:
            print("❌ No servers available in coordinator")
            
            # Debug server status
            for server in coordinator.load_balancer.servers:
                print(f"   Server {server.config.name}:")
                print(f"     Status: {server.status.value}")
                print(f"     Available: {server.is_available()}")
                print(f"     Endpoint: {server.config.endpoint}")
                print(f"     Active requests: {server.active_requests}")
                print(f"     Max requests: {server.config.max_concurrent_requests}")
            
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Step 4: Test API call
    print("\n4. Testing direct API call...")
    try:
        test_params = {
            "max_tokens": 10,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        result = await coordinator.call_model_api(
            "Hello, this is a test",
            test_params,
            preferred_gpu_id=2
        )
        
        print(f"✅ API call successful: {result}")
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False
    
    # Step 5: Test agent creation
    print("\n5. Testing agent creation...")
    try:
        agent = FixedBaseOpulenceAgent(coordinator, "test_agent", gpu_id=2)
        response = await agent.call_api("Test prompt")
        print(f"✅ Agent call successful: {response}")
        
    except Exception as e:
        print(f"❌ Agent call failed: {e}")
        return False
    
    # Cleanup
    print("\n6. Cleaning up...")
    try:
        await coordinator.shutdown()
        print("✅ Coordinator shutdown successful")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    return True

# ============================================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================================

def main_diagnostic():
    """Run comprehensive diagnostic"""
    
    print("\n🚀 OPULENCE MODEL SERVER DIAGNOSTIC")
    print("=" * 80)
    
    # Configuration
    server_endpoint = "http://171.201.3.165:8100"
    gpu_id = 2
    
    # Step 1: Debug server connectivity
    debugger = ModelServerDebugger(server_endpoint, gpu_id)
    debugger.debug_server_connectivity()
    
    # Step 2: Test coordinator functionality
    print("\n" + "=" * 80)
    success = asyncio.run(test_coordinator_functionality())
    
    # Step 3: Recommendations
    print("\n" + "=" * 80)
    print("🔧 RECOMMENDATIONS:")
    
    if success:
        print("✅ System appears to be working correctly!")
    else:
        print("❌ Issues found. Check the following:")
        print("\n1. Server Configuration:")
        print("   - Ensure server is running on http://171.201.3.165:8100")
        print("   - Check /health and /generate endpoints")
        print("   - Verify model is loaded and ready")
        
        print("\n2. Payload Format:")
        print("   - Use proper JSON format for /generate endpoint")
        print("   - Include required fields: prompt, max_tokens, temperature")
        print("   - Avoid None values in parameters")
        
        print("\n3. Coordinator Configuration:")
        print("   - Use correct GPU ID (2) in server config")
        print("   - Set appropriate timeouts and retry settings")
        print("   - Check load balancer health check settings")
        
        print("\n4. Network Issues:")
        print("   - Verify network connectivity between client and server")
        print("   - Check firewall settings for port 8100")
        print("   - Ensure no proxy issues")

if __name__ == "__main__":
    main_diagnostic()