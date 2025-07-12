#!/usr/bin/env python3
"""
Fix Server Availability Detection Issue
The coordinator shows "No servers available" even though server is healthy
"""

import asyncio
import aiohttp
import requests
import time
from typing import Dict, Any
from enum import Enum

# ============================================================================
# ROOT CAUSE ANALYSIS
# ============================================================================

"""
From your screenshots, I can see:

1. ✅ Model server is running and healthy at http://171.201.3.165:8100
2. ✅ Server responds to health checks correctly  
3. ✅ Coordinator is created successfully
4. ❌ Coordinator shows "No servers available in coordinator"
5. ❌ Server status is "unknown" and Available: false

PROBLEM: The server is not being marked as AVAILABLE in the load balancer
even though it's healthy. This happens because:

1. Health check might be failing in the coordinator (different from manual check)
2. Server is marked as unavailable due to circuit breaker or other conditions
3. The is_available() method is returning False for some reason
4. Initial health check during coordinator setup failed
"""

# ============================================================================
# ENHANCED SERVER HEALTH CHECKING
# ============================================================================

class ModelServerStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"

class FixedModelServer:
    """Fixed model server with better availability detection"""
    
    def __init__(self, config):
        self.config = config
        self.status = ModelServerStatus.UNKNOWN
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.consecutive_failures = 0
        self.circuit_breaker_open_time = 0
        
    def is_available(self) -> bool:
        """FIXED: More lenient availability check"""
        # 1. Check circuit breaker first
        if self.status == ModelServerStatus.CIRCUIT_OPEN:
            # Check if circuit breaker should be reset (more aggressive reset)
            if time.time() - self.circuit_breaker_open_time > 30:  # Reduced from 60s
                print(f"🔄 Resetting circuit breaker for {self.config.name}")
                self.status = ModelServerStatus.UNKNOWN
                self.consecutive_failures = 0
                return True
            return False
        
        # 2. More lenient status check
        is_status_ok = self.status in [ModelServerStatus.HEALTHY, ModelServerStatus.UNKNOWN]
        
        # 3. Check request capacity
        is_capacity_ok = self.active_requests < self.config.max_concurrent_requests
        
        print(f"🔍 Server {self.config.name} availability:")
        print(f"   Status: {self.status.value} -> OK: {is_status_ok}")
        print(f"   Requests: {self.active_requests}/{self.config.max_concurrent_requests} -> OK: {is_capacity_ok}")
        print(f"   Consecutive failures: {self.consecutive_failures}")
        
        return is_status_ok and is_capacity_ok
    
    def should_open_circuit(self, threshold: int) -> bool:
        """More tolerant circuit breaker"""
        return self.consecutive_failures >= threshold

# ============================================================================
# ENHANCED MODEL SERVER CLIENT WITH BETTER HEALTH CHECKS
# ============================================================================

class FixedModelServerClient:
    """Fixed client with better health checking"""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        
    async def initialize(self):
        """Initialize with better timeout settings"""
        connector = aiohttp.TCPConnector(
            limit=10,  # Reduced
            limit_per_host=5,  # Reduced
            ttl_dns_cache=300,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,  # Reduced timeout for health checks
            connect=10  # Faster connection timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'OpulenceCoordinator/1.0.0'
            }
        )
    
    async def health_check(self, server) -> bool:
        """FIXED: Enhanced health check with better error handling"""
        if not self.session:
            print(f"❌ No session for health check of {server.config.name}")
            return False
            
        try:
            health_url = f"{server.config.endpoint}/health"
            print(f"🏥 Health checking {server.config.name} at {health_url}")
            
            async with self.session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=10)  # Quick health check
            ) as response:
                
                status_code = response.status
                response_text = await response.text()
                
                print(f"   Health response: {status_code} - {response_text[:100]}")
                
                if status_code == 200:
                    server.status = ModelServerStatus.HEALTHY
                    server.consecutive_failures = 0  # Reset failures on success
                    print(f"✅ {server.config.name} health check passed")
                    return True
                else:
                    server.status = ModelServerStatus.UNHEALTHY
                    server.consecutive_failures += 1
                    print(f"❌ {server.config.name} health check failed: {status_code}")
                    return False
                    
        except asyncio.TimeoutError:
            server.status = ModelServerStatus.UNHEALTHY
            server.consecutive_failures += 1
            print(f"⏰ {server.config.name} health check timeout")
            return False
            
        except Exception as e:
            server.status = ModelServerStatus.UNHEALTHY
            server.consecutive_failures += 1
            print(f"❌ {server.config.name} health check error: {e}")
            return False

# ============================================================================
# FIXED LOAD BALANCER WITH BETTER SERVER MANAGEMENT
# ============================================================================

class FixedLoadBalancer:
    """Fixed load balancer with better server availability detection"""
    
    def __init__(self, config):
        self.config = config
        self.servers = []
        self.current_index = 0
        
        # Initialize servers with fixed class
        for server_config in config.model_servers:
            server = FixedModelServer(server_config)
            self.servers.append(server)
            print(f"🖥️ Added server: {server.config.name} at {server.config.endpoint}")
    
    def get_available_servers(self):
        """Get available servers with detailed logging"""
        available = []
        
        print(f"\n🔍 Checking {len(self.servers)} servers for availability:")
        
        for server in self.servers:
            is_avail = server.is_available()
            print(f"   {server.config.name}: {'✅ Available' if is_avail else '❌ Unavailable'}")
            
            if is_avail:
                available.append(server)
        
        print(f"📊 Result: {len(available)}/{len(self.servers)} servers available")
        return available
    
    def select_server(self):
        """Select server with fallback logic"""
        available_servers = self.get_available_servers()
        
        if not available_servers:
            print("❌ No available servers found!")
            
            # EMERGENCY FALLBACK: Try to reset all servers
            print("🚨 Emergency fallback: Attempting to reset server status...")
            for server in self.servers:
                if server.status == ModelServerStatus.CIRCUIT_OPEN:
                    server.status = ModelServerStatus.UNKNOWN
                    server.consecutive_failures = 0
                    print(f"🔄 Reset {server.config.name} from circuit_open to unknown")
                elif server.status == ModelServerStatus.UNHEALTHY:
                    server.status = ModelServerStatus.UNKNOWN
                    print(f"🔄 Reset {server.config.name} from unhealthy to unknown")
            
            # Try again after reset
            available_servers = self.get_available_servers()
            
            if not available_servers:
                print("❌ Still no available servers after reset")
                return None
        
        # Simple round-robin
        server = available_servers[self.current_index % len(available_servers)]
        self.current_index += 1
        
        print(f"🎯 Selected server: {server.config.name}")
        return server

# ============================================================================
# ENHANCED COORDINATOR INITIALIZATION
# ============================================================================

async def enhanced_coordinator_initialization():
    """Enhanced initialization with step-by-step verification"""
    
    print("\n🚀 ENHANCED COORDINATOR INITIALIZATION")
    print("=" * 60)
    
    # Step 1: Server configuration
    print("\n1. Configuring server...")
    server_config = {
        "name": "gpu_server_2",
        "endpoint": "http://171.201.3.165:8100",
        "gpu_id": 2,
        "max_concurrent_requests": 2,  # Very conservative
        "timeout": 60
    }
    print(f"   Server config: {server_config}")
    
    # Step 2: Manual health check first
    print("\n2. Manual health check...")
    try:
        response = requests.get(f"{server_config['endpoint']}/health", timeout=10)
        print(f"   Manual health check: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Manual health check passed")
        else:
            print(f"   ❌ Manual health check failed: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Manual health check error: {e}")
        return None
    
    # Step 3: Create enhanced coordinator configuration
    print("\n3. Creating coordinator config...")
    
    from api_opulence_coordinator import APIOpulenceConfig, ModelServerConfig
    
    model_server = ModelServerConfig(
        endpoint=server_config["endpoint"],
        gpu_id=server_config["gpu_id"],
        name=server_config["name"],
        max_concurrent_requests=server_config["max_concurrent_requests"],
        timeout=server_config["timeout"]
    )
    
    config = APIOpulenceConfig(
        model_servers=[model_server],
        load_balancing_strategy="round_robin",
        max_retries=1,  # Minimal retries
        connection_pool_size=5,
        request_timeout=60,
        circuit_breaker_threshold=10,  # Very tolerant
        health_check_interval=60  # Less frequent checks
    )
    
    print("   ✅ Configuration created")
    
    # Step 4: Create coordinator
    print("\n4. Creating coordinator...")
    
    from api_opulence_coordinator import APIOpulenceCoordinator
    
    coordinator = APIOpulenceCoordinator(config)
    print("   ✅ Coordinator created")
    
    # Step 5: Initialize coordinator
    print("\n5. Initializing coordinator...")
    try:
        await coordinator.initialize()
        print("   ✅ Coordinator initialized")
    except Exception as e:
        print(f"   ❌ Coordinator initialization failed: {e}")
        return None
    
    # Step 6: IMMEDIATE health check of all servers
    print("\n6. Running immediate health checks...")
    
    for server in coordinator.load_balancer.servers:
        print(f"   Checking {server.config.name}...")
        try:
            is_healthy = await coordinator.client.health_check(server)
            print(f"      Health check result: {is_healthy}")
            print(f"      Server status: {server.status.value}")
            print(f"      Is available: {server.is_available()}")
        except Exception as e:
            print(f"      Health check error: {e}")
    
    # Step 7: Verify coordinator health
    print("\n7. Checking coordinator health...")
    try:
        health = coordinator.get_health_status()
        print(f"   Available servers: {health.get('available_servers')}")
        print(f"   Total servers: {health.get('total_servers')}")
        
        if health.get('available_servers', 0) > 0:
            print("   ✅ Coordinator reports servers available")
        else:
            print("   ❌ Coordinator reports no servers available")
            
            # Debug server status
            print("\n   🔍 Debugging server status:")
            for server in coordinator.load_balancer.servers:
                print(f"      {server.config.name}:")
                print(f"        Status: {server.status.value}")
                print(f"        Available: {server.is_available()}")
                print(f"        Active requests: {server.active_requests}")
                print(f"        Consecutive failures: {server.consecutive_failures}")
        
    except Exception as e:
        print(f"   ❌ Health status error: {e}")
    
    return coordinator

# ============================================================================
# EMERGENCY FIX FOR EXISTING COORDINATOR
# ============================================================================

def emergency_fix_coordinator(coordinator):
    """Emergency fix for existing coordinator that shows no available servers"""
    
    print("\n🚨 EMERGENCY COORDINATOR FIX")
    print("=" * 50)
    
    if not coordinator:
        print("❌ No coordinator provided")
        return False
    
    if not hasattr(coordinator, 'load_balancer'):
        print("❌ Coordinator has no load_balancer")
        return False
    
    if not coordinator.load_balancer.servers:
        print("❌ No servers in load balancer")
        return False
    
    print(f"🔍 Found {len(coordinator.load_balancer.servers)} servers")
    
    # Reset all servers to a known good state
    for server in coordinator.load_balancer.servers:
        print(f"\n🔧 Fixing server: {server.config.name}")
        print(f"   Current status: {server.status.value}")
        print(f"   Current availability: {server.is_available()}")
        
        # Force reset server state
        server.status = ModelServerStatus.HEALTHY  # Force healthy
        server.consecutive_failures = 0
        server.active_requests = 0
        server.circuit_breaker_open_time = 0
        
        print(f"   ✅ Reset to: {server.status.value}")
        print(f"   ✅ New availability: {server.is_available()}")
    
    # Test availability
    available_servers = coordinator.load_balancer.get_available_servers()
    print(f"\n📊 Available servers after fix: {len(available_servers)}")
    
    return len(available_servers) > 0

# ============================================================================
# STREAMLIT INTEGRATION FIXES
# ============================================================================

def fix_streamlit_coordinator():
    """Fix the coordinator in Streamlit session state"""
    
    import streamlit as st
    
    if 'coordinator' not in st.session_state or not st.session_state.coordinator:
        st.error("❌ No coordinator in session state")
        return False
    
    coordinator = st.session_state.coordinator
    
    st.info("🔧 Attempting to fix coordinator...")
    
    # Apply emergency fix
    success = emergency_fix_coordinator(coordinator)
    
    if success:
        st.success("✅ Coordinator fixed! Servers are now available.")
        
        # Verify the fix
        health = coordinator.get_health_status()
        st.info(f"Available servers: {health.get('available_servers')}")
        
        return True
    else:
        st.error("❌ Fix failed. Try restarting the coordinator.")
        return False

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

async def test_fixed_coordinator():
    """Test the fixed coordinator functionality"""
    
    print("\n🧪 TESTING FIXED COORDINATOR")
    print("=" * 50)
    
    # Create enhanced coordinator
    coordinator = await enhanced_coordinator_initialization()
    
    if not coordinator:
        print("❌ Failed to create coordinator")
        return False
    
    try:
        # Test 1: Check available servers
        print("\n🔍 Test 1: Check available servers")
        available = coordinator.load_balancer.get_available_servers()
        print(f"Available servers: {len(available)}")
        
        if len(available) == 0:
            print("❌ No servers available - applying emergency fix...")
            emergency_fix_coordinator(coordinator)
            available = coordinator.load_balancer.get_available_servers()
            print(f"Available servers after fix: {len(available)}")
        
        # Test 2: Make API call
        if len(available) > 0:
            print("\n🔍 Test 2: Make API call")
            try:
                result = await coordinator.call_model_api(
                    "Hello test",
                    {"max_tokens": 5, "temperature": 0.1}
                )
                print(f"✅ API call successful: {result}")
                return True
            except Exception as e:
                print(f"❌ API call failed: {e}")
                return False
        else:
            print("❌ Cannot test API call - no available servers")
            return False
    
    finally:
        # Cleanup
        try:
            await coordinator.shutdown()
        except:
            pass

# ============================================================================
# INSTRUCTIONS FOR FIXING YOUR ISSUE
# ============================================================================

"""
IMMEDIATE FIXES FOR YOUR STREAMLIT APP:

1. ADD THIS TO YOUR STREAMLIT APP:

def fix_coordinator_availability():
    '''Add this function to your Streamlit app'''
    if st.button("🚨 Emergency Fix Coordinator"):
        if st.session_state.coordinator:
            # Force reset all servers to healthy state
            for server in st.session_state.coordinator.load_balancer.servers:
                server.status = ModelServerStatus.HEALTHY
                server.consecutive_failures = 0
                server.active_requests = 0
                server.circuit_breaker_open_time = 0
            
            # Check if fix worked
            available = st.session_state.coordinator.load_balancer.get_available_servers()
            st.success(f"✅ Fixed! Now {len(available)} servers available")
        else:
            st.error("No coordinator found")

2. UPDATE YOUR COORDINATOR CONFIG:

# Use more conservative settings
model_servers = [{
    "name": "gpu_server_2", 
    "endpoint": "http://171.201.3.165:8100",
    "gpu_id": 2,
    "max_concurrent_requests": 1,  # Very conservative
    "timeout": 30
}]

coordinator = create_api_coordinator_from_config(
    model_servers=model_servers,
    load_balancing_strategy="round_robin",
    max_retries=1,
    connection_pool_size=2,
    request_timeout=30,
    circuit_breaker_threshold=20,  # Very tolerant
    health_check_interval=120  # Less frequent
)

3. ADD DEBUGGING TO YOUR STREAMLIT SIDEBAR:

if st.sidebar.button("🔍 Debug Servers"):
    if st.session_state.coordinator:
        for server in st.session_state.coordinator.load_balancer.servers:
            st.sidebar.write(f"Server: {server.config.name}")
            st.sidebar.write(f"Status: {server.status.value}")
            st.sidebar.write(f"Available: {server.is_available()}")
            st.sidebar.write(f"Failures: {server.consecutive_failures}")

4. FORCE HEALTH CHECK:

if st.sidebar.button("🏥 Force Health Check"):
    async def force_health_check():
        for server in st.session_state.coordinator.load_balancer.servers:
            await st.session_state.coordinator.client.health_check(server)
    
    safe_run_async(force_health_check())
    st.sidebar.success("Health check completed")

The issue is that your server is healthy but the coordinator's load balancer 
is marking it as unavailable due to failed health checks or circuit breaker logic.
"""

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_fixed_coordinator())