#!/usr/bin/env python3
"""
GPU Detection and Availability Test Routine
Comprehensive testing to find where the model server is actually running
and why preferred GPU 2 shows as not available
"""

import requests
import json
import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# COMPREHENSIVE GPU DETECTION TEST ROUTINE
# ============================================================================

class GPUDetectionTester:
    """Comprehensive GPU detection and availability testing"""
    
    def __init__(self, server_endpoint: str = "http://171.201.3.165:8100"):
        self.server_endpoint = server_endpoint
        self.logger = logging.getLogger(f"{__name__}.GPUTester")
        
    async def run_comprehensive_gpu_test(self):
        """Run complete GPU detection and availability test"""
        
        print("\n🔍 COMPREHENSIVE GPU DETECTION TEST")
        print("=" * 80)
        
        # Test 1: Direct server interrogation
        print("\n1. 🖥️ DIRECT SERVER INTERROGATION")
        print("-" * 50)
        server_gpu_info = await self._interrogate_server_directly()
        
        # Test 2: API endpoint analysis
        print("\n2. 🔌 API ENDPOINT ANALYSIS")
        print("-" * 50)
        api_gpu_info = await self._analyze_api_endpoints()
        
        # Test 3: Load test with GPU tracking
        print("\n3. 🏋️ LOAD TEST WITH GPU TRACKING")
        print("-" * 50)
        load_test_info = await self._run_load_test_with_gpu_tracking()
        
        # Test 4: Coordinator integration test
        print("\n4. 🔧 COORDINATOR INTEGRATION TEST")
        print("-" * 50)
        coordinator_info = await self._test_coordinator_gpu_detection()
        
        # Test 5: Availability diagnosis
        print("\n5. 🏥 AVAILABILITY DIAGNOSIS")
        print("-" * 50)
        availability_info = await self._diagnose_availability_issues()
        
        # Summary and recommendations
        print("\n📋 SUMMARY AND RECOMMENDATIONS")
        print("=" * 80)
        self._generate_recommendations(
            server_gpu_info, api_gpu_info, load_test_info, 
            coordinator_info, availability_info
        )
        
        return {
            'server_gpu_info': server_gpu_info,
            'api_gpu_info': api_gpu_info,
            'load_test_info': load_test_info,
            'coordinator_info': coordinator_info,
            'availability_info': availability_info
        }
    
    async def _interrogate_server_directly(self) -> Dict[str, Any]:
        """Directly interrogate the server to find GPU information"""
        
        gpu_info = {
            'detected_gpu_id': None,
            'gpu_processes': [],
            'server_responses': {},
            'nvidia_smi_equivalent': None
        }
        
        try:
            # Test 1: Check /status endpoint for GPU info
            print("   📊 Checking /status endpoint...")
            try:
                response = requests.get(f"{self.server_endpoint}/status", timeout=10)
                if response.status_code == 200:
                    status_data = response.json()
                    gpu_info['server_responses']['status'] = status_data
                    
                    # Look for GPU information in response
                    if 'gpu_info' in status_data:
                        print(f"   ✅ Found gpu_info: {status_data['gpu_info']}")
                        gpu_info['gpu_processes'] = status_data['gpu_info']
                    
                    if 'device' in status_data:
                        print(f"   📱 Device info: {status_data['device']}")
                    
                    if 'model' in status_data:
                        print(f"   🤖 Model: {status_data['model']}")
                    
                    # Try to extract GPU ID from various fields
                    for key in ['gpu_id', 'device_id', 'cuda_device', 'gpu', 'device']:
                        if key in status_data:
                            potential_gpu = status_data[key]
                            print(f"   🎯 Potential GPU from {key}: {potential_gpu}")
                            if isinstance(potential_gpu, int):
                                gpu_info['detected_gpu_id'] = potential_gpu
                
                else:
                    print(f"   ❌ Status endpoint returned: {response.status_code}")
                    
            except Exception as e:
                print(f"   ⚠️ Status endpoint error: {e}")
            
            # Test 2: Check /health endpoint
            print("   🏥 Checking /health endpoint...")
            try:
                response = requests.get(f"{self.server_endpoint}/health", timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    gpu_info['server_responses']['health'] = health_data
                    print(f"   ✅ Health response: {health_data}")
                else:
                    print(f"   ❌ Health endpoint returned: {response.status_code}")
            except Exception as e:
                print(f"   ⚠️ Health endpoint error: {e}")
            
            # Test 3: Try to infer GPU from generate response metadata
            print("   🔬 Testing generate endpoint for GPU metadata...")
            try:
                test_payload = {
                    "prompt": "GPU test",
                    "max_tokens": 1,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    f"{self.server_endpoint}/generate",
                    json=test_payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    generate_data = response.json()
                    gpu_info['server_responses']['generate'] = generate_data
                    
                    # Look for GPU information in response
                    for key in ['gpu_id', 'device_id', 'server_id', 'node_id']:
                        if key in generate_data:
                            potential_gpu = generate_data[key]
                            print(f"   🎯 GPU from generate response {key}: {potential_gpu}")
                            if isinstance(potential_gpu, int):
                                gpu_info['detected_gpu_id'] = potential_gpu
                
                else:
                    print(f"   ❌ Generate test failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ⚠️ Generate test error: {e}")
            
        except Exception as e:
            print(f"   💥 Server interrogation failed: {e}")
        
        if gpu_info['detected_gpu_id'] is not None:
            print(f"   🎉 DETECTED GPU ID: {gpu_info['detected_gpu_id']}")
        else:
            print(f"   ❓ Could not detect GPU ID from server responses")
        
        return gpu_info
    
    async def _analyze_api_endpoints(self) -> Dict[str, Any]:
        """Analyze all available API endpoints for GPU information"""
        
        endpoints_to_test = [
            "/health", "/status", "/info", "/metrics", "/stats", 
            "/config", "/model", "/version", "/gpu", "/device"
        ]
        
        endpoint_results = {}
        detected_gpus = []
        
        print("   🔍 Testing various API endpoints...")
        
        for endpoint in endpoints_to_test:
            try:
                url = f"{self.server_endpoint}{endpoint}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        endpoint_results[endpoint] = {
                            'status': 'success',
                            'data': data,
                            'contains_gpu_info': any(
                                key in str(data).lower() 
                                for key in ['gpu', 'cuda', 'device']
                            )
                        }
                        print(f"   ✅ {endpoint}: Available")
                        
                        # Look for GPU references
                        gpu_refs = self._extract_gpu_references(data)
                        if gpu_refs:
                            detected_gpus.extend(gpu_refs)
                            print(f"      🎯 GPU references: {gpu_refs}")
                    except:
                        endpoint_results[endpoint] = {
                            'status': 'success',
                            'data': response.text,
                            'contains_gpu_info': 'gpu' in response.text.lower()
                        }
                else:
                    endpoint_results[endpoint] = {
                        'status': f'http_{response.status_code}',
                        'data': None
                    }
                    
            except requests.exceptions.ConnectionError:
                endpoint_results[endpoint] = {'status': 'connection_error'}
            except Exception as e:
                endpoint_results[endpoint] = {'status': f'error_{str(e)}'}
        
        return {
            'endpoint_results': endpoint_results,
            'detected_gpus': list(set(detected_gpus)),
            'working_endpoints': [
                ep for ep, result in endpoint_results.items() 
                if result['status'] == 'success'
            ]
        }
    
    def _extract_gpu_references(self, data: Any) -> List[int]:
        """Extract GPU ID references from API response data"""
        gpu_refs = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                if any(gpu_key in key_lower for gpu_key in ['gpu', 'device', 'cuda']):
                    if isinstance(value, int):
                        gpu_refs.append(value)
                    elif isinstance(value, str) and value.isdigit():
                        gpu_refs.append(int(value))
                
                # Recursively check nested dictionaries
                if isinstance(value, dict):
                    gpu_refs.extend(self._extract_gpu_references(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            gpu_refs.extend(self._extract_gpu_references(item))
        
        return gpu_refs
    
    async def _run_load_test_with_gpu_tracking(self) -> Dict[str, Any]:
        """Run load test while tracking GPU usage patterns"""
        
        print("   🏋️ Running load test with GPU tracking...")
        
        load_results = {
            'total_requests': 0,
            'successful_requests': 0,
            'gpu_usage_patterns': {},
            'response_times': [],
            'unique_responses': set()
        }
        
        # Make multiple rapid requests to see GPU patterns
        for i in range(5):
            try:
                start_time = time.time()
                
                test_payload = {
                    "prompt": f"Test {i+1}: What GPU are you running on?",
                    "max_tokens": 10,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    f"{self.server_endpoint}/generate",
                    json=test_payload,
                    timeout=20
                )
                
                response_time = time.time() - start_time
                load_results['total_requests'] += 1
                load_results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    load_results['successful_requests'] += 1
                    result = response.json()
                    
                    # Track any GPU information in responses
                    for key in ['gpu_id', 'device_id', 'server_id', 'node_id']:
                        if key in result:
                            gpu_id = result[key]
                            if gpu_id not in load_results['gpu_usage_patterns']:
                                load_results['gpu_usage_patterns'][gpu_id] = 0
                            load_results['gpu_usage_patterns'][gpu_id] += 1
                    
                    # Track response text for patterns
                    response_text = result.get('text', result.get('response', ''))
                    load_results['unique_responses'].add(response_text[:50])
                    
                    print(f"      Request {i+1}: {response_time:.2f}s - {response_text[:30]}...")
                
                else:
                    print(f"      Request {i+1}: Failed with {response.status_code}")
                
                # Small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"      Request {i+1}: Error - {e}")
        
        # Convert set to list for JSON serialization
        load_results['unique_responses'] = list(load_results['unique_responses'])
        
        success_rate = (load_results['successful_requests'] / load_results['total_requests'] * 100) if load_results['total_requests'] > 0 else 0
        avg_response_time = sum(load_results['response_times']) / len(load_results['response_times']) if load_results['response_times'] else 0
        
        print(f"   📊 Load test results: {success_rate:.1f}% success rate, {avg_response_time:.2f}s avg response time")
        
        return load_results
    
    async def _test_coordinator_gpu_detection(self) -> Dict[str, Any]:
        """Test GPU detection through coordinator integration"""
        
        coordinator_info = {
            'coordinator_created': False,
            'servers_configured': [],
            'available_servers': [],
            'gpu_lookup_test': {},
            'load_balancer_test': {}
        }
        
        try:
            print("   🔧 Testing coordinator GPU detection...")
            
            # Try to import and create coordinator
            try:
                from api_opulence_coordinator import create_api_coordinator_from_config, ModelServerStatus
                
                # Test different GPU ID configurations
                test_configs = [
                    {"gpu_id": 2, "name": "test_gpu_2"},
                    {"gpu_id": 0, "name": "test_gpu_0"},
                    {"gpu_id": 1, "name": "test_gpu_1"},
                ]
                
                for config in test_configs:
                    print(f"      🧪 Testing coordinator with GPU {config['gpu_id']}...")
                    
                    model_servers = [{
                        "name": config['name'],
                        "endpoint": self.server_endpoint,
                        "gpu_id": config['gpu_id'],
                        "max_concurrent_requests": 1,
                        "timeout": 30
                    }]
                    
                    coordinator = create_api_coordinator_from_config(
                        model_servers=model_servers,
                        load_balancing_strategy="round_robin",
                        max_retries=1,
                        connection_pool_size=1,
                        request_timeout=30
                    )
                    
                    coordinator_info['coordinator_created'] = True
                    
                    # Initialize coordinator
                    await coordinator.initialize()
                    
                    # Test server configuration
                    servers = coordinator.load_balancer.servers
                    coordinator_info['servers_configured'].extend([
                        {
                            'name': s.config.name,
                            'gpu_id': s.config.gpu_id,
                            'endpoint': s.config.endpoint
                        } for s in servers
                    ])
                    
                    # Force server to healthy status for testing
                    for server in servers:
                        server.status = ModelServerStatus.HEALTHY
                        server.consecutive_failures = 0
                        server.active_requests = 0
                    
                    # Test availability
                    available_servers = coordinator.load_balancer.get_available_servers()
                    coordinator_info['available_servers'].extend([
                        {
                            'name': s.config.name,
                            'gpu_id': s.config.gpu_id,
                            'is_available': s.is_available()
                        } for s in available_servers
                    ])
                    
                    # Test GPU lookup
                    gpu_server = coordinator.load_balancer.get_server_by_gpu_id(config['gpu_id'])
                    coordinator_info['gpu_lookup_test'][config['gpu_id']] = {
                        'found': gpu_server is not None,
                        'server_name': gpu_server.config.name if gpu_server else None,
                        'is_available': gpu_server.is_available() if gpu_server else False
                    }
                    
                    # Test API call if server is available
                    if gpu_server and gpu_server.is_available():
                        try:
                            result = await coordinator.call_model_api(
                                "Test GPU detection",
                                {"max_tokens": 5, "temperature": 0.1},
                                preferred_gpu_id=config['gpu_id']
                            )
                            
                            coordinator_info['gpu_lookup_test'][config['gpu_id']]['api_call'] = {
                                'success': True,
                                'gpu_used': result.get('gpu_id', 'unknown'),
                                'response': result.get('text', str(result))[:50]
                            }
                            
                            print(f"         ✅ API call successful on GPU {config['gpu_id']}")
                            
                        except Exception as api_error:
                            coordinator_info['gpu_lookup_test'][config['gpu_id']]['api_call'] = {
                                'success': False,
                                'error': str(api_error)
                            }
                            print(f"         ❌ API call failed on GPU {config['gpu_id']}: {api_error}")
                    
                    # Cleanup
                    await coordinator.shutdown()
                    
                    # Break if we found a working configuration
                    if available_servers:
                        print(f"         🎉 Found working configuration with GPU {config['gpu_id']}")
                        break
                
            except ImportError as e:
                coordinator_info['import_error'] = str(e)
                print(f"      ❌ Cannot import coordinator: {e}")
            
        except Exception as e:
            coordinator_info['test_error'] = str(e)
            print(f"      💥 Coordinator test failed: {e}")
        
        return coordinator_info
    
    async def _diagnose_availability_issues(self) -> Dict[str, Any]:
        """Diagnose why preferred GPU 2 shows as not available"""
        
        availability_diagnosis = {
            'server_reachable': False,
            'health_check_passes': False,
            'generate_endpoint_works': False,
            'coordinator_config_issues': [],
            'load_balancer_issues': [],
            'circuit_breaker_status': None,
            'recommended_fixes': []
        }
        
        print("   🏥 Diagnosing availability issues...")
        
        # Test 1: Basic server reachability
        try:
            response = requests.get(f"{self.server_endpoint}/health", timeout=10)
            availability_diagnosis['server_reachable'] = True
            
            if response.status_code == 200:
                availability_diagnosis['health_check_passes'] = True
                print("      ✅ Server is reachable and healthy")
            else:
                print(f"      ⚠️ Server reachable but health check returns {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Server not reachable: {e}")
            availability_diagnosis['recommended_fixes'].append("Check if server is running and accessible")
        
        # Test 2: Generate endpoint functionality
        try:
            test_payload = {
                "prompt": "Availability test",
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.server_endpoint}/generate",
                json=test_payload,
                timeout=20
            )
            
            if response.status_code == 200:
                availability_diagnosis['generate_endpoint_works'] = True
                print("      ✅ Generate endpoint works correctly")
            else:
                print(f"      ❌ Generate endpoint returns {response.status_code}")
                availability_diagnosis['recommended_fixes'].append("Check generate endpoint configuration")
                
        except Exception as e:
            print(f"      ❌ Generate endpoint error: {e}")
            availability_diagnosis['recommended_fixes'].append("Fix generate endpoint issues")
        
        # Test 3: Coordinator configuration issues
        if availability_diagnosis['server_reachable'] and availability_diagnosis['generate_endpoint_works']:
            availability_diagnosis['coordinator_config_issues'].append("Server works but coordinator can't access it")
            availability_diagnosis['recommended_fixes'].extend([
                "Check coordinator server configuration",
                "Verify GPU ID mapping in coordinator",
                "Check load balancer health check logic",
                "Force server status to HEALTHY"
            ])
        
        return availability_diagnosis
    
    def _generate_recommendations(self, server_info, api_info, load_info, coordinator_info, availability_info):
        """Generate comprehensive recommendations based on test results"""
        
        print("\n🎯 DETECTED CONFIGURATION:")
        
        # Primary findings
        detected_gpu = server_info.get('detected_gpu_id')
        if detected_gpu is not None:
            print(f"   🎮 Server GPU ID: {detected_gpu}")
        else:
            print(f"   ❓ Could not detect GPU ID from server")
        
        working_endpoints = api_info.get('working_endpoints', [])
        if working_endpoints:
            print(f"   🔌 Working endpoints: {working_endpoints}")
        
        success_rate = (load_info['successful_requests'] / load_info['total_requests'] * 100) if load_info['total_requests'] > 0 else 0
        print(f"   📊 Server success rate: {success_rate:.1f}%")
        
        # Coordinator findings
        if coordinator_info.get('coordinator_created'):
            available_gpus = [info for info in coordinator_info.get('gpu_lookup_test', {}).values() if info.get('found')]
            print(f"   🔧 Coordinator can find {len(available_gpus)} GPU configurations")
        
        print("\n💡 RECOMMENDATIONS:")
        
        # Recommendation 1: GPU ID configuration
        if detected_gpu is not None:
            print(f"   1. ✅ Use GPU ID {detected_gpu} in your coordinator configuration")
            print(f"      model_servers = [{{\"gpu_id\": {detected_gpu}, \"endpoint\": \"{self.server_endpoint}\"}}]")
        else:
            print(f"   1. ⚠️ Could not auto-detect GPU ID. Try testing with different GPU IDs (0, 1, 2)")
        
        # Recommendation 2: Server availability
        if availability_info['server_reachable'] and availability_info['generate_endpoint_works']:
            print(f"   2. ✅ Server is working correctly")
            print(f"      Problem is likely in coordinator configuration or health checks")
        else:
            print(f"   2. ❌ Fix server connectivity issues first")
        
        # Recommendation 3: Coordinator fixes
        print(f"   3. 🔧 Coordinator fixes needed:")
        print(f"      - Force server status to HEALTHY after initialization")
        print(f"      - Use emergency fix button to reset server availability")
        print(f"      - Check load balancer health check logic")
        
        # Recommendation 4: Emergency fixes
        print(f"   4. 🚨 Emergency fixes to try:")
        print(f"      - Add GPU fix button to Streamlit app")
        print(f"      - Force all servers to GPU ID that works")
        print(f"      - Bypass coordinator health checks temporarily")

# ============================================================================
# STREAMLIT INTEGRATION
# ============================================================================

def run_gpu_detection_in_streamlit():
    """Run GPU detection test in Streamlit interface"""
    
    st.markdown("### 🔍 GPU Detection Test")
    
    server_endpoint = st.text_input(
        "Server Endpoint", 
        value="http://171.201.3.165:8100",
        help="Enter your model server endpoint"
    )
    
    if st.button("🚀 Run GPU Detection Test", type="primary"):
        
        if not server_endpoint:
            st.error("Please enter a server endpoint")
            return
        
        try:
            with st.spinner("Running comprehensive GPU detection test..."):
                
                # Create tester
                tester = GPUDetectionTester(server_endpoint)
                
                # Run async test
                import asyncio
                results = asyncio.run(tester.run_comprehensive_gpu_test())
            
            # Display results
            st.success("✅ GPU detection test completed!")
            
            # Show key findings
            server_gpu = results['server_gpu_info'].get('detected_gpu_id')
            if server_gpu is not None:
                st.info(f"🎯 **Detected GPU ID: {server_gpu}**")
                
                # Provide fix button
                if st.button(f"🔧 Configure Coordinator for GPU {server_gpu}"):
                    if st.session_state.get('coordinator'):
                        # Apply the detected GPU configuration
                        for server in st.session_state.coordinator.load_balancer.servers:
                            server.config.gpu_id = server_gpu
                        
                        st.success(f"✅ Configured coordinator for GPU {server_gpu}")
                        st.info("🎯 Try your COBOL parsing now!")
            else:
                st.warning("⚠️ Could not auto-detect GPU ID")
            
            # Show detailed results
            with st.expander("📊 Detailed Test Results", expanded=False):
                st.json(results)
                
        except Exception as e:
            st.error(f"❌ GPU detection test failed: {e}")

# ============================================================================
# STANDALONE TEST RUNNER
# ============================================================================

async def main():
    """Main function to run GPU detection test"""
    
    # You can customize the server endpoint here
    server_endpoint = "http://171.201.3.165:8100"
    
    tester = GPUDetectionTester(server_endpoint)
    results = await tester.run_comprehensive_gpu_test()
    
    return results

if __name__ == "__main__":
    # Run the test
    results = asyncio.run(main())
    
    # Save results to file for analysis
    import json
    with open("gpu_detection_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to gpu_detection_results.json")

# ============================================================================
# QUICK TEST FUNCTION FOR STREAMLIT
# ============================================================================

def quick_gpu_test():
    """Quick GPU test function for immediate use in Streamlit"""
    
    st.markdown("### ⚡ Quick GPU Test")
    
    if st.button("⚡ Quick Test"):
        try:
            # Test server directly
            response = requests.get("http://171.201.3.165:8100/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Server is responding")
                
                # Look for GPU information
                gpu_refs = []
                for key, value in data.items():
                    if 'gpu' in key.lower() or 'device' in key.lower():
                        gpu_refs.append(f"{key}: {value}")
                
                if gpu_refs:
                    st.info("🎯 GPU references found:")
                    for ref in gpu_refs:
                        st.write(f"- {ref}")
                else:
                    st.warning("⚠️ No GPU references found in /status response")
                
                # Show raw response
                with st.expander("📄 Raw Response"):
                    st.json(data)
            else:
                st.error(f"❌ Server returned {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Test failed: {e}")

"""
USAGE INSTRUCTIONS:

1. For standalone testing:
   python gpu_detection_test.py

2. For Streamlit integration:
   Add run_gpu_detection_in_streamlit() to your Streamlit app

3. For quick testing:
   Add quick_gpu_test() to your Streamlit app

This will help you find:
- What GPU your server is actually running on
- Why coordinator can't find GPU 2
- What configuration to use
- How to fix availability issues
"""