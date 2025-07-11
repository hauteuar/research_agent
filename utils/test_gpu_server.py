#!/usr/bin/env python3
"""
Diagnostic Script for GPU Server and Coordinator Issues
Run this to identify the exact problem with your setup
"""

import requests
import json
import time
import sys
from typing import Dict, Any

def test_server_endpoint(endpoint: str, verbose: bool = True) -> Dict[str, Any]:
    """Test a server endpoint thoroughly"""
    results = {
        'endpoint': endpoint,
        'health_check': {'status': 'failed', 'message': '', 'response_time': None},
        'status_check': {'status': 'failed', 'message': '', 'data': None},
        'metrics_check': {'status': 'failed', 'message': '', 'data': None},
        'overall_status': 'failed'
    }
    
    if verbose:
        print(f"\n🔍 Testing endpoint: {endpoint}")
        print("=" * 50)
    
    # Test 1: Health Check
    try:
        if verbose:
            print("1. Testing health endpoint...")
        start_time = time.time()
        response = requests.get(f"{endpoint}/health", timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results['health_check'] = {
                'status': 'success',
                'message': 'Health check passed',
                'response_time': response_time,
                'data': response.json() if response.text else {}
            }
            if verbose:
                print(f"   ✅ Health check: OK ({response_time:.3f}s)")
        else:
            results['health_check']['message'] = f"HTTP {response.status_code}"
            if verbose:
                print(f"   ❌ Health check: HTTP {response.status_code}")
                
    except requests.exceptions.ConnectionError:
        results['health_check']['message'] = "Connection refused - server not running"
        if verbose:
            print("   ❌ Health check: Connection refused (server not running)")
    except requests.exceptions.Timeout:
        results['health_check']['message'] = "Request timeout"
        if verbose:
            print("   ❌ Health check: Timeout")
    except Exception as e:
        results['health_check']['message'] = f"Error: {str(e)}"
        if verbose:
            print(f"   ❌ Health check: {str(e)}")
    
    # Test 2: Status Check
    try:
        if verbose:
            print("2. Testing status endpoint...")
        response = requests.get(f"{endpoint}/status", timeout=10)
        
        if response.status_code == 200:
            status_data = response.json()
            results['status_check'] = {
                'status': 'success',
                'message': 'Status check passed',
                'data': status_data
            }
            if verbose:
                print("   ✅ Status check: OK")
                print(f"      Model: {status_data.get('model', 'Unknown')}")
                print(f"      Active requests: {status_data.get('active_requests', 0)}")
                print(f"      GPUs: {len(status_data.get('gpu_info', {}))}")
        else:
            results['status_check']['message'] = f"HTTP {response.status_code}"
            if verbose:
                print(f"   ❌ Status check: HTTP {response.status_code}")
                
    except Exception as e:
        results['status_check']['message'] = f"Error: {str(e)}"
        if verbose:
            print(f"   ❌ Status check: {str(e)}")
    
    # Test 3: Metrics Check
    try:
        if verbose:
            print("3. Testing metrics endpoint...")
        response = requests.get(f"{endpoint}/metrics", timeout=10)
        
        if response.status_code == 200:
            metrics_data = response.json()
            results['metrics_check'] = {
                'status': 'success',
                'message': 'Metrics check passed',
                'data': metrics_data
            }
            if verbose:
                print("   ✅ Metrics check: OK")
                print(f"      RPS: {metrics_data.get('requests_per_second', 0):.2f}")
                print(f"      Latency: {metrics_data.get('average_latency', 0):.3f}s")
        else:
            results['metrics_check']['message'] = f"HTTP {response.status_code}"
            if verbose:
                print(f"   ⚠️  Metrics check: HTTP {response.status_code} (optional)")
                
    except Exception as e:
        results['metrics_check']['message'] = f"Error: {str(e)}"
        if verbose:
            print(f"   ⚠️  Metrics check: {str(e)} (optional)")
    
    # Overall status
    if (results['health_check']['status'] == 'success' and 
        results['status_check']['status'] == 'success'):
        results['overall_status'] = 'success'
        if verbose:
            print("\n🟢 Overall Status: HEALTHY")
    else:
        if verbose:
            print("\n🔴 Overall Status: UNHEALTHY")
    
    return results

def run_comprehensive_diagnostics():
    """Run comprehensive diagnostics"""
    print("🚀 GPU Server Diagnostic Tool")
    print("=" * 50)
    
    # Common endpoints to test
    endpoints_to_test = [
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
    ]
    
    results = []
    healthy_servers = []
    
    for endpoint in endpoints_to_test:
        result = test_server_endpoint(endpoint)
        results.append(result)
        
        if result['overall_status'] == 'success':
            healthy_servers.append(endpoint)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if healthy_servers:
        print(f"✅ Found {len(healthy_servers)} healthy server(s):")
        for server in healthy_servers:
            print(f"   • {server}")
        
        print("\n🔧 RECOMMENDED CONFIGURATION:")
        for i, server in enumerate(healthy_servers):
            print(f"""
Server {i+1}:
  Name: gpu_server_{i+1}
  Endpoint: {server}
  GPU ID: {i}
  Max Requests: 10
  Timeout: 300
""")
    else:
        print("❌ No healthy servers found!")
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Check if your model server is running")
        print("2. Verify the correct port number")
        print("3. Test manual connection:")
        for endpoint in endpoints_to_test[:2]:
            print(f"   curl {endpoint}/health")
        print("4. Check firewall settings")
        print("5. Verify GPU drivers and CUDA installation")
    
    # Detailed error analysis
    print("\n📊 DETAILED ERROR ANALYSIS:")
    for result in results:
        if result['overall_status'] != 'success':
            print(f"\n❌ {result['endpoint']}:")
            print(f"   Health: {result['health_check']['message']}")
            print(f"   Status: {result['status_check']['message']}")
    
    return results, healthy_servers

def generate_streamlit_config(healthy_servers):
    """Generate Streamlit configuration code"""
    if not healthy_servers:
        return None
    
    config_code = """
# Add this configuration to your Streamlit app:

st.session_state.model_servers = [
"""
    
    for i, server in enumerate(healthy_servers):
        config_code += f"""    {{
        "name": "gpu_server_{i+1}",
        "endpoint": "{server}",
        "gpu_id": {i},
        "max_concurrent_requests": 10,
        "timeout": 300
    }},
"""
    
    config_code += """]

# Then initialize the coordinator:
# Go to System Health tab and click "🚀 Initialize Coordinator"
"""
    
    return config_code

if __name__ == "__main__":
    print("Starting diagnostic scan...")
    
    results, healthy_servers = run_comprehensive_diagnostics()
    
    if healthy_servers:
        config = generate_streamlit_config(healthy_servers)
        print("\n" + "=" * 50)
        print("🛠️  CONFIGURATION CODE")
        print("=" * 50)
        print(config)
        
        # Save to file
        with open("gpu_server_config.txt", "w") as f:
            f.write(config)
        print("💾 Configuration saved to: gpu_server_config.txt")
    
    print("\n✅ Diagnostic complete!")
    
    # Exit codes for automation
    sys.exit(0 if healthy_servers else 1)