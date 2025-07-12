#!/usr/bin/env python3
"""
Enhanced Streamlit Application for Opulence API Research Agent - Part 1
Core setup, utilities, and single GPU initialization
"""
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import asyncio
import time
import traceback
import json
import os
import sqlite3
from datetime import datetime as dt
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import mimetypes
import hashlib
import tempfile
from pathlib import Path

# ============================================================================
# GLOBAL CONSTANTS AND CONFIGURATION
# ============================================================================

COORDINATOR_AVAILABLE = True
import_error = None

try:
    from api_opulence_coordinator import create_api_coordinator_from_config, APIOpulenceCoordinator
except ImportError as e:
    COORDINATOR_AVAILABLE = False
    import_error = str(e)


def cleanup_on_session_end():
    """Cleanup function to call when session ends"""
    try:
        if st.session_state.get('coordinator'):
            # Use sync cleanup for session end
            if hasattr(st.session_state.coordinator, 'cleanup'):
                st.session_state.coordinator.cleanup()
    except Exception as e:
        print(f"Cleanup error: {e}")  # Use print since st may not be available

# Register cleanup
import atexit
atexit.register(cleanup_on_session_end)

# Comprehensive mainframe file types and extensions
MAINFRAME_FILE_TYPES = {
    'cobol': {
        'extensions': ['.cbl', '.cob', '.cobol', '.cpy', '.copybook'],
        'description': 'COBOL Programs and Copybooks',
        'mime_types': ['text/plain', 'application/octet-stream'],
        'agent': 'code_parser'
    },
    'jcl': {
        'extensions': ['.jcl', '.job', '.proc', '.prc'],
        'description': 'Job Control Language',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    'pli': {
        'extensions': ['.pli', '.pl1', '.pls'],
        'description': 'PL/I Programs',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    'sql': {
        'extensions': ['.sql', '.db2', '.ddl', '.dml'],
        'description': 'SQL Scripts',
        'mime_types': ['text/plain', 'application/sql'],
        'agent': 'db2_comparator'
    },
    'data': {
        'extensions': ['.dat', '.txt', '.csv', '.tsv', '.fixed'],
        'description': 'Data Files',
        'mime_types': ['text/plain', 'text/csv', 'application/octet-stream'],
        'agent': 'data_loader'
    }
}

# Agent types
AGENT_TYPES = [
    'code_parser', 'chat_agent', 'vector_index', 'data_loader',
    'lineage_analyzer', 'logic_analyzer', 'documentation', 'db2_comparator'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def safe_run_async(coroutine, timeout=30):
    """FIXED: Simple version without timeout context manager"""
    import nest_asyncio
    nest_asyncio.apply()
    
    try:
        # Just run the coroutine without wait_for timeout
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coroutine)
            return loop.run_until_complete(task)
        except RuntimeError:
            return asyncio.run(coroutine)
            
    except Exception as e:
        st.error(f"Async operation failed: {str(e)}")
        return {"error": str(e)}    
def with_error_handling(func):
    """Decorator to add error handling to functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)
            return None
    return wrapper

def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'chat_history': [],
        'processing_history': [],
        'uploaded_files': [],
        'file_analysis_results': {},
        'agent_status': {agent: {'status': 'unknown', 'last_used': None, 'total_calls': 0, 'errors': 0} 
                        for agent in AGENT_TYPES},
        'model_servers': [],
        'coordinator': None,
        'debug_mode': False,
        'initialization_status': 'not_started',
        'import_error': import_error if not COORDINATOR_AVAILABLE else None,
        'auto_refresh_gpu': False,
        'gpu_refresh_interval': 10,
        'current_query': '',
        'analysis_results': {},
        'dashboard_metrics': {
            'files_processed': 0,
            'queries_answered': 0,
            'components_analyzed': 0,
            'system_uptime': 0
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def detect_single_gpu_servers():
    """Auto-detect single GPU server configurations - FIXED VERSION"""
    # FIXED: Use only your working server
    potential_endpoints = [
        "http://171.201.3.165:8100",  # Your working server
    ]
    detected_servers = []
    
    st.info("🔍 Scanning for GPU servers...")
    
    for i, endpoint in enumerate(potential_endpoints):
        try:
            with st.spinner(f"Checking {endpoint}..."):
                response = requests.get(f"{endpoint}/health", timeout=5)
                if response.status_code == 200:
                    detected_servers.append({
                        "name": "gpu_server_2",
                        "endpoint": endpoint,
                        "gpu_id": 2,
                        "max_concurrent_requests": 3,
                        "timeout": 120
                    })
                    st.success(f"✅ Detected server at {endpoint} (GPU 2)")
                    break  # Stop after finding the working server
        except requests.exceptions.ConnectionError:
            st.info(f"ℹ️ No server found at {endpoint}")
        except Exception as e:
            st.warning(f"⚠️ Error checking {endpoint}: {str(e)}")
    
    return detected_servers

def debug_initialization_state():
    """Debug helper to show initialization state"""
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Debug Information", expanded=False):
            st.json({
                "coordinator_available": COORDINATOR_AVAILABLE,
                "import_error": st.session_state.get('import_error'),
                "initialization_status": st.session_state.get('initialization_status'),
                "coordinator_exists": st.session_state.get('coordinator') is not None,
                "model_servers": st.session_state.get('model_servers', []),
                "agent_status": {k: v['status'] for k, v in st.session_state.get('agent_status', {}).items()}
            })


def validate_server_endpoint(endpoint: str, timeout: int = 5) -> Dict[str, Any]:
    """Validate a server endpoint and return detailed status"""
    try:
        response = requests.get(f"{endpoint}/health", timeout=timeout)
        if response.status_code == 200:
            try:
                status_response = requests.get(f"{endpoint}/status", timeout=timeout)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    gpu_info = status_data.get('gpu_info', {})
                    gpu_count = len(gpu_info)
                    model_name = status_data.get('model', 'Unknown')
                    
                    return {
                        "status": "healthy",
                        "message": f"Model: {model_name}, GPUs: {gpu_count}",
                        "response_time": response.elapsed.total_seconds(),
                        "accessible": True,
                        "detailed_status": status_data
                    }
                else:
                    return {
                        "status": "healthy",
                        "message": "Server healthy, detailed status unavailable",
                        "response_time": response.elapsed.total_seconds(),
                        "accessible": True,
                        "detailed_status": None
                    }
            except:
                return {
                    "status": "healthy",
                    "message": "Basic health check passed",
                    "response_time": response.elapsed.total_seconds(),
                    "accessible": True,
                    "detailed_status": None
                }
        else:
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}",
                "response_time": None,
                "accessible": False,
                "detailed_status": None
            }
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Connection failed: {str(e)}",
            "response_time": None,
            "accessible": False,
            "detailed_status": None
        }

def detect_mainframe_file_type(filename: str, content: str = None) -> Dict[str, Any]:
    """Detect mainframe file type from filename and content"""
    filename_lower = filename.lower()
    
    # Check by extension first
    for file_type, info in MAINFRAME_FILE_TYPES.items():
        for ext in info['extensions']:
            if filename_lower.endswith(ext.lower()):
                return {
                    'type': file_type,
                    'description': info['description'],
                    'agent': info['agent'],
                    'confidence': 'high'
                }
    
    # Content-based detection if no extension match
    if content:
        content_upper = content.upper()
        
        if any(keyword in content_upper for keyword in ['IDENTIFICATION DIVISION', 'PROGRAM-ID', 'WORKING-STORAGE']):
            return {
                'type': 'cobol',
                'description': 'COBOL Program (detected from content)',
                'agent': 'code_parser',
                'confidence': 'medium'
            }
        
        if any(keyword in content_upper for keyword in ['//JOB ', '//EXEC ', '//DD ']):
            return {
                'type': 'jcl',
                'description': 'JCL Job (detected from content)',
                'agent': 'code_parser',
                'confidence': 'medium'
            }
        
        if any(keyword in content_upper for keyword in ['CREATE TABLE', 'SELECT ', 'INSERT INTO']):
            return {
                'type': 'sql',
                'description': 'SQL Script (detected from content)',
                'agent': 'db2_comparator',
                'confidence': 'medium'
            }
    
    return {
        'type': 'data',
        'description': 'Generic File',
        'agent': 'code_parser',
        'confidence': 'low'
    }

def add_custom_css():
    """Add custom CSS styles"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 15px 0;
        border-bottom: 1px solid #bdc3c7;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-unknown { color: #6c757d; font-weight: bold; }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        background-color: #f8f9fa;
    }
    .analysis-result {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# ENHANCED INITIALIZATION FOR SINGLE GPU
# ============================================================================

@with_error_handling
async def init_api_coordinator_single_gpu():
    """FIXED: Enhanced coordinator initialization optimized for single GPU"""
    if not COORDINATOR_AVAILABLE:
        return {"error": "API Coordinator module not available"}
    
    # Ensure session state is initialized
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = None
    
    # Clean up existing coordinator first
    if st.session_state.coordinator is not None:    
    
        try:
            await st.session_state.coordinator.shutdown()
        except Exception as e:
            st.warning(f"Error cleaning up existing coordinator: {e}")
        finally:
            st.session_state.coordinator = None
    
    try:
        with st.spinner("🔍 Detecting available GPU servers..."):
            # FIXED: Force use your known working server
            if not st.session_state.model_servers:
                # First try auto-detection
                detected_servers = detect_single_gpu_servers()
                
                if detected_servers:
                    st.session_state.model_servers = detected_servers
                    st.info(f"🎯 Auto-detected {len(detected_servers)} GPU server(s)")
                else:
                    # FIXED: Fallback with correct GPU ID
                    st.session_state.model_servers = [{
                        "name": "main_gpu_server",
                        "endpoint": "http://171.201.3.165:8100",
                        "gpu_id": 2,  # FIXED: Use GPU ID 2
                        "max_concurrent_requests": 3,
                        "timeout": 120
                    }]
                    st.warning("⚠️ No servers detected, using known server with GPU ID 2")
            
            st.info(f"📋 Using servers: {st.session_state.model_servers}")
            
            # FIXED: Test server connectivity before creating coordinator
            st.info("🔍 Testing server connectivity...")
            server_config = st.session_state.model_servers[0]
            
            # Test the server before proceeding
            try:
                test_response = requests.get(f"{server_config['endpoint']}/health", timeout=10)
                if test_response.status_code != 200:
                    st.error(f"❌ Server health check failed: {test_response.status_code}")
                    return {"error": f"Server not healthy: {test_response.status_code}"}
                else:
                    st.success(f"✅ Server health check passed")
            except Exception as e:
                st.error(f"❌ Server connectivity test failed: {e}")
                return {"error": f"Server connectivity failed: {e}"}
            
            # Create coordinator with conservative settings
            st.info("🔧 Creating coordinator...")
            coordinator = create_api_coordinator_from_config(
                model_servers=st.session_state.model_servers,
                load_balancing_strategy="round_robin",  # Simple for single server
                max_retries=2,  # REDUCED retries
                connection_pool_size=5,  # SMALLER pool
                request_timeout=120,  # LONGER timeout
                circuit_breaker_threshold=5,  # MORE tolerance
                retry_delay=2.0  # LONGER delay between retries
            )
            
            # Initialize with validation
            st.info("🚀 Initializing coordinator...")
            await coordinator.initialize()
            st.success("✅ Coordinator initialized")
            
            # FIXED: More thorough validation
            st.info("🔍 Running validation...")
            validation_results = await validate_single_gpu_setup_fixed(coordinator)
            st.info(f"📊 Validation results: {validation_results}")
            
            if validation_results['success']:
                st.session_state.coordinator = coordinator
                st.session_state.initialization_status = "completed"
                
                # Initialize agent status more carefully
                for agent_type in AGENT_TYPES:
                    try:
                        agent = coordinator.get_agent(agent_type)
                        if agent:
                            st.session_state.agent_status[agent_type]['status'] = 'available'
                        else:
                            st.session_state.agent_status[agent_type]['status'] = 'unavailable'
                    except Exception as e:
                        st.session_state.agent_status[agent_type]['status'] = 'error'
                        st.session_state.agent_status[agent_type]['error_message'] = str(e)
                
                st.success("🎉 System fully initialized and validated!")
                return {"success": True, "validation": validation_results}
            else:
                st.error(f"❌ Validation failed: {validation_results['message']}")
                # Still keep coordinator for debugging
                st.session_state.coordinator = coordinator
                st.session_state.initialization_status = f"validation_failed: {validation_results['message']}"
                return {"success": False, "validation_error": validation_results['message']}
    
    except Exception as e:
        st.error(f"❌ Coordinator creation failed: {str(e)}")
        st.exception(e)
        
        # Clean up on failure
        if 'coordinator' in locals():
            try:
                await coordinator.shutdown()
            except:
                pass
        
        st.session_state.initialization_status = f"error: {str(e)}"
        return {"error": str(e)}
        
def cleanup_on_session_end():
    """Cleanup function to call when session ends"""
    try:
        if st.session_state.get('coordinator'):
            # Use sync cleanup for session end
            if hasattr(st.session_state.coordinator, 'cleanup'):
                st.session_state.coordinator.cleanup()
    except Exception as e:
        print(f"Cleanup error: {e}")  # Use print since st may not be available

    # Register cleanup
    import atexit
    atexit.register(cleanup_on_session_end)

async def validate_single_gpu_setup_fixed(coordinator) -> Dict[str, Any]:
    """FIXED: Validate single GPU coordinator setup with better error handling"""
    try:
        # Step 1: Check coordinator health
        health = coordinator.get_health_status()
        available_servers = health.get('available_servers', 0)
        total_servers = health.get('total_servers', 0)
        
        st.info(f"Health check: {available_servers}/{total_servers} servers available")
        
        if available_servers == 0:
            # Debug server status
            server_details = []
            for server in coordinator.load_balancer.servers:
                status_detail = {
                    'name': server.config.name,
                    'endpoint': server.config.endpoint,
                    'status': server.status.value,
                    'available': server.is_available(),
                    'active_requests': server.active_requests,
                    'max_requests': server.config.max_concurrent_requests
                }
                server_details.append(status_detail)
            
            return {
                'success': False,
                'message': f'No servers available. Server details: {server_details}',
                'servers': 0,
                'server_details': server_details
            }
        
        # Step 2: Test API call with minimal parameters
        st.info("Testing API call...")
        try:
            test_result = await coordinator.call_model_api(
                "Hello", 
                {
                    "max_tokens": 5,
                    "temperature": 0.1,
                    "stream": False
                }
            )
            
            if test_result and not test_result.get('error'):
                return {
                    'success': True,
                    'message': f'Validation successful with {available_servers} server(s)',
                    'servers': available_servers,
                    'test_result': test_result
                }
            else:
                return {
                    'success': False,
                    'message': f'API test failed: {test_result}',
                    'servers': available_servers
                }
                
        except Exception as api_error:
            return {
                'success': False,
                'message': f'API call exception: {str(api_error)}',
                'servers': available_servers
            }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Validation error: {str(e)}',
            'servers': 0
        }

def show_initialization_interface():
    """Show initialization interface optimized for single GPU"""
    st.markdown('<div class="sub-header">🚀 System Initialization</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.error("🔴 API Coordinator module not available")
        st.code(st.session_state.get('import_error', 'Unknown import error'))
        return
    debug_initialization_state()
    # Initialization status
    status = st.session_state.initialization_status
    
    if status == 'not_started':
        st.info("🟡 System not initialized")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Auto-Initialize (Recommended)", type="primary", use_container_width=True):
                with st.spinner("Initializing system..."):
                    result = safe_run_async(init_api_coordinator_single_gpu())
                    
                    if result and result.get('success'):
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        error_msg = result.get('error') if result else "Unknown error"
                        st.error(f"❌ Initialization failed: {error_msg}")
        
        with col2:
            if st.button("⚙️ Manual Configuration", use_container_width=True):
                st.session_state.show_manual_config = True
        
        # Manual configuration
        if st.session_state.get('show_manual_config', False):
            show_manual_server_configuration()
    
    elif status == 'completed':
        st.success("🟢 System initialized and ready")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart System"):
                restart_coordinator()
        
        with col2:
            if st.button("🏥 Health Check"):
                run_health_check()
        
        with col3:
            if st.button("📊 System Status"):
                st.session_state.show_system_status = True
        
        # Show system status if requested
        if st.session_state.get('show_system_status', False):
            show_system_status_summary()
    
    else:
        # Error or other status
        st.error(f"🔴 System Status: {status}")
        
        if st.button("🔄 Retry Initialization"):
            st.session_state.initialization_status = 'not_started'
            st.rerun()

def show_manual_server_configuration():
    """Show manual server configuration for single GPU"""
    st.markdown("#### ⚙️ Manual Server Configuration")
    
    with st.form("manual_server_config"):
        st.markdown("Configure your GPU server endpoint:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            endpoint = st.text_input("Server Endpoint", value="http://171.201.3.165:8100")
            gpu_id = st.number_input("GPU ID", min_value=0, value=0)
        
        with col2:
            server_name = st.text_input("Server Name", value="my_gpu_server")
            max_requests = st.number_input("Max Concurrent Requests", min_value=1, value=3)
        
        test_connection = st.checkbox("Test connection before adding", value=True)
        
        if st.form_submit_button("Add Server"):
            if test_connection:
                with st.spinner("Testing connection..."):
                    result = validate_server_endpoint(endpoint)
                    
                    if result['accessible']:
                        st.success(f"✅ Connection successful: {result['message']}")
                        add_server_to_config(server_name, endpoint, gpu_id, max_requests)
                    else:
                        st.error(f"❌ Connection failed: {result['message']}")
            else:
                add_server_to_config(server_name, endpoint, gpu_id, max_requests)

def add_server_to_config(name: str, endpoint: str, gpu_id: int, max_requests: int):
    """Add server to configuration"""
    new_server = {
        "name": name,
        "endpoint": endpoint,
        "gpu_id": gpu_id,
        "max_concurrent_requests": max_requests,
        "timeout": 300
    }
    
    st.session_state.model_servers.append(new_server)
    st.success(f"✅ Added server: {name}")

def restart_coordinator():
    """Restart the coordinator"""
    try:
        if st.session_state.coordinator:
            # Properly shutdown the coordinator
            safe_run_async(st.session_state.coordinator.shutdown())
        
        st.session_state.coordinator = None
        st.session_state.initialization_status = 'not_started'
        
        # Reset agent status
        for agent_type in AGENT_TYPES:
            st.session_state.agent_status[agent_type] = {
                'status': 'unknown', 
                'last_used': None, 
                'total_calls': 0, 
                'errors': 0
            }
        
        st.success("✅ System restarted successfully")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Restart failed: {str(e)}")

def run_health_check():
    """Run comprehensive health check"""
    if not st.session_state.coordinator:
        st.error("❌ No coordinator available")
        return
    
    try:
        health = st.session_state.coordinator.get_health_status()
        
        if health.get('status') == 'healthy':
            st.success(f"🟢 System Healthy - {health.get('available_servers', 0)} servers available")
        else:
            st.warning(f"⚠️ System Issues - {health.get('available_servers', 0)} servers available")
        
        # Show detailed health info
        with st.expander("📊 Detailed Health Information"):
            st.json(health)
    
    except Exception as e:
        st.error(f"❌ Health check failed: {str(e)}")

def show_system_status_summary():
    """Show system status summary"""
    if not st.session_state.coordinator:
        st.warning("No coordinator available")
        return
    
    try:
        health = st.session_state.coordinator.get_health_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", health.get('status', 'unknown').upper())
        
        with col2:
            st.metric("Available Servers", f"{health.get('available_servers', 0)}")
        
        with col3:
            st.metric("Active Agents", f"{health.get('active_agents', 0)}")
        
        with col4:
            uptime = health.get('uptime_seconds', 0)
            st.metric("Uptime", f"{uptime:.0f}s")
        
        # Agent status summary
        available_agents = sum(1 for status in st.session_state.agent_status.values() 
                             if status['status'] == 'available')
        total_agents = len(AGENT_TYPES)
        
        if available_agents == total_agents:
            st.success(f"🤖 All {total_agents} agents operational")
        elif available_agents > 0:
            st.warning(f"⚠️ {available_agents}/{total_agents} agents operational")
        else:
            st.error(f"❌ No agents operational")
    
    except Exception as e:
        st.error(f"❌ Status check failed: {str(e)}")

# ============================================================================
# ENHANCED DASHBOARD IMPLEMENTATION
# ============================================================================

def show_enhanced_dashboard():
    """Enhanced dashboard with comprehensive metrics and status"""
    st.markdown('<div class="sub-header">🏠 Enhanced System Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.warning("🟡 System not initialized. Please initialize in the sidebar.")
        show_initialization_interface()
        return
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📊 Real-time System Overview")
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Get system metrics
    system_metrics = get_dashboard_metrics()
    
    # Main metrics row
    show_main_metrics(system_metrics)
    
    # System health overview
    show_system_health_overview()
    
    # Two-column layout for detailed information
    col1, col2 = st.columns(2)
    
    with col1:
        show_processing_statistics()
        show_recent_activity()
    
    with col2:
        show_server_status_cards()
        show_agent_status_overview()
    
    # Performance charts
    show_performance_charts(system_metrics)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()

def get_dashboard_metrics():
    """Get comprehensive dashboard metrics"""
    try:
        if not st.session_state.coordinator:
            return {}
        
        # Get coordinator health and stats
        health = st.session_state.coordinator.get_health_status()
        stats = safe_run_async(st.session_state.coordinator.get_statistics())
        
        # Calculate processing metrics
        total_files = len(st.session_state.processing_history)
        successful_files = sum(1 for h in st.session_state.processing_history if h.get('status') == 'success')
        total_queries = len(st.session_state.chat_history)
        total_components = len(st.session_state.get('analysis_results', {}))
        
        # Calculate success rates
        file_success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
        
        # Server metrics
        available_servers = health.get('available_servers', 0)
        total_servers = health.get('total_servers', 0)
        
        # Agent metrics
        available_agents = sum(1 for status in st.session_state.agent_status.values() 
                             if status['status'] == 'available')
        
        # System uptime
        uptime = health.get('uptime_seconds', 0)
        
        return {
            'files_processed': total_files,
            'file_success_rate': file_success_rate,
            'queries_answered': total_queries,
            'components_analyzed': total_components,
            'available_servers': available_servers,
            'total_servers': total_servers,
            'available_agents': available_agents,
            'total_agents': len(AGENT_TYPES),
            'system_uptime': uptime,
            'health_status': health.get('status', 'unknown'),
            'api_calls': health.get('stats', {}).get('total_api_calls', 0),
            'system_stats': stats
        }
    
    except Exception as e:
        st.error(f"Failed to get dashboard metrics: {str(e)}")
        return {}

def show_main_metrics(metrics):
    """Show main dashboard metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        files_processed = metrics.get('files_processed', 0)
        success_rate = metrics.get('file_success_rate', 0)
        
        st.metric(
            label="📄 Files Processed",
            value=files_processed,
            delta=f"{success_rate:.1f}% success rate"
        )
    
    with col2:
        queries = metrics.get('queries_answered', 0)
        api_calls = metrics.get('api_calls', 0)
        
        st.metric(
            label="💬 Queries Answered", 
            value=queries,
            delta=f"{api_calls} API calls"
        )
    
    with col3:
        components = metrics.get('components_analyzed', 0)
        available_agents = metrics.get('available_agents', 0)
        total_agents = metrics.get('total_agents', 0)
        
        st.metric(
            label="🔍 Components Analyzed",
            value=components,
            delta=f"{available_agents}/{total_agents} agents ready"
        )
    
    with col4:
        uptime = metrics.get('system_uptime', 0)
        health = metrics.get('health_status', 'unknown')
        
        uptime_str = f"{uptime/3600:.1f}h" if uptime > 3600 else f"{uptime:.0f}s"
        st.metric(
            label="⏱️ System Uptime",
            value=uptime_str,
            delta=f"Status: {health}"
        )

def show_system_health_overview():
    """Show system health overview"""
    st.markdown("### 🏥 System Health Overview")
    
    try:
        health = st.session_state.coordinator.get_health_status()
        
        # Health indicator
        status = health.get('status', 'unknown')
        available_servers = health.get('available_servers', 0)
        total_servers = health.get('total_servers', 0)
        
        if status == 'healthy' and available_servers > 0:
            st.success(f"🟢 System Operational - {available_servers}/{total_servers} servers healthy")
        elif available_servers > 0:
            st.warning(f"⚠️ Partial Service - {available_servers}/{total_servers} servers available")
        else:
            st.error(f"🔴 Service Unavailable - {available_servers}/{total_servers} servers responding")
        
        # Quick health metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            server_health = "Healthy" if available_servers > 0 else "Unhealthy"
            st.metric("Server Health", server_health)
        
        with col2:
            available_agents = sum(1 for status in st.session_state.agent_status.values() 
                                 if status['status'] == 'available')
            agent_health = f"{available_agents}/{len(AGENT_TYPES)}"
            st.metric("Agent Health", agent_health)
        
        with col3:
            total_errors = sum(status.get('errors', 0) for status in st.session_state.agent_status.values())
            st.metric("Total Errors", total_errors)
        
        with col4:
            db_status = "Connected" if os.path.exists(st.session_state.coordinator.db_path) else "Disconnected"
            st.metric("Database", db_status)
    
    except Exception as e:
        st.error(f"Failed to get health overview: {str(e)}")

def show_processing_statistics():
    """Show processing statistics and trends"""
    st.markdown("#### 📈 Processing Statistics")
    
    if not st.session_state.processing_history:
        st.info("No processing history available yet")
        return
    
    # Processing summary
    history = st.session_state.processing_history
    total_files = len(history)
    successful = sum(1 for h in history if h.get('status') == 'success')
    failed = total_files - successful
    
    # Success rate chart
    success_data = pd.DataFrame({
        'Status': ['Successful', 'Failed'],
        'Count': [successful, failed],
        'Percentage': [successful/total_files*100, failed/total_files*100] if total_files > 0 else [0, 0]
    })
    
    fig = px.pie(success_data, values='Count', names='Status', 
                 title=f"Processing Success Rate ({total_files} files)",
                 color_discrete_map={'Successful': '#28a745', 'Failed': '#dc3545'})
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent processing times
    if len(history) > 0:
        recent_history = history[-10:]  # Last 10 files
        times = [h.get('processing_time', 0) for h in recent_history]
        files = [h.get('file_name', f'File {i}') for i, h in enumerate(recent_history)]
        
        time_df = pd.DataFrame({'File': files, 'Processing Time (s)': times})
        
        fig_time = px.bar(time_df, x='File', y='Processing Time (s)',
                          title="Recent Processing Times")
        fig_time.update_layout(height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig_time, use_container_width=True)

def show_recent_activity():
    """Show recent system activity"""
    st.markdown("#### 🕒 Recent Activity")
    
    # Combine recent activities from different sources
    activities = []
    
    # Recent file processing
    for h in st.session_state.processing_history[-5:]:
        activities.append({
            'time': h.get('timestamp', ''),
            'type': 'File Processing',
            'description': f"Processed {h.get('file_name', 'unknown')}",
            'status': h.get('status', 'unknown')
        })
    
    # Recent chat queries
    for h in st.session_state.chat_history[-3:]:
        activities.append({
            'time': h.get('timestamp', ''),
            'type': 'Chat Query',
            'description': f"Query: {h.get('query', '')[:50]}...",
            'status': 'completed'
        })
    
    # Recent analysis
    for name, result in list(st.session_state.get('analysis_results', {}).items())[-3:]:
        activities.append({
            'time': result.get('timestamp', ''),
            'type': 'Component Analysis',
            'description': f"Analyzed {name}",
            'status': result.get('status', 'unknown')
        })
    
    # Sort by time and show recent
    activities.sort(key=lambda x: x['time'], reverse=True)
    
    if activities:
        for activity in activities[:8]:  # Show last 8 activities
            status_icon = "✅" if activity['status'] == 'success' or activity['status'] == 'completed' else "❌"
            time_str = activity['time'][:19].replace('T', ' ') if activity['time'] else 'Unknown time'
            
            st.markdown(f"""
            <div class="chat-message">
                {status_icon} <strong>{activity['type']}</strong><br>
                {activity['description']}<br>
                <small>🕒 {time_str}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent activity")

def show_server_status_cards():
    """Show server status as cards"""
    st.markdown("#### 🖥️ Server Status")
    
    if not st.session_state.model_servers:
        st.info("No servers configured")
        return
    
    try:
        health = st.session_state.coordinator.get_health_status()
        server_stats = health.get('server_stats', {})
        
        for server_config in st.session_state.model_servers:
            server_name = server_config['name']
            stats = server_stats.get(server_name, {})
            
            # Determine status
            is_available = stats.get('available', False)
            status_color = "🟢" if is_available else "🔴"
            
            # Server card
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{status_color} {server_name}</h4>
                    <p><strong>Endpoint:</strong> {server_config['endpoint']}</p>
                    <p><strong>GPU ID:</strong> {server_config['gpu_id']}</p>
                    <p><strong>Active Requests:</strong> {stats.get('active_requests', 0)}</p>
                    <p><strong>Total Requests:</strong> {stats.get('total_requests', 0)}</p>
                    <p><strong>Success Rate:</strong> {stats.get('success_rate', 0):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Failed to get server status: {str(e)}")

def show_agent_status_overview():
    """Show agent status overview"""
    st.markdown("#### 🤖 Agent Status Overview")
    
    # Agent status summary
    status_counts = {'available': 0, 'error': 0, 'unavailable': 0, 'unknown': 0}
    
    for agent_type, status in st.session_state.agent_status.items():
        agent_status = status.get('status', 'unknown')
        status_counts[agent_status] = status_counts.get(agent_status, 0) + 1
    
    # Status chart
    status_df = pd.DataFrame([
        {'Status': status.title(), 'Count': count} 
        for status, count in status_counts.items() if count > 0
    ])
    
    if not status_df.empty:
        color_map = {
            'Available': '#28a745',
            'Error': '#dc3545', 
            'Unavailable': '#ffc107',
            'Unknown': '#6c757d'
        }
        
        fig = px.bar(status_df, x='Status', y='Count',
                     title="Agent Status Distribution",
                     color='Status',
                     color_discrete_map=color_map)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent usage statistics
    total_calls = sum(status.get('total_calls', 0) for status in st.session_state.agent_status.values())
    total_errors = sum(status.get('errors', 0) for status in st.session_state.agent_status.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Agent Calls", total_calls)
    with col2:
        st.metric("Total Agent Errors", total_errors)

def show_performance_charts(metrics):
    """Show performance charts"""
    st.markdown("### 📊 Performance Analytics")
    
    # Create sample time series data for demonstration
    # In a real implementation, this would come from historical data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing throughput over time
        st.markdown("#### 📈 Processing Throughput")
        
        # Generate sample data based on current metrics
        import numpy as np
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Simulate throughput data
        base_throughput = max(1, metrics.get('files_processed', 0) / 30)
        throughput_data = []
        
        for i, date in enumerate(dates):
            daily_files = max(0, base_throughput + np.random.normal(0, base_throughput * 0.3))
            throughput_data.append({
                'Date': date,
                'Files Processed': daily_files,
                'Success Rate': 85 + np.random.normal(0, 10)
            })
        
        throughput_df = pd.DataFrame(throughput_data)
        
        fig_throughput = px.line(throughput_df, x='Date', y='Files Processed',
                                title='Daily File Processing Throughput')
        fig_throughput.update_layout(height=300)
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    with col2:
        # Agent usage distribution
        st.markdown("#### 🤖 Agent Usage Distribution")
        
        agent_usage = []
        for agent_type, status in st.session_state.agent_status.items():
            calls = status.get('total_calls', 0)
            if calls > 0:
                agent_usage.append({
                    'Agent': agent_type.replace('_', ' ').title(),
                    'Calls': calls,
                    'Errors': status.get('errors', 0)
                })
        
        if agent_usage:
            usage_df = pd.DataFrame(agent_usage)
            fig_usage = px.bar(usage_df, x='Agent', y='Calls',
                              title='Agent Usage Statistics',
                              hover_data=['Errors'])
            fig_usage.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig_usage, use_container_width=True)
        else:
            st.info("No agent usage data available yet")
    
    # System resource utilization
    st.markdown("#### 💻 System Resource Utilization")
    
    try:
        # Get server metrics if available
        health = st.session_state.coordinator.get_health_status()
        server_stats = health.get('server_stats', {})
        
        if server_stats:
            resource_data = []
            
            for server_name, stats in server_stats.items():
                if stats.get('available', False):
                    resource_data.append({
                        'Server': server_name,
                        'Active Requests': stats.get('active_requests', 0),
                        'Avg Latency (s)': stats.get('average_latency', 0),
                        'Success Rate (%)': stats.get('success_rate', 0)
                    })
            
            if resource_data:
                resource_df = pd.DataFrame(resource_data)
                
                # Create subplots
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Active Requests', 'Average Latency', 'Success Rate'),
                    specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
                )
                
                # Add traces
                fig.add_trace(
                    go.Bar(x=resource_df['Server'], y=resource_df['Active Requests'], name='Active Requests'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=resource_df['Server'], y=resource_df['Avg Latency (s)'], name='Avg Latency'),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=resource_df['Server'], y=resource_df['Success Rate (%)'], name='Success Rate'),
                    row=1, col=3
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No server resource data available")
        else:
            st.info("No server statistics available")
    
    except Exception as e:
        st.error(f"Failed to get resource utilization: {str(e)}")

# ============================================================================
# DASHBOARD UTILITY FUNCTIONS
# ============================================================================

def update_dashboard_metrics():
    """Update dashboard metrics in session state"""
    try:
        if st.session_state.coordinator:
            metrics = get_dashboard_metrics()
            st.session_state.dashboard_metrics.update(metrics)
    except Exception as e:
        st.error(f"Failed to update dashboard metrics: {str(e)}")

def export_dashboard_data():
    """Export dashboard data to CSV"""
    try:
        # Prepare export data
        export_data = {
            'timestamp': dt.now().isoformat(),
            'system_metrics': st.session_state.dashboard_metrics,
            'processing_history': st.session_state.processing_history,
            'agent_status': st.session_state.agent_status
        }
        
        # Convert to JSON string for download
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Dashboard Data",
            data=json_data,
            file_name=f"opulence_dashboard_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Failed to export dashboard data: {str(e)}")

def reset_dashboard_metrics():
    """Reset dashboard metrics"""
    try:
        st.session_state.dashboard_metrics = {
            'files_processed': 0,
            'queries_answered': 0,
            'components_analyzed': 0,
            'system_uptime': 0
        }
        st.session_state.processing_history = []
        st.session_state.chat_history = []
        st.session_state.analysis_results = {}
        
        st.success("✅ Dashboard metrics reset successfully")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to reset metrics: {str(e)}")

# ============================================================================
# ENHANCED CHAT ANALYSIS IMPLEMENTATION
# ============================================================================

def show_enhanced_chat_analysis():
    """Enhanced chat analysis interface with conversation management"""
    st.markdown('<div class="sub-header">💬 Enhanced Chat Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in the sidebar.")
        show_initialization_interface()
        return
    
    # Chat interface layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🤖 Mainframe Code Assistant")
    
    with col2:
        # Chat controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
        
        if st.button("💾 Export Chat", use_container_width=True):
            export_chat_history()
    
    # Chat configuration
    with st.expander("⚙️ Chat Configuration", expanded=False):
        show_chat_configuration()
    
    # Main chat interface
    show_chat_interface()
    
    # Chat history and conversation management
    show_conversation_management()
    
    # Chat analytics
    if st.session_state.chat_history:
        show_chat_analytics()

def show_chat_configuration():
    """Show chat configuration options"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        response_mode = st.selectbox(
            "Response Mode",
            ["Detailed", "Concise", "Technical", "Business-friendly"],
            help="Choose the style of responses"
        )
        st.session_state.chat_response_mode = response_mode
    
    with col2:
        include_context = st.checkbox(
            "Include Context", 
            value=True,
            help="Include relevant code/data context in responses"
        )
        st.session_state.chat_include_context = include_context
    
    with col3:
        max_history = st.number_input(
            "Conversation History Length",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of previous messages to include as context"
        )
        st.session_state.chat_max_history = max_history

def show_chat_interface():
    """Show main chat interface"""
    # Chat input
    user_query = st.chat_input(
        "Ask about your mainframe code, data lineage, or system architecture...",
        key="chat_input"
    )
    
    # Suggested queries
    st.markdown("#### 💡 Suggested Queries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Analyze this program", use_container_width=True):
            user_query = "Can you analyze the structure and functionality of the COBOL programs in my system?"
    
    with col2:
        if st.button("🔍 Find data lineage", use_container_width=True):
            user_query = "Show me the data lineage for customer records in the system"
    
    with col3:
        if st.button("🏗️ System architecture", use_container_width=True):
            user_query = "Explain the overall architecture and data flow of the mainframe system"
    
    # Process user query
    if user_query:
        process_chat_query(user_query)
    
    # Display conversation
    display_chat_conversation()

def process_chat_query_fixed(query: str):
    """FIXED: Process user chat query with better error handling"""
    if not query.strip():
        return
    
    # Add user message to history
    user_message = {
        'role': 'user',
        'content': query,
        'timestamp': dt.now().isoformat(),
        'query_id': str(time.time())
    }
    
    st.session_state.chat_history.append(user_message)
    
    # Show processing indicator
    with st.spinner("🤖 Processing your query..."):
        try:
            # FIXED: Check coordinator first
            if not st.session_state.coordinator:
                raise RuntimeError("Coordinator not available")
            
            # FIXED: Check server availability
            health = st.session_state.coordinator.get_health_status()
            if health.get('available_servers', 0) == 0:
                raise RuntimeError("No model servers available")
            
            # Get conversation context
            conversation_history = get_conversation_context()
            
            # FIXED: Conservative query configuration
            query_config = {
                'response_mode': st.session_state.get('chat_response_mode', 'Concise'),  # Use concise mode
                'include_context': True,
                'max_history': 3  # Reduce history to avoid token limits
            }
            
            # Process with coordinator
            start_time = time.time()
            result = safe_run_async(
                st.session_state.coordinator.process_chat_query(
                    query, 
                    conversation_history,
                    **query_config
                ),
                timeout=120  # Longer timeout
            )
            
            processing_time = time.time() - start_time
            
            if result and not result.get('error'):
                # Add assistant response to history
                assistant_message = {
                    'role': 'assistant',
                    'content': result.get('response', 'No response generated'),
                    'response_type': result.get('response_type', 'general'),
                    'suggestions': result.get('suggestions', []),
                    'context_used': result.get('context_used', []),
                    'processing_time': processing_time,
                    'timestamp': dt.now().isoformat(),
                    'query_id': user_message['query_id']
                }
                
                st.session_state.chat_history.append(assistant_message)
                
                # Update agent status
                st.session_state.agent_status['chat_agent']['total_calls'] += 1
                st.session_state.agent_status['chat_agent']['last_used'] = dt.now().isoformat()
                st.session_state.agent_status['chat_agent']['status'] = 'available'
                
                st.success(f"✅ Response generated in {processing_time:.2f} seconds")
            
            else:
                error_message = result.get('error', 'Unknown error') if result else 'Processing failed'
                
                # Add error response
                error_response = {
                    'role': 'assistant',
                    'content': f"I apologize, but I encountered an error: {error_message}",
                    'response_type': 'error',
                    'processing_time': processing_time,
                    'timestamp': dt.now().isoformat(),
                    'query_id': user_message['query_id'],
                    'error': error_message
                }
                
                st.session_state.chat_history.append(error_response)
                st.session_state.agent_status['chat_agent']['errors'] += 1
                
                st.error(f"❌ Query failed: {error_message}")
        
        except Exception as e:
            error_msg = str(e)
            
            # Add exception response
            exception_response = {
                'role': 'assistant',
                'content': f"I encountered an unexpected error: {error_msg}",
                'response_type': 'exception',
                'timestamp': dt.now().isoformat(),
                'query_id': user_message['query_id'],
                'error': error_msg
            }
            
            st.session_state.chat_history.append(exception_response)
            st.session_state.agent_status['chat_agent']['errors'] += 1
            
            st.error(f"❌ Unexpected error: {error_msg}")
            
            # Show debug info if enabled
            if st.session_state.get('debug_mode', False):
                st.exception(e)
    
    # Rerun to show new messages
    st.rerun()

def test_manual_api_call():
    """Manual API call test for debugging"""
    if not st.session_state.get('coordinator'):
        st.error("❌ No coordinator available")
        return
    
    st.markdown("#### 🧪 Manual API Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_prompt = st.text_input("Test Prompt", value="Hello, how are you?")
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=100, value=10)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    with col2:
        gpu_id = st.number_input("GPU ID", min_value=0, max_value=10, value=2)
        timeout = st.number_input("Timeout", min_value=10, max_value=300, value=60)
    
    if st.button("🚀 Test API Call"):
        test_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            with st.spinner("Making API call..."):
                result = safe_run_async(
                    st.session_state.coordinator.call_model_api(
                        test_prompt, 
                        test_params, 
                        preferred_gpu_id=gpu_id
                    ),
                    timeout=timeout
                )
            
            if result and not result.get('error'):
                st.success("✅ API call successful!")
                st.json(result)
            else:
                st.error(f"❌ API call failed: {result}")
                
        except Exception as e:
            st.error(f"❌ API call exception: {str(e)}")
            st.exception(e)

def show_quick_diagnostic():
    """Show quick diagnostic in Streamlit sidebar"""
    st.sidebar.markdown("### 🔧 Quick Diagnostic")
    
    if st.sidebar.button("🏥 Health Check"):
        if st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                if health.get('available_servers', 0) > 0:
                    st.sidebar.success("✅ System Healthy")
                else:
                    st.sidebar.error("❌ No servers available")
                    st.sidebar.json(health)
            except Exception as e:
                st.sidebar.error(f"Health check failed: {e}")
        else:
            st.sidebar.error("❌ No coordinator")
    
    if st.sidebar.button("🧪 Test API"):
        test_manual_api_call()
    
    if st.sidebar.button("🐛 Debug Info"):
        debug_coordinator_state()

def get_conversation_context():
    """Get conversation context for chat agent"""
    max_history = st.session_state.get('chat_max_history', 5)
    
    # Get recent conversation history
    recent_history = st.session_state.chat_history[-(max_history * 2):] if st.session_state.chat_history else []
    
    # Format for chat agent
    formatted_history = []
    for message in recent_history:
        formatted_history.append({
            'role': message['role'],
            'content': message['content'],
            'timestamp': message.get('timestamp', ''),
            'response_type': message.get('response_type', 'general')
        })
    
    return formatted_history

def display_chat_conversation():
    """Display chat conversation"""
    st.markdown("#### 💬 Conversation")
    
    if not st.session_state.chat_history:
        st.info("👋 Welcome! Ask me anything about your mainframe system. I can help with:")
        st.markdown("""
        - **Code Analysis**: Understanding COBOL, JCL, and other mainframe programs
        - **Data Lineage**: Tracing how data flows through your system
        - **System Architecture**: Explaining overall system structure and dependencies
        - **Best Practices**: Recommendations for maintenance and optimization
        """)
        return
    
    # Display messages in chronological order
    for i, message in enumerate(st.session_state.chat_history):
        display_chat_message(message, i)

def display_chat_message(message: Dict[str, Any], index: int):
    """Display individual chat message"""
    role = message.get('role', 'unknown')
    content = message.get('content', '')
    timestamp = message.get('timestamp', '')
    
    # Format timestamp
    time_str = timestamp[:19].replace('T', ' ') if timestamp else 'Unknown time'
    
    if role == 'user':
        # User message
        st.markdown(f"""
        <div style="text-align: right; margin: 10px 0;">
            <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 15px; display: inline-block; max-width: 80%;">
                <strong>You</strong><br>
                {content}
            </div>
            <br><small style="color: #666;">🕒 {time_str}</small>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == 'assistant':
        # Assistant message
        response_type = message.get('response_type', 'general')
        processing_time = message.get('processing_time', 0)
        suggestions = message.get('suggestions', [])
        context_used = message.get('context_used', [])
        error = message.get('error')
        
        # Determine message style based on response type
        if response_type == 'error' or error:
            bg_color = "#f8d7da"
            border_color = "#dc3545"
            icon = "🚨"
        elif response_type == 'success':
            bg_color = "#d4edda"
            border_color = "#28a745"
            icon = "✅"
        else:
            bg_color = "#f8f9fa"
            border_color = "#6c757d"
            icon = "🤖"
        
        st.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 15px; border-left: 4px solid {border_color}; max-width: 90%;">
                <strong>{icon} Assistant</strong><br><br>
                {content}
            </div>
            <small style="color: #666;">
                🕒 {time_str} | ⏱️ {processing_time:.2f}s
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        # Show additional information
        if suggestions or context_used:
            with st.expander(f"📋 Additional Information (Message {index + 1})", expanded=False):
                
                if suggestions:
                    st.markdown("**💡 Suggestions:**")
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
                
                if context_used:
                    st.markdown("**📚 Context Used:**")
                    for context in context_used:
                        st.markdown(f"- {context}")

def show_conversation_management():
    """Show conversation management options"""
    st.markdown("#### 🔧 Conversation Management")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Conversation", use_container_width=True):
            save_conversation()
    
    with col2:
        if st.button("📤 Share Conversation", use_container_width=True):
            share_conversation()
    
    with col3:
        if st.button("🔄 New Conversation", use_container_width=True):
            start_new_conversation()
    
    with col4:
        if st.button("📊 Chat Stats", use_container_width=True):
            st.session_state.show_chat_stats = True

def show_chat_analytics():
    """Show chat analytics"""
    if not st.session_state.get('show_chat_stats', False):
        return
    
    st.markdown("#### 📊 Chat Analytics")
    
    # Basic statistics
    total_messages = len(st.session_state.chat_history)
    user_messages = sum(1 for msg in st.session_state.chat_history if msg.get('role') == 'user')
    assistant_messages = sum(1 for msg in st.session_state.chat_history if msg.get('role') == 'assistant')
    
    # Processing times
    processing_times = [msg.get('processing_time', 0) for msg in st.session_state.chat_history 
                       if msg.get('role') == 'assistant' and msg.get('processing_time')]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Response types
    response_types = {}
    for msg in st.session_state.chat_history:
        if msg.get('role') == 'assistant':
            resp_type = msg.get('response_type', 'general')
            response_types[resp_type] = response_types.get(resp_type, 0) + 1
    
    # Display analytics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", total_messages)
    
    with col2:
        st.metric("User Queries", user_messages)
    
    with col3:
        st.metric("Assistant Responses", assistant_messages)
    
    with col4:
        st.metric("Avg Response Time", f"{avg_processing_time:.2f}s")
    
    # Response type distribution
    if response_types:
        st.markdown("**Response Type Distribution:**")
        
        response_df = pd.DataFrame([
            {'Type': resp_type.title(), 'Count': count}
            for resp_type, count in response_types.items()
        ])
        
        fig = px.pie(response_df, values='Count', names='Type',
                     title="Response Types")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Processing time trends
    if len(processing_times) > 1:
        st.markdown("**Response Time Trends:**")
        
        time_df = pd.DataFrame({
            'Response #': range(1, len(processing_times) + 1),
            'Processing Time (s)': processing_times
        })
        
        fig_time = px.line(time_df, x='Response #', y='Processing Time (s)',
                          title="Response Time Over Conversation")
        fig_time.update_layout(height=300)
        st.plotly_chart(fig_time, use_container_width=True)

def clear_chat_history():
    """Clear chat history"""
    st.session_state.chat_history = []
    st.session_state.show_chat_stats = False
    st.success("✅ Chat history cleared")
    st.rerun()

def export_chat_history():
    """Export chat history"""
    try:
        if not st.session_state.chat_history:
            st.warning("No chat history to export")
            return
        
        # Prepare export data
        export_data = {
            'export_timestamp': dt.now().isoformat(),
            'total_messages': len(st.session_state.chat_history),
            'conversation_history': st.session_state.chat_history,
            'chat_settings': {
                'response_mode': st.session_state.get('chat_response_mode', 'Detailed'),
                'include_context': st.session_state.get('chat_include_context', True),
                'max_history': st.session_state.get('chat_max_history', 5)
            }
        }
        
        # Convert to JSON
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="💾 Download Chat History",
            data=json_data,
            file_name=f"opulence_chat_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Chat history ready for download")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def save_conversation():
    """Save conversation to database"""
    try:
        if not st.session_state.chat_history:
            st.warning("No conversation to save")
            return
        
        # Save to session state with a name
        conversation_name = f"Conversation_{dt.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'saved_conversations' not in st.session_state:
            st.session_state.saved_conversations = {}
        
        st.session_state.saved_conversations[conversation_name] = {
            'history': st.session_state.chat_history.copy(),
            'saved_at': dt.now().isoformat(),
            'message_count': len(st.session_state.chat_history)
        }
        
        st.success(f"✅ Conversation saved as '{conversation_name}'")
        
    except Exception as e:
        st.error(f"❌ Save failed: {str(e)}")

def share_conversation():
    """Share conversation (generate shareable link/code)"""
    try:
        if not st.session_state.chat_history:
            st.warning("No conversation to share")
            return
        
        # Generate a simple share code
        conversation_data = {
            'messages': len(st.session_state.chat_history),
            'timestamp': dt.now().isoformat()
        }
        
        share_code = hashlib.md5(json.dumps(conversation_data).encode()).hexdigest()[:8]
        
        st.info(f"🔗 Share Code: `{share_code}`")
        st.caption("Share this code with others to reference this conversation")
        
    except Exception as e:
        st.error(f"❌ Share generation failed: {str(e)}")

def start_new_conversation():
    """Start a new conversation"""
    if st.session_state.chat_history:
        # Ask for confirmation
        if st.button("⚠️ Confirm: Start New Conversation (current will be lost)"):
            st.session_state.chat_history = []
            st.session_state.show_chat_stats = False
            st.success("✅ New conversation started")
            st.rerun()
    else:
        st.info("Already in a new conversation")

# ============================================================================
# ADVANCED CHAT FEATURES
# ============================================================================

def show_advanced_chat_features():
    """Show advanced chat features"""
    st.markdown("#### 🔬 Advanced Chat Features")
    
    with st.expander("🧠 Knowledge Base Query", expanded=False):
        show_knowledge_base_query()
    
    with st.expander("🔍 Code Search & Analysis", expanded=False):
        show_code_search_interface()
    
    with st.expander("📊 Data Flow Analysis", expanded=False):
        show_data_flow_analysis()

def show_knowledge_base_query():
    """Show knowledge base query interface"""
    st.markdown("Search your processed codebase and documentation:")
    
    kb_query = st.text_input(
        "Knowledge Base Query",
        placeholder="e.g., 'customer data structures', 'error handling patterns'"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox(
            "Search Type",
            ["Semantic", "Keyword", "Code Pattern", "Data Structure"]
        )
    
    with col2:
        result_limit = st.number_input("Result Limit", min_value=1, max_value=50, value=10)
    
    if st.button("🔍 Search Knowledge Base") and kb_query:
        search_knowledge_base(kb_query, search_type, result_limit)

def search_knowledge_base(query: str, search_type: str, limit: int):
    """Search the knowledge base"""
    try:
        with st.spinner("🔍 Searching knowledge base..."):
            # Use vector index agent for search
            result = safe_run_async(
                st.session_state.coordinator.search_code_patterns(query, limit)
            )
            
            if result and result.get('status') == 'success':
                results = result.get('results', [])
                
                if results:
                    st.success(f"✅ Found {len(results)} results")
                    
                    for i, item in enumerate(results):
                        with st.expander(f"📄 Result {i+1}: {item.get('title', 'Unknown')}", expanded=False):
                            st.markdown(f"**Score:** {item.get('score', 0):.3f}")
                            st.markdown(f"**Type:** {item.get('type', 'Unknown')}")
                            st.code(item.get('content', 'No content'), language='cobol')
                            
                            if item.get('metadata'):
                                st.json(item['metadata'])
                else:
                    st.info("No results found for your query")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Search failed'
                st.error(f"❌ Search failed: {error_msg}")
    
    except Exception as e:
        st.error(f"❌ Knowledge base search error: {str(e)}")

def show_code_search_interface():
    """Show code search interface"""
    st.markdown("Search for specific code patterns and structures:")
    
    code_query = st.text_area(
        "Code Pattern",
        placeholder="Enter COBOL code pattern, SQL query, or JCL snippet to find similar code"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_types = st.multiselect(
            "File Types",
            ["COBOL", "JCL", "SQL", "PL/I", "CICS"],
            default=["COBOL"]
        )
    
    with col2:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    if st.button("🔍 Find Similar Code") and code_query:
        find_similar_code(code_query, file_types, similarity_threshold)

def find_similar_code(query: str, file_types: List[str], threshold: float):
    """Find similar code patterns"""
    try:
        with st.spinner("🔍 Searching for similar code..."):
            # Process through vector index agent
            vector_agent = st.session_state.coordinator.get_agent("vector_index")
            
            if vector_agent:
                result = safe_run_async(
                    vector_agent.semantic_search(query, top_k=20)
                )
                
                if result:
                    # Filter by similarity threshold and file types
                    filtered_results = []
                    for item in result:
                        if (item.get('score', 0) >= threshold and 
                            any(ft.lower() in item.get('type', '').lower() for ft in file_types)):
                            filtered_results.append(item)
                    
                    if filtered_results:
                        st.success(f"✅ Found {len(filtered_results)} similar code patterns")
                        
                        for i, item in enumerate(filtered_results):
                            similarity = item.get('score', 0) * 100
                            
                            with st.expander(f"📄 Match {i+1}: {similarity:.1f}% similar", expanded=False):
                                st.markdown(f"**File:** {item.get('source', 'Unknown')}")
                                st.markdown(f"**Type:** {item.get('type', 'Unknown')}")
                                st.markdown(f"**Similarity:** {similarity:.1f}%")
                                
                                code_content = item.get('content', 'No content available')
                                st.code(code_content, language='cobol')
                    else:
                        st.info(f"No code patterns found above {threshold*100:.0f}% similarity")
                else:
                    st.error("❌ Code search failed")
            else:
                st.error("❌ Vector index agent not available")
    
    except Exception as e:
        st.error(f"❌ Code search error: {str(e)}")

def show_data_flow_analysis():
    """Show data flow analysis interface"""
    st.markdown("Analyze data flow and dependencies:")
    
    analysis_target = st.text_input(
        "Analysis Target",
        placeholder="Enter field name, table name, or program name"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Field Lineage", "Program Dependencies", "Data Flow", "Impact Analysis"]
        )
    
    with col2:
        depth_level = st.selectbox(
            "Analysis Depth",
            ["Shallow (1 level)", "Medium (2-3 levels)", "Deep (full trace)"]
        )
    
    if st.button("🔍 Analyze Data Flow") and analysis_target:
        analyze_data_flow(analysis_target, analysis_type, depth_level)

def analyze_data_flow(target: str, analysis_type: str, depth: str):
    """Analyze data flow for target"""
    try:
        with st.spinner(f"🔍 Analyzing {analysis_type.lower()} for {target}..."):
            
            if analysis_type == "Field Lineage":
                lineage_agent = st.session_state.coordinator.get_agent("lineage_analyzer")
                if lineage_agent:
                    result = safe_run_async(
                        lineage_agent.analyze_field_lineage(target)
                    )
                    display_lineage_results(result, target)
                else:
                    st.error("❌ Lineage analyzer not available")
            
            elif analysis_type == "Program Dependencies":
                logic_agent = st.session_state.coordinator.get_agent("logic_analyzer")
                if logic_agent:
                    result = safe_run_async(
                        logic_agent.find_dependencies(target)
                    )
                    display_dependency_results(result, target)
                else:
                    st.error("❌ Logic analyzer not available")
            
            else:
                # Use component analysis for other types
                result = safe_run_async(
                    st.session_state.coordinator.analyze_component(target, analysis_type.lower())
                )
                display_component_analysis_results(result, target)
    
    except Exception as e:
        st.error(f"❌ Data flow analysis error: {str(e)}")

def display_lineage_results(result: Dict[str, Any], target: str):
    """Display lineage analysis results"""
    if not result:
        st.error("❌ No lineage data found")
        return
    
    st.success(f"✅ Lineage analysis completed for {target}")
    
    # Display lineage information
    if isinstance(result, dict):
        for key, value in result.items():
            if key == 'lineage_path' and isinstance(value, list):
                st.markdown("**📊 Data Lineage Path:**")
                for i, step in enumerate(value):
                    st.markdown(f"{i+1}. {step}")
            
            elif key == 'dependencies' and isinstance(value, list):
                st.markdown("**🔗 Dependencies:**")
                for dep in value:
                    st.markdown(f"- {dep}")
            
            elif key == 'usage_patterns':
                st.markdown("**📈 Usage Patterns:**")
                st.json(value)
    else:
        st.json(result)

def display_dependency_results(result: Dict[str, Any], target: str):
    """Display dependency analysis results"""
    if not result:
        st.error("❌ No dependency data found")
        return
    
    st.success(f"✅ Dependency analysis completed for {target}")
    
    # Create dependency visualization
    if isinstance(result, dict):
        dependencies = result.get('dependencies', [])
        
        if dependencies:
            # Create a simple dependency graph
            dep_data = []
            for dep in dependencies:
                if isinstance(dep, dict):
                    dep_data.append({
                        'Source': target,
                        'Target': dep.get('name', 'Unknown'),
                        'Type': dep.get('type', 'Unknown'),
                        'Relationship': dep.get('relationship', 'depends_on')
                    })
            
            if dep_data:
                dep_df = pd.DataFrame(dep_data)
                st.dataframe(dep_df, use_container_width=True)
        
        # Display other analysis results
        for key, value in result.items():
            if key != 'dependencies':
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                if isinstance(value, (list, dict)):
                    st.json(value)
                else:
                    st.markdown(str(value))

def display_component_analysis_results(result: Dict[str, Any], target: str):
    """Display component analysis results"""
    if not result:
        st.error("❌ No analysis data found")
        return
    
    status = result.get('status', 'unknown')
    
    if status == 'completed':
        st.success(f"✅ Component analysis completed for {target}")
    elif status == 'partial':
        st.warning(f"⚠️ Partial analysis completed for {target}")
    else:
        st.error(f"❌ Analysis failed for {target}")
    
    # Display analysis results
    analyses = result.get('analyses', {})
    
    for analysis_name, analysis_data in analyses.items():
        with st.expander(f"📊 {analysis_name.replace('_', ' ').title()}", expanded=True):
            
            if analysis_data.get('status') == 'success':
                data = analysis_data.get('data', {})
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        
                        if isinstance(value, list) and len(value) > 0:
                            for item in value:
                                st.markdown(f"- {item}")
                        elif isinstance(value, dict):
                            st.json(value)
                        else:
                            st.markdown(str(value))
                else:
                    st.json(data)
            else:
                st.error(f"❌ {analysis_name} failed: {analysis_data.get('error', 'Unknown error')}")

# ============================================================================
# ENHANCED COMPONENT ANALYSIS IMPLEMENTATION
# ============================================================================

def show_enhanced_component_analysis():
    """Enhanced component analysis with comprehensive investigation tools"""
    st.markdown('<div class="sub-header">🔍 Enhanced Component Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in the sidebar.")
        show_initialization_interface()
        return
    
    # Component analysis interface layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Component Investigation")
    
    with col2:
        # Analysis controls
        if st.button("📊 View Analysis History", use_container_width=True):
            st.session_state.show_analysis_history = True
        
        if st.button("📈 Analysis Reports", use_container_width=True):
            st.session_state.show_analysis_reports = True
    
    # Main analysis interface
    show_component_analysis_interface()
    
    # Analysis configuration
    with st.expander("⚙️ Analysis Configuration", expanded=False):
        show_analysis_configuration()
    
    # Quick analysis shortcuts
    show_quick_analysis_shortcuts()
    
    # Analysis results display
    show_analysis_results_display()
    
    # Analysis history and reports
    if st.session_state.get('show_analysis_history', False):
        show_analysis_history()
    
    if st.session_state.get('show_analysis_reports', False):
        show_analysis_reports()

def show_component_analysis_interface():
    """Show main component analysis interface"""
    st.markdown("#### 🎯 Component Target Selection")
    
    # Component input methods
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Database Search", "File Upload"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        show_manual_component_entry()
    elif input_method == "Database Search":
        show_database_component_search()
    else:
        show_file_component_extraction()

def show_manual_component_entry():
    """Show manual component entry interface"""
    col1, col2 = st.columns(2)
    
    with col1:
        component_name = st.text_input(
            "Component Name",
            placeholder="e.g., CUSTOMER-RECORD, CALC-INTEREST, PAY001",
            help="Enter the exact name of the component to analyze"
        )
        
        component_type = st.selectbox(
            "Component Type",
            ["Auto-detect", "Field", "Program", "COBOL", "JCL", "Copybook", "Table", "File"],
            help="Select the type of component or use auto-detect"
        )
    
    with col2:
        analysis_scope = st.selectbox(
            "Analysis Scope",
            ["Comprehensive", "Quick", "Custom"],
            help="Choose the depth of analysis"
        )
        
        include_dependencies = st.checkbox(
            "Include Dependencies",
            value=True,
            help="Analyze component dependencies and relationships"
        )
    
    if st.button("🚀 Start Analysis", type="primary") and component_name:
        start_component_analysis(
            component_name, 
            component_type if component_type != "Auto-detect" else None,
            analysis_scope,
            include_dependencies
        )

def show_database_component_search():
    """Show database component search interface"""
    st.markdown("Search for components in your processed database:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Search for component names, patterns, or descriptions"
        )
        
        search_type = st.selectbox(
            "Search Type",
            ["Name Match", "Pattern Match", "Description Search", "Full Text"]
        )
    
    with col2:
        component_filter = st.multiselect(
            "Component Types",
            ["Field", "Program", "COBOL", "JCL", "Copybook", "Table", "File"],
            help="Filter by component types"
        )
        
        result_limit = st.number_input(
            "Result Limit",
            min_value=1,
            max_value=100,
            value=20
        )
    
    if st.button("🔍 Search Components") and search_query:
        search_database_components(search_query, search_type, component_filter, result_limit)

def show_file_component_extraction():
    """Show file component extraction interface"""
    st.markdown("Extract components from uploaded files for analysis:")
    
    uploaded_file = st.file_uploader(
        "Upload File for Component Extraction",
        type=['cbl', 'cob', 'jcl', 'sql', 'txt'],
        help="Upload a mainframe file to extract and analyze its components"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            extraction_mode = st.selectbox(
                "Extraction Mode",
                ["All Components", "Programs Only", "Fields Only", "Dependencies Only"]
            )
        
        with col2:
            auto_analyze = st.checkbox(
                "Auto-analyze Extracted Components",
                value=True,
                help="Automatically start analysis for extracted components"
            )
        
        if st.button("📤 Extract Components"):
            extract_file_components(uploaded_file, extraction_mode, auto_analyze)

def show_analysis_configuration():
    """Show analysis configuration options"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Analysis Types:**")
        enable_lineage = st.checkbox("📊 Lineage Analysis", value=True)
        enable_logic = st.checkbox("🧠 Logic Analysis", value=True)
        enable_semantic = st.checkbox("🔍 Semantic Analysis", value=True)
        
        st.session_state.analysis_config = {
            'enable_lineage': enable_lineage,
            'enable_logic': enable_logic,
            'enable_semantic': enable_semantic
        }
    
    with col2:
        st.markdown("**Analysis Depth:**")
        lineage_depth = st.selectbox("Lineage Depth", ["Shallow", "Medium", "Deep"], index=1)
        dependency_levels = st.number_input("Dependency Levels", min_value=1, max_value=5, value=3)
        
        st.session_state.analysis_config.update({
            'lineage_depth': lineage_depth,
            'dependency_levels': dependency_levels
        })
    
    with col3:
        st.markdown("**Output Options:**")
        generate_report = st.checkbox("📄 Generate Report", value=True)
        save_results = st.checkbox("💾 Save to Database", value=True)
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "HTML"])
        
        st.session_state.analysis_config.update({
            'generate_report': generate_report,
            'save_results': save_results,
            'export_format': export_format
        })

def show_quick_analysis_shortcuts():
    """Show quick analysis shortcuts"""
    st.markdown("#### ⚡ Quick Analysis Shortcuts")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏢 Analyze Customer Data", use_container_width=True):
            start_predefined_analysis("customer", "Customer data components")
    
    with col2:
        if st.button("💰 Analyze Payment Flow", use_container_width=True):
            start_predefined_analysis("payment", "Payment processing components")
    
    with col3:
        if st.button("📊 Analyze Reports", use_container_width=True):
            start_predefined_analysis("report", "Reporting components")
    
    with col4:
        if st.button("🔄 Analyze Batch Jobs", use_container_width=True):
            start_predefined_analysis("batch", "Batch processing components")

def start_component_analysis(name: str, component_type: str, scope: str, include_deps: bool):
    """Start comprehensive component analysis"""
    try:
        with st.spinner(f"🔍 Analyzing component: {name}..."):
            start_time = time.time()
            
            # Determine analysis parameters based on scope
            if scope == "Quick":
                analysis_types = ["lineage_analysis"]
            elif scope == "Comprehensive":
                analysis_types = ["lineage_analysis", "logic_analysis", "semantic_analysis"]
            else:  # Custom
                config = st.session_state.get('analysis_config', {})
                analysis_types = []
                if config.get('enable_lineage', True):
                    analysis_types.append("lineage_analysis")
                if config.get('enable_logic', True):
                    analysis_types.append("logic_analysis")
                if config.get('enable_semantic', True):
                    analysis_types.append("semantic_analysis")
            
            # Start analysis with coordinator
            result = safe_run_async(
                st.session_state.coordinator.analyze_component(
                    name, 
                    component_type,
                    analysis_types=analysis_types,
                    include_dependencies=include_deps
                )
            )
            
            processing_time = time.time() - start_time
            
            if result:
                # Store results
                analysis_id = f"{name}_{int(time.time())}"
                
                st.session_state.analysis_results[analysis_id] = {
                    'component_name': name,
                    'component_type': component_type,
                    'analysis_scope': scope,
                    'result': result,
                    'processing_time': processing_time,
                    'timestamp': dt.now().isoformat(),
                    'analysis_id': analysis_id
                }
                
                # Update dashboard metrics
                st.session_state.dashboard_metrics['components_analyzed'] += 1
                
                # Show success and results
                status = result.get('status', 'unknown')
                
                if status == 'completed':
                    st.success(f"✅ Analysis completed for {name} in {processing_time:.2f} seconds")
                elif status == 'partial':
                    st.warning(f"⚠️ Partial analysis completed for {name} in {processing_time:.2f} seconds")
                else:
                    st.error(f"❌ Analysis failed for {name}")
                
                # Display results immediately
                display_analysis_result(result, name)
                
            else:
                st.error(f"❌ Analysis failed for {name}")
    
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")

def start_predefined_analysis(category: str, description: str):
    """Start predefined category analysis"""
    try:
        with st.spinner(f"🔍 Searching for {description}..."):
            # Search for components matching the category
            search_results = search_components_by_category(category)
            
            if search_results:
                st.success(f"✅ Found {len(search_results)} {description}")
                
                # Show found components
                st.markdown(f"#### 📋 Found {description.title()}:")
                
                selected_components = []
                
                for i, component in enumerate(search_results):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        if st.checkbox(f"Select", key=f"comp_{i}"):
                            selected_components.append(component)
                    
                    with col2:
                        st.markdown(f"**{component['name']}** ({component['type']})")
                        st.caption(component.get('description', 'No description'))
                    
                    with col3:
                        if st.button(f"Analyze", key=f"analyze_{i}"):
                            start_component_analysis(
                                component['name'],
                                component['type'],
                                "Comprehensive",
                                True
                            )
                
                if selected_components and st.button(f"🚀 Analyze Selected ({len(selected_components)})"):
                    analyze_multiple_components(selected_components)
            
            else:
                st.info(f"No {description} found in the database")
    
    except Exception as e:
        st.error(f"❌ Predefined analysis error: {str(e)}")

def search_database_components(query: str, search_type: str, filters: List[str], limit: int):
    """Search for components in database"""
    try:
        with st.spinner("🔍 Searching database..."):
            # Mock database search - in real implementation, query SQLite database
            results = mock_database_search(query, search_type, filters, limit)
            
            if results:
                st.success(f"✅ Found {len(results)} components")
                
                # Display results in a table
                results_df = pd.DataFrame(results)
                
                # Add selection column
                selected_indices = []
                
                for i, row in results_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
                    
                    with col1:
                        if st.checkbox("Select", key=f"search_result_{i}"):
                            selected_indices.append(i)
                    
                    with col2:
                        st.markdown(f"**{row['name']}**")
                        st.caption(row.get('description', 'No description'))
                    
                    with col3:
                        st.markdown(f"*{row['type']}*")
                    
                    with col4:
                        relevance = row.get('relevance', 0)
                        st.markdown(f"{relevance:.0f}%")
                    
                    with col5:
                        if st.button("Analyze", key=f"analyze_search_{i}"):
                            start_component_analysis(
                                row['name'],
                                row['type'],
                                "Comprehensive",
                                True
                            )
                
                # Bulk analysis option
                if selected_indices and st.button(f"🚀 Analyze Selected ({len(selected_indices)})"):
                    selected_components = [results[i] for i in selected_indices]
                    analyze_multiple_components(selected_components)
            
            else:
                st.info("No components found matching your search criteria")
    
    except Exception as e:
        st.error(f"❌ Database search error: {str(e)}")

def extract_file_components(uploaded_file, extraction_mode: str, auto_analyze: bool):
    """Extract components from uploaded file"""
    try:
        with st.spinner("📤 Extracting components from file..."):
            # Read file content
            file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            file_name = uploaded_file.name
            
            # Detect file type
            file_type_info = detect_mainframe_file_type(file_name, file_content)
            
            # Extract components based on file type and mode
            extracted_components = extract_components_from_content(
                file_content, 
                file_type_info, 
                extraction_mode
            )
            
            if extracted_components:
                st.success(f"✅ Extracted {len(extracted_components)} components from {file_name}")
                
                # Display extracted components
                st.markdown("#### 📋 Extracted Components:")
                
                selected_for_analysis = []
                
                for i, component in enumerate(extracted_components):
                    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                    
                    with col1:
                        if st.checkbox("Select", key=f"extract_{i}"):
                            selected_for_analysis.append(component)
                    
                    with col2:
                        st.markdown(f"**{component['name']}**")
                        st.caption(component.get('description', 'No description'))
                    
                    with col3:
                        st.markdown(f"*{component['type']}*")
                    
                    with col4:
                        if st.button("Analyze", key=f"analyze_extract_{i}"):
                            start_component_analysis(
                                component['name'],
                                component['type'],
                                "Comprehensive",
                                True
                            )
                
                # Auto-analyze if enabled
                if auto_analyze:
                    st.info("🔄 Auto-analyzing extracted components...")
                    analyze_multiple_components(extracted_components[:5])  # Limit to first 5
                
                # Manual bulk analysis
                elif selected_for_analysis and st.button(f"🚀 Analyze Selected ({len(selected_for_analysis)})"):
                    analyze_multiple_components(selected_for_analysis)
            
            else:
                st.warning(f"No components extracted from {file_name}")
    
    except Exception as e:
        st.error(f"❌ Component extraction error: {str(e)}")

def show_analysis_results_display():
    """Show analysis results display"""
    if not st.session_state.analysis_results:
        st.info("💡 No analysis results yet. Start analyzing components above.")
        return
    
    st.markdown("#### 📊 Recent Analysis Results")
    
    # Get recent results
    recent_results = list(st.session_state.analysis_results.items())[-5:]
    
    for analysis_id, analysis_data in recent_results:
        component_name = analysis_data['component_name']
        result = analysis_data['result']
        timestamp = analysis_data['timestamp']
        processing_time = analysis_data['processing_time']
        
        with st.expander(
            f"📄 {component_name} - {result.get('status', 'unknown').title()} "
            f"({processing_time:.2f}s)", 
            expanded=False
        ):
            display_analysis_result(result, component_name, show_metadata=True)

def display_analysis_result(result: Dict[str, Any], component_name: str, show_metadata: bool = False):
    """Display comprehensive analysis result"""
    if not result:
        st.error("No analysis results to display")
        return
    
    status = result.get('status', 'unknown')
    analyses = result.get('analyses', {})
    
    # Status indicator
    if status == 'completed':
        st.success(f"🎯 Complete analysis for **{component_name}**")
    elif status == 'partial':
        st.warning(f"⚠️ Partial analysis for **{component_name}**")
    else:
        st.error(f"❌ Failed analysis for **{component_name}**")
    
    # Metadata if requested
    if show_metadata:
        metadata = result.get('processing_metadata', {})
        if metadata:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration = metadata.get('total_duration_seconds', 0)
                st.metric("Duration", f"{duration:.2f}s")
            
            with col2:
                completed = metadata.get('analyses_completed', 0)
                total = metadata.get('analyses_total', 0)
                st.metric("Completed", f"{completed}/{total}")
            
            with col3:
                success_rate = metadata.get('success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Analysis results tabs
    if analyses:
        # Create tabs for different analysis types
        analysis_tabs = st.tabs([
            f"📊 {name.replace('_', ' ').title()}" 
            for name in analyses.keys()
        ])
        
        for tab, (analysis_name, analysis_data) in zip(analysis_tabs, analyses.items()):
            with tab:
                display_specific_analysis(analysis_name, analysis_data, component_name)
    
    # Component summary
    show_component_summary(result, component_name)

def display_specific_analysis(analysis_name: str, analysis_data: Dict[str, Any], component_name: str):
    """Display specific analysis results"""
    status = analysis_data.get('status', 'unknown')
    
    if status == 'success':
        data = analysis_data.get('data', {})
        agent_used = analysis_data.get('agent_used', 'unknown')
        completion_time = analysis_data.get('completion_time', 0)
        
        # Analysis header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Agent:** {agent_used}")
        with col2:
            st.markdown(f"**Time:** {completion_time:.2f}s")
        
        # Display based on analysis type
        if analysis_name == 'lineage_analysis':
            display_lineage_analysis(data, component_name)
        elif analysis_name == 'logic_analysis':
            display_logic_analysis(data, component_name)
        elif analysis_name == 'semantic_analysis':
            display_semantic_analysis(data, component_name)
        else:
            # Generic display for other analysis types
            st.json(data)
    
    elif status == 'error':
        error_msg = analysis_data.get('error', 'Unknown error')
        st.error(f"❌ {analysis_name} failed: {error_msg}")
    
    else:
        st.warning(f"⚠️ {analysis_name} status: {status}")

def display_lineage_analysis(data: Dict[str, Any], component_name: str):
    """Display lineage analysis results"""
    st.markdown("#### 📊 Data Lineage Analysis")
    
    # Lineage path
    if 'lineage_path' in data:
        st.markdown("**📈 Lineage Path:**")
        lineage_path = data['lineage_path']
        
        if isinstance(lineage_path, list) and lineage_path:
            for i, step in enumerate(lineage_path):
                icon = "🔹" if i % 2 == 0 else "🔸"
                st.markdown(f"{icon} {step}")
        else:
            st.info("No lineage path found")
    
    # Dependencies
    if 'dependencies' in data:
        dependencies = data['dependencies']
        if dependencies:
            st.markdown("**🔗 Dependencies:**")
            
            # Create dependency chart
            dep_data = []
            for dep in dependencies:
                if isinstance(dep, dict):
                    dep_data.append({
                        'Component': dep.get('name', 'Unknown'),
                        'Type': dep.get('type', 'Unknown'),
                        'Relationship': dep.get('relationship', 'Unknown')
                    })
            
            if dep_data:
                dep_df = pd.DataFrame(dep_data)
                st.dataframe(dep_df, use_container_width=True)
        else:
            st.info("No dependencies found")
    
    # Usage patterns
    if 'usage_patterns' in data:
        st.markdown("**📈 Usage Patterns:**")
        usage_patterns = data['usage_patterns']
        
        if isinstance(usage_patterns, dict):
            for pattern_name, pattern_data in usage_patterns.items():
                st.markdown(f"- **{pattern_name}**: {pattern_data}")
        else:
            st.json(usage_patterns)

def display_logic_analysis(data: Dict[str, Any], component_name: str):
    """Display logic analysis results"""
    st.markdown("#### 🧠 Logic Analysis")
    
    # Program structure
    if 'program_structure' in data:
        st.markdown("**🏗️ Program Structure:**")
        structure = data['program_structure']
        
        if isinstance(structure, dict):
            for section, details in structure.items():
                with st.expander(f"📂 {section.replace('_', ' ').title()}", expanded=False):
                    if isinstance(details, list):
                        for detail in details:
                            st.markdown(f"- {detail}")
                    else:
                        st.markdown(str(details))
    
    # Logic flows
    if 'logic_flows' in data:
        st.markdown("**🔄 Logic Flows:**")
        flows = data['logic_flows']
        
        if isinstance(flows, list):
            for i, flow in enumerate(flows):
                st.markdown(f"{i+1}. {flow}")
        else:
            st.json(flows)
    
    # Variables and operations
    if 'variables' in data:
        st.markdown("**📝 Variables:**")
        variables = data['variables']
        
        if isinstance(variables, list) and variables:
            var_df = pd.DataFrame(variables)
            st.dataframe(var_df, use_container_width=True)
        else:
            st.json(variables)

def display_semantic_analysis(data: Dict[str, Any], component_name: str):
    """Display semantic analysis results"""
    st.markdown("#### 🔍 Semantic Analysis")
    
    # Similar components
    if 'similar_components' in data:
        st.markdown("**🔗 Similar Components:**")
        similar = data['similar_components']
        
        if isinstance(similar, list) and similar:
            for component in similar:
                similarity = component.get('score', 0) * 100
                name = component.get('name', 'Unknown')
                comp_type = component.get('type', 'Unknown')
                
                st.markdown(f"- **{name}** ({comp_type}) - {similarity:.1f}% similar")
        else:
            st.info("No similar components found")
    
    # Semantic search results
    if 'semantic_search' in data:
        st.markdown("**🔍 Related Functionality:**")
        search_results = data['semantic_search']
        
        if isinstance(search_results, list) and search_results:
            for result in search_results:
                score = result.get('score', 0) * 100
                content = result.get('content', 'No content')
                source = result.get('source', 'Unknown source')
                
                with st.expander(f"📄 {source} ({score:.1f}% relevance)", expanded=False):
                    st.code(content[:500] + "..." if len(content) > 500 else content)
        else:
            st.info("No related functionality found")

def show_component_summary(result: Dict[str, Any], component_name: str):
    """Show component analysis summary"""
    st.markdown("#### 📋 Analysis Summary")
    
    # Component information
    component_type = result.get('component_type', 'Unknown')
    timestamp = result.get('analysis_timestamp', '')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Component:** {component_name}")
        st.markdown(f"**Type:** {component_type}")
    
    with col2:
        st.markdown(f"**Analysis Date:** {timestamp[:10] if timestamp else 'Unknown'}")
        status = result.get('status', 'Unknown')
        st.markdown(f"**Status:** {status.title()}")
    
    with col3:
        analyses = result.get('analyses', {})
        successful_analyses = sum(1 for a in analyses.values() if a.get('status') == 'success')
        st.markdown(f"**Analyses Completed:** {successful_analyses}/{len(analyses)}")
        
        if 'processing_metadata' in result:
            duration = result['processing_metadata'].get('total_duration_seconds', 0)
            st.markdown(f"**Total Time:** {duration:.2f}s")

def show_analysis_history():
    """Show analysis history"""
    st.markdown("#### 📈 Analysis History")
    
    if not st.session_state.analysis_results:
        st.info("No analysis history available")
        return
    
    # Analysis history table
    history_data = []
    for analysis_id, analysis_data in st.session_state.analysis_results.items():
        history_data.append({
            'Component': analysis_data['component_name'],
            'Type': analysis_data['component_type'],
            'Scope': analysis_data['analysis_scope'],
            'Status': analysis_data['result'].get('status', 'unknown'),
            'Duration': f"{analysis_data['processing_time']:.2f}s",
            'Timestamp': analysis_data['timestamp'][:19].replace('T', ' '),
            'ID': analysis_id
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=history_df['Status'].unique(),
            default=history_df['Status'].unique()
        )
    
    with col2:
        type_filter = st.multiselect(
            "Filter by Type",
            options=history_df['Type'].unique(),
            default=history_df['Type'].unique()
        )
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_results = {}
            st.rerun()
    
    # Apply filters
    filtered_df = history_df[
        (history_df['Status'].isin(status_filter)) &
        (history_df['Type'].isin(type_filter))
    ]
    
    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)
    
    # Re-analyze option
    if st.selectbox("Re-analyze Component", ["Select..."] + filtered_df['Component'].tolist()) != "Select...":
        selected_component = st.selectbox("Re-analyze Component", ["Select..."] + filtered_df['Component'].tolist())
        if selected_component != "Select..." and st.button("🔄 Re-analyze"):
            start_component_analysis(selected_component, None, "Comprehensive", True)

def show_analysis_reports():
    """Show analysis reports"""
    st.markdown("#### 📊 Analysis Reports")
    
    if not st.session_state.analysis_results:
        st.info("No analysis data available for reports")
        return
    
    # Report type selection
    report_type = st.selectbox(
        "Report Type",
        ["Summary Report", "Detailed Report", "Comparison Report", "Trend Analysis"]
    )
    
    if report_type == "Summary Report":
        generate_summary_report()
    elif report_type == "Detailed Report":
        generate_detailed_report()
    elif report_type == "Comparison Report":
        generate_comparison_report()
    else:
        generate_trend_analysis_report()

def generate_summary_report():
    """Generate summary report"""
    results = st.session_state.analysis_results
    
    # Summary statistics
    total_analyses = len(results)
    successful = sum(1 for r in results.values() if r['result'].get('status') == 'completed')
    avg_duration = sum(r['processing_time'] for r in results.values()) / total_analyses
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        st.metric("Success Rate", f"{(successful/total_analyses*100):.1f}%")
    
    with col3:
        st.metric("Avg Duration", f"{avg_duration:.2f}s")
    
    # Component type distribution
    type_counts = {}
    for analysis in results.values():
        comp_type = analysis.get('component_type', 'Unknown')
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
    
    if type_counts:
        type_df = pd.DataFrame([
            {'Type': comp_type, 'Count': count}
            for comp_type, count in type_counts.items()
        ])
        
        fig = px.pie(type_df, values='Count', names='Type',
                     title="Component Types Analyzed")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# UTILITY FUNCTIONS FOR COMPONENT ANALYSIS
# ============================================================================

def search_components_by_category(category: str) -> List[Dict[str, Any]]:
    """Mock function to search components by category"""
    # In real implementation, this would query the database
    mock_components = {
        'customer': [
            {'name': 'CUSTOMER-RECORD', 'type': 'Field', 'description': 'Main customer data structure'},
            {'name': 'CUST-VALIDATE', 'type': 'Program', 'description': 'Customer validation routine'},
            {'name': 'CUSTOMER-FILE', 'type': 'File', 'description': 'Customer master file'}
        ],
        'payment': [
            {'name': 'PAYMENT-CALC', 'type': 'Program', 'description': 'Payment calculation logic'},
            {'name': 'PAY-AMOUNT', 'type': 'Field', 'description': 'Payment amount field'},
            {'name': 'PROCESS-PAYMENT', 'type': 'Program', 'description': 'Payment processing routine'}
        ],
        'report': [
            {'name': 'MONTHLY-REPORT', 'type': 'Program', 'description': 'Monthly reporting program'},
            {'name': 'RPT-HEADER', 'type': 'Field', 'description': 'Report header structure'},
            {'name': 'GENERATE-RPT', 'type': 'Program', 'description': 'Report generation utility'}
        ],
        'batch': [
            {'name': 'BATCH-JOB', 'type': 'JCL', 'description': 'Main batch processing job'},
            {'name': 'BATCH-CONTROL', 'type': 'Program', 'description': 'Batch job controller'},
            {'name': 'JOB-STATUS', 'type': 'Field', 'description': 'Job execution status'}
        ]
    }
    
    return mock_components.get(category, [])

def mock_database_search(query: str, search_type: str, filters: List[str], limit: int) -> List[Dict[str, Any]]:
    """Mock database search function"""
    # In real implementation, this would query SQLite database
    mock_results = [
        {
            'name': f'COMPONENT-{i}',
            'type': filters[i % len(filters)] if filters else 'Program',
            'description': f'Mock component matching {query}',
            'relevance': max(50, 100 - i * 5)
        }
        for i in range(min(limit, 10))
    ]
    
    return mock_results

def extract_components_from_content(content: str, file_type_info: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    """Extract components from file content"""
    components = []
    file_type = file_type_info.get('type', 'unknown')
    
    if file_type == 'cobol':
        # Extract COBOL components
        if mode in ['All Components', 'Programs Only']:
            # Look for program name
            import re
            program_match = re.search(r'PROGRAM-ID\.\s+(\S+)', content, re.IGNORECASE)
            if program_match:
                components.append({
                    'name': program_match.group(1),
                    'type': 'Program',
                    'description': 'COBOL Program'
                })
        
        if mode in ['All Components', 'Fields Only']:
            # Look for field definitions
            field_matches = re.findall(r'^\s*\d+\s+([A-Z][A-Z0-9-]+)', content, re.MULTILINE)
            for field in field_matches[:10]:  # Limit to first 10
                components.append({
                    'name': field,
                    'type': 'Field',
                    'description': 'COBOL Data Field'
                })
    
    elif file_type == 'jcl':
        # Extract JCL components
        job_matches = re.findall(r'//(\w+)\s+JOB', content)
        for job in job_matches:
            components.append({
                'name': job,
                'type': 'JCL',
                'description': 'JCL Job'
            })
    
    return components

def analyze_multiple_components(components: List[Dict[str, Any]]):
    """Analyze multiple components"""
    try:
        total_components = len(components)
        st.info(f"🔄 Starting analysis of {total_components} components...")
        
        progress_bar = st.progress(0)
        
        for i, component in enumerate(components):
            start_component_analysis(
                component['name'],
                component['type'],
                "Quick",  # Use quick analysis for bulk processing
                False     # Skip dependencies for speed
            )
            
            # Update progress
            progress_bar.progress((i + 1) / total_components)
        
        st.success(f"✅ Completed analysis of {total_components} components")
        
    except Exception as e:
        st.error(f"❌ Bulk analysis error: {str(e)}")

def generate_detailed_report():
    """Generate detailed analysis report"""
    st.info("📄 Detailed report generation would be implemented here")

def generate_comparison_report():
    """Generate comparison report"""
    st.info("📊 Comparison report generation would be implemented here")

def generate_trend_analysis_report():
    """Generate trend analysis report"""
    st.info("📈 Trend analysis report generation would be implemented here")

# ============================================================================
# MAIN APPLICATION WITH ENHANCED NAVIGATION
# ============================================================================

def main():
    """Enhanced main application with comprehensive single GPU support"""
    
    # Page configuration
    st.set_page_config(
        page_title="Opulence Mainframe Analysis Platform",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Add custom CSS
    add_custom_css()
    
    # Header with system status
    show_application_header()
    
    # Enhanced sidebar navigation
    with st.sidebar:
        show_enhanced_sidebar()
    
    # Main content area
    show_main_content()
    
    # Footer with system information
    show_application_footer()

def show_application_header():
    """Show application header with system status"""
    st.markdown('<div class="main-header">🌐 Opulence Enhanced Mainframe Analysis Platform</div>', unsafe_allow_html=True)
    
    # System status bar
    if COORDINATOR_AVAILABLE and st.session_state.coordinator:
        show_header_status_bar()
    elif not COORDINATOR_AVAILABLE:
        st.error("⚠️ API Coordinator module not available - Please check installation")
    else:
        st.warning("🟡 System not initialized - Please initialize in the sidebar")

def show_header_status_bar():
    """Show condensed status bar in header"""
    try:
        health = st.session_state.coordinator.get_health_status()
        available_servers = health.get('available_servers', 0)
        total_servers = health.get('total_servers', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if available_servers > 0:
                st.success(f"🟢 {available_servers}/{total_servers} GPU Servers")
            else:
                st.error(f"🔴 {available_servers}/{total_servers} GPU Servers")
        
        with col2:
            available_agents = sum(1 for status in st.session_state.agent_status.values() 
                                 if status['status'] == 'available')
            total_agents = len(AGENT_TYPES)
            
            if available_agents == total_agents:
                st.success(f"🤖 {available_agents}/{total_agents} Agents")
            elif available_agents > 0:
                st.warning(f"🤖 {available_agents}/{total_agents} Agents")
            else:
                st.error(f"🤖 {available_agents}/{total_agents} Agents")
        
        with col3:
            files_processed = len(st.session_state.processing_history)
            if files_processed > 0:
                success_rate = sum(1 for h in st.session_state.processing_history 
                                 if h.get('status') == 'success') / files_processed * 100
                st.info(f"📄 {files_processed} files ({success_rate:.0f}% success)")
            else:
                st.info("📄 0 files processed")
        
        with col4:
            queries = len(st.session_state.chat_history)
            st.info(f"💬 {queries} queries answered")
    
    except Exception as e:
        st.warning(f"🟡 Status check failed: {str(e)}")

def debug_coordinator_state():
    """Debug coordinator state in Streamlit"""
    if not st.session_state.get('coordinator'):
        st.error("❌ No coordinator in session state")
        return
    
    coordinator = st.session_state.coordinator
    
    with st.expander("🐛 Coordinator Debug Information", expanded=True):
        # Basic coordinator info
        st.markdown("**Coordinator Status:**")
        
        try:
            health = coordinator.get_health_status()
            st.json(health)
        except Exception as e:
            st.error(f"Health check failed: {e}")
        
        # Server details
        st.markdown("**Server Details:**")
        for i, server in enumerate(coordinator.load_balancer.servers):
            st.markdown(f"**Server {i + 1}: {server.config.name}**")
            st.markdown(f"- Endpoint: {server.config.endpoint}")
            st.markdown(f"- GPU ID: {server.config.gpu_id}")
            st.markdown(f"- Status: {server.status.value}")
            st.markdown(f"- Available: {server.is_available()}")
            st.markdown(f"- Active requests: {server.active_requests}/{server.config.max_concurrent_requests}")
            st.markdown(f"- Total requests: {server.total_requests}")
            st.markdown(f"- Consecutive failures: {server.consecutive_failures}")
            
            if server.status.value == 'circuit_open':
                st.markdown(f"- Circuit opened at: {server.circuit_breaker_open_time}")
        
        # Load balancer info
        st.markdown("**Load Balancer:**")
        available_servers = coordinator.load_balancer.get_available_servers()
        st.markdown(f"- Available servers: {len(available_servers)}")
        st.markdown(f"- Strategy: {coordinator.config.load_balancing_strategy.value}")

def show_enhanced_sidebar():
    """Show enhanced sidebar with navigation and controls"""
    # Logo and branding
    st.image("https://via.placeholder.com/150x50/059669/ffffff?text=OPULENCE", use_container_width=True)
    
    # System initialization section
    with st.expander("🚀 System Control", expanded=not st.session_state.coordinator):
        show_sidebar_system_control()
    
    # Navigation
    st.markdown("### 📋 Navigation")
    
    page = st.selectbox(
        "Choose Page",
        [
            "🏠 Dashboard", 
            "📂 File Upload & Processing", 
            "💬 Chat Analysis", 
            "🔍 Component Analysis",
            "🤖 Agent Status",
            "🎮 GPU Monitoring",
            "⚙️ System Health",
            "📊 Analytics & Reports"
        ],
        key="main_navigation"
    )
    
    st.session_state.current_page = page
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            show_clear_options()
    
    # System information
    show_sidebar_system_info()
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        show_advanced_options()

def show_sidebar_system_control():
    """Show system control in sidebar"""
    if not COORDINATOR_AVAILABLE:
        st.error("❌ Coordinator Not Available")
        st.caption(f"Error: {st.session_state.get('import_error', 'Unknown error')}")
        return
    
    # Initialization status
    status = st.session_state.initialization_status
    
    if status == 'not_started':
        st.warning("🟡 System Not Initialized")
        if st.button("🚀 Initialize System", use_container_width=True, type="primary"):
            initialize_system()
    
    elif status == 'completed':
        st.success("🟢 System Ready")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart", use_container_width=True):
                restart_system()
        with col2:
            if st.button("🏥 Health", use_container_width=True):
                check_system_health()
    
    else:
        st.error(f"🔴 Status: {status}")
        if st.button("🔄 Retry", use_container_width=True):
            retry_initialization()

def show_sidebar_system_info():
    """Show system information in sidebar"""
    st.markdown("### 📊 System Info")
    
    try:
        if st.session_state.coordinator:
            health = st.session_state.coordinator.get_health_status()
            
            # System metrics
            st.metric("GPU Servers", f"{health.get('available_servers', 0)}")
            st.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
            
            # Processing stats
            files = len(st.session_state.processing_history)
            queries = len(st.session_state.chat_history)
            
            st.metric("Files Processed", files)
            st.metric("Queries Handled", queries)
        else:
            st.info("System not initialized")
    
    except Exception as e:
        st.error(f"Info error: {str(e)}")

def show_advanced_options():
    """Show advanced options"""
    # Debug mode
    debug_mode = st.checkbox(
        "🐛 Debug Mode", 
        value=st.session_state.get('debug_mode', False),
        help="Enable detailed error messages and logging"
    )
    st.session_state.debug_mode = debug_mode
    
    # Performance settings
    st.markdown("**Performance:**")
    
    auto_refresh = st.checkbox(
        "🔄 Auto-refresh", 
        value=st.session_state.get('auto_refresh_enabled', False),
        help="Automatically refresh data periodically"
    )
    st.session_state.auto_refresh_enabled = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh Interval (s)",
            min_value=5,
            max_value=60,
            value=10
        )
        st.session_state.refresh_interval = refresh_interval

def show_clear_options():
    """Show clear options modal"""
    st.markdown("#### 🧹 Clear Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared")
            st.rerun()
        
        if st.button("📄 Clear Files", use_container_width=True):
            st.session_state.processing_history = []
            st.session_state.uploaded_files = []
            st.success("File history cleared")
            st.rerun()
    
    with col2:
        if st.button("🔍 Clear Analysis", use_container_width=True):
            st.session_state.analysis_results = {}
            st.success("Analysis results cleared")
            st.rerun()
        
        if st.button("🧹 Clear All", use_container_width=True, type="secondary"):
            clear_all_data()

def show_main_content():
    """Show main content based on navigation"""
    page = st.session_state.get('current_page', '🏠 Dashboard')
    
    try:
        if page == "🏠 Dashboard":
            show_enhanced_dashboard()
        elif page == "📂 File Upload & Processing":
            show_enhanced_file_upload()
        elif page == "💬 Chat Analysis":
            show_enhanced_chat_analysis()
        elif page == "🔍 Component Analysis":
            show_enhanced_component_analysis()
        elif page == "🤖 Agent Status":
            show_comprehensive_agent_status()
        elif page == "🎮 GPU Monitoring":
            show_enhanced_gpu_monitoring()
        elif page == "⚙️ System Health":
            show_enhanced_system_health()
        elif page == "📊 Analytics & Reports":
            show_analytics_and_reports()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)

def show_application_footer():
    """Show application footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🌐 Opulence Mainframe Analysis Platform")
    
    with col2:
        if st.session_state.coordinator:
            health = st.session_state.coordinator.get_health_status()
            coordinator_type = health.get('coordinator_type', 'unknown')
            st.caption(f"🔧 Coordinator: {coordinator_type}")
        else:
            st.caption("🔧 Coordinator: Not initialized")
    
    with col3:
        current_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"🕒 {current_time}")

# ============================================================================
# ENHANCED FILE UPLOAD WITH MAINFRAME SUPPORT
# ============================================================================

def show_enhanced_file_upload():
    """Enhanced file upload with comprehensive mainframe support"""
    st.markdown('<div class="sub-header">📂 Mainframe File Upload & Processing</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in the sidebar.")
        return
    
    # File upload interface
    show_file_upload_interface()
    
    # Processing configuration
    with st.expander("⚙️ Processing Configuration", expanded=False):
        show_processing_configuration()
    
    # Processing history
    show_processing_history()

def show_file_upload_interface():
    """Show file upload interface"""
    st.markdown("#### 📤 Upload Mainframe Files")
    
    # File type information
    with st.expander("📋 Supported File Types", expanded=False):
        show_supported_file_types()
    
    # Upload area
    uploaded_files = st.file_uploader(
        "Choose mainframe files to upload",
        accept_multiple_files=True,
        type=None,  # Accept all file types
        help="Upload COBOL, JCL, SQL, PL/I, and other mainframe files"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)

def show_supported_file_types():
    """Show supported file types"""
    for file_type, info in MAINFRAME_FILE_TYPES.items():
        st.markdown(f"**{info['description']}**")
        st.markdown(f"- Extensions: {', '.join(info['extensions'])}")
        st.markdown(f"- Processed by: {info['agent']} agent")
        st.markdown("---")

def show_processing_configuration():
    """Show processing configuration"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_detect = st.checkbox("🔍 Auto-detect file types", value=True)
        parallel_processing = st.checkbox("⚡ Parallel processing", value=False)
    
    with col2:
        save_to_db = st.checkbox("💾 Save to database", value=True)
        generate_reports = st.checkbox("📄 Generate reports", value=False)
    
    with col3:
        processing_timeout = st.number_input("Timeout (seconds)", min_value=30, value=120)
        batch_size = st.number_input("Batch size", min_value=1, value=5)
    
    st.session_state.processing_config = {
        'auto_detect': auto_detect,
        'parallel_processing': parallel_processing,
        'save_to_db': save_to_db,
        'generate_reports': generate_reports,
        'timeout': processing_timeout,
        'batch_size': batch_size
    }

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    st.markdown("#### 📊 File Analysis & Processing")
    
    # Analyze uploaded files
    file_analysis = []
    
    for uploaded_file in uploaded_files:
        try:
            file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
        except:
            file_content = None
        
        file_type_info = detect_mainframe_file_type(uploaded_file.name, file_content)
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': file_type_info['type'],
            'description': file_type_info['description'],
            'agent': file_type_info['agent'],
            'confidence': file_type_info['confidence'],
            'hash': file_hash,
            'content_preview': file_content[:200] + '...' if file_content and len(file_content) > 200 else file_content,
            'upload_time': dt.now().isoformat()
        }
        
        file_analysis.append(file_info)
    
    # Display file analysis
    for i, file_info in enumerate(file_analysis):
        with st.expander(f"📄 {file_info['name']} ({file_info['size']} bytes)", expanded=False):
            show_file_analysis_details(file_info)
    
    # Processing controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Process All Files", type="primary", use_container_width=True):
            process_files_batch(uploaded_files, file_analysis)
    
    with col2:
        if st.button("👁️ Preview Processing", use_container_width=True):
            preview_processing(file_analysis)
    
    with col3:
        if st.button("📋 Save File List", use_container_width=True):
            save_file_list(file_analysis)

def show_file_analysis_details(file_info):
    """Show detailed file analysis"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Type:** {file_info['type'].upper()}")
        st.markdown(f"**Agent:** {file_info['agent']}")
    
    with col2:
        st.markdown(f"**Size:** {file_info['size']:,} bytes")
        st.markdown(f"**Confidence:** {file_info['confidence']}")
    
    with col3:
        st.markdown(f"**Hash:** {file_info['hash'][:8]}...")
        st.markdown(f"**Uploaded:** {file_info['upload_time'][:19]}")
    
    if file_info['content_preview']:
        st.markdown("**Content Preview:**")
        st.code(file_info['content_preview'], language='text')

def process_files_batch(uploaded_files, file_analysis):
    """Process files in batch"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        total_files = len(uploaded_files)
        results = []
        
        for i, (uploaded_file, file_info) in enumerate(zip(uploaded_files, file_analysis)):
            status_text.text(f"Processing {file_info['name']} ({i+1}/{total_files})...")
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f"_{file_info['name']}") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            try:
                # Process with coordinator
                start_time = time.time()
                
                result = safe_run_async(
                    st.session_state.coordinator.process_batch_files(
                        [Path(temp_file_path)], 
                        file_info['type']
                    )
                )
                
                processing_time = time.time() - start_time
                
                # Record result
                processing_result = {
                    'file_name': file_info['name'],
                    'file_type': file_info['type'],
                    'agent_used': file_info['agent'],
                    'status': 'success' if result and not result.get('error') else 'error',
                    'processing_time': processing_time,
                    'result': result,
                    'timestamp': dt.now().isoformat(),
                    'error': result.get('error') if result else 'Unknown error'
                }
                
                results.append(processing_result)
                st.session_state.processing_history.append(processing_result)
                
                # Update agent status
                agent_type = file_info['agent']
                st.session_state.agent_status[agent_type]['total_calls'] += 1
                st.session_state.agent_status[agent_type]['last_used'] = dt.now().isoformat()
                
                if processing_result['status'] == 'success':
                    st.session_state.agent_status[agent_type]['status'] = 'available'
                    with results_container:
                        st.success(f"✅ {file_info['name']} processed successfully in {processing_time:.2f}s")
                else:
                    st.session_state.agent_status[agent_type]['errors'] += 1
                    with results_container:
                        st.error(f"❌ {file_info['name']} processing failed: {processing_result['error']}")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            
            progress_bar.progress((i + 1) / total_files)
        
        # Final summary
        status_text.text("Processing complete!")
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = len(results) - success_count
        
        st.success(f"🎉 Processing Summary: {success_count} successful, {error_count} errors")
        
        # Update dashboard metrics
        st.session_state.dashboard_metrics['files_processed'] += success_count
        
    except Exception as e:
        st.error(f"❌ Batch processing failed: {str(e)}")

def show_processing_history():
    """Show processing history"""
    st.markdown("#### 📈 Processing History")
    
    if not st.session_state.processing_history:
        st.info("No files processed yet. Upload and process files to see history.")
        return
    
    # Summary metrics
    total_files = len(st.session_state.processing_history)
    successful = sum(1 for h in st.session_state.processing_history if h['status'] == 'success')
    failed = total_files - successful
    avg_time = sum(h.get('processing_time', 0) for h in st.session_state.processing_history) / total_files
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", total_files)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Failed", failed)
    with col4:
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    # Recent files table
    recent_files = st.session_state.processing_history[-20:]  # Last 20 files
    
    if recent_files:
        history_data = []
        for h in recent_files:
            history_data.append({
                'File': h['file_name'],
                'Type': h['file_type'].upper(),
                'Agent': h['agent_used'],
                'Status': '✅' if h['status'] == 'success' else '❌',
                'Time': f"{h.get('processing_time', 0):.2f}s",
                'Timestamp': h['timestamp'][:19].replace('T', ' ')
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Export Processing History"):
            export_processing_history()

# ============================================================================
# SYSTEM CONTROL FUNCTIONS
# ============================================================================

def initialize_system():
    """Initialize the system"""
    try:
        with st.spinner("🚀 Initializing system..."):
            result = safe_run_async(init_api_coordinator_single_gpu())
            
            if result and result.get('success'):
                st.success("✅ System initialized successfully!")
                time.sleep(1)  # Brief pause to show the message
                st.rerun()
            else:
                error_msg = result.get('error') if result else "Unknown error"
                st.error(f"❌ Initialization failed: {error_msg}")
                st.session_state.initialization_status = f"error: {error_msg}"
                
                # Show detailed error in debug mode
                if st.session_state.get('debug_mode', False):
                    st.json(result)
    except Exception as e:
        st.error(f"❌ System initialization exception: {str(e)}")
        st.session_state.initialization_status = f"exception: {str(e)}"
        
        if st.session_state.get('debug_mode', False):
            st.exception(e)

def restart_system():
    """Restart the system"""
    try:
        if st.session_state.coordinator:
            st.session_state.coordinator.cleanup()
        
        st.session_state.coordinator = None
        st.session_state.initialization_status = 'not_started'
        
        # Reset agent status
        for agent_type in AGENT_TYPES:
            st.session_state.agent_status[agent_type] = {
                'status': 'unknown', 
                'last_used': None, 
                'total_calls': 0, 
                'errors': 0
            }
        
        st.success("✅ System restarted successfully")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Restart failed: {str(e)}")

def retry_initialization():
    """Retry initialization"""
    st.session_state.initialization_status = 'not_started'
    st.rerun()

def check_system_health():
    """Check system health"""
    if not st.session_state.coordinator:
        st.error("❌ No coordinator available")
        return
    
    try:
        health = st.session_state.coordinator.get_health_status()
        
        if health.get('status') == 'healthy':
            st.success(f"🟢 System Healthy - {health.get('available_servers', 0)} servers available")
        else:
            st.warning(f"⚠️ System Issues - {health.get('available_servers', 0)} servers available")
        
        with st.expander("📊 Detailed Health Information"):
            st.json(health)
    
    except Exception as e:
        st.error(f"❌ Health check failed: {str(e)}")

def clear_all_data():
    """Clear all application data"""
    if st.button("⚠️ Confirm: Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.processing_history = []
        st.session_state.analysis_results = {}
        st.session_state.uploaded_files = []
        st.session_state.file_analysis_results = {}
        
        # Reset dashboard metrics
        st.session_state.dashboard_metrics = {
            'files_processed': 0,
            'queries_answered': 0,
            'components_analyzed': 0,
            'system_uptime': 0
        }
        
        st.success("✅ All data cleared successfully")
        st.rerun()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preview_processing(file_analysis):
    """Preview what processing would do"""
    st.markdown("#### 👁️ Processing Preview")
    
    for file_info in file_analysis:
        st.markdown(f"**{file_info['name']}**")
        st.markdown(f"- Will be processed by: {file_info['agent']} agent")
        st.markdown(f"- Detected as: {file_info['description']}")
        st.markdown(f"- Detection confidence: {file_info['confidence']}")
        st.markdown("---")

def save_file_list(file_analysis):
    """Save file list for later processing"""
    try:
        file_list = {
            'timestamp': dt.now().isoformat(),
            'files': file_analysis
        }
        
        json_data = json.dumps(file_list, indent=2)
        
        st.download_button(
            label="💾 Download File List",
            data=json_data,
            file_name=f"file_list_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ File list ready for download")
        
    except Exception as e:
        st.error(f"❌ Save failed: {str(e)}")

def export_processing_history():
    """Export processing history"""
    try:
        export_data = {
            'export_timestamp': dt.now().isoformat(),
            'total_files': len(st.session_state.processing_history),
            'processing_history': st.session_state.processing_history
        }
        
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="💾 Download Processing History",
            data=json_data,
            file_name=f"processing_history_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Processing history ready for download")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

# ============================================================================
# PLACEHOLDER IMPLEMENTATIONS
# ============================================================================

def show_comprehensive_agent_status():
    """Show comprehensive agent status - placeholder"""
    st.markdown("### 🤖 Agent Status & Monitoring")
    st.info("Comprehensive agent status implementation goes here")

def show_enhanced_gpu_monitoring():
    """Show enhanced GPU monitoring - placeholder"""
    st.markdown("### 🎮 GPU Monitoring & Performance")
    st.info("Enhanced GPU monitoring implementation goes here")

def show_enhanced_system_health():
    """Show enhanced system health - placeholder"""
    st.markdown("### ⚙️ System Health & Configuration")
    st.info("Enhanced system health implementation goes here")

def show_analytics_and_reports():
    """Show analytics and reports - placeholder"""
    st.markdown("### 📊 Analytics & Reports")
    st.info("Analytics and reports implementation goes here")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
        # Emergency debug mode
        st.markdown("### 🚨 Emergency Debug Information")
        st.json({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "coordinator_available": COORDINATOR_AVAILABLE,
            "session_state_keys": list(st.session_state.keys()) if 'st' in globals() else [],
            "mainframe_file_types_supported": len(MAINFRAME_FILE_TYPES),
            "agent_types_configured": len(AGENT_TYPES)
        })
        
        # Recovery options
        st.markdown("### 🔧 Recovery Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache"):
                if 'clear_all_data' in globals():
                    clear_all_data()
        
        with col3:
            if st.button("📊 Show Debug Info"):
                st.session_state.debug_mode = True
                st.rerun()