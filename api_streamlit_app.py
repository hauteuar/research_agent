#!/usr/bin/env python3
"""
FIXED Streamlit Application for Opulence API Research Agent
Addresses: timeout context manager issues, agent initialization failures, proper task timeouts
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
import concurrent.futures
import threading

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
# FIXED UTILITY FUNCTIONS
# ============================================================================

def safe_run_async(coroutine, timeout=120):
    """FIXED: Streamlit async runner with proper timeout handling and task management"""
    
    try:
        # CRITICAL FIX: Use ThreadPoolExecutor with asyncio for proper task isolation
        def run_in_thread():
            """Run coroutine in separate thread with new event loop and timeout"""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                # FIXED: Use asyncio.wait_for for timeout inside the task
                async def run_with_timeout():
                    return await asyncio.wait_for(coroutine, timeout=timeout)
                
                return new_loop.run_until_complete(run_with_timeout())
            except asyncio.TimeoutError:
                return {"error": f"Operation timed out after {timeout} seconds"}
            except Exception as e:
                return {"error": f"Execution failed: {str(e)}"}
            finally:
                # Clean shutdown of the loop
                try:
                    # Cancel remaining tasks
                    pending = asyncio.all_tasks(new_loop)
                    for task in pending:
                        task.cancel()
                    
                    if pending:
                        new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except:
                    pass
                finally:
                    new_loop.close()
        
        # Execute in thread with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_thread)
            try:
                return future.result(timeout=timeout + 10)  # Extra buffer for thread cleanup
            except concurrent.futures.TimeoutError:
                return {"error": f"Thread execution timed out after {timeout + 10} seconds"}
                
    except Exception as e:
        return {"error": f"Safe async execution failed: {str(e)}"}

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
    """FIXED: Conservative server detection"""
    detected_servers = [{
        "name": "gpu_server_2",
        "endpoint": "http://171.201.3.165:8100",
        "gpu_id": 2,
        "max_concurrent_requests": 1,
        "timeout": 180
    }]
    
    st.info("🎯 Using conservative single server configuration")
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
# FIXED INITIALIZATION FOR SINGLE GPU
# ============================================================================

@with_error_handling
async def init_api_coordinator_single_gpu_fixed():
    """FIXED: Ultra-conservative coordinator initialization with proper timeout handling"""
    
    if not COORDINATOR_AVAILABLE:
        return {"error": "API Coordinator module not available"}
    
    # Clean up existing coordinator
    if st.session_state.get('coordinator'):
        try:
            await asyncio.wait_for(st.session_state.coordinator.shutdown(), timeout=10)
        except (asyncio.TimeoutError, Exception) as e:
            st.warning(f"Cleanup warning: {e}")
        finally:
            st.session_state.coordinator = None
    
    try:
        with st.spinner("🔍 Setting up ultra-conservative configuration..."):
            # ULTRA-CONSERVATIVE: Force known working configuration
            st.session_state.model_servers = [{
                "name": "main_gpu_server",
                "endpoint": "http://171.201.3.165:8100",
                "gpu_id": 2,
                "max_concurrent_requests": 1,
                "timeout": 180
            }]
            
            st.info(f"📋 Using ultra-conservative server config: {st.session_state.model_servers}")
            
            # Test server connectivity FIRST with timeout
            st.info("🔍 Testing server connectivity...")
            server_config = st.session_state.model_servers[0]
            
            try:
                test_response = requests.get(f"{server_config['endpoint']}/health", timeout=15)
                if test_response.status_code != 200:
                    st.error(f"❌ Server health check failed: {test_response.status_code}")
                    return {"error": f"Server not healthy: {test_response.status_code}"}
                else:
                    st.success(f"✅ Server health check passed")
            except Exception as e:
                st.error(f"❌ Server connectivity test failed: {e}")
                return {"error": f"Server connectivity failed: {e}"}
            
            # Create coordinator with ULTRA-CONSERVATIVE settings
            st.info("🔧 Creating ultra-conservative coordinator...")
            
            # FIXED: Use the fixed coordinator with proper timeout handling
            coordinator = create_api_coordinator_from_config(
                model_servers=st.session_state.model_servers,
                load_balancing_strategy="round_robin",
                max_retries=1,
                connection_pool_size=2,
                request_timeout=120,  # Reduced from 180
                circuit_breaker_threshold=10,
                retry_delay=5.0,
                connection_timeout=30  # Reduced from 60
            )
            
            # FIXED: Initialize coordinator with timeout
            st.info("🚀 Initializing coordinator with timeout...")
            
            # Use asyncio.wait_for for proper timeout handling
            await asyncio.wait_for(coordinator.initialize(), timeout=60)
            
            st.success("✅ Coordinator initialized")
            
            # MINIMAL validation test with timeout
            st.info("🔍 Running minimal validation...")
            try:
                # Test with very tiny request and timeout
                test_result = await asyncio.wait_for(
                    coordinator.call_model_api(
                        "Hi", 
                        {
                            "max_tokens": 5,
                            "temperature": 0.1,
                            "stream": False
                        }
                    ),
                    timeout=30  # 30 second timeout for test
                )
                
                if test_result and not test_result.get('error'):
                    st.session_state.coordinator = coordinator
                    st.session_state.initialization_status = "completed"
                    
                    # Initialize agent status conservatively
                    for agent_type in AGENT_TYPES:
                        try:
                            st.session_state.agent_status[agent_type]['status'] = 'available'
                        except Exception as e:
                            st.session_state.agent_status[agent_type]['status'] = 'error'
                            st.session_state.agent_status[agent_type]['error_message'] = str(e)
                    
                    st.success("🎉 System fully initialized!")
                    return {"success": True}
                else:
                    st.error(f"❌ Validation failed: {test_result}")
                    return {"success": False, "error": f"Validation failed: {test_result}"}
                    
            except asyncio.TimeoutError:
                st.error(f"❌ Validation timeout after 30 seconds")
                # Still keep coordinator for debugging
                st.session_state.coordinator = coordinator
                st.session_state.initialization_status = "validation_timeout"
                return {"success": False, "error": "Validation timeout"}
            except Exception as validation_error:
                st.error(f"❌ Validation error: {validation_error}")
                # Still keep coordinator for debugging
                st.session_state.coordinator = coordinator
                st.session_state.initialization_status = f"validation_error: {validation_error}"
                return {"success": False, "error": f"Validation failed: {validation_error}"}
    
    except asyncio.TimeoutError:
        st.error(f"❌ Coordinator initialization timeout")
        st.session_state.initialization_status = "initialization_timeout"
        return {"error": "Coordinator initialization timeout"}
    except Exception as e:
        st.error(f"❌ Coordinator creation failed: {str(e)}")
        st.session_state.initialization_status = f"error: {str(e)}"
        return {"error": str(e)}

def cleanup_on_session_end():
    """Cleanup function to call when session ends"""
    try:
        if st.session_state.get('coordinator'):
            if hasattr(st.session_state.coordinator, 'cleanup'):
                st.session_state.coordinator.cleanup()
    except Exception as e:
        print(f"Cleanup error: {e}")

# Register cleanup
import atexit
atexit.register(cleanup_on_session_end)

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
                with st.spinner("Initializing system with proper timeouts..."):
                    result = safe_run_async(init_api_coordinator_single_gpu_fixed(), timeout=120)
                    
                    if result and result.get('success'):
                        st.success("✅ System initialized successfully!")
                        time.sleep(2)  # Brief pause to show success
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
            max_requests = st.number_input("Max Concurrent Requests", min_value=1, value=1)
        
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
        "timeout": 180
    }
    
    st.session_state.model_servers.append(new_server)
    st.success(f"✅ Added server: {name}")

def restart_coordinator():
    """Restart the coordinator"""
    try:
        if st.session_state.coordinator:
            # Use safe async for shutdown
            safe_run_async(st.session_state.coordinator.shutdown(), timeout=30)
        
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
# FIXED CHAT ANALYSIS IMPLEMENTATION
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
        process_chat_query_fixed(user_query)
    
    # Display conversation
    display_chat_conversation()

def process_chat_query_fixed(query: str):
    """FIXED: Process user chat query with proper timeout handling"""
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
            
            # FIXED: Ultra-conservative query configuration
            query_config = {
                'response_mode': st.session_state.get('chat_response_mode', 'Concise'),
                'include_context': True,
                'max_history': 2  # Reduce history to avoid token limits
            }
            
            # FIXED: Process with coordinator using safe_run_async with timeout
            start_time = time.time()
            
            # Create the coroutine for the chat query
            async def chat_query_task():
                return await st.session_state.coordinator.process_chat_query(
                    query, 
                    conversation_history,
                    **query_config
                )
            
            # Use safe_run_async with reduced timeout
            result = safe_run_async(chat_query_task(), timeout=60)
            
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

def show_application_header():
    """Show application header with system status"""
    st.markdown('<div class="main-header">🌐 Opulence Fixed Mainframe Analysis Platform</div>', unsafe_allow_html=True)
    
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

def show_enhanced_sidebar():
    """Show enhanced sidebar with navigation and controls"""
    # System initialization section
    with st.expander("🚀 System Control", expanded=not st.session_state.coordinator):
        show_sidebar_system_control()
    
    # Navigation
    st.markdown("### 📋 Navigation")
    
    page = st.selectbox(
        "Choose Page",
        [
            "🏠 Dashboard", 
            "💬 Chat Analysis", 
            "🔍 Component Analysis",
            "📂 File Upload & Processing", 
            "🤖 Agent Status",
            "⚙️ System Health"
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
        elif page == "💬 Chat Analysis":
            show_enhanced_chat_analysis()
        elif page == "🔍 Component Analysis":
            show_enhanced_component_analysis()
        elif page == "📂 File Upload & Processing":
            show_enhanced_file_upload()
        elif page == "🤖 Agent Status":
            show_comprehensive_agent_status()
        elif page == "⚙️ System Health":
            show_enhanced_system_health()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)

# ============================================================================
# SYSTEM CONTROL FUNCTIONS
# ============================================================================

def initialize_system():
    """Initialize the system"""
    try:
        with st.spinner("🚀 Initializing system with fixed timeouts..."):
            result = safe_run_async(init_api_coordinator_single_gpu_fixed(), timeout=120)
            
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
            safe_run_async(st.session_state.coordinator.shutdown(), timeout=30)
        
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
# PLACEHOLDER IMPLEMENTATIONS FOR MISSING FUNCTIONS
# ============================================================================

def show_enhanced_dashboard():
    """Show enhanced dashboard - simplified implementation"""
    st.markdown("### 🏠 Enhanced System Dashboard")
    
    if not st.session_state.coordinator:
        st.warning("🟡 System not initialized. Please initialize in the sidebar.")
        show_initialization_interface()
        return
    
    # Basic dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        files_processed = len(st.session_state.processing_history)
        st.metric("Files Processed", files_processed)
    
    with col2:
        queries = len(st.session_state.chat_history)
        st.metric("Queries Answered", queries)
    
    with col3:
        components = len(st.session_state.get('analysis_results', {}))
        st.metric("Components Analyzed", components)
    
    with col4:
        if st.session_state.coordinator:
            health = st.session_state.coordinator.get_health_status()
            uptime = health.get('uptime_seconds', 0)
            st.metric("System Uptime", f"{uptime:.0f}s")
        else:
            st.metric("System Uptime", "0s")
    
    # System health overview
    st.markdown("### 🏥 System Health Overview")
    
    try:
        if st.session_state.coordinator:
            health = st.session_state.coordinator.get_health_status()
            
            status = health.get('status', 'unknown')
            available_servers = health.get('available_servers', 0)
            total_servers = health.get('total_servers', 0)
            
            if status == 'healthy' and available_servers > 0:
                st.success(f"🟢 System Operational - {available_servers}/{total_servers} servers healthy")
            elif available_servers > 0:
                st.warning(f"⚠️ Partial Service - {available_servers}/{total_servers} servers available")
            else:
                st.error(f"🔴 Service Unavailable - {available_servers}/{total_servers} servers responding")
        else:
            st.error("🔴 Coordinator not initialized")
    
    except Exception as e:
        st.error(f"Failed to get system health: {str(e)}")

def show_enhanced_component_analysis():
    """Show enhanced component analysis - simplified implementation"""
    st.markdown("### 🔍 Enhanced Component Analysis")
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in the sidebar.")
        show_initialization_interface()
        return
    
    st.markdown("#### 🎯 Component Investigation")
    
    # Component input
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
        start_component_analysis_fixed(
            component_name, 
            component_type if component_type != "Auto-detect" else None,
            analysis_scope,
            include_dependencies
        )
    
    # Show analysis results
    if st.session_state.analysis_results:
        st.markdown("#### 📊 Recent Analysis Results")
        
        for analysis_id, analysis_data in list(st.session_state.analysis_results.items())[-5:]:
            component_name = analysis_data['component_name']
            result = analysis_data['result']
            timestamp = analysis_data['timestamp']
            processing_time = analysis_data['processing_time']
            
            with st.expander(
                f"📄 {component_name} - {result.get('status', 'unknown').title()} "
                f"({processing_time:.2f}s)", 
                expanded=False
            ):
                st.json(result)

def start_component_analysis_fixed(name: str, component_type: str, scope: str, include_deps: bool):
    """FIXED: Start component analysis with proper timeout handling"""
    try:
        with st.spinner(f"🔍 Analyzing component: {name}..."):
            start_time = time.time()
            
            # Create the coroutine for component analysis
            async def analysis_task():
                return await st.session_state.coordinator.analyze_component(
                    name, 
                    component_type,
                    include_dependencies=include_deps
                )
            
            # Use safe_run_async with timeout
            result = safe_run_async(analysis_task(), timeout=120)
            
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
                
                # Display basic results
                st.json(result)
                
            else:
                st.error(f"❌ Analysis failed for {name}")
    
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")

def show_enhanced_file_upload():
    """Show enhanced file upload - simplified implementation"""
    st.markdown("### 📂 Mainframe File Upload & Processing")
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in the sidebar.")
        return
    
    # File upload interface
    st.markdown("#### 📤 Upload Mainframe Files")
    
    uploaded_files = st.file_uploader(
        "Choose mainframe files to upload",
        accept_multiple_files=True,
        type=None,
        help="Upload COBOL, JCL, SQL, PL/I, and other mainframe files"
    )
    
    if uploaded_files:
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
        
        # Processing controls
        if st.button("🚀 Process All Files", type="primary", use_container_width=True):
            process_files_batch_fixed(uploaded_files, file_analysis)

def process_files_batch_fixed(uploaded_files, file_analysis):
    """FIXED: Process files in batch with proper timeout handling"""
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
                # Process with API-based coordinator using timeout
                start_time = time.time()
                
                # Create the coroutine for file processing
                async def file_processing_task():
                    return await st.session_state.coordinator.process_batch_files(
                        [Path(temp_file_path)], 
                        file_info['type']
                    )
                
                # Use safe_run_async with timeout
                result = safe_run_async(file_processing_task(), timeout=120)
                
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

def show_comprehensive_agent_status():
    """Show comprehensive agent status - simplified implementation"""
    st.markdown("### 🤖 Agent Status & Monitoring")
    
    if not st.session_state.coordinator:
        st.warning("🟡 System not initialized. Please initialize in the sidebar.")
        return
    
    # Agent status overview
    st.markdown("#### 📊 Agent Status Overview")
    
    # Status summary
    status_counts = {'available': 0, 'error': 0, 'unavailable': 0, 'unknown': 0}
    
    for agent_type, status in st.session_state.agent_status.items():
        agent_status = status.get('status', 'unknown')
        status_counts[agent_status] = status_counts.get(agent_status, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available", status_counts.get('available', 0))
    with col2:
        st.metric("Error", status_counts.get('error', 0))
    with col3:
        st.metric("Unavailable", status_counts.get('unavailable', 0))
    with col4:
        st.metric("Unknown", status_counts.get('unknown', 0))
    
    # Detailed agent status
    st.markdown("#### 🔍 Detailed Agent Status")
    
    for agent_type, status in st.session_state.agent_status.items():
        with st.expander(f"🤖 {agent_type.replace('_', ' ').title()}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Status:** {status['status']}")
                st.markdown(f"**Total Calls:** {status['total_calls']}")
            
            with col2:
                st.markdown(f"**Errors:** {status['errors']}")
                last_used = status['last_used']
                if last_used:
                    st.markdown(f"**Last Used:** {last_used[:19].replace('T', ' ')}")
                else:
                    st.markdown("**Last Used:** Never")
            
            with col3:
                if status['status'] == 'available':
                    st.success("✅ Available")
                elif status['status'] == 'error':
                    st.error("❌ Error")
                    if 'error_message' in status:
                        st.caption(f"Error: {status['error_message']}")
                else:
                    st.warning("⚠️ Unknown Status")

def show_enhanced_system_health():
    """Show enhanced system health - simplified implementation"""
    st.markdown("### ⚙️ System Health & Configuration")
    
    if not st.session_state.coordinator:
        st.warning("🟡 System not initialized. Please initialize in the sidebar.")
        return
    
    try:
        health = st.session_state.coordinator.get_health_status()
        
        # Overall health status
        st.markdown("#### 🏥 Overall Health Status")
        
        status = health.get('status', 'unknown')
        if status == 'healthy':
            st.success(f"🟢 System Status: {status.upper()}")
        else:
            st.error(f"🔴 System Status: {status.upper()}")
        
        # Server details
        st.markdown("#### 🖥️ Server Status")
        
        server_stats = health.get('server_stats', {})
        
        for server_name, stats in server_stats.items():
            with st.expander(f"🖥️ {server_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Endpoint:** {stats.get('endpoint', 'Unknown')}")
                    st.markdown(f"**Status:** {stats.get('status', 'Unknown')}")
                
                with col2:
                    st.markdown(f"**Active Requests:** {stats.get('active_requests', 0)}")
                    st.markdown(f"**Total Requests:** {stats.get('total_requests', 0)}")
                
                with col3:
                    st.markdown(f"**Success Rate:** {stats.get('success_rate', 0):.1f}%")
                    st.markdown(f"**Avg Latency:** {stats.get('average_latency', 0):.3f}s")
        
        # System statistics
        st.markdown("#### 📊 System Statistics")
        
        stats = health.get('stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Calls", stats.get('total_api_calls', 0))
        with col2:
            st.metric("Files Processed", stats.get('total_files_processed', 0))
        with col3:
            st.metric("Queries", stats.get('total_queries', 0))
        with col4:
            uptime = health.get('uptime_seconds', 0)
            st.metric("Uptime", f"{uptime:.0f}s")
    
    except Exception as e:
        st.error(f"❌ Failed to get system health: {str(e)}")

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