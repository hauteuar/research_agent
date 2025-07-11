# streamlit_app.py
"""
Opulence Deep Research Mainframe Agent - Streamlit UI (API-Based Version)
Updated for APIOpulenceCoordinator - No Direct GPU Management
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import time
from datetime import datetime as dt
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import sys
import os
import shutil
import sqlite3

# Set page config FIRST
st.set_page_config(
    page_title="Opulence - API-Based Deep Research Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = None
    if 'initialization_status' not in st.session_state:
        st.session_state.initialization_status = "pending"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'field_lineage_result' not in st.session_state:
        st.session_state.field_lineage_result = None
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'model_servers' not in st.session_state:
        st.session_state.model_servers = [
            {"endpoint": "http://localhost:8000", "gpu_id": 1, "name": "gpu_1"},
            {"endpoint": "http://localhost:8001", "gpu_id": 2, "name": "gpu_2"}
        ]

initialize_session_state()

# Try to import API coordinator
try:
    from api_opulence_coordinator import (
        APIOpulenceCoordinator,
        APIOpulenceConfig,
        ModelServerConfig,
        LoadBalancingStrategy,
        create_api_coordinator_from_config,
        get_global_api_coordinator
    )
    COORDINATOR_AVAILABLE = True
except ImportError as e:
    COORDINATOR_AVAILABLE = False
    st.session_state.import_error = str(e)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3730a3;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }

    /* API indicator styling */
    .api-indicator {
        background: linear-gradient(45deg, #059669, #0d9488, #0891b2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }

    .server-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .server-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #059669;
    }
</style>
""", unsafe_allow_html=True)

def show_debug_info():
    """Show debug information"""
    if st.session_state.debug_mode:
        st.sidebar.markdown("### 🐛 Debug Info")
        debug_info = {
            "coordinator_available": COORDINATOR_AVAILABLE,
            "coordinator_type": "api_based" if st.session_state.coordinator else None,
            "coordinator_exists": st.session_state.coordinator is not None,
            "init_status": st.session_state.initialization_status,
            "import_error": st.session_state.get('import_error', 'None'),
            "model_servers": st.session_state.model_servers,
            "session_keys": list(st.session_state.keys())
        }

        if st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                debug_info.update({
                    "available_servers": health.get('available_servers'),
                    "total_servers": health.get('total_servers'),
                    "coordinator_status": health.get('status'),
                    "active_agents": health.get('active_agents', 0),
                    "load_balancing": health.get('load_balancing_strategy')
                })
            except:
                debug_info["coordinator_status"] = "error_getting_status"

        st.sidebar.json(debug_info)

async def init_api_coordinator():
    """Initialize the API coordinator"""
    if not COORDINATOR_AVAILABLE:
        return False

    if st.session_state.coordinator is None:
        try:
            # Create API coordinator with configured servers
            coordinator = create_api_coordinator_from_config(
                model_servers=st.session_state.model_servers,
                load_balancing_strategy="least_busy"
            )

            # Initialize the coordinator
            await coordinator.initialize()

            st.session_state.coordinator = coordinator
            st.session_state.initialization_status = "completed"
            return True

        except Exception as e:
            st.session_state.initialization_status = f"error: {str(e)}"
            return False
    return True

def safe_run_async(coro):
    """Safely run async functions in Streamlit"""
    if not hasattr(coro, '__await__'):
        return coro

    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new thread for async operation
            import threading
            result = {}
            exception = {}

            def run_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result['value'] = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception['error'] = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if 'error' in exception:
                raise exception['error']
            return result.get('value')
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        return {"error": str(e)}

def process_chat_query(query: str) -> Dict[str, Any]:
    """Process chat query and return structured response"""
    if not COORDINATOR_AVAILABLE:
        return {
            "response": "❌ API Coordinator not available. Please check the import error in debug mode.",
            "response_type": "error",
            "suggestions": []
        }

    if not st.session_state.coordinator:
        return {
            "response": "❌ System not initialized. Please check system health.",
            "response_type": "error",
            "suggestions": ["Initialize system in System Health tab"]
        }

    try:
        # Get conversation history for context
        conversation_history = st.session_state.chat_history[-5:] if st.session_state.chat_history else []

        # Process with the API coordinator
        result = safe_run_async(
            st.session_state.coordinator.process_chat_query(query, conversation_history)
        )

        if isinstance(result, dict):
            # Add server info to response
            if "servers_used" not in result:
                health = st.session_state.coordinator.get_health_status()
                result["servers_used"] = list(health.get('server_stats', {}).keys())
            return result
        else:
            return {
                "response": str(result),
                "response_type": "general",
                "suggestions": [],
                "servers_used": "unknown"
            }

    except Exception as e:
        return {
            "response": f"❌ Error processing query: {str(e)}",
            "response_type": "error",
            "suggestions": ["Try rephrasing your question", "Check system status"],
            "servers_used": "unknown"
        }

def show_server_status():
    """Show current server status for API system"""
    if st.session_state.coordinator:
        try:
            health = st.session_state.coordinator.get_health_status()

            # API indicator
            available_servers = health.get('available_servers', 0)
            total_servers = health.get('total_servers', 0)
            st.markdown(f"""
                <div class="api-indicator">
                    🌐 API Mode: {available_servers}/{total_servers} Servers Available
                    {"🟢 Healthy" if health.get('status') == 'healthy' else "🔴 Issues"}
                </div>
            """, unsafe_allow_html=True)

            # Server metrics grid
            server_stats = health.get('server_stats', {})
            if server_stats:
                st.markdown("### Server Status")

                # Create columns for servers
                num_servers = len(server_stats)
                if num_servers > 0:
                    cols = st.columns(min(num_servers, 3))

                    for i, (server_name, stats) in enumerate(server_stats.items()):
                        with cols[i % 3]:
                            st.markdown(f"#### {server_name}")

                            # Status indicator
                            status = stats.get('status', 'unknown')
                            if status == 'healthy':
                                st.success("🟢 Healthy")
                            elif status == 'unhealthy':
                                st.error("🔴 Unhealthy")
                            else:
                                st.warning(f"🟡 {status}")

                            # Metrics
                            st.metric("Active Requests", stats.get('active_requests', 0))
                            st.metric("Total Requests", stats.get('total_requests', 0))

                            success_rate = stats.get('success_rate', 0)
                            st.metric("Success Rate", f"{success_rate:.1f}%")

                            avg_latency = stats.get('average_latency', 0)
                            st.metric("Avg Latency", f"{avg_latency:.3f}s")

            # Combined system metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Load Balancing", health.get('load_balancing_strategy', 'unknown'))

            with col2:
                uptime = health.get('uptime_seconds', 0)
                st.metric("System Uptime", f"{uptime:.0f}s")

            with col3:
                st.metric("Active Agents", health.get('active_agents', 0))

            with col4:
                total_api_calls = health.get('stats', {}).get('total_api_calls', 0)
                st.metric("Total API Calls", total_api_calls)

        except Exception as e:
            st.error(f"Error getting server status: {str(e)}")
    else:
        st.warning("🟡 API Coordinator not initialized")

def configure_model_servers():
    """Configure model servers interface"""
    st.markdown("### 🌐 Configure Model Servers")

    # Server configuration form
    with st.form("server_config"):
        st.markdown("#### Add/Edit Server")

        col1, col2, col3 = st.columns(3)

        with col1:
            server_name = st.text_input("Server Name", value="gpu_1")

        with col2:
            endpoint = st.text_input("Endpoint", value="http://localhost:8000")

        with col3:
            gpu_id = st.number_input("GPU ID", min_value=0, value=1)

        col1, col2 = st.columns(2)

        with col1:
            max_requests = st.number_input("Max Concurrent Requests", min_value=1, value=10)

        with col2:
            timeout = st.number_input("Timeout (seconds)", min_value=30, value=300)

        if st.form_submit_button("Add Server"):
            new_server = {
                "name": server_name,
                "endpoint": endpoint,
                "gpu_id": gpu_id,
                "max_concurrent_requests": max_requests,
                "timeout": timeout
            }

            # Check if server already exists
            existing_names = [s.get('name', '') for s in st.session_state.model_servers]
            if server_name in existing_names:
                # Update existing server
                for i, server in enumerate(st.session_state.model_servers):
                    if server.get('name') == server_name:
                        st.session_state.model_servers[i] = new_server
                        break
                st.success(f"Updated server: {server_name}")
            else:
                # Add new server
                st.session_state.model_servers.append(new_server)
                st.success(f"Added new server: {server_name}")

            st.rerun()

    # Current servers display
    st.markdown("#### Current Servers")
    if st.session_state.model_servers:
        for i, server in enumerate(st.session_state.model_servers):
            with st.expander(f"🖥️ {server.get('name', f'Server {i+1}')}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Endpoint:** {server.get('endpoint', 'N/A')}")
                    st.write(f"**GPU ID:** {server.get('gpu_id', 'N/A')}")

                with col2:
                    st.write(f"**Max Requests:** {server.get('max_concurrent_requests', 10)}")
                    st.write(f"**Timeout:** {server.get('timeout', 300)}s")

                if st.button(f"Remove {server.get('name', f'Server {i+1}')}", key=f"remove_{i}"):
                    st.session_state.model_servers.pop(i)
                    st.rerun()
    else:
        st.info("No servers configured. Add servers above to get started.")

def show_enhanced_chat_analysis():
    """Enhanced chat analysis interface for API system"""
    st.markdown('<div class="sub-header">💬 Chat with API-Based Opulence</div>', unsafe_allow_html=True)

    # Show server status first
    show_server_status()

    # Chat status indicator
    if st.session_state.coordinator:
        try:
            health = st.session_state.coordinator.get_health_status()
            available_servers = health.get('available_servers', 0)
            if available_servers > 0:
                st.success(f"🟢 Chat Agent Ready - {available_servers} servers available")
            else:
                st.error("🔴 No servers available for chat")
        except:
            st.info("🔵 Chat Agent: Status unknown")

    # Chat container with enhanced display
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    # Check if this is a structured response
                    if isinstance(message.get("content"), dict):
                        response_data = message["content"]
                        st.write(response_data.get("response", ""))

                        # Show server info if available
                        servers_used = response_data.get("servers_used")
                        if servers_used and servers_used != "unknown":
                            if isinstance(servers_used, list):
                                st.caption(f"🌐 Processed via servers: {', '.join(servers_used)}")
                            else:
                                st.caption(f"🌐 Processed via: {servers_used}")

                        # Show suggestions if available
                        suggestions = response_data.get("suggestions", [])
                        if suggestions:
                            st.markdown("**💡 Suggestions:**")
                            cols = st.columns(min(len(suggestions), 3))
                            for j, suggestion in enumerate(suggestions[:3]):
                                with cols[j]:
                                    if st.button(f"💬 {suggestion[:30]}...", key=f"suggestion_{i}_{j}"):
                                        # Add suggestion as new user message
                                        st.session_state.chat_history.append({
                                            "role": "user",
                                            "content": suggestion,
                                            "timestamp": dt.now().isoformat()
                                        })
                                        st.rerun()
                    else:
                        st.write(message["content"])

    # Enhanced chat input with processing
    user_input = st.chat_input("Ask about your mainframe systems...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": dt.now().isoformat()
        })

        # Show processing indicator
        with st.spinner("🌐 API-based Opulence is thinking..."):
            # Process query and generate response
            response_data = process_chat_query(user_input)

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_data,
            "timestamp": dt.now().isoformat()
        })

        st.rerun()

    # Enhanced chat controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("📄 Export Chat"):
            export_chat_history()

    with col3:
        if st.button("📊 Chat Summary"):
            generate_chat_summary()

    with col4:
        if st.session_state.chat_history:
            if st.button("🔮 Suggest Questions"):
                generate_follow_up_suggestions()

def process_uploaded_files(uploaded_files):
    """Process uploaded files using the API coordinator"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    # Save files temporarily and process
    temp_dir = Path(tempfile.mkdtemp())
    file_paths = []

    try:
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Saving {uploaded_file.name}...")

            # Save file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        # Process files in batch using API coordinator
        status_text.text("Processing files with API-based Opulence...")

        # Use the coordinator's process_batch_files method
        result = safe_run_async(
            st.session_state.coordinator.process_batch_files(file_paths)
        )

        # Display results
        with results_container:
            if isinstance(result, dict) and result.get("status") == "success":
                servers_used = result.get('servers_used', 'Unknown')
                st.success(f"✅ Successfully processed {result.get('files_processed', 0)} files in {result.get('processing_time', 0):.2f} seconds via {servers_used}")

                # Show detailed results
                results_list = result.get("results", [])
                for i, file_result in enumerate(results_list):
                    if isinstance(file_result, dict):
                        with st.expander(f"📄 {uploaded_files[i].name}"):
                            st.json(file_result)
                    else:
                        with st.expander(f"📄 {uploaded_files[i].name}"):
                            st.text(str(file_result))

            else:
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                st.error(f"❌ Processing failed: {error_msg}")

        # Update processing history
        st.session_state.processing_history.append({
            "timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_count": len(uploaded_files),
            "status": result.get("status", "error") if isinstance(result, dict) else "error",
            "processing_time": result.get("processing_time", 0) if isinstance(result, dict) else 0,
            "servers_used": result.get("servers_used", "unknown") if isinstance(result, dict) else "unknown"
        })

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.exception(e)

    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

        progress_bar.empty()
        status_text.empty()

def analyze_component_api(component_name: str, component_type: str, user_question: str = None):
    """Component analysis using API coordinator"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return

    with st.spinner(f"🌐 Analyzing {component_name} via API..."):
        try:
            # Use regular analysis
            component_type_param = None if component_type == "auto-detect" else component_type
            result = safe_run_async(
                st.session_state.coordinator.analyze_component(
                    component_name,
                    component_type_param
                )
            )

            st.session_state.current_analysis = result

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def show_system_health():
    """Show system health and statistics for API system"""
    st.markdown('<div class="sub-header">⚙️ API System Health & Statistics</div>', unsafe_allow_html=True)

    if not COORDINATOR_AVAILABLE:
        st.error("🔴 System Status: API Coordinator Not Available")
        st.markdown("### Import Error")
        st.code(st.session_state.get('import_error', 'Unknown import error'))
        st.info("Please ensure the api_opulence_coordinator module is properly installed and configured.")
        return

    # Server configuration section
    configure_model_servers()

    st.markdown("---")

    if not st.session_state.coordinator:
        st.warning("🟡 System Status: Not Initialized")
        if st.button("🔄 Initialize API System"):
            with st.spinner("Initializing API system..."):
                try:
                    success = safe_run_async(init_api_coordinator())
                    if success and not isinstance(success, dict):
                        st.success("✅ API system initialized successfully")
                        st.rerun()
                    else:
                        error_msg = success.get('error') if isinstance(success, dict) else st.session_state.initialization_status
                        st.error(f"❌ Initialization failed: {error_msg}")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
        return

    try:
        # Get health status
        health = st.session_state.coordinator.get_health_status()
        stats_result = safe_run_async(st.session_state.coordinator.get_statistics())

        # Show server status prominently
        show_server_status()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        system_stats = stats_result.get("system_stats", {}) if isinstance(stats_result, dict) else {}

        with col1:
            st.metric("Files Processed", system_stats.get("total_files_processed", 0))

        with col2:
            st.metric("Total Queries", system_stats.get("total_queries", 0))

        with col3:
            api_calls = system_stats.get("total_api_calls", 0)
            st.metric("API Calls", api_calls)

        with col4:
            avg_time = system_stats.get("avg_response_time", 0)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")

        # Server performance chart
        server_stats = stats_result.get("server_stats", []) if isinstance(stats_result, dict) else []
        if server_stats:
            st.markdown("### Server Performance")
            df_servers = pd.DataFrame(server_stats)

            if not df_servers.empty and 'name' in df_servers.columns:
                col1, col2 = st.columns(2)

                with col1:
                    if 'total_requests' in df_servers.columns:
                        fig_requests = px.bar(df_servers, x="name", y="total_requests",
                                            title="Total Requests by Server")
                        st.plotly_chart(fig_requests, use_container_width=True)

                with col2:
                    if 'success_rate' in df_servers.columns:
                        fig_success = px.bar(df_servers, x="name", y="success_rate",
                                           title="Success Rate by Server (%)")
                        st.plotly_chart(fig_success, use_container_width=True)

        # Recent activity
        st.markdown("### Recent Activity")
        if st.session_state.processing_history:
            for activity in st.session_state.processing_history[-5:]:
                servers_info = f" (via {activity.get('servers_used', 'Unknown')})" if activity.get('servers_used') else ""
                st.info(f"🕐 {activity['timestamp']}: Processed {activity['files_count']} files{servers_info}")
        else:
            st.info("No recent activities to display")

    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def show_file_upload():
    """Show file upload interface for API system"""
    st.markdown('<div class="sub-header">📂 File Upload & Processing (API-Based)</div>', unsafe_allow_html=True)

    if not COORDINATOR_AVAILABLE:
        st.warning("⚠️ API Coordinator not available. File processing is disabled in demo mode.")
        st.info("To enable file processing, ensure the api_opulence_coordinator module is properly installed.")
        return

    if not st.session_state.coordinator:
        st.error("System not initialized. Please go to System Health tab to initialize.")
        return

    # Show current server status
    show_server_status()

    # Upload options
    upload_type = st.radio(
        "Upload Type",
        ["Single Files", "Batch Upload (ZIP)"]
    )

    if upload_type == "Single Files":
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['cbl', 'cob', 'jcl', 'csv', 'ddl', 'sql', 'dcl', 'copy', 'cpy'],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} files")

            if st.button("Process Files via API"):
                process_uploaded_files(uploaded_files)

    elif upload_type == "Batch Upload (ZIP)":
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing multiple files",
            type=['zip']
        )

        if uploaded_zip:
            if st.button("Extract and Process ZIP via API"):
                process_zip_file(uploaded_zip)

    # Processing history with server information
    st.markdown("### Processing History")
    if st.session_state.processing_history:
        df_history = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No processing history available")

def process_zip_file(uploaded_zip):
    """Process uploaded ZIP file using API coordinator"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Extract ZIP
        zip_path = temp_dir / uploaded_zip.name
        with open(zip_path, 'wb') as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find all processable files
        file_extensions = ['.cbl', '.cob', '.jcl', '.csv', '.ddl', '.sql', '.dcl', '.copy', '.cpy']
        file_paths = []

        for ext in file_extensions:
            file_paths.extend(temp_dir.rglob(f"*{ext}"))

        if not file_paths:
            st.warning("No processable files found in ZIP")
            return

        st.info(f"Found {len(file_paths)} files to process via API")

        # Process files using API coordinator
        result = safe_run_async(
            st.session_state.coordinator.process_batch_files(file_paths)
        )

        if isinstance(result, dict) and result.get("status") == "success":
            servers_used = result.get('servers_used', 'Unknown')
            st.success(f"✅ Successfully processed {result.get('files_processed', 0)} files via {servers_used}")
        else:
            error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
            st.error(f"❌ Processing failed: {error_msg}")

    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")

    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def show_enhanced_component_analysis():
    """Enhanced component analysis with API integration"""
    st.markdown('<div class="sub-header">🔍 Enhanced Component Analysis (API-Based)</div>', unsafe_allow_html=True)

    # Show server status
    show_server_status()

    # Component selection
    col1, col2 = st.columns([2, 1])

    with col1:
        component_name = st.text_input("Component Name (file, table, program, field):")

    with col2:
        component_type = st.selectbox(
            "Component Type",
            ["auto-detect", "file", "table", "program", "jcl", "field"]
        )

    # Optional user question
    user_question = st.text_input("Specific question about this component (optional):")

    if st.button("🔍 Analyze Component via API") and component_name:
        analyze_component_api(component_name, component_type, user_question)

    # Display current analysis
    if st.session_state.current_analysis:
        display_component_analysis(st.session_state.current_analysis)

def display_component_analysis(analysis: dict):
    """Display component analysis results with API info"""
    if isinstance(analysis, dict) and "error" in analysis:
        st.error(f"Analysis error: {analysis['error']}")
        return

    if not isinstance(analysis, dict):
        st.error(f"Invalid analysis result: {type(analysis)}")
        return

    component_name = analysis.get("component_name", "Unknown")
    component_type = analysis.get("component_type", "unknown")

    st.success(f"✅ Analysis completed for {component_type}: **{component_name}** via API")

    # Show processing metadata if available
    if "processing_metadata" in analysis:
        metadata = analysis["processing_metadata"]
        coordinator_type = metadata.get("coordinator_type", "unknown")
        if coordinator_type == "api_based":
            st.info("🌐 Processed using API-based architecture")

    # Create tabs for different aspects of analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔄 Analyses", "📋 Report", "🐛 Debug"])

    with tab1:
        show_analysis_overview(analysis)

    with tab2:
        show_analyses_results(analysis)

    with tab3:
        show_analysis_report(analysis)

    with tab4:
        show_analysis_debug(analysis)

def show_analysis_overview(analysis: dict):
    """Analysis overview for API-based processing"""
    st.markdown("### 📊 Component Overview")

    # Basic information
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        component_type = analysis.get("component_type", "unknown")
        st.metric("Component Type", component_type.title())

    with col2:
        status = analysis.get("status", "unknown")
        st.metric("Analysis Status", status.title())

    with col3:
        processing_time = analysis.get("processing_time", 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")

    with col4:
        if "processing_metadata" in analysis:
            coordinator_type = analysis["processing_metadata"].get("coordinator_type", "unknown")
            st.metric("Processing Type", coordinator_type.replace("_", " ").title())

    # Show completion statistics
    if "analyses" in analysis:
        analyses = analysis["analyses"]
        completed = sum(1 for a in analyses.values() if a.get("status") == "success")
        total = len(analyses)

        st.markdown("### Analysis Completion")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Analyses", total)

        with col2:
            st.metric("Completed", completed)

        with col3:
            success_rate = (completed / total * 100) if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

def show_analyses_results(analysis: dict):
    """Show detailed analysis results"""
    st.markdown("### 🔄 Analysis Results")

    if "analyses" not in analysis:
        st.info("No detailed analyses available")
        return

    analyses = analysis["analyses"]

    for analysis_type, analysis_data in analyses.items():
        with st.expander(f"📊 {analysis_type.replace('_', ' ').title()}"):
            if analysis_data.get("status") == "success":
                st.success("✅ Completed successfully")

                # Show completion time
                completion_time = analysis_data.get("completion_time", 0)
                st.info(f"⏱️ Completed in {completion_time:.2f} seconds")

                # Show data if available
                if "data" in analysis_data:
                    data = analysis_data["data"]
                    if isinstance(data, dict):
                        # Show specific metrics based on analysis type
                        if analysis_type == "lineage_analysis":
                            if "usage_analysis" in data:
                                usage_stats = data["usage_analysis"].get("statistics", {})
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total References", usage_stats.get("total_references", 0))
                                with col2:
                                    st.metric("Programs Using", len(usage_stats.get("programs_using", [])))
                                with col3:
                                    st.metric("Operation Types", len(usage_stats.get("operation_types", {})))

                        elif analysis_type == "logic_analysis":
                            if "total_chunks" in data:
                                st.metric("Code Chunks", data["total_chunks"])
                            if "complexity_score" in data:
                                complexity = data["complexity_score"]
                                st.metric("Complexity Score", f"{complexity:.2f}")

                        elif analysis_type == "semantic_analysis":
                            if "similar_components" in data:
                                similar_count = len(data["similar_components"].get("results", []))
                                st.metric("Similar Components", similar_count)
                    else:
                        st.write(str(data)[:500] + "..." if len(str(data)) > 500 else str(data))
            else:
                st.error(f"❌ Failed: {analysis_data.get('error', 'Unknown error')}")

                # Show agent used
                agent_used = analysis_data.get("agent_used", "unknown")
                st.caption(f"Agent: {agent_used}")

def show_analysis_report(analysis: dict):
    """Comprehensive analysis report for API-based processing"""
    st.markdown("### 📋 Comprehensive Analysis Report")

    component_name = analysis.get("component_name", "Unknown")
    component_type = analysis.get("component_type", "unknown")

    # Generate report based on available analysis
    report_sections = []

    # Executive Summary
    report_sections.append("## Executive Summary")
    report_sections.append(f"**Component**: {component_name}")
    report_sections.append(f"**Type**: {component_type.title()}")
    report_sections.append(f"**Analysis Status**: {analysis.get('status', 'unknown').title()}")

    # Processing information
    if "processing_metadata" in analysis:
        metadata = analysis["processing_metadata"]
        coordinator_type = metadata.get("coordinator_type", "unknown")
        report_sections.append(f"**Processing Method**: {coordinator_type.replace('_', ' ').title()}")

        if "total_duration_seconds" in metadata:
            duration = metadata["total_duration_seconds"]
            report_sections.append(f"**Total Processing Time**: {duration:.2f} seconds")

    # Analysis results summary
    if "analyses" in analysis:
        analyses = analysis["analyses"]
        completed = sum(1 for a in analyses.values() if a.get("status") == "success")
        total = len(analyses)

        report_sections.append(f"\n## Analysis Summary")
        report_sections.append(f"**Analyses Completed**: {completed}/{total}")

        if completed > 0:
            success_rate = (completed / total) * 100
            report_sections.append(f"**Success Rate**: {success_rate:.1f}%")

        # Detailed results for each analysis
        for analysis_type, analysis_data in analyses.items():
            if analysis_data.get("status") == "success":
                report_sections.append(f"\n### {analysis_type.replace('_', ' ').title()}")

                completion_time = analysis_data.get("completion_time", 0)
                report_sections.append(f"**Duration**: {completion_time:.2f} seconds")

                agent_used = analysis_data.get("agent_used", "unknown")
                report_sections.append(f"**Agent**: {agent_used}")

                # Add specific insights based on analysis type
                data = analysis_data.get("data", {})
                if analysis_type == "lineage_analysis" and isinstance(data, dict):
                    if "usage_analysis" in data:
                        usage_stats = data["usage_analysis"].get("statistics", {})
                        total_refs = usage_stats.get("total_references", 0)
                        programs_using = len(usage_stats.get("programs_using", []))
                        report_sections.append(f"- Found {total_refs} total references")
                        report_sections.append(f"- Used by {programs_using} programs")

                elif analysis_type == "logic_analysis" and isinstance(data, dict):
                    if "total_chunks" in data:
                        report_sections.append(f"- Analyzed {data['total_chunks']} code chunks")
                    if "complexity_score" in data:
                        complexity = data["complexity_score"]
                        report_sections.append(f"- Average complexity: {complexity:.2f}")

    # Display the report
    report_content = "\n".join(report_sections)
    st.markdown(report_content)

    # Download button
    if report_content:
        st.download_button(
            "📄 Download Report",
            report_content,
            file_name=f"opulence_api_analysis_{component_name}_{dt.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def show_analysis_debug(analysis: dict):
    """Show debug information for API-based analysis"""
    st.markdown("### 🐛 Debug Information")

    # Show raw analysis data
    with st.expander("Raw Analysis Data"):
        st.json(analysis)

    # Show processing metadata
    if "processing_metadata" in analysis:
        with st.expander("Processing Metadata"):
            st.json(analysis["processing_metadata"])

    # Show individual analysis debug info
    if "analyses" in analysis:
        st.markdown("#### Individual Analysis Results")
        for name, result in analysis["analyses"].items():
            with st.expander(f"{name.replace('_', ' ').title()} Debug"):
                if isinstance(result, dict):
                    if "error" in result:
                        st.error(f"❌ {name} failed: {result['error']}")
                    else:
                        st.success(f"✅ {name} succeeded")
                    st.json(result)
                else:
                    st.write(result)

    # Show API processing info
    coordinator_type = analysis.get("processing_metadata", {}).get("coordinator_type", "unknown")
    st.info(f"🌐 Analysis performed using {coordinator_type.replace('_', ' ').title()} architecture")

def export_chat_history():
    """Export chat history with API processing info"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return

    # Create formatted export
    export_data = {
        "export_info": {
            "timestamp": dt.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "session_id": st.session_state.get("session_id", "unknown"),
            "coordinator_type": "api_based",
            "model_servers": st.session_state.model_servers
        },
        "conversation": []
    }

    for message in st.session_state.chat_history:
        if isinstance(message.get("content"), dict):
            # Structured response
            export_data["conversation"].append({
                "role": message["role"],
                "timestamp": message["timestamp"],
                "response": message["content"].get("response", ""),
                "response_type": message["content"].get("response_type", "unknown"),
                "suggestions": message["content"].get("suggestions", []),
                "servers_used": message["content"].get("servers_used", "unknown"),
                "metadata": message["content"]
            })
        else:
            # Simple message
            export_data["conversation"].append({
                "role": message["role"],
                "timestamp": message["timestamp"],
                "content": message["content"]
            })

    # Create download
    export_json = json.dumps(export_data, indent=2)
    st.download_button(
        "📥 Download API Chat History",
        export_json,
        file_name=f"opulence_api_chat_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def generate_chat_summary():
    """Generate conversation summary using API coordinator"""
    if not st.session_state.chat_history:
        st.info("No conversation to summarize")
        return

    if not st.session_state.coordinator:
        st.error("System not initialized")
        return

    with st.spinner("🌐 Generating conversation summary via API..."):
        try:
            # Create a summary query
            summary_query = "Please provide a summary of our conversation so far, highlighting the key points and findings."

            summary_result = safe_run_async(
                st.session_state.coordinator.process_chat_query(summary_query, st.session_state.chat_history)
            )

            summary = summary_result.get("response", "Unable to generate summary") if isinstance(summary_result, dict) else str(summary_result)
            servers_used = summary_result.get("servers_used", "unknown") if isinstance(summary_result, dict) else "unknown"

            # Display summary in a nice format
            st.markdown("### 📋 Conversation Summary")
            st.markdown(summary)
            st.caption(f"Generated via {servers_used}")

            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": {
                    "response": f"**Conversation Summary:**\n\n{summary}",
                    "response_type": "summary",
                    "suggestions": ["Continue analysis", "Export summary", "Start new topic"],
                    "servers_used": servers_used
                },
                "timestamp": dt.now().isoformat()
            })

        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")

def generate_follow_up_suggestions():
    """Generate follow-up question suggestions using API coordinator"""
    if len(st.session_state.chat_history) < 2:
        st.info("Need more conversation for suggestions")
        return

    if not st.session_state.coordinator:
        st.error("System not initialized")
        return

    # Get last exchange
    last_messages = st.session_state.chat_history[-2:]
    if len(last_messages) >= 2:
        last_query = last_messages[0].get("content", "")
        last_response = last_messages[1].get("content", "")

        # Extract response text if structured
        if isinstance(last_response, dict):
            last_response = last_response.get("response", "")
    else:
        return

    with st.spinner("🔮 Generating suggestions via API..."):
        try:
            suggestion_query = f"Based on my question '{last_query}' and your response '{last_response[:200]}...', what are 3 good follow-up questions I should ask?"

            suggestion_result = safe_run_async(
                st.session_state.coordinator.process_chat_query(suggestion_query, [])
            )

            if isinstance(suggestion_result, dict):
                suggestions_text = suggestion_result.get("response", "")
                servers_used = suggestion_result.get("servers_used", "unknown")

                # Extract suggestions from response (simple parsing)
                lines = suggestions_text.split('\n')
                suggestions = []
                for line in lines:
                    if line.strip() and ('?' in line or line.strip().startswith(('-', '•', '1.', '2.', '3.'))):
                        clean_suggestion = line.strip().lstrip('-•123. ')
                        if clean_suggestion and len(clean_suggestion) > 10:
                            suggestions.append(clean_suggestion)

                if suggestions:
                    st.markdown("### 💡 Suggested Follow-up Questions")
                    st.caption(f"Generated via {servers_used}")
                    for i, suggestion in enumerate(suggestions[:3]):
                        if st.button(f"❓ {suggestion}", key=f"followup_{i}"):
                            # Add as new user message
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": suggestion,
                                "timestamp": dt.now().isoformat()
                            })
                            st.rerun()
                else:
                    st.info("Could not parse specific suggestions from response")
                    st.write(suggestions_text)

        except Exception as e:
            st.error(f"Failed to generate suggestions: {str(e)}")

def show_example_queries():
    """Show example queries in sidebar with API info"""
    st.markdown("### 💡 Example Queries")

    examples = [
        "Analyze the lifecycle of TRADE_DATE field",
        "Trace lineage of TRANSACTION_HISTORY_FILE",
        "Find programs using security settlement logic",
        "Show impact of changing ACCOUNT_ID field",
        "Compare data between CUSTOMER_TABLE and customer.csv",
        "Generate documentation for BKPG_TRD001 program"
    ]

    for i, example in enumerate(examples):
        if st.button(f"💬 {example[:30]}...", key=f"example_{i}"):
            # Add example to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": example,
                "timestamp": dt.now().isoformat()
            })
            # Process the example query
            response = process_chat_query(example)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": dt.now().isoformat()
            })
            st.rerun()

def show_footer():
    """Show footer with system information including API details"""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌐 Opulence API-Based Research Agent**")
        st.markdown("Powered by HTTP API calls to model servers")
        if st.session_state.coordinator:
            health = st.session_state.coordinator.get_health_status()
            available_servers = health.get('available_servers', 0)
            total_servers = health.get('total_servers', 0)
            st.markdown(f"API Servers: {available_servers}/{total_servers} available")

    with col2:
        st.markdown("**📊 Current Session**")
        st.markdown(f"Chat Messages: {len(st.session_state.chat_history)}")
        st.markdown(f"Files Processed: {len(st.session_state.processing_history)}")

    with col3:
        st.markdown("**🕐 System Time**")
        st.markdown(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def show_dashboard():
    """Show main dashboard for API system"""
    st.markdown('<div class="sub-header">🌐 API System Overview</div>', unsafe_allow_html=True)

    if not st.session_state.coordinator:
        st.warning("System not initialized")
        return

    # Get system statistics
    try:
        health = st.session_state.coordinator.get_health_status()
        stats_result = safe_run_async(st.session_state.coordinator.get_statistics())

        # Overall health indicator
        if health.get("status") == "healthy":
            st.success("🟢 API System Status: Healthy")
        else:
            st.error("🔴 API System Status: Issues Detected")

        # Show server information prominently
        show_server_status()

        # System metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Agents", health.get("active_agents", 0))

        with col2:
            db_status = "Available" if health.get("database_available", False) else "Not Available"
            st.metric("Database", db_status)

        with col3:
            coordinator_type = health.get("coordinator_type", "unknown")
            st.metric("Coordinator Type", coordinator_type.replace("_", " ").title())

        with col4:
            available_servers = health.get("available_servers", 0)
            total_servers = health.get("total_servers", 0)
            st.metric("Server Availability", f"{available_servers}/{total_servers}")

        # Server performance statistics
        if isinstance(stats_result, dict) and "server_stats" in stats_result:
            server_stats = stats_result["server_stats"]
            if server_stats:
                st.markdown("### Server Performance")
                df_servers = pd.DataFrame(server_stats)

                col1, col2 = st.columns(2)

                with col1:
                    if 'total_requests' in df_servers.columns and 'name' in df_servers.columns:
                        fig_requests = px.bar(df_servers, x="name", y="total_requests",
                                            title="Total Requests by Server")
                        st.plotly_chart(fig_requests, use_container_width=True)

                with col2:
                    if 'average_latency' in df_servers.columns and 'name' in df_servers.columns:
                        fig_latency = px.bar(df_servers, x="name", y="average_latency",
                                           title="Average Latency by Server (seconds)")
                        st.plotly_chart(fig_latency, use_container_width=True)

        # Recent activity
        st.markdown("### Recent Activity")
        if st.session_state.processing_history:
            for activity in st.session_state.processing_history[-5:]:
                servers_info = f" (via {activity.get('servers_used', 'Unknown')})" if activity.get('servers_used') else ""
                st.info(f"🕐 {activity['timestamp']}: Processed {activity['files_count']} files{servers_info}")
        else:
            st.info("No recent activities to display")

        # System controls
        st.markdown("### System Controls")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔄 Refresh Statistics"):
                st.rerun()

        with col2:
            if st.button("🧹 Clean Memory"):
                try:
                    st.session_state.coordinator.cleanup()
                    st.success("Memory cleaned")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clean memory: {str(e)}")

        with col3:
            if st.button("📊 Health Check"):
                run_health_check()

        with col4:
            if st.button("📥 Export Logs"):
                export_system_logs()

    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def run_health_check():
    """Run health check on all servers"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return

    with st.spinner("Running health check on all servers..."):
        try:
            # Force a health check by getting fresh status
            health = st.session_state.coordinator.get_health_status()

            server_stats = health.get('server_stats', {})
            healthy_count = sum(1 for stats in server_stats.values() if stats.get('available', False))
            total_count = len(server_stats)

            if healthy_count == total_count:
                st.success(f"✅ All {total_count} servers are healthy")
            elif healthy_count > 0:
                st.warning(f"⚠️ {healthy_count}/{total_count} servers are healthy")
            else:
                st.error("❌ No servers are responding")

            # Show detailed status
            for server_name, stats in server_stats.items():
                if stats.get('available', False):
                    st.success(f"🟢 {server_name}: Healthy")
                else:
                    st.error(f"🔴 {server_name}: Unhealthy")

        except Exception as e:
            st.error(f"Health check failed: {str(e)}")

def export_system_logs():
    """Export system logs"""
    try:
        # Read log file
        log_file = Path("opulence_api.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()

            st.download_button(
                "📥 Download API System Logs",
                log_content,
                file_name=f"opulence_api_logs_{dt.now().strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain"
            )
        else:
            st.warning("No log file found")

    except Exception as e:
        st.error(f"Failed to export logs: {str(e)}")

def main():
    """Main application function for API system"""

    # Header
    st.markdown('<div class="main-header">🌐 Opulence API-Based Deep Research Agent</div>', unsafe_allow_html=True)

    # Show import status
    if not COORDINATOR_AVAILABLE:
        st.error("⚠️ API Coordinator module not available - Running in demo mode")
        if st.button("🐛 Toggle Debug Mode"):
            st.session_state.debug_mode = not st.session_state.debug_mode
            st.rerun()

    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/059669/ffffff?text=OPULENCE", use_container_width=True)

        page = st.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📂 File Upload", "💬 Enhanced Chat", "🔍 Enhanced Analysis",
             "⚙️ System Health"]
        )

        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh API System"):
            st.rerun()

        # System status indicator with API info
        if COORDINATOR_AVAILABLE and st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                available_servers = health.get('available_servers', 0)
                total_servers = health.get('total_servers', 0)
                if available_servers > 0:
                    st.success(f"🟢 API System Healthy ({available_servers}/{total_servers})")
                else:
                    st.error("🔴 No API Servers Available")
            except:
                st.success("🟢 API System Ready")
        elif COORDINATOR_AVAILABLE:
            st.warning("🟡 API System Not Initialized")
        else:
            st.error("🔴 Demo Mode")

        # Show example queries
        show_example_queries()

        # Show debug info if enabled
        show_debug_info()

    # Main content based on selected page
    try:
        if page == "🏠 Dashboard":
            show_dashboard()
        elif page == "📂 File Upload":
            show_file_upload()
        elif page == "💬 Enhanced Chat":
            show_enhanced_chat_analysis()
        elif page == "🔍 Enhanced Analysis":
            show_enhanced_component_analysis()
        elif page == "⚙️ System Health":
            show_system_health()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.exception(e)

    # Show footer
    show_footer()

# Main execution
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

        # Emergency debug mode
        st.markdown("### Emergency Debug Info")
        st.json({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "session_state": dict(st.session_state),
            "coordinator_type": "api_based"
        })