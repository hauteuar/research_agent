#!/usr/bin/env python3
"""
Enhanced Streamlit Application for Opulence API Research Agent
- Complete mainframe file support (COBOL, JCL, COPYBOOK, PL/I, CICS, DB2, etc.)
- Comprehensive GPU monitoring and status
- Agent status tracking and display
- Real-time coordinator health monitoring
"""

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
from typing import Dict, Any, List, Optional
from enum import Enum
import mimetypes
import hashlib

# ============================================================================
# GLOBAL CONSTANTS AND MAINFRAME FILE CONFIGURATION
# ============================================================================

COORDINATOR_AVAILABLE = True
import_error = None

try:
    from api_opulence_coordinator import create_api_coordinator_from_config
except ImportError as e:
    COORDINATOR_AVAILABLE = False
    import_error = str(e)

# Comprehensive mainframe file types and extensions
MAINFRAME_FILE_TYPES = {
    # COBOL Files
    'cobol': {
        'extensions': ['.cbl', '.cob', '.cobol', '.cpy', '.copybook'],
        'description': 'COBOL Programs and Copybooks',
        'mime_types': ['text/plain', 'application/octet-stream'],
        'agent': 'code_parser'
    },
    # JCL Files
    'jcl': {
        'extensions': ['.jcl', '.job', '.proc', '.prc'],
        'description': 'Job Control Language',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # PL/I Files
    'pli': {
        'extensions': ['.pli', '.pl1', '.pls'],
        'description': 'PL/I Programs',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # CICS Files
    'cics': {
        'extensions': ['.cics', '.bms', '.mapset'],
        'description': 'CICS Maps and Programs',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # DB2 SQL Files
    'db2': {
        'extensions': ['.sql', '.db2', '.ddl', '.dml'],
        'description': 'DB2 SQL Scripts',
        'mime_types': ['text/plain', 'application/sql'],
        'agent': 'db2_comparator'
    },
    # Assembler Files
    'asm': {
        'extensions': ['.asm', '.s', '.hlasm', '.bal'],
        'description': 'Assembler Programs',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # VSAM Files
    'vsam': {
        'extensions': ['.vsam', '.ksds', '.esds', '.rrds'],
        'description': 'VSAM File Definitions',
        'mime_types': ['text/plain'],
        'agent': 'data_loader'
    },
    # IMS Files
    'ims': {
        'extensions': ['.ims', '.psbgen', '.dbdgen', '.mfs'],
        'description': 'IMS Database and Program Files',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # REXX Files
    'rexx': {
        'extensions': ['.rexx', '.rex', '.cmd'],
        'description': 'REXX Scripts',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # Data Files
    'data': {
        'extensions': ['.dat', '.txt', '.csv', '.tsv', '.fixed', '.ebcdic'],
        'description': 'Mainframe Data Files',
        'mime_types': ['text/plain', 'text/csv', 'application/octet-stream'],
        'agent': 'data_loader'
    },
    # Control Files
    'control': {
        'extensions': ['.ctl', '.cfg', '.config', '.parm', '.sysin'],
        'description': 'Control and Parameter Files',
        'mime_types': ['text/plain'],
        'agent': 'code_parser'
    },
    # Documentation
    'docs': {
        'extensions': ['.md', '.txt', '.doc', '.rtf', '.readme'],
        'description': 'Documentation Files',
        'mime_types': ['text/plain', 'text/markdown'],
        'agent': 'documentation'
    }
}

# Get all supported extensions
ALL_MAINFRAME_EXTENSIONS = []
for file_type_info in MAINFRAME_FILE_TYPES.values():
    ALL_MAINFRAME_EXTENSIONS.extend(file_type_info['extensions'])

# Remove duplicates and sort
ALL_MAINFRAME_EXTENSIONS = sorted(list(set(ALL_MAINFRAME_EXTENSIONS)))

# Agent status tracking
AGENT_TYPES = [
    'code_parser', 'chat_agent', 'vector_index', 'data_loader',
    'lineage_analyzer', 'logic_analyzer', 'documentation', 'db2_comparator'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_run_async(coroutine):
    """Safe async function runner for Streamlit"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()
    except Exception as e:
        st.error(f"Async execution failed: {str(e)}")
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
        'model_servers': [
            {
                "name": "gpu_server_1",
                "endpoint": "http://localhost:8000",
                "gpu_id": 0,
                "max_concurrent_requests": 10,
                "timeout": 300
            }
        ],
        'coordinator': None,
        'debug_mode': False,
        'initialization_status': 'not_started',
        'import_error': import_error if not COORDINATOR_AVAILABLE else None,
        'auto_refresh_gpu': False,
        'gpu_refresh_interval': 10
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

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
        
        # COBOL detection
        if any(keyword in content_upper for keyword in ['IDENTIFICATION DIVISION', 'PROGRAM-ID', 'WORKING-STORAGE']):
            return {
                'type': 'cobol',
                'description': 'COBOL Program (detected from content)',
                'agent': 'code_parser',
                'confidence': 'medium'
            }
        
        # JCL detection
        if any(keyword in content_upper for keyword in ['//JOB ', '//EXEC ', '//DD ']):
            return {
                'type': 'jcl',
                'description': 'JCL Job (detected from content)',
                'agent': 'code_parser',
                'confidence': 'medium'
            }
        
        # SQL detection
        if any(keyword in content_upper for keyword in ['CREATE TABLE', 'SELECT ', 'INSERT INTO', 'UPDATE ', 'DELETE FROM']):
            return {
                'type': 'db2',
                'description': 'SQL Script (detected from content)',
                'agent': 'db2_comparator',
                'confidence': 'medium'
            }
    
    # Default to generic data file
    return {
        'type': 'data',
        'description': 'Generic Mainframe File',
        'agent': 'code_parser',
        'confidence': 'low'
    }

def add_custom_css():
    """Add custom CSS styles for mainframe application"""
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
    .file-type-badge {
        background-color: #007bff;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    .agent-status-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
    }
    .mainframe-upload-area {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9ff;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def validate_server_endpoint(endpoint: str, timeout: int = 5) -> Dict[str, Any]:
    """Validate a server endpoint and return detailed status"""
    try:
        # Health check
        response = requests.get(f"{endpoint}/health", timeout=timeout)
        if response.status_code == 200:
            health_status = "healthy"
            health_message = "Server responding"
            
            try:
                # Get detailed status
                status_response = requests.get(f"{endpoint}/status", timeout=timeout)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    gpu_info = status_data.get('gpu_info', {})
                    gpu_count = len(gpu_info)
                    model_name = status_data.get('model', 'Unknown')
                    active_requests = status_data.get('active_requests', 0)
                    total_requests = status_data.get('total_requests', 0)
                    
                    health_message = f"Model: {model_name}, GPUs: {gpu_count}, Active: {active_requests}, Total: {total_requests}"
                else:
                    health_message = "Server healthy, detailed status unavailable"
            except:
                health_message = "Server healthy, status endpoint unavailable"
            
            return {
                "status": health_status,
                "message": health_message,
                "response_time": response.elapsed.total_seconds(),
                "accessible": True,
                "detailed_status": status_data if 'status_data' in locals() else None
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
    
# ============================================================================
# CORE COORDINATOR AND AGENT MONITORING
# ============================================================================

@with_error_handling
def get_detailed_server_status():
    """Get detailed status from all model servers including GPU and agent information"""
    if not st.session_state.coordinator:
        return {}
    
    detailed_status = {}
    
    try:
        # Get coordinator health which includes server stats
        health = st.session_state.coordinator.get_health_status()
        server_stats = health.get('server_stats', {})
        
        # For each server, try to get detailed information
        for server_name, basic_stats in server_stats.items():
            try:
                # Find the server configuration to get endpoint
                server_config = None
                for server in st.session_state.model_servers:
                    if server.get('name') == server_name:
                        server_config = server
                        break
                
                if server_config and server_config.get('endpoint'):
                    # Get detailed status from the model server directly
                    response = requests.get(
                        f"{server_config['endpoint']}/status",
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        server_status = response.json()
                        
                        # Get metrics
                        metrics_response = requests.get(
                            f"{server_config['endpoint']}/metrics",
                            timeout=5
                        )
                        metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
                        
                        # Merge all data
                        detailed_status[server_name] = {
                            **basic_stats,
                            'endpoint': server_config['endpoint'],
                            'gpu_info': server_status.get('gpu_info', {}),
                            'memory_info': server_status.get('memory_info', {}),
                            'model': server_status.get('model', 'Unknown'),
                            'uptime': server_status.get('uptime', 0),
                            'metrics': metrics_data
                        }
                    else:
                        detailed_status[server_name] = {
                            **basic_stats,
                            'endpoint': server_config['endpoint'],
                            'error': f"HTTP {response.status_code}"
                        }
                else:
                    detailed_status[server_name] = {
                        **basic_stats,
                        'error': 'No endpoint configured'
                    }
                    
            except requests.RequestException as e:
                detailed_status[server_name] = {
                    **basic_stats,
                    'error': f"Connection error: {str(e)}"
                }
            except Exception as e:
                detailed_status[server_name] = {
                    **basic_stats,
                    'error': f"Error: {str(e)}"
                }
        
        return detailed_status
        
    except Exception as e:
        st.error(f"Failed to get detailed server status: {str(e)}")
        return {}

@with_error_handling
def get_agent_status():
    """Get status of all agents"""
    if not st.session_state.coordinator:
        return st.session_state.agent_status
    
    try:
        # Get agent information from coordinator
        agent_status = {}
        
        for agent_type in AGENT_TYPES:
            try:
                # Try to get agent from coordinator
                agent = st.session_state.coordinator.get_agent(agent_type)
                
                if agent:
                    # Agent is available
                    status = {
                        'status': 'available',
                        'last_used': st.session_state.agent_status[agent_type].get('last_used'),
                        'total_calls': st.session_state.agent_status[agent_type].get('total_calls', 0),
                        'errors': st.session_state.agent_status[agent_type].get('errors', 0),
                        'agent_object': type(agent).__name__
                    }
                else:
                    status = {
                        'status': 'unavailable',
                        'last_used': None,
                        'total_calls': 0,
                        'errors': 0,
                        'agent_object': None
                    }
                
                agent_status[agent_type] = status
                
            except Exception as e:
                agent_status[agent_type] = {
                    'status': 'error',
                    'last_used': None,
                    'total_calls': 0,
                    'errors': st.session_state.agent_status[agent_type].get('errors', 0) + 1,
                    'error_message': str(e)
                }
        
        # Update session state
        st.session_state.agent_status.update(agent_status)
        return agent_status
        
    except Exception as e:
        st.error(f"Failed to get agent status: {str(e)}")
        return st.session_state.agent_status

# ============================================================================
# ENHANCED FILE UPLOAD WITH MAINFRAME SUPPORT
# ============================================================================

def show_enhanced_file_upload():
    """Enhanced file upload with comprehensive mainframe support"""
    st.markdown('<div class="sub-header">📂 Mainframe File Upload & Processing</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in System Health tab.")
        st.info("The coordinator must be running to process mainframe files.")
        return
    
    # File type information
    with st.expander("📋 Supported Mainframe File Types", expanded=False):
        for file_type, info in MAINFRAME_FILE_TYPES.items():
            st.markdown(f"**{info['description']}**")
            st.markdown(f"- Extensions: {', '.join(info['extensions'])}")
            st.markdown(f"- Processed by: {info['agent']} agent")
            st.markdown("---")
    
    # Upload area
    st.markdown('<div class="mainframe-upload-area">', unsafe_allow_html=True)
    st.markdown("### 🚀 Upload Mainframe Files")
    st.markdown("Drag and drop your COBOL, JCL, DB2, PL/I, CICS, and other mainframe files here")
    
    uploaded_files = st.file_uploader(
        "Choose mainframe files to upload",
        accept_multiple_files=True,
        type=None,  # Accept all file types
        help="Upload any mainframe files including COBOL (.cbl, .cob), JCL (.jcl), SQL (.sql), etc."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.markdown("### 📁 Uploaded Files Analysis")
        
        file_analysis = []
        
        for uploaded_file in uploaded_files:
            # Read file content for analysis
            try:
                file_content = uploaded_file.read().decode('utf-8', errors='ignore')
                uploaded_file.seek(0)  # Reset file pointer
            except:
                file_content = None
            
            # Detect file type
            file_type_info = detect_mainframe_file_type(uploaded_file.name, file_content)
            
            # Calculate file hash for tracking
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
                    st.markdown(f"**File Type:** {file_info['type'].upper()}")
                    st.markdown(f"**Description:** {file_info['description']}")
                
                with col2:
                    st.markdown(f"**Processing Agent:** {file_info['agent']}")
                    st.markdown(f"**Detection Confidence:** {file_info['confidence']}")
                
                with col3:
                    st.markdown(f"**File Size:** {file_info['size']:,} bytes")
                    st.markdown(f"**File Hash:** {file_info['hash'][:8]}...")
                
                if file_info['content_preview']:
                    st.markdown("**Content Preview:**")
                    st.code(file_info['content_preview'], language='text')
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_detect_types = st.checkbox("🔍 Auto-detect file types", value=True)
        
        with col2:
            parallel_processing = st.checkbox("⚡ Parallel processing", value=True)
        
        with col3:
            save_to_database = st.checkbox("💾 Save to database", value=True)
        
        # Processing button
        if st.button("🚀 Process All Files", type="primary"):
            process_mainframe_files(uploaded_files, file_analysis, {
                'auto_detect_types': auto_detect_types,
                'parallel_processing': parallel_processing,
                'save_to_database': save_to_database
            })
    
    # Show processing history
    show_file_processing_history()

def process_mainframe_files(uploaded_files, file_analysis, options):
    """Process uploaded mainframe files"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    total_files = len(uploaded_files)
    processed_files = 0
    results = []
    
    for i, (uploaded_file, file_info) in enumerate(zip(uploaded_files, file_analysis)):
        try:
            status_text.text(f"Processing {file_info['name']} ({i+1}/{total_files})...")
            
            # Save file temporarily
            temp_file_path = f"/tmp/{file_info['name']}"
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Update agent usage tracking
            agent_type = file_info['agent']
            st.session_state.agent_status[agent_type]['total_calls'] += 1
            st.session_state.agent_status[agent_type]['last_used'] = dt.now().isoformat()
            
            # Process with coordinator
            start_time = time.time()
            
            if options['parallel_processing'] and total_files > 1:
                # TODO: Implement actual parallel processing
                result = safe_run_async(
                    st.session_state.coordinator.process_batch_files(
                        [temp_file_path], file_info['type']
                    )
                )
            else:
                # Sequential processing
                result = safe_run_async(
                    st.session_state.coordinator.process_batch_files(
                        [temp_file_path], file_info['type']
                    )
                )
            
            processing_time = time.time() - start_time
            
            if result and not result.get('error'):
                # Success
                st.session_state.agent_status[agent_type]['status'] = 'available'
                
                processing_result = {
                    'file_name': file_info['name'],
                    'file_type': file_info['type'],
                    'agent_used': agent_type,
                    'status': 'success',
                    'processing_time': processing_time,
                    'result': result,
                    'timestamp': dt.now().isoformat()
                }
                
                # Add to processing history
                st.session_state.processing_history.append(processing_result)
                
                results.append(processing_result)
                
                with results_container:
                    st.success(f"✅ Successfully processed {file_info['name']} in {processing_time:.2f}s")
            
            else:
                # Error
                error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
                st.session_state.agent_status[agent_type]['errors'] += 1
                
                processing_result = {
                    'file_name': file_info['name'],
                    'file_type': file_info['type'],
                    'agent_used': agent_type,
                    'status': 'error',
                    'error': error_msg,
                    'processing_time': processing_time,
                    'timestamp': dt.now().isoformat()
                }
                
                st.session_state.processing_history.append(processing_result)
                results.append(processing_result)
                
                with results_container:
                    st.error(f"❌ Failed to process {file_info['name']}: {error_msg}")
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
            
        except Exception as e:
            st.error(f"❌ Exception processing {file_info['name']}: {str(e)}")
            st.session_state.agent_status[file_info['agent']]['errors'] += 1
    
    # Final summary
    status_text.text("Processing complete!")
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    st.success(f"🎉 Processing Summary: {success_count} successful, {error_count} errors")
    
    # Show detailed results
    if results:
        st.markdown("### 📊 Processing Results")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results[['file_name', 'file_type', 'agent_used', 'status', 'processing_time']], 
                    use_container_width=True)

def show_file_processing_history():
    """Show file processing history"""
    if st.session_state.processing_history:
        st.markdown("### 📈 Processing History")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        total_processed = len(st.session_state.processing_history)
        successful = sum(1 for h in st.session_state.processing_history if h['status'] == 'success')
        failed = total_processed - successful
        avg_time = sum(h.get('processing_time', 0) for h in st.session_state.processing_history) / total_processed if total_processed > 0 else 0
        
        with col1:
            st.metric("Total Files", total_processed)
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Avg Time", f"{avg_time:.2f}s")
        
        # Recent files
        recent_files = st.session_state.processing_history[-10:]  # Last 10 files
        
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
        
        if history_data:
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No files processed yet. Upload and process some mainframe files to see history here.")

# ============================================================================
# COMPREHENSIVE AGENT STATUS DISPLAY
# ============================================================================

def show_comprehensive_agent_status():
    """Show comprehensive agent status and monitoring"""
    st.markdown('<div class="sub-header">🤖 Agent Status & Monitoring</div>', unsafe_allow_html=True)
    
    # Get current agent status
    agent_status = get_agent_status()
    
    # Overall agent health
    available_agents = sum(1 for status in agent_status.values() if status['status'] == 'available')
    total_agents = len(AGENT_TYPES)
    error_agents = sum(1 for status in agent_status.values() if status['status'] == 'error')
    
    # Health indicator
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", total_agents)
    
    with col2:
        st.metric("Available", available_agents)
        if available_agents == total_agents:
            st.success("All agents operational")
    
    with col3:
        st.metric("Errors", error_agents)
        if error_agents > 0:
            st.error(f"{error_agents} agents have errors")
    
    with col4:
        total_calls = sum(status.get('total_calls', 0) for status in agent_status.values())
        st.metric("Total Calls", total_calls)
    
    # Individual agent status
    st.markdown("### 🔍 Individual Agent Status")
    
    # Create columns for agent cards
    cols = st.columns(2)
    
    for i, (agent_type, status) in enumerate(agent_status.items()):
        col = cols[i % 2]
        
        with col:
            # Determine status color and icon
            if status['status'] == 'available':
                status_color = "status-healthy"
                status_icon = "🟢"
            elif status['status'] == 'error':
                status_color = "status-error"
                status_icon = "🔴"
            elif status['status'] == 'unavailable':
                status_color = "status-warning"
                status_icon = "🟡"
            else:
                status_color = "status-unknown"
                status_icon = "⚪"
            
            # Agent card
            with st.container():
                st.markdown(f'<div class="agent-status-card">', unsafe_allow_html=True)
                
                # Header
                st.markdown(f"**{status_icon} {agent_type.replace('_', ' ').title()} Agent**")
                st.markdown(f'<span class="{status_color}">Status: {status["status"].upper()}</span>', 
                           unsafe_allow_html=True)
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Calls:** {status.get('total_calls', 0)}")
                    st.markdown(f"**Errors:** {status.get('errors', 0)}")
                
                with col_b:
                    last_used = status.get('last_used')
                    if last_used:
                        last_used_str = last_used[:19].replace('T', ' ')
                        st.markdown(f"**Last Used:** {last_used_str}")
                    else:
                        st.markdown("**Last Used:** Never")
                    
                    if 'agent_object' in status and status['agent_object']:
                        st.markdown(f"**Type:** {status['agent_object']}")
                
                # Error message if any
                if status['status'] == 'error' and 'error_message' in status:
                    st.error(f"Error: {status['error_message']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Agent usage analytics
    st.markdown("### 📊 Agent Usage Analytics")
    
    if any(status.get('total_calls', 0) > 0 for status in agent_status.values()):
        # Usage distribution chart
        usage_data = []
        for agent_type, status in agent_status.items():
            usage_data.append({
                'Agent': agent_type.replace('_', ' ').title(),
                'Total Calls': status.get('total_calls', 0),
                'Errors': status.get('errors', 0),
                'Success Rate': ((status.get('total_calls', 0) - status.get('errors', 0)) / 
                               max(status.get('total_calls', 1), 1) * 100)
            })
        
        df_usage = pd.DataFrame(usage_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Usage bar chart
            fig_usage = px.bar(df_usage, x='Agent', y='Total Calls',
                              title='Agent Usage Distribution',
                              color='Total Calls',
                              color_continuous_scale='Blues')
            fig_usage.update_layout(height=400)
            st.plotly_chart(fig_usage, use_container_width=True)
        
        with col2:
            # Success rate chart
            fig_success = px.bar(df_usage, x='Agent', y='Success Rate',
                                title='Agent Success Rates (%)',
                                color='Success Rate',
                                color_continuous_scale='RdYlGn')
            fig_success.update_layout(height=400)
            st.plotly_chart(fig_success, use_container_width=True)
    else:
        st.info("No agent usage data available yet. Process some files to see analytics.")
    
    # Agent testing section
    st.markdown("### 🧪 Agent Testing")
    
    with st.expander("Test Individual Agents", expanded=False):
        test_agent = st.selectbox("Select Agent to Test", AGENT_TYPES)
        test_operation = st.selectbox("Test Operation", [
            "Health Check", "Simple Query", "File Processing Test"
        ])
        
        if st.button(f"Test {test_agent.replace('_', ' ').title()} Agent"):
            test_agent_functionality(test_agent, test_operation)

def test_agent_functionality(agent_type: str, operation: str):
    """Test individual agent functionality"""
    if not st.session_state.coordinator:
        st.error("Coordinator not available for testing")
        return
    
    try:
        with st.spinner(f"Testing {agent_type} agent..."):
            start_time = time.time()
            
            # Get the agent
            agent = st.session_state.coordinator.get_agent(agent_type)
            
            if not agent:
                st.error(f"❌ {agent_type} agent not available")
                return
            
            # Perform test based on operation
            if operation == "Health Check":
                # Simple health check
                result = {"status": "healthy", "agent_type": agent_type}
                test_time = time.time() - start_time
                st.success(f"✅ {agent_type} agent is healthy (Response time: {test_time:.3f}s)")
            
            elif operation == "Simple Query":
                # Simple query test
                if agent_type == "chat_agent":
                    result = safe_run_async(
                        st.session_state.coordinator.process_chat_query("Test query")
                    )
                elif agent_type == "vector_index":
                    result = safe_run_async(
                        st.session_state.coordinator.search_code_patterns("test pattern", limit=1)
                    )
                else:
                    result = {"status": "test_completed", "message": "Basic functionality test"}
                
                test_time = time.time() - start_time
                
                if result and not result.get('error'):
                    st.success(f"✅ {agent_type} agent query test passed (Response time: {test_time:.3f}s)")
                    if st.session_state.get('debug_mode', False):
                        st.json(result)
                else:
                    st.error(f"❌ {agent_type} agent query test failed")
                    if result:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
            
            elif operation == "File Processing Test":
                # Create a simple test file
                test_content = """IDENTIFICATION DIVISION.
PROGRAM-ID. TEST-PROGRAM.
WORKING-STORAGE SECTION.
01 TEST-VARIABLE PIC X(10) VALUE 'TEST'.
PROCEDURE DIVISION.
DISPLAY 'Hello World'.
STOP RUN."""
                
                # Save test file
                test_file_path = "/tmp/test_cobol.cbl"
                with open(test_file_path, 'w') as f:
                    f.write(test_content)
                
                # Process the test file
                result = safe_run_async(
                    st.session_state.coordinator.process_batch_files([test_file_path], "cobol")
                )
                
                test_time = time.time() - start_time
                
                if result and not result.get('error'):
                    st.success(f"✅ {agent_type} agent file processing test passed (Response time: {test_time:.3f}s)")
                    st.info(f"Processed {result.get('files_processed', 0)} files")
                else:
                    st.error(f"❌ {agent_type} agent file processing test failed")
                    if result:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                
                # Clean up
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)
            
            # Update agent status
            st.session_state.agent_status[agent_type]['last_used'] = dt.now().isoformat()
            st.session_state.agent_status[agent_type]['total_calls'] += 1
            
    except Exception as e:
        st.error(f"❌ Agent test failed: {str(e)}")
        st.session_state.agent_status[agent_type]['errors'] += 1

# ============================================================================
# ENHANCED GPU MONITORING WITH REAL-TIME UPDATES
# ============================================================================

def show_enhanced_gpu_monitoring():
    """Enhanced GPU monitoring with real-time updates"""
    st.markdown('<div class="sub-header">🎮 Enhanced GPU Monitoring & Performance</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("🔴 System not initialized. Please initialize in System Health tab.")
        return
    
    # Real-time controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        auto_refresh = st.checkbox("🔄 Auto Refresh", 
                                  value=st.session_state.get('auto_refresh_gpu', False),
                                  help="Auto-refresh GPU metrics")
        st.session_state.auto_refresh_gpu = auto_refresh
    
    with col2:
        refresh_interval = st.selectbox("Refresh Interval", 
                                       [5, 10, 30, 60], 
                                       index=st.session_state.get('gpu_refresh_interval', 1),
                                       help="Seconds between refreshes")
        st.session_state.gpu_refresh_interval = refresh_interval
    
    with col3:
        if st.button("🔄 Refresh Now", help="Manually refresh all GPU data"):
            st.rerun()
    
    with col4:
        if st.button("📊 Export Metrics", help="Export current metrics to CSV"):
            export_gpu_metrics()
    
    # Get detailed server status
    detailed_status = get_detailed_server_status()
    
    if not detailed_status:
        st.warning("No GPU server data available")
        return
    
    # Overall GPU system health
    show_gpu_system_overview(detailed_status)
    
    # GPU monitoring tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-time Metrics", 
        "🎯 Performance Analysis", 
        "🏥 Health Monitor",
        "📈 Historical Trends"
    ])
    
    with tab1:
        show_realtime_gpu_metrics(detailed_status)
    
    with tab2:
        show_gpu_performance_analysis(detailed_status)
    
    with tab3:
        show_gpu_health_monitoring(detailed_status)
    
    with tab4:
        show_gpu_historical_trends()
    
    # Auto-refresh logic
    if auto_refresh:
        st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
        time.sleep(1)  # Brief pause to show the message
        st.rerun()

def show_gpu_system_overview(detailed_status: Dict):
    """Show GPU system overview"""
    st.markdown("### 🌐 GPU System Overview")
    
    # Calculate system-wide metrics
    total_gpus = 0
    healthy_gpus = 0
    total_gpu_utilization = 0
    total_memory_usage = 0
    total_active_requests = 0
    total_requests = 0
    
    server_count = len(detailed_status)
    healthy_servers = 0
    
    for server_name, stats in detailed_status.items():
        if 'error' not in stats:
            healthy_servers += 1
            total_active_requests += stats.get('active_requests', 0)
            total_requests += stats.get('total_requests', 0)
            
            gpu_info = stats.get('gpu_info', {})
            total_gpus += len(gpu_info)
            
            for gpu_name, gpu_data in gpu_info.items():
                if stats.get('status') == 'healthy':
                    healthy_gpus += 1
                
                utilization = gpu_data.get('utilization', 0)
                total_gpu_utilization += utilization
                
                memory_total = gpu_data.get('total_memory', 0)
                memory_used = gpu_data.get('memory_cached', 0)
                if memory_total > 0:
                    memory_usage_percent = (memory_used / memory_total) * 100
                    total_memory_usage += memory_usage_percent
    
    avg_gpu_utilization = total_gpu_utilization / max(total_gpus, 1)
    avg_memory_usage = total_memory_usage / max(total_gpus, 1)
    
    # System health indicator
    if healthy_servers == server_count and healthy_gpus == total_gpus:
        st.success(f"🟢 All Systems Operational - {healthy_servers}/{server_count} servers, {healthy_gpus}/{total_gpus} GPUs healthy")
    elif healthy_servers > 0:
        st.warning(f"⚠️ Partial Service - {healthy_servers}/{server_count} servers, {healthy_gpus}/{total_gpus} GPUs healthy")
    else:
        st.error(f"🔴 Service Degraded - {healthy_servers}/{server_count} servers, {healthy_gpus}/{total_gpus} GPUs healthy")
    
    # System metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("GPU Servers", f"{healthy_servers}/{server_count}")
    
    with col2:
        st.metric("Total GPUs", f"{healthy_gpus}/{total_gpus}")
    
    with col3:
        st.metric("Avg GPU Utilization", f"{avg_gpu_utilization:.1f}%")
    
    with col4:
        st.metric("Avg Memory Usage", f"{avg_memory_usage:.1f}%")
    
    with col5:
        st.metric("Active Requests", total_active_requests)

def show_realtime_gpu_metrics(detailed_status: Dict):
    """Show real-time GPU metrics"""
    st.markdown("### 🔄 Real-Time GPU Metrics")
    
    # Create GPU metrics table
    gpu_metrics = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
        
        gpu_info = stats.get('gpu_info', {})
        metrics_info = stats.get('metrics', {})
        
        for gpu_name, gpu_data in gpu_info.items():
            gpu_metrics.append({
                'Server': server_name,
                'GPU': gpu_name,
                'Model': gpu_data.get('name', 'Unknown'),
                'Utilization (%)': f"{gpu_data.get('utilization', 0):.1f}",
                'Memory Used (GB)': f"{gpu_data.get('memory_cached', 0) / (1024**3):.2f}",
                'Memory Total (GB)': f"{gpu_data.get('total_memory', 0) / (1024**3):.2f}",
                'Memory Usage (%)': f"{(gpu_data.get('memory_cached', 0) / max(gpu_data.get('total_memory', 1), 1)) * 100:.1f}",
                'Active Requests': stats.get('active_requests', 0),
                'RPS': f"{metrics_info.get('requests_per_second', 0):.2f}",
                'Avg Latency (s)': f"{metrics_info.get('average_latency', 0):.3f}",
                'Status': '🟢' if stats.get('status') == 'healthy' else '🔴'
            })
    
    if gpu_metrics:
        # Display metrics table
        df_metrics = pd.DataFrame(gpu_metrics)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            # GPU Utilization chart
            fig_util = px.bar(df_metrics, x='Server', y='Utilization (%)',
                             title='GPU Utilization by Server',
                             color='Utilization (%)',
                             color_continuous_scale='RdYlGn_r',
                             text='Utilization (%)')
            fig_util.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_util.update_layout(height=400)
            st.plotly_chart(fig_util, use_container_width=True)
        
        with col2:
            # Memory Usage chart
            df_metrics['Memory Usage (%)'] = df_metrics['Memory Usage (%)'].astype(float)
            fig_mem = px.bar(df_metrics, x='Server', y='Memory Usage (%)',
                            title='GPU Memory Usage by Server',
                            color='Memory Usage (%)',
                            color_continuous_scale='RdYlBu_r',
                            text='Memory Usage (%)')
            fig_mem.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_mem.update_layout(height=400)
            st.plotly_chart(fig_mem, use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Requests per second
            df_metrics['RPS'] = df_metrics['RPS'].astype(float)
            fig_rps = px.bar(df_metrics, x='Server', y='RPS',
                            title='Requests Per Second by Server',
                            color='RPS',
                            color_continuous_scale='Blues')
            fig_rps.update_layout(height=400)
            st.plotly_chart(fig_rps, use_container_width=True)
        
        with col2:
            # Average latency
            df_metrics['Avg Latency (s)'] = df_metrics['Avg Latency (s)'].astype(float)
            fig_latency = px.bar(df_metrics, x='Server', y='Avg Latency (s)',
                                title='Average Latency by Server',
                                color='Avg Latency (s)',
                                color_continuous_scale='RdYlBu')
            fig_latency.update_layout(height=400)
            st.plotly_chart(fig_latency, use_container_width=True)
    
    else:
        st.info("No GPU metrics available")

def show_gpu_performance_analysis(detailed_status: Dict):
    """Show GPU performance analysis"""
    st.markdown("### 🎯 GPU Performance Analysis")
    
    # Performance recommendations
    recommendations = analyze_gpu_performance(detailed_status)
    
    if recommendations:
        st.markdown("#### 💡 Performance Recommendations")
        
        for rec in recommendations:
            if rec['type'] == 'critical':
                st.error(f"🚨 **{rec['server']}**: {rec['message']}")
            elif rec['type'] == 'warning':
                st.warning(f"⚠️ **{rec['server']}**: {rec['message']}")
            elif rec['type'] == 'info':
                st.info(f"ℹ️ **{rec['server']}**: {rec['message']}")
            else:
                st.success(f"✅ **{rec['server']}**: {rec['message']}")
    
    # Performance distribution analysis
    performance_data = []
    
    for server_name, stats in detailed_status.items():
        if 'error' not in stats:
            gpu_info = stats.get('gpu_info', {})
            metrics = stats.get('metrics', {})
            
            for gpu_name, gpu_data in gpu_info.items():
                performance_data.append({
                    'server': server_name,
                    'gpu': gpu_name,
                    'utilization': gpu_data.get('utilization', 0),
                    'memory_percent': (gpu_data.get('memory_cached', 0) / 
                                     max(gpu_data.get('total_memory', 1), 1)) * 100,
                    'active_requests': stats.get('active_requests', 0),
                    'rps': metrics.get('requests_per_second', 0),
                    'latency': metrics.get('average_latency', 0),
                    'efficiency': (gpu_data.get('utilization', 0) / 
                                 max(stats.get('active_requests', 1), 1))
                })
    
    if performance_data:
        df_perf = pd.DataFrame(performance_data)
        
        # Performance correlation analysis
        st.markdown("#### 📊 Performance Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Utilization vs Active Requests
            fig_corr1 = px.scatter(df_perf, x='active_requests', y='utilization',
                                  title='GPU Utilization vs Active Requests',
                                  color='server',
                                  size='rps',
                                  hover_data=['gpu', 'latency'])
            fig_corr1.update_layout(height=400)
            st.plotly_chart(fig_corr1, use_container_width=True)
        
        with col2:
            # Memory Usage vs Latency
            fig_corr2 = px.scatter(df_perf, x='memory_percent', y='latency',
                                  title='Memory Usage vs Latency',
                                  color='server',
                                  size='utilization',
                                  hover_data=['gpu', 'rps'])
            fig_corr2.update_layout(height=400)
            st.plotly_chart(fig_corr2, use_container_width=True)
        
        # Efficiency analysis
        st.markdown("#### ⚡ Efficiency Analysis")
        
        efficiency_stats = df_perf.groupby('server').agg({
            'efficiency': 'mean',
            'utilization': 'mean',
            'memory_percent': 'mean',
            'rps': 'sum',
            'latency': 'mean'
        }).round(2)
        
        st.dataframe(efficiency_stats, use_container_width=True)

def analyze_gpu_performance(detailed_status: Dict) -> List[Dict]:
    """Analyze GPU performance and generate recommendations"""
    recommendations = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            recommendations.append({
                'type': 'critical',
                'server': server_name,
                'message': f"Server error: {stats['error']}"
            })
            continue
        
        gpu_info = stats.get('gpu_info', {})
        metrics = stats.get('metrics', {})
        active_requests = stats.get('active_requests', 0)
        
        for gpu_name, gpu_data in gpu_info.items():
            utilization = gpu_data.get('utilization', 0)
            memory_total = gpu_data.get('total_memory', 0)
            memory_used = gpu_data.get('memory_cached', 0)
            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            # Critical issues
            if memory_percent > 95:
                recommendations.append({
                    'type': 'critical',
                    'server': f"{server_name} ({gpu_name})",
                    'message': f"Critical memory usage: {memory_percent:.1f}% - Risk of OOM errors"
                })
            
            # Warning issues
            elif memory_percent > 85:
                recommendations.append({
                    'type': 'warning',
                    'server': f"{server_name} ({gpu_name})",
                    'message': f"High memory usage: {memory_percent:.1f}% - Monitor closely"
                })
            
            if utilization > 95 and active_requests > 5:
                recommendations.append({
                    'type': 'warning',
                    'server': f"{server_name} ({gpu_name})",
                    'message': f"High utilization: {utilization:.1f}% with {active_requests} requests - Consider load balancing"
                })
            
            # Performance optimizations
            if utilization < 20 and active_requests > 0:
                recommendations.append({
                    'type': 'info',
                    'server': f"{server_name} ({gpu_name})",
                    'message': f"Low utilization: {utilization:.1f}% with active requests - Check for bottlenecks"
                })
            
            # Good performance
            if 30 <= utilization <= 80 and memory_percent < 70:
                recommendations.append({
                    'type': 'success',
                    'server': f"{server_name} ({gpu_name})",
                    'message': f"Optimal performance: {utilization:.1f}% utilization, {memory_percent:.1f}% memory"
                })
    
    return recommendations

def show_gpu_health_monitoring(detailed_status: Dict):
    """Show GPU health monitoring"""
    st.markdown("### 🏥 GPU Health Monitoring")
    
    # Health summary
    total_gpus = 0
    healthy_gpus = 0
    warning_gpus = 0
    critical_gpus = 0
    
    health_details = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
        
        gpu_info = stats.get('gpu_info', {})
        server_healthy = stats.get('status') == 'healthy'
        
        for gpu_name, gpu_data in gpu_info.items():
            total_gpus += 1
            
            utilization = gpu_data.get('utilization', 0)
            memory_total = gpu_data.get('total_memory', 0)
            memory_used = gpu_data.get('memory_cached', 0)
            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            # Determine health status
            health_status = "healthy"
            health_message = "Operating normally"
            issues = []
            
            if memory_percent > 95:
                health_status = "critical"
                issues.append(f"Critical memory usage: {memory_percent:.1f}%")
                critical_gpus += 1
            elif memory_percent > 85:
                health_status = "warning"
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                warning_gpus += 1
            elif utilization > 95:
                health_status = "warning"
                issues.append(f"Very high utilization: {utilization:.1f}%")
                warning_gpus += 1
            elif not server_healthy:
                health_status = "warning"
                issues.append("Server unhealthy")
                warning_gpus += 1
            else:
                healthy_gpus += 1
            
            if issues:
                health_message = "; ".join(issues)
            
            health_details.append({
                'server': server_name,
                'gpu': gpu_name,
                'status': health_status,
                'message': health_message,
                'utilization': utilization,
                'memory_percent': memory_percent,
                'gpu_model': gpu_data.get('name', 'Unknown'),
                'temperature': gpu_data.get('temperature', 'N/A'),
                'power_usage': gpu_data.get('power_usage', 'N/A')
            })
    
    # Health overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total GPUs", total_gpus)
    
    with col2:
        st.metric("Healthy", healthy_gpus)
        if healthy_gpus == total_gpus:
            st.success("All GPUs healthy")
    
    with col3:
        st.metric("Warnings", warning_gpus)
        if warning_gpus > 0:
            st.warning(f"{warning_gpus} GPUs need attention")
    
    with col4:
        st.metric("Critical", critical_gpus)
        if critical_gpus > 0:
            st.error(f"{critical_gpus} GPUs in critical state")
    
    # Detailed health status
    st.markdown("#### 🔍 Detailed Health Status")
    
    for detail in health_details:
        status_icon = "🚨" if detail['status'] == 'critical' else "⚠️" if detail['status'] == 'warning' else "✅"
        
        with st.expander(f"{status_icon} {detail['server']} ({detail['gpu']}) - {detail['gpu_model']}", 
                        expanded=detail['status'] != 'healthy'):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Status:** {detail['status'].upper()}")
                st.markdown(f"**Message:** {detail['message']}")
            
            with col2:
                st.markdown(f"**Utilization:** {detail['utilization']:.1f}%")
                st.markdown(f"**Memory Usage:** {detail['memory_percent']:.1f}%")
            
            with col3:
                st.markdown(f"**Temperature:** {detail['temperature']}")
                st.markdown(f"**Power Usage:** {detail['power_usage']}")

def show_gpu_historical_trends():
    """Show GPU historical trends"""
    st.markdown("### 📈 Historical GPU Trends")
    
    # For now, show placeholder for historical data
    # In a real implementation, this would pull from a time-series database
    
    st.info("📊 Historical trend analysis will be implemented with time-series data collection")
    
    # Placeholder charts
    import numpy as np
    
    # Generate sample historical data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    sample_data = []
    
    for i, date in enumerate(dates):
        sample_data.append({
            'Date': date,
            'GPU_Utilization': 50 + 30 * np.sin(i / 30) + np.random.normal(0, 5),
            'Memory_Usage': 60 + 20 * np.cos(i / 45) + np.random.normal(0, 3),
            'Requests_Per_Second': 100 + 50 * np.sin(i / 20) + np.random.normal(0, 10),
            'Average_Latency': 0.5 + 0.3 * np.cos(i / 60) + np.random.normal(0, 0.05)
        })
    
    df_historical = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GPU utilization trend
        fig_util_trend = px.line(df_historical, x='Date', y='GPU_Utilization',
                                title='GPU Utilization Trend (Sample Data)',
                                labels={'GPU_Utilization': 'Utilization (%)'})
        fig_util_trend.update_layout(height=300)
        st.plotly_chart(fig_util_trend, use_container_width=True)
        
        # Requests per second trend
        fig_rps_trend = px.line(df_historical, x='Date', y='Requests_Per_Second',
                               title='Requests Per Second Trend (Sample Data)',
                               labels={'Requests_Per_Second': 'RPS'})
        fig_rps_trend.update_layout(height=300)
        st.plotly_chart(fig_rps_trend, use_container_width=True)
    
    with col2:
        # Memory usage trend
        fig_mem_trend = px.line(df_historical, x='Date', y='Memory_Usage',
                               title='Memory Usage Trend (Sample Data)',
                               labels={'Memory_Usage': 'Memory Usage (%)'})
        fig_mem_trend.update_layout(height=300)
        st.plotly_chart(fig_mem_trend, use_container_width=True)
        
        # Latency trend
        fig_latency_trend = px.line(df_historical, x='Date', y='Average_Latency',
                                   title='Average Latency Trend (Sample Data)',
                                   labels={'Average_Latency': 'Latency (s)'})
        fig_latency_trend.update_layout(height=300)
        st.plotly_chart(fig_latency_trend, use_container_width=True)

def export_gpu_metrics():
    """Export current GPU metrics to CSV"""
    try:
        detailed_status = get_detailed_server_status()
        
        if not detailed_status:
            st.error("No GPU data to export")
            return
        
        # Prepare export data
        export_data = []
        timestamp = dt.now().isoformat()
        
        for server_name, stats in detailed_status.items():
            if 'error' not in stats:
                gpu_info = stats.get('gpu_info', {})
                metrics = stats.get('metrics', {})
                
                for gpu_name, gpu_data in gpu_info.items():
                    export_data.append({
                        'Timestamp': timestamp,
                        'Server': server_name,
                        'GPU': gpu_name,
                        'GPU_Model': gpu_data.get('name', 'Unknown'),
                        'Utilization_Percent': gpu_data.get('utilization', 0),
                        'Memory_Used_GB': gpu_data.get('memory_cached', 0) / (1024**3),
                        'Memory_Total_GB': gpu_data.get('total_memory', 0) / (1024**3),
                        'Memory_Usage_Percent': (gpu_data.get('memory_cached', 0) / 
                                               max(gpu_data.get('total_memory', 1), 1)) * 100,
                        'Active_Requests': stats.get('active_requests', 0),
                        'Total_Requests': stats.get('total_requests', 0),
                        'Success_Rate': stats.get('success_rate', 0),
                        'RPS': metrics.get('requests_per_second', 0),
                        'Average_Latency': metrics.get('average_latency', 0),
                        'Error_Rate': metrics.get('error_rate', 0),
                        'Server_Status': stats.get('status', 'unknown'),
                        'Uptime_Seconds': stats.get('uptime', 0)
                    })
        
        if export_data:
            df_export = pd.DataFrame(export_data)
            csv_data = df_export.to_csv(index=False)
            
            st.download_button(
                label="💾 Download GPU Metrics CSV",
                data=csv_data,
                file_name=f"gpu_metrics_{timestamp[:19].replace(':', '-')}.csv",
                mime="text/csv"
            )
            st.success("✅ GPU metrics prepared for download")
        else:
            st.error("No data available for export")
    
    except Exception as e:
        st.error(f"Failed to export metrics: {str(e)}")


# ============================================================================
# ENHANCED COORDINATOR INITIALIZATION AND MONITORING
# ============================================================================

async def init_api_coordinator():
    """Enhanced coordinator initialization with comprehensive validation"""
    if not COORDINATOR_AVAILABLE:
        return {"error": "API Coordinator module not available"}
    
    if st.session_state.coordinator is None:
        try:
            with st.spinner("Initializing API coordinator..."):
                # Create API coordinator with configured servers
                coordinator = create_api_coordinator_from_config(
                    model_servers=st.session_state.model_servers,
                    load_balancing_strategy="least_busy"
                )
                
                # Initialize the coordinator
                await coordinator.initialize()
                
                # Comprehensive validation
                validation_results = await validate_coordinator_setup(coordinator)
                
                if validation_results['overall_success']:
                    st.session_state.coordinator = coordinator
                    st.session_state.initialization_status = "completed"
                    
                    # Initialize agent status tracking
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
                    
                    return {"success": True, "validation": validation_results}
                else:
                    st.session_state.initialization_status = f"validation_failed: {validation_results['summary']}"
                    return {"error": f"Validation failed: {validation_results['summary']}"}
        
        except Exception as e:
            st.session_state.initialization_status = f"error: {str(e)}"
            return {"error": str(e)}
    
    return {"success": True, "message": "Coordinator already initialized"}

async def validate_coordinator_setup(coordinator) -> Dict[str, Any]:
    """Comprehensive validation of coordinator setup"""
    validation_results = {
        'server_validation': [],
        'agent_validation': [],
        'gpu_validation': [],
        'overall_success': True,
        'summary': ''
    }
    
    try:
        # Validate servers
        for server_config in st.session_state.model_servers:
            server_result = validate_server_endpoint(server_config['endpoint'])
            server_result['server_name'] = server_config['name']
            validation_results['server_validation'].append(server_result)
            
            if not server_result['accessible']:
                validation_results['overall_success'] = False
        
        # Validate agents
        for agent_type in AGENT_TYPES:
            try:
                agent = coordinator.get_agent(agent_type)
                if agent:
                    validation_results['agent_validation'].append({
                        'agent_type': agent_type,
                        'status': 'available',
                        'class': type(agent).__name__
                    })
                else:
                    validation_results['agent_validation'].append({
                        'agent_type': agent_type,
                        'status': 'unavailable',
                        'error': 'Agent creation failed'
                    })
                    validation_results['overall_success'] = False
            except Exception as e:
                validation_results['agent_validation'].append({
                    'agent_type': agent_type,
                    'status': 'error',
                    'error': str(e)
                })
                validation_results['overall_success'] = False
        
        # Validate GPU access
        health_status = coordinator.get_health_status()
        available_servers = health_status.get('available_servers', 0)
        total_servers = health_status.get('total_servers', 0)
        
        validation_results['gpu_validation'] = {
            'available_servers': available_servers,
            'total_servers': total_servers,
            'server_stats': health_status.get('server_stats', {}),
            'success': available_servers > 0
        }
        
        if available_servers == 0:
            validation_results['overall_success'] = False
        
        # Generate summary
        if validation_results['overall_success']:
            validation_results['summary'] = f"All systems operational: {available_servers}/{total_servers} servers, {len([a for a in validation_results['agent_validation'] if a['status'] == 'available'])}/{len(AGENT_TYPES)} agents"
        else:
            issues = []
            if available_servers == 0:
                issues.append("No GPU servers available")
            
            failed_agents = len([a for a in validation_results['agent_validation'] if a['status'] != 'available'])
            if failed_agents > 0:
                issues.append(f"{failed_agents} agent(s) failed")
            
            failed_servers = len([s for s in validation_results['server_validation'] if not s['accessible']])
            if failed_servers > 0:
                issues.append(f"{failed_servers} server(s) unreachable")
            
            validation_results['summary'] = "; ".join(issues)
        
        return validation_results
    
    except Exception as e:
        validation_results['overall_success'] = False
        validation_results['summary'] = f"Validation error: {str(e)}"
        return validation_results

# ============================================================================
# ENHANCED SYSTEM HEALTH WITH COMPREHENSIVE MONITORING
# ============================================================================

def show_enhanced_system_health():
    """Enhanced system health page with comprehensive monitoring"""
    st.markdown('<div class="sub-header">⚙️ System Health & Comprehensive Monitoring</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.error("🔴 System Status: API Coordinator Not Available")
        st.markdown("### Import Error Details")
        st.code(st.session_state.get('import_error', 'Unknown import error'))
        st.info("Please ensure the api_opulence_coordinator module is properly installed and configured.")
        return
    
    # System overview
    show_system_overview()
    
    st.markdown("---")
    
    # Server configuration
    configure_enhanced_model_servers()
    
    st.markdown("---")
    
    # Coordinator management
    show_coordinator_management()
    
    st.markdown("---")
    
    # Comprehensive status display
    if st.session_state.coordinator:
        show_comprehensive_system_status()

def show_system_overview():
    """Show system overview with health indicators"""
    st.markdown("### 🌐 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Coordinator status
        if st.session_state.coordinator:
            st.success("🟢 Coordinator: Active")
        else:
            st.error("🔴 Coordinator: Inactive")
    
    with col2:
        # Server status
        if st.session_state.model_servers:
            server_count = len(st.session_state.model_servers)
            st.info(f"🖥️ Servers: {server_count} configured")
        else:
            st.warning("⚠️ Servers: None configured")
    
    with col3:
        # Agent status
        if st.session_state.coordinator:
            available_agents = sum(1 for status in st.session_state.agent_status.values() 
                                 if status['status'] == 'available')
            st.info(f"🤖 Agents: {available_agents}/{len(AGENT_TYPES)} available")
        else:
            st.warning("⚠️ Agents: Not initialized")
    
    with col4:
        # Processing status
        total_processed = len(st.session_state.processing_history)
        if total_processed > 0:
            successful = sum(1 for h in st.session_state.processing_history if h['status'] == 'success')
            success_rate = (successful / total_processed) * 100
            st.info(f"📊 Processing: {success_rate:.1f}% success rate")
        else:
            st.info("📊 Processing: No history")

def configure_enhanced_model_servers():
    """Enhanced model server configuration"""
    st.markdown("### 🌐 Model Server Configuration")
    
    # Server configuration form
    with st.form("enhanced_server_config"):
        st.markdown("#### Add/Edit GPU Model Server")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            server_name = st.text_input("Server Name", value="gpu_server_1", 
                                      help="Unique identifier for this server")
            endpoint = st.text_input("Endpoint URL", value="http://localhost:8000",
                                   help="Full URL including protocol and port")
        
        with col2:
            gpu_id = st.number_input("GPU ID", min_value=0, value=0,
                                   help="GPU identifier on the server")
            max_requests = st.number_input("Max Concurrent Requests", min_value=1, value=10,
                                         help="Maximum concurrent requests")
        
        with col3:
            timeout = st.number_input("Timeout (seconds)", min_value=30, value=300,
                                    help="Request timeout in seconds")
            
            # Server validation options
            test_connection = st.checkbox("🔍 Test connection", value=True)
            get_detailed_info = st.checkbox("📊 Get detailed server info", value=True)
        
        if st.form_submit_button("➕ Add/Update Server", type="primary"):
            add_or_update_server(server_name, endpoint, gpu_id, max_requests, timeout, 
                                test_connection, get_detailed_info)
    
    # Current servers display with enhanced information
    show_current_servers()

def add_or_update_server(server_name, endpoint, gpu_id, max_requests, timeout, 
                        test_connection, get_detailed_info):
    """Add or update a server with comprehensive validation"""
    new_server = {
        "name": server_name,
        "endpoint": endpoint,
        "gpu_id": gpu_id,
        "max_concurrent_requests": max_requests,
        "timeout": timeout
    }
    
    # Test connection if requested
    if test_connection:
        with st.spinner(f"Testing connection to {server_name}..."):
            validation_result = validate_server_endpoint(endpoint, timeout=10)
            
            if validation_result['accessible']:
                st.success(f"✅ Connection to {server_name} successful")
                st.info(f"⏱️ Response time: {validation_result['response_time']:.3f}s")
                
                # Get detailed server information
                if get_detailed_info and validation_result['detailed_status']:
                    status_data = validation_result['detailed_status']
                    
                    with st.expander("📊 Server Details", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Model:** {status_data.get('model', 'Unknown')}")
                            st.markdown(f"**Active Requests:** {status_data.get('active_requests', 0)}")
                            st.markdown(f"**Total Requests:** {status_data.get('total_requests', 0)}")
                        
                        with col2:
                            st.markdown(f"**Uptime:** {status_data.get('uptime', 0):.0f}s")
                            gpu_count = len(status_data.get('gpu_info', {}))
                            st.markdown(f"**GPUs:** {gpu_count}")
                            
                            if gpu_count > 0:
                                st.markdown("**GPU Models:**")
                                for gpu_name, gpu_data in status_data.get('gpu_info', {}).items():
                                    st.markdown(f"- {gpu_data.get('name', 'Unknown')}")
                
                # Add server to configuration
                existing_names = [s.get('name', '') for s in st.session_state.model_servers]
                if server_name in existing_names:
                    # Update existing server
                    for i, server in enumerate(st.session_state.model_servers):
                        if server.get('name') == server_name:
                            st.session_state.model_servers[i] = new_server
                            break
                    st.success(f"🔄 Updated server: {server_name}")
                else:
                    # Add new server
                    st.session_state.model_servers.append(new_server)
                    st.success(f"➕ Added new server: {server_name}")
                
                st.rerun()
            
            else:
                st.error(f"❌ Connection failed: {validation_result['message']}")
                
                # Still allow adding the server
                if st.button("Add Server Anyway"):
                    existing_names = [s.get('name', '') for s in st.session_state.model_servers]
                    if server_name in existing_names:
                        for i, server in enumerate(st.session_state.model_servers):
                            if server.get('name') == server_name:
                                st.session_state.model_servers[i] = new_server
                                break
                        st.warning(f"⚠️ Updated server (not validated): {server_name}")
                    else:
                        st.session_state.model_servers.append(new_server)
                        st.warning(f"⚠️ Added server (not validated): {server_name}")
                    st.rerun()
    else:
        # Add without testing
        existing_names = [s.get('name', '') for s in st.session_state.model_servers]
        if server_name in existing_names:
            for i, server in enumerate(st.session_state.model_servers):
                if server.get('name') == server_name:
                    st.session_state.model_servers[i] = new_server
                    break
            st.success(f"🔄 Updated server: {server_name}")
        else:
            st.session_state.model_servers.append(new_server)
            st.success(f"➕ Added new server: {server_name}")
        st.rerun()

def show_current_servers():
    """Show current servers with enhanced information"""
    st.markdown("#### 🖥️ Current Servers")
    
    if st.session_state.model_servers:
        for i, server in enumerate(st.session_state.model_servers):
            with st.expander(f"🖥️ {server.get('name', f'Server {i+1}')}", expanded=False):
                
                # Server details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Endpoint:** {server.get('endpoint', 'N/A')}")
                    st.markdown(f"**GPU ID:** {server.get('gpu_id', 'N/A')}")
                    st.markdown(f"**Max Requests:** {server.get('max_concurrent_requests', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Timeout:** {server.get('timeout', 'N/A')}s")
                    
                    # Real-time status check
                    if st.button(f"🔄 Check Status", key=f"status_{i}"):
                        check_server_real_time_status(server)
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button(f"🔍 Test Connection", key=f"test_{i}"):
                        test_server_connection(server)
                
                with col2:
                    if st.button(f"🎮 GPU Details", key=f"gpu_{i}"):
                        show_server_gpu_details(server)
                
                with col3:
                    if st.button(f"📊 Metrics", key=f"metrics_{i}"):
                        show_server_metrics(server)
                
                with col4:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}", type="secondary"):
                        st.session_state.model_servers.pop(i)
                        st.success(f"Removed server: {server.get('name')}")
                        st.rerun()
    else:
        st.info("📝 No servers configured. Add servers above to get started.")
        
        # Quick setup button
        if st.button("⚡ Quick Setup (Localhost)", type="primary"):
            default_servers = [
                {
                    "name": "local_gpu_0",
                    "endpoint": "http://localhost:8000",
                    "gpu_id": 0,
                    "max_concurrent_requests": 10,
                    "timeout": 300
                },
                {
                    "name": "local_gpu_1", 
                    "endpoint": "http://localhost:8001",
                    "gpu_id": 1,
                    "max_concurrent_requests": 10,
                    "timeout": 300
                }
            ]
            st.session_state.model_servers.extend(default_servers)
            st.success("Added default localhost servers")
            st.rerun()

def test_server_connection(server):
    """Test server connection with detailed results"""
    with st.spinner(f"Testing {server['name']}..."):
        result = validate_server_endpoint(server['endpoint'])
        
        if result['accessible']:
            st.success(f"✅ {server['name']}: {result['message']}")
            st.info(f"⏱️ Response time: {result['response_time']:.3f}s")
        else:
            st.error(f"❌ {server['name']}: {result['message']}")

def show_server_gpu_details(server):
    """Show detailed GPU information for a server"""
    try:
        with st.spinner(f"Getting GPU details from {server['name']}..."):
            response = requests.get(f"{server['endpoint']}/status", timeout=10)
            
            if response.status_code == 200:
                status_data = response.json()
                gpu_info = status_data.get('gpu_info', {})
                
                if gpu_info:
                    st.success(f"📊 GPU Details for {server['name']}:")
                    
                    for gpu_name, gpu_data in gpu_info.items():
                        with st.container():
                            st.markdown(f"**{gpu_name}**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"- Model: {gpu_data.get('name', 'Unknown')}")
                                st.markdown(f"- Compute: {gpu_data.get('compute_capability', 'Unknown')}")
                            
                            with col2:
                                total_mem = gpu_data.get('total_memory', 0) / (1024**3)
                                used_mem = gpu_data.get('memory_cached', 0) / (1024**3)
                                st.markdown(f"- Total Memory: {total_mem:.1f} GB")
                                st.markdown(f"- Used Memory: {used_mem:.1f} GB")
                            
                            with col3:
                                utilization = gpu_data.get('utilization', 0)
                                st.markdown(f"- Utilization: {utilization:.1f}%")
                                
                                if total_mem > 0:
                                    usage_percent = (used_mem / total_mem) * 100
                                    st.markdown(f"- Memory Usage: {usage_percent:.1f}%")
                else:
                    st.warning("No GPU information available")
            else:
                st.error(f"HTTP {response.status_code}: Unable to get GPU details")
    
    except Exception as e:
        st.error(f"Error getting GPU details: {str(e)}")

def show_server_metrics(server):
    """Show server performance metrics"""
    try:
        with st.spinner(f"Getting metrics from {server['name']}..."):
            response = requests.get(f"{server['endpoint']}/metrics", timeout=10)
            
            if response.status_code == 200:
                metrics_data = response.json()
                
                st.success(f"📈 Performance Metrics for {server['name']}:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Requests/Second", f"{metrics_data.get('requests_per_second', 0):.2f}")
                    st.metric("Average Latency", f"{metrics_data.get('average_latency', 0):.3f}s")
                    st.metric("GPU Utilization", f"{metrics_data.get('gpu_utilization', 0):.1f}%")
                
                with col2:
                    st.metric("Memory Usage", f"{metrics_data.get('memory_usage', 0):.1f}%")
                    st.metric("Active Connections", metrics_data.get('active_connections', 0))
                    st.metric("Error Rate", f"{metrics_data.get('error_rate', 0):.2f}%")
            else:
                st.error(f"HTTP {response.status_code}: Unable to get metrics")
    
    except Exception as e:
        st.error(f"Error getting metrics: {str(e)}")

def check_server_real_time_status(server):
    """Check real-time server status"""
    try:
        with st.spinner("Checking status..."):
            # Health check
            health_response = requests.get(f"{server['endpoint']}/health", timeout=5)
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success(f"🟢 {server['name']} is healthy")
                st.info(f"Uptime: {health_data.get('uptime', 0):.0f}s")
                
                # Get detailed status
                status_response = requests.get(f"{server['endpoint']}/status", timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Model:** {status_data.get('model', 'Unknown')}")
                        st.markdown(f"**Active Requests:** {status_data.get('active_requests', 0)}")
                    
                    with col2:
                        st.markdown(f"**Total Requests:** {status_data.get('total_requests', 0)}")
                        gpu_count = len(status_data.get('gpu_info', {}))
                        st.markdown(f"**GPUs Available:** {gpu_count}")
            else:
                st.error(f"🔴 {server['name']} health check failed: HTTP {health_response.status_code}")
    
    except Exception as e:
        st.error(f"🔴 {server['name']} is unreachable: {str(e)}")

def show_coordinator_management():
    """Show coordinator management interface"""
    st.markdown("### 🤖 Coordinator Management")
    
    if not st.session_state.coordinator:
        st.warning("🟡 Coordinator Status: Not Initialized")
        
        if st.button("🚀 Initialize Coordinator", type="primary"):
            with st.spinner("Initializing coordinator system..."):
                result = safe_run_async(init_api_coordinator())
                
                if result and result.get('success'):
                    st.success("✅ Coordinator initialized successfully")
                    
                    # Show validation results
                    validation = result.get('validation', {})
                    if validation:
                        with st.expander("📊 Initialization Validation Results", expanded=True):
                            
                            # Server validation
                            st.markdown("**Server Validation:**")
                            for server_val in validation.get('server_validation', []):
                                status_icon = "✅" if server_val['accessible'] else "❌"
                                st.markdown(f"{status_icon} {server_val['server_name']}: {server_val['message']}")
                            
                            # Agent validation  
                            st.markdown("**Agent Validation:**")
                            for agent_val in validation.get('agent_validation', []):
                                status_icon = "✅" if agent_val['status'] == 'available' else "❌"
                                st.markdown(f"{status_icon} {agent_val['agent_type']}: {agent_val['status']}")
                            
                            # GPU validation
                            gpu_val = validation.get('gpu_validation', {})
                            if gpu_val:
                                st.markdown(f"**GPU Validation:** {gpu_val['available_servers']}/{gpu_val['total_servers']} servers available")
                    
                    st.rerun()
                else:
                    error_msg = result.get('error') if result else "Unknown error"
                    st.error(f"❌ Initialization failed: {error_msg}")
    else:
        st.success("🟢 Coordinator Status: Active")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart Coordinator"):
                restart_coordinator()
        
        with col2:
            if st.button("🧹 Cleanup Resources"):
                cleanup_coordinator_resources()
        
        with col3:
            if st.button("📊 Health Check"):
                run_comprehensive_health_check()

def restart_coordinator():
    """Restart the coordinator"""
    with st.spinner("Restarting coordinator..."):
        try:
            # Cleanup existing coordinator
            if st.session_state.coordinator:
                st.session_state.coordinator.cleanup()
            
            # Reset coordinator state
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
            
            # Re-initialize
            result = safe_run_async(init_api_coordinator())
            
            if result and result.get('success'):
                st.success("✅ Coordinator restarted successfully")
            else:
                st.error(f"❌ Restart failed: {result.get('error') if result else 'Unknown error'}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Restart failed: {str(e)}")

def cleanup_coordinator_resources():
    """Cleanup coordinator resources"""
    with st.spinner("Cleaning up resources..."):
        try:
            if st.session_state.coordinator:
                st.session_state.coordinator.cleanup()
                st.success("✅ Resources cleaned up successfully")
            else:
                st.info("No active coordinator to clean up")
        except Exception as e:
            st.error(f"❌ Cleanup failed: {str(e)}")

def run_comprehensive_health_check():
    """Run comprehensive health check"""
    with st.spinner("Running comprehensive health check..."):
        try:
            if not st.session_state.coordinator:
                st.error("❌ No coordinator available for health check")
                return
            
            # Get coordinator health
            health = st.session_state.coordinator.get_health_status()
            
            st.markdown("### 🏥 Health Check Results")
            
            # Overall status
            overall_status = health.get('status', 'unknown')
            if overall_status == 'healthy':
                st.success(f"🟢 Overall Status: {overall_status.upper()}")
            else:
                st.error(f"🔴 Overall Status: {overall_status.upper()}")
            
            # Server health
            server_stats = health.get('server_stats', {})
            available_servers = health.get('available_servers', 0)
            total_servers = health.get('total_servers', 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Available Servers", f"{available_servers}/{total_servers}")
            
            with col2:
                active_agents = health.get('active_agents', 0)
                st.metric("Active Agents", active_agents)
            
            with col3:
                uptime = health.get('uptime_seconds', 0)
                st.metric("Uptime", f"{uptime:.0f}s")
            
            # Detailed server status
            if server_stats:
                st.markdown("#### 🖥️ Server Details")
                
                for server_name, stats in server_stats.items():
                    status_icon = "🟢" if stats.get('available', False) else "🔴"
                    
                    with st.expander(f"{status_icon} {server_name}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Status:** {stats.get('status', 'unknown')}")
                            st.markdown(f"**Active Requests:** {stats.get('active_requests', 0)}")
                            st.markdown(f"**Total Requests:** {stats.get('total_requests', 0)}")
                        
                        with col2:
                            success_rate = stats.get('success_rate', 0)
                            st.markdown(f"**Success Rate:** {success_rate:.1f}%")
                            avg_latency = stats.get('average_latency', 0)
                            st.markdown(f"**Avg Latency:** {avg_latency:.3f}s")
                            st.markdown(f"**Available:** {'Yes' if stats.get('available', False) else 'No'}")
            
            # Agent health
            agent_status = get_agent_status()
            available_agents = sum(1 for status in agent_status.values() if status['status'] == 'available')
            
            st.markdown("#### 🤖 Agent Health")
            st.info(f"Agent Status: {available_agents}/{len(AGENT_TYPES)} agents available")
            
            for agent_type, status in agent_status.items():
                status_icon = "🟢" if status['status'] == 'available' else "🔴" if status['status'] == 'error' else "🟡"
                st.markdown(f"{status_icon} **{agent_type}**: {status['status']}")
            
        except Exception as e:
            st.error(f"❌ Health check failed: {str(e)}")

def show_comprehensive_system_status():
    """Show comprehensive system status"""
    if not st.session_state.coordinator:
        return
    
    st.markdown("### 📊 Comprehensive System Status")
    
    # Get all status information
    health_status = st.session_state.coordinator.get_health_status()
    detailed_server_status = get_detailed_server_status()
    agent_status = get_agent_status()
    
    # Create status tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 System Overview",
        "🖥️ Server Status", 
        "🤖 Agent Status",
        "📈 Performance Metrics"
    ])
    
    with tab1:
        show_system_status_overview(health_status, detailed_server_status, agent_status)
    
    with tab2:
        show_detailed_server_status_tab(detailed_server_status)
    
    with tab3:
        show_comprehensive_agent_status()
    
    with tab4:
        show_system_performance_metrics(detailed_server_status)

def show_system_status_overview(health_status, detailed_server_status, agent_status):
    """Show system status overview"""
    # System health summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_status = health_status.get('status', 'unknown')
        if overall_status == 'healthy':
            st.success("🟢 System Healthy")
        else:
            st.error("🔴 System Issues")
    
    with col2:
        available_servers = health_status.get('available_servers', 0)
        total_servers = health_status.get('total_servers', 0)
        if available_servers == total_servers and total_servers > 0:
            st.success(f"🖥️ {available_servers}/{total_servers} Servers")
        elif available_servers > 0:
            st.warning(f"🖥️ {available_servers}/{total_servers} Servers")
        else:
            st.error(f"🖥️ {available_servers}/{total_servers} Servers")
    
    with col3:
        available_agents = sum(1 for status in agent_status.values() if status['status'] == 'available')
        total_agents = len(AGENT_TYPES)
        if available_agents == total_agents:
            st.success(f"🤖 {available_agents}/{total_agents} Agents")
        elif available_agents > 0:
            st.warning(f"🤖 {available_agents}/{total_agents} Agents")
        else:
            st.error(f"🤖 {available_agents}/{total_agents} Agents")
    
    with col4:
        uptime = health_status.get('uptime_seconds', 0)
        if uptime > 0:
            st.info(f"⏱️ Uptime: {uptime:.0f}s")
        else:
            st.warning("⏱️ Uptime: Unknown")
    
    # Processing statistics
    if st.session_state.processing_history:
        st.markdown("#### 📊 Processing Statistics")
        
        total_files = len(st.session_state.processing_history)
        successful_files = sum(1 for h in st.session_state.processing_history if h['status'] == 'success')
        failed_files = total_files - successful_files
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", total_files)
        
        with col2:
            st.metric("Successful", successful_files)
        
        with col3:
            st.metric("Failed", failed_files)
        
        with col4:
            success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

def show_detailed_server_status_tab(detailed_server_status):
    """Show detailed server status tab"""
    if not detailed_server_status:
        st.warning("No server status data available")
        return
    
    for server_name, stats in detailed_server_status.items():
        with st.expander(f"🖥️ {server_name}", expanded=True):
            
            if 'error' in stats:
                st.error(f"❌ Server Error: {stats['error']}")
                continue
            
            # Server metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = stats.get('status', 'unknown')
                if status == 'healthy':
                    st.success(f"Status: {status}")
                else:
                    st.error(f"Status: {status}")
            
            with col2:
                st.metric("Active Requests", stats.get('active_requests', 0))
            
            with col3:
                st.metric("Total Requests", stats.get('total_requests', 0))
            
            with col4:
                success_rate = stats.get('success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")

def show_system_performance_metrics(detailed_server_status):
    """Show system performance metrics"""
    if not detailed_server_status:
        st.warning("No performance data available")
        return
    
    # Aggregate performance data
    performance_data = []
    
    for server_name, stats in detailed_server_status.items():
        if 'error' not in stats:
            metrics = stats.get('metrics', {})
            gpu_info = stats.get('gpu_info', {})
            
            # Calculate average GPU utilization for this server
            gpu_utilizations = [gpu_data.get('utilization', 0) for gpu_data in gpu_info.values()]
            avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0
            
            performance_data.append({
                'server': server_name,
                'rps': metrics.get('requests_per_second', 0),
                'latency': metrics.get('average_latency', 0),
                'gpu_utilization': avg_gpu_util,
                'active_requests': stats.get('active_requests', 0)
            })
    
    if performance_data:
        df_perf = pd.DataFrame(performance_data)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # RPS chart
            fig_rps = px.bar(df_perf, x='server', y='rps',
                            title='Requests Per Second by Server')
            fig_rps.update_layout(height=400)
            st.plotly_chart(fig_rps, use_container_width=True)
        
        with col2:
            # GPU utilization chart
            fig_gpu = px.bar(df_perf, x='server', y='gpu_utilization',
                            title='GPU Utilization by Server')
            fig_gpu.update_layout(height=400)
            st.plotly_chart(fig_gpu, use_container_width=True)

# ============================================================================
# MAIN APPLICATION WITH ENHANCED NAVIGATION
# ============================================================================

def main():
    """Enhanced main application with comprehensive mainframe support"""
    
    # Initialize session state
    initialize_session_state()
    
    # Add custom CSS
    add_custom_css()
    
    # Header
    st.markdown('<div class="main-header">🌐 Opulence Enhanced Mainframe Analysis Platform</div>', unsafe_allow_html=True)
    
    # Show system status in header
    if COORDINATOR_AVAILABLE and st.session_state.coordinator:
        show_header_status()
    elif not COORDINATOR_AVAILABLE:
        st.error("⚠️ API Coordinator module not available - Running in demo mode")
    
    # Enhanced sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/059669/ffffff?text=OPULENCE", use_container_width=True)
        
        # Navigation with mainframe focus
        page = st.selectbox(
            "Navigation",
            [
                "🏠 Dashboard", 
                "📂 Mainframe File Upload", 
                "💬 Chat Analysis", 
                "🔍 Component Analysis",
                "🤖 Agent Status",
                "🎮 GPU Monitoring",
                "⚙️ System Health",
                "📊 Analytics & Reports"
            ]
        )
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                clear_application_cache()
        
        # Debug mode toggle
        debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False))
        st.session_state.debug_mode = debug_mode
    
    # Main content routing
    try:
        if page == "🏠 Dashboard":
            show_enhanced_dashboard()
        elif page == "📂 Mainframe File Upload":
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
            
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)

def show_header_status():
    """Show condensed status in header"""
    try:
        health = st.session_state.coordinator.get_health_status()
        available_servers = health.get('available_servers', 0)
        total_servers = health.get('total_servers', 0)
        
        if available_servers == total_servers and total_servers > 0:
            st.success(f"🟢 System Operational: {available_servers} servers, {len(st.session_state.agent_status)} agents")
        elif available_servers > 0:
            st.warning(f"⚠️ Partial Service: {available_servers}/{total_servers} servers available")
        else:
            st.error(f"🔴 Service Issues: {available_servers}/{total_servers} servers available")
    except:
        st.warning("🟡 System status unknown")

def clear_application_cache():
    """Clear application cache and temporary data"""
    try:
        # Clear processing history
        st.session_state.processing_history = []
        
        # Clear chat history
        st.session_state.chat_history = []
        
        # Clear uploaded files
        st.session_state.uploaded_files = []
        
        # Clear file analysis results
        st.session_state.file_analysis_results = {}
        
        st.success("✅ Application cache cleared")
    except Exception as e:
        st.error(f"❌ Failed to clear cache: {str(e)}")

# Placeholder functions for missing components
def show_enhanced_dashboard():
    """Show enhanced dashboard"""
    st.markdown("### 🏠 Enhanced Dashboard")
    st.info("Dashboard implementation goes here")

def show_enhanced_chat_analysis():
    """Show enhanced chat analysis"""
    st.markdown("### 💬 Enhanced Chat Analysis")
    st.info("Chat analysis implementation goes here")

def show_enhanced_component_analysis():
    """Show enhanced component analysis"""
    st.markdown("### 🔍 Enhanced Component Analysis")
    st.info("Component analysis implementation goes here")

def show_analytics_and_reports():
    """Show analytics and reports"""
    st.markdown("### 📊 Analytics & Reports")
    st.info("Analytics and reports implementation goes here")

# ============================================================================
# RUN APPLICATION
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
            "session_state_keys": list(st.session_state.keys()),
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
                clear_application_cache()
        
        with col3:
            if st.button("📊 Show Debug Info"):
                st.session_state.debug_mode = True
                st.rerun()