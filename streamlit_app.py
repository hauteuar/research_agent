# streamlit_app.py
"""
Opulence Deep Research Mainframe Agent - Streamlit UI (Single GPU Version)
Updated for SingleGPUOpulenceCoordinator
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

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Opulence - Single GPU Deep Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state FIRST
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

# Call initialization immediately
initialize_session_state()

# Try to import single GPU coordinator with error handling
try:
    from opulence_coordinator import (
        SingleGPUOpulenceCoordinator,
        SingleGPUOpulenceConfig,
        SingleGPUChatEnhancer,
        create_single_gpu_coordinator,
        create_shared_server_coordinator,
        create_dedicated_server_coordinator,
        get_global_coordinator
    )
    COORDINATOR_AVAILABLE = True
except ImportError as e:
    COORDINATOR_AVAILABLE = False
    st.session_state.import_error = str(e)

# Custom CSS (same as before)
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
    
    /* Custom button styling */
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
    
    /* GPU indicator styling */
    .gpu-indicator {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def show_debug_info():
    """Show debug information"""
    if st.session_state.debug_mode:
        st.sidebar.markdown("### 🐛 Debug Info")
        debug_info = {
            "coordinator_available": COORDINATOR_AVAILABLE,
            "coordinator_type": "single_gpu" if st.session_state.coordinator else None,
            "coordinator_exists": st.session_state.coordinator is not None,
            "init_status": st.session_state.initialization_status,
            "import_error": st.session_state.get('import_error', 'None'),
            "session_keys": list(st.session_state.keys())
        }
        
        if st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                debug_info.update({
                    "selected_gpu": health.get('selected_gpu'),
                    "gpu_locked": health.get('gpu_status', {}).get('is_locked', False),
                    "active_agents": health.get('active_agents', 0),
                    "llm_engine": health.get('llm_engine_available', False)
                })
            except:
                debug_info["coordinator_status"] = "error_getting_status"
        
        st.sidebar.json(debug_info)


async def init_single_gpu_coordinator():
    """Initialize the single GPU coordinator"""
    if not COORDINATOR_AVAILABLE:
        return False
        
    if st.session_state.coordinator is None:
        try:
            # Try to get global coordinator first
            try:
                st.session_state.coordinator = get_global_coordinator()
                st.session_state.initialization_status = "completed"
                return True
            except Exception as e:
                # Create new coordinator if global one fails
                st.session_state.coordinator = create_single_gpu_coordinator()
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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop for async operations
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
            "response": "❌ Single GPU Coordinator not available. Please check the import error in debug mode.",
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
        
        # Process with the single GPU coordinator
        result = safe_run_async(
            st.session_state.coordinator.process_chat_query(query, conversation_history)
        )
        
        if isinstance(result, dict):
            # Add GPU info to response
            if "gpu_used" not in result:
                result["gpu_used"] = getattr(st.session_state.coordinator, 'selected_gpu', 'unknown')
            return result
        else:
            return {
                "response": str(result),
                "response_type": "general",
                "suggestions": [],
                "gpu_used": getattr(st.session_state.coordinator, 'selected_gpu', 'unknown')
            }
    
    except Exception as e:
        return {
            "response": f"❌ Error processing query: {str(e)}",
            "response_type": "error",
            "suggestions": ["Try rephrasing your question", "Check system status"],
            "gpu_used": getattr(st.session_state.coordinator, 'selected_gpu', 'unknown')
        }


def show_gpu_status():
    """Show current GPU status for single GPU system"""
    if st.session_state.coordinator:
        try:
            health = st.session_state.coordinator.get_health_status()
            gpu_id = health.get('selected_gpu', 'Unknown')
            gpu_status = health.get('gpu_status', {})
            
            # GPU indicator
            st.markdown(f"""
                <div class="gpu-indicator">
                    🎯 Single GPU Mode: GPU {gpu_id} 
                    {"🔒 Locked" if gpu_status.get('is_locked', False) else "🔓 Available"}
                </div>
            """, unsafe_allow_html=True)
            
            # GPU metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                memory_used = gpu_status.get('memory_usage_gb', 0)
                st.metric("GPU Memory", f"{memory_used:.1f}GB")
            
            with col2:
                active_tasks = gpu_status.get('active_tasks', 0)
                st.metric("Active Tasks", active_tasks)
            
            with col3:
                total_tasks = gpu_status.get('total_tasks_processed', 0)
                st.metric("Tasks Completed", total_tasks)
            
            with col4:
                uptime = health.get('uptime_seconds', 0)
                st.metric("Uptime", f"{uptime:.0f}s")
                
        except Exception as e:
            st.error(f"Error getting GPU status: {str(e)}")
    else:
        st.warning("🟡 Single GPU Coordinator not initialized")


def show_enhanced_chat_analysis():
    """Enhanced chat analysis interface for single GPU"""
    st.markdown('<div class="sub-header">💬 Chat with Single GPU Opulence</div>', unsafe_allow_html=True)
    
    # Show GPU status first
    show_gpu_status()
    
    # Chat status indicator
    if st.session_state.coordinator:
        try:
            health = st.session_state.coordinator.get_health_status()
            if health.get("llm_engine_available", False):
                gpu_id = health.get('selected_gpu', 'Unknown')
                st.success(f"🟢 Chat Agent Ready on GPU {gpu_id}")
            else:
                st.warning("🟡 Chat Agent: LLM Engine not available")
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
                        
                        # Show GPU info if available
                        gpu_used = response_data.get("gpu_used")
                        if gpu_used:
                            st.caption(f"🎯 Processed on GPU {gpu_used}")
                        
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
        with st.spinner("🧠 Single GPU Opulence is thinking..."):
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
    """Process uploaded files using the single GPU coordinator"""
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
        
        # Process files in batch using single GPU coordinator
        status_text.text("Processing files with Single GPU Opulence...")
        
        # Use the coordinator's process_batch_files method
        result = safe_run_async(
            st.session_state.coordinator.process_batch_files(file_paths)
        )
        
        # Display results
        with results_container:
            if isinstance(result, dict) and result.get("status") == "success":
                gpu_used = result.get('gpu_used', 'Unknown')
                st.success(f"✅ Successfully processed {result.get('files_processed', 0)} files in {result.get('processing_time', 0):.2f} seconds on GPU {gpu_used}")
                
                # Show detailed results
                results_list = result.get("results", [])
                for i, file_result in enumerate(results_list):
                    if isinstance(file_result, dict):
                        with st.expander(f"📄 {uploaded_files[i].name}"):
                            st.json(file_result)
                    else:
                        with st.expander(f"📄 {uploaded_files[i].name}"):
                            st.text(str(file_result))
                            
                # Show vector indexing status
                vector_status = result.get('vector_indexing', 'unknown')
                if vector_status == 'completed':
                    st.info("✅ Vector embeddings created successfully")
                elif vector_status == 'skipped':
                    st.info("ℹ️ Vector embedding creation skipped")
                    
            else:
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                st.error(f"❌ Processing failed: {error_msg}")
        
        # Update processing history
        st.session_state.processing_history.append({
            "timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_count": len(uploaded_files),
            "status": result.get("status", "error") if isinstance(result, dict) else "error",
            "processing_time": result.get("processing_time", 0) if isinstance(result, dict) else 0,
            "gpu_used": result.get("gpu_used", "unknown") if isinstance(result, dict) else "unknown"
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


def analyze_component_single_gpu(component_name: str, component_type: str, user_question: str = None, chat_enhanced: bool = True):
    """Enhanced component analysis using single GPU coordinator"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"🧠 Analyzing {component_name} on single GPU..."):
        try:
            if chat_enhanced and user_question:
                # Use chat-enhanced analysis if available
                try:
                    enhancer = SingleGPUChatEnhancer(st.session_state.coordinator)
                    result = safe_run_async(
                        enhancer.chat_analyze_component(
                            component_name, 
                            user_question,
                            st.session_state.chat_history[-3:] if st.session_state.chat_history else []
                        )
                    )
                except Exception as e:
                    st.warning(f"Chat enhancement failed, using regular analysis: {str(e)}")
                    # Fallback to regular analysis
                    component_type_param = None if component_type == "auto-detect" else component_type
                    result = safe_run_async(
                        st.session_state.coordinator.analyze_component(
                            component_name, 
                            component_type_param
                        )
                    )
            else:
                # Use regular analysis
                component_type_param = None if component_type == "auto-detect" else component_type
                result = safe_run_async(
                    st.session_state.coordinator.analyze_component(
                        component_name, 
                        component_type_param
                    )
                )
                
                # Add chat explanation if requested
                if chat_enhanced:
                    try:
                        chat_query = f"Explain the analysis results for {component_name} in a conversational way."
                        chat_result = safe_run_async(
                            st.session_state.coordinator.process_chat_query(chat_query, [])
                        )
                        result = {
                            "component_name": component_name,
                            "analysis": result,
                            "chat_explanation": chat_result.get("response", ""),
                            "suggestions": chat_result.get("suggestions", []),
                            "response_type": "enhanced_analysis",
                            "gpu_used": getattr(st.session_state.coordinator, 'selected_gpu', 'unknown')
                        }
                    except Exception as e:
                        st.warning(f"Chat explanation failed: {str(e)}")
            
            st.session_state.current_analysis = result
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


def show_system_health():
    """Show system health and statistics for single GPU system"""
    st.markdown('<div class="sub-header">⚙️ Single GPU System Health & Statistics</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.error("🔴 System Status: Single GPU Coordinator Not Available")
        st.markdown("### Import Error")
        st.code(st.session_state.get('import_error', 'Unknown import error'))
        st.info("Please ensure the opulence_coordinator_single_gpu module is properly installed and configured.")
        return
    
    if not st.session_state.coordinator:
        st.warning("🟡 System Status: Not Initialized")
        if st.button("🔄 Initialize Single GPU System"):
            with st.spinner("Initializing single GPU system..."):
                try:
                    success = safe_run_async(init_single_gpu_coordinator())
                    if success and not isinstance(success, dict):
                        st.success("✅ Single GPU system initialized successfully")
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
        
        # Overall health indicator
        if health.get("status") == "healthy":
            st.success("🟢 Single GPU System Status: Healthy")
        else:
            st.error("🔴 Single GPU System Status: Issues Detected")
        
        # Show GPU information prominently
        show_gpu_status()
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Agents", health.get("active_agents", 0))
        
        with col2:
            llm_status = "Available" if health.get("llm_engine_available", False) else "Not Available"
            st.metric("LLM Engine", llm_status)
        
        with col3:
            db_status = "Available" if health.get("database_available", False) else "Not Available"
            st.metric("Database", db_status)
        
        with col4:
            coordinator_type = health.get("coordinator_type", "unknown")
            st.metric("Coordinator Type", coordinator_type.replace("_", " ").title())
        
        # GPU utilization chart
        gpu_status = health.get("gpu_status", {})
        if gpu_status:
            st.markdown("### GPU Utilization")
            
            # Create GPU metrics visualization
            gpu_data = {
                "Metric": ["Memory Used", "Active Tasks", "Total Tasks"],
                "Value": [
                    gpu_status.get("memory_usage_gb", 0),
                    gpu_status.get("active_tasks", 0),
                    gpu_status.get("total_tasks_processed", 0)
                ],
                "Unit": ["GB", "Tasks", "Tasks"]
            }
            
            df_gpu = pd.DataFrame(gpu_data)
            fig = px.bar(df_gpu, x="Metric", y="Value", 
                        title=f"GPU {health.get('selected_gpu', 'Unknown')} Status")
            st.plotly_chart(fig, use_container_width=True)
        
        # Processing statistics
        if isinstance(stats_result, dict) and "processing_stats" in stats_result:
            processing_stats = stats_result["processing_stats"]
            if processing_stats:
                st.markdown("### Processing Performance")
                df_perf = pd.DataFrame(processing_stats)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'avg_duration' in df_perf.columns and 'operation' in df_perf.columns:
                        fig_duration = px.bar(df_perf, x="operation", y="avg_duration", 
                                            title="Average Processing Time by Operation")
                        st.plotly_chart(fig_duration, use_container_width=True)
                
                with col2:
                    if 'count' in df_perf.columns and 'operation' in df_perf.columns:
                        fig_count = px.bar(df_perf, x="operation", y="count", 
                                         title="Operation Count")
                        st.plotly_chart(fig_count, use_container_width=True)
        
        # System controls
        st.markdown("### System Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Statistics"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clean GPU Memory"):
                try:
                    st.session_state.coordinator.cleanup()
                    st.success("GPU memory cleaned")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clean memory: {str(e)}")
        
        with col3:
            if st.button("📊 Rebuild Indices"):
                rebuild_indices()
        
        with col4:
            if st.button("📥 Export Logs"):
                export_system_logs()
        
        # Show database statistics
        if isinstance(stats_result, dict) and "database_stats" in stats_result:
            db_stats = stats_result["database_stats"]
            if db_stats and not db_stats.get("error"):
                st.markdown("### Database Statistics")
                
                # Create database stats visualization
                db_data = []
                for key, value in db_stats.items():
                    if key.endswith("_count") and isinstance(value, (int, float)):
                        table_name = key.replace("_count", "").replace("_", " ").title()
                        db_data.append({"Table": table_name, "Records": value})
                
                if db_data:
                    df_db = pd.DataFrame(db_data)
                    fig_db = px.bar(df_db, x="Table", y="Records", 
                                   title="Database Records by Table")
                    st.plotly_chart(fig_db, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error getting system health: {str(e)}")
        st.exception(e)


def rebuild_indices():
    """Rebuild vector indices using single GPU"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Rebuilding indices on single GPU..."):
        try:
            vector_agent = st.session_state.coordinator.get_agent("vector_index")
            result = safe_run_async(vector_agent.rebuild_index())
            
            if isinstance(result, dict) and result.get("status") == "success":
                st.success("✅ Indices rebuilt successfully")
            else:
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                st.error(f"❌ Index rebuild failed: {error_msg}")
                
        except Exception as e:
            st.error(f"Index rebuild failed: {str(e)}")


def export_system_logs():
    """Export system logs"""
    try:
        # Read log file
        log_file = Path("opulence_single_gpu.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            st.download_button(
                "📥 Download Single GPU System Logs",
                log_content,
                file_name=f"opulence_single_gpu_logs_{dt.now().strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain"
            )
        else:
            st.warning("No log file found")
            
    except Exception as e:
        st.error(f"Failed to export logs: {str(e)}")


def show_dashboard():
    """Show main dashboard for single GPU system"""
    st.markdown('<div class="sub-header">Single GPU System Overview</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.warning("System not initialized")
        return
    
    # Get system statistics
    try:
        stats_result = safe_run_async(st.session_state.coordinator.get_statistics())
        health = st.session_state.coordinator.get_health_status()
        
        # Show GPU status prominently
        show_gpu_status()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        system_stats = stats_result.get("system_stats", {}) if isinstance(stats_result, dict) else {}
        
        with col1:
            st.metric("Files Processed", system_stats.get("total_files_processed", 0))
        
        with col2:
            st.metric("Total Queries", system_stats.get("total_queries", 0))
        
        with col3:
            avg_time = system_stats.get("avg_response_time", 0)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        with col4:
            tasks_completed = system_stats.get("tasks_completed", 0)
            st.metric("Tasks Completed", tasks_completed)
        
        # Processing statistics chart
        if isinstance(stats_result, dict) and "processing_stats" in stats_result:
            processing_stats = stats_result["processing_stats"]
            if processing_stats:
                st.markdown("### Processing Statistics")
                df_stats = pd.DataFrame(processing_stats)
                
                fig = px.bar(df_stats, x="operation", y="avg_duration", 
                            title="Average Processing Time by Operation")
                st.plotly_chart(fig, use_container_width=True)
        
        # File statistics
        if isinstance(stats_result, dict) and "file_stats" in stats_result:
            file_stats = stats_result["file_stats"]
            if file_stats:
                st.markdown("### File Processing Status")
                df_files = pd.DataFrame(file_stats)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'file_type' in df_files.columns and 'count' in df_files.columns:
                        fig_pie = px.pie(df_files, values="count", names="file_type", 
                                       title="Files by Type")
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    if 'processing_status' in df_files.columns:
                        fig_status = px.bar(df_files, x="processing_status", y="count", 
                                          color="file_type", title="Processing Status")
                        st.plotly_chart(fig_status, use_container_width=True)
        
        # Recent activity
        st.markdown("### Recent Activity")
        if st.session_state.processing_history:
            for activity in st.session_state.processing_history[-5:]:
                gpu_info = f" (GPU {activity.get('gpu_used', 'Unknown')})" if activity.get('gpu_used') else ""
                st.info(f"🕐 {activity['timestamp']}: Processed {activity['files_count']} files{gpu_info}")
        else:
            st.info("No recent activities to display")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")


def show_file_upload():
    """Show file upload interface for single GPU system"""
    st.markdown('<div class="sub-header">📂 File Upload & Processing (Single GPU)</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.warning("⚠️ Single GPU Coordinator not available. File processing is disabled in demo mode.")
        st.info("To enable file processing, ensure the opulence_coordinator_single_gpu module is properly installed.")
        return
    
    if not st.session_state.coordinator:
        st.error("System not initialized. Please go to System Health tab to initialize.")
        return
    
    # Show current GPU status
    show_gpu_status()
    
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
            
            if st.button("Process Files on Single GPU"):
                process_uploaded_files(uploaded_files)
    
    elif upload_type == "Batch Upload (ZIP)":
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing multiple files",
            type=['zip']
        )
        
        if uploaded_zip:
            if st.button("Extract and Process ZIP on Single GPU"):
                process_zip_file(uploaded_zip)
    
    # Processing history with GPU information
    st.markdown("### Processing History")
    if st.session_state.processing_history:
        df_history = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No processing history available")


def process_zip_file(uploaded_zip):
    """Process uploaded ZIP file using single GPU"""
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
        
        st.info(f"Found {len(file_paths)} files to process on single GPU")
        
        # Process files using single GPU coordinator
        result = safe_run_async(
            st.session_state.coordinator.process_batch_files(file_paths)
        )
        
        if isinstance(result, dict) and result.get("status") == "success":
            gpu_used = result.get('gpu_used', 'Unknown')
            st.success(f"✅ Successfully processed {result.get('files_processed', 0)} files on GPU {gpu_used}")
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
    """Enhanced component analysis with single GPU chat integration"""
    st.markdown('<div class="sub-header">🔍 Enhanced Component Analysis (Single GPU)</div>', unsafe_allow_html=True)
    
    # Show GPU status
    show_gpu_status()
    
    # Component selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        component_name = st.text_input("Component Name (file, table, program, field):")
    
    with col2:
        component_type = st.selectbox(
            "Component Type",
            ["auto-detect", "file", "table", "program", "jcl", "field"]
        )
    
    with col3:
        chat_enhanced = st.checkbox("Use Chat Enhancement", value=True)
    
    # Optional user question
    user_question = st.text_input("Specific question about this component (optional):")
    
    if st.button("🔍 Analyze Component on Single GPU") and component_name:
        analyze_component_single_gpu(component_name, component_type, user_question, chat_enhanced)
    
    # Display current analysis with chat integration
    if st.session_state.current_analysis:
        display_enhanced_component_analysis(st.session_state.current_analysis)


def show_enhanced_search():
    """Enhanced search interface with single GPU chat integration"""
    st.markdown('<div class="sub-header">🔍 Enhanced Code Search (Single GPU)</div>', unsafe_allow_html=True)
    
    # Show GPU status
    show_gpu_status()
    
    # Search input
    search_query = st.text_input("Describe what you're looking for:")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        search_type = st.selectbox("Search Type", ["Functionality", "Code Pattern", "Business Logic", "Data Flow"])
    
    with col2:
        result_count = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
    
    if st.button("🔍 Search with Single GPU Enhancement") and search_query:
        perform_enhanced_search_single_gpu(search_query, search_type, result_count)


def perform_enhanced_search_single_gpu(search_query: str, search_type: str, result_count: int):
    """Perform enhanced search with single GPU chat explanations"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"🔍 Searching for '{search_query}' on single GPU..."):
        try:
            # Use chat-enhanced search if available
            try:
                enhancer = SingleGPUChatEnhancer(st.session_state.coordinator)
                result = safe_run_async(
                    enhancer.chat_search_patterns(
                        f"{search_type}: {search_query}",
                        st.session_state.chat_history[-3:] if st.session_state.chat_history else []
                    )
                )
            except Exception as e:
                st.warning(f"Chat enhancement failed, using regular search: {str(e)}")
                # Fallback to regular search
                result = safe_run_async(
                    st.session_state.coordinator.search_code_patterns(search_query, limit=result_count)
                )
                # Convert to enhanced format
                result = {
                    "search_description": search_query,
                    "search_results": result.get("results", []),
                    "chat_explanation": f"Found {len(result.get('results', []))} code patterns matching '{search_query}'",
                    "total_found": len(result.get("results", [])),
                    "suggestions": ["Refine search terms", "Try different keywords"],
                    "response_type": "enhanced_search",
                    "gpu_used": getattr(st.session_state.coordinator, 'selected_gpu', 'unknown')
                }
            
            # Display results
            if result.get("response_type") == "enhanced_search":
                gpu_used = result.get("gpu_used", "Unknown")
                st.success(f"✅ Found {result.get('total_found', 0)} results on GPU {gpu_used}")
                
                # Show chat explanation
                if result.get("chat_explanation"):
                    st.markdown("### 🧠 Search Analysis")
                    st.markdown(result["chat_explanation"])
                
                # Show search results
                search_results = result.get("search_results", [])
                if search_results:
                    st.markdown("### 📋 Search Results")
                    for i, search_result in enumerate(search_results[:result_count], 1):
                        with st.expander(f"Result {i}: {search_result.get('metadata', {}).get('program_name', 'Unknown')}"):
                            metadata = search_result.get('metadata', {})
                            st.write(f"**Type:** {metadata.get('chunk_type', 'Unknown')}")
                            st.write(f"**Similarity:** {search_result.get('similarity_score', 0):.3f}")
                            st.code(search_result.get('content', '')[:500] + "...")
                            
                            if st.button(f"Analyze {metadata.get('program_name', 'Component')}", key=f"analyze_result_{i}"):
                                # Trigger analysis of this component
                                component_name = metadata.get('program_name', '')
                                if component_name:
                                    st.session_state.current_analysis = None  # Clear previous
                                    analyze_component_single_gpu(component_name, "auto-detect", f"Tell me about this component found in search for '{search_query}'", True)
                
                # Show suggestions
                suggestions = result.get("suggestions", [])
                if suggestions:
                    st.markdown("### 💡 Suggestions")
                    for suggestion in suggestions:
                        if st.button(suggestion, key=f"search_suggestion_{hash(suggestion)}"):
                            # Add to chat
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": suggestion,
                                "timestamp": dt.now().isoformat()
                            })
            else:
                st.error("Search failed or returned unexpected results")
                
        except Exception as e:
            st.error(f"Enhanced search failed: {str(e)}")


def export_chat_history():
    """Export chat history with enhanced formatting including GPU info"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
    
    # Create formatted export
    export_data = {
        "export_info": {
            "timestamp": dt.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "session_id": st.session_state.get("session_id", "unknown"),
            "coordinator_type": "single_gpu",
            "gpu_used": getattr(st.session_state.coordinator, 'selected_gpu', 'unknown') if st.session_state.coordinator else 'unknown'
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
                "gpu_used": message["content"].get("gpu_used", "unknown"),
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
        "📥 Download Single GPU Chat History",
        export_json,
        file_name=f"opulence_single_gpu_chat_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def generate_chat_summary():
    """Generate an intelligent summary of the chat conversation using single GPU"""
    if not st.session_state.chat_history:
        st.info("No conversation to summarize")
        return
    
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("🔄 Generating conversation summary on single GPU..."):
        try:
            # Create a summary query
            summary_query = "Please provide a summary of our conversation so far, highlighting the key points and findings."
            
            summary_result = safe_run_async(
                st.session_state.coordinator.process_chat_query(summary_query, st.session_state.chat_history)
            )
            
            summary = summary_result.get("response", "Unable to generate summary") if isinstance(summary_result, dict) else str(summary_result)
            gpu_used = summary_result.get("gpu_used", "unknown") if isinstance(summary_result, dict) else "unknown"
            
            # Display summary in a nice format
            st.markdown("### 📋 Conversation Summary")
            st.markdown(summary)
            st.caption(f"Generated on GPU {gpu_used}")
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": {
                    "response": f"**Conversation Summary:**\n\n{summary}",
                    "response_type": "summary",
                    "suggestions": ["Continue analysis", "Export summary", "Start new topic"],
                    "gpu_used": gpu_used
                },
                "timestamp": dt.now().isoformat()
            })
            
        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")


def generate_follow_up_suggestions():
    """Generate intelligent follow-up question suggestions using single GPU"""
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
    
    with st.spinner("🔮 Generating suggestions on single GPU..."):
        try:
            suggestion_query = f"Based on my question '{last_query}' and your response '{last_response[:200]}...', what are 3 good follow-up questions I should ask?"
            
            suggestion_result = safe_run_async(
                st.session_state.coordinator.process_chat_query(suggestion_query, [])
            )
            
            if isinstance(suggestion_result, dict):
                suggestions_text = suggestion_result.get("response", "")
                gpu_used = suggestion_result.get("gpu_used", "unknown")
                
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
                    st.caption(f"Generated on GPU {gpu_used}")
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


def display_enhanced_component_analysis(analysis: dict):
    """Display enhanced component analysis results with GPU information"""
    if isinstance(analysis, dict) and "error" in analysis:
        st.error(f"Analysis error: {analysis['error']}")
        return
    
    # Check if this is chat-enhanced
    if analysis.get("response_type") == "enhanced_analysis":
        component_name = analysis.get("component_name", "Unknown")
        gpu_used = analysis.get("gpu_used", "Unknown")
        st.success(f"✅ Enhanced analysis completed for: **{component_name}** on GPU {gpu_used}")
        
        # Show chat explanation first
        if analysis.get("chat_explanation"):
            st.markdown("### 🧠 AI Explanation")
            st.markdown(analysis["chat_explanation"])
        
        # Show suggestions
        suggestions = analysis.get("suggestions", [])
        if suggestions:
            st.markdown("### 💡 Suggested Actions")
            cols = st.columns(min(len(suggestions), 3))
            for i, suggestion in enumerate(suggestions[:3]):
                with cols[i]:
                    if st.button(suggestion, key=f"analysis_suggestion_{i}"):
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": f"For {component_name}: {suggestion}",
                            "timestamp": dt.now().isoformat()
                        })
        
        # Show detailed analysis in tabs
        tab1, tab2, tab3 = st.tabs(["📊 Technical Analysis", "💬 Chat View", "📋 Export"])
        
        with tab1:
            # Display the underlying technical analysis
            technical_analysis = analysis.get("analysis", {})
            if technical_analysis:
                display_component_analysis(technical_analysis)
        
        with tab2:
            # Show this analysis in chat format
            if st.button("💬 Add to Chat History"):
                st.session_state.chat_history.extend([
                    {
                        "role": "user",
                        "content": f"Analyze {component_name}",
                        "timestamp": dt.now().isoformat()
                    },
                    {
                        "role": "assistant",
                        "content": {
                            "response": analysis.get("chat_explanation", "Analysis completed."),
                            "response_type": "analysis",
                            "suggestions": suggestions,
                            "gpu_used": gpu_used
                        },
                        "timestamp": dt.now().isoformat()
                    }
                ])
                st.success("Added to chat history!")
        
        with tab3:
            # Export options
            if st.button("📄 Export Enhanced Analysis"):
                export_data = json.dumps(analysis, indent=2, default=str)
                st.download_button(
                    "Download Enhanced Analysis",
                    export_data,
                    file_name=f"enhanced_analysis_{component_name}_gpu{gpu_used}_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        # Regular analysis display
        display_component_analysis(analysis)


def display_component_analysis(analysis: dict):
    """Display component analysis results with better error handling and GPU info"""
    if isinstance(analysis, dict) and "error" in analysis:
        st.error(f"Analysis error: {analysis['error']}")
        
        # Show debug info if available
        if "debug_info" in analysis:
            with st.expander("🐛 Debug Information"):
                st.json(analysis["debug_info"])
        
        if "traceback" in analysis:
            with st.expander("🔍 Technical Details"):
                st.code(analysis["traceback"])
        return
    
    if not isinstance(analysis, dict):
        st.error(f"Invalid analysis result: {type(analysis)}")
        return
    
    component_name = analysis.get("component_name", "Unknown")
    component_type = analysis.get("component_type", "unknown")
    gpu_used = analysis.get("gpu_used", "Unknown")
    
    st.success(f"✅ Analysis completed for {component_type}: **{component_name}** on GPU {gpu_used}")
    
    # Show what data was found
    if "debug_info" in analysis:
        debug_info = analysis["debug_info"]
        st.info(f"📊 Found {debug_info.get('chunk_count', 0)} chunks in database")
    
    # Create tabs for different aspects of analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔄 Lineage", "📈 Impact", "📋 Report", "🐛 Debug"])
    
    with tab1:
        show_analysis_overview(analysis)
    
    with tab2:
        show_lineage_analysis(analysis)
    
    with tab3:
        show_impact_analysis(analysis)
    
    with tab4:
        show_analysis_report(analysis)
    
    with tab5:
        show_analysis_debug(analysis)


def show_analysis_overview(analysis: dict):
    """Fixed analysis overview to show actual data with GPU info"""
    st.markdown("### 📊 Component Overview")
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    
    component_type = analysis.get("component_type", "unknown")
    
    with col1:
        st.metric("Component Type", component_type.title())
    
    with col2:
        chunks_found = analysis.get("chunks_found", 0)
        st.metric("Chunks Found", chunks_found)
    
    with col3:
        processing_time = analysis.get("processing_time", 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col4:
        gpu_used = analysis.get("gpu_used", "Unknown")
        st.metric("GPU Used", gpu_used)
    
    # Show logic analysis results if available
    if "logic_analysis" in analysis:
        logic_data = analysis["logic_analysis"]
        if isinstance(logic_data, dict) and "error" not in logic_data:
            st.markdown("### 🧠 Logic Analysis Summary")
            
            # Show key metrics from logic analysis
            if "total_chunks" in logic_data:
                st.info(f"📊 Analyzed {logic_data['total_chunks']} code chunks")
            
            if "complexity_score" in logic_data:
                complexity = logic_data["complexity_score"]
                st.info(f"🎯 Average complexity score: {complexity:.2f}")
            
            if "metrics" in logic_data:
                metrics = logic_data["metrics"]
                if "total_patterns" in metrics:
                    st.info(f"🔍 Found {metrics['total_patterns']} logic patterns")
                
                if "maintainability_score" in metrics:
                    maintainability = metrics["maintainability_score"]
                    st.info(f"🔧 Maintainability score: {maintainability:.1f}/10")
            
            # Show business rules if available
            if "business_rules" in logic_data and logic_data["business_rules"]:
                st.markdown("#### 📋 Business Rules Found")
                rules_count = len(logic_data["business_rules"])
                st.write(f"Extracted {rules_count} business rules from the program")
                
                # Show first few rules
                for i, rule in enumerate(logic_data["business_rules"][:3], 1):
                    if isinstance(rule, dict):
                        rule_type = rule.get("rule_type", "unknown")
                        condition = rule.get("condition", "No condition")
                        st.write(f"**Rule {i}** ({rule_type}): {condition}")
            
            # Show recommendations if available
            if "recommendations" in logic_data and logic_data["recommendations"]:
                st.markdown("#### 💡 Recommendations")
                for rec in logic_data["recommendations"][:3]:
                    st.write(f"• {rec}")
    
    # Basic info section
    if "basic_info" in analysis and isinstance(analysis["basic_info"], dict):
        basic_info = analysis["basic_info"]
        
        if "chunk_summary" in basic_info and basic_info["chunk_summary"]:
            st.markdown("### 📋 Code Structure Breakdown")
            chunk_summary = basic_info["chunk_summary"]
            
            for chunk_type, count in chunk_summary.items():
                st.write(f"• **{chunk_type.replace('_', ' ').title()}**: {count} chunks")
        
        if "total_content_length" in basic_info:
            content_length = basic_info["total_content_length"]
            st.write(f"📄 **Total content**: {content_length:,} characters")


def show_lineage_analysis(analysis: dict):
    """Fixed lineage analysis display with GPU info"""
    st.markdown("### 📈 Data Lineage Analysis")
    
    if "lineage" not in analysis:
        st.info("💡 Lineage analysis is available for fields and data components. This is a program component.")
        st.write("**For program analysis, check the Logic tab for:**")
        st.write("• Code structure and complexity")
        st.write("• Business rules extraction") 
        st.write("• Logic patterns and recommendations")
        return
    
    lineage = analysis["lineage"]
    
    if isinstance(lineage, dict) and "error" not in lineage:
        # Show lineage summary
        if "usage_analysis" in lineage:
            usage_stats = lineage["usage_analysis"].get("statistics", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total References", usage_stats.get("total_references", 0))
            with col2:
                st.metric("Programs Using", len(usage_stats.get("programs_using", [])))
            with col3:
                st.metric("Operation Types", len(usage_stats.get("operation_types", {})))
        
        # Show lineage graph info
        if "lineage_graph" in lineage:
            graph = lineage["lineage_graph"]
            st.write(f"**Lineage Graph**: {graph.get('total_nodes', 0)} nodes, {graph.get('total_edges', 0)} relationships")
    else:
        st.warning("Lineage analysis not available or failed for this component")


def show_impact_analysis(analysis: dict):
    """Fixed impact analysis display with GPU info"""
    st.markdown("### 📈 Impact Analysis")
    
    # Check if we have logic analysis (which contains impact info for programs)
    if "logic_analysis" in analysis:
        logic_data = analysis["logic_analysis"]
        if isinstance(logic_data, dict) and "error" not in logic_data:
            
            # Show program impact metrics
            col1, col2 = st.columns(2)
            
            with col1:
                if "total_chunks" in logic_data:
                    st.metric("Code Chunks", logic_data["total_chunks"])
                if "metrics" in logic_data and "total_lines" in logic_data["metrics"]:
                    st.metric("Total Lines", logic_data["metrics"]["total_lines"])
            
            with col2:
                if "complexity_score" in logic_data:
                    complexity = logic_data["complexity_score"]
                    if complexity > 7:
                        risk_level = "HIGH"
                        risk_color = "🔴"
                    elif complexity > 4:
                        risk_level = "MEDIUM" 
                        risk_color = "🟡"
                    else:
                        risk_level = "LOW"
                        risk_color = "🟢"
                    st.metric("Change Risk", f"{risk_color} {risk_level}")
                
                if "metrics" in logic_data and "maintainability_score" in logic_data["metrics"]:
                    maintainability = logic_data["metrics"]["maintainability_score"]
                    st.metric("Maintainability", f"{maintainability:.1f}/10")
            
            # Impact recommendations
            st.markdown("#### 🎯 Change Impact Considerations")
            
            if "recommendations" in logic_data and logic_data["recommendations"]:
                st.write("**Before modifying this program:**")
                for rec in logic_data["recommendations"]:
                    st.write(f"• {rec}")
            else:
                st.write("• Review code complexity before making changes")
                st.write("• Test thoroughly due to business logic complexity")
                st.write("• Consider impact on dependent systems")
    
    elif "impact_analysis" in analysis:
        # Handle other component types
        impact = analysis["impact_analysis"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Affected Programs", len(impact.get("affected_programs", [])))
        
        with col2:
            st.metric("Risk Level", impact.get("risk_level", "Unknown"))
    
    else:
        st.info("💡 Impact analysis shows how changes to this component might affect the system.")
        st.write("For programs, this includes:")
        st.write("• Code complexity assessment")
        st.write("• Maintainability evaluation") 
        st.write("• Change risk factors")


def show_analysis_report(analysis: dict):
    """Fixed comprehensive analysis report with GPU info"""
    st.markdown("### 📋 Comprehensive Analysis Report")
    
    component_name = analysis.get("component_name", "Unknown")
    component_type = analysis.get("component_type", "unknown")
    gpu_used = analysis.get("gpu_used", "Unknown")
    
    # Generate report based on available analysis
    report_sections = []
    
    # Executive Summary
    report_sections.append("## Executive Summary")
    report_sections.append(f"**Component**: {component_name}")
    report_sections.append(f"**Type**: {component_type.title()}")
    report_sections.append(f"**Analysis Status**: {analysis.get('status', 'unknown').title()}")
    report_sections.append(f"**GPU Used**: {gpu_used}")
    
    if analysis.get("chunks_found", 0) > 0:
        report_sections.append(f"**Code Size**: {analysis['chunks_found']} chunks analyzed")
    
    # Logic Analysis Section
    if "logic_analysis" in analysis:
        logic_data = analysis["logic_analysis"]
        if isinstance(logic_data, dict) and "error" not in logic_data:
            report_sections.append("\n## Program Analysis")
            
            if "complexity_score" in logic_data:
                complexity = logic_data["complexity_score"]
                report_sections.append(f"**Average Complexity**: {complexity:.2f}")
            
            if "metrics" in logic_data:
                metrics = logic_data["metrics"]
                if "maintainability_score" in metrics:
                    maintainability = metrics["maintainability_score"]
                    report_sections.append(f"**Maintainability Score**: {maintainability:.1f}/10")
                
                if "total_patterns" in metrics:
                    report_sections.append(f"**Logic Patterns Found**: {metrics['total_patterns']}")
            
            # Business Rules
            if "business_rules" in logic_data and logic_data["business_rules"]:
                report_sections.append(f"\n**Business Rules**: {len(logic_data['business_rules'])} rules extracted")
            
            # Recommendations
            if "recommendations" in logic_data and logic_data["recommendations"]:
                report_sections.append("\n## Recommendations")
                for rec in logic_data["recommendations"]:
                    report_sections.append(f"• {rec}")
        
        if "basic_info" in analysis:
            basic_info = analysis["basic_info"]
            if isinstance(basic_info, dict) and "chunk_summary" in basic_info:
                report_sections.append("\n## Technical Structure")
                for chunk_type, count in basic_info["chunk_summary"].items():
                    report_sections.append(f"• **{chunk_type.replace('_', ' ').title()}**: {count} chunks")
    
    # Processing Information
    if "processing_time" in analysis:
        processing_time = analysis["processing_time"]
        report_sections.append(f"\n**Analysis completed in**: {processing_time:.2f} seconds on GPU {gpu_used}")
    
    # Display the report
    report_content = "\n".join(report_sections)
    st.markdown(report_content)
    
    # Download button
    if report_content:
        st.download_button(
            "📄 Download Report",
            report_content,
            file_name=f"opulence_analysis_{component_name}_gpu{gpu_used}_{dt.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    # Show comprehensive report if available
    if "comprehensive_report" in analysis:
        st.markdown("### 📊 Detailed Analysis Report")
        st.markdown(analysis["comprehensive_report"])
    elif "lineage" in analysis and isinstance(analysis["lineage"], dict) and "comprehensive_report" in analysis["lineage"]:
        st.markdown("### 📊 Detailed Lineage Report")
        st.markdown(analysis["lineage"]["comprehensive_report"])


def show_analysis_debug(analysis: dict):
    """Show debug information for analysis with GPU info"""
    st.markdown("### 🐛 Debug Information")
    
    # Show raw analysis data
    with st.expander("Raw Analysis Data"):
        st.json(analysis)
    
    # Show what analyses were attempted
    attempted_analyses = []
    if "lineage" in analysis:
        attempted_analyses.append(("Lineage Analysis", analysis["lineage"]))
    if "logic_analysis" in analysis:
        attempted_analyses.append(("Logic Analysis", analysis["logic_analysis"]))
    if "jcl_analysis" in analysis:
        attempted_analyses.append(("JCL Analysis", analysis["jcl_analysis"]))
    if "semantic_search" in analysis:
        attempted_analyses.append(("Semantic Search", analysis["semantic_search"]))
    if "basic_info" in analysis:
        attempted_analyses.append(("Basic Info", analysis["basic_info"]))
    
    for name, result in attempted_analyses:
        with st.expander(f"{name} Results"):
            if isinstance(result, dict):
                if "error" in result:
                    st.error(f"❌ {name} failed: {result['error']}")
                else:
                    st.success(f"✅ {name} succeeded")
                    st.json(result)
            else:
                st.write(result)
    
    # Show GPU information
    gpu_used = analysis.get("gpu_used", "Unknown")
    st.info(f"🎯 Analysis performed on GPU {gpu_used}")


def show_example_queries():
    """Show example queries in sidebar with GPU info"""
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
    """Show footer with system information including GPU details"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🧠 Opulence Single GPU Research Agent**")
        st.markdown("Powered by vLLM, FAISS, and ChromaDB")
        if st.session_state.coordinator:
            gpu_id = getattr(st.session_state.coordinator, 'selected_gpu', 'Unknown')
            st.markdown(f"Running on GPU {gpu_id}")
    
    with col2:
        st.markdown("**📊 Current Session**")
        st.markdown(f"Chat Messages: {len(st.session_state.chat_history)}")
        st.markdown(f"Files Processed: {len(st.session_state.processing_history)}")
    
    with col3:
        st.markdown("**🕐 System Time**")
        st.markdown(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main application function for single GPU system"""
    
    # Header
    st.markdown('<div class="main-header">🧠 Opulence Single GPU Deep Research Agent</div>', unsafe_allow_html=True)
    
    # Show import status
    if not COORDINATOR_AVAILABLE:
        st.error("⚠️ Single GPU Coordinator module not available - Running in demo mode")
        if st.button("🐛 Toggle Debug Mode"):
            st.session_state.debug_mode = not st.session_state.debug_mode
            st.rerun()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1e3a8a/ffffff?text=OPULENCE", use_container_width=True)
        
        page = st.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📂 File Upload", "💬 Enhanced Chat", "🔍 Enhanced Analysis", 
             "🔍 Enhanced Search", "📊 Field Lineage", "🔄 DB2 Comparison", "📋 Documentation", "⚙️ System Health"]
        )
        
        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh System"):
            st.rerun()
        
        # System status indicator with GPU info
        if COORDINATOR_AVAILABLE and st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                gpu_id = health.get('selected_gpu', 'Unknown')
                st.success(f"🟢 System Healthy (GPU {gpu_id})")
            except:
                st.success("🟢 System Healthy")
        elif COORDINATOR_AVAILABLE:
            st.warning("🟡 System Not Initialized")
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
        elif page == "🔍 Enhanced Search":
            show_enhanced_search()
        elif page == "📊 Field Lineage":
            show_field_lineage()
        elif page == "🔄 DB2 Comparison":
            show_db2_comparison()
        elif page == "📋 Documentation":
            show_documentation()
        elif page == "⚙️ System Health":
            show_system_health()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.exception(e)
    
    # Show footer
    show_footer()


# Additional functions that weren't updated but are still needed
def show_field_lineage():
    """Show field lineage analysis interface for single GPU"""
    st.markdown('<div class="sub-header">📊 Field Lineage Analysis (Single GPU)</div>', unsafe_allow_html=True)
    
    # Show GPU status
    show_gpu_status()
    
    # Field selection
    field_name = st.text_input("Field Name to Trace:")
    
    if st.button("🔍 Trace Field Lineage on Single GPU") and field_name:
        trace_field_lineage(field_name)
    
    # Display lineage results
    if st.session_state.field_lineage_result:
        display_field_lineage_results(st.session_state.field_lineage_result)


def trace_field_lineage(field_name: str):
    """Trace field lineage using single GPU"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"Tracing lineage for {field_name} on single GPU..."):
        try:
            lineage_agent = st.session_state.coordinator.get_agent("lineage_analyzer")
            result = safe_run_async(lineage_agent.analyze_field_lineage(field_name))
            st.session_state.field_lineage_result = result
            
        except Exception as e:
            st.error(f"Lineage tracing failed: {str(e)}")


def display_field_lineage_results(result: dict):
    """Display field lineage results with GPU info"""
    if isinstance(result, dict) and "error" in result:
        st.error(f"Lineage analysis error: {result['error']}")
        return
    
    if not isinstance(result, dict):
        st.error(f"Invalid lineage result: {type(result)}")
        return
    
    field_name = result.get("field_name", "Unknown")
    gpu_used = result.get("gpu_used", "Unknown")
    st.success(f"✅ Lineage analysis completed for field: **{field_name}** on GPU {gpu_used}")
    
    # Usage statistics
    if "usage_analysis" in result:
        usage = result["usage_analysis"].get("statistics", {})
        
        st.markdown("### Usage Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total References", usage.get("total_references", 0))
        
        with col2:
            st.metric("Programs Using", len(usage.get("programs_using", [])))
        
        with col3:
            st.metric("Operation Types", len(usage.get("operation_types", {})))


def show_db2_comparison():
    """Show DB2 comparison interface for single GPU"""
    st.markdown('<div class="sub-header">🔄 DB2 Data Comparison (Single GPU)</div>', unsafe_allow_html=True)
    
    # Show GPU status
    show_gpu_status()
    
    st.info("DB2 Comparison feature will compare data between DB2 tables and loaded SQLite data (limited to 10K rows)")
    
    # Component selection for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### SQLite Component")
        sqlite_component = st.text_input("SQLite Table/File Name:")
    
    with col2:
        st.markdown("#### DB2 Component")
        db2_component = st.text_input("DB2 Table Name:")
    
    if st.button("🔄 Compare Data on Single GPU") and sqlite_component and db2_component:
        compare_data_sources(sqlite_component, db2_component)


def compare_data_sources(sqlite_comp: str, db2_comp: str):
    """Compare data between SQLite and DB2 using single GPU"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Comparing data sources on single GPU..."):
        try:
            db2_agent = st.session_state.coordinator.get_agent("db2_comparator")
            result = safe_run_async(db2_agent.compare_data(db2_comp))
            display_comparison_results(result)
            
        except Exception as e:
            st.error(f"Comparison failed: {str(e)}")


def display_comparison_results(result: dict):
    """Display data comparison results with GPU info"""
    if isinstance(result, dict) and "error" in result:
        st.error(f"Comparison error: {result['error']}")
        return
    
    if not isinstance(result, dict):
        st.error(f"Invalid comparison result: {type(result)}")
        return
    
    gpu_used = result.get("gpu_used", "Unknown")
    st.success(f"✅ Data comparison completed on GPU {gpu_used}")
    
    # Summary metrics
    if "data_comparison" in result:
        data_comp = result["data_comparison"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SQLite Records", data_comp.get("sqlite_rows", 0))
        
        with col2:
            st.metric("DB2 Records", data_comp.get("db2_rows", 0))
        
        with col3:
            match_rate = data_comp.get("comparison_score", 0) * 100
            st.metric("Match Rate", f"{match_rate:.1f}%")
        
        with col4:
            st.metric("Differences Found", data_comp.get("mismatched_rows", 0))


def show_documentation():
    """Show documentation generation interface for single GPU"""
    st.markdown('<div class="sub-header">📋 Documentation Generation (Single GPU)</div>', unsafe_allow_html=True)
    
    # Show GPU status
    show_gpu_status()
    
    # Documentation type selection
    doc_type = st.selectbox(
        "Documentation Type",
        ["Field Lineage Report", "Component Analysis Report", "System Overview"]
    )
    
    # Component selection for documentation
    component_name = st.text_input("Component Name (for component-specific docs):")
    
    # Additional options
    include_diagrams = st.checkbox("Include Data Flow Diagrams", value=True)
    include_sample_data = st.checkbox("Include Sample Data", value=False)
    
    if st.button("📋 Generate Documentation on Single GPU"):
        generate_documentation(doc_type, component_name, include_diagrams, include_sample_data)


def generate_documentation(doc_type: str, component_name: str, include_diagrams: bool, include_sample_data: bool):
    """Generate documentation using single GPU"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Generating documentation on single GPU..."):
        try:
            doc_agent = st.session_state.coordinator.get_agent("documentation")
            if component_name:
                result = safe_run_async(doc_agent.generate_program_documentation(component_name))
            else:
                # Generate system overview
                result = safe_run_async(doc_agent.generate_system_documentation(["SYSTEM_OVERVIEW"]))
            
            display_generated_documentation(result)
            
        except Exception as e:
            st.error(f"Documentation generation failed: {str(e)}")


def display_generated_documentation(result: dict):
    """Display generated documentation with GPU info"""
    if isinstance(result, dict) and "error" in result:
        st.error(f"Documentation error: {result['error']}")
        return
    
    if not isinstance(result, dict):
        st.error(f"Invalid documentation result: {type(result)}")
        return
    
    gpu_used = result.get("gpu_used", "Unknown")
    st.success(f"✅ Documentation generated successfully on GPU {gpu_used}")
    
    # Display documentation content
    if "documentation" in result:
        st.markdown("### Generated Documentation")
        st.markdown(result["documentation"])
        
        # Download button
        st.download_button(
            "📄 Download Documentation",
            result["documentation"],
            file_name=f"opulence_documentation_gpu{gpu_used}_{dt.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


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
            "coordinator_type": "single_gpu"
        })