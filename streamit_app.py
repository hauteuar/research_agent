# streamlit_app.py
"""
Opulence Deep Research Mainframe Agent - Streamlit UI
Main interface for the Opulence system
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our coordinator and agents
from opulence_coordinator import get_coordinator, initialize_system
import sys
import os

# Set page config
st.set_page_config(
    page_title="Opulence - Deep Research Mainframe Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def show_db2_comparison():
    """Show DB2 comparison interface"""
    st.markdown('<div class="sub-header">🔄 DB2 Data Comparison</div>', unsafe_allow_html=True)
    
    st.info("DB2 Comparison feature will compare data between DB2 tables and loaded SQLite data (limited to 10K rows)")
    
    # Component selection for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### SQLite Component")
        sqlite_component = st.text_input("SQLite Table/File Name:")
    
    with col2:
        st.markdown("#### DB2 Component")
        db2_component = st.text_input("DB2 Table Name:")
    
    if st.button("🔄 Compare Data") and sqlite_component and db2_component:
        compare_data_sources(sqlite_component, db2_component)

def compare_data_sources(sqlite_comp: str, db2_comp: str):
    """Compare data between SQLite and DB2"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Comparing data sources..."):
        try:
            result = asyncio.run(
                st.session_state.coordinator.agents["db2_comparator"].compare_data(db2_comp)
            )
            
            display_comparison_results(result)
            
        except Exception as e:
            st.error(f"Comparison failed: {str(e)}")

def display_comparison_results(result: dict):
    """Display data comparison results"""
    if "error" in result:
        st.error(f"Comparison error: {result['error']}")
        return
    
    st.success("✅ Data comparison completed")
    
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
    """Show documentation generation interface"""
    st.markdown('<div class="sub-header">📋 Documentation Generation</div>', unsafe_allow_html=True)
    
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
    
    if st.button("📋 Generate Documentation"):
        generate_documentation(doc_type, component_name, include_diagrams, include_sample_data)

def generate_documentation(doc_type: str, component_name: str, include_diagrams: bool, include_sample_data: bool):
    """Generate documentation"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Generating documentation..."):
        try:
            if component_name:
                result = asyncio.run(
                    st.session_state.coordinator.agents["documentation"].generate_program_documentation(component_name)
                )
            else:
                # Generate system overview
                result = asyncio.run(
                    st.session_state.coordinator.agents["documentation"].generate_system_documentation(["SYSTEM_OVERVIEW"])
                )
            
            display_generated_documentation(result)
            
        except Exception as e:
            st.error(f"Documentation generation failed: {str(e)}")

def display_generated_documentation(result: dict):
    """Display generated documentation"""
    if "error" in result:
        st.error(f"Documentation error: {result['error']}")
        return
    
    st.success("✅ Documentation generated successfully")
    
    # Display documentation content
    if "documentation" in result:
        st.markdown("### Generated Documentation")
        st.markdown(result["documentation"])
        
        # Download button
        st.download_button(
            "📄 Download Documentation",
            result["documentation"],
            file_name=f"opulence_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def show_system_health():
    """Show system health and statistics"""
    st.markdown('<div class="sub-header">⚙️ System Health & Statistics</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    # Get health status
    health = st.session_state.coordinator.get_health_status()
    stats = st.session_state.coordinator.get_statistics()
    
    # Overall health indicator
    if health["status"] == "healthy":
        st.success("🟢 System Status: Healthy")
    else:
        st.error("🔴 System Status: Issues Detected")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Agents", health.get("active_agents", 0))
    
    with col2:
        memory_usage = health.get("memory_usage", {})
        if "virtual" in memory_usage:
            st.metric("Memory Usage", f"{memory_usage['virtual'].get('percent', 0):.1f}%")
        else:
            st.metric("Memory Usage", "N/A")
    
    with col3:
        gpu_status = health.get("gpu_status", {})
        st.metric("GPUs Available", len(gpu_status))
    
    with col4:
        cache_stats = health.get("cache_stats", {})
        st.metric("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
    
    # GPU utilization
    if health.get("gpu_status"):
        st.markdown("### GPU Utilization")
        gpu_data = []
        for gpu_id, gpu_info in health["gpu_status"].items():
            gpu_data.append({
                "GPU": gpu_id,
                "Utilization": gpu_info.get("utilization", 0),
                "Memory": gpu_info.get("memory_used_gb", 0),
                "Temperature": gpu_info.get("temperature", 0)
            })
        
        if gpu_data:
            df_gpu = pd.DataFrame(gpu_data)
            fig = px.bar(df_gpu, x="GPU", y="Utilization", title="GPU Utilization %")
            st.plotly_chart(fig, use_container_width=True)
    
    # Processing statistics
    if stats.get("processing_stats"):
        st.markdown("### Processing Performance")
        df_perf = pd.DataFrame(stats["processing_stats"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_duration = px.bar(df_perf, x="operation", y="avg_duration", 
                                title="Average Processing Time by Operation")
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col2:
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
        if st.button("🗑️ Clear Cache"):
            st.info("Cache cleared")
    
    with col3:
        if st.button("📊 Rebuild Indices"):
            rebuild_indices()
    
    with col4:
        if st.button("📥 Export Logs"):
            export_system_logs()

def rebuild_indices():
    """Rebuild vector indices"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Rebuilding indices..."):
        try:
            result = asyncio.run(
                st.session_state.coordinator.agents["vector_index"].rebuild_index()
            )
            
            if result["status"] == "success":
                st.success("✅ Indices rebuilt successfully")
            else:
                st.error(f"❌ Index rebuild failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Index rebuild failed: {str(e)}")

def export_system_logs():
    """Export system logs"""
    try:
        # Read log file
        log_file = Path("opulence.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            st.download_button(
                "📥 Download System Logs",
                log_content,
                file_name=f"opulence_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain"
            )
        else:
            st.warning("No log file found")
            
    except Exception as e:
        st.error(f"Failed to export logs: {str(e)}")

def show_example_queries():
    """Show example queries in sidebar"""
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
                "timestamp": datetime.now().isoformat()
            })
            # Process the example query
            response = process_chat_query(example)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat()
            })

def show_footer():
    """Show footer with system information"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🧠 Opulence Deep Research Agent**")
        st.markdown("Powered by vLLM, FAISS, and ChromaDB")
    
    with col2:
        st.markdown("**📊 Current Session**")
        st.markdown(f"Chat Messages: {len(st.session_state.chat_history)}")
        st.markdown(f"Files Processed: {len(st.session_state.processing_history)}")
    
    with col3:
        st.markdown("**🕐 System Time**")
        st.markdown(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Helper function to run async functions in Streamlit
def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Error handling decorator
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    return wrapper

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator for performance monitoring"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log performance if it takes more than 1 second
        if end_time - start_time > 1.0:
            st.info(f"⏱️ Operation completed in {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper

# Add loading spinner for async operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_statistics():
    """Get cached statistics to improve performance"""
    if st.session_state.coordinator:
        return st.session_state.coordinator.get_statistics()
    return {}

# Main application entry point
if __name__ == "__main__":
    main() initialize_session_state():
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

# Call initialization
initialize_session_state()

async def init_coordinator():
    """Initialize the coordinator asynchronously"""
    if st.session_state.coordinator is None:
        try:
            st.session_state.coordinator = await initialize_system()
            st.session_state.initialization_status = "completed"
            return True
        except Exception as e:
            st.session_state.initialization_status = f"error: {str(e)}"
            return False
    return True

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🧠 Opulence Deep Research Mainframe Agent</div>', unsafe_allow_html=True)
    
    # Initialize system
    if st.session_state.initialization_status == "pending":
        with st.spinner("Initializing Opulence system..."):
            success = asyncio.run(init_coordinator())
            if not success:
                st.error(f"Failed to initialize system: {st.session_state.initialization_status}")
                return
    
    # Check if initialization failed
    if st.session_state.initialization_status.startswith("error"):
        st.error(f"System initialization failed: {st.session_state.initialization_status}")
        if st.button("Retry Initialization"):
            st.session_state.initialization_status = "pending"
            st.rerun()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1e3a8a/ffffff?text=OPULENCE", use_column_width=True)
        
        page = st.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📂 File Upload", "💬 Chat Analysis", "🔍 Component Analysis", 
             "📊 Field Lineage", "🔄 DB2 Comparison", "📋 Documentation", "⚙️ System Health"]
        )
        
        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh System"):
            st.rerun()
        
        if st.button("📈 Update Statistics"):
            if st.session_state.coordinator:
                stats = st.session_state.coordinator.get_statistics()
                st.session_state.system_stats = stats
        
        # System status indicator
        if st.session_state.coordinator:
            health = st.session_state.coordinator.get_health_status()
            if health["status"] == "healthy":
                st.success("🟢 System Healthy")
            else:
                st.warning("🟡 System Issues")
        
        # Show example queries
        show_example_queries()
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📂 File Upload":
        show_file_upload()
    elif page == "💬 Chat Analysis":
        show_chat_analysis()
    elif page == "🔍 Component Analysis":
        show_component_analysis()
    elif page == "📊 Field Lineage":
        show_field_lineage()
    elif page == "🔄 DB2 Comparison":
        show_db2_comparison()
    elif page == "📋 Documentation":
        show_documentation()
    elif page == "⚙️ System Health":
        show_system_health()
    
    # Show footer
    show_footer()

def show_dashboard():
    """Show main dashboard"""
    st.markdown('<div class="sub-header">System Overview</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.warning("System not initialized")
        return
    
    # Get system statistics
    stats = st.session_state.coordinator.get_statistics()
    health = st.session_state.coordinator.get_health_status()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", stats["system_stats"]["total_files_processed"])
    
    with col2:
        st.metric("Total Queries", stats["system_stats"]["total_queries"])
    
    with col3:
        avg_time = stats["system_stats"]["avg_response_time"]
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    with col4:
        cache_rate = stats["system_stats"]["cache_hit_rate"]
        st.metric("Cache Hit Rate", f"{cache_rate:.1%}")
    
    # Processing statistics chart
    if stats["processing_stats"]:
        st.markdown("### Processing Statistics")
        df_stats = pd.DataFrame(stats["processing_stats"])
        
        fig = px.bar(df_stats, x="operation", y="avg_duration", 
                    title="Average Processing Time by Operation")
        st.plotly_chart(fig, use_container_width=True)
    
    # File statistics
    if stats["file_stats"]:
        st.markdown("### File Processing Status")
        df_files = pd.DataFrame(stats["file_stats"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(df_files, values="count", names="file_type", 
                           title="Files by Type")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_status = px.bar(df_files, x="processing_status", y="count", 
                              color="file_type", title="Processing Status")
            st.plotly_chart(fig_status, use_container_width=True)
    
    # Recent activity
    st.markdown("### Recent Activity")
    if st.session_state.processing_history:
        for activity in st.session_state.processing_history[-5:]:
            st.info(f"🕐 {activity['timestamp']}: Processed {activity['files_count']} files")
    else:
        st.info("No recent activities to display")

def show_file_upload():
    """Show file upload interface"""
    st.markdown('<div class="sub-header">File Upload & Processing</div>', unsafe_allow_html=True)
    
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
            
            if st.button("Process Files"):
                process_uploaded_files(uploaded_files)
    
    elif upload_type == "Batch Upload (ZIP)":
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing multiple files",
            type=['zip']
        )
        
        if uploaded_zip:
            if st.button("Extract and Process ZIP"):
                process_zip_file(uploaded_zip)
    
    # Processing history
    st.markdown("### Processing History")
    if st.session_state.processing_history:
        df_history = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No processing history available")

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
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
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Process files in batch
        status_text.text("Processing files with Opulence...")
        result = asyncio.run(
            st.session_state.coordinator.process_batch_files(file_paths)
        )
        
        # Display results
        with results_container:
            if result["status"] == "success":
                st.success(f"✅ Successfully processed {result['files_processed']} files in {result['processing_time']:.2f} seconds")
                
                # Show detailed results
                for i, file_result in enumerate(result["results"]):
                    if isinstance(file_result, dict):
                        with st.expander(f"📄 {uploaded_files[i].name}"):
                            st.json(file_result)
            else:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Update processing history
        st.session_state.processing_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_count": len(uploaded_files),
            "status": result["status"],
            "processing_time": result.get("processing_time", 0)
        })
        
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()
        temp_dir.rmdir()
        
        progress_bar.empty()
        status_text.empty()

def process_zip_file(uploaded_zip):
    """Process uploaded ZIP file"""
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
        
        st.info(f"Found {len(file_paths)} files to process")
        
        # Process files
        result = asyncio.run(
            st.session_state.coordinator.process_batch_files(file_paths)
        )
        
        if result["status"] == "success":
            st.success(f"✅ Successfully processed {result['files_processed']} files")
        else:
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

def show_chat_analysis():
    """Show chat-based analysis interface"""
    st.markdown('<div class="sub-header">💬 Chat with Opulence</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your mainframe systems...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Process query and generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = process_chat_query(user_input)
                st.write(response)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
    
    # Chat controls
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("Export Chat"):
            chat_export = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "Download Chat History",
                chat_export,
                file_name=f"opulence_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def process_chat_query(query: str) -> str:
    """Process chat query and return response"""
    if not st.session_state.coordinator:
        return "❌ System not initialized. Please check system health."
    
    try:
        # Determine query type and route to appropriate agent
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['lifecycle', 'lineage', 'trace', 'impact']):
            # Extract component name from query
            component_name = extract_component_name(query)
            if component_name:
                result = asyncio.run(
                    st.session_state.coordinator.analyze_component(component_name)
                )
                return format_analysis_response(result)
            else:
                return "Could you please specify which component (file, table, program, or field) you'd like me to analyze?"
        
        elif any(word in query_lower for word in ['compare', 'difference', 'db2']):
            return "For data comparison, please use the DB2 Comparison tab to select specific components."
        
        elif any(word in query_lower for word in ['search', 'find', 'pattern']):
            # Use semantic search
            if st.session_state.coordinator.agents.get("vector_index"):
                results = asyncio.run(
                    st.session_state.coordinator.agents["vector_index"].search_code_by_pattern(query)
                )
                return format_search_results(results)
            else:
                return "Search functionality is not available. Please check if files have been processed."
        
        else:
            # General query - try to provide helpful guidance
            return generate_general_response(query)
    
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def extract_component_name(query: str) -> str:
    """Extract component name from natural language query"""
    words = query.split()
    
    # Look for patterns like "analyze COMPONENT_NAME" or "trace FIELD_NAME"
    trigger_words = ['analyze', 'trace', 'lifecycle', 'lineage', 'impact', 'of', 'for']
    
    for i, word in enumerate(words):
        if word.lower() in trigger_words and i + 1 < len(words):
            potential_component = words[i + 1].strip('.,!?')
            if len(potential_component) > 2:  # Basic validation
                return potential_component
    
    # Look for uppercase words (likely component names)
    for word in words:
        if word.isupper() and len(word) > 2:
            return word
    
    return ""

def format_analysis_response(result: dict) -> str:
    """Format analysis result for chat display"""
    if "error" in result:
        return f"❌ Analysis failed: {result['error']}"
    
    component_name = result.get("component_name", "Unknown")
    component_type = result.get("component_type", "unknown")
    
    response = f"## 📊 Analysis of {component_type.title()}: {component_name}\n\n"
    
    if "lineage" in result:
        lineage = result["lineage"]
        usage_stats = lineage.get("usage_analysis", {}).get("statistics", {})
        response += f"**Usage Summary:**\n"
        response += f"- Total references: {usage_stats.get('total_references', 0)}\n"
        response += f"- Programs using: {len(usage_stats.get('programs_using', []))}\n\n"
    
    response += "💡 For detailed analysis, please check the Component Analysis tab."
    
    return response

def format_search_results(results: list) -> str:
    """Format search results for chat display"""
    if not results:
        return "🔍 No matching code patterns found. Try refining your search terms."
    
    response = f"## 🔍 Found {len(results)} Matching Code Patterns\n\n"
    
    for i, result in enumerate(results[:3], 1):  # Show top 3 results
        metadata = result.get("metadata", {})
        response += f"**{i}. {metadata.get('program_name', 'Unknown Program')}**\n"
        response += f"- Type: {metadata.get('chunk_type', 'code')}\n"
        response += f"- Similarity: {result.get('similarity_score', 0):.2f}\n"
        response += f"- Content preview: {result.get('content', '')[:100]}...\n\n"
    
    if len(results) > 3:
        response += f"💡 And {len(results) - 3} more results. Use the Component Analysis tab for detailed exploration."
    
    return response

def generate_general_response(query: str) -> str:
    """Generate general helpful response"""
    return f"""
I'm Opulence, your deep research mainframe agent! I can help you with:

🔍 **Component Analysis**: Analyze the lifecycle and usage of files, tables, programs, or fields
📊 **Field Lineage**: Trace data flow and transformations for specific fields  
🔄 **DB2 Comparison**: Compare data between DB2 tables and loaded files
📋 **Documentation**: Generate technical documentation and reports

**Examples of what you can ask:**
- "Analyze the lifecycle of TRADE_DATE field"
- "Trace the lineage of TRANSACTION_FILE"
- "Find programs that use security settlement logic"
- "Show me the impact of changing ACCOUNT_ID"

Your query: "{query}"

Would you like me to help you with any specific analysis?
    """

def show_component_analysis():
    """Show component analysis interface"""
    st.markdown('<div class="sub-header">🔍 Component Analysis</div>', unsafe_allow_html=True)
    
    # Component selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        component_name = st.text_input("Component Name (file, table, program, field):")
    
    with col2:
        component_type = st.selectbox(
            "Component Type",
            ["auto-detect", "file", "table", "program", "jcl", "field"]
        )
    
    if st.button("🔍 Analyze Component") and component_name:
        analyze_component(component_name, component_type)
    
    # Display current analysis
    if st.session_state.current_analysis:
        display_component_analysis(st.session_state.current_analysis)

def analyze_component(component_name: str, component_type: str):
    """Analyze a specific component"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"Analyzing {component_name}..."):
        try:
            result = asyncio.run(
                st.session_state.coordinator.analyze_component(
                    component_name, 
                    None if component_type == "auto-detect" else component_type
                )
            )
            st.session_state.current_analysis = result
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def display_component_analysis(analysis: dict):
    """Display component analysis results"""
    if "error" in analysis:
        st.error(f"Analysis error: {analysis['error']}")
        return
    
    component_name = analysis.get("component_name", "Unknown")
    component_type = analysis.get("component_type", "unknown")
    
    st.success(f"✅ Analysis completed for {component_type}: **{component_name}**")
    
    # Create tabs for different aspects of analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔄 Lineage", "📈 Impact", "📋 Report"])
    
    with tab1:
        show_analysis_overview(analysis)
    
    with tab2:
        show_lineage_analysis(analysis)
    
    with tab3:
        show_impact_analysis(analysis)
    
    with tab4:
        show_analysis_report(analysis)

def show_analysis_overview(analysis: dict):
    """Show analysis overview"""
    st.markdown("### Component Overview")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    component_type = analysis.get("component_type", "unknown")
    
    with col1:
        st.metric("Component Type", component_type.title())
    
    if "lineage" in analysis:
        lineage = analysis["lineage"]
        usage_stats = lineage.get("usage_analysis", {}).get("statistics", {})
        
        with col2:
            st.metric("Total References", usage_stats.get("total_references", 0))
        
        with col3:
            st.metric("Programs Using", len(usage_stats.get("programs_using", [])))

def show_lineage_analysis(analysis: dict):
    """Show lineage analysis"""
    st.markdown("### Data Lineage Analysis")
    
    if "lineage" not in analysis:
        st.info("Lineage analysis not available for this component type")
        return
    
    lineage = analysis["lineage"]
    
    # Lineage graph visualization
    if "lineage_graph" in lineage:
        graph = lineage["lineage_graph"]
        st.markdown(f"**Lineage Graph:** {len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} relationships")
        st.info("Interactive lineage graph visualization available!")

def show_impact_analysis(analysis: dict):
    """Show impact analysis"""
    st.markdown("### Impact Analysis")
    
    if "impact_analysis" in analysis:
        impact = analysis["impact_analysis"]
        
        # Impact metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Affected Programs", len(impact.get("affected_programs", [])))
        
        with col2:
            st.metric("Change Complexity", impact.get("change_complexity", "Unknown"))
    
    else:
        st.info("Impact analysis not available for this component")

def show_analysis_report(analysis: dict):
    """Show comprehensive analysis report"""
    st.markdown("### Comprehensive Analysis Report")
    
    if "comprehensive_report" in analysis:
        st.markdown(analysis["comprehensive_report"])
    elif "lineage" in analysis and "comprehensive_report" in analysis["lineage"]:
        st.markdown(analysis["lineage"]["comprehensive_report"])
    else:
        st.info("Comprehensive report not available")
    
    # Download report button
    report_content = analysis.get("comprehensive_report") or analysis.get("lineage", {}).get("comprehensive_report")
    
    if report_content:
        st.download_button(
            "📄 Download Report",
            report_content,
            file_name=f"opulence_analysis_{analysis.get('component_name', 'component')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def show_field_lineage():
    """Show field lineage analysis interface"""
    st.markdown('<div class="sub-header">📊 Field Lineage Analysis</div>', unsafe_allow_html=True)
    
    # Field selection
    field_name = st.text_input("Field Name to Trace:")
    
    if st.button("🔍 Trace Field Lineage") and field_name:
        trace_field_lineage(field_name)
    
    # Display lineage results
    if st.session_state.field_lineage_result:
        display_field_lineage_results(st.session_state.field_lineage_result)

def trace_field_lineage(field_name: str):
    """Trace field lineage"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"Tracing lineage for {field_name}..."):
        try:
            result = asyncio.run(
                st.session_state.coordinator.agents["lineage_analyzer"].analyze_field_lineage(field_name)
            )
            st.session_state.field_lineage_result = result
            
        except Exception as e:
            st.error(f"Lineage tracing failed: {str(e)}")

def display_field_lineage_results(result: dict):
    """Display field lineage results"""
    if "error" in result:
        st.error(f"Lineage analysis error: {result['error']}")
        return
    
    field_name = result.get("field_name", "Unknown")
    st.success(f"✅ Lineage analysis completed for field: **{field_name}**")
    
    # Usage statistics
    if "usage_analysis" in result:
        usage = result["usage_analysis"]["statistics"]
        
        st.markdown("### Usage Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total References", usage.get("total_references", 0))
        
        with col2:
            st.metric("Programs Using", len(usage.get("programs_using", [])))
        
        with col3:
            st.metric("Operation Types", len(usage.get("operation_types", {})))

