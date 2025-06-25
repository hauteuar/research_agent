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
import traceback
import sys
import os

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Opulence - Deep Research Mainframe Agent",
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

# Try to import coordinator with error handling
try:
    from opulence_coordinator import get_coordinator, initialize_system
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


def show_debug_info():
    """Show debug information"""
    if st.session_state.debug_mode:
        st.sidebar.markdown("### 🐛 Debug Info")
        st.sidebar.json({
            "coordinator_available": COORDINATOR_AVAILABLE,
            "coordinator": st.session_state.coordinator is not None,
            "init_status": st.session_state.initialization_status,
            "import_error": st.session_state.get('import_error', 'None'),
            "session_keys": list(st.session_state.keys())
        })


async def init_coordinator():
    """Initialize the coordinator asynchronously"""
    if not COORDINATOR_AVAILABLE:
        return False
        
    if st.session_state.coordinator is None:
        try:
            st.session_state.coordinator = await initialize_system()
            st.session_state.initialization_status = "completed"
            return True
        except Exception as e:
            st.session_state.initialization_status = f"error: {str(e)}"
            return False
    return True


def safe_run_async(coro):
    """Safely run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we can't run another coroutine
            return {"error": "Event loop already running"}
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        return {"error": str(e)}


def process_chat_query(query: str) -> str:
    """Process chat query and return response"""
    if not COORDINATOR_AVAILABLE:
        return "❌ Coordinator not available. Please check the import error in debug mode."
    
    if not st.session_state.coordinator:
        return "❌ System not initialized. Please check system health."
    
    try:
        # For now, return a simple response since we don't have the actual coordinator
        return f"📝 Query received: '{query}'\n\n🔧 This is a placeholder response. The actual coordinator integration needs to be implemented."
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
            st.rerun()


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


def show_demo_dashboard():
    """Show demo dashboard when coordinator is not available"""
    st.markdown('<div class="sub-header">System Overview (Demo Mode)</div>', unsafe_allow_html=True)
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", 42)
    
    with col2:
        st.metric("Total Queries", 156)
    
    with col3:
        st.metric("Avg Response Time", "2.3s")
    
    with col4:
        st.metric("Cache Hit Rate", "87.5%")
    
    # Demo chart
    st.markdown("### Processing Statistics (Demo)")
    demo_data = pd.DataFrame({
        'operation': ['File Analysis', 'Lineage Trace', 'Query Processing', 'DB2 Compare'],
        'avg_duration': [1.2, 3.4, 0.8, 5.1],
        'count': [25, 8, 45, 12]
    })
    
    fig = px.bar(demo_data, x="operation", y="avg_duration", 
                title="Average Processing Time by Operation")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📝 This is demo mode. Connect the actual coordinator to see real data.")


def show_chat_analysis():
    """Show chat-based analysis interface"""
    st.markdown('<div class="sub-header">💬 Chat with Opulence</div>', unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
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
        
        # Process query and generate response
        response = process_chat_query(user_input)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        st.rerun()
    
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


def show_file_upload():
    """Show file upload interface"""
    st.markdown('<div class="sub-header">📂 File Upload & Processing</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.warning("⚠️ Coordinator not available. File processing is disabled in demo mode.")
        st.info("To enable file processing, ensure the opulence_coordinator module is properly installed.")
        return
    
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
                st.info("🔧 File processing feature needs coordinator integration.")
    
    elif upload_type == "Batch Upload (ZIP)":
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing multiple files",
            type=['zip']
        )
        
        if uploaded_zip:
            if st.button("Extract and Process ZIP"):
                st.info("🔧 ZIP processing feature needs coordinator integration.")
    
    # Processing history
    st.markdown("### Processing History")
    if st.session_state.processing_history:
        df_history = pd.DataFrame(st.session_state.processing_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No processing history available")


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
        if not COORDINATOR_AVAILABLE:
            st.warning("⚠️ Coordinator not available. Analysis is disabled in demo mode.")
        else:
            st.info(f"🔧 Analyzing {component_name}... (Feature needs coordinator integration)")
    
    # Display current analysis
    if st.session_state.current_analysis:
        st.info("Analysis results would appear here when coordinator is connected.")


def show_field_lineage():
    """Show field lineage analysis interface"""
    st.markdown('<div class="sub-header">📊 Field Lineage Analysis</div>', unsafe_allow_html=True)
    
    # Field selection
    field_name = st.text_input("Field Name to Trace:")
    
    if st.button("🔍 Trace Field Lineage") and field_name:
        if not COORDINATOR_AVAILABLE:
            st.warning("⚠️ Coordinator not available. Lineage tracing is disabled in demo mode.")
        else:
            st.info(f"🔧 Tracing lineage for {field_name}... (Feature needs coordinator integration)")
    
    # Display lineage results
    if st.session_state.field_lineage_result:
        st.info("Lineage results would appear here when coordinator is connected.")


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
        if not COORDINATOR_AVAILABLE:
            st.warning("⚠️ Coordinator not available. DB2 comparison is disabled in demo mode.")
        else:
            st.info("🔧 Data comparison feature needs coordinator integration.")


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
        if not COORDINATOR_AVAILABLE:
            st.warning("⚠️ Coordinator not available. Documentation generation is disabled in demo mode.")
        else:
            st.info("🔧 Documentation generation feature needs coordinator integration.")


def show_system_health():
    """Show system health and statistics"""
    st.markdown('<div class="sub-header">⚙️ System Health & Statistics</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.error("🔴 System Status: Coordinator Not Available")
        st.markdown("### Import Error")
        st.code(st.session_state.get('import_error', 'Unknown import error'))
        st.info("Please ensure the opulence_coordinator module is properly installed and configured.")
        return
    
    if not st.session_state.coordinator:
        st.warning("🟡 System Status: Not Initialized")
        if st.button("🔄 Initialize System"):
            with st.spinner("Initializing system..."):
                success = safe_run_async(init_coordinator())
                if success:
                    st.success("✅ System initialized successfully")
                    st.rerun()
                else:
                    st.error(f"❌ Initialization failed: {st.session_state.initialization_status}")
        return
    
    st.success("🟢 System Status: Healthy")
    
    # System metrics would go here when coordinator is available
    st.info("System health monitoring active.")


def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🧠 Opulence Deep Research Mainframe Agent</div>', unsafe_allow_html=True)
    
    # Show import status
    if not COORDINATOR_AVAILABLE:
        st.error("⚠️ Coordinator module not available - Running in demo mode")
        if st.button("🐛 Toggle Debug Mode"):
            st.session_state.debug_mode = not st.session_state.debug_mode
            st.rerun()
    
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
        
        # System status indicator
        if COORDINATOR_AVAILABLE and st.session_state.coordinator:
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
            show_demo_dashboard()
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
            "session_state": dict(st.session_state)
        })