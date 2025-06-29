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
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import sys
import os
import shutil
import sqlite3
import traceback

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
    from opulence_coordinator import get_dynamic_coordinator, initialize_dynamic_system
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
            "response": "❌ Coordinator not available. Please check the import error in debug mode.",
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
        
        # Process with the enhanced chat agent
        result = safe_run_async(
            st.session_state.coordinator.process_chat_query(query, conversation_history)
        )
        
        if isinstance(result, dict):
            return result
        else:
            return {
                "response": str(result),
                "response_type": "general",
                "suggestions": []
            }
    
    except Exception as e:
        return {
            "response": f"❌ Error processing query: {str(e)}",
            "response_type": "error",
            "suggestions": ["Try rephrasing your question", "Check system status"]
        }

def show_chat_analysis():
    """Enhanced chat analysis interface with intelligent responses"""
    st.markdown('<div class="sub-header">💬 Chat with Opulence</div>', unsafe_allow_html=True)
    
    # Chat status indicator
    if st.session_state.coordinator:
        try:
            chat_status = safe_run_async(st.session_state.coordinator.get_chat_agent_status())
            if chat_status.get("status") == "available":
                st.success(f"🟢 Chat Agent Ready (GPU {chat_status.get('gpu_id', 'N/A')})")
            else:
                st.warning(f"🟡 Chat Agent: {chat_status.get('status', 'Unknown')}")
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
                                            "timestamp": datetime.now().isoformat()
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
            "timestamp": datetime.now().isoformat()
        })
        
        # Show processing indicator
        with st.spinner("🧠 Opulence is thinking..."):
            # Process query and generate response
            response_data = process_chat_query(user_input)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_data,
            "timestamp": datetime.now().isoformat()
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


def export_chat_history():
    """Export chat history with enhanced formatting"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
    
    # Create formatted export
    export_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "session_id": st.session_state.get("session_id", "unknown")
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
        "📥 Download Enhanced Chat History",
        export_json,
        file_name=f"opulence_chat_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def generate_chat_summary():
    """Generate an intelligent summary of the chat conversation"""
    if not st.session_state.chat_history:
        st.info("No conversation to summarize")
        return
    
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("🔄 Generating conversation summary..."):
        try:
            summary = safe_run_async(
                st.session_state.coordinator.get_conversation_summary(st.session_state.chat_history)
            )
            
            # Display summary in a nice format
            st.markdown("### 📋 Conversation Summary")
            st.markdown(summary)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": {
                    "response": f"**Conversation Summary:**\n\n{summary}",
                    "response_type": "summary",
                    "suggestions": ["Continue analysis", "Export summary", "Start new topic"]
                },
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")


def generate_follow_up_suggestions():
    """Generate intelligent follow-up question suggestions"""
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
    
    with st.spinner("🔮 Generating suggestions..."):
        try:
            suggestions = safe_run_async(
                st.session_state.coordinator.suggest_follow_up_questions(last_query, last_response)
            )
            
            if suggestions:
                st.markdown("### 💡 Suggested Follow-up Questions")
                for i, suggestion in enumerate(suggestions):
                    if st.button(f"❓ {suggestion}", key=f"followup_{i}"):
                        # Add as new user message
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": suggestion,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.rerun()
            
        except Exception as e:
            st.error(f"Failed to generate suggestions: {str(e)}")


def show_enhanced_component_analysis():
    """Enhanced component analysis with chat integration"""
    st.markdown('<div class="sub-header">🔍 Enhanced Component Analysis</div>', unsafe_allow_html=True)
    
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
    
    if st.button("🔍 Analyze Component") and component_name:
        analyze_component_enhanced(component_name, component_type, user_question, chat_enhanced)
    
    # Display current analysis with chat integration
    if st.session_state.current_analysis:
        display_enhanced_component_analysis(st.session_state.current_analysis)


def analyze_component_enhanced(component_name: str, component_type: str, user_question: str = None, chat_enhanced: bool = True):
    """Enhanced component analysis with optional chat integration"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"🧠 Analyzing {component_name}..."):
        try:
            if chat_enhanced and user_question:
                # Use chat-enhanced analysis
                result = safe_run_async(
                    st.session_state.coordinator.chat_analyze_component(
                        component_name, 
                        user_question,
                        st.session_state.chat_history[-3:] if st.session_state.chat_history else []
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
                    chat_query = f"Explain the analysis results for {component_name} in a conversational way."
                    chat_result = safe_run_async(
                        st.session_state.coordinator.process_chat_query(chat_query, [])
                    )
                    result = {
                        "component_name": component_name,
                        "analysis": result,
                        "chat_explanation": chat_result.get("response", ""),
                        "suggestions": chat_result.get("suggestions", []),
                        "response_type": "enhanced_analysis"
                    }
            
            st.session_state.current_analysis = result
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


def display_enhanced_component_analysis(analysis: dict):
    """Display enhanced component analysis results"""
    if isinstance(analysis, dict) and "error" in analysis:
        st.error(f"Analysis error: {analysis['error']}")
        return
    
    # Check if this is chat-enhanced
    if analysis.get("response_type") == "enhanced_analysis":
        component_name = analysis.get("component_name", "Unknown")
        st.success(f"✅ Enhanced analysis completed for: **{component_name}**")
        
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
                            "timestamp": datetime.now().isoformat()
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
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "role": "assistant",
                        "content": {
                            "response": analysis.get("chat_explanation", "Analysis completed."),
                            "response_type": "analysis",
                            "suggestions": suggestions
                        },
                        "timestamp": datetime.now().isoformat()
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
                    file_name=f"enhanced_analysis_{component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        # Regular analysis display
        display_component_analysis(analysis)


def show_enhanced_search():
    """Enhanced search interface with chat integration"""
    st.markdown('<div class="sub-header">🔍 Enhanced Code Search</div>', unsafe_allow_html=True)
    
    # Search input
    search_query = st.text_input("Describe what you're looking for:")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        search_type = st.selectbox("Search Type", ["Functionality", "Code Pattern", "Business Logic", "Data Flow"])
    
    with col2:
        result_count = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
    
    if st.button("🔍 Search with Chat Enhancement") and search_query:
        perform_enhanced_search(search_query, search_type, result_count)


def perform_enhanced_search(search_query: str, search_type: str, result_count: int):
    """Perform enhanced search with chat explanations"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner(f"🔍 Searching for '{search_query}'..."):
        try:
            # Use chat-enhanced search
            result = safe_run_async(
                st.session_state.coordinator.chat_search_patterns(
                    f"{search_type}: {search_query}",
                    st.session_state.chat_history[-3:] if st.session_state.chat_history else []
                )
            )
            
            # Display results
            if result.get("response_type") == "enhanced_search":
                st.success(f"✅ Found {result.get('total_found', 0)} results")
                
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
                                    analyze_component_enhanced(component_name, "auto-detect", f"Tell me about this component found in search for '{search_query}'", True)
                
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
                                "timestamp": datetime.now().isoformat()
                            })
            else:
                st.error("Search failed or returned unexpected results")
                
        except Exception as e:
            st.error(f"Enhanced search failed: {str(e)}")



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
    if isinstance(result, dict) and "error" in result:
        return f"❌ Analysis failed: {result['error']}"
    
    if not isinstance(result, dict):
        return f"❌ Unexpected result format: {type(result)}"
    
    component_name = result.get("component_name", "Unknown")
    component_type = result.get("component_type", "unknown")
    
    response = f"## 📊 Analysis of {component_type.title()}: {component_name}\n\n"
    
    if "lineage" in result:
        lineage = result["lineage"]
        if isinstance(lineage, dict):
            usage_stats = lineage.get("usage_analysis", {}).get("statistics", {})
            response += f"**Usage Summary:**\n"
            response += f"- Total references: {usage_stats.get('total_references', 0)}\n"
            response += f"- Programs using: {len(usage_stats.get('programs_using', []))}\n\n"
    
    if "logic_analysis" in result:
        response += f"**Logic Analysis:** Available\n\n"
    
    if "jcl_analysis" in result:
        response += f"**JCL Analysis:** Available\n\n"
    
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


def show_dashboard():
    """Show main dashboard"""
    st.markdown('<div class="sub-header">System Overview</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.warning("System not initialized")
        return
    
    # Get system statistics
    try:
        stats = st.session_state.coordinator.get_statistics()
        health = st.session_state.coordinator.get_health_status()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        system_stats = stats.get("system_stats", {})
        
        with col1:
            st.metric("Files Processed", system_stats.get("total_files_processed", 0))
        
        with col2:
            st.metric("Total Queries", system_stats.get("total_queries", 0))
        
        with col3:
            avg_time = system_stats.get("avg_response_time", 0)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        with col4:
            cache_rate = system_stats.get("cache_hit_rate", 0)
            if isinstance(cache_rate, (int, float)):
                st.metric("Cache Hit Rate", f"{cache_rate:.1%}")
            else:
                st.metric("Cache Hit Rate", "N/A")
        
        # Processing statistics chart
        processing_stats = stats.get("processing_stats", [])
        if processing_stats:
            st.markdown("### Processing Statistics")
            df_stats = pd.DataFrame(processing_stats)
            
            fig = px.bar(df_stats, x="operation", y="avg_duration", 
                        title="Average Processing Time by Operation")
            st.plotly_chart(fig, use_container_width=True)
        
        # File statistics
        file_stats = stats.get("file_stats", [])
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
                st.info(f"🕐 {activity['timestamp']}: Processed {activity['files_count']} files")
        else:
            st.info("No recent activities to display")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")


def show_file_upload():
    """Show file upload interface"""
    st.markdown('<div class="sub-header">📂 File Upload & Processing</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.warning("⚠️ Coordinator not available. File processing is disabled in demo mode.")
        st.info("To enable file processing, ensure the opulence_coordinator module is properly installed.")
        return
    
    if not st.session_state.coordinator:
        st.error("System not initialized. Please go to System Health tab to initialize.")
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
    """Process uploaded files using the coordinator"""
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
        
        # Process files in batch using coordinator
        status_text.text("Processing files with Opulence...")
        
        # Use the coordinator's process_batch_files method
        result = safe_run_async(
            st.session_state.coordinator.process_batch_files(file_paths)
        )
        
        # Display results
        with results_container:
            if isinstance(result, dict) and result.get("status") == "success":
                st.success(f"✅ Successfully processed {result.get('files_processed', 0)} files in {result.get('processing_time', 0):.2f} seconds")
                
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_count": len(uploaded_files),
            "status": result.get("status", "error") if isinstance(result, dict) else "error",
            "processing_time": result.get("processing_time", 0) if isinstance(result, dict) else 0
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
        
        # Process files using coordinator
        result = safe_run_async(
            st.session_state.coordinator.process_batch_files(file_paths)
        )
        
        if isinstance(result, dict) and result.get("status") == "success":
            st.success(f"✅ Successfully processed {result.get('files_processed', 0)} files")
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


def display_component_analysis(analysis: dict):
    """Display component analysis results with better error handling"""
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
    
    st.success(f"✅ Analysis completed for {component_type}: **{component_name}**")
    
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

def show_analysis_debug(analysis: dict):
    """Show debug information for analysis"""
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

def show_analysis_overview(analysis: dict):
    """Fixed analysis overview to show actual data"""
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
        status = analysis.get("status", "unknown")
        st.metric("Analysis Status", status.title())
    
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

def debug_database_content():
    """Debug function to check database content"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    try:
        conn = sqlite3.connect(st.session_state.coordinator.db_path)
        cursor = conn.cursor()
        
        st.markdown("### 🗃️ Database Content Check")
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        st.write(f"**Tables found:** {tables}")
        
        # Check program_chunks
        cursor.execute("SELECT COUNT(*) FROM program_chunks")
        chunk_count = cursor.fetchone()[0]
        st.write(f"**Total chunks:** {chunk_count}")
        
        if chunk_count > 0:
            cursor.execute("""
                SELECT program_name, COUNT(*) as count 
                FROM program_chunks 
                GROUP BY program_name 
                ORDER BY count DESC 
                LIMIT 10
            """)
            programs = cursor.fetchall()
            st.write("**Top programs by chunk count:**")
            for prog, count in programs:
                st.write(f"  - {prog}: {count} chunks")
        
        # Check chunk types
        if chunk_count > 0:
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count 
                FROM program_chunks 
                GROUP BY chunk_type
            """)
            chunk_types = cursor.fetchall()
            st.write("**Chunk types:**")
            for chunk_type, count in chunk_types:
                st.write(f"  - {chunk_type}: {count}")
        
        # Sample content
        if chunk_count > 0:
            cursor.execute("SELECT program_name, chunk_id, chunk_type, content FROM program_chunks LIMIT 3")
            samples = cursor.fetchall()
            st.write("**Sample chunks:**")
            for prog, chunk_id, chunk_type, content in samples:
                st.write(f"  - {prog}.{chunk_id} ({chunk_type}): {content[:100]}...")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Database debug failed: {str(e)}")

def show_lineage_analysis(analysis: dict):
    """Fixed lineage analysis display"""
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
    """Fixed impact analysis display"""
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
    """Fixed comprehensive analysis report"""
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
        report_sections.append(f"\n**Analysis completed in**: {processing_time:.2f} seconds")
    
    # Display the report
    report_content = "\n".join(report_sections)
    st.markdown(report_content)
    
    # Download button
    if report_content:
        st.download_button(
            "📄 Download Report",
            report_content,
            file_name=f"opulence_analysis_{component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    # Show comprehensive report if available
    if "comprehensive_report" in analysis:
        st.markdown("### 📊 Detailed Analysis Report")
        st.markdown(analysis["comprehensive_report"])
    elif "lineage" in analysis and isinstance(analysis["lineage"], dict) and "comprehensive_report" in analysis["lineage"]:
        st.markdown("### 📊 Detailed Lineage Report")
        st.markdown(analysis["lineage"]["comprehensive_report"])
    

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
            if hasattr(st.session_state.coordinator, 'agents') and "lineage_analyzer" in st.session_state.coordinator.agents:
                result = safe_run_async(
                    st.session_state.coordinator.agents["lineage_analyzer"].analyze_field_lineage(field_name)
                )
                st.session_state.field_lineage_result = result
            else:
                st.error("Lineage analyzer not available")
            
        except Exception as e:
            st.error(f"Lineage tracing failed: {str(e)}")


def display_field_lineage_results(result: dict):
    """Display field lineage results"""
    if isinstance(result, dict) and "error" in result:
        st.error(f"Lineage analysis error: {result['error']}")
        return
    
    if not isinstance(result, dict):
        st.error(f"Invalid lineage result: {type(result)}")
        return
    
    field_name = result.get("field_name", "Unknown")
    st.success(f"✅ Lineage analysis completed for field: **{field_name}**")
    
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
            if hasattr(st.session_state.coordinator, 'agents') and "db2_comparator" in st.session_state.coordinator.agents:
                result = safe_run_async(
                    st.session_state.coordinator.agents["db2_comparator"].compare_data(db2_comp)
                )
                display_comparison_results(result)
            else:
                st.error("DB2 comparator not available")
            
        except Exception as e:
            st.error(f"Comparison failed: {str(e)}")


def display_comparison_results(result: dict):
    """Display data comparison results"""
    if isinstance(result, dict) and "error" in result:
        st.error(f"Comparison error: {result['error']}")
        return
    
    if not isinstance(result, dict):
        st.error(f"Invalid comparison result: {type(result)}")
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
            if hasattr(st.session_state.coordinator, 'agents') and "documentation" in st.session_state.coordinator.agents:
                if component_name:
                    result = safe_run_async(
                        st.session_state.coordinator.agents["documentation"].generate_program_documentation(component_name)
                    )
                else:
                    # Generate system overview
                    result = safe_run_async(
                        st.session_state.coordinator.agents["documentation"].generate_system_documentation(["SYSTEM_OVERVIEW"])
                    )
                
                display_generated_documentation(result)
            else:
                st.error("Documentation agent not available")
            
        except Exception as e:
            st.error(f"Documentation generation failed: {str(e)}")


def display_generated_documentation(result: dict):
    """Display generated documentation"""
    if isinstance(result, dict) and "error" in result:
        st.error(f"Documentation error: {result['error']}")
        return
    
    if not isinstance(result, dict):
        st.error(f"Invalid documentation result: {type(result)}")
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
                try:
                    success = safe_run_async(init_coordinator())
                    if success and not isinstance(success, dict):
                        st.success("✅ System initialized successfully")
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
        stats = st.session_state.coordinator.get_statistics()
        
        # Overall health indicator
        if health.get("status") == "healthy":
            st.success("🟢 System Status: Healthy")
        else:
            st.error("🔴 System Status: Issues Detected")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Agents", health.get("active_agents", 0))
        
        with col2:
            memory_usage = health.get("memory_usage", {})
            if isinstance(memory_usage, dict) and "virtual" in memory_usage:
                st.metric("Memory Usage", f"{memory_usage['virtual'].get('percent', 0):.1f}%")
            else:
                st.metric("Memory Usage", "N/A")
        
        with col3:
            gpu_status = health.get("gpu_status", {})
            st.metric("GPUs Available", len(gpu_status) if isinstance(gpu_status, dict) else 0)
        
        with col4:
            cache_stats = health.get("cache_stats", {})
            hit_rate = cache_stats.get('hit_rate', 0) if isinstance(cache_stats, dict) else 0
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        
        # GPU utilization
        if health.get("gpu_status") and isinstance(health["gpu_status"], dict):
            st.markdown("### GPU Utilization")
            gpu_data = []
            for gpu_id, gpu_info in health["gpu_status"].items():
                if isinstance(gpu_info, dict):
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
        processing_stats = stats.get("processing_stats", [])
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
            if st.button("🗑️ Clear Cache"):
                try:
                    if hasattr(st.session_state.coordinator, 'cache_manager'):
                        st.session_state.coordinator.cache_manager.clear()
                    st.success("Cache cleared")
                except Exception as e:
                    st.error(f"Failed to clear cache: {str(e)}")
        
        with col3:
            if st.button("📊 Rebuild Indices"):
                rebuild_indices()
        
        with col4:
            if st.button("📥 Export Logs"):
                export_system_logs()
    
    except Exception as e:
        st.error(f"Error getting system health: {str(e)}")


def rebuild_indices():
    """Rebuild vector indices"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Rebuilding indices..."):
        try:
            if hasattr(st.session_state.coordinator, 'agents') and "vector_index" in st.session_state.coordinator.agents:
                result = safe_run_async(
                    st.session_state.coordinator.agents["vector_index"].rebuild_index()
                )
                
                if isinstance(result, dict) and result.get("status") == "success":
                    st.success("✅ Indices rebuilt successfully")
                else:
                    error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                    st.error(f"❌ Index rebuild failed: {error_msg}")
            else:
                st.error("Vector index agent not available")
                
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
             ["🏠 Dashboard", "📂 File Upload", "💬 Enhanced Chat", "🔍 Enhanced Analysis", 
            "🔍 Enhanced Search", "📊 Field Lineage", "🔄 DB2 Comparison", "📋 Documentation", "⚙️ System Health"]
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
            show_dashboard()
        elif page == "📂 File Upload":
            show_file_upload()
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
        elif page == "💬 Enhanced Chat":
            show_chat_analysis()
        elif page == "🔍 Enhanced Analysis":
            show_enhanced_component_analysis()
        elif page == "🔍 Enhanced Search":
            show_enhanced_search()
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