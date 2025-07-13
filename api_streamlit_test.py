#!/usr/bin/env python3
"""
Synchronous Coordinator that uses your REAL agent system
This adapts your existing APIOpulenceCoordinator to work synchronously
"""

import streamlit as st
import requests
import json
import time
import logging
import sqlite3
import hashlib
import tempfile
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import your existing agent classes
try:
    from agents.code_parser_agent_api import CodeParserAgent
    CODE_PARSER_AVAILABLE = True
except ImportError as e:
    CODE_PARSER_AVAILABLE = False
    st.error(f"CodeParserAgent import failed: {e}")

try:
    from agents.base_agent_api import BaseOpulenceAgent
    BASE_AGENT_AVAILABLE = True
except ImportError as e:
    BASE_AGENT_AVAILABLE = False
    st.error(f"BaseOpulenceAgent import failed: {e}")

# Import your coordinator components
try:
    from api_opulence_coordinator import (
        ModelServerConfig, 
        APIOpulenceConfig, 
        LoadBalancingStrategy,
        ModelServer,
        LoadBalancer
    )
    COORDINATOR_COMPONENTS_AVAILABLE = True
except ImportError as e:
    COORDINATOR_COMPONENTS_AVAILABLE = False
    st.error(f"Coordinator components import failed: {e}")

# ==================== Synchronous Wrapper for Your Coordinator ====================

class SyncCoordinatorWrapper:
    """Synchronous wrapper around your existing coordinator architecture"""
    
    def __init__(self, server_endpoint: str):
        self.server_endpoint = server_endpoint
        self.db_path = "opulence_sync_test.db"
        
        # Create config using your existing classes
        if COORDINATOR_COMPONENTS_AVAILABLE:
            self.server_config = ModelServerConfig(
                endpoint=server_endpoint,
                gpu_id=2,
                name="sync_test_server",
                max_concurrent_requests=1,
                timeout=60
            )
            
            self.config = APIOpulenceConfig(
                model_servers=[self.server_config],
                load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
                max_retries=1,
                connection_pool_size=1,
                request_timeout=60,
                db_path=self.db_path
            )
            
            self.load_balancer = LoadBalancer(self.config)
        
        self.stats = {
            "total_api_calls": 0,
            "total_files_processed": 0,
            "start_time": time.time(),
            "coordinator_type": "sync_wrapper"
        }
        
        self.agents = {}
        self.logger = logging.getLogger(f"{__name__}.SyncCoordinatorWrapper")
        
        # Initialize database using your schema
        self._init_database()
        
        # Test server
        self._test_server()
        
        self.logger.info(f"Sync coordinator wrapper initialized")
    
    def _init_database(self):
        """Initialize database with your existing schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            cursor = conn.cursor()
            
            # Use your existing database schema
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT UNIQUE NOT NULL,
                    file_type TEXT,
                    table_name TEXT,
                    fields TEXT,
                    source_type TEXT,
                    last_modified TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS program_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding_id TEXT,
                    file_hash TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    line_start INTEGER,
                    line_end INTEGER,
                    UNIQUE(program_name, chunk_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_program_chunks_name ON program_chunks(program_name);
                CREATE INDEX IF NOT EXISTS idx_program_chunks_type ON program_chunks(chunk_type);
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized with your schema")
            
        except Exception as e:
            self.logger.error(f"Database init failed: {e}")
            raise
    
    def _test_server(self):
        """Test server connectivity"""
        try:
            response = requests.get(f"{self.server_endpoint}/health", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Server connectivity verified")
            else:
                raise RuntimeError(f"Server health check failed: {response.status_code}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to server: {e}")
    
    def call_model_api(self, prompt: str, params: Dict[str, Any] = None, 
                      preferred_gpu_id: int = None) -> Dict[str, Any]:
        """Synchronous API call compatible with your existing interface"""
        try:
            params = params or {}
            request_data = {
                "prompt": prompt,
                "max_tokens": min(params.get("max_tokens", 512), 1024),
                "temperature": max(0.0, min(params.get("temperature", 0.1), 1.0)),
                "top_p": max(0.1, min(params.get("top_p", 0.9), 1.0)),
                "stream": False
            }
            
            self.logger.debug(f"Making sync API call")
            
            response = requests.post(
                f"{self.server_endpoint}/generate",
                json=request_data,
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.stats["total_api_calls"] += 1
                
                # Add metadata in the format your agents expect
                result.update({
                    "server_used": "sync_test_server",
                    "gpu_id": 2,
                    "latency": 0.1
                })
                
                return result
            else:
                raise RuntimeError(f"API call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise RuntimeError(f"Model server call failed: {e}")
    
    def get_agent(self, agent_type: str):
        """Get agent using your existing agent classes"""
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)
        return self.agents[agent_type]
    
    def _create_agent(self, agent_type: str):
        """Create agent using your existing agent classes"""
        if agent_type == "code_parser" and CODE_PARSER_AVAILABLE:
            # Create your actual CodeParserAgent
            agent = CodeParserAgent(
                llm_engine=None,  # Not used in API mode
                db_path=self.db_path,
                gpu_id=2,
                coordinator=self
            )
            
            # Replace async methods with sync versions
            agent.call_api_sync = self._make_sync_call_api(agent)
            
            self.logger.info("Created real CodeParserAgent")
            return agent
        
        elif BASE_AGENT_AVAILABLE:
            # Fallback to base agent
            agent = BaseOpulenceAgent(
                coordinator=self,
                agent_type=agent_type,
                db_path=self.db_path,
                gpu_id=2
            )
            
            agent.call_api_sync = self._make_sync_call_api(agent)
            
            self.logger.info(f"Created base agent for {agent_type}")
            return agent
        
        else:
            raise RuntimeError(f"Cannot create agent {agent_type} - imports failed")
    
    def _make_sync_call_api(self, agent):
        """Create sync version of call_api for the agent"""
        def call_api_sync(prompt: str, params: Dict[str, Any] = None) -> str:
            try:
                # Use the agent's existing API parameters
                final_params = agent.api_params.copy()
                if params:
                    final_params.update(params)
                
                # Make synchronous call through coordinator
                result = self.call_model_api(prompt, final_params)
                
                # Extract response in the format the agent expects
                if isinstance(result, dict):
                    response_text = (
                        result.get('text') or 
                        result.get('response') or 
                        result.get('content') or
                        result.get('generated_text') or
                        'No response generated'
                    )
                else:
                    response_text = str(result)
                
                return response_text
                
            except Exception as e:
                agent.logger.error(f"Sync API call failed: {e}")
                return f"API call failed: {str(e)}"
        
        return call_api_sync
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status compatible with your interface"""
        try:
            response = requests.get(f"{self.server_endpoint}/health", timeout=5)
            server_healthy = response.status_code == 200
            
            return {
                "status": "healthy" if server_healthy else "unhealthy",
                "coordinator_type": "sync_wrapper",
                "selected_gpus": [2],
                "available_servers": 1 if server_healthy else 0,
                "total_servers": 1,
                "server_stats": {
                    "sync_test_server": {
                        "endpoint": self.server_endpoint,
                        "status": "healthy" if server_healthy else "unhealthy",
                        "available": server_healthy
                    }
                },
                "stats": self.stats,
                "uptime_seconds": time.time() - self.stats["start_time"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "coordinator_type": "sync_wrapper",
                "stats": self.stats
            }
    
    def cleanup(self):
        """Cleanup method compatible with your interface"""
        for agent_type, agent in self.agents.items():
            try:
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup {agent_type}: {e}")
        
        self.agents.clear()
        self.logger.info("Sync coordinator cleanup completed")

# ==================== Streamlit Application ====================

def initialize_session_state():
    """Initialize session state"""
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []

def show_header():
    """Show application header"""
    st.title("🔧 Sync Coordinator with Real Agents")
    st.markdown("**Testing your actual CodeParserAgent synchronously**")

def show_server_config():
    """Show server configuration"""
    st.sidebar.markdown("### 🖥️ Server Configuration")
    
    server_endpoint = st.sidebar.text_input(
        "Model Server Endpoint", 
        value="http://171.201.3.165:8100"
    )
    
    if st.sidebar.button("🔧 Initialize System"):
        try:
            with st.spinner("Initializing with real agents..."):
                # Create coordinator wrapper
                coordinator = SyncCoordinatorWrapper(server_endpoint)
                
                # Store in session state
                st.session_state.coordinator = coordinator
                
                st.success("✅ Real agent system initialized!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.exception(e)
    
    # Show status
    if st.session_state.coordinator:
        health = st.session_state.coordinator.get_health_status()
        if health['status'] == 'healthy':
            st.sidebar.success("🟢 System Healthy")
        else:
            st.sidebar.error("🔴 System Unhealthy")
        
        stats = health['stats']
        st.sidebar.metric("API Calls", stats['total_api_calls'])
        st.sidebar.metric("Files Processed", stats['total_files_processed'])

def show_file_upload():
    """Show file upload interface"""
    st.markdown("### 📁 File Upload & Processing with Real CodeParser")
    
    if not st.session_state.coordinator:
        st.warning("⚠️ Please initialize the system first")
        return
    
    uploaded_file = st.file_uploader(
        "Upload a mainframe file",
        type=['cbl', 'cob', 'cobol', 'jcl', 'sql', 'txt']
    )
    
    if uploaded_file is not None:
        st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        try:
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            
            with st.expander("👁️ File Preview"):
                st.code(content[:500] + "..." if len(content) > 500 else content)
            
            if st.button("🚀 Process with Real CodeParser", type="primary"):
                process_with_real_agent(uploaded_file.name, content)
                
        except Exception as e:
            st.error(f"❌ Failed to read file: {str(e)}")

def process_with_real_agent(filename: str, content: str):
    """Process file using your real CodeParserAgent"""
    coordinator = st.session_state.coordinator
    
    with st.spinner(f"Processing {filename} with real CodeParserAgent..."):
        try:
            # Get your actual CodeParserAgent
            code_parser = coordinator.get_agent("code_parser")
            
            st.info(f"Using agent: {type(code_parser).__name__}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = Path(tmp_file.name)
            
            try:
                # Process using your agent's actual method
                # But call it synchronously if it's async
                start_time = time.time()
                
                if hasattr(code_parser, 'process_file'):
                    # Try to call process_file synchronously
                    try:
                        # If it's an async method, we need to adapt it
                        result = call_agent_method_sync(code_parser, 'process_file', tmp_path)
                    except Exception as e:
                        st.warning(f"process_file failed: {e}, trying alternative...")
                        # Fallback to manual processing
                        result = manual_code_parsing(code_parser, filename, content)
                else:
                    # Manual processing if no process_file method
                    result = manual_code_parsing(code_parser, filename, content)
                
                processing_time = time.time() - start_time
                
                # Store result
                result['processing_time'] = processing_time
                result['timestamp'] = dt.now().isoformat()
                st.session_state.processing_history.append(result)
                
                # Show results
                show_processing_result(result)
                
                coordinator.stats["total_files_processed"] += 1
                
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.exception(e)

def call_agent_method_sync(agent, method_name: str, *args, **kwargs):
    """Call agent method synchronously even if it's async"""
    method = getattr(agent, method_name)
    
    if hasattr(method, '__call__'):
        try:
            # Try calling directly (sync method)
            return method(*args, **kwargs)
        except Exception as e:
            if "coroutine" in str(e).lower():
                # It's an async method, we need a different approach
                raise RuntimeError(f"Method {method_name} is async - not supported in sync mode")
            else:
                raise

def manual_code_parsing(agent, filename: str, content: str) -> Dict[str, Any]:
    """Manual code parsing using the agent's API capabilities"""
    try:
        # Use the agent's sync API call method
        prompt = f"""
        Analyze this mainframe code file and provide a structured analysis:
        
        Filename: {filename}
        Content: {content[:2000]}
        
        Please provide:
        1. File type identification
        2. Main program structure
        3. Key components found
        4. Summary of functionality
        
        Format your response as a clear analysis:
        """
        
        analysis = agent.call_api_sync(prompt, {
            "max_tokens": 500,
            "temperature": 0.1
        })
        
        return {
            'status': 'success',
            'file_name': filename,
            'file_type': 'detected',
            'analysis': analysis,
            'method': 'manual_parsing_with_real_agent'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file_name': filename,
            'error': str(e),
            'method': 'manual_parsing_failed'
        }

def show_processing_result(result: Dict[str, Any]):
    """Show processing results"""
    st.markdown("### 📊 Processing Results")
    
    status = result.get('status', 'unknown')
    if status == 'success':
        st.success(f"✅ Successfully processed {result['file_name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Method", result.get('method', 'unknown'))
        with col2:
            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
        
        if 'analysis' in result:
            st.markdown("#### 🤖 Agent Analysis")
            st.markdown(result['analysis'])
        
        if 'structure' in result:
            st.markdown("#### 🏗️ Code Structure")
            st.json(result['structure'])
    
    else:
        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Sync Coordinator with Real Agents",
        page_icon="🔧",
        layout="wide"
    )
    
    initialize_session_state()
    show_header()
    
    # Check imports
    if not CODE_PARSER_AVAILABLE or not COORDINATOR_COMPONENTS_AVAILABLE:
        st.error("❌ Required agent imports failed. Check your agent files.")
        return
    
    show_server_config()
    show_file_upload()

if __name__ == "__main__":
    main()