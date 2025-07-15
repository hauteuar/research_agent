#!/usr/bin/env python3
"""
API-Based Opulence Coordinator System - FIXED VERSION
CRITICAL FIXES: Timeout context manager issues, proper asyncio task handling, streamlit compatibility
Keeps ALL existing class names and function signatures for compatibility
"""

import asyncio
import aiohttp
import json
import logging
import time
import uuid
import random
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime as dt
from contextlib import asynccontextmanager
from urllib.parse import urljoin
import sqlite3
from pathlib import Path
import os
from enum import Enum

# Import existing agents unchanged
try:
    from agents.code_parser_agent_api import CodeParserAgent
except ImportError:
    CodeParserAgent = None

try:
    from agents.chat_agent_api import OpulenceChatAgent
except ImportError:
    OpulenceChatAgent = None

try:
    from agents.vector_index_agent_api import VectorIndexAgent
except ImportError:
    VectorIndexAgent = None

try:
    from agents.data_loader_agent_api import DataLoaderAgent
except ImportError:
    DataLoaderAgent = None

try:
    from agents.lineage_analyzer_agent_api import LineageAnalyzerAgent    
except ImportError:
    LineageAnalyzerAgent = None

try:
    from agents.logic_analyzer_agent_api import LogicAnalyzerAgent
except ImportError:
    LogicAnalyzerAgent = None

try:
    from agents.documentation_agent_api import DocumentationAgent
except ImportError:
    DocumentationAgent = None

try:
    from agents.db2_comparator_agent_api import DB2ComparatorAgent
except ImportError:
    DB2ComparatorAgent = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration Classes ====================

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    LEAST_LATENCY = "least_latency"
    RANDOM = "random"

@dataclass
class ModelServerConfig:
    """Configuration for individual model server"""
    endpoint: str
    gpu_id: int  # Keep for compatibility but used as identifier only
    name: str = ""
    max_concurrent_requests: int = 1  # CONSERVATIVE: Only 1 request at a time
    timeout: int = 120  # FIXED: Reduced to 2 minutes
    
    def __post_init__(self):
        if not self.name:
            self.name = f"gpu_{self.gpu_id}"

@dataclass
class APIOpulenceConfig:
    """Configuration for API-based Opulence Coordinator - FIXED VERSION"""
    # Model server endpoints
    model_servers: List[ModelServerConfig] = field(default_factory=list)
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    
    # FIXED connection pool settings
    connection_pool_size: int = 1  # Even smaller for stability
    connection_timeout: int = 30  # FIXED: Reduced to 30 seconds
    request_timeout: int = 120  # FIXED: Reduced to 2 minutes
    
    # FIXED retry settings
    max_retries: int = 1  # Keep minimal retries
    retry_delay: float = 3.0  # FIXED: Reduced delay
    exponential_backoff: bool = False
    
    # FIXED health checking
    health_check_interval: int = 30  # FIXED: More frequent checks
    circuit_breaker_threshold: int = 5  # FIXED: Less tolerant
    circuit_breaker_timeout: int = 60  # FIXED: Shorter timeout
    
    # Database
    db_path: str = "opulence_api_data.db"
    
    # FIXED backwards compatibility settings
    max_tokens: int = 20  # FIXED: Even smaller default
    temperature: float = 0.1  # Low for speed
    auto_cleanup: bool = True
    
    @classmethod
    def from_gpu_endpoints(cls, gpu_endpoints: Dict[int, str]) -> 'APIOpulenceConfig':
        """Create config from GPU ID to endpoint mapping"""
        servers = []
        for gpu_id, endpoint in gpu_endpoints.items():
            servers.append(ModelServerConfig(
                endpoint=endpoint,
                gpu_id=gpu_id,
                name=f"gpu_{gpu_id}",
                max_concurrent_requests=1,  # CONSERVATIVE
                timeout=120  # FIXED: Reduced timeout
            ))
        return cls(model_servers=servers)

# ==================== Model Server Management ====================

class ModelServerStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"

class ModelServer:
    """Represents a single model server instance"""
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.status = ModelServerStatus.UNKNOWN
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.consecutive_failures = 0
        self.circuit_breaker_open_time = 0
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    @property
    def average_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    def is_available(self) -> bool:
        """Check if server is available for requests"""
        if self.status == ModelServerStatus.CIRCUIT_OPEN:
            # Check if circuit breaker should be reset
            if time.time() - self.circuit_breaker_open_time > self.config.timeout:
                self.status = ModelServerStatus.UNKNOWN
                return True
            return False
        
        return (self.status in [ModelServerStatus.HEALTHY, ModelServerStatus.UNKNOWN] and 
                self.active_requests < self.config.max_concurrent_requests)
    
    def record_success(self, latency: float):
        """Record successful request"""
        self.successful_requests += 1
        self.total_latency += latency
        self.consecutive_failures = 0
        self.status = ModelServerStatus.HEALTHY
        
    def record_failure(self):
        """Record failed request"""
        self.failed_requests += 1
        self.consecutive_failures += 1
        
    def should_open_circuit(self, threshold: int) -> bool:
        """Check if circuit breaker should open"""
        return self.consecutive_failures >= threshold
    
    def open_circuit(self):
        """Open circuit breaker"""
        self.status = ModelServerStatus.CIRCUIT_OPEN
        self.circuit_breaker_open_time = time.time()
        self.logger.warning(f"Circuit breaker opened for {self.config.name}")

# ==================== Load Balancer ====================

class LoadBalancer:
    """Load balancer for routing requests to available model servers"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.config = config
        self.servers: List[ModelServer] = []
        self.current_index = 0
        self.logger = logging.getLogger(f"{__name__}.LoadBalancer")
        
        # Initialize servers
        for server_config in config.model_servers:
            self.servers.append(ModelServer(server_config))
    
    def get_available_servers(self) -> List[ModelServer]:
        """Get list of available servers"""
        return [server for server in self.servers if server.is_available()]
    
    def select_server(self) -> Optional[ModelServer]:
        """Select best server based on load balancing strategy"""
        available_servers = self.get_available_servers()
        
        if not available_servers:
            return None
        
        if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            server = available_servers[self.current_index % len(available_servers)]
            self.current_index += 1
            return server
        
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_BUSY:
            return min(available_servers, key=lambda s: s.active_requests)
        
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_LATENCY:
            return min(available_servers, key=lambda s: s.average_latency)
        
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_servers)
        
        else:
            return available_servers[0]
    
    def get_server_by_gpu_id(self, gpu_id: int) -> Optional[ModelServer]:
        """Get server by GPU ID - for compatibility only"""
        for server in self.servers:
            if server.config.gpu_id == gpu_id:
                return server
        return None

# ==================== API Client for Model Servers ====================

class ModelServerClient:
    """HTTP client for calling model servers - FIXED VERSION"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"{__name__}.ModelServerClient")
        
    async def initialize(self):
        """ULTRA-CONSERVATIVE: Session initialization without complex timeouts"""
        
        # ULTRA-SIMPLE connector - no fancy settings
        connector = aiohttp.TCPConnector(
            limit=1,
            limit_per_host=1,
            enable_cleanup_closed=False,  # CRITICAL: Don't auto-cleanup
            force_close=False  # CRITICAL: Don't force close connections
        )
        
        # CRITICAL: Very simple timeout - only total timeout
        timeout = aiohttp.ClientTimeout(total=60)  # 1 minute total, that's it
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        
        self.logger.info("ULTRA-CONSERVATIVE model server client initialized")
        
    
    async def close(self):
        """FIXED: Safe session cleanup"""
        if self.session:
            try:
                await self.session.close()
                # FIXED: Give time for cleanup
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.warning(f"Session cleanup warning: {e}")
            finally:
                self.session = None
    


    async def call_generate(self, server: ModelServer, prompt: str, 
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """ULTRA-SIMPLE: API call without complex error handling"""
        
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        params = params or {}
        
        # ULTRA-CONSERVATIVE request - minimal parameters
        request_data = {
            "prompt": prompt[:100],  # Truncate prompt
            "max_tokens": min(params.get("max_tokens", 5), 10),  # Very small
            "temperature": 0.1,
            "stream": False
        }
        
        server.active_requests += 1
        server.total_requests += 1
        start_time = time.time()
        
        try:
            generate_url = f"{server.config.endpoint.rstrip('/')}/generate"
            
            # ULTRA-SIMPLE: Direct post without complex error handling
            response = await self.session.post(generate_url, json=request_data)
            
            if response.status == 200:
                result = await response.json()
                latency = time.time() - start_time
                server.record_success(latency)
                
                result["server_used"] = server.config.name
                result["gpu_id"] = server.config.gpu_id
                result["latency"] = latency
                
                return result
            else:
                error_text = f"HTTP {response.status}"
                server.record_failure()
                return {"error": error_text}
                
        except Exception as e:
            server.record_failure()
            return {"error": f"Request failed: {str(e)}"}
            
        finally:
            server.active_requests = max(0, server.active_requests - 1)
            
    async def health_check(self, server: ModelServer) -> bool:
        """FIXED: Ultra-simple health check"""
        try:
            if not self.session:
                return False
                
            health_url = f"{server.config.endpoint.rstrip('/')}/health"
            
            # FIXED: Direct call with session's timeout only
            response = await self.session.get(health_url)
            
            try:
                if response.status == 200:
                    server.status = ModelServerStatus.HEALTHY
                    return True
                else:
                    server.status = ModelServerStatus.UNHEALTHY
                    return False
            finally:
                response.close()
                    
        except Exception as e:
            server.status = ModelServerStatus.UNHEALTHY
            self.logger.debug(f"Health check failed for {server.config.name}: {e}")
            return False
        
# ==================== API-Compatible Engine Context ====================

class APIEngineContext:
    """FIXED: Provides engine-like interface for existing agents using API calls"""
    
    def __init__(self, coordinator, preferred_gpu_id: int = None):
        self.coordinator = coordinator
        self.preferred_gpu_id = preferred_gpu_id  # Keep for compatibility but ignore
        self.logger = logging.getLogger(f"{__name__}.APIEngineContext")
    
    async def generate(self, prompt: str, sampling_params, request_id: str = None):
        """FIXED: Generate text via API (compatible with vLLM interface)"""
        # Convert sampling_params to API parameters
        params = {}
        
        if hasattr(sampling_params, '__dict__'):
            for attr in ['max_tokens', 'temperature', 'top_p', 'top_k', 
                        'frequency_penalty', 'presence_penalty', 'stop', 'seed']:
                if hasattr(sampling_params, attr):
                    value = getattr(sampling_params, attr)
                    if value is not None:
                        params[attr] = value
        elif isinstance(sampling_params, dict):
            params = sampling_params.copy()
        else:
            params = {"max_tokens": 10, "temperature": 0.1, "top_p": 0.9}  # FIXED: Even smaller
        
        # FIXED: Ultra-conservative parameter validation
        validated_params = {}
        for key, value in params.items():
            if value is not None:
                if key == "max_tokens":
                    validated_params[key] = max(1, min(value, 20))  # FIXED: Smaller limit
                elif key == "temperature":
                    validated_params[key] = max(0.0, min(value, 0.2))  # FIXED: Lower max
                elif key == "top_p":
                    validated_params[key] = max(0.0, min(value, 1.0))
                else:
                    validated_params[key] = value
        
        # Call API - ignore preferred_gpu_id, just use load balancer
        result = await self.coordinator.call_model_api(
            prompt=prompt, 
            params=validated_params
        )
        
        # Convert API response to vLLM-compatible format
        class MockOutput:
            def __init__(self, text: str, finish_reason: str):
                self.text = text
                self.finish_reason = finish_reason
                self.token_ids = []
        
        class MockRequestOutput:
            def __init__(self, result: Dict[str, Any]):
                if isinstance(result, dict):
                    text = (
                        result.get('text') or 
                        result.get('response') or 
                        result.get('content') or
                        result.get('generated_text') or
                        str(result.get('choices', [{}])[0].get('text', '')) or
                        ''
                    )
                    finish_reason = result.get('finish_reason', 'stop')
                else:
                    text = str(result)
                    finish_reason = 'stop'
                
                self.outputs = [MockOutput(text, finish_reason)]
                self.finished = True
                self.prompt_token_ids = []
        
        yield MockRequestOutput(result)

# ==================== API-Based Coordinator (Keep exact class name) ====================

class APIOpulenceCoordinator:
    """FIXED: API-based Opulence Coordinator with proper timeout handling"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.logger = logging.getLogger(f"{__name__}.APIOpulenceCoordinator")
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.client = ModelServerClient(config)
        self.db_path = config.db_path
        
        # Agent storage - keep existing agents unchanged
        self.agents = {}
        
        # FIXED: No health check task for simplicity
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize database
        self._init_database()

        # Keep these for compatibility
        self.primary_gpu_id = None
        self.available_gpu_ids = [server.config.gpu_id for server in self.load_balancer.servers]
        
        # Statistics
        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "total_api_calls": 0,
            "avg_response_time": 0,
            "start_time": time.time(),
            "coordinator_type": "api_based_fixed"
        }
        
        # FIXED: Ultra-conservative agent configurations
        self.agent_configs = {
            "code_parser": {
                "max_tokens": 20,  # FIXED: Very small
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Analyzes and parses code structures"
            },
            "vector_index": {
                "max_tokens": 15,  # FIXED: Even smaller
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Handles vector embeddings and similarity search"
            },
            "data_loader": {
                "max_tokens": 20,  # FIXED: Small
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Processes and loads data files"
            },
            "lineage_analyzer": {
                "max_tokens": 25,  # FIXED: Small
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Analyzes data and code lineage"
            },
            "logic_analyzer": {
                "max_tokens": 25,  # FIXED: Small
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Analyzes program logic and flow"
            },
            "documentation": {
                "max_tokens": 30,  # FIXED: Medium
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Generates documentation"
            },
            "db2_comparator": {
                "max_tokens": 20,  # FIXED: Small
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Compares database schemas and data"
            },
            "chat_agent": {
                "max_tokens": 25,  # FIXED: Small
                "temperature": 0.15,  # Slightly higher for chat
                "top_p": 0.9,
                "description": "Handles interactive chat queries"
            }
        }

        # For backwards compatibility
        self.selected_gpus = [server.config.gpu_id for server in self.load_balancer.servers]
        
        self.logger.info(f"FIXED API Coordinator initialized with servers: {[s.config.name for s in self.load_balancer.servers]}")
    
    async def initialize(self):
        """FIXED: Initialize the coordinator with proper agent handling"""
        try:
            await self.client.initialize()
            
            # Test server connectivity with fixed approach
            await self._test_connectivity_fixed()
            
            # Initialize task list for tracking
            self._initialization_tasks = []
            
            self.logger.info("FIXED API Coordinator initialized successfully")
        except Exception as e:
            self.logger.error(f"FIXED coordinator initialization failed: {e}")
            raise

    
    async def _test_connectivity_fixed(self):
        """FIXED: Test connectivity to all servers without timeout conflicts"""
        healthy_count = 0
        for server in self.load_balancer.servers:
            try:
                # Simple delay between tests
                if healthy_count > 0:
                    await asyncio.sleep(0.5)
                
                if await self.client.health_check(server):
                    healthy_count += 1
                    self.logger.info(f"✅ {server.config.name} ({server.config.endpoint}) - Connected")
                else:
                    self.logger.warning(f"❌ {server.config.name} ({server.config.endpoint}) - Failed")
            except Exception as e:
                self.logger.error(f"Server test error for {server.config.name}: {e}")
        
        if healthy_count == 0:
            raise RuntimeError("No model servers are accessible!")
        
        self.logger.info(f"Connected to {healthy_count}/{len(self.load_balancer.servers)} servers")
    
    async def shutdown(self):
        """FIXED: Safe shutdown with agent initialization cleanup"""
        try:
            # Cancel any pending initialization tasks
            if hasattr(self, '_initialization_tasks'):
                for task in self._initialization_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.warning(f"Task cleanup warning: {e}")
            
            # Close client safely
            if self.client:
                await self.client.close()
            
            self.logger.info("FIXED API Coordinator shut down successfully")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    def _init_database(self):
        """Initialize database (same as original)"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            
            cursor = conn.cursor()
            
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
                
                CREATE INDEX IF NOT EXISTS idx_file_metadata_name ON file_metadata(file_name);
                CREATE INDEX IF NOT EXISTS idx_file_metadata_type ON file_metadata(file_type);
                
                CREATE TABLE IF NOT EXISTS field_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    program_name TEXT,
                    paragraph TEXT,
                    operation TEXT,
                    source_file TEXT,
                    target_file TEXT,
                    transformation_logic TEXT,
                    data_type TEXT,
                    business_domain TEXT,
                    sensitivity_level TEXT,
                    last_used TIMESTAMP,
                    read_in TEXT,
                    updated_in TEXT,
                    purged_in TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_field_lineage_field ON field_lineage(field_name);
                CREATE INDEX IF NOT EXISTS idx_field_lineage_program ON field_lineage(program_name);
                
                CREATE TABLE IF NOT EXISTS program_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    business_context TEXT,
                    embedding_id TEXT,
                    file_hash TEXT,
                    confidence_score REAL DEFAULT 1.0,
                    llm_analysis TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    line_start INTEGER,
                    line_end INTEGER,
                    UNIQUE(program_name, chunk_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_program_chunks_name ON program_chunks(program_name);
                CREATE INDEX IF NOT EXISTS idx_program_chunks_type ON program_chunks(chunk_type);
                CREATE INDEX IF NOT EXISTS idx_file_hash ON program_chunks(file_hash);
                
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    operation TEXT,
                    duration REAL,
                    server_used TEXT,
                    gpu_id INTEGER,
                    status TEXT,
                    details TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_processing_stats_timestamp ON processing_stats(timestamp);
                
                CREATE TABLE IF NOT EXISTS vector_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id INTEGER,
                    embedding_id TEXT,
                    faiss_id INTEGER,
                    embedding_vector TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES program_chunks (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_vector_embeddings_chunk ON vector_embeddings(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_vector_embeddings_faiss ON vector_embeddings(faiss_id);
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    async def call_model_api(self, prompt: str, params: Dict[str, Any] = None, 
                            preferred_gpu_id: int = None) -> Dict[str, Any]:
        """FIXED: API call with simplified error handling"""
        
        server = self.load_balancer.select_server()        
        if not server:
            raise RuntimeError("No available servers found")
        
        try:
            self.logger.debug(f"FIXED API call to {server.config.name}")
            
            # Direct call with fixed parameters
            result = await self.client.call_generate(server, prompt, params)
            self.stats["total_api_calls"] += 1
            return result
            
        except Exception as e:
            self.logger.warning(f"Request failed on {server.config.name}: {e}")
            
            # Simple retry with different server if available
            retry_server = self.load_balancer.select_server()
            if retry_server and retry_server != server:
                try:
                    self.logger.info(f"Retrying with {retry_server.config.name}")
                    result = await self.client.call_generate(retry_server, prompt, params)
                    self.stats["total_api_calls"] += 1
                    return result
                except Exception as retry_e:
                    self.logger.error(f"Retry failed: {retry_e}")
            
            raise RuntimeError(f"All servers failed: {str(e)}")
    
    # ==================== Keep all existing methods unchanged ====================
    
    def get_agent(self, agent_type: str):
        """Get agent - creates instances that use API calls instead of direct GPU access"""
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)
        return self.agents[agent_type]
    
    def _create_agent(self, agent_type: str):
        """FIXED: Create agent with proper async initialization handling"""
        self.logger.info(f"🔗 Creating {agent_type} agent (FIXED API-based)")
        
        # Get agent configuration
        agent_config = self.agent_configs.get(agent_type, {})
        
        # Use first available server's GPU ID for compatibility
        selected_gpu_id = self.available_gpu_ids[0] if self.available_gpu_ids else 0
        
        try:
            # Import the base agent class
            from agents.base_agent_api import BaseOpulenceAgent
            
            # Create agents with FIXED initialization
            if agent_type == "code_parser" and CodeParserAgent:
                agent = CodeParserAgent(
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id,
                    coordinator=self
                )
            elif agent_type == "vector_index" and VectorIndexAgent:
                agent = VectorIndexAgent(
                    coordinator=self,  # FIXED: Pass coordinator first
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id
                )
                # CRITICAL FIX: Don't call async methods in __init__
                # The agent will handle its own initialization
                
            elif agent_type == "data_loader" and DataLoaderAgent:
                agent = DataLoaderAgent(
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id,
                    coordinator=self
                )
            elif agent_type == "lineage_analyzer" and LineageAnalyzerAgent:
                agent = LineageAnalyzerAgent(
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id,
                    coordinator=self
                )
            elif agent_type == "logic_analyzer" and LogicAnalyzerAgent:
                agent = LogicAnalyzerAgent(
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id,
                    coordinator=self
                )
            elif agent_type == "documentation" and DocumentationAgent:
                agent = DocumentationAgent(
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id,
                    coordinator=self
                )
            elif agent_type == "db2_comparator" and DB2ComparatorAgent:
                agent = DB2ComparatorAgent(
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id,
                    max_rows=10000,
                    coordinator=self
                )
            elif agent_type == "chat_agent" and OpulenceChatAgent:
                agent = OpulenceChatAgent(
                    coordinator=self,
                    llm_engine=None,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id
                )
            else:
                # Fallback to base agent
                agent = BaseOpulenceAgent(
                    coordinator=self,
                    agent_type=agent_type,
                    db_path=self.db_path,
                    gpu_id=selected_gpu_id
                )
            
            # Configure agent with FIXED API parameters
            if hasattr(agent, 'update_api_params') and agent_config:
                fixed_params = {
                    'max_tokens': min(agent_config.get('max_tokens', 15), 30),
                    'temperature': min(agent_config.get('temperature', 0.1), 0.15),
                    'top_p': min(agent_config.get('top_p', 0.9), 0.9),
                    'stream': False,
                    'stop': ["\n\n", "###"]
                }
                agent.update_api_params(**fixed_params)
                self.logger.info(f"Applied FIXED configuration to {agent_type}: {fixed_params}")
            
            # Inject API-based engine context
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create {agent_type} agent: {str(e)}")
            raise RuntimeError(f"Agent creation failed for {agent_type}: {str(e)}")
        
    def _schedule_agent_initialization(self, agent):
        """CRITICAL FIX: Properly schedule async agent initialization"""
        if hasattr(agent, '_initialize_components'):
            try:
                # Create a task to initialize the agent components
                loop = asyncio.get_event_loop()
                
                # Schedule the initialization to run later
                task = loop.create_task(self._initialize_agent_safely(agent))
                
                # Store task reference to prevent garbage collection
                if not hasattr(self, '_initialization_tasks'):
                    self._initialization_tasks = []
                self._initialization_tasks.append(task)
                
                self.logger.info(f"✅ Scheduled initialization for {agent.__class__.__name__}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not schedule agent initialization: {e}")
                # Try to initialize synchronously as fallback
                try:
                    if hasattr(agent, '_initialize_components_sync'):
                        agent._initialize_components_sync()
                except Exception as sync_error:
                    self.logger.error(f"❌ Sync initialization also failed: {sync_error}")


    async def _initialize_agent_safely(self, agent):
        """FIXED: Safely initialize agent components"""
        try:
            if hasattr(agent, '_initialize_components'):
                await agent._initialize_components()
                self.logger.info(f"✅ {agent.__class__.__name__} initialized successfully")
            else:
                self.logger.info(f"ℹ️ {agent.__class__.__name__} has no async initialization")
        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")


    def _create_engine_context_for_agent(self, agent):
        """FIXED: Create API-based engine context for agent"""
        @asynccontextmanager
        async def api_engine_context():
            # Create API-based engine context - ignore GPU ID preference
            api_context = APIEngineContext(self, preferred_gpu_id=None)
            try:
                yield api_context
            except Exception as e:
                self.logger.error(f"Engine context error: {e}")
                raise
            finally:
                # No cleanup needed for API calls
                pass
        
        return api_engine_context
    
    def list_available_agents(self) -> Dict[str, Any]:
        """List all available agent types and their configurations"""
        return {
            agent_type: {
                "config": config,
                "available": agent_type in [
                    "code_parser", "vector_index", "data_loader", 
                    "lineage_analyzer", "logic_analyzer", "documentation",
                    "db2_comparator", "chat_agent"
                ],
                "loaded": agent_type in self.agents
            }
            for agent_type, config in self.agent_configs.items()
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all loaded agents"""
        status = {}
        for agent_type, agent in self.agents.items():
            if hasattr(agent, 'get_agent_stats'):
                try:
                    status[agent_type] = agent.get_agent_stats()
                except Exception as e:
                    status[agent_type] = {
                        "agent_type": agent_type,
                        "status": "error",
                        "error": str(e)
                    }
            else:
                status[agent_type] = {
                    "agent_type": agent_type,
                    "gpu_id": getattr(agent, 'gpu_id', None),
                    "api_based": True,
                    "status": "loaded"
                }
        return status
    
    def update_agent_config(self, agent_type: str, **config_updates):
        """Update configuration for a specific agent type"""
        if agent_type not in self.agent_configs:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Update stored configuration
        self.agent_configs[agent_type].update(config_updates)
        
        # Update live agent if it exists
        if agent_type in self.agents:
            agent = self.agents[agent_type]
            if hasattr(agent, 'update_api_params'):
                api_params = {k: v for k, v in config_updates.items() 
                             if k in ['max_tokens', 'temperature', 'top_p', 'top_k', 
                                     'frequency_penalty', 'presence_penalty']}
                if api_params:
                    try:
                        agent.update_api_params(**api_params)
                        self.logger.info(f"Updated live agent {agent_type} configuration: {api_params}")
                    except Exception as e:
                        self.logger.error(f"Failed to update {agent_type} config: {e}")
    
    def reload_agent(self, agent_type: str):
        """Reload a specific agent with updated configuration"""
        if agent_type in self.agents:
            # Cleanup old agent
            old_agent = self.agents[agent_type]
            if hasattr(old_agent, 'cleanup'):
                try:
                    old_agent.cleanup()
                except Exception as e:
                    self.logger.warning(f"Agent cleanup warning: {e}")
            
            # Remove from cache
            del self.agents[agent_type]
            
            self.logger.info(f"Reloaded {agent_type} agent")
        
        # Agent will be recreated on next access
        return self.get_agent(agent_type)
    
    # ==================== Existing Interface Methods (Keep Unchanged) ====================
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """FIXED: Process files with immediate vector index update"""
        start_time = time.time()
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"🚀 Processing {total_files} files with vector index update")
        
        try:
            for i, file_path in enumerate(file_paths):
                try:
                    self.logger.info(f"📄 Processing file {i+1}/{total_files}: {file_path.name}")
                    
                    # Determine agent type
                    if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                        agent_type = "code_parser"
                    elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                        agent_type = "data_loader"
                    else:
                        agent_type = "code_parser"
                    
                    # Get agent
                    agent = self.get_agent(agent_type)
                    
                    # Process file
                    result = await agent.process_file(file_path)
                    
                    # CRITICAL FIX: Immediately update vector index
                    if result and result.get('status') == 'success':
                        await self._update_vector_index_for_file(file_path, result)
                    
                    await self._ensure_file_stored_in_db(file_path, result, file_type)
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {file_path}: {str(e)}")
                    results.append({
                        "status": "error",
                        "file": str(file_path),
                        "error": str(e)
                    })
            
            # Final vector index rebuild to ensure consistency
            await self._ensure_vector_index_ready()
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_files_processed"] += total_files
            
            return {
                "status": "success",
                "files_processed": total_files,
                "processing_time": processing_time,
                "results": results,
                "servers_used": [s.config.name for s in self.load_balancer.servers],
                "vector_index_updated": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "files_processed": 0,
                "results": []
            }
    
    async def _update_vector_index_for_file(self, file_path: Path, process_result: Dict[str, Any]):
        """FIXED: Update vector index immediately after file processing"""
        try:
            # Get new chunks for this file
            chunks = await self._get_chunks_for_file(file_path.name)
            
            if chunks:
                vector_agent = self.get_agent("vector_index")
                
                # FIXED: Use correct method name from vector agent
                update_result = await self._safe_agent_call(
                    vector_agent.create_embeddings_for_chunks,  # FIXED: Correct method
                    chunks
                )
                
                if update_result and not update_result.get('error'):
                    self.logger.info(f"✅ Vector index updated for {file_path.name} with {len(chunks)} chunks")
                else:
                    self.logger.warning(f"⚠️ Vector index update failed for {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Vector index update failed for {file_path.name}: {e}")

    async def _get_chunks_for_file(self, file_name: str):
        """Get chunks for a specific file"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT program_name, chunk_id, chunk_type, content, metadata
                FROM program_chunks
                WHERE program_name = ? AND content IS NOT NULL AND content != ''
                ORDER BY created_timestamp DESC
            """, (file_name,))
            
            rows = cursor.fetchall()
            conn.close()
            
            chunks = []
            for row in rows:
                chunks.append({
                    'program_name': row[0],
                    'chunk_id': row[1],
                    'chunk_type': row[2],
                    'content': row[3],
                    'metadata': row[4]
                })
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get chunks for {file_name}: {e}")
            return []


    async def analyze_component(self, component_name: str, component_type: str = None, **kwargs) -> Dict[str, Any]:
        """ENHANCED: Complete component analysis with documentation summary"""
        start_time = time.time()
        
        try:
            # Auto-detect component type if not provided
            if not component_type or component_type == "auto-detect":
                component_type = await self._determine_component_type(component_name)
            
            # CRITICAL FIX: Check for prefixed component names during upload
            original_component_name = component_name
            cleaned_component_name = self._clean_component_name(component_name)
            
            # Use cleaned name for analysis
            analysis_component_name = cleaned_component_name
            
            # Normalize the component type for processing
            normalized_type = self._normalize_component_type(component_type)
            
            analysis_result = {
                "component_name": original_component_name,
                "cleaned_component_name": cleaned_component_name,
                "component_type": component_type,
                "normalized_type": normalized_type,
                "analysis_timestamp": dt.now().isoformat(),
                "status": "in_progress",
                "analyses": {},
                "processing_metadata": {
                    "start_time": start_time,
                    "coordinator_type": "api_based_enhanced_with_docs"
                }
            }
            
            completed_count = 0
            
            # Ensure all required agents are ready
            await self._ensure_agents_ready()
            
            # STEP 1: LINEAGE ANALYSIS (FOUNDATIONAL)
            try:
                self.logger.info(f"🔄 Step 1: Running lineage analysis for {analysis_component_name} (type: {normalized_type})")
                lineage_agent = self.get_agent("lineage_analyzer")
                
                # Call appropriate lineage method based on type
                if normalized_type == "field":
                    lineage_result = await self._safe_agent_call(
                        lineage_agent.analyze_field_lineage,
                        analysis_component_name
                    )
                else:
                    lineage_result = await self._safe_agent_call(
                        lineage_agent.analyze_full_lifecycle,
                        analysis_component_name,
                        normalized_type
                    )
                
                if lineage_result and not lineage_result.get('error'):
                    analysis_result["analyses"]["lineage_analysis"] = {
                        "status": "success",
                        "data": lineage_result,
                        "agent_used": "lineage_analyzer",
                        "completion_time": time.time() - start_time,
                        "step": 1
                    }
                    completed_count += 1
                    self.logger.info(f"✅ Step 1: Lineage analysis completed successfully")
                else:
                    error_msg = lineage_result.get('error', 'No result returned') if lineage_result else 'No result returned'
                    analysis_result["analyses"]["lineage_analysis"] = {
                        "status": "error",
                        "error": error_msg,
                        "agent_used": "lineage_analyzer",
                        "step": 1
                    }
                    self.logger.warning(f"⚠️ Step 1: Lineage analysis failed: {error_msg}")
                
            except Exception as e:
                self.logger.error(f"❌ Step 1: Lineage analysis exception: {str(e)}")
                analysis_result["analyses"]["lineage_analysis"] = {
                    "status": "error",
                    "error": str(e),
                    "agent_used": "lineage_analyzer",
                    "step": 1
                }
            
            # STEP 2: LOGIC ANALYSIS (FOR COBOL/PROGRAM TYPES)
            if normalized_type in ["cobol", "copybook", "program", "jcl"]:
                try:
                    self.logger.info(f"🔄 Step 2: Running logic analysis for {analysis_component_name} (type: {normalized_type})")
                    logic_agent = self.get_agent("logic_analyzer")
                    
                    # Call appropriate logic method based on type
                    if normalized_type in ["cobol", "program"]:
                        logic_result = await self._safe_agent_call(
                            logic_agent.analyze_program,
                            analysis_component_name
                        )
                    else:
                        logic_result = await self._safe_agent_call(
                            logic_agent.find_dependencies,
                            analysis_component_name
                        )
                    
                    if logic_result and not logic_result.get('error'):
                        analysis_result["analyses"]["logic_analysis"] = {
                            "status": "success",
                            "data": logic_result,
                            "agent_used": "logic_analyzer",
                            "completion_time": time.time() - start_time,
                            "step": 2,
                            "normalized_type": normalized_type
                        }
                        completed_count += 1
                        self.logger.info(f"✅ Step 2: Logic analysis completed successfully")
                    else:
                        error_msg = logic_result.get('error', 'No result returned') if logic_result else 'No result returned'
                        analysis_result["analyses"]["logic_analysis"] = {
                            "status": "error",
                            "error": error_msg,
                            "agent_used": "logic_analyzer",
                            "step": 2
                        }
                        self.logger.warning(f"⚠️ Step 2: Logic analysis failed: {error_msg}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Step 2: Logic analysis exception: {str(e)}")
                    analysis_result["analyses"]["logic_analysis"] = {
                        "status": "error",
                        "error": str(e),
                        "agent_used": "logic_analyzer",
                        "step": 2
                    }
            else:
                self.logger.info(f"ℹ️ Step 2: Skipping logic analysis for type: {normalized_type}")
            
            # STEP 3: SEMANTIC ANALYSIS (VECTOR SEARCH)
            try:
                self.logger.info(f"🔄 Step 3: Running semantic analysis for {analysis_component_name}")
                vector_agent = self.get_agent("vector_index")
                
                # Ensure vector index is ready
                vector_ready = await self._ensure_vector_index_ready()
                if not vector_ready:
                    self.logger.warning(f"⚠️ Vector index not ready, skipping semantic analysis")
                    analysis_result["analyses"]["semantic_analysis"] = {
                        "status": "skipped",
                        "error": "Vector index not available",
                        "agent_used": "vector_index",
                        "step": 3
                    }
                else:
                    # Perform similarity search
                    similarity_result = await self._safe_agent_call(
                        vector_agent.search_similar_components,
                        analysis_component_name,
                        3
                    )
                    
                    # Perform semantic search
                    semantic_result = await self._safe_agent_call(
                        vector_agent.semantic_search,
                        f"{analysis_component_name} similar functionality",
                        2
                    )
                    
                    # Validate and normalize results
                    validated_similarity = self._validate_search_result(similarity_result)
                    validated_semantic = self._validate_search_result(semantic_result)
                    
                    if validated_similarity or validated_semantic:
                        analysis_result["analyses"]["semantic_analysis"] = {
                            "status": "success",
                            "data": {
                                "similar_components": validated_similarity,
                                "semantic_search": validated_semantic
                            },
                            "agent_used": "vector_index",
                            "completion_time": time.time() - start_time,
                            "step": 3
                        }
                        completed_count += 1
                        self.logger.info(f"✅ Step 3: Semantic analysis completed successfully")
                    else:
                        analysis_result["analyses"]["semantic_analysis"] = {
                            "status": "error",
                            "error": "No valid search results returned",
                            "agent_used": "vector_index",
                            "step": 3
                        }
                        self.logger.warning(f"⚠️ Step 3: Semantic analysis failed - no valid results")
                    
            except Exception as e:
                self.logger.error(f"❌ Step 3: Semantic analysis exception: {str(e)}")
                analysis_result["analyses"]["semantic_analysis"] = {
                    "status": "error",
                    "error": str(e),
                    "agent_used": "vector_index",
                    "step": 3
                }
            
            # STEP 4: DOCUMENTATION SUMMARY (NEW!)
            try:
                self.logger.info(f"🔄 Step 4: Generating documentation summary for {analysis_component_name}")
                doc_agent = self.get_agent("documentation")
                
                # Prepare analysis summary for documentation agent
                analysis_summary = self._prepare_analysis_summary(analysis_result)
                
                # Generate documentation based on component type
                if normalized_type == "field":
                    doc_result = await self._safe_agent_call(
                        doc_agent.generate_field_lineage_documentation,
                        analysis_component_name
                    )
                elif normalized_type in ["cobol", "program", "copybook", "jcl"]:
                    doc_result = await self._safe_agent_call(
                        doc_agent.generate_program_documentation,
                        analysis_component_name,
                        "markdown"
                    )
                else:
                    # Generate custom analysis summary
                    doc_result = await self._generate_analysis_summary_doc(
                        analysis_component_name, analysis_summary, doc_agent
                    )
                
                if doc_result and not doc_result.get('error'):
                    analysis_result["analyses"]["documentation_summary"] = {
                        "status": "success",
                        "data": doc_result,
                        "agent_used": "documentation",
                        "completion_time": time.time() - start_time,
                        "step": 4
                    }
                    completed_count += 1
                    self.logger.info(f"✅ Step 4: Documentation summary completed successfully")
                else:
                    error_msg = doc_result.get('error', 'No documentation generated') if doc_result else 'No documentation generated'
                    analysis_result["analyses"]["documentation_summary"] = {
                        "status": "error",
                        "error": error_msg,
                        "agent_used": "documentation",
                        "step": 4
                    }
                    self.logger.warning(f"⚠️ Step 4: Documentation summary failed: {error_msg}")
                    
            except Exception as e:
                self.logger.error(f"❌ Step 4: Documentation summary exception: {str(e)}")
                analysis_result["analyses"]["documentation_summary"] = {
                    "status": "error",
                    "error": str(e),
                    "agent_used": "documentation",
                    "step": 4
                }
            
            # Determine final status
            total_analyses = len(analysis_result["analyses"])
            if completed_count == total_analyses and total_analyses > 0:
                analysis_result["status"] = "completed"
                self.logger.info(f"🎉 All {completed_count} analyses completed successfully")
            elif completed_count > 0:
                analysis_result["status"] = "partial"
                self.logger.warning(f"⚠️ Partial completion: {completed_count}/{total_analyses} analyses succeeded")
            else:
                analysis_result["status"] = "failed"
                self.logger.error(f"❌ All analyses failed for {analysis_component_name}")
            
            # Add final metadata
            analysis_result["processing_metadata"].update({
                "end_time": time.time(),
                "total_duration_seconds": time.time() - start_time,
                "analyses_completed": completed_count,
                "analyses_total": total_analyses,
                "success_rate": (completed_count / total_analyses) * 100 if total_analyses > 0 else 0,
                "analysis_sequence": ["lineage_analysis", "logic_analysis", "semantic_analysis", "documentation_summary"],
                "component_name_cleaned": cleaned_component_name != original_component_name
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Component analysis system error: {str(e)}")
            return {
                "component_name": component_name,
                "status": "system_error",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "coordinator_type": "api_based_enhanced_with_docs"
            }

    def _clean_component_name(self, component_name: str) -> str:
        """CRITICAL FIX: Clean component names that may have been prefixed during upload"""
        import re
        
        # Handle common prefixing patterns during file upload
        # Pattern: tmpewmlf88a_component_name or similar temporary prefixes
        
        # Remove temporary file prefixes (tmp + random chars + underscore)
        cleaned = re.sub(r'^tmp[a-zA-Z0-9]+_', '', component_name)
        
        # Remove session/upload ID prefixes (numbers + underscore)
        cleaned = re.sub(r'^[0-9a-f]{8,}_', '', cleaned)
        
        # Remove UUID-like prefixes (hex patterns)
        cleaned = re.sub(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}_', '', cleaned)
        
        # Remove generic temp prefixes
        temp_patterns = [
            r'^temp_[0-9]+_',
            r'^upload_[0-9]+_', 
            r'^session_[a-zA-Z0-9]+_',
            r'^file_[a-zA-Z0-9]+_'
        ]
        
        for pattern in temp_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Remove file extensions if present
        cleaned = re.sub(r'\.(cbl|cob|copy|cpy|jcl|job|proc)$', '', cleaned, flags=re.IGNORECASE)
        
        # Log if we cleaned the name
        if cleaned != component_name:
            self.logger.info(f"🧹 Cleaned component name: '{component_name}' → '{cleaned}'")
        
        return cleaned


    def _prepare_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a summary of all analyses for documentation generation"""
        summary = {
            "component_name": analysis_result.get("cleaned_component_name", analysis_result.get("component_name")),
            "component_type": analysis_result.get("normalized_type"),
            "analysis_timestamp": analysis_result.get("analysis_timestamp"),
            "total_analyses": len(analysis_result.get("analyses", {})),
            "successful_analyses": len([a for a in analysis_result.get("analyses", {}).values() if a.get("status") == "success"]),
            "findings": {}
        }
        
        # Extract key findings from each analysis
        analyses = analysis_result.get("analyses", {})
        
        # Lineage findings
        if "lineage_analysis" in analyses and analyses["lineage_analysis"].get("status") == "success":
            lineage_data = analyses["lineage_analysis"].get("data", {})
            summary["findings"]["lineage"] = {
                "programs_found": len(lineage_data.get("programs_using", [])),
                "operations": lineage_data.get("operations", []),
                "lifecycle_stages": lineage_data.get("lifecycle_stages", [])
            }
        
        # Logic findings
        if "logic_analysis" in analyses and analyses["logic_analysis"].get("status") == "success":
            logic_data = analyses["logic_analysis"].get("data", {})
            summary["findings"]["logic"] = {
                "complexity_score": logic_data.get("complexity_score", 0),
                "dependencies": logic_data.get("dependencies", []),
                "business_rules": logic_data.get("business_rules", [])
            }
        
        # Semantic findings
        if "semantic_analysis" in analyses and analyses["semantic_analysis"].get("status") == "success":
            semantic_data = analyses["semantic_analysis"].get("data", {})
            summary["findings"]["semantic"] = {
                "similar_components": len(semantic_data.get("similar_components", [])),
                "semantic_matches": len(semantic_data.get("semantic_search", []))
            }
        
        return summary

    async def _generate_analysis_summary_doc(self, component_name: str, analysis_summary: Dict[str, Any], 
                                        doc_agent) -> Dict[str, Any]:
        """FIXED: Generate a readable analysis summary document"""
        try:
            # Create a structured summary for readable documentation
            findings_text = self._format_findings_as_text(analysis_summary.get('findings', {}))
            
            prompt = f"""
            Create a comprehensive business analysis summary for: {component_name}
            
            Component Type: {analysis_summary.get('component_type', 'Unknown')}
            Analysis Timestamp: {analysis_summary.get('analysis_timestamp', 'Unknown')}
            Total Analyses: {analysis_summary.get('total_analyses', 0)}
            Successful Analyses: {analysis_summary.get('successful_analyses', 0)}
            
            Key Findings:
            {findings_text}
            
            Write a professional executive summary that includes:
            
            1. **Executive Summary**
            - What this component does in business terms
            - Its importance to the organization
            
            2. **Key Findings**
            - Most important discoveries from the analysis
            - Critical dependencies and relationships
            
            3. **Component Characteristics**
            - Technical characteristics that matter to business
            - Performance and reliability indicators
            
            4. **Recommendations**
            - Actions to improve or maintain this component
            - Risk mitigation strategies
            
            5. **Next Steps**
            - Immediate actions required
            - Long-term considerations
            
            Write in clear, professional prose suitable for both technical and business audiences.
            Do not use JSON format. Use proper headings and paragraphs.
            Maximum 800 words.
            """
            
            # Use the documentation agent's enhanced API call method
            if hasattr(doc_agent, '_call_api_for_readable_analysis'):
                doc_content = await doc_agent._call_api_for_readable_analysis(prompt, max_tokens=1000)
            else:
                # Fallback to regular API call with enhanced prompt
                doc_content = await doc_agent._call_api_for_analysis(prompt, max_tokens=1000)
                # Clean the response if needed
                doc_content = self._ensure_readable_output(doc_content, component_name)
            
            return {
                "status": "success",
                "component_name": component_name,
                "documentation": doc_content,
                "format": "markdown",
                "analysis_summary": analysis_summary,
                "generation_timestamp": dt.now().isoformat(),
                "content_type": "readable_summary"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis summary doc: {e}")
            return {
                "status": "error",
                "error": str(e),
                "component_name": component_name,
                "fallback_summary": self._generate_fallback_summary(component_name, analysis_summary)
            }

    def _format_findings_as_text(self, findings: Dict[str, Any]) -> str:
        """Format findings dictionary as readable text"""
        text_parts = []
        
        if 'lineage' in findings:
            lineage = findings['lineage']
            programs = lineage.get('programs_found', 0)
            operations = len(lineage.get('operations', []))
            text_parts.append(f"Lineage Analysis: Found usage in {programs} programs with {operations} different operations")
        
        if 'logic' in findings:
            logic = findings['logic']
            complexity = logic.get('complexity_score', 0)
            dependencies = len(logic.get('dependencies', []))
            text_parts.append(f"Logic Analysis: Complexity score of {complexity:.1f} with {dependencies} dependencies")
        
        if 'semantic' in findings:
            semantic = findings['semantic']
            similar = semantic.get('similar_components', 0)
            matches = semantic.get('semantic_matches', 0)
            text_parts.append(f"Semantic Analysis: {similar} similar components found with {matches} semantic matches")
        
        return '\n'.join(text_parts) if text_parts else "Analysis completed successfully"

    def _ensure_readable_output(self, content: str, component_name: str) -> str:
        """Ensure the output is readable prose, not JSON"""
        if not content or len(content.strip()) < 10:
            return self._generate_fallback_summary(component_name, {})
        
        # Check if content is JSON-like and convert if needed
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            try:
                import json
                json_data = json.loads(content)
                # Extract readable content from JSON
                if isinstance(json_data, dict):
                    readable_parts = []
                    for key, value in json_data.items():
                        if isinstance(value, str) and len(value) > 20:
                            readable_parts.append(f"**{key.title()}:** {value}")
                    
                    if readable_parts:
                        content = '\n\n'.join(readable_parts)
                    else:
                        content = self._generate_fallback_summary(component_name, json_data)
            except:
                pass
        
        return content

    def _generate_fallback_summary(self, component_name: str, analysis_summary: Dict[str, Any]) -> str:
        """Generate a fallback readable summary"""
        summary = f"# Analysis Summary for {component_name}\n\n"
        
        summary += f"**Component:** {component_name}\n"
        summary += f"**Analysis Date:** {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"**Analysis Type:** Comprehensive Component Analysis\n\n"
        
        total_analyses = analysis_summary.get('total_analyses', 0)
        successful = analysis_summary.get('successful_analyses', 0)
        
        summary += "## Executive Summary\n\n"
        summary += f"This analysis examined {component_name} using {total_analyses} different analytical approaches, "
        summary += f"with {successful} analyses completing successfully. "
        
        if successful > 0:
            summary += "The component appears to be actively used within the system "
            summary += "and has identifiable patterns and dependencies.\n\n"
        else:
            summary += "Limited analysis data is available for this component.\n\n"
        
        summary += "## Key Findings\n\n"
        
        findings = analysis_summary.get('findings', {})
        if findings:
            for category, data in findings.items():
                if isinstance(data, dict) and data:
                    summary += f"**{category.title()} Analysis:** "
                    summary += f"Analysis completed with {len(data)} data points identified.\n"
        else:
            summary += "Component analysis completed. Detailed findings available in technical sections.\n"
        
        summary += "\n## Recommendations\n\n"
        summary += "- Review component usage patterns for optimization opportunities\n"
        summary += "- Ensure proper documentation is maintained\n"
        summary += "- Monitor component dependencies for system stability\n"
        
        return summary

    
    def _normalize_component_type(self, component_type: str) -> str:
        """Normalize component type for proper agent handling"""
        if not component_type:
            return "cobol"
        
        component_type_lower = component_type.lower()
        
        # Map file extensions and variations to standard types
        if component_type_lower in ['cbl', 'cob', 'cobol', 'program']:
            return "cobol"
        elif component_type_lower in ['copy', 'cpy', 'copybook']:
            return "copybook" 
        elif component_type_lower in ['jcl', 'job', 'proc']:
            return "jcl"
        elif component_type_lower in ['field', 'data_field', 'variable']:
            return "field"
        else:
            return "cobol"  # Safe default


    def _validate_search_result(self, result):
        """Validate and normalize search results"""
        try:
            if not result or result.get('error'):
                return []
            
            # If result is already a list, return as-is
            if isinstance(result, list):
                return result
            
            # If result is a dict, extract data
            if isinstance(result, dict):
                if 'data' in result:
                    data = result['data']
                    return data if isinstance(data, list) else []
                elif 'results' in result:
                    results = result['results']
                    return results if isinstance(results, list) else []
                elif 'matches' in result:
                    matches = result['matches']
                    return matches if isinstance(matches, list) else []
                else:
                    return [result]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Result validation failed: {e}")
            return []
        
    async def _ensure_agents_ready(self):
        """Ensure all required agents are loaded"""
        required_agents = ["lineage_analyzer", "logic_analyzer", "vector_index"]
        
        for agent_type in required_agents:
            if agent_type not in self.agents:
                try:
                    self.logger.info(f"Loading {agent_type} agent...")
                    self.get_agent(agent_type)  # This will create the agent
                    self.logger.info(f"✅ {agent_type} agent loaded")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {agent_type}: {e}")
                    raise RuntimeError(f"Required agent {agent_type} failed to load")

    async def _safe_agent_call(self, agent_method, *args, **kwargs):
        """Safely call agent method with proper async handling"""
        try:
            if not callable(agent_method):
                return {"error": "Method is not callable"}
            
            method_name = getattr(agent_method, '__name__', str(agent_method))
            self.logger.debug(f"🔧 Calling {method_name}")
            
            # Check if method is async
            import inspect
            if inspect.iscoroutinefunction(agent_method):
                result = await asyncio.wait_for(
                    agent_method(*args, **kwargs),
                    timeout=120
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: agent_method(*args, **kwargs)
                    ),
                    timeout=120
                )
            
            if result is None:
                return {"error": f"Method {method_name} returned None"}
            
            return result
            
        except asyncio.TimeoutError:
            return {"error": f"Method timed out after 120 seconds"}
        except Exception as e:
            return {"error": f"Method failed: {str(e)}"}

        
    async def _ensure_vector_index_ready(self) -> bool:
        """Ensure vector index is ready"""
        try:
            vector_agent = self.get_agent("vector_index")
            
            # Check if index has embeddings
            stats_result = await self._safe_agent_call(vector_agent.get_embedding_statistics)
            
            if stats_result and not stats_result.get('error'):
                total_embeddings = stats_result.get('total_embeddings', 0)
                if total_embeddings > 0:
                    self.logger.info(f"✅ Vector index ready with {total_embeddings} embeddings")
                    return True
            
            # Try to build index if not ready
            self.logger.info("🔄 Building vector index...")
            chunks = await self._get_processed_chunks()
            
            if chunks:
                build_result = await self._safe_agent_call(
                    vector_agent.create_embeddings_for_chunks,
                    chunks
                )
                
                if build_result and not build_result.get('error'):
                    self.logger.info(f"✅ Vector index built with {len(chunks)} chunks")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Vector index preparation failed: {e}")
            return False
    async def rebuild_vector_index(self) -> Dict[str, Any]:
        """FIXED: Rebuild vector index using correct method name"""
        try:
            vector_agent = self.get_agent("vector_index")
            
            # FIXED: Use the correct method name that exists in vector agent
            result = await self._safe_agent_call(vector_agent.rebuild_index)
            
            return {
                "status": "success" if result and not result.get('error') else "error",
                "result": result,
                "coordinator_type": "api_based_fixed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Vector index rebuild failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "coordinator_type": "api_based_fixed"
            }

    async def get_vector_index_stats(self) -> Dict[str, Any]:
        """FIXED: Get vector index statistics using correct method"""
        try:
            vector_agent = self.get_agent("vector_index")
            
            # FIXED: Use correct method name
            stats = await self._safe_agent_call(vector_agent.get_embedding_statistics)
            
            return {
                "status": "success" if stats and not stats.get('error') else "error",
                "stats": stats if stats else {},
                "coordinator_type": "api_based_fixed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Vector stats failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "coordinator_type": "api_based_fixed"
            }

    async def _get_processed_chunks(self):
        """Get processed chunks from database"""
        try:
            def get_chunks():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        SELECT program_name, chunk_id, chunk_type, content, metadata
                        FROM program_chunks
                        WHERE content IS NOT NULL AND content != '' AND length(content) > 10
                        ORDER BY created_timestamp DESC
                        LIMIT 500
                    """)
                    
                    rows = cursor.fetchall()
                    
                    chunks = []
                    for row in rows:
                        chunks.append({
                            'program_name': row[0],
                            'chunk_id': row[1],
                            'chunk_type': row[2],
                            'content': row[3][:1000],  # Limit content size
                            'metadata': row[4]
                        })
                    
                    return chunks
                    
                finally:
                    conn.close()
            
            chunks = await asyncio.get_event_loop().run_in_executor(None, get_chunks)
            self.logger.info(f"📊 Retrieved {len(chunks)} chunks from database")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get chunks: {e}")
            return []

    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """FIXED: Process chat query with proper error handling"""
        try:
            chat_agent = self.get_agent("chat_agent")
            result = await chat_agent.process_chat_query(query, conversation_history, **kwargs)
            
            self.stats["total_queries"] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chat query failed: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try rephrasing your question", "Check system status"],
                "coordinator_type": "api_based_fixed"
            }
    
    async def search_code_patterns(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """FIXED: Search code patterns with correct vector agent method"""
        try:
            vector_agent = self.get_agent("vector_index")
            
            # FIXED: Use advanced semantic search if available, otherwise fall back
            if hasattr(vector_agent, 'advanced_semantic_search'):
                results = await self._safe_agent_call(
                    vector_agent.advanced_semantic_search, 
                    query, 
                    limit
                )
            else:
                results = await self._safe_agent_call(
                    vector_agent.semantic_search, 
                    query, 
                    limit
                )
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_found": len(results) if results else 0,
                "coordinator_type": "api_based_fixed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pattern search failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "coordinator_type": "api_based_fixed"
            }
    
    # ==================== Helper Methods (Same as Original) ====================
    
    async def _ensure_file_stored_in_db(self, file_path: Path, result: Dict, file_type: str):
        """Store file processing result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO file_metadata 
                    (file_name, file_type, processing_status, last_modified)
                    VALUES (?, ?, ?, ?)
                """, (
                    file_path.name,
                    file_type,
                    result.get("status", "processed"),
                    dt.now().isoformat()
                ))
                
                cursor.execute("COMMIT")
                self.logger.debug(f"✅ Stored {file_path.name} in database")
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
                
            finally:
                conn.close()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store {file_path} in database: {str(e)}")
    
    async def _determine_component_type(self, component_name: str) -> str:
        """Determine component type with proper COBOL file extension handling"""
        
        # Check file extension first
        component_lower = component_name.lower()
        
        # Handle COBOL file extensions
        if any(component_lower.endswith(ext) for ext in ['.cbl', '.cob', '.cobol']):
            self.logger.info(f"✅ Detected COBOL program file: {component_name}")
            return "cobol"
        elif any(component_lower.endswith(ext) for ext in ['.copy', '.cpy', '.copybook']):
            self.logger.info(f"✅ Detected COBOL copybook file: {component_name}")
            return "copybook"
        elif any(component_lower.endswith(ext) for ext in ['.jcl', '.job', '.proc']):
            self.logger.info(f"✅ Detected JCL file: {component_name}")
            return "jcl"
        
        # Database-based detection
        try:
            def check_database():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                try:
                    # Check chunk types
                    cursor.execute("""
                        SELECT chunk_type, COUNT(*) as count 
                        FROM program_chunks 
                        WHERE program_name = ? OR program_name LIKE ?
                        GROUP BY chunk_type
                        ORDER BY count DESC
                    """, (component_name, f"%{component_name}%"))
                    
                    chunk_types = cursor.fetchall()
                    
                    if chunk_types:
                        chunk_type_names = [ct.lower() for ct, _ in chunk_types]
                        
                        if any('job' in ct for ct in chunk_type_names):
                            return "jcl"
                        elif any(ct in ['working_storage', 'procedure_division', 'data_division', 'identification_division'] for ct in chunk_type_names):
                            return "cobol"
                        elif any(ct in ['copybook', 'copy'] for ct in chunk_type_names):
                            return "copybook"
                    
                    # Check if it's a field
                    cursor.execute("""
                        SELECT COUNT(*) FROM program_chunks
                        WHERE (content LIKE ? OR metadata LIKE ?) AND chunk_type NOT IN ('file_header', 'comment')
                        LIMIT 1
                    """, (f"%{component_name}%", f"%{component_name}%"))
                    
                    field_count = cursor.fetchone()[0]
                    
                    if (field_count > 0 and component_name.isupper() and 
                        ('_' in component_name or '-' in component_name or len(component_name) <= 30) and
                        not any(component_lower.endswith(ext) for ext in ['.cbl', '.cob', '.copy', '.cpy'])):
                        return "field"
                    
                    return "cobol"  # Default
                    
                finally:
                    conn.close()
            
            # Run database check in executor
            result = await asyncio.get_event_loop().run_in_executor(None, check_database)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Component type determination failed: {e}")
            return "cobol"

    
    def get_health_status(self) -> Dict[str, Any]:
        """FIXED: Get coordinator health status with vector index info"""
        try:
            available_servers = len(self.load_balancer.get_available_servers())
            total_servers = len(self.load_balancer.servers)
            
            server_stats = {}
            for server in self.load_balancer.servers:
                try:
                    status_value = server.status.value if hasattr(server.status, 'value') else str(server.status)
                    
                    server_stats[server.config.name] = {
                        "endpoint": server.config.endpoint,
                        "status": status_value,
                        "active_requests": getattr(server, 'active_requests', 0),
                        "total_requests": getattr(server, 'total_requests', 0),
                        "success_rate": (
                            (server.successful_requests / server.total_requests * 100) 
                            if getattr(server, 'total_requests', 0) > 0 else 0
                        ),
                        "average_latency": getattr(server, 'average_latency', 0),
                        "available": server.is_available() if hasattr(server, 'is_available') else False
                    }
                except Exception as e:
                    server_stats[server.config.name] = {
                        "status": "error",
                        "error": str(e),
                        "available": False
                    }
            
            try:
                agent_status = self.get_agent_status()
                available_agent_types = self.list_available_agents()
                
                # FIXED: Add vector index status
                vector_index_status = "unknown"
                vector_embeddings_count = 0
                
                try:
                    if "vector_index" in self.agents:
                        vector_agent = self.agents["vector_index"]
                        if hasattr(vector_agent, 'faiss_index') and vector_agent.faiss_index:
                            vector_embeddings_count = vector_agent.faiss_index.ntotal
                            vector_index_status = "ready" if vector_embeddings_count > 0 else "empty"
                        else:
                            vector_index_status = "not_initialized"
                except Exception as ve:
                    vector_index_status = f"error: {str(ve)}"
                
            except Exception as e:
                agent_status = {"error": str(e)}
                available_agent_types = {}
                vector_index_status = "unknown"
                vector_embeddings_count = 0
            
            return {
                "status": "healthy" if available_servers > 0 else "unhealthy",
                "coordinator_type": "api_based_fixed",
                "selected_gpus": getattr(self, 'selected_gpus', []),
                "available_servers": available_servers,
                "total_servers": total_servers,
                "server_stats": server_stats,
                "active_agents": len(getattr(self, 'agents', {})),
                "agent_status": agent_status,
                "available_agent_types": available_agent_types,
                "vector_index_status": vector_index_status,
                "vector_embeddings_count": vector_embeddings_count,
                "stats": getattr(self, 'stats', {}),
                "uptime_seconds": time.time() - self.stats.get("start_time", time.time()),
                "database_available": os.path.exists(self.db_path),
                "load_balancing_strategy": self.config.load_balancing_strategy.value
            }
        except Exception as e:
            self.logger.error(f"Health status check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "coordinator_type": "api_based_fixed"
            }

    
    async def get_statistics(self) -> Dict[str, Any]:
        """FIXED: Get comprehensive system statistics with error handling"""
        try:
            # Get database stats
            database_stats = await self._get_database_stats()
            
            # Get server stats
            server_stats = []
            for server in self.load_balancer.servers:
                try:
                    server_stats.append({
                        "name": server.config.name,
                        "gpu_id": server.config.gpu_id,
                        "endpoint": server.config.endpoint,
                        "status": server.status.value,
                        "active_requests": server.active_requests,
                        "total_requests": server.total_requests,
                        "successful_requests": server.successful_requests,
                        "failed_requests": server.failed_requests,
                        "success_rate": (server.successful_requests / server.total_requests * 100) if server.total_requests > 0 else 0,
                        "average_latency": server.average_latency,
                        "consecutive_failures": server.consecutive_failures,
                        "available": server.is_available()
                    })
                except Exception as e:
                    server_stats.append({
                        "name": server.config.name,
                        "error": str(e)
                    })
            
            return {
                "system_stats": self.stats,
                "server_stats": server_stats,
                "database_stats": database_stats,
                "timestamp": dt.now().isoformat(),
                "coordinator_type": "api_based_fixed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {str(e)}")
            return {
                "system_stats": self.stats,
                "error": str(e),
                "coordinator_type": "api_based_fixed"
            }
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """FIXED: Get database statistics with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            tables = ['program_chunks', 'file_metadata', 'field_lineage', 'vector_embeddings']
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    stats[f"{table}_count"] = 0
            
            if os.path.exists(self.db_path):
                stats["database_size_bytes"] = os.path.getsize(self.db_path)
            else:
                stats["database_size_bytes"] = 0
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Database stats failed: {str(e)}")
            return {"error": str(e)}
    
    # ==================== Backwards Compatibility Methods ====================
    
    def cleanup(self):
        """FIXED: Cleanup method for backwards compatibility"""
        self.logger.info("🧹 Cleaning up FIXED API coordinator resources...")
        
        # Clean up agents
        for agent_type, agent in self.agents.items():
            try:
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                    self.logger.info(f"✅ Cleaned up {agent_type} agent")
            except Exception as e:
                self.logger.error(f"❌ Failed to cleanup {agent_type}: {e}")
        
        # Clear agent cache
        self.agents.clear()
        
        self.logger.info("✅ FIXED API Coordinator cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __repr__(self):
        return (f"APIOpulenceCoordinator("
                f"servers={len(self.load_balancer.servers)}, "
                f"agents={len(self.agents)}, "
                f"strategy={self.config.load_balancing_strategy.value}, "
                f"type=fixed)")

# ==================== Keep ALL Factory Functions Unchanged ====================

def create_api_coordinator_from_endpoints(gpu_endpoints: Dict[int, str]) -> APIOpulenceCoordinator:
    """Create API coordinator from GPU endpoints mapping"""
    config = APIOpulenceConfig.from_gpu_endpoints(gpu_endpoints)
    return APIOpulenceCoordinator(config)

def create_api_coordinator_from_config(
    model_servers: List[Dict[str, Any]],
    load_balancing_strategy: str = "round_robin",
    **kwargs
) -> APIOpulenceCoordinator:
    """FIXED: Create API coordinator from configuration"""
    server_configs = []
    for server_config in model_servers:
        server_configs.append(ModelServerConfig(
            endpoint=server_config["endpoint"],
            gpu_id=server_config["gpu_id"],
            name=server_config.get("name", f"gpu_{server_config['gpu_id']}"),
            max_concurrent_requests=server_config.get("max_concurrent_requests", 1),  # FIXED: Conservative
            timeout=server_config.get("timeout", 120)  # FIXED: Reduced timeout
        ))
    
    # FIXED: Apply conservative overrides to kwargs
    fixed_kwargs = kwargs.copy()
    fixed_kwargs.setdefault('connection_timeout', 30)  # FIXED: Shorter
    fixed_kwargs.setdefault('request_timeout', 120)   # FIXED: Shorter
    fixed_kwargs.setdefault('max_retries', 1)         # FIXED: Minimal
    fixed_kwargs.setdefault('circuit_breaker_threshold', 5)  # FIXED: Less tolerant
    
    config = APIOpulenceConfig(
        model_servers=server_configs,
        load_balancing_strategy=LoadBalancingStrategy(load_balancing_strategy),
        **fixed_kwargs
    )
    
    return APIOpulenceCoordinator(config)

def create_dual_gpu_coordinator_api(
    model_servers: List[Dict[str, Any]] = None,
    load_balancing_strategy: str = "round_robin"
) -> APIOpulenceCoordinator:
    """FIXED: Drop-in replacement for create_dual_gpu_coordinator using API"""
    if model_servers is None:
        # FIXED: Default to single working server with conservative settings
        model_servers = [
            {
                "endpoint": "http://171.201.3.165:8100", 
                "gpu_id": 2, 
                "name": "gpu_2", 
                "max_concurrent_requests": 1, 
                "timeout": 120  # FIXED: Reduced timeout
            }
        ]
    
    return create_api_coordinator_from_config(model_servers, load_balancing_strategy)

def get_global_api_coordinator() -> APIOpulenceCoordinator:
    """Get or create global API coordinator instance"""
    global _global_api_coordinator
    if _global_api_coordinator is None:
        _global_api_coordinator = create_dual_gpu_coordinator_api()
    return _global_api_coordinator

# Global coordinator instance
_global_api_coordinator: Optional[APIOpulenceCoordinator] = None

# ==================== Keep ALL Utility Functions Unchanged ====================

async def quick_file_processing_api(file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
    """Quick file processing using API coordinator"""
    coordinator = get_global_api_coordinator()
    await coordinator.initialize()
    try:
        return await coordinator.process_batch_files(file_paths, file_type)
    finally:
        await coordinator.shutdown()

async def quick_component_analysis_api(component_name: str, component_type: str = None) -> Dict[str, Any]:
    """Quick component analysis using API coordinator"""
    coordinator = get_global_api_coordinator()
    await coordinator.initialize()
    try:
        return await coordinator.analyze_component(component_name, component_type)
    finally:
        await coordinator.shutdown()

async def quick_chat_query_api(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Quick chat query using API coordinator"""
    coordinator = get_global_api_coordinator()
    await coordinator.initialize()
    try:
        return await coordinator.process_chat_query(query, conversation_history)
    finally:
        await coordinator.shutdown()

def get_system_status_api() -> Dict[str, Any]:
    """Get system status using API coordinator"""
    coordinator = get_global_api_coordinator()
    return coordinator.get_health_status()


# ==================== STREAMLIT SESSION STATE FIX ====================

def ensure_streamlit_session_state():
    """CRITICAL FIX: Ensure all session state variables are properly initialized"""
    import streamlit as st
    
    # AGENT_TYPES definition for session state
    STREAMLIT_AGENT_TYPES = [
        'code_parser', 'chat_agent', 'vector_index', 'data_loader',
        'lineage_analyzer', 'logic_analyzer', 'documentation', 'db2_comparator'
    ]
    
    defaults = {
        'chat_history': [],
        'processing_history': [],
        'uploaded_files': [],
        'file_analysis_results': {},
        'agent_status': {agent: {'status': 'unknown', 'last_used': None, 'total_calls': 0, 'errors': 0} 
                        for agent in STREAMLIT_AGENT_TYPES},  # FIXED: Proper agent_status initialization
        'model_servers': [],
        'coordinator': None,
        'debug_mode': False,
        'initialization_status': 'not_started',
        'import_error': None,
        'auto_refresh_gpu': False,
        'gpu_refresh_interval': 10,
        'current_query': '',
        'analysis_results': {},
        'dashboard_metrics': {
            'files_processed': 0,
            'queries_answered': 0,
            'components_analyzed': 0,
            'system_uptime': 0
        },
        'show_manual_config': False,
        'show_system_status': False,
        'show_chat_stats': False,
        'saved_conversations': {},
        'current_page': '🏠 Dashboard',
        'chat_response_mode': 'Detailed',
        'chat_include_context': True,
        'chat_max_history': 5,
        'auto_refresh_enabled': False,
        'refresh_interval': 10
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            
    # CRITICAL FIX: Ensure agent_status is always properly structured
    if not isinstance(st.session_state.get('agent_status'), dict):
        st.session_state.agent_status = {agent: {'status': 'unknown', 'last_used': None, 'total_calls': 0, 'errors': 0} 
                                        for agent in STREAMLIT_AGENT_TYPES}
    
    # Ensure all required agents are in agent_status
    for agent in STREAMLIT_AGENT_TYPES:
        if agent not in st.session_state.agent_status:
            st.session_state.agent_status[agent] = {
                'status': 'unknown', 
                'last_used': None, 
                'total_calls': 0, 
                'errors': 0
            }

async def example_usage():
    """Example of how to use the FIXED API coordinator"""
    
    # FIXED model server configuration
    model_servers = [
        {
            "endpoint": "http://171.201.3.165:8100", 
            "gpu_id": 2, 
            "name": "gpu_2",
            "max_concurrent_requests": 1,  # Only 1 request at a time
            "timeout": 120  # FIXED: 2 minutes
        }
    ]
    
    # Create FIXED coordinator
    coordinator = create_api_coordinator_from_config(
        model_servers=model_servers,
        load_balancing_strategy="round_robin",
        max_retries=1,  # Minimal retries
        connection_pool_size=1,  # Very small
        request_timeout=120,  # 2 minutes
        circuit_breaker_threshold=5  # Less tolerant
    )
    
    # Initialize
    await coordinator.initialize()
    
    try:
        # Process files (same interface as before)
        file_paths = [Path("example.cbl"), Path("example.jcl")]
        result = await coordinator.process_batch_files(file_paths)
        print(f"Processed {result['files_processed']} files")
        
        # Analyze component (same interface as before)
        analysis = await coordinator.analyze_component("CUSTOMER-RECORD", "field")
        print(f"Analysis status: {analysis['status']}")
        
        # Chat query (same interface as before)
        chat_result = await coordinator.process_chat_query("What is the purpose of CUSTOMER-RECORD?")
        print(f"Chat response: {chat_result['response']}")
        
        # Get health status
        health = coordinator.get_health_status()
        print(f"System health: {health['status']}")
        
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())