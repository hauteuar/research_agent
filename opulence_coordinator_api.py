#!/usr/bin/env python3
"""
API-Based Opulence Coordinator System
Replaces direct GPU management with HTTP API calls to model servers
Keeps existing agent modules unchanged - just changes the orchestration layer
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
    from agents.code_parser_agent import CodeParserAgent
except ImportError:
    CodeParserAgent = None

try:
    from agents.chat_agent import OpulenceChatAgent
except ImportError:
    OpulenceChatAgent = None

try:
    from agents.vector_index_agent import VectorIndexAgent
except ImportError:
    VectorIndexAgent = None

try:
    from agents.data_loader_agent import DataLoaderAgent
except ImportError:
    DataLoaderAgent = None

try:
    from agents.lineage_analyzer_agent import LineageAnalyzerAgent
except ImportError:
    LineageAnalyzerAgent = None

try:
    from agents.logic_analyzer_agent import LogicAnalyzerAgent
except ImportError:
    LogicAnalyzerAgent = None

try:
    from agents.documentation_agent import DocumentationAgent
except ImportError:
    DocumentationAgent = None

try:
    from agents.db2_comparator_agent import DB2ComparatorAgent
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
    gpu_id: int
    name: str = ""
    max_concurrent_requests: int = 10
    timeout: int = 300
    
    def __post_init__(self):
        if not self.name:
            self.name = f"gpu_{self.gpu_id}"

@dataclass
class APIOpulenceConfig:
    """Configuration for API-based Opulence Coordinator"""
    # Model server endpoints
    model_servers: List[ModelServerConfig] = field(default_factory=list)
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_BUSY
    
    # Connection pool settings
    connection_pool_size: int = 50
    connection_timeout: int = 30
    request_timeout: int = 300
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Health checking
    health_check_interval: int = 30
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Database
    db_path: str = "opulence_api_data.db"
    
    # Backwards compatibility settings
    max_tokens: int = 1024
    temperature: float = 0.1
    auto_cleanup: bool = True
    
    @classmethod
    def from_gpu_endpoints(cls, gpu_endpoints: Dict[int, str]) -> 'APIOpulenceConfig':
        """Create config from GPU ID to endpoint mapping"""
        servers = []
        for gpu_id, endpoint in gpu_endpoints.items():
            servers.append(ModelServerConfig(
                endpoint=endpoint,
                gpu_id=gpu_id,
                name=f"gpu_{gpu_id}"
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
            if time.time() - self.circuit_breaker_open_time > 60:
                self.status = ModelServerStatus.UNKNOWN
                return True
            return False
        
        return (self.status == ModelServerStatus.HEALTHY and 
                self.active_requests < self.config.max_concurrent_requests)
    
    def record_success(self, latency: float):
        """Record successful request"""
        self.successful_requests += 1
        self.total_latency += latency
        self.consecutive_failures = 0
        
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
        """Get server by GPU ID"""
        for server in self.servers:
            if server.config.gpu_id == gpu_id:
                return server
        return None

# ==================== API Client for Model Servers ====================

class ModelServerClient:
    """HTTP client for calling model servers"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"{__name__}.ModelServerClient")
        
    async def initialize(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=10,
            ttl_dns_cache=300,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=self.config.connection_timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'OpulenceCoordinator/1.0.0'
            }
        )
        
        self.logger.info("Model server client initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def call_generate(self, server: ModelServer, prompt: str, 
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call model server generate endpoint"""
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        params = params or {}
        request_data = {
            "prompt": prompt,
            "max_tokens": params.get("max_tokens", 512),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "stream": params.get("stream", False),
            **params
        }
        
        server.active_requests += 1
        server.total_requests += 1
        start_time = time.time()
        
        try:
            for attempt in range(self.config.max_retries + 1):
                try:
                    generate_url = urljoin(server.config.endpoint, "/generate")
                    
                    async with self.session.post(
                        generate_url,
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=server.config.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Record success
                            latency = time.time() - start_time
                            server.record_success(latency)
                            
                            # Add metadata
                            result["server_used"] = server.config.name
                            result["gpu_id"] = server.config.gpu_id
                            result["latency"] = latency
                            
                            return result
                        
                        elif response.status == 503:
                            # Service unavailable - don't retry on same server
                            error_text = await response.text()
                            raise aiohttp.ClientError(f"Service unavailable: {error_text}")
                        
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                
                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    if attempt < self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** attempt if self.config.exponential_backoff else 1)
                        self.logger.warning(f"Request failed, retrying in {delay:.1f}s: {e}")
                        await asyncio.sleep(delay)
                        continue
                    raise
        
        except Exception as e:
            server.record_failure()
            
            # Check if circuit breaker should open
            if server.should_open_circuit(self.config.circuit_breaker_threshold):
                server.open_circuit()
            
            raise RuntimeError(f"Model server call failed: {e}")
        
        finally:
            server.active_requests -= 1
    
    async def health_check(self, server: ModelServer) -> bool:
        """Check server health"""
        try:
            health_url = urljoin(server.config.endpoint, "/health")
            
            async with self.session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    server.status = ModelServerStatus.HEALTHY
                    return True
                else:
                    server.status = ModelServerStatus.UNHEALTHY
                    return False
                    
        except Exception:
            server.status = ModelServerStatus.UNHEALTHY
            return False

# ==================== API-Compatible Engine Context ====================

class APIEngineContext:
    """Provides engine-like interface for existing agents using API calls"""
    
    def __init__(self, coordinator, preferred_gpu_id: int = None):
        self.coordinator = coordinator
        self.preferred_gpu_id = preferred_gpu_id
        self.logger = logging.getLogger(f"{__name__}.APIEngineContext")
    
    async def generate(self, prompt: str, sampling_params, request_id: str = None):
        """Generate text via API (compatible with vLLM interface)"""
        # Convert sampling_params to API parameters
        params = {
            "max_tokens": getattr(sampling_params, 'max_tokens', 512),
            "temperature": getattr(sampling_params, 'temperature', 0.7),
            "top_p": getattr(sampling_params, 'top_p', 0.9),
            "top_k": getattr(sampling_params, 'top_k', 50),
            "frequency_penalty": getattr(sampling_params, 'frequency_penalty', 0.0),
            "presence_penalty": getattr(sampling_params, 'presence_penalty', 0.0),
            "stop": getattr(sampling_params, 'stop', None),
            "seed": getattr(sampling_params, 'seed', None),
        }
        
        # Call API
        result = await self.coordinator.call_model_api(prompt, params, self.preferred_gpu_id)
        
        # Convert API response to vLLM-compatible format
        class MockOutput:
            def __init__(self, text: str, finish_reason: str):
                self.text = text
                self.finish_reason = finish_reason
                self.token_ids = []  # Not available from API
        
        class MockRequestOutput:
            def __init__(self, result: Dict[str, Any]):
                self.outputs = [MockOutput(result.get('text', ''), result.get('finish_reason', 'stop'))]
                self.finished = True
                self.prompt_token_ids = []  # Not available from API
        
        # Yield result (to match async generator interface)
        yield MockRequestOutput(result)

# ==================== API-Based Coordinator ====================

class APIOpulenceCoordinator:
    """API-based Opulence Coordinator - orchestrates existing agents through API calls"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.client = ModelServerClient(config)
        self.db_path = config.db_path
        
        # Agent storage - keep existing agents unchanged
        self.agents = {}
        
        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "total_api_calls": 0,
            "avg_response_time": 0,
            "start_time": time.time(),
            "coordinator_type": "api_based"
        }
        
        # For backwards compatibility - selected GPUs are now API endpoints
        self.selected_gpus = [server.config.gpu_id for server in self.load_balancer.servers]
        
        self.logger = logging.getLogger(f"{__name__}.APIOpulenceCoordinator")
        self.logger.info(f"API Coordinator initialized with servers: {[s.config.name for s in self.load_balancer.servers]}")
    
    async def initialize(self):
        """Initialize the coordinator"""
        await self.client.initialize()
        
        # Start health checking
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("API Coordinator initialized successfully")
    
    async def shutdown(self):
        """Shutdown the coordinator"""
        if self.health_check_task:
            self.health_check_task.cancel()
        
        await self.client.close()
        self.logger.info("API Coordinator shut down")
    
    def _init_database(self):
        """Initialize database (same as original)"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            
            cursor = conn.cursor()
            
            # Create tables (same as original coordinator)
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
                    embedding_id TEXT,
                    file_hash TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
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
    
    async def _health_check_loop(self):
        """Background health checking loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check all servers
                for server in self.load_balancer.servers:
                    await self.client.health_check(server)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def call_model_api(self, prompt: str, params: Dict[str, Any] = None, 
                           preferred_gpu_id: int = None) -> Dict[str, Any]:
        """Call model API with load balancing"""
        # Try preferred GPU first
        if preferred_gpu_id is not None:
            server = self.load_balancer.get_server_by_gpu_id(preferred_gpu_id)
            if server and server.is_available():
                try:
                    return await self.client.call_generate(server, prompt, params)
                except Exception as e:
                    self.logger.warning(f"Preferred GPU {preferred_gpu_id} failed: {e}")
        
        # Use load balancer
        server = self.load_balancer.select_server()
        if not server:
            raise RuntimeError("No available model servers")
        
        result = await self.client.call_generate(server, prompt, params)
        self.stats["total_api_calls"] += 1
        return result
    
    # ==================== Existing Agent Interface (Backwards Compatibility) ====================
    
    def get_agent(self, agent_type: str):
        """Get agent - creates instances that use API calls instead of direct GPU access"""
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)
        return self.agents[agent_type]
    
    def _create_agent(self, agent_type: str):
        """Create agent using API-based engine context"""
        self.logger.info(f"🔗 Creating {agent_type} agent (API-based)")
        
        # Create agents but inject API-based engine context
        if agent_type == "code_parser" and CodeParserAgent:
            agent = CodeParserAgent(
                llm_engine=None,  # Will be replaced with API context
                db_path=self.db_path,
                gpu_id=None,
                coordinator=self
            )
            # Monkey patch the get_engine_context method
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "vector_index" and VectorIndexAgent:
            agent = VectorIndexAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None,
                coordinator=self
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "data_loader" and DataLoaderAgent:
            agent = DataLoaderAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None,
                coordinator=self
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "lineage_analyzer" and LineageAnalyzerAgent:
            agent = LineageAnalyzerAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None,
                coordinator=self
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "logic_analyzer" and LogicAnalyzerAgent:
            agent = LogicAnalyzerAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None,
                coordinator=self
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "documentation" and DocumentationAgent:
            agent = DocumentationAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None,
                coordinator=self
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "db2_comparator" and DB2ComparatorAgent:
            agent = DB2ComparatorAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None,
                max_rows=10000,
                coordinator=self
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        elif agent_type == "chat_agent" and OpulenceChatAgent:
            agent = OpulenceChatAgent(
                coordinator=self,
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=None
            )
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            return agent
            
        else:
            raise ValueError(f"Unknown or unavailable agent type: {agent_type}")
    
    def _create_engine_context_for_agent(self, agent):
        """Create API-based engine context for agent"""
        @asynccontextmanager
        async def api_engine_context():
            # Create API-based engine context
            api_context = APIEngineContext(self, getattr(agent, 'gpu_id', None))
            try:
                yield api_context
            finally:
                # No cleanup needed for API calls
                pass
        
        return api_engine_context
    
    # ==================== Existing Interface Methods (Unchanged) ====================
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """Process files - same interface as original"""
        start_time = time.time()
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"🚀 Processing {total_files} files using API-based agents")
        
        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"📄 Processing file {i+1}/{total_files}: {file_path.name}")
                
                # Get appropriate agent (same logic as original)
                if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                    agent = self.get_agent("code_parser")
                elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                    agent = self.get_agent("data_loader")
                else:
                    agent = self.get_agent("code_parser")
                
                # Process with API-based agent
                result = await agent.process_file(file_path)
                await self._ensure_file_stored_in_db(file_path, result, file_type)
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path}: {str(e)}")
                results.append({
                    "status": "error",
                    "file": str(file_path),
                    "error": str(e)
                })
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["total_files_processed"] += total_files
        
        return {
            "status": "success",
            "files_processed": total_files,
            "processing_time": processing_time,
            "results": results,
            "servers_used": [s.config.name for s in self.load_balancer.servers]
        }
    
    async def analyze_component(self, component_name: str, component_type: str = None) -> Dict[str, Any]:
        """Analyze component - same interface as original"""
        start_time = time.time()
        
        try:
            if not component_type or component_type == "auto-detect":
                component_type = await self._determine_component_type(component_name)
            
            analysis_result = {
                "component_name": component_name,
                "component_type": component_type,
                "analysis_timestamp": dt.now().isoformat(),
                "status": "in_progress",
                "analyses": {},
                "processing_metadata": {
                    "start_time": start_time,
                    "coordinator_type": "api_based"
                }
            }
            
            completed_count = 0
            
            # Lineage Analysis (same logic, different execution)
            try:
                self.logger.info(f"🔄 Running lineage analysis for {component_name}")
                lineage_agent = self.get_agent("lineage_analyzer")
                
                if component_type == "field":
                    lineage_result = await lineage_agent.analyze_field_lineage(component_name)
                else:
                    lineage_result = await lineage_agent.analyze_full_lifecycle(component_name, component_type)
                
                analysis_result["analyses"]["lineage_analysis"] = {
                    "status": "success",
                    "data": lineage_result,
                    "agent_used": "lineage_analyzer",
                    "completion_time": time.time() - start_time
                }
                completed_count += 1
                
            except Exception as e:
                self.logger.error(f"❌ Lineage analysis failed: {str(e)}")
                analysis_result["analyses"]["lineage_analysis"] = {
                    "status": "error",
                    "error": str(e),
                    "agent_used": "lineage_analyzer"
                }
            
            # Logic Analysis (same logic, different execution)
            if component_type in ["program", "cobol", "copybook"]:
                try:
                    self.logger.info(f"🔄 Running logic analysis for {component_name}")
                    logic_agent = self.get_agent("logic_analyzer")
                    
                    if component_type in ["program", "cobol"]:
                        logic_result = await logic_agent.analyze_program(component_name)
                    else:
                        logic_result = await logic_agent.find_dependencies(component_name)
                    
                    analysis_result["analyses"]["logic_analysis"] = {
                        "status": "success",
                        "data": logic_result,
                        "agent_used": "logic_analyzer",
                        "completion_time": time.time() - start_time
                    }
                    completed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Logic analysis failed: {str(e)}")
                    analysis_result["analyses"]["logic_analysis"] = {
                        "status": "error",
                        "error": str(e),
                        "agent_used": "logic_analyzer"
                    }
            
            # Semantic Analysis (same logic, different execution)
            try:
                self.logger.info(f"🔄 Running semantic analysis for {component_name}")
                vector_agent = self.get_agent("vector_index")
                
                similarity_result = await vector_agent.search_similar_components(component_name, top_k=10)
                semantic_result = await vector_agent.semantic_search(f"{component_name} similar functionality", top_k=5)
                
                analysis_result["analyses"]["semantic_analysis"] = {
                    "status": "success",
                    "data": {
                        "similar_components": similarity_result,
                        "semantic_search": semantic_result
                    },
                    "agent_used": "vector_index",
                    "completion_time": time.time() - start_time
                }
                completed_count += 1
                
            except Exception as e:
                self.logger.error(f"❌ Semantic analysis failed: {str(e)}")
                analysis_result["analyses"]["semantic_analysis"] = {
                    "status": "error",
                    "error": str(e),
                    "agent_used": "vector_index"
                }
            
            # Determine final status
            total_analyses = len(analysis_result["analyses"])
            if completed_count == total_analyses:
                analysis_result["status"] = "completed"
            elif completed_count > 0:
                analysis_result["status"] = "partial"
            else:
                analysis_result["status"] = "failed"
            
            # Add final metadata
            analysis_result["processing_metadata"].update({
                "end_time": time.time(),
                "total_duration_seconds": time.time() - start_time,
                "analyses_completed": completed_count,
                "analyses_total": total_analyses,
                "success_rate": (completed_count / total_analyses) * 100 if total_analyses > 0 else 0
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Component analysis failed: {str(e)}")
            return {
                "component_name": component_name,
                "status": "system_error",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "coordinator_type": "api_based"
            }
    
    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process chat query - same interface as original"""
        try:
            chat_agent = self.get_agent("chat_agent")
            result = await chat_agent.process_chat_query(query, conversation_history)
            
            self.stats["total_queries"] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chat query failed: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try rephrasing your question", "Check system status"],
                "coordinator_type": "api_based"
            }
    
    async def search_code_patterns(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search code patterns - same interface as original"""
        try:
            vector_agent = self.get_agent("vector_index")
            results = await vector_agent.semantic_search(query, top_k=limit)
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_found": len(results),
                "coordinator_type": "api_based"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pattern search failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "coordinator_type": "api_based"
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
        """Determine component type from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count 
                FROM program_chunks 
                WHERE program_name = ?
                GROUP BY chunk_type
                ORDER BY count DESC
            """, (component_name,))
            
            chunk_types = cursor.fetchall()
            
            if chunk_types:
                if any('job' in ct.lower() for ct, _ in chunk_types):
                    return "jcl"
                elif any(ct in ['working_storage', 'procedure_division', 'data_division'] for ct, _ in chunk_types):
                    return "program"
                else:
                    return "program"
            
            # Check if it's a field
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks
                WHERE content LIKE ? OR metadata LIKE ?
                LIMIT 1
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            if cursor.fetchone()[0] > 0:
                if component_name.isupper() and ('_' in component_name or len(component_name) <= 20):
                    return "field"
            
            return "program"  # Default
            
        except Exception as e:
            self.logger.error(f"❌ Component type determination failed: {e}")
            return "program"
        finally:
            conn.close()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get coordinator health status"""
        available_servers = len(self.load_balancer.get_available_servers())
        total_servers = len(self.load_balancer.servers)
        
        server_stats = {}
        for server in self.load_balancer.servers:
            server_stats[server.config.name] = {
                "status": server.status.value,
                "active_requests": server.active_requests,
                "total_requests": server.total_requests,
                "success_rate": (server.successful_requests / server.total_requests * 100) if server.total_requests > 0 else 0,
                "average_latency": server.average_latency,
                "available": server.is_available()
            }
        
        return {
            "status": "healthy" if available_servers > 0 else "unhealthy",
            "coordinator_type": "api_based",
            "selected_gpus": self.selected_gpus,  # For backwards compatibility
            "available_servers": available_servers,
            "total_servers": total_servers,
            "server_stats": server_stats,
            "active_agents": len(self.agents),
            "stats": self.stats,
            "uptime_seconds": time.time() - self.stats["start_time"],
            "database_available": os.path.exists(self.db_path),
            "load_balancing_strategy": self.config.load_balancing_strategy.value
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get database stats
            database_stats = await self._get_database_stats()
            
            # Get server stats
            server_stats = []
            for server in self.load_balancer.servers:
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
            
            return {
                "system_stats": self.stats,
                "server_stats": server_stats,
                "database_stats": database_stats,
                "timestamp": dt.now().isoformat(),
                "coordinator_type": "api_based"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {str(e)}")
            return {
                "system_stats": self.stats,
                "error": str(e),
                "coordinator_type": "api_based"
            }
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
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
        """Cleanup method for backwards compatibility"""
        self.logger.info("🧹 Cleaning up API coordinator resources...")
        
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
        
        self.logger.info("✅ API Coordinator cleanup completed")
    
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
                f"strategy={self.config.load_balancing_strategy.value})")

# ==================== Factory Functions ====================

def create_api_coordinator_from_endpoints(gpu_endpoints: Dict[int, str]) -> APIOpulenceCoordinator:
    """Create API coordinator from GPU endpoints mapping"""
    config = APIOpulenceConfig.from_gpu_endpoints(gpu_endpoints)
    return APIOpulenceCoordinator(config)

def create_api_coordinator_from_config(
    model_servers: List[Dict[str, Any]],
    load_balancing_strategy: str = "least_busy",
    **kwargs
) -> APIOpulenceCoordinator:
    """Create API coordinator from configuration"""
    server_configs = []
    for server_config in model_servers:
        server_configs.append(ModelServerConfig(
            endpoint=server_config["endpoint"],
            gpu_id=server_config["gpu_id"],
            name=server_config.get("name", f"gpu_{server_config['gpu_id']}"),
            max_concurrent_requests=server_config.get("max_concurrent_requests", 10),
            timeout=server_config.get("timeout", 300)
        ))
    
    config = APIOpulenceConfig(
        model_servers=server_configs,
        load_balancing_strategy=LoadBalancingStrategy(load_balancing_strategy),
        **kwargs
    )
    
    return APIOpulenceCoordinator(config)

# ==================== Drop-in Replacement Functions ====================

def create_dual_gpu_coordinator_api(
    model_servers: List[Dict[str, Any]] = None,
    load_balancing_strategy: str = "least_busy"
) -> APIOpulenceCoordinator:
    """Drop-in replacement for create_dual_gpu_coordinator using API"""
    if model_servers is None:
        # Default to localhost model servers
        model_servers = [
            {"endpoint": "http://localhost:8000", "gpu_id": 1, "name": "gpu_1"},
            {"endpoint": "http://localhost:8001", "gpu_id": 2, "name": "gpu_2"}
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

# ==================== Utility Functions ====================

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

# ==================== Example Usage ====================

async def example_usage():
    """Example of how to use the API coordinator"""
    
    # Define your model server endpoints
    model_servers = [
        {"endpoint": "http://gpu-server-1:8000", "gpu_id": 1, "name": "gpu_1"},
        {"endpoint": "http://gpu-server-2:8001", "gpu_id": 2, "name": "gpu_2"},
        {"endpoint": "http://gpu-server-3:8002", "gpu_id": 3, "name": "gpu_3"}
    ]
    
    # Create coordinator
    coordinator = create_api_coordinator_from_config(
        model_servers=model_servers,
        load_balancing_strategy="least_busy"
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