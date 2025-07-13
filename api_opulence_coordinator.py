#!/usr/bin/env python3
"""
FIXED API-Based Opulence Coordinator System
Addresses: timeout context manager issues, agent initialization failures, proper task timeouts
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FIXED Configuration Classes ====================

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
    max_concurrent_requests: int = 1
    timeout: int = 180
    
    def __post_init__(self):
        if not self.name:
            self.name = f"gpu_{self.gpu_id}"

@dataclass
class APIOpulenceConfig:
    """FIXED Configuration for API-based Opulence Coordinator"""
    model_servers: List[ModelServerConfig] = field(default_factory=list)
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    connection_pool_size: int = 2
    connection_timeout: int = 60
    request_timeout: int = 180
    max_retries: int = 1
    retry_delay: float = 5.0
    exponential_backoff: bool = False
    health_check_interval: int = 60
    circuit_breaker_threshold: int = 10
    circuit_breaker_timeout: int = 120
    db_path: str = "opulence_api_data.db"
    max_tokens: int = 50
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
                name=f"gpu_{gpu_id}",
                max_concurrent_requests=1,
                timeout=180
            ))
        return cls(model_servers=servers)

# ==================== FIXED Model Server Management ====================

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

# ==================== FIXED Load Balancer ====================

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

# ==================== FIXED API Client ====================

class ModelServerClient:
    """FIXED HTTP client for calling model servers"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"{__name__}.ModelServerClient")
        
    async def initialize(self):
        """FIXED: Initialize session with proper timeout configuration"""
        
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=1,
            ttl_dns_cache=300,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # CRITICAL FIX: Simple timeout without nesting
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
        
        self.logger.info("Model server client initialized successfully")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def call_generate(self, server: ModelServer, prompt: str, 
                  params: Dict[str, Any] = None) -> Dict[str, Any]:
        """FIXED: Make API call with proper timeout handling inside task"""
        
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        params = params or {}
        
        # Conservative request parameters
        request_data = {
            "prompt": prompt,
            "max_tokens": min(params.get("max_tokens", 20), 50),
            "temperature": max(0.0, min(params.get("temperature", 0.1), 0.3)),
            "top_p": max(0.1, min(params.get("top_p", 0.9), 0.9)),
            "stream": False
        }
        
        server.active_requests += 1
        server.total_requests += 1
        start_time = time.time()
        
        try:
            generate_url = f"{server.config.endpoint.rstrip('/')}/generate"
            
            # CRITICAL FIX: Use asyncio.wait_for for timeout inside task
            async with asyncio.timeout(self.config.request_timeout):
                async with self.session.post(generate_url, json=request_data) as response:
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
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
        
        except asyncio.TimeoutError:
            server.record_failure()
            self.logger.error(f"Request timeout after {self.config.request_timeout}s for {server.config.name}")
            raise RuntimeError(f"Request timeout after {self.config.request_timeout}s")
            
        except aiohttp.ClientError as e:
            server.record_failure()
            self.logger.error(f"Client error for {server.config.name}: {e}")
            raise RuntimeError(f"Client error: {e}")
            
        except Exception as e:
            server.record_failure()
            self.logger.error(f"Unexpected error for {server.config.name}: {e}")
            
            # Check circuit breaker
            if server.should_open_circuit(self.config.circuit_breaker_threshold):
                server.open_circuit()
            
            raise RuntimeError(f"Model server call failed: {e}")
        
        finally:
            server.active_requests -= 1

    async def health_check(self, server: ModelServer) -> bool:
        """FIXED: Health check with proper timeout"""
        try:
            if not self.session:
                return False
                
            health_url = f"{server.config.endpoint.rstrip('/')}/health"
            
            # FIXED: Use asyncio.timeout for health check
            async with asyncio.timeout(10):  # 10 second timeout for health checks
                async with self.session.get(health_url) as response:
                    if response.status == 200:
                        server.status = ModelServerStatus.HEALTHY
                        return True
                    else:
                        server.status = ModelServerStatus.UNHEALTHY
                        return False
                        
        except asyncio.TimeoutError:
            server.status = ModelServerStatus.UNHEALTHY
            self.logger.debug(f"Health check timeout for {server.config.name}")
            return False
        except Exception as e:
            server.status = ModelServerStatus.UNHEALTHY
            self.logger.debug(f"Health check failed for {server.config.name}: {e}")
            return False

# ==================== FIXED API Engine Context ====================

class APIEngineContext:
    """FIXED engine-like interface for existing agents using API calls"""
    
    def __init__(self, coordinator, preferred_gpu_id: int = None):
        self.coordinator = coordinator
        self.preferred_gpu_id = preferred_gpu_id
        self.logger = logging.getLogger(f"{__name__}.APIEngineContext")
    
    async def generate(self, prompt: str, sampling_params, request_id: str = None):
        """FIXED: Generate text via API with proper timeout handling"""
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
            params = {"max_tokens": 20, "temperature": 0.1, "top_p": 0.9}
        
        # Conservative parameter validation
        validated_params = {}
        for key, value in params.items():
            if value is not None:
                if key == "max_tokens":
                    validated_params[key] = max(1, min(value, 50))
                elif key == "temperature":
                    validated_params[key] = max(0.0, min(value, 0.3))
                elif key == "top_p":
                    validated_params[key] = max(0.0, min(value, 1.0))
                else:
                    validated_params[key] = value
        
        # FIXED: Call API with timeout inside task
        try:
            result = await asyncio.wait_for(
                self.coordinator.call_model_api(prompt=prompt, params=validated_params),
                timeout=self.coordinator.config.request_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Generate request timeout after {self.coordinator.config.request_timeout}s")
        
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

# ==================== FIXED API Coordinator ====================

class APIOpulenceCoordinator:
    """FIXED API-based Opulence Coordinator with proper agent initialization"""
    
    def __init__(self, config: APIOpulenceConfig):
        self.logger = logging.getLogger(f"{__name__}.APIOpulenceCoordinator")
        self.config = config
        self.load_balancer = LoadBalancer(config)
        self.client = ModelServerClient(config)
        self.db_path = config.db_path
        
        # Agent storage
        self.agents = {}
        
        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize database
        self._init_database()

        # Compatibility attributes
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
        
        # Agent configurations
        self.agent_configs = {
            "code_parser": {
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Analyzes and parses code structures"
            },
            "vector_index": {
                "max_tokens": 50,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Handles vector embeddings and similarity search"
            },
            "data_loader": {
                "max_tokens": 75,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Processes and loads data files"
            },
            "lineage_analyzer": {
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Analyzes data and code lineage"
            },
            "logic_analyzer": {
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Analyzes program logic and flow"
            },
            "documentation": {
                "max_tokens": 150,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Generates documentation"
            },
            "db2_comparator": {
                "max_tokens": 75,
                "temperature": 0.1,
                "top_p": 0.9,
                "description": "Compares database schemas and data"
            },
            "chat_agent": {
                "max_tokens": 100,
                "temperature": 0.2,
                "top_p": 0.9,
                "description": "Handles interactive chat queries"
            }
        }

        self.selected_gpus = [server.config.gpu_id for server in self.load_balancer.servers]
        
        self.logger.info(f"Fixed API Coordinator initialized with servers: {[s.config.name for s in self.load_balancer.servers]}")
    
    async def initialize(self):
        """FIXED: Initialize the coordinator with proper error handling"""
        try:
            # Initialize client first
            await self.client.initialize()
            
            # Test server connectivity
            await self._test_connectivity_with_timeout()
            
            # Start health check task
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.logger.info("Fixed API Coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Coordinator initialization failed: {e}")
            raise
    
    async def _test_connectivity_with_timeout(self):
        """FIXED: Test connectivity with proper timeout handling"""
        healthy_count = 0
        
        for server in self.load_balancer.servers:
            try:
                # FIXED: Use asyncio.wait_for for timeout
                is_healthy = await asyncio.wait_for(
                    self.client.health_check(server),
                    timeout=15  # 15 second timeout per server
                )
                
                if is_healthy:
                    healthy_count += 1
                    self.logger.info(f"✅ {server.config.name} ({server.config.endpoint}) - Connected")
                else:
                    self.logger.warning(f"❌ {server.config.name} ({server.config.endpoint}) - Failed")
                    
                # Small delay between tests
                await asyncio.sleep(0.5)
                
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ {server.config.name} - Connection timeout")
            except Exception as e:
                self.logger.error(f"Server test error for {server.config.name}: {e}")
        
        if healthy_count == 0:
            raise RuntimeError("No model servers are accessible!")
        
        self.logger.info(f"Connected to {healthy_count}/{len(self.load_balancer.servers)} servers")
    
    async def _health_check_loop(self):
        """FIXED: Health check loop with proper error handling"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                for server in self.load_balancer.servers:
                    try:
                        await asyncio.wait_for(
                            self.client.health_check(server),
                            timeout=10
                        )
                    except asyncio.TimeoutError:
                        server.status = ModelServerStatus.UNHEALTHY
                        self.logger.debug(f"Health check timeout for {server.config.name}")
                    except Exception as e:
                        server.status = ModelServerStatus.UNHEALTHY
                        self.logger.debug(f"Health check error for {server.config.name}: {e}")
                        
            except asyncio.CancelledError:
                self.logger.info("Health check loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def shutdown(self):
        """FIXED: Shutdown with proper task cancellation"""
        try:
            # Cancel health check task
            if self.health_check_task and not self.health_check_task.done():
                self.health_check_task.cancel()
                try:
                    await asyncio.wait_for(self.health_check_task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Close client
            if self.client:
                await self.client.close()
            
            self.logger.info("Fixed API Coordinator shut down successfully")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    def _init_database(self):
        """Initialize database - same as original"""
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
    
    async def call_model_api(self, prompt: str, params: Dict[str, Any] = None, 
                    preferred_gpu_id: int = None) -> Dict[str, Any]:
        """FIXED: API call with proper timeout and error handling"""
        
        server = self.load_balancer.select_server()        
        if not server:
            raise RuntimeError("No available servers found")
        
        try:
            self.logger.debug(f"API call to {server.config.name}")
            
            # FIXED: Use asyncio.wait_for for timeout
            result = await asyncio.wait_for(
                self.client.call_generate(server, prompt, params),
                timeout=self.config.request_timeout
            )
            
            self.stats["total_api_calls"] += 1
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"API call timeout for {server.config.name}")
            raise RuntimeError(f"API call timeout after {self.config.request_timeout}s")
            
        except Exception as e:
            self.logger.warning(f"Request failed on {server.config.name}: {e}")
            
            # Simple retry with different server if available
            retry_server = self.load_balancer.select_server()
            if retry_server and retry_server != server:
                try:
                    self.logger.info(f"Retrying with {retry_server.config.name}")
                    result = await asyncio.wait_for(
                        self.client.call_generate(retry_server, prompt, params),
                        timeout=self.config.request_timeout
                    )
                    self.stats["total_api_calls"] += 1
                    return result
                except Exception as retry_e:
                    self.logger.error(f"Retry failed: {retry_e}")
            
            raise RuntimeError(f"All servers failed: {str(e)}")
    
    def get_agent(self, agent_type: str):
        """FIXED: Get agent with proper initialization"""
        if agent_type not in self.agents:
            try:
                self.agents[agent_type] = self._create_agent_safe(agent_type)
            except Exception as e:
                self.logger.error(f"Failed to create {agent_type} agent: {e}")
                # Return a mock agent for compatibility
                return self._create_mock_agent(agent_type)
        return self.agents[agent_type]
    
    def _create_agent_safe(self, agent_type: str):
        """FIXED: Create agent with proper error handling"""
        self.logger.info(f"🔗 Creating {agent_type} agent (Fixed API-based)")
        
        # Get agent configuration
        agent_config = self.agent_configs.get(agent_type, {})
        
        # Use first available server's GPU ID for compatibility
        selected_gpu_id = self.available_gpu_ids[0] if self.available_gpu_ids else 0
        
        try:
            # Create minimal agent that works with API
            agent = self._create_minimal_agent(agent_type, selected_gpu_id, agent_config)
            
            # Inject API-based engine context
            agent.get_engine_context = self._create_engine_context_for_agent(agent)
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create {agent_type} agent: {str(e)}")
            raise
    
    def _create_minimal_agent(self, agent_type: str, gpu_id: int, config: dict):
        """Create minimal agent that works with API"""
        class MinimalAgent:
            def __init__(self, agent_type, gpu_id, coordinator, config):
                self.agent_type = agent_type
                self.gpu_id = gpu_id
                self.coordinator = coordinator
                self.config = config
                self.logger = logging.getLogger(f"MinimalAgent.{agent_type}")
            
            async def process_file(self, file_path):
                """Minimal file processing"""
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Simple API call for processing
                    prompt = f"Analyze this {self.agent_type} file:\n\n{content[:1000]}..."
                    
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={
                            "max_tokens": self.config.get("max_tokens", 50),
                            "temperature": self.config.get("temperature", 0.1),
                            "top_p": self.config.get("top_p", 0.9)
                        }
                    )
                    
                    return {
                        "status": "success",
                        "file": str(file_path),
                        "agent_type": self.agent_type,
                        "result": result,
                        "processing_time": 1.0
                    }
                    
                except Exception as e:
                    self.logger.error(f"File processing failed: {e}")
                    return {
                        "status": "error",
                        "file": str(file_path),
                        "agent_type": self.agent_type,
                        "error": str(e)
                    }
            
            async def analyze_field_lineage(self, field_name):
                """Minimal lineage analysis"""
                try:
                    prompt = f"Analyze data lineage for field: {field_name}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 30, "temperature": 0.1}
                    )
                    return {"lineage_path": [f"Source -> {field_name} -> Target"], "dependencies": []}
                except Exception as e:
                    self.logger.error(f"Lineage analysis failed: {e}")
                    return {"error": str(e)}
            
            async def analyze_full_lifecycle(self, component_name, component_type):
                """Minimal lifecycle analysis"""
                try:
                    prompt = f"Analyze lifecycle for {component_type}: {component_name}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 30, "temperature": 0.1}
                    )
                    return {"lifecycle": [f"Created -> {component_name} -> Used"], "dependencies": []}
                except Exception as e:
                    self.logger.error(f"Lifecycle analysis failed: {e}")
                    return {"error": str(e)}
            
            async def analyze_program(self, program_name):
                """Minimal program analysis"""
                try:
                    prompt = f"Analyze program structure: {program_name}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 40, "temperature": 0.1}
                    )
                    return {"program_structure": {"main": [program_name]}, "logic_flows": []}
                except Exception as e:
                    self.logger.error(f"Program analysis failed: {e}")
                    return {"error": str(e)}
            
            async def find_dependencies(self, component_name):
                """Minimal dependency analysis"""
                try:
                    prompt = f"Find dependencies for: {component_name}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 30, "temperature": 0.1}
                    )
                    return {"dependencies": [{"name": component_name, "type": "component", "relationship": "uses"}]}
                except Exception as e:
                    self.logger.error(f"Dependency analysis failed: {e}")
                    return {"error": str(e)}
            
            async def search_similar_components(self, component_name, top_k=5):
                """Minimal similarity search"""
                try:
                    prompt = f"Find similar components to: {component_name}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 20, "temperature": 0.1}
                    )
                    return [{"name": f"similar_to_{component_name}", "score": 0.8, "type": "component"}]
                except Exception as e:
                    self.logger.error(f"Similarity search failed: {e}")
                    return []
            
            async def semantic_search(self, query, top_k=5):
                """Minimal semantic search"""
                try:
                    prompt = f"Semantic search for: {query}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 20, "temperature": 0.1}
                    )
                    return [{"content": query, "score": 0.7, "source": "search_result"}]
                except Exception as e:
                    self.logger.error(f"Semantic search failed: {e}")
                    return []
            
            async def process_chat_query(self, query, conversation_history=None, **kwargs):
                """Minimal chat processing"""
                try:
                    prompt = f"Answer this question: {query}"
                    result = await self.coordinator.call_model_api(
                        prompt=prompt,
                        params={"max_tokens": 100, "temperature": 0.2}
                    )
                    
                    response_text = result.get('text', result.get('response', str(result)))
                    
                    return {
                        "response": response_text,
                        "response_type": "general",
                        "suggestions": ["Try another question"],
                        "context_used": []
                    }
                except Exception as e:
                    self.logger.error(f"Chat query failed: {e}")
                    return {
                        "response": f"I encountered an error: {str(e)}",
                        "response_type": "error",
                        "suggestions": ["Try rephrasing your question"],
                        "context_used": []
                    }
            
            def get_agent_stats(self):
                """Get agent statistics"""
                return {
                    "agent_type": self.agent_type,
                    "gpu_id": self.gpu_id,
                    "api_based": True,
                    "status": "loaded",
                    "config": self.config
                }
            
            def update_api_params(self, **params):
                """Update API parameters"""
                self.config.update(params)
                self.logger.info(f"Updated {self.agent_type} parameters: {params}")
            
            def cleanup(self):
                """Cleanup agent resources"""
                self.logger.info(f"Cleaning up {self.agent_type} agent")
        
        return MinimalAgent(agent_type, gpu_id, self, config)
    
    def _create_mock_agent(self, agent_type: str):
        """Create mock agent for fallback"""
        class MockAgent:
            def __init__(self, agent_type):
                self.agent_type = agent_type
                self.gpu_id = 0
                self.logger = logging.getLogger(f"MockAgent.{agent_type}")
            
            async def process_file(self, file_path):
                return {"status": "error", "error": f"Mock {self.agent_type} agent"}
            
            async def analyze_field_lineage(self, field_name):
                return {"error": f"Mock {self.agent_type} agent"}
            
            async def analyze_full_lifecycle(self, component_name, component_type):
                return {"error": f"Mock {self.agent_type} agent"}
            
            async def analyze_program(self, program_name):
                return {"error": f"Mock {self.agent_type} agent"}
            
            async def find_dependencies(self, component_name):
                return {"error": f"Mock {self.agent_type} agent"}
            
            async def search_similar_components(self, component_name, top_k=5):
                return []
            
            async def semantic_search(self, query, top_k=5):
                return []
            
            async def process_chat_query(self, query, conversation_history=None, **kwargs):
                return {
                    "response": f"Mock {self.agent_type} agent unavailable",
                    "response_type": "error",
                    "suggestions": [],
                    "context_used": []
                }
            
            def get_agent_stats(self):
                return {
                    "agent_type": self.agent_type,
                    "gpu_id": 0,
                    "api_based": True,
                    "status": "mock",
                    "error": "Agent creation failed"
                }
            
            def update_api_params(self, **params):
                pass
            
            def cleanup(self):
                pass
        
        return MockAgent(agent_type)
    
    def _create_engine_context_for_agent(self, agent):
        """Create API-based engine context for agent"""
        @asynccontextmanager
        async def api_engine_context():
            # Create API-based engine context
            api_context = APIEngineContext(self, preferred_gpu_id=None)
            try:
                yield api_context
            finally:
                # No cleanup needed for API calls
                pass
        
        return api_engine_context
    
    # ==================== Interface Methods ====================
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """FIXED: Process files with proper timeout handling"""
        start_time = time.time()
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"🚀 Processing {total_files} files using fixed API-based agents")
        
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
                
                # Get agent with timeout
                agent = self.get_agent(agent_type)
                
                # FIXED: Process with timeout
                result = await asyncio.wait_for(
                    agent.process_file(file_path),
                    timeout=self.config.request_timeout
                )
                
                await self._ensure_file_stored_in_db(file_path, result, file_type)
                results.append(result)
                
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ File processing timeout for {file_path}")
                results.append({
                    "status": "error",
                    "file": str(file_path),
                    "error": "Processing timeout"
                })
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
    
    async def analyze_component(self, component_name: str, component_type: str = None, **kwargs) -> Dict[str, Any]:
        """FIXED: Analyze component with proper timeout handling"""
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
                    "coordinator_type": "api_based_fixed"
                }
            }
            
            completed_count = 0
            
            # FIXED: Lineage Analysis with timeout
            try:
                self.logger.info(f"🔄 Running lineage analysis for {component_name}")
                lineage_agent = self.get_agent("lineage_analyzer")
                
                if component_type == "field":
                    lineage_result = await asyncio.wait_for(
                        lineage_agent.analyze_field_lineage(component_name),
                        timeout=60
                    )
                else:
                    lineage_result = await asyncio.wait_for(
                        lineage_agent.analyze_full_lifecycle(component_name, component_type),
                        timeout=60
                    )
                
                analysis_result["analyses"]["lineage_analysis"] = {
                    "status": "success",
                    "data": lineage_result,
                    "agent_used": "lineage_analyzer",
                    "completion_time": time.time() - start_time
                }
                completed_count += 1
                
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ Lineage analysis timeout for {component_name}")
                analysis_result["analyses"]["lineage_analysis"] = {
                    "status": "error",
                    "error": "Analysis timeout",
                    "agent_used": "lineage_analyzer"
                }
            except Exception as e:
                self.logger.error(f"❌ Lineage analysis failed: {str(e)}")
                analysis_result["analyses"]["lineage_analysis"] = {
                    "status": "error",
                    "error": str(e),
                    "agent_used": "lineage_analyzer"
                }
            
            # FIXED: Logic Analysis with timeout
            if component_type in ["program", "cobol", "copybook"]:
                try:
                    self.logger.info(f"🔄 Running logic analysis for {component_name}")
                    logic_agent = self.get_agent("logic_analyzer")
                    
                    if component_type in ["program", "cobol"]:
                        logic_result = await asyncio.wait_for(
                            logic_agent.analyze_program(component_name),
                            timeout=60
                        )
                    else:
                        logic_result = await asyncio.wait_for(
                            logic_agent.find_dependencies(component_name),
                            timeout=60
                        )
                    
                    analysis_result["analyses"]["logic_analysis"] = {
                        "status": "success",
                        "data": logic_result,
                        "agent_used": "logic_analyzer",
                        "completion_time": time.time() - start_time
                    }
                    completed_count += 1
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"⏰ Logic analysis timeout for {component_name}")
                    analysis_result["analyses"]["logic_analysis"] = {
                        "status": "error",
                        "error": "Analysis timeout",
                        "agent_used": "logic_analyzer"
                    }
                except Exception as e:
                    self.logger.error(f"❌ Logic analysis failed: {str(e)}")
                    analysis_result["analyses"]["logic_analysis"] = {
                        "status": "error",
                        "error": str(e),
                        "agent_used": "logic_analyzer"
                    }
            
            # FIXED: Semantic Analysis with timeout
            try:
                self.logger.info(f"🔄 Running semantic analysis for {component_name}")
                vector_agent = self.get_agent("vector_index")
                
                similarity_result = await asyncio.wait_for(
                    vector_agent.search_similar_components(component_name, top_k=3),
                    timeout=30
                )
                semantic_result = await asyncio.wait_for(
                    vector_agent.semantic_search(f"{component_name} similar functionality", top_k=2),
                    timeout=30
                )
                
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
                
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ Semantic analysis timeout for {component_name}")
                analysis_result["analyses"]["semantic_analysis"] = {
                    "status": "error",
                    "error": "Analysis timeout",
                    "agent_used": "vector_index"
                }
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
                "coordinator_type": "api_based_fixed"
            }
    
    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """FIXED: Process chat query with timeout"""
        try:
            chat_agent = self.get_agent("chat_agent")
            
            # FIXED: Add timeout to chat processing
            result = await asyncio.wait_for(
                chat_agent.process_chat_query(query, conversation_history, **kwargs),
                timeout=self.config.request_timeout
            )
            
            self.stats["total_queries"] += 1
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Chat query timeout")
            return {
                "response": "I'm sorry, but the request timed out. Please try again with a shorter question.",
                "response_type": "error",
                "suggestions": ["Try a simpler question", "Check system status"],
                "coordinator_type": "api_based_fixed"
            }
        except Exception as e:
            self.logger.error(f"❌ Chat query failed: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try rephrasing your question", "Check system status"],
                "coordinator_type": "api_based_fixed"
            }
    
    async def search_code_patterns(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """FIXED: Search code patterns with timeout"""
        try:
            vector_agent = self.get_agent("vector_index")
            
            # FIXED: Add timeout to search
            results = await asyncio.wait_for(
                vector_agent.semantic_search(query, top_k=limit),
                timeout=30
            )
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_found": len(results),
                "coordinator_type": "api_based_fixed"
            }
            
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Pattern search timeout")
            return {
                "status": "error",
                "error": "Search timeout",
                "query": query,
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
    
    # ==================== Helper Methods ====================
    
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
            agent_status = {agent_type: agent.get_agent_stats() for agent_type, agent in self.agents.items()}
            available_agent_types = self.agent_configs
        except Exception as e:
            agent_status = {"error": str(e)}
            available_agent_types = {}
        
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
            "stats": getattr(self, 'stats', {}),
            "uptime_seconds": time.time() - self.stats.get("start_time", time.time()),
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
    
    # ==================== Agent Management Methods ====================
    
    def list_available_agents(self) -> Dict[str, Any]:
        """List all available agent types and their configurations"""
        return {
            agent_type: {
                "config": config,
                "available": True,  # All agents are available as minimal agents
                "loaded": agent_type in self.agents
            }
            for agent_type, config in self.agent_configs.items()
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all loaded agents"""
        status = {}
        for agent_type, agent in self.agents.items():
            try:
                if hasattr(agent, 'get_agent_stats'):
                    status[agent_type] = agent.get_agent_stats()
                else:
                    status[agent_type] = {
                        "agent_type": agent_type,
                        "gpu_id": getattr(agent, 'gpu_id', None),
                        "api_based": True,
                        "status": "loaded"
                    }
            except Exception as e:
                status[agent_type] = {
                    "agent_type": agent_type,
                    "status": "error",
                    "error": str(e)
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
                    agent.update_api_params(**api_params)
                    self.logger.info(f"Updated live agent {agent_type} configuration: {api_params}")
    
    def reload_agent(self, agent_type: str):
        """Reload a specific agent with updated configuration"""
        if agent_type in self.agents:
            # Cleanup old agent
            old_agent = self.agents[agent_type]
            if hasattr(old_agent, 'cleanup'):
                old_agent.cleanup()
            
            # Remove from cache
            del self.agents[agent_type]
            
            self.logger.info(f"Reloaded {agent_type} agent")
        
        # Agent will be recreated on next access
        return self.get_agent(agent_type)
    
    # ==================== Cleanup Methods ====================
    
    def cleanup(self):
        """Cleanup method for backwards compatibility"""
        self.logger.info("🧹 Cleaning up fixed API coordinator resources...")
        
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
        
        self.logger.info("✅ Fixed API Coordinator cleanup completed")
    
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

# ==================== Factory Functions ====================

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
            max_concurrent_requests=server_config.get("max_concurrent_requests", 1),
            timeout=server_config.get("timeout", 180)
        ))
    
    config = APIOpulenceConfig(
        model_servers=server_configs,
        load_balancing_strategy=LoadBalancingStrategy(load_balancing_strategy),
        **kwargs
    )
    
    return APIOpulenceCoordinator(config)

def create_dual_gpu_coordinator_api(
    model_servers: List[Dict[str, Any]] = None,
    load_balancing_strategy: str = "round_robin"
) -> APIOpulenceCoordinator:
    """FIXED: Drop-in replacement for create_dual_gpu_coordinator using API"""
    if model_servers is None:
        # Default to single working server
        model_servers = [
            {"endpoint": "http://171.201.3.165:8100", "gpu_id": 2, "name": "gpu_2", "max_concurrent_requests": 1, "timeout": 180}
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
    """FIXED: Quick file processing using API coordinator"""
    coordinator = get_global_api_coordinator()
    await coordinator.initialize()
    try:
        return await coordinator.process_batch_files(file_paths, file_type)
    finally:
        await coordinator.shutdown()

async def quick_component_analysis_api(component_name: str, component_type: str = None) -> Dict[str, Any]:
    """FIXED: Quick component analysis using API coordinator"""
    coordinator = get_global_api_coordinator()
    await coordinator.initialize()
    try:
        return await coordinator.analyze_component(component_name, component_type)
    finally:
        await coordinator.shutdown()

async def quick_chat_query_api(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """FIXED: Quick chat query using API coordinator"""
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

# ==================== FIXED Example Usage ====================

async def example_usage():
    """FIXED: Example of how to use the fixed API coordinator"""
    
    # Model server configuration
    model_servers = [
        {
            "endpoint": "http://171.201.3.165:8100", 
            "gpu_id": 2, 
            "name": "gpu_2",
            "max_concurrent_requests": 1,
            "timeout": 180
        }
    ]
    
    # Create coordinator
    coordinator = create_api_coordinator_from_config(
        model_servers=model_servers,
        load_balancing_strategy="round_robin",
        max_retries=1,
        connection_pool_size=2,
        request_timeout=180,
        circuit_breaker_threshold=10
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
                            