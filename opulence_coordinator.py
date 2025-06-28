# opulence_coordinator.py
"""
Opulence - Deep Research Mainframe Agent Coordinator with Dynamic GPU Allocation
Handles dynamic GPU distribution, agent orchestration, and parallel processing
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path
import json
import uuid
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager

import streamlit as st
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import faiss
import chromadb
import pandas as pd

# Import our agents
from agents.code_parser_agent import CompleteEnhancedCodeParserAgent as CodeParserAgent
from agents.vector_index_agent import VectorIndexAgent
  
from agents.data_loader_agent import DataLoaderAgent
from agents.lineage_analyzer_agent import LineageAnalyzerAgent
from agents.logic_analyzer_agent import LogicAnalyzerAgent
from agents.documentation_agent import DocumentationAgent
from agents.db2_comparator_agent import DB2ComparatorAgent
from utils.gpu_manager import ImprovedDynamicGPUManager, SafeGPUContext
from gpu_force_fix import EnhancedGPUForcer
from utils.dynamic_config_manager import DynamicConfigManager, get_dynamic_config, GPUConfig
from utils.health_monitor import HealthMonitor
from utils.cache_manager import CacheManager

def _ensure_airgap_environment(self):
    """Ensure no external connections are possible"""
    import os
    import socket
    
    # Set environment variables to disable external connections
    os.environ.update({
        'NO_PROXY': '*',
        'DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
        'REQUESTS_CA_BUNDLE': '',
        'CURL_CA_BUNDLE': '',
        'SSL_VERIFY': 'false',
        'PYTHONHTTPSVERIFY': '0'
    })
    
    # Block socket connections
    original_socket = socket.socket
    def blocked_socket(*args, **kwargs):
        raise OSError("Network connections disabled in airgap mode")
    socket.socket = blocked_socket
    
    # Block requests library
    try:
        import requests
        def blocked_request(*args, **kwargs):
            raise requests.exceptions.ConnectionError("External connections disabled")
        requests.request = blocked_request
        requests.get = blocked_request
        requests.post = blocked_request
    except ImportError:
        pass

@dataclass
class OpulenceConfig:
    """Configuration for Opulence system"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 4096
    temperature: float = 0.1
    total_gpu_count: int = 4  # Changed to 4 GPUs
    max_processing_time: int = 900  # 15 minutes
    batch_size: int = 32
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600  # 1 hour
    memory_threshold: float = 0.85
    utilization_threshold: float = 80.0

class DynamicOpulenceCoordinator:
    """Enhanced coordinator with dynamic GPU allocation"""
    
    def __init__(self, config: OpulenceConfig = None):
        # Initialize configuration manager first
        self._ensure_airgap_environment()
        self.config_manager = DynamicConfigManager()
        
        # Use provided config or create from config manager
        if config is None:
            runtime_config = self.config_manager.create_runtime_config()
            self.config = OpulenceConfig(**{k: v for k, v in runtime_config.items() 
                                          if k in OpulenceConfig.__dataclass_fields__})
        else:
            self.config = config
            
        self.logger = self._setup_logging()
        
        # Get GPU configuration from config manager
        gpu_config = self.config_manager.get_gpu_config()
        
        # Initialize dynamic GPU manager with config
        self.gpu_manager = ImprovedDynamicGPUManager(
            total_gpu_count=gpu_config.total_gpu_count,
            memory_threshold=gpu_config.memory_threshold,
            utilization_threshold=gpu_config.utilization_threshold
        )
        
        self.health_monitor = HealthMonitor()
        self._request_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent LLM requests
        self._last_request_time = 0
        self._min_request_interval = 0.5  # Minimum 500ms between requests
        self._active_llm_requests = 0
        
        # Get cache configuration
        cache_ttl = self.config_manager.get("system.cache_ttl", 3600)
        self.cache_manager = CacheManager(cache_ttl)
        
        # Initialize SQLite database
        self.db_path = "opulence_data.db"
        self._init_database()
        
        # LLM engine pool - create engines dynamically
        self.llm_engine_pool = {}
        self.engine_lock = asyncio.Lock()
        
        # Agent instances - will be created with dynamic GPU allocation
        self.agents = {}
        
        # Processing statistics
        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "avg_response_time": 0,
            "cache_hit_rate": 0,
            "gpu_utilization": {},
            "dynamic_allocations": 0,
            "failed_allocations": 0
        }
        
        self.logger.info("Dynamic Opulence Coordinator initialized with configuration management")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('opulence.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def _safe_llm_generate(self, engine: AsyncLLMEngine, prompt: str, sampling_params) -> str:
        """Safely generate with LLM to prevent request flooding"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        async with self._request_semaphore:
            try:
                self._active_llm_requests += 1
                self._last_request_time = time.time()
                
                request_id = str(uuid.uuid4())
                
                # Use async generator properly
                result_generator = engine.generate(prompt, sampling_params, request_id=request_id)
                
                async for result in result_generator:
                    if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                        return result.outputs[0].text.strip()
                    break
                
                return ""
                
            except Exception as e:
                self.logger.error(f"Safe LLM generation failed: {e}")
                return ""
            finally:
                self._active_llm_requests -= 1

    # FIX 12: Enhanced database initialization in coordinator
    def _init_database(self):
        """Initialize SQLite database with required tables and proper configuration"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # FIX 13: Set proper SQLite configuration for async use
            conn.execute("PRAGMA journal_mode=WAL")  # Better for concurrent access
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            conn.execute("PRAGMA temp_store=memory")  # Use memory for temp storage
            
            cursor = conn.cursor()
            
            # Create tables with proper indexes
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
                    chunk_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    embedding_id TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, chunk_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_program_chunks_name ON program_chunks(program_name);
                CREATE INDEX IF NOT EXISTS idx_program_chunks_type ON program_chunks(chunk_type);
                CREATE INDEX IF NOT EXISTS idx_program_chunks_id ON program_chunks(chunk_id);
                
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    operation TEXT,
                    duration REAL,
                    gpu_used INTEGER,
                    status TEXT,
                    details TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_processing_stats_timestamp ON processing_stats(timestamp);
                
                CREATE TABLE IF NOT EXISTS gpu_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    agent_type TEXT,
                    gpu_id INTEGER,
                    preferred_gpu INTEGER,
                    allocation_success BOOLEAN,
                    duration REAL,
                    workload_type TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_gpu_allocations_timestamp ON gpu_allocations(timestamp);
                
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
            
            # FIX 14: Explicit commit for table creation
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully with proper configuration")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    async def get_or_create_llm_engine(self, gpu_id: int, force_reload: bool = False) -> AsyncLLMEngine:
        """Enhanced LLM engine creation with improved error handling"""
        async with self.engine_lock:
            engine_key = f"gpu_{gpu_id}"
            
            # Return existing engine if available and not forcing reload
            if engine_key in self.llm_engine_pool and not force_reload:
                # Test if engine is still working
                try:
                    # Quick test
                    test_params = SamplingParams(temperature=0.1, max_tokens=5)
                    test_generator = self.llm_engine_pool[engine_key].generate("test", test_params)
                    async for result in test_generator:
                        break  # Just test if it works
                    self.logger.info(f"Reusing verified LLM engine on GPU {gpu_id}")
                    return self.llm_engine_pool[engine_key]
                except Exception as e:
                    self.logger.warning(f"Existing engine on GPU {gpu_id} failed test: {e}")
                    # Remove failed engine and create new one
                    del self.llm_engine_pool[engine_key]
            
            # Use SafeGPUContext for engine creation
            try:
                with SafeGPUContext(
                    self.gpu_manager, 
                    f"llm_engine_{engine_key}", 
                    preferred_gpu=gpu_id,
                    cleanup_on_exit=False
                ) as allocated_gpu:
                    
                    if allocated_gpu != gpu_id:
                        if allocated_gpu is None:
                            raise RuntimeError(f"Failed to allocate GPU {gpu_id} for LLM engine")
                        else:
                            self.logger.warning(f"GPU {gpu_id} not available, using GPU {allocated_gpu}")
                            gpu_id = allocated_gpu
                            engine_key = f"gpu_{gpu_id}"
                    
                    # Check memory before creation
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    free_gb = memory_info['free_gb']
                    
                    if free_gb < 1.5:
                        raise RuntimeError(f"Insufficient memory on GPU {gpu_id}: {free_gb:.1f}GB free")
                    
                    self.logger.info(f"Creating LLM engine on GPU {gpu_id} with {free_gb:.1f}GB available")
                    
                    # Force GPU environment safely
                    success = EnhancedGPUForcer.safe_force_gpu_environment(
                        gpu_id, cleanup_first=True, verify_success=True
                    )
                    
                    if not success:
                        raise RuntimeError(f"Failed to force GPU environment for GPU {gpu_id}")
                    
                    # Create conservative engine args
                    engine_args = EnhancedGPUForcer.create_conservative_vllm_args(
                        self.config.model_name,
                        self.config.max_tokens
                    )
                    
                    # Create engine
                    engine = AsyncLLMEngine.from_engine_args(engine_args)
                    self.llm_engine_pool[engine_key] = engine
                    
                    # Verify final state
                    final_memory = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    used_gb = free_gb - final_memory['free_gb']
                    
                    self.logger.info(f"✅ LLM engine created on GPU {gpu_id}. "
                                f"Used {used_gb:.1f}GB, {final_memory['free_gb']:.1f}GB remaining")
                    
                    return engine
                    
            except Exception as e:
                self.logger.error(f"Failed to create LLM engine on GPU {gpu_id}: {e}")
                
                # Cleanup on failure
                if engine_key in self.llm_engine_pool:
                    del self.llm_engine_pool[engine_key]
                
                # Perform cleanup
                EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
                
                raise
    
    @asynccontextmanager
    async def get_agent_with_gpu(self, agent_type: str, preferred_gpu: Optional[int] = None):
        """Enhanced context manager with better error handling and recovery"""
        start_time = time.time()
        allocated_gpu = None
        agent = None
        
        try:
            # Get GPU allocation using enhanced manager
            allocated_gpu = await self.get_available_gpu_for_agent(agent_type, preferred_gpu)
            
            if allocated_gpu is None:
                # Try emergency recovery
                self.logger.warning(f"No GPU available for {agent_type}, attempting recovery...")
                await self.recover_gpu_errors()
                
                # Try again after recovery
                allocated_gpu = await self.get_available_gpu_for_agent(agent_type, preferred_gpu)
                
                if allocated_gpu is None:
                    self.stats["failed_allocations"] += 1
                    raise RuntimeError(f"No GPU available for {agent_type} even after recovery")
            
            self.stats["dynamic_allocations"] += 1
            
            # Get or create LLM engine for this GPU
            llm_engine = await self.get_or_create_llm_engine(allocated_gpu)
            
            # Create or get agent
            agent_key = f"{agent_type}_gpu_{allocated_gpu}"
            
            if agent_key not in self.agents:
                self.agents[agent_key] = self._create_agent(agent_type, llm_engine, allocated_gpu)
            
            agent = self.agents[agent_key]
            
            # Log allocation
            self._log_gpu_allocation(agent_type, allocated_gpu, preferred_gpu, True, time.time() - start_time)
            
            self.logger.info(f"✅ Allocated GPU {allocated_gpu} for {agent_type}")
            
            yield agent, allocated_gpu
            
        except Exception as e:
            if allocated_gpu is not None:
                self._log_gpu_allocation(agent_type, allocated_gpu, preferred_gpu, False, time.time() - start_time)
            self.logger.error(f"Failed to allocate GPU for {agent_type}: {e}")
            raise
        
        finally:
            # Release GPU workload when context exits
            if allocated_gpu is not None:
                try:
                    self.gpu_manager.release_gpu_workload(allocated_gpu, f"{agent_type}_agent")
                    allocation_duration = time.time() - start_time
                    self.logger.info(f"Released GPU {allocated_gpu} for {agent_type} after {allocation_duration:.2f}s")
                except Exception as e:
                    self.logger.warning(f"Error releasing GPU {allocated_gpu}: {e}")

    def start_health_monitoring(self, interval: int = 30):
    """Start continuous health monitoring"""
    import threading
    
    def monitor_loop():
        while True:
            try:
                health = self.get_system_health()
                
                if health["overall_health"] == "critical":
                    self.logger.error("🚨 CRITICAL: System health is critical!")
                    # Could trigger automatic recovery here
                    
                elif health["overall_health"] == "degraded":
                    self.logger.warning("⚠️ WARNING: System health is degraded")
                
                # Log GPU status
                for gpu_name, status in health["gpu_details"].items():
                    if not status.get("healthy", False):
                        self.logger.warning(f"GPU {gpu_name} is unhealthy: {status}")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="Health_Monitor")
    monitor_thread.start()
    self.logger.info(f"Started health monitoring with {interval}s interval")
    def _is_gpu_available(self, gpu_id: int) -> bool:
        """Check if GPU is actually available for use"""
        try:
            from gpu_force_fix import GPUForcer
            
            # Check if GPU exists
            if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
                return False
            
            # Check GPU memory using GPUForcer
            memory_info = GPUForcer.check_gpu_memory(gpu_id)
            free_gb = memory_info.get('free_gb', 0)
            
            # Get status from our GPU manager
            gpu_status_detailed = self.gpu_manager.get_gpu_status_detailed()
            gpu_key = f"gpu_{gpu_id}"
            
            if gpu_key not in gpu_status_detailed:
                return False
                
            gpu_status = gpu_status_detailed[gpu_key]
            
            # GPU is available if:
            # 1. Has sufficient free memory (at least 1.5GB)
            # 2. Utilization is below threshold  
            # 3. Status is available in our manager
            # 4. Active workloads are manageable
            if (free_gb >= 1.5 and 
                gpu_status.get('utilization_percent', 100) < self.config.utilization_threshold and
                gpu_status.get('is_available', False) and
                gpu_status.get('active_workloads', 10) < 5):  # Allow some concurrent workloads
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking GPU {gpu_id} availability: {e}")
            return False

    async def get_available_gpu_for_agent(self, agent_type: str, preferred_gpu: Optional[int] = None) -> Optional[int]:
        """Enhanced GPU allocation for agents using improved GPU manager"""
        
        # Force refresh GPU status first
        self.gpu_manager.force_refresh()
        
        # Check for existing engines that can be shared (except for LLM engines)
        if "llm_engine" not in agent_type.lower():
            existing_engines = list(self.llm_engine_pool.keys())
            for engine_key in existing_engines:
                gpu_id = int(engine_key.split('_')[1])
                
                try:
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    if memory_info.get('can_share', False):
                        self.logger.info(f"Sharing existing LLM engine on GPU {gpu_id} for {agent_type}")
                        return gpu_id
                except Exception as e:
                    self.logger.warning(f"Error checking GPU {gpu_id} for sharing: {e}")
                    continue
        
        # Use enhanced GPU allocation
        best_gpu = self.gpu_manager.get_available_gpu_smart(
            preferred_gpu=preferred_gpu,
            workload_type=agent_type,
            allow_sharing="llm_engine" not in agent_type.lower(),
            exclude_gpu_0=True  # Prefer to avoid GPU 0
        )
        
        if best_gpu is not None:
            self.logger.info(f"Allocated GPU {best_gpu} for {agent_type}")
            return best_gpu
        
        # Last resort: try any GPU including GPU 0
        best_gpu = self.gpu_manager.get_available_gpu_smart(
            preferred_gpu=None,
            workload_type=agent_type,
            allow_sharing=True,
            exclude_gpu_0=False
        )
        
        if best_gpu is not None:
            self.logger.warning(f"Using GPU {best_gpu} for {agent_type} as last resort")
            return best_gpu
        
        self.logger.error(f"No GPU available for {agent_type}")
        return None

    async def _cleanup_gpu_engine(self, gpu_id: int):
        """Enhanced GPU engine cleanup using new GPU forcer"""
        engine_key = f"gpu_{gpu_id}"
        
        try:
            if engine_key in self.llm_engine_pool:
                self.logger.info(f"Cleaning up LLM engine on GPU {gpu_id}")
                
                # Remove from pool first
                engine = self.llm_engine_pool.pop(engine_key, None)
                
                # Use enhanced cleanup
                success = EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
                
                # Release from GPU manager
                try:
                    self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
                except:
                    pass  # Ignore errors during cleanup
                
                # Wait for cleanup to take effect
                await asyncio.sleep(2)
                
                if success:
                    self.logger.info(f"✅ GPU {gpu_id} cleaned up successfully")
                else:
                    self.logger.warning(f"⚠️ GPU {gpu_id} cleanup may have failed")
                    
            else:
                self.logger.info(f"No engine to clean up on GPU {gpu_id}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up GPU {gpu_id}: {e}")

    async def emergency_reset(self) -> Dict[str, Any]:
        """Emergency reset of all GPU resources"""
        try:
            self.logger.warning("🚨 Performing emergency reset of all GPU resources")
            
            # Clear all LLM engines
            engine_keys = list(self.llm_engine_pool.keys())
            for engine_key in engine_keys:
                gpu_id = int(engine_key.split('_')[1])
                await self._cleanup_gpu_engine(gpu_id)
            
            # Clear all agents
            self.agents.clear()
            
            # Reset GPU manager
            self.gpu_manager.force_refresh()
            
            # Perform recovery
            recovery_results = await self.recover_gpu_errors()
            
            self.logger.info("✅ Emergency reset completed")
            
            return {
                "status": "success",
                "message": "Emergency reset completed",
                "engines_cleared": len(engine_keys),
                "agents_cleared": True,
                "recovery_results": recovery_results
            }
            
        except Exception as e:
            self.logger.error(f"Emergency reset failed: {e}")
            return {"status": "error", "error": str(e)}

# Fix 14: Add health check method
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get GPU status
            gpu_status = {}
            for gpu_id in range(self.config.total_gpu_count):
                try:
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    gpu_status[f"gpu_{gpu_id}"] = {
                        "healthy": memory_info.get('is_healthy', False),
                        "available": memory_info.get('is_available', False),
                        "free_gb": memory_info.get('free_gb', 0),
                        "error": memory_info.get('error', None)
                    }
                except Exception as e:
                    gpu_status[f"gpu_{gpu_id}"] = {"healthy": False, "error": str(e)}
            
            # System health indicators
            healthy_gpus = sum(1 for status in gpu_status.values() if status.get('healthy', False))
            available_gpus = sum(1 for status in gpu_status.values() if status.get('available', False))
            
            overall_health = "healthy" if healthy_gpus >= 2 else "degraded" if healthy_gpus >= 1 else "critical"
            
            return {
                "overall_health": overall_health,
                "healthy_gpus": healthy_gpus,
                "available_gpus": available_gpus,
                "total_gpus": self.config.total_gpu_count,
                "active_engines": len(self.llm_engine_pool),
                "active_agents": len(self.agents),
                "gpu_details": gpu_status,
                "active_llm_requests": getattr(self, '_active_llm_requests', 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "overall_health": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
                
    async def reload_model_on_gpu(self, gpu_id: int) -> bool:
        """Force reload model on specific GPU"""
        try:
            self.logger.info(f"Force reloading model on GPU {gpu_id}")
            
            # Cleanup existing model
            await self._cleanup_gpu_engine(gpu_id)
            
            # Wait a bit for cleanup
            await asyncio.sleep(2)
            
            # Create new engine
            await self.get_or_create_llm_engine(gpu_id, force_reload=True)
            
            self.logger.info(f"✅ Successfully reloaded model on GPU {gpu_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload model on GPU {gpu_id}: {e}")
            return False

    def get_gpu_availability_status(self) -> Dict[str, Any]:
        """Get detailed GPU availability status using both GPUForcer and GPU manager"""
        status = {}
        
        # Force refresh GPU manager first
        self.gpu_manager.force_refresh()
        gpu_manager_status = self.gpu_manager.get_gpu_status_detailed()
        
        for gpu_id in range(self.config.total_gpu_count):
            try:
                from gpu_force_fix import GPUForcer
                
                # Get memory info from GPUForcer
                memory_info = GPUForcer.check_gpu_memory(gpu_id)
                
                # Get status from GPU manager
                gpu_key = f"gpu_{gpu_id}"
                manager_info = gpu_manager_status.get(gpu_key, {})
                
                status[gpu_key] = {
                    "available": self._is_gpu_available(gpu_id),
                    "free_memory_gb": memory_info.get('free_gb', 0),
                    "total_memory_gb": memory_info.get('total_gb', 0),
                    "utilization_percent": manager_info.get('utilization_percent', 0),
                    "has_llm_engine": f"gpu_{gpu_id}" in self.llm_engine_pool,
                    "active_workloads": manager_info.get('active_workloads', 0),
                    "can_allocate": (self._is_gpu_available(gpu_id) and 
                                   memory_info.get('free_gb', 0) >= 2.0),
                    "manager_status": manager_info.get('status', 'unknown'),
                    "manager_available": manager_info.get('is_available', False),
                    "process_count": manager_info.get('process_count', 0)
                }
                
            except Exception as e:
                status[f"gpu_{gpu_id}"] = {
                    "available": False,
                    "error": str(e)
                }
        
        return status
    
        # Add this method to DynamicOpulenceCoordinator class in opulence_coordinator.py

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get processing statistics from database
            conn = sqlite3.connect(self.db_path)
            
            # Processing stats
            processing_stats = pd.read_sql_query("""
                SELECT 
                    operation,
                    COUNT(*) as count,
                    AVG(duration) as avg_duration,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration
                FROM processing_stats
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY operation
            """, conn)
            
            # File stats
            file_stats = pd.read_sql_query("""
                SELECT 
                    file_type,
                    COUNT(*) as count,
                    processing_status
                FROM file_metadata
                GROUP BY file_type, processing_status
            """, conn)
            
            conn.close()
            
            return {
                "system_stats": self.stats,
                "processing_stats": processing_stats.to_dict('records') if not processing_stats.empty else [],
                "file_stats": file_stats.to_dict('records') if not file_stats.empty else [],
                "gpu_stats": self.get_gpu_utilization_stats(),
                "database_stats": await self._get_database_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {
                "system_stats": self.stats,
                "processing_stats": [],
                "file_stats": [],
                "error": str(e)
            }
    
    def _create_agent(self, agent_type: str, llm_engine: AsyncLLMEngine, gpu_id: int):
        """Create agent instance with enhanced configuration"""
        
        # Common parameters for all agents
        common_params = {
            "llm_engine": llm_engine,
            "db_path": self.db_path,
            "gpu_id": gpu_id,
            "coordinator": self,
            "enable_llm": True,
            "conservative_mode": True  # Use conservative mode to prevent errors
        }
        
        if agent_type == "code_parser":
            return CompleteEnhancedCodeParserAgent(**common_params)
        elif agent_type == "vector_index":
            return VectorIndexAgent(
                local_model_path="./models/microsoft-codebert-base",
                **common_params
            )
        elif agent_type == "data_loader":
            return DataLoaderAgent(**common_params)
        elif agent_type == "lineage_analyzer":
            return LineageAnalyzerAgent(**common_params)
        elif agent_type == "logic_analyzer":
            return LogicAnalyzerAgent(**common_params)
        elif agent_type == "documentation":
            return DocumentationAgent(**common_params)
        elif agent_type == "db2_comparator":
            return DB2ComparatorAgent(
                max_rows=self.config.max_db_rows,
                **common_params
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _log_gpu_allocation(self, agent_type: str, gpu_id: int, preferred_gpu: Optional[int], 
                           success: bool, duration: float):
        """Log GPU allocation for tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO gpu_allocations 
                (agent_type, gpu_id, preferred_gpu, allocation_success, duration, workload_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (agent_type, gpu_id, preferred_gpu, success, duration, f"{agent_type}_workload"))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log GPU allocation: {str(e)}")
    
    # REPLACE the process_batch_files method with proper indentation:

    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """Process multiple files with PROPER batch processing and database storage"""
        start_time = time.time()

        try:
            # CHECK: Ensure database is initialized
            await self._ensure_database_initialized()
        
            # FIX 1: Import and use the BatchProcessor
            from utils.batch_processor import BatchProcessor
            batch_processor = BatchProcessor(max_workers=4, gpu_count=self.config.total_gpu_count)
        
            results = []
            total_files = len(file_paths)
        
            # Group files by type for efficient processing
            file_groups = {
                'cobol': [],
                'jcl': [],
                'csv': [],
                'auto': []
            }
        
            for file_path in file_paths:
                if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                    file_groups['cobol'].append(file_path)
                elif file_type == "jcl" or file_path.suffix.lower() == '.jcl':
                    file_groups['jcl'].append(file_path)
                elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                    file_groups['csv'].append(file_path)
                else:
                    file_groups['auto'].append(file_path)
        
            # FIX 2: Process each group using BatchProcessor with proper GPU distribution
            for group_type, group_files in file_groups.items():
                if not group_files:
                    continue
                
                self.logger.info(f"Processing {len(group_files)} {group_type} files")
            
                if group_type == 'cobol':
                    # Use batch processor with GPU distribution
                    group_results = await batch_processor.process_files_batch(
                        group_files,
                        self._process_cobol_file_with_storage,  # NEW method with proper storage
                        batch_size=8,
                        use_gpu_distribution=True
                    )
                    results.extend(group_results)
                
                elif group_type == 'jcl':
                    group_results = await batch_processor.process_files_batch(
                        group_files,
                        self._process_jcl_file_with_storage,   # NEW method with proper storage
                        batch_size=8,
                        use_gpu_distribution=True
                    )
                    results.extend(group_results)
                
                elif group_type == 'csv':
                    group_results = await batch_processor.process_files_batch(
                        group_files,
                        self._process_csv_file_with_storage,   # NEW method with proper storage
                        batch_size=6,
                        use_gpu_distribution=True
                    )
                    results.extend(group_results)
                
                else:
                    # Auto-detect and process
                    group_results = await batch_processor.process_files_batch(
                        group_files,
                        self._auto_detect_and_process_with_storage,  # NEW method with proper storage
                        batch_size=8,
                        use_gpu_distribution=True
                    )
                    results.extend(group_results)
        
            # FIX 3: Verify database storage
            await self._verify_database_storage(file_paths)

            # Vector embedding creation (only if successful results)
            if results:  # Only if we have successful results
                try:
                    await self._create_vector_embeddings_for_processed_files(file_paths)
                    self.logger.info("Vector embeddings created for processed files")
                except Exception as e:
                    self.logger.warning(f"Vector embedding creation failed: {str(e)}")
                    # Don't fail the entire batch if vector indexing fails
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_files_processed"] += total_files
            self._update_processing_stats("batch_processing", processing_time)
            
            # Cleanup batch processor
            batch_processor.shutdown()
        
            return {
                "status": "success",
                "files_processed": total_files,
                "processing_time": processing_time,
                "results": results,
                "database_verification": await self._get_database_stats(),
                "vector_indexing": "completed" if results else "skipped"
            }
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "files_processed": 0
            }
    # REPLACE the _create_vector_embeddings_for_processed_files method:

    async def _create_vector_embeddings_for_processed_files(self, file_paths: List[Path]):
        """Create vector embeddings for all successfully processed files"""
        try:
            async with self.get_agent_with_gpu("vector_index") as (vector_agent, gpu_id):
                
                # Get all program chunks that were just created
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get chunks from recently processed files
                program_names = [fp.stem for fp in file_paths]
                placeholders = ','.join(['?' for _ in program_names])
                
                cursor.execute(f"""
                    SELECT id, program_name, chunk_id, chunk_type, content, metadata
                    FROM program_chunks 
                    WHERE program_name IN ({placeholders})
                    AND created_timestamp > datetime('now', '-1 hour')
                """, program_names)
                
                chunks = cursor.fetchall()
                conn.close()
                
                if chunks:
                    # Create embeddings for all chunks
                    embedding_result = await vector_agent.create_embeddings_for_chunks(chunks)
                    self.logger.info(f"Created embeddings for {len(chunks)} chunks from {len(file_paths)} files")
                    return embedding_result
                else:
                    self.logger.warning("No chunks found for vector embedding creation")
                    return {"status": "no_chunks"}
                    
        except Exception as e:
            self.logger.error(f"Vector embedding creation failed: {str(e)}")
            # REMOVED: raise - Don't propagate the error, just log it
            return {"status": "error", "error": str(e)}
        
    async def _process_cobol_file_with_storage(self, file_path: Path) -> Dict[str, Any]:
        """Process COBOL file with guaranteed database storage"""
        try:
            async with self.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                # Process the file
                result = await agent.process_file(file_path)
            
                # FIX: Ensure database storage is completed
                if result.get("status") == "success":
                    await self._ensure_file_stored_in_db(file_path, result, "cobol")
            
                result["gpu_used"] = gpu_id
                return result
            
        except Exception as e:
            self.logger.error(f"Failed to process COBOL file {file_path}: {str(e)}")
            return {"status": "error", "file": str(file_path), "error": str(e)}

    async def _process_jcl_file_with_storage(self, file_path: Path) -> Dict[str, Any]:
        """Process JCL file with guaranteed database storage"""
        try:
            async with self.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                result = await agent.process_file(file_path)
            
                if result.get("status") == "success":
                    await self._ensure_file_stored_in_db(file_path, result, "jcl")
            
                result["gpu_used"] = gpu_id
                return result
            
        except Exception as e:
            self.logger.error(f"Failed to process JCL file {file_path}: {str(e)}")
            return {"status": "error", "file": str(file_path), "error": str(e)}

    async def _process_csv_file_with_storage(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV file with guaranteed database storage"""
        try:
            async with self.get_agent_with_gpu("data_loader") as (agent, gpu_id):
                result = await agent.process_file(file_path)
            
                if result.get("status") == "success":
                    await self._ensure_file_stored_in_db(file_path, result, "csv")
            
                result["gpu_used"] = gpu_id
                return result
            
        except Exception as e:
            self.logger.error(f"Failed to process CSV file {file_path}: {str(e)}")
            return {"status": "error", "file": str(file_path), "error": str(e)}

    async def _auto_detect_and_process_with_storage(self, file_path: Path) -> Dict[str, Any]:
        """Auto-detect and process file with guaranteed database storage"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)
        
            if 'IDENTIFICATION DIVISION' in content or 'PROGRAM-ID' in content:
                return await self._process_cobol_file_with_storage(file_path)
            elif content.startswith('//') and 'JOB' in content:
                return await self._process_jcl_file_with_storage(file_path)
            elif ',' in content and '\n' in content:
                return await self._process_csv_file_with_storage(file_path)
            else:
                return {"status": "unknown_file_type", "file": str(file_path)}
            
        except Exception as e:
            return {"status": "error", "error": str(e), "file": str(file_path)}

    # FIX 5: Database verification and storage methods
    async def _ensure_database_initialized(self):
        """Ensure database is properly initialized"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        # Verify tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
        
            required_tables = ['program_chunks', 'file_metadata', 'field_lineage']
            existing_tables = [table[0] for table in tables]
        
            for required_table in required_tables:
                if required_table not in existing_tables:
                    self.logger.warning(f"Table {required_table} not found, reinitializing database")
                    conn.close()
                    self._init_database()  # Reinitialize
                    break
            else:
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization check failed: {str(e)}")
            self._init_database()  # Force reinitialize

    async def _ensure_file_stored_in_db(self, file_path: Path, result: Dict, file_type: str):
        """Ensure file processing result is stored in database with proper commit"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
            # FIX: Add explicit transaction management
            cursor.execute("BEGIN TRANSACTION")
        
            try:
                # Store file metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO file_metadata 
                    (file_name, file_type, processing_status, last_modified)
                    VALUES (?, ?, ?, ?)
                """, (
                    file_path.name,
                    file_type,
                    result.get("status", "processed"),
                    datetime.now().isoformat()
                ))
            
                # Verify chunks were stored
                if "chunks_created" in result and result["chunks_created"] > 0:
                    cursor.execute("""
                        SELECT COUNT(*) FROM program_chunks 
                        WHERE program_name = ?
                    """, (file_path.stem,))
                
                    chunk_count = cursor.fetchone()[0]
                
                    if chunk_count != result["chunks_created"]:
                        self.logger.warning(
                            f"Chunk count mismatch for {file_path.name}: "
                            f"expected {result['chunks_created']}, found {chunk_count}"
                        )
            
                # FIX: Explicit commit
                cursor.execute("COMMIT")
                self.logger.debug(f"Successfully stored {file_path.name} in database")
            
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
            
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"Failed to verify database storage for {file_path}: {str(e)}")

    async def _verify_database_storage(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Verify all files were properly stored in database"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count total chunks
            cursor.execute("SELECT COUNT(*) FROM program_chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Count total files
            cursor.execute("SELECT COUNT(*) FROM file_metadata")
            total_files_in_db = cursor.fetchone()[0]
            
            # Count by file type
            cursor.execute("""
                SELECT file_type, COUNT(*) 
                FROM file_metadata 
                GROUP BY file_type
            """)
            file_type_counts = dict(cursor.fetchall())
            
            conn.close()
            
            verification_result = {
                "files_processed": len(file_paths),
                "files_in_database": total_files_in_db,
                "total_chunks": total_chunks,
                "file_type_breakdown": file_type_counts,
                "storage_success_rate": (total_files_in_db / len(file_paths)) * 100 if file_paths else 0
            }
            
            self.logger.info(f"Database verification: {verification_result}")
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Database verification failed: {str(e)}")
            return {"error": str(e)}

    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Get table sizes
            tables = ['program_chunks', 'file_metadata', 'field_lineage', 'lineage_nodes', 'lineage_edges']
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    stats[f"{table}_count"] = 0  # Table doesn't exist
            
            # Get database file size
            import os
            if os.path.exists(self.db_path):
                stats["database_size_bytes"] = os.path.getsize(self.db_path)
            else:
                stats["database_size_bytes"] = 0
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            return {"error": str(e)}
    async def process_regular_chat_query(self, query: str) -> str:
        """Process regular chat query without vector search"""
        try:
            # Determine query type and route to appropriate agent
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['lifecycle', 'lineage', 'trace', 'impact']):
                # Extract component name from query
                component_name = self._extract_component_name(query)
                if component_name:
                    result = await self.analyze_component(component_name)
                    return self._format_analysis_response(result)
                else:
                    return "Could you please specify which component (file, table, program, or field) you'd like me to analyze?"
            
            elif any(word in query_lower for word in ['compare', 'difference', 'db2']):
                return "For data comparison, please use the DB2 Comparison tab to select specific components."
            
            else:
                # General query - provide helpful guidance
                return self._generate_general_response(query)
        
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"

    def _extract_component_name(self, query: str) -> str:
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

    def _format_analysis_response(self, result: dict) -> str:
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

    def _generate_general_response(self, query: str) -> str:
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

    async def analyze_component(self, component_name: str, component_type: str = None, 
                          preferred_gpu: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced component analysis with better error handling"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting analysis for component: {component_name}, type: {component_type}")
            
            # Check cache first
            cache_key = f"analyze_{component_name}_{component_type}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached result for {component_name}")
                return cached_result
            
            # Determine component type if not specified
            if not component_type:
                component_type = await self._determine_component_type(component_name, preferred_gpu)
                self.logger.info(f"Determined component type: {component_type}")
            
            analysis_result = {
                "component_name": component_name,
                "component_type": component_type,
                "status": "processing"
            }
            
            try:
                # Semantic search with better error handling
                async with self.get_agent_with_gpu("vector_index", preferred_gpu) as (vector_agent, gpu_id):
                    try:
                        semantic_results = await vector_agent.search_similar_components(component_name)
                        analysis_result["semantic_search"] = semantic_results
                        analysis_result["vector_gpu_used"] = gpu_id
                        self.logger.info(f"Semantic search completed for {component_name}")
                    except Exception as e:
                        self.logger.error(f"Semantic search failed: {e}")
                        analysis_result["semantic_search"] = {"error": str(e)}
            except Exception as e:
                self.logger.error(f"Vector agent allocation failed: {e}")
                analysis_result["semantic_search"] = {"error": "Vector agent unavailable"}
            
            # Component-specific analysis with fallbacks
            if component_type in ["file", "table"]:
                try:
                    async with self.get_agent_with_gpu("lineage_analyzer", preferred_gpu) as (lineage_agent, gpu1):
                        lineage_result = await lineage_agent.analyze_field_lineage(component_name)
                        analysis_result["lineage"] = lineage_result
                        self.logger.info(f"Lineage analysis completed for {component_name}")
                except Exception as e:
                    self.logger.error(f"Lineage analysis failed: {e}")
                    analysis_result["lineage"] = {"error": str(e)}
                
                try:
                    async with self.get_agent_with_gpu("data_loader", preferred_gpu) as (data_agent, gpu2):
                        data_result = await data_agent.get_component_info(component_name)
                        analysis_result["data_info"] = data_result
                        self.logger.info(f"Data info completed for {component_name}")
                except Exception as e:
                    self.logger.error(f"Data info failed: {e}")
                    analysis_result["data_info"] = {"error": str(e)}
            
            elif component_type in ["program", "cobol"]:
                try:
                    async with self.get_agent_with_gpu("logic_analyzer", preferred_gpu) as (logic_agent, gpu_id):
                        logic_result = await logic_agent.analyze_program(component_name)
                        analysis_result["logic_analysis"] = logic_result
                        self.logger.info(f"Logic analysis completed for {component_name}")
                except Exception as e:
                    self.logger.error(f"Logic analysis failed: {e}")
                    analysis_result["logic_analysis"] = {"error": str(e)}
            
            # Always try to provide some basic info from database
            if not any(key in analysis_result for key in ["lineage", "data_info", "logic_analysis"]):
                try:
                    basic_info = await self._get_basic_component_info(component_name)
                    analysis_result["basic_info"] = basic_info
                    self.logger.info(f"Basic info retrieved for {component_name}")
                except Exception as e:
                    self.logger.error(f"Basic info failed: {e}")
                    analysis_result["basic_info"] = {"error": str(e)}
            
            analysis_result["status"] = "completed"
            
            # Cache the result
            self.cache_manager.set(cache_key, analysis_result)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Component analysis completed for {component_name} in {processing_time:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Component analysis failed for {component_name}: {str(e)}")
            return {
                "component_name": component_name,
                "component_type": component_type or "unknown",
                "status": "error",
                "error": str(e)
            }

    async def _get_basic_component_info(self, component_name: str) -> Dict[str, Any]:
        """Get basic component info directly from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check program_chunks
            cursor.execute("""
                SELECT COUNT(*), chunk_type, program_name 
                FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
                GROUP BY chunk_type, program_name
            """, (component_name, f"%{component_name}%"))
            
            chunk_info = cursor.fetchall()
            
            # Check file_metadata if available
            try:
                cursor.execute("""
                    SELECT COUNT(*), file_type, table_name 
                    FROM file_metadata 
                    WHERE table_name = ? OR table_name LIKE ?
                """, (component_name, f"%{component_name}%"))
                
                file_info = cursor.fetchall()
            except:
                file_info = []
            
            conn.close()
            
            return {
                "chunk_info": chunk_info,
                "file_info": file_info,
                "found_in_chunks": len(chunk_info) > 0,
                "found_in_files": len(file_info) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Basic component info failed: {e}")
            return {"error": str(e)}
    
    async def search_code_patterns(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search code patterns using vector similarity"""
        try:
            async with self.get_agent_with_gpu("vector_index") as (vector_agent, gpu_id):
                results = await vector_agent.search_code_by_pattern(query, limit=limit)
                
                return {
                    "status": "success",
                    "query": query,
                    "results": results,
                    "total_found": len(results),
                    "gpu_used": gpu_id
                }
                
        except Exception as e:
            self.logger.error(f"Code pattern search failed: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "query": query
            }

    # 5. NEW METHOD: Rebuild vector index
    async def rebuild_vector_index(self) -> Dict[str, Any]:
        """Rebuild the entire vector index from stored chunks"""
        try:
            async with self.get_agent_with_gpu("vector_index") as (vector_agent, gpu_id):
                
                # Get all chunks from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, program_name, chunk_id, chunk_type, content, metadata
                    FROM program_chunks 
                    ORDER BY created_timestamp DESC
                """)
                
                all_chunks = cursor.fetchall()
                conn.close()
                
                if not all_chunks:
                    return {"status": "error", "error": "No chunks found in database"}
                
                # Rebuild index
                result = await vector_agent.rebuild_index_from_chunks(all_chunks)
                
                return {
                    "status": "success",
                    "chunks_processed": len(all_chunks),
                    "rebuild_result": result,
                    "gpu_used": gpu_id
                }
                
        except Exception as e:
            self.logger.error(f"Vector index rebuild failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def process_chat_query_with_vector_search(self, query: str) -> str:
        """Process chat query with vector search enhancement"""
        try:
            # Check if this is a search query
            if any(word in query.lower() for word in ['search', 'find', 'pattern', 'similar', 'like']):
                
                # Use vector search
                search_results = await self.search_code_patterns(query, limit=5)
                
                if search_results.get("status") == "success" and search_results.get("results"):
                    response = f"## 🔍 Found {len(search_results['results'])} Similar Code Patterns\n\n"
                    
                    for i, result in enumerate(search_results["results"][:3], 1):
                        metadata = result.get("metadata", {})
                        response += f"**{i}. {metadata.get('program_name', 'Unknown Program')}**\n"
                        response += f"- Chunk: {metadata.get('chunk_id', 'unknown')}\n"
                        response += f"- Type: {metadata.get('chunk_type', 'code')}\n"
                        response += f"- Similarity: {result.get('similarity_score', 0):.2f}\n"
                        response += f"- Preview: {result.get('content', '')[:100]}...\n\n"
                    
                    return response
                else:
                    return "🔍 No similar code patterns found. Try refining your search terms."
            
            # Fall back to regular query processing
            return await self.process_regular_chat_query(query)
            
        except Exception as e:
            return f"❌ Search failed: {str(e)}"
        
    
    async def _determine_component_type(self, component_name: str, preferred_gpu: Optional[int] = None) -> str:
        """Safely determine component type with fallback logic"""
        try:
            # First try simple heuristics
            name_lower = component_name.lower()
            
            # File patterns
            if any(pattern in name_lower for pattern in ['file', 'dat', 'txt', 'csv']):
                return "file"
            
            # Table patterns
            if any(pattern in name_lower for pattern in ['tbl', 'table', 'tab']):
                return "table"
            
            # Program patterns
            if any(pattern in name_lower for pattern in ['prog', 'pgm', 'program']):
                return "program"
            
            # JCL patterns
            if any(pattern in name_lower for pattern in ['jcl', 'job', 'proc']):
                return "jcl"
            
            # If heuristics fail, try LLM
            prompt = f"""
            Analyze the component name '{component_name}' and determine its type.
            
            Types:
            - file: Data file or dataset
            - table: DB2 table
            - program: COBOL program
            - jcl: JCL job or procedure
            
            Based on naming conventions, determine the most likely type.
            Respond with only the type name.
            """
            
            sampling_params = SamplingParams(temperature=0.1, max_tokens=20, stop=["\n"])
            
            # Use safe GPU allocation
            async with self.get_agent_with_gpu("logic_analyzer", preferred_gpu) as (agent, gpu_id):
                if hasattr(agent, 'llm_engine') and agent.llm_engine:
                    result = await self._safe_llm_generate(agent.llm_engine, prompt, sampling_params)
                    if result and result.lower() in ['file', 'table', 'program', 'jcl']:
                        return result.lower()
            
            # Final fallback
            return "file"  # Default to file
            
        except Exception as e:
            self.logger.warning(f"Component type determination failed: {e}")
            return "file"  # Safe fallback
    
    async def _analyze_lifecycle(self, component_name: str, component_type: str, 
                                preferred_gpu: Optional[int] = None) -> Dict[str, Any]:
        """Analyze complete lifecycle of a component with dynamic GPU allocation"""
        async with self.get_agent_with_gpu("lineage_analyzer", preferred_gpu) as (agent, gpu_id):
            return await agent.analyze_full_lifecycle(component_name, component_type)
    
    async def _find_dependencies(self, component_name: str, 
                                preferred_gpu: Optional[int] = None) -> List[str]:
        """Find all dependencies for a component with dynamic GPU allocation"""
        async with self.get_agent_with_gpu("lineage_analyzer", preferred_gpu) as (agent, gpu_id):
            return await agent.find_dependencies(component_name)
    
    async def _analyze_job_flow(self, jcl_name: str, 
                               preferred_gpu: Optional[int] = None) -> Dict[str, Any]:
        """Analyze JCL job flow with dynamic GPU allocation"""
        async with self.get_agent_with_gpu("code_parser", preferred_gpu) as (agent, gpu_id):
            return await agent.analyze_job_flow(jcl_name)
    
    async def _auto_detect_and_process(self, file_path: Path) -> Dict[str, Any]:
        """Auto-detect file type and process with dynamic GPU allocation"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)
            
            if 'IDENTIFICATION DIVISION' in content or 'PROGRAM-ID' in content:
                async with self.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                    result = await agent.process_file(file_path)
                    result["gpu_used"] = gpu_id
                    return result
                    
            elif content.startswith('//') and 'JOB' in content:
                async with self.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                    result = await agent.process_file(file_path)
                    result["gpu_used"] = gpu_id
                    return result
                    
            elif ',' in content and '\n' in content:
                async with self.get_agent_with_gpu("data_loader") as (agent, gpu_id):
                    result = await agent.process_file(file_path)
                    result["gpu_used"] = gpu_id
                    return result
                    
            else:
                return {"status": "unknown_file_type", "file": str(file_path)}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _update_processing_stats(self, operation: str, duration: float, gpu_id: int = None):
        """Update processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO processing_stats (operation, duration, gpu_used, status)
            VALUES (?, ?, ?, ?)
        """, (operation, duration, gpu_id, "completed"))
        
        conn.commit()
        conn.close()
        
        # Update in-memory stats
        self.stats["avg_response_time"] = (
            self.stats["avg_response_time"] + duration
        ) / 2
    
    async def recover_gpu_errors(self) -> Dict[str, Any]:
        """Recover from GPU errors and clean up problematic engines"""
        recovery_results = {}
        
        try:
            # Check all GPUs for errors
            for gpu_id in range(self.config.total_gpu_count):
                try:
                    memory_info = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                    if 'error' in memory_info or not memory_info.get('is_healthy', False):
                        self.logger.info(f"Attempting recovery for GPU {gpu_id}")
                        
                        # Clean up existing engine if any
                        await self._cleanup_gpu_engine(gpu_id)
                        
                        # Perform aggressive cleanup
                        success = EnhancedGPUForcer.aggressive_gpu_cleanup(gpu_id)
                        
                        # Re-check GPU state
                        final_memory = EnhancedGPUForcer.check_gpu_memory(gpu_id)
                        
                        recovery_results[f"gpu_{gpu_id}"] = {
                            "attempted": True,
                            "cleanup_success": success,
                            "final_state": final_memory,
                            "recovered": final_memory.get('is_available', False)
                        }
                    else:
                        recovery_results[f"gpu_{gpu_id}"] = {
                            "attempted": False,
                            "reason": "GPU appears healthy",
                            "state": memory_info
                        }
                        
                except Exception as e:
                    recovery_results[f"gpu_{gpu_id}"] = {
                        "attempted": True,
                        "error": str(e),
                        "recovered": False
                    }
            
            # Update GPU manager state
            self.gpu_manager.force_refresh()
            
            return {
                "status": "completed",
                "recovery_results": recovery_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GPU recovery failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            "status": "healthy",
            "gpu_status": self.gpu_manager.get_gpu_status_detailed(),
            "workload_distribution": self.gpu_manager.get_workload_distribution(),
            "memory_usage": self.health_monitor.get_memory_usage(),
            "processing_stats": self.stats,
            "active_agents": len(self.agents),
            "cache_stats": self.cache_manager.get_stats(),
            "llm_engines": list(self.llm_engine_pool.keys()),
            "configuration": {
                "gpu_config": self.config_manager.get_gpu_config().__dict__,
                "agent_preferences": self.config_manager.get_gpu_agent_mapping(),
                "performance_config": self.config_manager.get_performance_config()
            },
            "gpu_recommendations": {
                agent_type: self.gpu_manager.get_recommendation(f"{agent_type}_agent")
                for agent_type in ["code_parser", "vector_index", "data_loader", 
                                  "lineage_analyzer", "logic_analyzer", "documentation", "db2_comparator"]
            }
        }
    
    def get_gpu_utilization_stats(self) -> Dict[str, Any]:
        """Get detailed GPU utilization statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get GPU allocation statistics
        df_allocations = pd.read_sql_query("""
            SELECT 
                agent_type,
                gpu_id,
                COUNT(*) as allocation_count,
                AVG(duration) as avg_duration,
                SUM(CASE WHEN allocation_success THEN 1 ELSE 0 END) as successful_allocations,
                AVG(CASE WHEN allocation_success THEN duration ELSE NULL END) as avg_successful_duration
            FROM gpu_allocations
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY agent_type, gpu_id
            ORDER BY gpu_id, agent_type
        """, conn)
        
        # Get processing statistics by GPU
        df_processing = pd.read_sql_query("""
            SELECT 
                gpu_used,
                operation,
                COUNT(*) as operation_count,
                AVG(duration) as avg_duration
            FROM processing_stats
            WHERE timestamp > datetime('now', '-24 hours') AND gpu_used IS NOT NULL
            GROUP BY gpu_used, operation
            ORDER BY gpu_used, operation
        """, conn)
        
        conn.close()
        
        return {
            "allocation_stats": df_allocations.to_dict('records'),
            "processing_stats": df_processing.to_dict('records'),
            "current_gpu_status": self.gpu_manager.get_gpu_status_detailed(),
            "system_stats": self.stats
        }
    
    async def optimize_gpu_allocation(self) -> Dict[str, Any]:
        """Optimize GPU allocation based on current workloads"""
        try:
            # Force refresh GPU status
            self.gpu_manager.force_refresh()
            
            # Clean up completed workloads
            self.gpu_manager.cleanup_completed_workloads()
            
            # Get current status
            gpu_status = self.gpu_manager.get_gpu_status_detailed()
            workload_distribution = self.gpu_manager.get_workload_distribution()
            
            optimization_suggestions = []
            
            # Analyze GPU utilization patterns
            for gpu_id, status in gpu_status.items():
                gpu_num = int(gpu_id.split('_')[1])
                
                if status['utilization_percent'] > 90:
                    optimization_suggestions.append({
                        "type": "high_utilization",
                        "gpu_id": gpu_num,
                        "message": f"GPU {gpu_num} is heavily utilized ({status['utilization_percent']:.1f}%)",
                        "recommendation": "Consider redistributing workloads to other GPUs"
                    })
                
                elif status['utilization_percent'] < 10 and status['active_workloads'] == 0:
                    optimization_suggestions.append({
                        "type": "underutilized",
                        "gpu_id": gpu_num,
                        "message": f"GPU {gpu_num} is underutilized ({status['utilization_percent']:.1f}%)",
                        "recommendation": "Available for new workloads"
                    })
            
            # Analyze workload distribution
            total_workloads = sum(len(workloads) for workloads in workload_distribution.values())
            if total_workloads > 0:
                workload_balance = max(len(workloads) for workloads in workload_distribution.values()) - \
                                 min(len(workloads) for workloads in workload_distribution.values())
                
                if workload_balance > 2:
                    optimization_suggestions.append({
                        "type": "workload_imbalance",
                        "message": f"Workload distribution is unbalanced (difference: {workload_balance})",
                        "recommendation": "Consider rebalancing workloads across GPUs"
                    })
            
            return {
                "status": "success",
                "current_gpu_status": gpu_status,
                "workload_distribution": workload_distribution,
                "optimization_suggestions": optimization_suggestions,
                "total_suggestions": len(optimization_suggestions)
            }
            
        except Exception as e:
            self.logger.error(f"GPU optimization failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def process_with_preferred_gpu(self, operation: str, preferred_gpu: int = 0, **kwargs) -> Dict[str, Any]:
        """Process operation with preferred GPU (fallback to others if unavailable)"""
        start_time = time.time()
        
        try:
            if operation == "analyze_component":
                return await self.analyze_component(
                    kwargs.get("component_name"),
                    kwargs.get("component_type"),
                    preferred_gpu
                )
            
            elif operation == "process_files":
                # For file processing, we'll try to use preferred GPU but allow fallback
                file_paths = kwargs.get("file_paths", [])
                results = []
                
                for file_path in file_paths:
                    try:
                        result = await self._auto_detect_and_process(file_path)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {str(e)}")
                        results.append({"status": "error", "file": str(file_path), "error": str(e)})
                
                return {
                    "status": "success",
                    "results": results,
                    "processing_time": time.time() - start_time,
                    "preferred_gpu": preferred_gpu
                }
            
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            self.logger.error(f"Operation {operation} failed: {str(e)}")
            return {"status": "error", "error": str(e), "operation": operation}
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> bool:
        """Update system configuration at runtime"""
        try:
            for key_path, value in config_updates.items():
                self.config_manager.set(key_path, value)
            
            # Apply GPU configuration changes
            if any(key.startswith('gpu.') for key in config_updates.keys()):
                gpu_config = self.config_manager.get_gpu_config()
                
                # Update GPU manager with new thresholds
                self.gpu_manager.memory_threshold = gpu_config.memory_threshold
                self.gpu_manager.utilization_threshold = gpu_config.utilization_threshold
                
                self.logger.info("GPU configuration updated at runtime")
            
            # Apply cache configuration changes
            if 'system.cache_ttl' in config_updates:
                new_ttl = config_updates['system.cache_ttl']
                self.cache_manager.default_ttl = new_ttl
                self.logger.info(f"Cache TTL updated to {new_ttl}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            return False
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            "config_file": str(self.config_manager.config_file),
            "gpu_config": self.config_manager.get_gpu_config().__dict__,
            "agent_mappings": self.config_manager.get_gpu_agent_mapping(),
            "performance_config": self.config_manager.get_performance_config(),
            "optimization_config": self.config_manager.get_optimization_config(),
            "validation_issues": self.config_manager.validate_config()
        }
    
    async def reload_configuration(self) -> bool:
        """Reload configuration from file"""
        try:
            # Reload configuration
            success = self.config_manager.load_config()
            if not success:
                return False
            
            # Apply new configuration
            gpu_config = self.config_manager.get_gpu_config()
            
            # Update GPU manager
            self.gpu_manager.memory_threshold = gpu_config.memory_threshold
            self.gpu_manager.utilization_threshold = gpu_config.utilization_threshold
            
            # Update cache TTL
            cache_ttl = self.config_manager.get("system.cache_ttl", 3600)
            self.cache_manager.default_ttl = cache_ttl
            
            self.logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {str(e)}")
            return False
        

    def shutdown(self):
        """Shutdown the coordinator and all resources"""
        try:
            # Save configuration before shutdown
            self.config_manager.save_config()
            
            # Shutdown GPU manager
            self.gpu_manager.shutdown()
            
            # Clear LLM engine pool
            self.llm_engine_pool.clear()
            
            # Clear agents
            self.agents.clear()
            
            self.logger.info("Opulence Coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    def create_standalone_agent(agent_type: str = "data_loader") -> DataLoaderAgent:
        """Create standalone agent that will use coordinator's shared LLMs"""
        try:
            # Get global coordinator
            coordinator = get_dynamic_coordinator()
        
            # Create agent with coordinator reference
            if agent_type == "data_loader":
                return DataLoaderAgent(coordinator=coordinator)
            # Add other agent types as needed...
        
        except Exception as e:
            # Fallback: create agent without coordinator
            logging.warning(f"Creating agent without coordinator: {e}")
            return DataLoaderAgent()

# Global coordinator instance
coordinator = None

def get_dynamic_coordinator() -> DynamicOpulenceCoordinator:
    """Get or create global dynamic coordinator instance"""
    global coordinator
    if coordinator is None:
        config = OpulenceConfig()
        coordinator = DynamicOpulenceCoordinator(config)
    return coordinator

async def initialize_dynamic_system():
    """Initialize the dynamic Opulence system"""
    global coordinator
    if coordinator is None:
        config = OpulenceConfig()
        coordinator = DynamicOpulenceCoordinator(config)
    return coordinator

# Utility functions for easy access
async def process_with_auto_gpu(operation: str, **kwargs) -> Dict[str, Any]:
    """Process operation with automatic GPU selection"""
    coord = get_dynamic_coordinator()
    return await coord.process_with_preferred_gpu(operation, preferred_gpu=None, **kwargs)

async def process_with_preferred_gpu(operation: str, preferred_gpu: int, **kwargs) -> Dict[str, Any]:
    """Process operation with preferred GPU"""
    coord = get_dynamic_coordinator()
    return await coord.process_with_preferred_gpu(operation, preferred_gpu, **kwargs)

def get_gpu_status() -> Dict[str, Any]:
    """Get current GPU status"""
    coord = get_dynamic_coordinator()
    return coord.gpu_manager.get_gpu_status_detailed()

def get_gpu_recommendations(workload_type: str) -> Dict[str, Any]:
    """Get GPU recommendations for workload type"""
    coord = get_dynamic_coordinator()
    return coord.gpu_manager.get_recommendation(workload_type)

# Configuration management functions
def update_system_config(config_updates: Dict[str, Any]) -> bool:
    """Update system configuration"""
    coord = get_dynamic_coordinator()
    return coord.update_configuration(config_updates)

def get_current_config() -> Dict[str, Any]:
    """Get current system configuration"""
    coord = get_dynamic_coordinator()
    return coord.get_configuration_status()

def set_agent_gpu_preference(agent_type: str, gpu_id: Optional[int]) -> bool:
    """Set GPU preference for specific agent"""
    coord = get_dynamic_coordinator()
    return coord.config_manager.set_agent_preferred_gpu(agent_type, gpu_id)

def optimize_gpu_assignments() -> Dict[str, int]:
    """Optimize GPU assignments for all agents"""
    coord = get_dynamic_coordinator()
    return coord.config_manager.optimize_agent_gpu_assignment()

async def reload_system_config() -> bool:
    """Reload system configuration from file"""
    coord = get_dynamic_coordinator()
    return await coord.reload_configuration()

def backup_config() -> str:
    """Create backup of current configuration"""
    coord = get_dynamic_coordinator()
    return coord.config_manager.backup_config()

def get_config_validation_status() -> List[str]:
    """Get configuration validation issues"""
    coord = get_dynamic_coordinator()
    return coord.config_manager.validate_config()