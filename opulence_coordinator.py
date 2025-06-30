# opulence_coordinator.py
"""
Opulence - Deep Research Mainframe Agent Coordinator with Dynamic GPU Allocation
Handles dynamic GPU distribution, agent orchestration, and parallel processing
"""

import tracemalloc
import os
import gc
import psutil

# CRITICAL: Enable tracemalloc BEFORE any other imports
#if not tracemalloc.is_tracing():
#    tracemalloc.start(25)
#    print("✅ Tracemalloc enabled")

#os.environ['PYTHONTRACEMALLOC'] = '25'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Don't block
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path
import json
import uuid
import sqlite3
from datetime import datetime as dt
from contextlib import asynccontextmanager
import re
import streamlit as st
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import faiss
import chromadb
import pandas as pd
import subprocess

# Import our agents
from agents.code_parser_agent import CodeParserAgent
from agents.chat_agent import OpulenceChatAgent
from agents.vector_index_agent import VectorIndexAgent  
from agents.data_loader_agent import DataLoaderAgent
from agents.lineage_analyzer_agent import LineageAnalyzerAgent
from agents.logic_analyzer_agent import LogicAnalyzerAgent
from agents.documentation_agent import DocumentationAgent
from agents.db2_comparator_agent import DB2ComparatorAgent
from utils.gpu_manager import OptimizedDynamicGPUManager, SafeGPUForcer
from utils.dynamic_config_manager import DynamicConfigManager, get_dynamic_config, GPUConfig
from utils.health_monitor import HealthMonitor
from utils.cache_manager import CacheManager


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class SimpleRequestManager:
    def __init__(self, max_concurrent: int = 2):  # Reduce to 2
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire_request_slot(self):
        await self.semaphore.acquire()
        return "req_id"
    
    async def release_request_slot(self, request_id: str):
        self.semaphore.release()
    
    

@dataclass
class OpulenceConfig:
    """Configuration for Opulence system"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 2048  # Reduced from 4096
    temperature: float = 0.1
    total_gpu_count: int = 4
    max_processing_time: int = 600  # Reduced to 10 minutes
    batch_size: int = 8  # Reduced from 32
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600
    memory_threshold: float = 0.7
    utilization_threshold: float = 70.0

class DynamicOpulenceCoordinator:
    """Enhanced coordinator with dynamic GPU allocation and memory tracking"""
    
    def __init__(self, config: OpulenceConfig = None):
        """Enhanced coordinator initialization with memory tracking"""
        # Initialize logging
        self.logger = self._setup_logging()
        # Initialize configuration manager
        self.config_manager = DynamicConfigManager()
        
        # Use provided config or create from config manager
        if config is None:
                # FIX: Make sure total_gpu_count is properly mapped
            runtime_config = self.config_manager.create_runtime_config()
            
            # Ensure total_gpu_count is available
            if 'total_gpu_count' not in runtime_config:
                runtime_config['total_gpu_count'] = self.config_manager.system.total_gpu_count
            
            self.config = OpulenceConfig(**{k: v for k, v in runtime_config.items() 
                                        if k in OpulenceConfig.__dataclass_fields__})
        else:
            self.config = config
            

        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0.0,
            "response_count": 0,
            "dynamic_allocations": 0,
            "failed_allocations": 0,
            "memory_usage_mb": 0.0,
            "peak_memory_mb": 0.0,
            "gpu_allocation_failures": 0,
            "last_operation_time": None
        }
        
        
        self._refreshing = False
        
        # FIX: Initialize ALL attributes BEFORE use
        self._refreshing = False  # ADD THIS LINE
        
        self._last_refresh = 0
        self._refresh_interval = 120
        self._cache_duration = 60
        
        # Auto-detect GPU count if not provided
        self.total_gpu_count = self._detect_gpu_count() if total_gpu_count is None else total_gpu_count
        self.available_gpu_count = self._get_available_gpu_count()
        
        # Initialize collections
        self.gpu_status: Dict[int, GPUStatus] = {}
        self.active_workloads: Dict[int, List[str]] = {}
        self.workload_history: List[Dict] = []
        
        # Initialize GPU status
        self._initialize_gpu_status()          
        # Get GPU configuration from config manager
        gpu_config = self.config_manager.get_gpu_config()
        
        # Initialize dynamic GPU manager with config
        self.gpu_manager = OptimizedDynamicGPUManager(
                total_gpu_count=None,  # Auto-detect
                memory_threshold=0.9,  # Conservative
                utilization_threshold=90.0  # Conservative
            )
        
        self.health_monitor = HealthMonitor(
            history_size=100,  # Reduce from 1000
            alert_retention_hours=6  # Reduce from 24
        )
        
        # Set conservative thresholds
        self.health_monitor.set_thresholds({
            "cpu_percent": {"warning": 85, "critical": 95},
            "memory_percent": {"warning": 80, "critical": 90},  # Lower thresholds
            "gpu_memory_percent": {"warning": 80, "critical": 90}
        })
        
        # Get cache configuration
        cache_ttl = self.config_manager.get("system.cache_ttl", 3600)
        self.cache_manager = CacheManager(cache_ttl)
        
        # Initialize request manager with conservative settings
        self.request_manager = SimpleRequestManager(max_concurrent=4)  # Very conservative
              
        # Initialize SQLite database
        self.db_path = "opulence_data.db"
        self._init_database()
        
        # LLM engine pool - create engines dynamically
        self.llm_engine_pool = {}
        self.engine_lock = asyncio.Lock()
        
        # Agent instances - will be created with dynamic GPU allocation
        self.agents = {}
        
                   
        
        self.logger.info("Enhanced Opulence Coordinator initialized with memory tracking and recovery management")
    
    
    
        
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
        """FIXED: Enhanced LLM engine creation"""
        async with self.engine_lock:
            engine_key = f"gpu_{gpu_id}"
            
            # Return existing engine if available
            if engine_key in self.llm_engine_pool and not force_reload:
                return self.llm_engine_pool[engine_key]
            
            # Clean up existing if force reload
            if force_reload and engine_key in self.llm_engine_pool:
                await self._cleanup_gpu_engine_simple(gpu_id)
            
            try:
                # FIXED: Use proper GPU memory check
                memory_info = SafeGPUForcer.check_gpu_memory_safe(gpu_id)
                
                if not memory_info['is_available']:
                    # Try cleanup first
                    SafeGPUForcer.safe_gpu_cleanup(gpu_id)
                    
                    # Recheck after cleanup
                    memory_info = SafeGPUForcer.check_gpu_memory_safe(gpu_id)
                    if not memory_info['is_available']:
                        raise RuntimeError(f"GPU {gpu_id} unavailable after cleanup")
                
                logger.info(f"Creating LLM engine on GPU {gpu_id}: {memory_info['free_gb']:.1f}GB free")
                
                # FIXED: Proper GPU environment setup
                SafeGPUForcer.force_gpu_environment(gpu_id, cleanup_first=True)
                
                # FIXED: Conservative engine args
                engine_args = AsyncEngineArgs(
                    model=self.config.model_name,
                    tensor_parallel_size=1,
                    max_num_seqs=4,  # Conservative
                    max_model_len=2048,  # Reduced
                    gpu_memory_utilization=0.6,  # Conservative
                    swap_space=2,
                    enforce_eager=True  # Avoid compilation overhead
                )
                
                # Create with timeout
                engine = await asyncio.wait_for(
                    asyncio.to_thread(AsyncLLMEngine.from_engine_args, engine_args),
                    timeout=60.0  # Reduced timeout
                )
                
                self.llm_engine_pool[engine_key] = engine
                
                # FIXED: Use proper GPU manager reservation
                self.gpu_manager.reserve_gpu_for_workload(
                    workload_type=f"llm_engine_{engine_key}",
                    preferred_gpu=gpu_id,
                    duration_estimate=3600,
                    allow_sharing=True
                )
                
                logger.info(f"✅ LLM engine created on GPU {gpu_id}")
                return engine
                
            except Exception as e:
                logger.error(f"❌ Failed to create LLM engine for GPU {gpu_id}: {str(e)}")
                SafeGPUForcer.cleanup_gpu_memory_aggressive(gpu_id)
                # Cleanup on failure
                if engine_key in self.llm_engine_pool:
                    del self.llm_engine_pool[engine_key]
                
                raise

    async def _cleanup_gpu_engine_simple(self, gpu_id: int):
        """FIXED: Simplified cleanup"""
        engine_key = f"gpu_{gpu_id}"
        if engine_key in self.llm_engine_pool:
            del self.llm_engine_pool[engine_key]
            SafeGPUForcer.safe_gpu_cleanup(gpu_id)
            
            # Release from GPU manager
            self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
            await asyncio.sleep(1)

    async def _cleanup_gpu_engine_enhanced(self, gpu_id: int):
        """FIXED: Enhanced GPU engine cleanup"""
        engine_key = f"gpu_{gpu_id}"
        
        try:
            if engine_key in self.llm_engine_pool:
                logger.info(f"🧹 Cleaning up LLM engine on GPU {gpu_id}")
                
                # Remove from pool first
                engine = self.llm_engine_pool.pop(engine_key, None)
                
                # FIXED: Use SafeGPUForcer for cleanup
                SafeGPUForcer.force_gpu_environment(gpu_id, cleanup_first=False)
                SafeGPUForcer.safe_gpu_cleanup(gpu_id)
                
                # FIXED: Use proper GPU manager release
                self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
                
                await asyncio.sleep(2)
                logger.info(f"✅ Cleanup completed for GPU {gpu_id}")
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed for GPU {gpu_id}: {e}")
        
    @asynccontextmanager
    async def get_agent_with_gpu(self, agent_type: str, preferred_gpu: Optional[int] = None):
        """FIXED: Enhanced agent allocation with proper cleanup"""
        start_time = time.time()
        allocated_gpu = None
        agent_key = None
        
        try:
            # GPU allocation logic (existing code)
            if preferred_gpu is not None:
                memory_info = SafeGPUForcer.check_gpu_memory_safe(preferred_gpu)
                if memory_info['is_available'] and memory_info['free_gb'] >= 2.0:
                    allocated_gpu = preferred_gpu
            
            if allocated_gpu is None:
                allocated_gpu = self.gpu_manager.get_optimal_gpu(
                    workload_type=agent_type,
                    min_memory_gb=2.0
                )
            
            if allocated_gpu is None:
                raise RuntimeError("No suitable GPU available")
            
            # Reserve GPU
            success = self.gpu_manager.reserve_gpu_for_workload(
                workload_type=f"{agent_type}_agent",
                preferred_gpu=allocated_gpu,
                duration_estimate=300,
                allow_sharing=True
            )
            
            if not success:
                self.logger.warning(f"GPU manager reservation failed for GPU {allocated_gpu}")
            
            # Get or create LLM engine
            llm_engine = await self.get_or_create_llm_engine(allocated_gpu)
            
            # Create/reuse agent
            agent_key = f"{agent_type}_gpu_{allocated_gpu}"
            
            if agent_key not in self.agents:
                self.agents[agent_key] = self._create_agent(agent_type, llm_engine, allocated_gpu)
                self.logger.info(f"🆕 Created agent {agent_key}")
            else:
                self.logger.info(f"♻️ Reusing agent {agent_key}")
            
            self.stats["dynamic_allocations"] += 1
            self.logger.info(f"✅ Allocated GPU {allocated_gpu} for {agent_type}")
            
            yield self.agents[agent_key], allocated_gpu
            
        except Exception as e:
            self.stats["failed_allocations"] += 1
            self.logger.error(f"❌ GPU allocation failed for {agent_type}: {e}")
            raise
            
        finally:
            # FIX: ALWAYS cleanup resources, even on exception
            if allocated_gpu is not None:
                try:
                    self.gpu_manager.release_gpu_workload(allocated_gpu, f"{agent_type}_agent")
                    allocation_duration = time.time() - start_time
                    self.logger.info(f"Released GPU {allocated_gpu} after {allocation_duration:.2f}s")
                except Exception as cleanup_error:
                    self.logger.warning(f"GPU cleanup failed: {cleanup_error}")
    
  
    
    def _is_gpu_available(self, gpu_id: int) -> bool:
        """Enhanced GPU availability check using SafeGPUForcer helpers"""
        try:
            # STEP 1: Basic physical check
            if not SafeGPUForcer.is_gpu_physically_available(gpu_id):
                return False
            
            # STEP 2: Check requirements
            requirements_check = SafeGPUForcer.check_gpu_meets_requirements(
                gpu_id, 
                min_memory_gb=2.0, 
                max_utilization=85.0
            )
            
            if not requirements_check['meets_requirements']:
                logger.debug(f"GPU {gpu_id}: {requirements_check['reason']}")
                return False
            
            # STEP 3: Check with GPU manager
            try:
                gpu_status_detailed = self.gpu_manager.get_gpu_status_detailed()
                gpu_key = f"gpu_{gpu_id}"
                
                if gpu_key in gpu_status_detailed:
                    gpu_status = gpu_status_detailed[gpu_key]
                    
                    # Check manager availability and workload limits
                    if (not gpu_status.get('is_available', False) or 
                        gpu_status.get('active_workloads', 10) >= 3):
                        return False
                        
            except Exception as e:
                logger.warning(f"GPU manager check failed for GPU {gpu_id}: {e}")
                # Continue with basic checks only
            
            logger.debug(f"GPU {gpu_id} is available: {requirements_check['reason']}")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking GPU {gpu_id} availability: {e}")
            return False
    
    def get_gpu_availability_summary(self) -> Dict[str, Any]:
        """Get summary of all GPU availability"""
        summary = {}
        
        for gpu_id in range(self.gpu_manager.total_gpu_count):
            requirements_check = SafeGPUForcer.check_gpu_meets_requirements(gpu_id)
            is_available = self._is_gpu_available(gpu_id)
            
            summary[f"gpu_{gpu_id}"] = {
                "available": is_available,
                "reason": requirements_check['reason'],
                "memory_free_gb": requirements_check['memory_free_gb'],
                "utilization_percent": requirements_check['utilization_percent']
            }
        
        return summary

    async def get_available_gpu_for_agent(self, agent_type: str, preferred_gpu: Optional[int] = None) -> Optional[int]:
        """Enhanced GPU allocation with better error handling"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        process_count = len(psutil.pids())
        
        if cpu_percent > 90 or memory_percent > 85 or process_count > 1200:
            raise RuntimeError("System under critical load - rejecting new tasks")
        
        # Use enhanced GPU forcer
        
        
        # Find optimal GPU avoiding GPU 0
        best_gpu = SafeGPUForcer.find_optimal_gpu(
            min_free_gb=3.0,  # Higher requirement
            exclude_gpu_0=True
        )
        
        if best_gpu is None:
            # Emergency: try GPU 0 with lower requirement
            best_gpu = SafeGPUForcer.find_optimal_gpu(
                min_free_gb=2.0,
                exclude_gpu_0=False
            )
        
        if best_gpu is None:
            raise RuntimeError("No suitable GPU available")
        
        return best_gpu   
    
    async def _cleanup_gpu_engine(self, gpu_id: int):
        """Standard GPU engine cleanup - KEEP IN COORDINATOR"""
        engine_key = f"gpu_{gpu_id}"
        
        try:
            if engine_key in self.llm_engine_pool:
                self.logger.info(f"Cleaning up LLM engine on GPU {gpu_id}")
                
                # Remove from pool first
                engine = self.llm_engine_pool.pop(engine_key, None)
                
                # FIXED: Use SafeGPUForcer methods
                SafeGPUForcer.force_gpu_environment(gpu_id, cleanup_first=True)
                
                # Release from GPU manager
                self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
                
                # Give a moment for cleanup
                await asyncio.sleep(1)
                
                self.logger.info(f"✅ GPU {gpu_id} cleaned up successfully")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up GPU {gpu_id}: {e}")

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

    def get_gpu_status_simple(self) -> Dict[str, Any]:
        """Get simple GPU status using GPU manager"""
        try:
            return self.gpu_manager.get_gpu_status_detailed()
        except Exception as e:
            logger.error(f"GPU status check failed: {e}")
            return {"error": str(e)}

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
                "timestamp": dt.now().isoformat()
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
        """Create agent instance with coordinator reference"""
        if agent_type == "code_parser":
            return CodeParserAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "vector_index":
            return VectorIndexAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "data_loader":
            return DataLoaderAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "lineage_analyzer":
            return LineageAnalyzerAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "logic_analyzer":
            return LogicAnalyzerAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "documentation":
            return DocumentationAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "db2_comparator":
            return DB2ComparatorAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                max_rows=self.config.max_db_rows,
                coordinator=self  # ADD coordinator reference
            )
        elif agent_type == "chat_agent":  # ADD THIS
            return OpulenceChatAgent(
                    coordinator=self,
                    llm_engine=llm_engine,
                    db_path=self.db_path,
                    gpu_id=gpu_id
                    
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
            elif file_path.suffix.lower() in ['.cpy', '.copy']:
                return await self._process_csv_file_with_storage(file_path)
            #elif re.search(r'^\s*\d+\s+[A-Z][A-Z0-9-]*\s+PIC', content.upper(), re.MULTILINE):  # ADD THIS LINE
            #    return await self._process_csv_file_with_storage(file_path) 
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
                    dt.now().isoformat()
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
    
    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process chat query using the chat agent"""
        try:
            async with self.get_agent_with_gpu("chat_agent") as (chat_agent, gpu_id):
                result = await chat_agent.process_chat_query(query, conversation_history)
                result["gpu_used"] = gpu_id
                
                # Update statistics
                self.stats["total_queries"] += 1
                
                return result
                
        except Exception as e:
            self.logger.error(f"Chat query processing failed: {str(e)}")
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try rephrasing your question", "Check if the system is properly initialized"]
            }

    async def get_chat_agent_status(self) -> Dict[str, Any]:
        """Get chat agent status and capabilities"""
        try:
            async with self.get_agent_with_gpu("chat_agent") as (chat_agent, gpu_id):
                return {
                    "status": "available",
                    "gpu_id": gpu_id,
                    "capabilities": [
                        "Component analysis conversations",
                        "Lineage tracing discussions", 
                        "Code pattern searches",
                        "Impact analysis explanations",
                        "Technical documentation assistance"
                    ],
                    "supported_queries": [
                        "Natural language component analysis",
                        "Conversational lineage tracing",
                        "Interactive code exploration",
                        "Guided impact assessment"
                    ]
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        
    async def process_regular_chat_query(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Process regular chat query using the intelligent chat agent"""
        try:
            result = await self.process_chat_query(query, conversation_history)
            
            if isinstance(result, dict):
                response = result.get("response", "")
                
                # Add suggestions if available
                suggestions = result.get("suggestions", [])
                if suggestions:
                    response += "\n\n💡 **Suggestions:**\n"
                    for suggestion in suggestions[:3]:
                        response += f"• {suggestion}\n"
                
                return response
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"Chat query processing failed: {str(e)}")
            return f"❌ I encountered an error processing your query: {str(e)}"

    # 5. Add enhanced chat capabilities

    async def get_conversation_summary(self, conversation_history: List[Dict]) -> str:
        """Generate a summary of the conversation using the chat agent"""
        try:
            if not conversation_history:
                return "No conversation to summarize."
            
            # Create a summary query
            summary_query = "Please summarize our conversation and the key points discussed."
            
            async with self.get_agent_with_gpu("chat_agent") as (chat_agent, gpu_id):
                result = await chat_agent.process_chat_query(summary_query, conversation_history)
                return result.get("response", "Unable to generate summary.")
                
        except Exception as e:
            self.logger.error(f"Conversation summary failed: {str(e)}")
            return f"Error generating summary: {str(e)}"

    async def suggest_follow_up_questions(self, last_query: str, last_response: str) -> List[str]:
        """Suggest follow-up questions based on the conversation"""
        try:
            # Use the chat agent to generate intelligent follow-ups
            follow_up_query = f"Based on the previous question '{last_query}' and response, suggest 3 relevant follow-up questions a user might ask."
            
            conversation_context = [
                {"role": "user", "content": last_query},
                {"role": "assistant", "content": last_response}
            ]
            
            async with self.get_agent_with_gpu("chat_agent") as (chat_agent, gpu_id):
                result = await chat_agent.process_chat_query(follow_up_query, conversation_context)
                
                # Extract suggestions from response
                response = result.get("response", "")
                suggestions = result.get("suggestions", [])
                
                # If no structured suggestions, try to extract from response
                if not suggestions and response:
                    # Simple extraction of questions from response
                    import re
                    questions = re.findall(r'[0-9]+\.\s*([^?\n]+\?)', response)
                    suggestions = questions[:3]
                
                return suggestions[:3] if suggestions else [
                    "Tell me more about this component",
                    "Show me the impact analysis", 
                    "Find similar components"
                ]
                
        except Exception as e:
            self.logger.error(f"Follow-up suggestion failed: {str(e)}")
            return ["Analyze another component", "Search for code patterns", "Check system status"]

    # 6. Add chat-enhanced component analysis

    async def chat_analyze_component(self, component_name: str, user_question: str = None, 
                                    conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze component with chat-enhanced explanations"""
        try:
            # First get the regular analysis
            analysis_result = await self.analyze_component(component_name, "auto-detect")
            
            # Then get chat-enhanced explanation
            if user_question:
                chat_query = f"Explain the analysis of {component_name}. {user_question}"
            else:
                chat_query = f"Provide a detailed explanation of {component_name} based on the analysis."
            
            # Create context with analysis results
            enhanced_history = conversation_history or []
            enhanced_history.append({
                "role": "system",
                "content": f"Analysis data for {component_name}: {json.dumps(analysis_result, default=str)}"
            })
            
            chat_result = await self.process_chat_query(chat_query, enhanced_history)
            
            # Combine results
            return {
                "component_name": component_name,
                "analysis": analysis_result,
                "chat_explanation": chat_result.get("response", ""),
                "suggestions": chat_result.get("suggestions", []),
                "response_type": "enhanced_analysis"
            }
            
        except Exception as e:
            self.logger.error(f"Chat-enhanced analysis failed: {str(e)}")
            return {
                "component_name": component_name,
                "analysis": analysis_result if 'analysis_result' in locals() else {},
                "error": str(e),
                "response_type": "error"
            }

    # 7. Add chat-enhanced search

    async def chat_search_patterns(self, search_description: str, 
                                conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Search for code patterns with chat-enhanced results"""
        try:
            # Use vector search first
            async with self.get_agent_with_gpu("vector_index") as (vector_agent, gpu_id):
                search_results = await vector_agent.search_by_functionality(search_description, top_k=10)
            
            # Get chat explanation of results
            chat_query = f"Explain these search results for '{search_description}' and help me understand what was found."
            
            # Create context with search results
            search_context = [
                {
                    "role": "system", 
                    "content": f"Search results for '{search_description}': {json.dumps(search_results[:5], default=str)}"
                }
            ]
            if conversation_history:
                search_context.extend(conversation_history)
            
            chat_result = await self.process_chat_query(chat_query, search_context)
            
            return {
                "search_description": search_description,
                "search_results": search_results,
                "chat_explanation": chat_result.get("response", ""),
                "total_found": len(search_results),
                "suggestions": chat_result.get("suggestions", []),
                "response_type": "enhanced_search"
            }
            
        except Exception as e:
            self.logger.error(f"Chat-enhanced search failed: {str(e)}")
            return {
                "search_description": search_description,
                "search_results": [],
                "error": str(e),
                "response_type": "error"
            }
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
        """Enhanced component analysis with detailed logging and debugging"""
        start_time = time.time()
        analysis_result = {
        "component_name": component_name,
        "component_type": component_type or "unknown",
        "status": "initializing",
        "timestamp": dt.now().isoformat(),
        "processing_time": 0.0
        }
        
        try:
            self.logger.info(f"🔍 Starting analysis for component: {component_name}, type: {component_type}")
            
            # STEP 1: Verify database connection and data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if component exists in database
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            chunk_count = cursor.fetchone()[0]
            self.logger.info(f"📊 Found {chunk_count} chunks for {component_name}")
            
            # Get available tables for debugging
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if chunk_count == 0:
                # Try content search as fallback
                cursor.execute("""
                    SELECT COUNT(*) FROM program_chunks 
                    WHERE content LIKE ? OR metadata LIKE ?
                """, (f"%{component_name}%", f"%{component_name}%"))
                
                content_matches = cursor.fetchone()[0]
                self.logger.info(f"📊 Found {content_matches} content matches for {component_name}")
                
                if content_matches == 0:
                    conn.close()
                    return {
                        "component_name": component_name,
                        "component_type": component_type or "unknown",
                        "status": "error",
                        "error": f"Component '{component_name}' not found in database",
                        "suggestion": "Check component name spelling or ensure files were processed",
                        "debug_info": {
                            "available_tables": tables,
                            "chunk_count": chunk_count,
                            "content_matches": content_matches
                        }
                    }
            
            conn.close()
            
            # STEP 2: Determine component type with database-first approach
            if not component_type or component_type == "auto-detect":
                component_type = await self._determine_component_type_fixed(component_name, preferred_gpu)
                self.logger.info(f"🎯 Determined component type: {component_type}")
            
            # STEP 3: Initialize analysis result
            analysis_result = {
                "component_name": component_name,
                "component_type": component_type,
                "status": "processing",
                "chunks_found": chunk_count,
                "timestamp": dt.now().isoformat(),
                "debug_info": {
                    "available_tables": tables,
                    "chunk_count": chunk_count,
                    "determined_type": component_type
                }
            }
            
            # STEP 4: Component-specific analysis with proper error handling
            analysis_success = False
            
            if component_type in ["field"]:
                try:
                    self.logger.info(f"🔧 Starting lineage analysis for field {component_name}")
                    async with self.get_agent_with_gpu("lineage_analyzer", preferred_gpu) as (lineage_agent, gpu_id):
                        lineage_result = await lineage_agent.analyze_field_lineage(component_name)
                        analysis_result["lineage"] = lineage_result
                        analysis_result["lineage_gpu_used"] = gpu_id
                        analysis_success = True
                        self.logger.info(f"✅ Lineage analysis completed: {lineage_result.get('status', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"❌ Lineage analysis failed: {e}")
                    analysis_result["lineage"] = {"error": str(e), "status": "error"}
            
            elif component_type in ["program", "cobol"]:
                try:
                    self.logger.info(f"🔧 Starting logic analysis for program {component_name}")
                    async with self.get_agent_with_gpu("logic_analyzer", preferred_gpu) as (logic_agent, gpu_id):
                        logic_result = await logic_agent.analyze_program(component_name)
                        analysis_result["logic_analysis"] = logic_result
                        analysis_result["logic_gpu_used"] = gpu_id
                        analysis_success = True
                        self.logger.info(f"✅ Logic analysis completed: {logic_result.get('status', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"❌ Logic analysis failed: {e}")
                    analysis_result["logic_analysis"] = {"error": str(e), "status": "error"}
            
            elif component_type in ["jcl"]:
                try:
                    self.logger.info(f"🔧 Starting JCL analysis for {component_name}")
                    async with self.get_agent_with_gpu("lineage_analyzer", preferred_gpu) as (lineage_agent, gpu_id):
                        jcl_result = await lineage_agent.analyze_full_lifecycle(component_name, "jcl")
                        analysis_result["jcl_analysis"] = jcl_result
                        analysis_result["jcl_gpu_used"] = gpu_id
                        analysis_success = True
                        self.logger.info(f"✅ JCL analysis completed: {jcl_result.get('status', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"❌ JCL analysis failed: {e}")
                    analysis_result["jcl_analysis"] = {"error": str(e), "status": "error"}
            
            # STEP 5: Always try semantic search for additional context
            try:
                self.logger.info(f"🔧 Starting semantic search for {component_name}")
                async with self.get_agent_with_gpu("vector_index", preferred_gpu) as (vector_agent, gpu_id):
                    semantic_results = await vector_agent.search_similar_components(component_name)
                    analysis_result["semantic_search"] = semantic_results
                    analysis_result["vector_gpu_used"] = gpu_id
                    self.logger.info(f"✅ Semantic search completed")
            except Exception as e:
                self.logger.error(f"❌ Semantic search failed: {e}")
                analysis_result["semantic_search"] = {"error": str(e), "status": "error"}
            
            # STEP 6: Always provide basic database info
            try:
                basic_info = await self._get_basic_component_info_fixed(component_name)
                analysis_result["basic_info"] = basic_info
                self.logger.info(f"✅ Basic info retrieved")
            except Exception as e:
                self.logger.error(f"❌ Basic info failed: {e}")
                analysis_result["basic_info"] = {"error": str(e)}
            
            # STEP 7: Finalize result
            analysis_result["status"] = "completed" if analysis_success else "partial"
            analysis_result["processing_time"] = time.time() - start_time
            
            self.logger.info(f"🎯 Component analysis completed for {component_name}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Component analysis failed for {component_name}: {str(e)}")
            return {
                "component_name": component_name,
                "component_type": component_type or "unknown",
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": time.time() - start_time
            }

    # 4. FIX: Add the helper methods to opulence_coordinator.py

    async def _determine_component_type_fixed(self, component_name: str, preferred_gpu: Optional[int] = None) -> str:
        """Fixed component type determination with database-first approach"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            self.logger.info(f"🔍 Determining type for {component_name}")
            
            # Check in program_chunks for exact match
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count 
                FROM program_chunks 
                WHERE program_name = ?
                GROUP BY chunk_type
                ORDER BY count DESC
            """, (component_name,))
            
            chunk_types = cursor.fetchall()
            
            if chunk_types:
                # Determine type based on chunk types found
                dominant_chunk_type = chunk_types[0][0]  # Most common chunk type
                
                # Map chunk types to component types
                if any('job' in ct.lower() for ct, _ in chunk_types):
                    conn.close()
                    self.logger.info(f"🎯 Determined type: jcl (found job chunks)")
                    return "jcl"
                elif any(ct in ['working_storage', 'procedure_division', 'data_division'] for ct, _ in chunk_types):
                    conn.close()
                    self.logger.info(f"🎯 Determined type: program (found COBOL chunks)")
                    return "program"
                else:
                    conn.close()
                    self.logger.info(f"🎯 Determined type: program (found code chunks)")
                    return "program"
            
            # If no exact program match, check if it might be a field
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks
                WHERE content LIKE ? OR metadata LIKE ?
                LIMIT 1
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            content_matches = cursor.fetchone()[0]
            
            if content_matches > 0:
                # Check if it looks like a field name (all uppercase, contains underscore, etc.)
                if (component_name.isupper() and 
                    ('_' in component_name or len(component_name) <= 20)):
                    conn.close()
                    self.logger.info(f"🎯 Determined type: field (found in content, looks like field)")
                    return "field"
            
            conn.close()
            
            # Default to program if found in database but type unclear
            self.logger.info(f"🎯 Determined type: program (default)")
            return "program"
            
        except Exception as e:
            self.logger.error(f"❌ Component type determination failed: {e}")
            conn.close()
            return "program"  # Safe default

    async def _get_basic_component_info_fixed(self, component_name: str) -> Dict[str, Any]:
        """Get basic component info with better error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check program_chunks
            cursor.execute("""
                SELECT chunk_type, chunk_id, 
                    LENGTH(content) as content_length,
                    CASE WHEN metadata IS NOT NULL THEN 1 ELSE 0 END as has_metadata
                FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
                ORDER BY chunk_id
            """, (component_name, f"%{component_name}%"))
            
            chunk_info = cursor.fetchall()
            
            # Summarize chunk info
            chunk_summary = {}
            total_content = 0
            metadata_count = 0
            
            for chunk_type, chunk_id, content_length, has_metadata in chunk_info:
                if chunk_type not in chunk_summary:
                    chunk_summary[chunk_type] = 0
                chunk_summary[chunk_type] += 1
                total_content += content_length or 0
                metadata_count += has_metadata
            
            # Check file_metadata if available
            file_info = []
            try:
                cursor.execute("""
                    SELECT file_name, file_type, table_name 
                    FROM file_metadata 
                    WHERE file_name LIKE ? OR table_name LIKE ?
                """, (f"%{component_name}%", f"%{component_name}%"))
                
                file_info = cursor.fetchall()
            except sqlite3.OperationalError:
                # Table doesn't exist
                pass
            
            conn.close()
            
            return {
                "chunk_summary": chunk_summary,
                "total_chunks": len(chunk_info),
                "total_content_length": total_content,
                "chunks_with_metadata": metadata_count,
                "file_info": file_info,
                "found_in_chunks": len(chunk_info) > 0,
                "found_in_files": len(file_info) > 0,
                "analysis_timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Basic component info failed: {e}")
            return {"error": str(e)}

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
        """Determine component type with database checks first"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check what tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check in program_chunks first
            cursor.execute("""
                SELECT COUNT(*), chunk_type FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
                GROUP BY chunk_type
            """, (component_name, f"%{component_name}%"))
            
            chunk_results = cursor.fetchall()
            
            if chunk_results:
                # Determine type based on chunk types found
                chunk_types = [row[1] for row in chunk_results]
                if any('job' in ct.lower() for ct in chunk_types):
                    conn.close()
                    return "jcl"
                elif any(ct in ['working_storage', 'procedure_division'] for ct in chunk_types):
                    conn.close()
                    return "program"
            
            # Check if it might be a field (search in content)
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks
                WHERE content LIKE ? OR metadata LIKE ?
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            field_count = cursor.fetchone()[0]
            if field_count > 0:
                conn.close()
                return "field"
            
            # Check file_metadata if exists
            if 'file_metadata' in tables:
                cursor.execute("""
                    SELECT COUNT(*) FROM file_metadata 
                    WHERE table_name = ? OR file_name = ?
                """, (component_name, component_name))
                
                if cursor.fetchone()[0] > 0:
                    conn.close()
                    return "table"
        
        except Exception as e:
            self.logger.error(f"Error determining component type: {e}")
        
        finally:
            conn.close()
        
        # Fallback to LLM determination
        return await self._llm_determine_component_type(component_name, preferred_gpu)
    
    async def _llm_determine_component_type(self, component_name: str, preferred_gpu: Optional[int] = None) -> str:
        """Use LLM to determine component type as fallback"""
        prompt = f"""
        Analyze the component name '{component_name}' and determine its type.
        
        Types:
        - file: Data file or dataset
        - table: DB2 table
        - program: COBOL program
        - jcl: JCL job or procedure
        - field: Data field name
        
        Based on naming conventions, determine the most likely type.
        Respond with only the type name.
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=50, stop=["\n"])
        
        try:
            async with self.get_agent_with_gpu("logic_analyzer", preferred_gpu) as (agent, gpu_id):
                engine = await self.get_or_create_llm_engine(gpu_id)
                
                # Handle both old and new vLLM API
                try:
                    request_id = str(uuid.uuid4())
                    result = await engine.generate(prompt, sampling_params, request_id=request_id)
                except TypeError:
                    result = await engine.generate(prompt, sampling_params)
                
                return result.outputs[0].text.strip().lower()
        except Exception as e:
            self.logger.error(f"LLM component type determination failed: {e}")
            return "unknown"

    
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
        """Update processing statistics with proper error handling"""
        try:
            # Database insertion
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_stats (operation, duration, gpu_used, status)
                VALUES (?, ?, ?, ?)
            """, (operation, duration, gpu_id, "completed"))
            
            conn.commit()
            conn.close()
            
            # Update in-memory stats with proper initialization
            if "avg_response_time" not in self.stats:
                self.stats["avg_response_time"] = 0.0
                self.stats["response_count"] = 0
            
            # Proper running average
            self.stats["response_count"] += 1
            count = self.stats["response_count"]
            current_avg = self.stats["avg_response_time"]
            self.stats["avg_response_time"] = ((current_avg * (count - 1)) + duration) / count
            
            # Update operation-specific stats
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = duration
            
            self.logger.debug(f"Stats updated: {operation} took {duration:.2f}s, avg now {self.stats['avg_response_time']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to update processing stats: {e}")
            self.stats["failed_operations"] += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Use HealthMonitor instead of manual checks"""
        try:
            # Use the health monitor we properly initialized
            return self.health_monitor.get_current_status()
            
        except Exception as e:
            logger.error(f"Health status failed: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": dt.now().isoformat()
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
            
            # FIXED: Create workload distribution manually
            workload_distribution = {}
            for gpu_id, workloads in self.gpu_manager.active_workloads.items():
                workload_distribution[f"gpu_{gpu_id}"] = workloads
            
            optimization_suggestions = []
            
            # Analyze GPU utilization patterns
            for gpu_key, status in gpu_status.items():
                gpu_num = int(gpu_key.split('_')[1])
                
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
                workload_counts = [len(workloads) for workloads in workload_distribution.values()]
                if workload_counts:  # Check if list is not empty
                    workload_balance = max(workload_counts) - min(workload_counts)
                    
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