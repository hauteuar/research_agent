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
from agents.code_parser_agent import CodeParserAgent
from agents.vector_index_agent import VectorIndexAgent  
from agents.data_loader_agent import DataLoaderAgent
from agents.lineage_analyzer_agent import LineageAnalyzerAgent
from agents.logic_analyzer_agent import LogicAnalyzerAgent
from agents.documentation_agent import DocumentationAgent
from agents.db2_comparator_agent import DB2ComparatorAgent
from utils.enhanced_gpu_manager import DynamicGPUManager, GPUContext
from utils.dynamic_config_manager import DynamicConfigManager, get_dynamic_config, GPUConfig
from utils.health_monitor import HealthMonitor
from utils.cache_manager import CacheManager

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
        self.gpu_manager = DynamicGPUManager(
            total_gpu_count=gpu_config.total_gpu_count,
            memory_threshold=gpu_config.memory_threshold,
            utilization_threshold=gpu_config.utilization_threshold
        )
        
        self.health_monitor = HealthMonitor()
        
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
            """)
            
            # FIX 14: Explicit commit for table creation
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully with proper configuration")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    async def get_or_create_llm_engine(self, gpu_id: int, force_reload: bool = False) -> AsyncLLMEngine:
        """Get or create LLM engine for specific GPU with availability check and reload capability"""
        async with self.engine_lock:
            engine_key = f"gpu_{gpu_id}"
            
            # If engine exists and not forcing reload, return it
            if engine_key in self.llm_engine_pool and not force_reload:
                self.logger.info(f"Reusing existing LLM engine on GPU {gpu_id}")
                return self.llm_engine_pool[engine_key]
            
            try:
                # Import the GPU forcer
                from gpu_force_fix import GPUForcer
                
                # CHECK GPU AVAILABILITY FIRST
                if not self._is_gpu_available(gpu_id):
                    raise RuntimeError(f"GPU {gpu_id} is not available or already occupied")
                
                # Check memory before attempting
                memory_info = GPUForcer.check_gpu_memory(gpu_id)
                free_gb = memory_info['free_gb']
                
                # If GPU has existing model and insufficient memory, try to clean it up
                if free_gb < 2.0:
                    self.logger.warning(f"GPU {gpu_id} has insufficient memory: {free_gb:.1f}GB free")
                    
                    # Try to cleanup existing model on this GPU
                    if engine_key in self.llm_engine_pool:
                        self.logger.info(f"Cleaning up existing model on GPU {gpu_id}")
                        await self._cleanup_gpu_engine(gpu_id)
                        
                        # Check memory again after cleanup
                        memory_info = GPUForcer.check_gpu_memory(gpu_id)
                        free_gb = memory_info['free_gb']
                        
                    if free_gb < 2.0:
                        raise RuntimeError(f"GPU {gpu_id} still has insufficient memory after cleanup: {free_gb:.1f}GB free (need at least 2GB)")
                
                self.logger.info(f"Creating LLM engine on GPU {gpu_id} with {free_gb:.1f}GB free memory")
                
                # AGGRESSIVE GPU FORCING - This changes the entire CUDA environment
                original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                
                try:
                    # Force GPU environment - this makes CUDA only see our target GPU
                    GPUForcer.force_gpu_environment(gpu_id)
                    
                    # Now create engine args (device 0 now maps to our target GPU)
                    engine_args = GPUForcer.create_vllm_engine_args(
                        self.config.model_name, 
                        self.config.max_tokens
                    )
                    
                    self.logger.info(f"Creating VLLM engine with forced GPU environment...")
                    
                    # Create the engine - it will use the GPU forced by environment
                    engine = AsyncLLMEngine.from_engine_args(engine_args)
                    
                    self.llm_engine_pool[engine_key] = engine
                    
                    # Mark GPU as occupied in our manager  
                    self.gpu_manager.reserve_gpu_for_workload(
                        workload_type=f"llm_engine_{engine_key}",
                        preferred_gpu=gpu_id,
                        duration_estimate=3600  # 1 hour estimated
                    )
                    
                    # Verify final memory usage
                    final_memory = GPUForcer.check_gpu_memory(gpu_id)
                    final_free_gb = final_memory['free_gb']
                    used_gb = free_gb - final_free_gb
                    
                    self.logger.info(f"✅ LLM engine created on GPU {gpu_id}. Used {used_gb:.1f}GB, {final_free_gb:.1f}GB remaining")
                    
                finally:
                    # Restore original CUDA_VISIBLE_DEVICES if it existed
                    if original_cuda_visible is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
                
            except Exception as e:
                self.logger.error(f"Failed to create LLM engine for GPU {gpu_id}: {str(e)}")
                
                # Clean up on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Remove from pool if it was partially created
                if engine_key in self.llm_engine_pool:
                    del self.llm_engine_pool[engine_key]
                
                # Release GPU workload on failure
                try:
                    self.gpu_manager.release_gpu_workload(gpu_id, f"llm_engine_{engine_key}")
                except:
                    pass  # Ignore errors during cleanup
                
                raise
            
            return self.llm_engine_pool[engine_key]
    
    @asynccontextmanager
    async def get_agent_with_gpu(self, agent_type: str, preferred_gpu: Optional[int] = None):
        """Context manager to get agent with smart GPU allocation using GPU manager"""
        start_time = time.time()
        allocated_gpu = None
        
        try:
            # Use GPU manager's workload reservation system
            allocated_gpu = self.gpu_manager.reserve_gpu_for_workload(
                workload_type=f"{agent_type}_agent",
                preferred_gpu=preferred_gpu,
                duration_estimate=300  # 5 minutes default
            )
            
            if allocated_gpu is None:
                # Fallback to our manual allocation
                allocated_gpu = await self.get_available_gpu_for_agent(agent_type, preferred_gpu)
            
            if allocated_gpu is None:
                self.stats["failed_allocations"] += 1
                
                # Get detailed status for better error message
                gpu_status = self.gpu_manager.get_gpu_status_detailed()
                available_info = {
                    gpu_id: {
                        "free_gb": status.get("memory_free_gb", 0),
                        "utilization": status.get("utilization_percent", 100),
                        "available": status.get("is_available", False)
                    }
                    for gpu_id, status in gpu_status.items()
                }
                
                raise RuntimeError(f"No available GPU for {agent_type}. GPU Status: {available_info}")
            
            self.stats["dynamic_allocations"] += 1
            
            # Get or create LLM engine for this GPU
            llm_engine = await self.get_or_create_llm_engine(allocated_gpu)
            
            # Create or get agent
            agent_key = f"{agent_type}_gpu_{allocated_gpu}"
            
            if agent_key not in self.agents:
                self.agents[agent_key] = self._create_agent(agent_type, llm_engine, allocated_gpu)
            
            # Log allocation
            self._log_gpu_allocation(agent_type, allocated_gpu, preferred_gpu, True, time.time() - start_time)
            
            self.logger.info(f"✅ Allocated GPU {allocated_gpu} for {agent_type} (preferred: {preferred_gpu})")
            
            yield self.agents[agent_key], allocated_gpu
            
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
        """Get best available GPU for agent with smart allocation using GPU manager"""
    
        # Force refresh GPU status first
        self.gpu_manager.force_refresh()
    
        # Check if we already have an LLM engine that can be shared
        existing_engines = list(self.llm_engine_pool.keys())
        if existing_engines:
            # Try to reuse existing engine if memory allows
            for engine_key in existing_engines:
                gpu_id = int(engine_key.split('_')[1])
                try:
                    from gpu_force_fix import GPUForcer
                    memory_info = GPUForcer.check_gpu_memory(gpu_id)
                    free_gb = memory_info.get('free_gb', 0)
                
                # If GPU has enough free memory for another workload
                    if free_gb >= 1.0:  # Reduced threshold for sharing
                        self.logger.info(f"Reusing existing LLM engine on GPU {gpu_id} for {agent_type}")
                        return gpu_id
                except Exception as e:
                    self.logger.warning(f"Error checking GPU {gpu_id} for reuse: {e}")
                    continue
    
        # Use GPU manager's built-in allocation logic
        best_gpu = self.gpu_manager.get_available_gpu(
            preferred_gpu=preferred_gpu, 
            fallback=True
        )
    
        if best_gpu is not None:
            self.logger.info(f"GPU manager selected GPU {best_gpu} for {agent_type}")
            return best_gpu
    
        # If no GPU available, check if we can wait and retry
        await asyncio.sleep(2)  # Wait 2 seconds
        self.gpu_manager.force_refresh()
    
        best_gpu = self.gpu_manager.get_available_gpu(
            preferred_gpu=preferred_gpu, 
            fallback=True
        )
    
        if best_gpu is not None:
            self.logger.info(f"GPU manager selected GPU {best_gpu} for {agent_type} (after retry)")
            return best_gpu
    
        self.logger.warning("No GPUs currently available even after retry")
        return None


    async def _cleanup_gpu_engine(self, gpu_id: int):
        """Clean up existing LLM engine on GPU using GPUForcer"""
        engine_key = f"gpu_{gpu_id}"
        
        try:
            if engine_key in self.llm_engine_pool:
                self.logger.info(f"Cleaning up LLM engine on GPU {gpu_id}")
                
                # Remove from pool first
                engine = self.llm_engine_pool.pop(engine_key, None)
                
                # Force GPU cleanup using GPUForcer
                from gpu_force_fix import GPUForcer
                
                # Set GPU environment for cleanup
                original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                try:
                    GPUForcer.force_gpu_environment(gpu_id)
                    
                    if torch.cuda.is_available():
                        with torch.cuda.device(0):  # 0 now maps to our target GPU
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            
                finally:
                    # Restore original environment
                    if original_cuda_visible is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
                
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
    
    # Fix 1: Update process_batch_files in opulence_coordinator.py
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
        
            # FIX 3: Verify database storage - ADD THIS LINE
            await self._verify_database_storage(file_paths)

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
                "database_verification": await self._get_database_stats()
            }
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "files_processed": 0
                }

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
            raise
        
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
    
    async def analyze_component(self, component_name: str, component_type: str = None, 
                               preferred_gpu: Optional[int] = None) -> Dict[str, Any]:
        """Deep analysis of a component with dynamic GPU allocation"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"analyze_{component_name}_{component_type}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.stats["cache_hit_rate"] += 1
                return cached_result
            
            # If component type not specified, determine it
            if not component_type:
                component_type = await self._determine_component_type(component_name, preferred_gpu)
            
            analysis_result = {}
            
            if component_type in ["file", "table"]:
                # Analyze file/table lifecycle with dynamic GPU allocation
                async with self.get_agent_with_gpu("lineage_analyzer", preferred_gpu) as (lineage_agent, gpu1):
                    async with self.get_agent_with_gpu("data_loader", preferred_gpu) as (data_agent, gpu2):
                        lineage_result = await lineage_agent.analyze_field_lineage(component_name)
                        data_result = await data_agent.get_component_info(component_name)
                
                analysis_result = {
                    "component_name": component_name,
                    "component_type": component_type,
                    "lineage": lineage_result,
                    "data_info": data_result,
                    "lifecycle": await self._analyze_lifecycle(component_name, component_type, preferred_gpu),
                    "gpus_used": [gpu1, gpu2]
                }
                
            elif component_type in ["program", "cobol"]:
                # Analyze program logic with dynamic GPU allocation
                async with self.get_agent_with_gpu("logic_analyzer", preferred_gpu) as (logic_agent, gpu_id):
                    logic_result = await logic_agent.analyze_program(component_name)
                    dependencies = await self._find_dependencies(component_name, preferred_gpu)
                    
                analysis_result = {
                    "component_name": component_name,
                    "component_type": component_type,
                    "logic_analysis": logic_result,
                    "dependencies": dependencies,
                    "gpu_used": gpu_id
                }
                
            elif component_type == "jcl":
                # Analyze JCL job flow with dynamic GPU allocation
                async with self.get_agent_with_gpu("code_parser", preferred_gpu) as (parser_agent, gpu_id):
                    jcl_result = await parser_agent.analyze_jcl(component_name)
                    job_flow = await self._analyze_job_flow(component_name, preferred_gpu)
                    
                analysis_result = {
                    "component_name": component_name,
                    "component_type": component_type,
                    "jcl_analysis": jcl_result,
                    "job_flow": job_flow,
                    "gpu_used": gpu_id
                }
            
            # Add comparison with DB2 if applicable
            if component_type in ["file", "table"]:
                async with self.get_agent_with_gpu("db2_comparator", preferred_gpu) as (db2_agent, gpu_id):
                    db2_comparison = await db2_agent.compare_data(component_name)
                    analysis_result["db2_comparison"] = db2_comparison
                    analysis_result["db2_gpu_used"] = gpu_id
            try:
                async with self.get_agent_with_gpu("vector_index", preferred_gpu) as (vector_agent, gpu_id):
                    semantic_results = await vector_agent.search_similar_components(component_name)
                    analysis_result["semantic_search"] = semantic_results
                    analysis_result["vector_gpu_used"] = gpu_id
            except Exception as e:
                self.logger.warning(f"Semantic search failed for {component_name}: {str(e)}")
                analysis_result["semantic_search"] = {"error": str(e)}

            # Cache the result
            self.cache_manager.set(cache_key, analysis_result)
            
            processing_time = time.time() - start_time
            self._update_processing_stats("component_analysis", processing_time)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Component analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
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
        """Use LLM to determine component type with dynamic GPU allocation"""
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
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50,
            stop=["\n"]
        )
        
        # Use dynamic GPU allocation for component type determination
        async with self.get_agent_with_gpu("logic_analyzer", preferred_gpu) as (agent, gpu_id):
            # Get the LLM engine from the agent
            engine = await self.get_or_create_llm_engine(gpu_id)
            result = await engine.generate(prompt, sampling_params)
            
            return result.outputs[0].text.strip().lower()
    
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