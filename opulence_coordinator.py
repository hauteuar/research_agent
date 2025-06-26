# opulence_coordinator.py
"""
Opulence - Deep Research Mainframe Agent Coordinator with Dynamic GPU Allocation
Handles dynamic GPU distribution, agent orchestration, and parallel processing
"""

import asyncio
import logging
import time
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
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for metadata storage
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE,
                file_type TEXT,
                table_name TEXT,
                fields TEXT,
                source_type TEXT,
                last_modified TIMESTAMP,
                processing_status TEXT DEFAULT 'pending'
            );
            
            CREATE TABLE IF NOT EXISTS field_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT,
                program_name TEXT,
                paragraph TEXT,
                operation TEXT,
                source_file TEXT,
                last_used TIMESTAMP,
                read_in TEXT,
                updated_in TEXT,
                purged_in TEXT
            );
            
            CREATE TABLE IF NOT EXISTS program_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT,
                chunk_id TEXT,
                chunk_type TEXT,
                content TEXT,
                metadata TEXT,
                embedding_id TEXT
            );
            
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                operation TEXT,
                duration REAL,
                gpu_used INTEGER,
                status TEXT
            );
            
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
        """)
        
        conn.commit()
        conn.close()
    
    async def get_or_create_llm_engine(self, gpu_id: int) -> AsyncLLMEngine:
        """Get or create LLM engine for specific GPU with proper device management"""
        async with self.engine_lock:
            engine_key = f"gpu_{gpu_id}"
            
            if engine_key not in self.llm_engine_pool:
                try:
                    # Force GPU memory cleanup first
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Set CUDA device explicitly
                    torch.cuda.set_device(gpu_id)
                    
                    # Verify we're on the correct device
                    current_device = torch.cuda.current_device()
                    if current_device != gpu_id:
                        raise RuntimeError(f"Failed to set CUDA device to {gpu_id}, current device is {current_device}")
                    
                    # Get memory info for the target GPU
                    with torch.cuda.device(gpu_id):
                        memory_info = torch.cuda.mem_get_info()
                        free_memory_gb = memory_info[0] / (1024**3)
                        total_memory_gb = memory_info[1] / (1024**3)
                        
                        self.logger.info(f"GPU {gpu_id} memory: {free_memory_gb:.1f}GB free / {total_memory_gb:.1f}GB total")
                        
                        # Check if we have enough memory (at least 2GB)
                        if free_memory_gb < 2.0:
                            raise RuntimeError(f"GPU {gpu_id} has insufficient memory: {free_memory_gb:.1f}GB free (need at least 2GB)")
                    
                    # Create engine args with specific GPU
                    engine_args = AsyncEngineArgs(
                        model=self.config.model_name,
                        tensor_parallel_size=1,
                        max_model_len=self.config.max_tokens,
                        gpu_memory_utilization=0.7,  # More conservative memory usage
                        device=f"cuda:{gpu_id}",
                        trust_remote_code=True,
                        enforce_eager=True,  # Disable CUDA graphs for better memory management
                        disable_log_stats=True
                    )
                    
                    # Set environment variables to force device
                    import os
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    
                    self.logger.info(f"Creating LLM engine on GPU {gpu_id}...")
                    engine = AsyncLLMEngine.from_engine_args(engine_args)
                    
                    # Reset CUDA_VISIBLE_DEVICES
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    
                    self.llm_engine_pool[engine_key] = engine
                    
                    # Verify memory usage after engine creation
                    with torch.cuda.device(gpu_id):
                        final_memory_info = torch.cuda.mem_get_info()
                        final_free_gb = final_memory_info[0] / (1024**3)
                        used_gb = free_memory_gb - final_free_gb
                        
                        self.logger.info(f"LLM engine created on GPU {gpu_id}. Used {used_gb:.1f}GB, {final_free_gb:.1f}GB remaining")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create LLM engine for GPU {gpu_id}: {str(e)}")
                    
                    # Clean up on failure
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Remove from pool if it was partially created
                    if engine_key in self.llm_engine_pool:
                        del self.llm_engine_pool[engine_key]
                    
                    raise
            
            return self.llm_engine_pool[engine_key]
    
    @asynccontextmanager
    async def get_agent_with_gpu(self, agent_type: str, preferred_gpu: Optional[int] = None):
        """Context manager to get agent with dynamic GPU allocation"""
        start_time = time.time()
        allocated_gpu = None
        
        # Get agent-specific preferred GPU from config if not specified
        if preferred_gpu is None:
            agent_config = self.config_manager.get_agent_config(agent_type)
            preferred_gpu = agent_config.get("preferred_gpu")
        
        try:
            # Get available GPU
            with GPUContext(self.gpu_manager, f"{agent_type}_agent", preferred_gpu, 300) as gpu_id:
                if gpu_id is None:
                    self.stats["failed_allocations"] += 1
                    raise RuntimeError(f"No GPU available for {agent_type}")
                
                allocated_gpu = gpu_id
                self.stats["dynamic_allocations"] += 1
                
                # Get or create LLM engine for this GPU
                llm_engine = await self.get_or_create_llm_engine(gpu_id)
                
                # Create or get agent
                agent_key = f"{agent_type}_gpu_{gpu_id}"
                
                if agent_key not in self.agents:
                    self.agents[agent_key] = self._create_agent(agent_type, llm_engine, gpu_id)
                
                # Log allocation
                self._log_gpu_allocation(agent_type, gpu_id, preferred_gpu, True, time.time() - start_time)
                
                self.logger.info(f"Allocated GPU {gpu_id} for {agent_type} (preferred: {preferred_gpu})")
                
                yield self.agents[agent_key], gpu_id
                
        except Exception as e:
            if allocated_gpu is not None:
                self._log_gpu_allocation(agent_type, allocated_gpu, preferred_gpu, False, time.time() - start_time)
            raise
    
    def _create_agent(self, agent_type: str, llm_engine: AsyncLLMEngine, gpu_id: int):
        """Create agent instance"""
        if agent_type == "code_parser":
            return CodeParserAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id
            )
        elif agent_type == "vector_index":
            return VectorIndexAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id
            )
        elif agent_type == "data_loader":
            return DataLoaderAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id
            )
        elif agent_type == "lineage_analyzer":
            return LineageAnalyzerAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id
            )
        elif agent_type == "logic_analyzer":
            return LogicAnalyzerAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id
            )
        elif agent_type == "documentation":
            return DocumentationAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id
            )
        elif agent_type == "db2_comparator":
            return DB2ComparatorAgent(
                llm_engine=llm_engine,
                db_path=self.db_path,
                gpu_id=gpu_id,
                max_rows=self.config.max_db_rows
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
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """Process multiple files with dynamic GPU allocation"""
        start_time = time.time()
        
        try:
            # Check processing time limit
            if time.time() - start_time > self.config.max_processing_time:
                raise TimeoutError("Processing time limit exceeded")
            
            results = []
            
            # Process files concurrently with dynamic GPU allocation
            for file_path in file_paths:
                try:
                    if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                        async with self.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                            result = await agent.process_file(file_path)
                            results.append(result)
                            
                    elif file_type == "jcl" or file_path.suffix.lower() == '.jcl':
                        async with self.get_agent_with_gpu("code_parser") as (agent, gpu_id):
                            result = await agent.process_file(file_path)
                            results.append(result)
                            
                    elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                        async with self.get_agent_with_gpu("data_loader") as (agent, gpu_id):
                            result = await agent.process_file(file_path)
                            results.append(result)
                            
                    else:
                        # Auto-detect file type
                        result = await self._auto_detect_and_process(file_path)
                        results.append(result)
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    results.append({"status": "error", "file": str(file_path), "error": str(e)})
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_files_processed"] += len(file_paths)
            self._update_processing_stats("batch_processing", processing_time)
            
            return {
                "status": "success",
                "files_processed": len(file_paths),
                "processing_time": processing_time,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "files_processed": 0
            }
    
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
            
            # Cache the result
            self.cache_manager.set(cache_key, analysis_result)
            
            processing_time = time.time() - start_time
            self._update_processing_stats("component_analysis", processing_time)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Component analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
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
        """Shutdown the coordinator and all resources"""
        try:
            # Shutdown GPU manager
            self.gpu_manager.shutdown()
            
            # Clear LLM engine pool
            self.llm_engine_pool.clear()
            
            # Clear agents
            self.agents.clear()
            
            self.logger.info("Opulence Coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


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