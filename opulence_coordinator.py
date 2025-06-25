# opulence_coordinator.py
"""
Opulence - Deep Research Mainframe Agent Coordinator
Handles GPU distribution, agent orchestration, and parallel processing
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
from utils.gpu_manager import GPUManager
from utils.health_monitor import HealthMonitor
from utils.cache_manager import CacheManager

@dataclass
class OpulenceConfig:
    """Configuration for Opulence system"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 4096
    temperature: float = 0.1
    gpu_count: int = 3
    max_processing_time: int = 900  # 15 minutes
    batch_size: int = 32
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600  # 1 hour

class OpulenceCoordinator:
    """Main coordinator for the Opulence deep research system"""
    
    def __init__(self, config: OpulenceConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.gpu_manager = GPUManager(config.gpu_count)
        self.health_monitor = HealthMonitor()
        self.cache_manager = CacheManager(config.cache_ttl)
        
        # Initialize SQLite database
        self.db_path = "opulence_data.db"
        self._init_database()
        
        # Initialize agents
        self.agents = {}
        self.llm_engines = {}
        self._init_agents()
        
        # Processing statistics
        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "avg_response_time": 0,
            "cache_hit_rate": 0,
            "gpu_utilization": {"gpu_0": 0, "gpu_1": 0, "gpu_2": 0}
        }
        
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
        """)
        
        conn.commit()
        conn.close()
    
    async def _init_agents(self):
        """Initialize all agents with GPU distribution"""
        try:
            # Distribute LLM engines across GPUs
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=1,
                max_model_len=self.config.max_tokens,
                gpu_memory_utilization=0.8
            )
            
            # Create LLM engines for different GPUs
            for gpu_id in range(self.config.gpu_count):
                with torch.cuda.device(gpu_id):
                    engine = AsyncLLMEngine.from_engine_args(engine_args)
                    self.llm_engines[f"gpu_{gpu_id}"] = engine
            
            # Initialize agents with assigned GPUs
            self.agents = {
                "code_parser": CodeParserAgent(
                    llm_engine=self.llm_engines["gpu_0"],
                    db_path=self.db_path,
                    gpu_id=0
                ),
                "vector_index": VectorIndexAgent(
                    llm_engine=self.llm_engines["gpu_1"],
                    db_path=self.db_path,
                    gpu_id=1
                ),
                "data_loader": DataLoaderAgent(
                    llm_engine=self.llm_engines["gpu_1"],
                    db_path=self.db_path,
                    gpu_id=1
                ),
                "lineage_analyzer": LineageAnalyzerAgent(
                    llm_engine=self.llm_engines["gpu_2"],
                    db_path=self.db_path,
                    gpu_id=2
                ),
                "logic_analyzer": LogicAnalyzerAgent(
                    llm_engine=self.llm_engines["gpu_0"],
                    db_path=self.db_path,
                    gpu_id=0
                ),
                "documentation": DocumentationAgent(
                    llm_engine=self.llm_engines["gpu_2"],
                    db_path=self.db_path,
                    gpu_id=2
                ),
                "db2_comparator": DB2ComparatorAgent(
                    llm_engine=self.llm_engines["gpu_1"],
                    db_path=self.db_path,
                    gpu_id=1,
                    max_rows=self.config.max_db_rows
                )
            }
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """Process multiple files in parallel across GPUs"""
        start_time = time.time()
        
        try:
            # Check processing time limit
            if time.time() - start_time > self.config.max_processing_time:
                raise TimeoutError("Processing time limit exceeded")
            
            # Distribute files across available GPUs
            tasks = []
            for i, file_path in enumerate(file_paths):
                gpu_id = i % self.config.gpu_count
                
                if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                    task = self.agents["code_parser"].process_file(file_path)
                elif file_type == "jcl" or file_path.suffix.lower() == '.jcl':
                    task = self.agents["code_parser"].process_file(file_path)
                elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                    task = self.agents["data_loader"].process_file(file_path)
                else:
                    # Auto-detect file type
                    task = self._auto_detect_and_process(file_path)
                
                tasks.append(task)
            
            # Execute tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
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
    
    async def analyze_component(self, component_name: str, component_type: str = None) -> Dict[str, Any]:
        """Deep analysis of a component (file, table, program, JCL)"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"analyze_{component_name}_{component_type}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.stats["cache_hit_rate"] += 1
                return cached_result
            
            # If component type not specified, ask LLM to determine
            if not component_type:
                component_type = await self._determine_component_type(component_name)
            
            analysis_result = {}
            
            if component_type in ["file", "table"]:
                # Analyze file/table lifecycle
                lineage_task = self.agents["lineage_analyzer"].analyze_field_lineage(component_name)
                data_task = self.agents["data_loader"].get_component_info(component_name)
                
                lineage_result, data_result = await asyncio.gather(lineage_task, data_task)
                
                analysis_result = {
                    "component_name": component_name,
                    "component_type": component_type,
                    "lineage": lineage_result,
                    "data_info": data_result,
                    "lifecycle": await self._analyze_lifecycle(component_name, component_type)
                }
                
            elif component_type in ["program", "cobol"]:
                # Analyze program logic
                logic_result = await self.agents["logic_analyzer"].analyze_program(component_name)
                analysis_result = {
                    "component_name": component_name,
                    "component_type": component_type,
                    "logic_analysis": logic_result,
                    "dependencies": await self._find_dependencies(component_name)
                }
                
            elif component_type == "jcl":
                # Analyze JCL job flow
                jcl_result = await self.agents["code_parser"].analyze_jcl(component_name)
                analysis_result = {
                    "component_name": component_name,
                    "component_type": component_type,
                    "jcl_analysis": jcl_result,
                    "job_flow": await self._analyze_job_flow(component_name)
                }
            
            # Add comparison with DB2 if applicable
            if component_type in ["file", "table"]:
                db2_comparison = await self.agents["db2_comparator"].compare_data(component_name)
                analysis_result["db2_comparison"] = db2_comparison
            
            # Cache the result
            self.cache_manager.set(cache_key, analysis_result)
            
            processing_time = time.time() - start_time
            self._update_processing_stats("component_analysis", processing_time)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Component analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _determine_component_type(self, component_name: str) -> str:
        """Use LLM to determine component type"""
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
        
        # Use first available GPU
        engine = self.llm_engines["gpu_0"]
        result = await engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip().lower()
    
    async def _analyze_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """Analyze complete lifecycle of a component"""
        return await self.agents["lineage_analyzer"].analyze_full_lifecycle(component_name, component_type)
    
    async def _find_dependencies(self, component_name: str) -> List[str]:
        """Find all dependencies for a component"""
        return await self.agents["lineage_analyzer"].find_dependencies(component_name)
    
    async def _analyze_job_flow(self, jcl_name: str) -> Dict[str, Any]:
        """Analyze JCL job flow"""
        return await self.agents["code_parser"].analyze_job_flow(jcl_name)
    
    async def _auto_detect_and_process(self, file_path: Path) -> Dict[str, Any]:
        """Auto-detect file type and process accordingly"""
        # Read first few lines to determine file type
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)
            
            if 'IDENTIFICATION DIVISION' in content or 'PROGRAM-ID' in content:
                return await self.agents["code_parser"].process_file(file_path)
            elif content.startswith('//') and 'JOB' in content:
                return await self.agents["code_parser"].process_file(file_path)
            elif ',' in content and '\n' in content:
                return await self.agents["data_loader"].process_file(file_path)
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
        """Get system health status"""
        return {
            "status": "healthy",
            "gpu_status": self.gpu_manager.get_gpu_status(),
            "memory_usage": self.health_monitor.get_memory_usage(),
            "processing_stats": self.stats,
            "active_agents": len(self.agents),
            "cache_stats": self.cache_manager.get_stats()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get processing statistics
        df_stats = pd.read_sql_query("""
            SELECT operation, AVG(duration) as avg_duration, COUNT(*) as count
            FROM processing_stats
            GROUP BY operation
        """, conn)
        
        # Get file statistics
        df_files = pd.read_sql_query("""
            SELECT file_type, processing_status, COUNT(*) as count
            FROM file_metadata
            GROUP BY file_type, processing_status
        """, conn)
        
        conn.close()
        
        return {
            "processing_stats": df_stats.to_dict('records'),
            "file_stats": df_files.to_dict('records'),
            "system_stats": self.stats
        }

# Global coordinator instance
coordinator = None

def get_coordinator() -> OpulenceCoordinator:
    """Get or create global coordinator instance"""
    global coordinator
    if coordinator is None:
        config = OpulenceConfig()
        coordinator = OpulenceCoordinator(config)
    return coordinator

async def initialize_system():
    """Initialize the Opulence system"""
    global coordinator
    if coordinator is None:
        config = OpulenceConfig()
        coordinator = OpulenceCoordinator(config)
        await coordinator._init_agents()
    return coordinator