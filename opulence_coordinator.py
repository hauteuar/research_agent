# opulence_coordinator_single_gpu.py
"""
Opulence - Single GPU Coordinator for Simplified and Reliable GPU Management
Uses one GPU exclusively for all operations - much simpler than multi-GPU coordination
"""

import traceback
import subprocess
import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import uuid
import sqlite3
from datetime import datetime as dt
from contextlib import asynccontextmanager
import re
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import pandas as pd

# Import the single GPU manager we created
from utils.single_gpu_manager import DualGPUManager, DualGPUConfig

# Import our agents with error handling
try:
    from agents.code_parser_agent import CodeParserAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("CodeParserAgent not available")
    CodeParserAgent = None

try:
    from agents.chat_agent import OpulenceChatAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("OpulenceChatAgent not available")
    OpulenceChatAgent = None

try:
    from agents.vector_index_agent import VectorIndexAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("VectorIndexAgent not available")
    VectorIndexAgent = None

try:
    from agents.data_loader_agent import DataLoaderAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("DataLoaderAgent not available")
    DataLoaderAgent = None

try:
    from agents.lineage_analyzer_agent import LineageAnalyzerAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("LineageAnalyzerAgent not available")
    LineageAnalyzerAgent = None

try:
    from agents.logic_analyzer_agent import LogicAnalyzerAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("LogicAnalyzerAgent not available")
    LogicAnalyzerAgent = None

try:
    from agents.documentation_agent import DocumentationAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("DocumentationAgent not available")
    DocumentationAgent = None

try:
    from agents.db2_comparator_agent import DB2ComparatorAgent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("DB2ComparatorAgent not available")
    DB2ComparatorAgent = None

# Set up logging FIRST
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opulence_single_gpu.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class DualGPUOpulenceConfig:
    """Configuration for Dual GPU Opulence system"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 1024
    temperature: float = 0.1
    exclude_gpu_0: bool = True  # Don't use GPU 0 in shared systems
    min_memory_gb: float = 6.0
    max_processing_time: int = 900
    batch_size: int = 8  # Per GPU
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600
    auto_cleanup: bool = True
    force_gpu_ids: Optional[List[int]] = None  # Force specific GPUs [1, 2]

class DualGPUOpulenceCoordinator:
    def __init__(self, config: DualGPUOpulenceConfig = None):
        self.config = config or DualGPUOpulenceConfig()
        self.logger = self._setup_logging()
        
        # Create dual GPU manager
        self.gpu_manager = DualGPUManager(self.config)
        self.selected_gpus = self.gpu_manager.selected_gpus
        
        # Initialize database
        self.db_path = "opulence_data.db"
        self._init_database()
        
        # LAZY LOADING: Don't create engines during init
        # REMOVE: self._initialize_llm_engines()
        #self.llm_engines = {}  # Will be populated on demand
        #self.engine_reference_count = {}  # Track how many agents use each engine
        self.engine_lock = asyncio.Lock()  # Prevent concurrent engine creation
        self.engine_reference_count = {}  # Track references per GPU
        self.engine_lock = asyncio.Lock()
        self.task_engine_mapping = {}     # {task_id: gpu_id} for cleanup
        
        # Agent storage with GPU assignment
        self.agents = {}
        self.agent_gpu_assignments = {}
        
        # Assign agent types to specific GPUs
        self._assign_agents_to_gpus()
        
        # Initialize stats
        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "avg_response_time": 0,
            "gpus_used": self.selected_gpus,
            "tasks_completed": 0,
            "start_time": time.time(),
            "engines_loaded": 0,  # Track loaded engines
            "lazy_loading": True   # Flag for lazy loading mode
        }
        
        self.logger.info(f"✅ Dual GPU Coordinator initialized on GPUs {self.selected_gpus} (LAZY LOADING)")
    
    @property
    def llm_engines(self):
        """Access GPU manager's engine cache directly"""
        return self.gpu_manager.gpu_engines
    
    async def get_shared_llm_engine(self, gpu_id: int):
        """FIXED: Get shared LLM engine with proper reuse logic"""
        async with self.engine_lock:
            
            # ✅ CRITICAL: Check if engine already exists and is healthy
            if self.gpu_manager.has_llm_engine(gpu_id):
                existing_engine = self.gpu_manager.gpu_engines[gpu_id]
                
                # ✅ Verify engine is still functional
                try:
                    # Simple health check - engine should respond to basic operations
                    if hasattr(existing_engine, 'engine') and existing_engine.engine:
                        # Increment reference count for tracking
                        self.engine_reference_count[gpu_id] = self.engine_reference_count.get(gpu_id, 0) + 1
                        self.logger.info(f"♻️ REUSING healthy engine on GPU {gpu_id} (refs: {self.engine_reference_count[gpu_id]})")
                        return existing_engine
                    else:
                        # Engine exists but is unhealthy - remove it
                        self.logger.warning(f"⚠️ Engine on GPU {gpu_id} unhealthy, recreating...")
                        self.gpu_manager.remove_llm_engine(gpu_id)
                except Exception as e:
                    self.logger.warning(f"⚠️ Engine health check failed on GPU {gpu_id}: {e}, recreating...")
                    self.gpu_manager.remove_llm_engine(gpu_id)
            
            # ✅ Create new engine only if none exists or previous was unhealthy
            self.logger.info(f"🔧 Creating NEW engine on GPU {gpu_id}...")
            engine = await self.gpu_manager.get_llm_engine_safe(
                gpu_id, 
                self.config.model_name, 
                self.config.max_tokens
            )
            
            # Initialize reference count
            self.engine_reference_count[gpu_id] = 1
            self.stats["engines_loaded"] += 1
            
            self.logger.info(f"✅ NEW engine created on GPU {gpu_id} (refs: 1)")
            return engine
            
    # 3. ADD engine reference management
    def release_engine_reference(self, gpu_id: int, task_id: str = None):
        """FIXED: Release engine reference with optional cleanup"""
        if gpu_id in self.engine_reference_count:
            self.engine_reference_count[gpu_id] -= 1
            self.logger.debug(f"📉 Released engine reference for GPU {gpu_id} (refs: {self.engine_reference_count[gpu_id]})")
            
            # Remove task mapping if provided
            if task_id and task_id in self.task_engine_mapping:
                del self.task_engine_mapping[task_id]
            
            # Auto cleanup if no references and auto_cleanup enabled
            if self.engine_reference_count[gpu_id] <= 0 and self.config.auto_cleanup:
                self._cleanup_unused_engine(gpu_id)

    def _cleanup_unused_engine(self, gpu_id: int):
        """FIXED: Clean up engine using GPU manager"""
        try:
            if self.gpu_manager.has_llm_engine(gpu_id):
                self.logger.info(f"🗑️ Auto-cleaning unused engine on GPU {gpu_id}")
                self.gpu_manager.remove_llm_engine(gpu_id)  # Use GPU manager's method
                
                # Clean up reference tracking
                if gpu_id in self.engine_reference_count:
                    del self.engine_reference_count[gpu_id]
                    
                self.stats["engines_loaded"] -= 1
                
        except Exception as e:
            self.logger.error(f"❌ Engine cleanup failed for GPU {gpu_id}: {e}")

    # Fix 2: Add missing _setup_logging method
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        return logging.getLogger(__name__)
    
    async def start_task_with_engine(self, task_name: str, agent_type: str, preferred_gpu: int = None) -> Tuple[str, Any, int]:
        """NEW: Start task and get engine atomically"""
        # Get assigned GPU for agent type or use preferred
        if preferred_gpu and preferred_gpu in self.selected_gpus:
            assigned_gpu = preferred_gpu
        else:
            assigned_gpu = self.agent_gpu_assignments.get(agent_type, self.selected_gpus[0])
        
        # Start task tracking
        task_id = self.gpu_manager.start_task(task_name, assigned_gpu)
        
        try:
            # Get engine for this task
            engine = await self.get_shared_llm_engine(assigned_gpu)
            
            # Link task to engine for cleanup
            self.task_engine_mapping[task_id] = assigned_gpu
            
            self.logger.info(f"🚀 Started task {task_id} with engine on GPU {assigned_gpu}")
            return task_id, engine, assigned_gpu
            
        except Exception as e:
            # Cleanup task if engine creation failed
            self.gpu_manager.finish_task(task_id)
            raise e

    def finish_task_with_engine(self, task_id: str):
        """NEW: Finish task and release engine reference"""
        try:
            # Get GPU from task mapping
            gpu_id = self.task_engine_mapping.get(task_id)
            
            # Finish task in GPU manager
            self.gpu_manager.finish_task(task_id)
            
            # Release engine reference
            if gpu_id is not None:
                self.release_engine_reference(gpu_id, task_id)
            
            self.logger.info(f"✅ Finished task {task_id} and released engine reference")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to finish task {task_id}: {e}")

    # Fix 3: Add missing get_agent method
    def get_agent(self, agent_type: str):
        """Get agent - uses assigned GPU automatically"""
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)
        return self.agents[agent_type]
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            
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
                                 
                CREATE TABLE IF NOT EXISTS partial_analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    partial_data TEXT NOT NULL,
                    progress_percent INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'in_progress',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX(component_name, agent_type, timestamp)
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    
    
    def _create_agent(self, agent_type: str):
        """Create agent using LAZY LOADED shared engine"""
        # Get assigned GPU for this agent type
        assigned_gpu = self.agent_gpu_assignments.get(agent_type, self.selected_gpus[0])
        
        self.logger.info(f"🔗 Creating {agent_type} agent (will lazy load engine on GPU {assigned_gpu})")
        
        # Pass coordinator reference instead of engine directly
        # Agents will call get_shared_llm_engine() when they need the engine
        
        if agent_type == "code_parser" and CodeParserAgent:
            return CodeParserAgent(
                llm_engine=None,  # Will be set via lazy loading
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                coordinator=self  # Agent uses coordinator.get_shared_llm_engine()
            )
        elif agent_type == "vector_index" and VectorIndexAgent:
            return VectorIndexAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                coordinator=self
            )
        elif agent_type == "data_loader" and DataLoaderAgent:
            return DataLoaderAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                coordinator=self
            )
        elif agent_type == "lineage_analyzer" and LineageAnalyzerAgent:
            return LineageAnalyzerAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                coordinator=self
            )
        elif agent_type == "logic_analyzer" and LogicAnalyzerAgent:
            return LogicAnalyzerAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                coordinator=self
            )
        elif agent_type == "documentation" and DocumentationAgent:
            return DocumentationAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                coordinator=self
            )
        elif agent_type == "db2_comparator" and DB2ComparatorAgent:
            return DB2ComparatorAgent(
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu,
                max_rows=self.config.max_db_rows,
                coordinator=self
            )
        elif agent_type == "chat_agent" and OpulenceChatAgent:
            return OpulenceChatAgent(
                coordinator=self,
                llm_engine=None,
                db_path=self.db_path,
                gpu_id=assigned_gpu
            )
        else:
            raise ValueError(f"Unknown or unavailable agent type: {agent_type}")
                
    def _assign_agents_to_gpus(self):
        """Assign different agent types to different GPUs for load balancing"""
        if len(self.selected_gpus) >= 2:
            # GPU 0: Code parsing (heavier workload)
            self.agent_gpu_assignments["code_parser"] = self.selected_gpus[0]
            self.agent_gpu_assignments["lineage_analyzer"] = self.selected_gpus[0]
            self.agent_gpu_assignments["logic_analyzer"] = self.selected_gpus[0]
            
            # GPU 1: Data processing (lighter workload)
            self.agent_gpu_assignments["data_loader"] = self.selected_gpus[1]
            self.agent_gpu_assignments["vector_index"] = self.selected_gpus[1]
            self.agent_gpu_assignments["chat_agent"] = self.selected_gpus[1]
        else:
            # Fallback to single GPU if only one available
            gpu_id = self.selected_gpus[0]
            for agent_type in ["code_parser", "data_loader", "vector_index", "chat_agent"]:
                self.agent_gpu_assignments[agent_type] = gpu_id
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """UPDATED: Process files with proper engine management"""
        start_time = time.time()
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"🚀 Processing {total_files} files on GPUs {self.selected_gpus}")
        
        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"📄 Processing file {i+1}/{total_files}: {file_path.name}")
                
                # Get appropriate agent
                if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                    agent = self.get_agent("code_parser")
                elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                    agent = self.get_agent("data_loader")
                else:
                    agent = self.get_agent("code_parser")  # Default
                
                # Process with automatic engine management
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
        self.stats["tasks_completed"] += 1
        
        return {
            "status": "success",
            "files_processed": total_files,
            "processing_time": processing_time,
            "results": results,
            "gpus_used": self.selected_gpus
        }

    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """UPDATED: Process chat with automatic engine management"""
        try:
            chat_agent = self.get_agent("chat_agent")
            # Agent handles engine management internally via context manager
            result = await chat_agent.process_chat_query(query, conversation_history)
            
            self.stats["total_queries"] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chat query failed: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try rephrasing your question", "Check system status"]
            }
    
    async def _auto_detect_and_process(self, file_path: Path) -> Dict[str, Any]:
        """Auto-detect file type and process"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)
            
            if 'IDENTIFICATION DIVISION' in content or 'PROGRAM-ID' in content:
                agent = self.get_agent("code_parser")
                result = await agent.process_file(file_path)
                await self._ensure_file_stored_in_db(file_path, result, "cobol")
                return result
                
            elif content.startswith('//') and 'JOB' in content:
                agent = self.get_agent("code_parser")
                result = await agent.process_file(file_path)
                await self._ensure_file_stored_in_db(file_path, result, "jcl")
                return result
                
            elif file_path.suffix.lower() in ['.cpy', '.copy']:
                agent = self.get_agent("data_loader")
                result = await agent.process_file(file_path)
                await self._ensure_file_stored_in_db(file_path, result, "copybook")
                return result
                
            elif ',' in content and '\n' in content:
                agent = self.get_agent("data_loader")
                result = await agent.process_file(file_path)
                await self._ensure_file_stored_in_db(file_path, result, "csv")
                return result
                
            else:
                return {"status": "unknown_file_type", "file": str(file_path)}
                
        except Exception as e:
            return {"status": "error", "error": str(e), "file": str(file_path)}
    
    async def _ensure_file_stored_in_db(self, file_path: Path, result: Dict, file_type: str):
        """Ensure file processing result is stored in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
                
                # Verify chunks were stored if applicable
                if "chunks_created" in result and result["chunks_created"] > 0:
                    cursor.execute("""
                        SELECT COUNT(*) FROM program_chunks 
                        WHERE program_name = ?
                    """, (file_path.stem,))
                    
                    chunk_count = cursor.fetchone()[0]
                    
                    if chunk_count != result["chunks_created"]:
                        self.logger.warning(
                            f"⚠️ Chunk count mismatch for {file_path.name}: "
                            f"expected {result['chunks_created']}, found {chunk_count}"
                        )
                
                cursor.execute("COMMIT")
                self.logger.debug(f"✅ Stored {file_path.name} in database")
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                raise e
                
            finally:
                conn.close()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store {file_path} in database: {str(e)}")
    
    async def _create_vector_embeddings_for_processed_files(self, file_paths: List[Path]):
        """Create vector embeddings for processed files"""
        try:
            vector_agent = self.get_agent("vector_index")
            
            # Get all program chunks from recently processed files
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
                embedding_result = await vector_agent.create_embeddings_for_chunks(chunks)
                self.logger.info(f"✅ Created embeddings for {len(chunks)} chunks")
                return embedding_result
            else:
                self.logger.warning("⚠️ No chunks found for vector embedding creation")
                return {"status": "no_chunks"}
                
        except Exception as e:
            self.logger.error(f"❌ Vector embedding creation failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def analyze_component_enhanced(self, component_name: str, component_type: str = None) -> Dict[str, Any]:
        """Enhanced component analysis with graceful degradation and partial results"""
        start_time = time.time()
        
        try:
            # Determine component type
            if not component_type or component_type == "auto-detect":
                component_type = await self._determine_component_type(component_name)
            
            # Initialize comprehensive result structure
            analysis_result = {
                "component_name": component_name,
                "component_type": component_type,
                "analysis_timestamp": dt.now().isoformat(),
                "status": "in_progress",
                "analyses": {},
                "executive_summary": "",
                "recommendations": [],
                "processing_metadata": {
                    "start_time": start_time,
                    "strategy": "graceful_degradation"
                }
            }
            
            # Define analysis strategy based on component type
            analysis_tasks = []
            
            if component_type == "field":
                analysis_tasks = [
                    ("lineage_analysis", "lineage_analyzer", "analyze_field_lineage_with_fallback"),
                    ("semantic_similarity", "vector_index", "search_similar_components")
                ]
            elif component_type in ["program", "cobol"]:
                analysis_tasks = [
                    ("logic_analysis", "logic_analyzer", "analyze_program"),
                    ("lifecycle_analysis", "lineage_analyzer", "analyze_full_lifecycle"),
                    ("dependency_analysis", "vector_index", "find_code_dependencies")
                ]
            else:  # copybook, jcl, etc.
                analysis_tasks = [
                    ("usage_analysis", "lineage_analyzer", "analyze_field_lineage_with_fallback"),
                    ("dependency_mapping", "logic_analyzer", "find_dependencies"),
                    ("similarity_search", "vector_index", "search_similar_components")
                ]
            
            # Execute analyses with error isolation
            completed_count = 0
            
            for analysis_name, agent_type, method_name in analysis_tasks:
                try:
                    self.logger.info(f"🔄 Running {analysis_name} for {component_name}")
                    
                    # Get agent and method
                    agent = self.get_agent(agent_type)
                    analysis_method = getattr(agent, method_name)
                    
                    # Execute with timeout (10 minutes max per analysis)
                    async with asyncio.timeout(600):
                        if analysis_name == "lifecycle_analysis":
                            result = await analysis_method(component_name, component_type)
                        else:
                            result = await analysis_method(component_name)
                    
                    analysis_result["analyses"][analysis_name] = {
                        "status": "success",
                        "data": result,
                        "agent_used": agent_type,
                        "completion_time": time.time() - start_time
                    }
                    
                    completed_count += 1
                    self.logger.info(f"✅ {analysis_name} completed successfully")
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"⏰ {analysis_name} timed out after 10 minutes")
                    analysis_result["analyses"][analysis_name] = {
                        "status": "timeout",
                        "error": "Analysis exceeded 10-minute timeout",
                        "agent_used": agent_type,
                        "partial_data": "Check partial_analysis_cache table for intermediate results"
                    }
                    
                except Exception as e:
                    self.logger.error(f"❌ {analysis_name} failed: {str(e)}")
                    analysis_result["analyses"][analysis_name] = {
                        "status": "error",
                        "error": str(e),
                        "agent_used": agent_type
                    }
            
            # Determine final status
            total_analyses = len(analysis_tasks)
            if completed_count == total_analyses:
                analysis_result["status"] = "completed"
            elif completed_count > 0:
                analysis_result["status"] = "partial"
            else:
                analysis_result["status"] = "failed"
            
            # Generate executive summary using available results
            analysis_result["executive_summary"] = await self._generate_executive_summary(
                component_name, analysis_result
            )
            
            # Generate actionable recommendations
            analysis_result["recommendations"] = self._generate_actionable_recommendations(
                analysis_result, completed_count, total_analyses
            )
            
            # Add final metadata
            analysis_result["processing_metadata"].update({
                "end_time": time.time(),
                "total_duration_seconds": time.time() - start_time,
                "analyses_completed": completed_count,
                "analyses_total": total_analyses,
                "success_rate": (completed_count / total_analyses) * 100
            })
            
            return self._add_processing_info(analysis_result)
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced component analysis failed: {str(e)}")
            return self._add_processing_info({
                "component_name": component_name,
                "status": "system_error",
                "error": str(e),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            })

    async def _generate_executive_summary(self, component_name: str, analysis_result: Dict) -> str:
        """Generate executive summary from available analysis results"""
        try:
            # Collect findings from successful analyses
            findings = {}
            for analysis_name, analysis_data in analysis_result["analyses"].items():
                if analysis_data["status"] == "success":
                    findings[analysis_name] = analysis_data["data"]
            
            if not findings:
                return f"Analysis of {component_name} encountered technical difficulties. No complete results available at this time."
            
            # Use chat agent for summary generation
            chat_agent = self.get_agent("chat_agent")
            
            # Prepare context for LLM
            context = {
                "component": component_name,
                "type": analysis_result["component_type"],
                "status": analysis_result["status"],
                "findings_available": list(findings.keys()),
                "completion_rate": f"{analysis_result['processing_metadata']['analyses_completed']}/{analysis_result['processing_metadata']['analyses_total']}"
            }
            
            # Extract key insights from available data
            key_insights = []
            if "lineage_analysis" in findings:
                lineage_data = findings["lineage_analysis"]
                if isinstance(lineage_data, dict):
                    programs_count = len(lineage_data.get("lineage_data", {}).get("programs", []))
                    refs_analyzed = lineage_data.get("references_analyzed", 0)
                    key_insights.append(f"Found usage in {programs_count} programs, analyzed {refs_analyzed} references")
            
            if "logic_analysis" in findings:
                logic_data = findings["logic_analysis"]
                if isinstance(logic_data, dict):
                    complexity = logic_data.get("complexity_score", 0)
                    rules_count = len(logic_data.get("business_rules", []))
                    key_insights.append(f"Complexity score: {complexity:.1f}, {rules_count} business rules identified")
            
            prompt = f"""
            Generate an executive summary for this component analysis:
            
            Component: {context['component']} ({context['type']})
            Analysis Status: {context['status']} 
            Completion Rate: {context['completion_rate']}
            Available Findings: {', '.join(context['findings_available'])}
            
            Key Insights:
            {chr(10).join(f"• {insight}" for insight in key_insights)}
            
            Provide a business-focused executive summary covering:
            1. Component purpose and business function
            2. Key findings from available analyses  
            3. Dependencies and business impact
            4. Analysis completeness and confidence level
            5. Immediate actionable insights
            
            Target audience: Business stakeholders and IT management
            Length: 200-250 words, professional tone
            """
            
            async with chat_agent.get_engine_context() as engine:
                sampling_params = SamplingParams(temperature=0.3, max_tokens=500)
                request_id = str(uuid.uuid4())
                
                async for result in engine.generate(prompt, sampling_params, request_id=request_id):
                    return result.outputs[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            # Fallback summary
            completed = analysis_result["processing_metadata"]["analyses_completed"]
            total = analysis_result["processing_metadata"]["analyses_total"]
            return f"Analysis of {component_name} ({analysis_result['component_type']}) completed {completed} of {total} analyses successfully. Status: {analysis_result['status']}. See detailed results below for available findings."
    
    def _generate_actionable_recommendations(self, analysis_result: Dict, completed: int, total: int) -> List[str]:
        """Generate actionable recommendations based on analysis results"""
        recommendations = []
        
        # Status-based recommendations
        if analysis_result["status"] == "partial":
            completion_rate = (completed / total) * 100
            if completion_rate >= 60:
                recommendations.append(f"✅ Analysis {completion_rate:.0f}% complete - sufficient for initial assessment")
            else:
                recommendations.append(f"⚠️ Analysis only {completion_rate:.0f}% complete - consider re-running for full insights")
        
        elif analysis_result["status"] == "failed":
            recommendations.append("❌ Analysis failed completely - check component complexity and system resources")
            recommendations.append("🔧 Consider breaking component into smaller units for analysis")
        
        # Analysis-specific recommendations
        for analysis_name, analysis_data in analysis_result["analyses"].items():
            if analysis_data["status"] == "timeout":
                if analysis_name == "lineage_analysis":
                    recommendations.append("⏰ Lineage analysis timed out - check partial_analysis_cache for intermediate results")
                elif analysis_name == "logic_analysis":
                    recommendations.append("⏰ Logic analysis timed out - consider manual code review for business rules")
        
        # Success-based recommendations
        successful_analyses = [name for name, data in analysis_result["analyses"].items() if data["status"] == "success"]
        
        if "lineage_analysis" in successful_analyses:
            recommendations.append("📊 Lineage analysis available - review dependency impacts before making changes")
        
        if "logic_analysis" in successful_analyses:
            recommendations.append("🧠 Business logic analysis complete - validate identified rules with business stakeholders")
        
        # Generic recommendations
        if len(successful_analyses) > 0:
            recommendations.append("📋 Use available analysis results for impact assessment and change planning")
        
        return recommendations
    async def _determine_component_type(self, component_name: str) -> str:
        """Determine component type from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check chunk types in database
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count 
                FROM program_chunks 
                WHERE program_name = ?
                GROUP BY chunk_type
                ORDER BY count DESC
            """, (component_name,))
            
            chunk_types = cursor.fetchall()
            
            if chunk_types:
                dominant_chunk_type = chunk_types[0][0]
                
                if any('job' in ct.lower() for ct, _ in chunk_types):
                    return "jcl"
                elif any(ct in ['working_storage', 'procedure_division', 'data_division'] for ct, _ in chunk_types):
                    return "program"
                else:
                    return "program"
            
            # Check if it's a field (found in content)
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
    
    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process chat query using single GPU"""
        task_id = self.gpu_manager.start_task("chat_query")
        
        try:
            chat_agent = self.get_agent("chat_agent")
            result = await chat_agent.process_chat_query(query, conversation_history)
            result["gpu_used"] = self.selected_gpu
            
            self.stats["total_queries"] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chat query failed: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "response_type": "error",
                "gpu_used": self.selected_gpu,
                "suggestions": ["Try rephrasing your question", "Check system status"]
            }
        finally:
            self.gpu_manager.finish_task(task_id)
    
    async def search_code_patterns(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search code patterns using single GPU - FIXED"""
        task_id = self.gpu_manager.start_task("pattern_search")
        
        try:
            # FIX: Use single GPU coordinator approach
            vector_agent = self.get_agent("vector_index")
            results = await vector_agent.semantic_search(query, top_k=limit)
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_found": len(results),
                "gpu_used": self.selected_gpu
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pattern search failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "gpu_used": self.selected_gpu
            }
        finally:
            self.gpu_manager.finish_task(task_id)

    
    def _update_processing_stats(self, operation: str, duration: float):
        """Update processing statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_stats (operation, duration, gpu_used, status)
                VALUES (?, ?, ?, ?)
            """, (operation, duration, str(self.selected_gpus), "completed"))
            
            conn.commit()
            conn.close()
            
            # Update in-memory stats
            self.stats["avg_response_time"] = (
                self.stats["avg_response_time"] + duration
            ) / 2
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update stats: {str(e)}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
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
            
            # Get GPU manager status
            gpu_status = self.gpu_manager.get_status()
            
            return {
                "system_stats": self.stats,
                "processing_stats": processing_stats.to_dict('records') if not processing_stats.empty else [],
                "file_stats": file_stats.to_dict('records') if not file_stats.empty else [],
                "gpu_status": gpu_status,
                "database_stats": await self._get_database_stats(),
                "timestamp": dt.now().isoformat(),
                "coordinator_type": "single_gpu"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {str(e)}")
            return {
                "system_stats": self.stats,
                "error": str(e),
                "coordinator_type": "single_gpu"
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
            
            # Database file size
            if os.path.exists(self.db_path):
                stats["database_size_bytes"] = os.path.getsize(self.db_path)
            else:
                stats["database_size_bytes"] = 0
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Database stats failed: {str(e)}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """FIXED: Enhanced health status with engine sharing info"""
        gpu_status = self.gpu_manager.get_status()
        
        # ✅ Add engine sharing metrics
        engine_sharing_stats = {}
        for gpu_id, ref_count in self.engine_reference_count.items():
            agents_on_gpu = [
                agent_type for agent_type, assigned_gpu 
                in self.agent_gpu_assignments.items() 
                if assigned_gpu == gpu_id
            ]
            engine_sharing_stats[f"GPU_{gpu_id}"] = {
                "reference_count": ref_count,
                "agents_assigned": agents_on_gpu,
                "engine_loaded": self.gpu_manager.has_llm_engine(gpu_id)
            }
        
        return {
            "status": "healthy" if self.selected_gpus else "no_gpus",
            "coordinator_type": "dual_gpu_lazy_shared",
            "selected_gpus": self.selected_gpus,
            "active_agents": len(self.agents),
            "stats": self.stats,
            "uptime_seconds": time.time() - self.stats["start_time"],
            "database_available": os.path.exists(self.db_path),
            "engine_sharing": engine_sharing_stats,  # ✅ New sharing metrics
            "lazy_loading": {
                "enabled": True,
                "engines_loaded": len(self.llm_engines),
                "engines_available": len(self.selected_gpus),
                "sharing_efficient": all(
                    ref_count <= len([a for a, g in self.agent_gpu_assignments.items() if g == gpu_id])
                    for gpu_id, ref_count in self.engine_reference_count.items()
                ),
                "on_demand_loading": True
            }
        }

    # 6. ADD method to preload engines if needed
    async def preload_engines(self, gpu_ids: List[int] = None):
        """Manually preload engines on specific GPUs (optional)"""
        target_gpus = gpu_ids or self.selected_gpus
        
        self.logger.info(f"🚀 Preloading engines on GPUs: {target_gpus}")
        
        for gpu_id in target_gpus:
            try:
                await self.get_shared_llm_engine(gpu_id)
                self.logger.info(f"✅ Preloaded engine on GPU {gpu_id}")
            except Exception as e:
                self.logger.error(f"❌ Failed to preload engine on GPU {gpu_id}: {e}")

    # 7. UPDATE cleanup method
    def cleanup(self):
        """UPDATED: Comprehensive cleanup with agent management"""
        self.logger.info("🧹 Cleaning up coordinator resources...")
        
        # Clean up all agents
        for agent_type, agent in self.agents.items():
            try:
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                    self.logger.info(f"✅ Cleaned up {agent_type} agent")
            except Exception as e:
                self.logger.error(f"❌ Failed to cleanup {agent_type}: {e}")
        
        # Clear agent cache
        self.agents.clear()
        
        # Clean up all engines and references
        for gpu_id in list(self.engine_reference_count.keys()):
            self._cleanup_unused_engine(gpu_id)
        
        # Clean up GPU manager
        self.gpu_manager.release_gpus()
        
        # Clear task mappings
        self.task_engine_mapping.clear()
        self.engine_reference_count.clear()
        
        self.logger.info("✅ Cleanup completed")

    
    def shutdown(self):
        """Shutdown coordinator and release all resources"""
        self.logger.info("🔄 Shutting down Single GPU Coordinator...")
        
        # Clear agents
        self.agents.clear()
        
        # Release GPU
        self.gpu_manager.release_gpu()
        
        self.logger.info("✅ Single GPU Coordinator shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
    
    def __repr__(self):
        return (f"DualGPUCoordinator("
                f"gpu={self.selected_gpu}, "
                f"agents={len(self.agents)}, "
                f"tasks_completed={self.stats['tasks_completed']})")


# Enhanced Chat Capabilities
class SingleGPUChatEnhancer:
    """Enhanced chat capabilities for single GPU coordinator"""
    
    def __init__(self, coordinator: DualGPUOpulenceCoordinator):
        self.coordinator = coordinator
        self.logger = coordinator.logger
    
    async def process_regular_chat_query(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Process regular chat query with enhanced responses"""
        try:
            result = await self.coordinator.process_chat_query(query, conversation_history)
            
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
            self.logger.error(f"❌ Chat query failed: {str(e)}")
            return f"❌ I encountered an error: {str(e)}"
    
    async def chat_analyze_component(self, component_name: str, user_question: str = None, 
                                    conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze component with chat-enhanced explanations"""
        try:
            # Get regular analysis
            analysis_result = await self.coordinator.analyze_component_enhanced(component_name)
            
            # Get chat explanation
            if user_question:
                chat_query = f"Explain the analysis of {component_name}. {user_question}"
            else:
                chat_query = f"Provide a detailed explanation of {component_name} based on the analysis."
            
            # Create context
            enhanced_history = conversation_history or []
            enhanced_history.append({
                "role": "system",
                "content": f"Analysis data for {component_name}: {json.dumps(analysis_result, default=str)}"
            })
            
            chat_result = await self.coordinator.process_chat_query(chat_query, enhanced_history)
            
            return {
                "component_name": component_name,
                "analysis": analysis_result,
                "chat_explanation": chat_result.get("response", ""),
                "suggestions": chat_result.get("suggestions", []),
                "response_type": "enhanced_analysis",
                "gpu_used": self.coordinator.selected_gpu
            }
            
        except Exception as e:
            self.logger.error(f"❌ Chat-enhanced analysis failed: {str(e)}")
            return {
                "component_name": component_name,
                "analysis": analysis_result if 'analysis_result' in locals() else {},
                "error": str(e),
                "response_type": "error"
            }
    
    async def chat_search_patterns(self, search_description: str, 
                              conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Search patterns with chat-enhanced results - FIXED"""
        try:
            # Use vector search - FIXED
            # FIX: Use single GPU coordinator approach
            vector_agent = self.coordinator.get_agent("vector_index")
            search_results = await vector_agent.semantic_search(search_description, top_k=10)
            
            # Get chat explanation
            chat_query = f"Explain these search results for '{search_description}' and help me understand what was found."
            
            # Create context
            search_context = [
                {
                    "role": "system", 
                    "content": f"Search results for '{search_description}': {json.dumps(search_results[:5], default=str)}"
                }
            ]
            if conversation_history:
                search_context.extend(conversation_history)
            
            chat_result = await self.coordinator.process_chat_query(chat_query, search_context)
            
            return {
                "search_description": search_description,
                "search_results": search_results,
                "chat_explanation": chat_result.get("response", ""),
                "total_found": len(search_results),
                "suggestions": chat_result.get("suggestions", []),
                "response_type": "enhanced_search",
                "gpu_used": self.coordinator.selected_gpu
            }
            
        except Exception as e:
            self.logger.error(f"❌ Chat-enhanced search failed: {str(e)}")
            return {
                "search_description": search_description,
                "search_results": [],
                "error": str(e),
                "response_type": "error"
            }



def create_dual_gpu_coordinator(
        model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
        exclude_gpu_0: bool = True,
        min_memory_gb: float = 6.0,
        force_gpu_ids: Optional[List[int]] = None
         ) -> DualGPUOpulenceCoordinator:
        """Create a dual GPU coordinator"""
        
        config = DualGPUOpulenceConfig(
            model_name=model_name,
            exclude_gpu_0=exclude_gpu_0,
            min_memory_gb=min_memory_gb,
            force_gpu_ids=force_gpu_ids,
            max_tokens=1024,
            auto_cleanup=True
        )
        
        return DualGPUOpulenceCoordinator(config)




def reset_global_coordinator():
    """Reset global coordinator (useful for testing)"""
    global _global_coordinator
    if _global_coordinator is not None:
        _global_coordinator.shutdown()
        _global_coordinator = None


# Utility Functions
async def quick_file_processing(file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
    """Quick file processing using global coordinator"""
    coordinator = get_global_coordinator()
    return await coordinator.process_batch_files(file_paths, file_type)


async def quick_component_analysis(component_name: str, component_type: str = None) -> Dict[str, Any]:
    """Quick component analysis using global coordinator"""
    coordinator = get_global_coordinator()
    return await coordinator.analyze_component_enhanced(component_name, component_type)


async def quick_chat_query(query: str, conversation_history: List[Dict] = None) -> str:
    """Quick chat query using global coordinator"""
    coordinator = get_global_coordinator()
    enhancer = SingleGPUChatEnhancer(coordinator)
    return await enhancer.process_regular_chat_query(query, conversation_history)


def get_system_status() -> Dict[str, Any]:
    """Get system status using global coordinator"""
    coordinator = get_global_coordinator()
    return coordinator.get_health_status()

# Add these functions to your opulence_coordinator.py file

def create_shared_server_coordinator() -> DualGPUOpulenceCoordinator:
    """Create coordinator optimized for shared server environments"""
    
    config = DualGPUOpulenceConfig(
        model_name="microsoft/DialoGPT-medium",  # Smaller model for shared environments
        exclude_gpu_0=True,  # Don't compete with others on GPU 0
        min_memory_gb=4.0,   # Lower memory requirement per GPU
        max_tokens=512,      # Smaller context window
        auto_cleanup=True,
        force_gpu_ids=None,  # Auto-select best available GPUs
        batch_size=4         # Smaller batch size per GPU
    )
    
    return DualGPUOpulenceCoordinator(config)


def create_dedicated_server_coordinator() -> DualGPUOpulenceCoordinator:
    """Create coordinator optimized for dedicated servers"""
    
    config = DualGPUOpulenceConfig(
        model_name="codellama/CodeLlama-7b-Instruct-hf",
        exclude_gpu_0=False,  # Can use any GPU on dedicated server
        min_memory_gb=8.0,    # Higher memory requirement per GPU
        max_tokens=2048,      # Larger context window
        auto_cleanup=True,
        force_gpu_ids=None,   # Auto-select best available GPUs
        batch_size=8          # Larger batch size per GPU
    )
    
    return DualGPUOpulenceCoordinator(config)


# Also update the get_global_coordinator function
def get_global_coordinator() -> DualGPUOpulenceCoordinator:
    """Get or create global coordinator instance"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = create_dual_gpu_coordinator(force_gpu_ids=[1, 2])
    return _global_coordinator

class DualGPUProductionCoordinatorManager:
    """Production-ready dual GPU coordinator manager with monitoring and recovery"""
    
    def __init__(self, config: DualGPUOpulenceConfig = None):
        self.config = config or DualGPUOpulenceConfig()
        self.coordinator = None
        self.health_check_interval = 60  # seconds
        self.max_restart_attempts = 3
        self.restart_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Dual GPU specific monitoring
        self.gpu_failure_counts = {}  # Track failures per GPU
        self.last_gpu_check = {}      # Last health check per GPU
        
    async def start(self):
        """Start coordinator with health monitoring"""
        try:
            self.coordinator = DualGPUOpulenceCoordinator(self.config)
            
            # Initialize GPU monitoring
            for gpu_id in self.coordinator.selected_gpus:
                self.gpu_failure_counts[gpu_id] = 0
                self.last_gpu_check[gpu_id] = time.time()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            self.logger.info(f"✅ Production coordinator started on GPUs {self.coordinator.selected_gpus}")
            return self.coordinator
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start coordinator: {e}")
            raise
    
    async def _health_monitor(self):
        """Monitor coordinator health and restart if needed"""
        while self.coordinator is not None:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.coordinator is None:
                    break
                
                # Check overall system status
                status = self.coordinator.get_health_status()
                
                if status['status'] != 'healthy':
                    self.logger.warning(f"⚠️ Unhealthy status: {status['status']}")
                    
                    # Check individual GPU health
                    unhealthy_gpus = await self._check_individual_gpu_health()
                    
                    if unhealthy_gpus:
                        self.logger.warning(f"⚠️ Unhealthy GPUs detected: {unhealthy_gpus}")
                        
                        # Try GPU-specific recovery first
                        recovery_success = await self._attempt_gpu_recovery(unhealthy_gpus)
                        
                        if not recovery_success and self.restart_count < self.max_restart_attempts:
                            await self._restart_coordinator()
                        elif self.restart_count >= self.max_restart_attempts:
                            self.logger.error("❌ Max restart attempts reached")
                            break
                
            except Exception as e:
                self.logger.error(f"❌ Health check failed: {e}")
    
    async def _check_individual_gpu_health(self) -> List[int]:
        """Check health of individual GPUs"""
        unhealthy_gpus = []
        
        try:
            for gpu_id in self.coordinator.selected_gpus:
                gpu_status = self.coordinator.gpu_manager.get_gpu_status(gpu_id)
                
                # Check GPU health criteria
                if (gpu_status.get('memory_usage_gb', 0) > 20 or  # High memory usage
                    gpu_status.get('active_tasks', 0) > 10 or      # Too many active tasks
                    gpu_status.get('error_count', 0) > 5):         # Too many errors
                    
                    unhealthy_gpus.append(gpu_id)
                    self.gpu_failure_counts[gpu_id] += 1
                    
                    self.logger.warning(f"⚠️ GPU {gpu_id} health issues detected")
                
                self.last_gpu_check[gpu_id] = time.time()
                
        except Exception as e:
            self.logger.error(f"❌ GPU health check failed: {e}")
        
        return unhealthy_gpus
    
    async def _attempt_gpu_recovery(self, unhealthy_gpus: List[int]) -> bool:
        """Attempt to recover specific GPUs without full restart"""
        try:
            self.logger.info(f"🔧 Attempting GPU recovery for GPUs: {unhealthy_gpus}")
            
            for gpu_id in unhealthy_gpus:
                try:
                    # Clean up GPU memory
                    await self.coordinator.gpu_manager.cleanup_gpu_memory(gpu_id)
                    
                    # Reset GPU engine if needed
                    if gpu_id in self.coordinator.gpu_manager.gpu_engines:
                        del self.coordinator.gpu_manager.gpu_engines[gpu_id]
                    
                    # Recreate engine for this GPU
                    self.coordinator.gpu_manager.get_llm_engine_safe(gpu_id)
                    
                    self.logger.info(f"✅ GPU {gpu_id} recovery successful")
                    
                except Exception as e:
                    self.logger.error(f"❌ GPU {gpu_id} recovery failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GPU recovery failed: {e}")
            return False
    
    async def _restart_coordinator(self):
        """Restart entire coordinator"""
        try:
            self.logger.info("🔄 Restarting dual GPU coordinator...")
            
            if self.coordinator:
                self.coordinator.shutdown()
            
            # Wait before restart
            await asyncio.sleep(5)
            
            self.coordinator = DualGPUOpulenceCoordinator(self.config)
            self.restart_count += 1
            
            # Reset GPU monitoring
            for gpu_id in self.coordinator.selected_gpus:
                self.gpu_failure_counts[gpu_id] = 0
                self.last_gpu_check[gpu_id] = time.time()
            
            self.logger.info(f"✅ Coordinator restarted (attempt {self.restart_count}) on GPUs {self.coordinator.selected_gpus}")
            
        except Exception as e:
            self.logger.error(f"❌ Restart failed: {e}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "restart_count": self.restart_count,
            "max_restart_attempts": self.max_restart_attempts,
            "gpu_failure_counts": self.gpu_failure_counts.copy(),
            "last_gpu_checks": self.last_gpu_check.copy(),
            "health_check_interval": self.health_check_interval,
            "coordinator_status": "running" if self.coordinator else "stopped"
        }
    
    def stop(self):
        """Stop coordinator"""
        if self.coordinator:
            self.coordinator.shutdown()
            self.coordinator = None


# Keep the old name for backward compatibility
ProductionCoordinatorManager = DualGPUProductionCoordinatorManager


# Alternative: Create a factory function for production managers
def create_production_manager(
    server_type: str = "dedicated",
    force_gpu_ids: Optional[List[int]] = None
) -> DualGPUProductionCoordinatorManager:
    """Create production coordinator manager based on server type"""
    
    if server_type == "shared":
        config = DualGPUOpulenceConfig(
            model_name="microsoft/DialoGPT-medium",
            exclude_gpu_0=True,
            min_memory_gb=4.0,
            max_tokens=512,
            force_gpu_ids=force_gpu_ids,
            auto_cleanup=True
        )
    else:  # dedicated
        config = DualGPUOpulenceConfig(
            model_name="codellama/CodeLlama-7b-Instruct-hf",
            exclude_gpu_0=False,
            min_memory_gb=8.0,
            max_tokens=2048,
            force_gpu_ids=force_gpu_ids,
            auto_cleanup=True
        )
    
    return DualGPUProductionCoordinatorManager(config)


# Update the __all__ export list at the bottom of the file
__all__ = [
    'DualGPUOpulenceCoordinator',
    'DualGPUOpulenceConfig', 
    'SingleGPUChatEnhancer',           # Note: keeping same name as it works with dual GPU
    'create_dual_gpu_coordinator',
    'create_shared_server_coordinator',    # Add this
    'create_dedicated_server_coordinator', # Add this
    'get_global_coordinator',
    'quick_file_processing',
    'quick_component_analysis',
    'quick_chat_query',
    'get_system_status',
    'ProductionCoordinatorManager'
]