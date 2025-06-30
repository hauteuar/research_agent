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
from typing import Dict, List, Optional, Any
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
from utils.single_gpu_manager import SingleGPUManager, SingleGPUConfig

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
class SingleGPUOpulenceConfig:
    """Configuration for Single GPU Opulence system"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_tokens: int = 1024  # Conservative for single GPU
    temperature: float = 0.1
    exclude_gpu_0: bool = True  # Don't use GPU 0 in shared systems
    min_memory_gb: float = 6.0  # Minimum GPU memory required
    max_processing_time: int = 900  # 15 minutes
    batch_size: int = 16  # Smaller batch for single GPU
    vector_dim: int = 768
    max_db_rows: int = 10000
    cache_ttl: int = 3600  # 1 hour
    auto_cleanup: bool = True
    force_gpu_id: Optional[int] = None  # Force specific GPU

class SingleGPUOpulenceCoordinator:
    """Single GPU Opulence Coordinator - Uses one GPU for all operations"""
    
    def __init__(self, config: SingleGPUOpulenceConfig = None):
        """Initialize with single GPU approach"""
        self.config = config or SingleGPUOpulenceConfig()
        self.logger = self._setup_logging()
        
        # Create single GPU manager configuration
        gpu_config = SingleGPUConfig(
            exclude_gpu_0=self.config.exclude_gpu_0,
            min_memory_gb=self.config.min_memory_gb,
            force_gpu_id=self.config.force_gpu_id,
            cleanup_on_exit=self.config.auto_cleanup
        )
        
        # Initialize single GPU manager
        self.gpu_manager = SingleGPUManager(gpu_config)
        self.selected_gpu = self.gpu_manager.selected_gpu
        
        # Initialize database
        self.db_path = "opulence_data.db"
        self._init_database()
        
        # Create single LLM engine that all agents will share
        self.llm_engine = self.gpu_manager.get_llm_engine(
            self.config.model_name,
            self.config.max_tokens
        )
        
        # Simple agent storage - no complex allocation needed
        self.agents = {}
        
        # Processing statistics
        self.stats = {
            "total_files_processed": 0,
            "total_queries": 0,
            "avg_response_time": 0,
            "gpu_used": self.selected_gpu,
            "tasks_completed": 0,
            "start_time": time.time()
        }
        
        self.logger.info(f"✅ Single GPU Opulence Coordinator initialized on GPU {self.selected_gpu}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        return logging.getLogger(__name__)
    
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
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def get_agent(self, agent_type: str):
        """Get agent - all agents use the same GPU automatically"""
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)
        return self.agents[agent_type]
    
    def _create_agent(self, agent_type: str):
        """Create agent using shared LLM engine and GPU"""
        if agent_type == "code_parser" and CodeParserAgent:
            return CodeParserAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                coordinator=self
            )
        elif agent_type == "vector_index" and VectorIndexAgent:
            return VectorIndexAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                coordinator=self
            )
        elif agent_type == "data_loader" and DataLoaderAgent:
            return DataLoaderAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                coordinator=self
            )
        elif agent_type == "lineage_analyzer" and LineageAnalyzerAgent:
            return LineageAnalyzerAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                coordinator=self
            )
        elif agent_type == "logic_analyzer" and LogicAnalyzerAgent:
            return LogicAnalyzerAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                coordinator=self
            )
        elif agent_type == "documentation" and DocumentationAgent:
            return DocumentationAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                coordinator=self
            )
        elif agent_type == "db2_comparator" and DB2ComparatorAgent:
            return DB2ComparatorAgent(
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu,
                max_rows=self.config.max_db_rows,
                coordinator=self
            )
        elif agent_type == "chat_agent" and OpulenceChatAgent:
            return OpulenceChatAgent(
                coordinator=self,
                llm_engine=self.llm_engine,
                db_path=self.db_path,
                gpu_id=self.selected_gpu
            )
        else:
            raise ValueError(f"Unknown or unavailable agent type: {agent_type}")
    
    async def process_batch_files(self, file_paths: List[Path], file_type: str = "auto") -> Dict[str, Any]:
        """Process files using single GPU"""
        task_id = self.gpu_manager.start_task("batch_file_processing")
        start_time = time.time()
        
        try:
            results = []
            total_files = len(file_paths)
            
            self.logger.info(f"🚀 Processing {total_files} files on GPU {self.selected_gpu}")
            
            for i, file_path in enumerate(file_paths):
                try:
                    self.logger.info(f"📄 Processing file {i+1}/{total_files}: {file_path.name}")
                    
                    # Determine file type and get appropriate agent
                    if file_type == "cobol" or file_path.suffix.lower() in ['.cbl', '.cob']:
                        agent = self.get_agent("code_parser")
                        result = await agent.process_file(file_path)
                        await self._ensure_file_stored_in_db(file_path, result, "cobol")
                        
                    elif file_type == "jcl" or file_path.suffix.lower() == '.jcl':
                        agent = self.get_agent("code_parser")
                        result = await agent.process_file(file_path)
                        await self._ensure_file_stored_in_db(file_path, result, "jcl")
                        
                    elif file_type == "csv" or file_path.suffix.lower() == '.csv':
                        agent = self.get_agent("data_loader")
                        result = await agent.process_file(file_path)
                        await self._ensure_file_stored_in_db(file_path, result, "csv")
                        
                    else:
                        # Auto-detect
                        result = await self._auto_detect_and_process(file_path)
                    
                    result["gpu_used"] = self.selected_gpu
                    results.append(result)
                    
                    # Log progress
                    if (i + 1) % 5 == 0:
                        self.logger.info(f"📊 Progress: {i+1}/{total_files} files processed")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {file_path}: {str(e)}")
                    results.append({
                        "status": "error",
                        "file": str(file_path),
                        "error": str(e),
                        "gpu_used": self.selected_gpu
                    })
            
            # Create vector embeddings for successful results
            successful_files = [r for r in results if r.get("status") == "success"]
            if successful_files:
                try:
                    await self._create_vector_embeddings_for_processed_files(file_paths)
                    self.logger.info("✅ Vector embeddings created for processed files")
                except Exception as e:
                    self.logger.warning(f"⚠️ Vector embedding creation failed: {str(e)}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_files_processed"] += total_files
            self.stats["tasks_completed"] += 1
            self._update_processing_stats("batch_processing", processing_time)
            
            return {
                "status": "success",
                "files_processed": total_files,
                "successful_files": len(successful_files),
                "failed_files": total_files - len(successful_files),
                "processing_time": processing_time,
                "results": results,
                "gpu_used": self.selected_gpu,
                "vector_indexing": "completed" if successful_files else "skipped"
            }
            
        finally:
            self.gpu_manager.finish_task(task_id)
            # Optional: cleanup memory after large batch
            if len(file_paths) > 10:
                self.gpu_manager.cleanup_gpu_memory()
    
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
    
    async def analyze_component(self, component_name: str, component_type: str = None) -> Dict[str, Any]:
        """Analyze component using single GPU"""
        task_id = self.gpu_manager.start_task(f"analyze_{component_name}")
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Analyzing component: {component_name}")
            
            # Verify component exists in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            chunk_count = cursor.fetchone()[0]
            
            if chunk_count == 0:
                # Try content search
                cursor.execute("""
                    SELECT COUNT(*) FROM program_chunks 
                    WHERE content LIKE ? OR metadata LIKE ?
                """, (f"%{component_name}%", f"%{component_name}%"))
                
                content_matches = cursor.fetchone()[0]
                
                if content_matches == 0:
                    conn.close()
                    return {
                        "component_name": component_name,
                        "status": "error",
                        "error": f"Component '{component_name}' not found in database",
                        "suggestion": "Check component name or ensure files were processed"
                    }
            
            conn.close()
            
            # Determine component type
            if not component_type or component_type == "auto-detect":
                component_type = await self._determine_component_type(component_name)
            
            analysis_result = {
                "component_name": component_name,
                "component_type": component_type,
                "status": "processing",
                "chunks_found": chunk_count,
                "gpu_used": self.selected_gpu,
                "timestamp": dt.now().isoformat()
            }
            
            # Perform component-specific analysis
            analysis_success = False
            
            if component_type == "field":
                try:
                    lineage_agent = self.get_agent("lineage_analyzer")
                    lineage_result = await lineage_agent.analyze_field_lineage(component_name)
                    analysis_result["lineage"] = lineage_result
                    analysis_success = True
                except Exception as e:
                    self.logger.error(f"❌ Lineage analysis failed: {e}")
                    analysis_result["lineage"] = {"error": str(e)}
            
            elif component_type in ["program", "cobol"]:
                try:
                    logic_agent = self.get_agent("logic_analyzer")
                    logic_result = await logic_agent.analyze_program(component_name)
                    analysis_result["logic_analysis"] = logic_result
                    analysis_success = True
                except Exception as e:
                    self.logger.error(f"❌ Logic analysis failed: {e}")
                    analysis_result["logic_analysis"] = {"error": str(e)}
            
            elif component_type == "jcl":
                try:
                    lineage_agent = self.get_agent("lineage_analyzer")
                    jcl_result = await lineage_agent.analyze_full_lifecycle(component_name, "jcl")
                    analysis_result["jcl_analysis"] = jcl_result
                    analysis_success = True
                except Exception as e:
                    self.logger.error(f"❌ JCL analysis failed: {e}")
                    analysis_result["jcl_analysis"] = {"error": str(e)}
            
            # Always try semantic search
            try:
                vector_agent = self.get_agent("vector_index")
                semantic_results = await vector_agent.search_similar_components(component_name)
                analysis_result["semantic_search"] = semantic_results
            except Exception as e:
                self.logger.error(f"❌ Semantic search failed: {e}")
                analysis_result["semantic_search"] = {"error": str(e)}
            
            # Finalize result
            analysis_result["status"] = "completed" if analysis_success else "partial"
            analysis_result["processing_time"] = time.time() - start_time
            
            self.logger.info(f"✅ Component analysis completed for {component_name}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Component analysis failed: {str(e)}")
            return {
                "component_name": component_name,
                "status": "error",
                "error": str(e),
                "gpu_used": self.selected_gpu,
                "processing_time": time.time() - start_time
            }
        finally:
            self.gpu_manager.finish_task(task_id)
    
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
        """Search code patterns using single GPU"""
        task_id = self.gpu_manager.start_task("pattern_search")
        
        try:
            vector_agent = self.get_agent("vector_index")
            results = await vector_agent.search_code_by_pattern(query, limit=limit)
            
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
            """, (operation, duration, self.selected_gpu, "completed"))
            
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
        """Get system health status"""
        gpu_status = self.gpu_manager.get_status()
        
        return {
            "status": "healthy" if gpu_status['is_locked'] else "gpu_not_available",
            "coordinator_type": "single_gpu",
            "selected_gpu": self.selected_gpu,
            "gpu_status": gpu_status,
            "active_agents": len(self.agents),
            "stats": self.stats,
            "uptime_seconds": time.time() - self.stats["start_time"],
            "llm_engine_available": self.llm_engine is not None,
            "database_available": os.path.exists(self.db_path)
        }
    
    def cleanup(self):
        """Clean up resources without releasing GPU lock"""
        self.logger.info("🧹 Cleaning up coordinator resources...")
        
        # Clear agent cache
        self.agents.clear()
        
        # Clean up GPU memory
        self.gpu_manager.cleanup_gpu_memory()
        
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
        return (f"SingleGPUOpulenceCoordinator("
                f"gpu={self.selected_gpu}, "
                f"agents={len(self.agents)}, "
                f"tasks_completed={self.stats['tasks_completed']})")


# Enhanced Chat Capabilities
class SingleGPUChatEnhancer:
    """Enhanced chat capabilities for single GPU coordinator"""
    
    def __init__(self, coordinator: SingleGPUOpulenceCoordinator):
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
            analysis_result = await self.coordinator.analyze_component(component_name)
            
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
        """Search patterns with chat-enhanced results"""
        try:
            # Use vector search
            search_results = await self.coordinator.search_code_patterns(search_description, limit=10)
            
            # Get chat explanation
            chat_query = f"Explain these search results for '{search_description}' and help me understand what was found."
            
            # Create context
            search_context = [
                {
                    "role": "system", 
                    "content": f"Search results for '{search_description}': {json.dumps(search_results.get('results', [])[:5], default=str)}"
                }
            ]
            if conversation_history:
                search_context.extend(conversation_history)
            
            chat_result = await self.coordinator.process_chat_query(chat_query, search_context)
            
            return {
                "search_description": search_description,
                "search_results": search_results.get('results', []),
                "chat_explanation": chat_result.get("response", ""),
                "total_found": len(search_results.get('results', [])),
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


# Factory Functions
def create_single_gpu_coordinator(
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
    exclude_gpu_0: bool = True,
    min_memory_gb: float = 6.0,
    force_gpu_id: Optional[int] = None
) -> SingleGPUOpulenceCoordinator:
    """Create a single GPU coordinator with specified configuration"""
    
    config = SingleGPUOpulenceConfig(
        model_name=model_name,
        exclude_gpu_0=exclude_gpu_0,
        min_memory_gb=min_memory_gb,
        force_gpu_id=force_gpu_id,
        max_tokens=1024,  # Conservative for stability
        auto_cleanup=True
    )
    
    return SingleGPUOpulenceCoordinator(config)


def create_shared_server_coordinator() -> SingleGPUOpulenceCoordinator:
    """Create coordinator optimized for shared server environments"""
    
    config = SingleGPUOpulenceConfig(
        model_name="microsoft/DialoGPT-medium",  # Smaller model
        exclude_gpu_0=True,  # Don't compete with others
        min_memory_gb=4.0,   # Lower requirement
        max_tokens=512,      # Small context
        auto_cleanup=True
    )
    
    return SingleGPUOpulenceCoordinator(config)


def create_dedicated_server_coordinator() -> SingleGPUOpulenceCoordinator:
    """Create coordinator optimized for dedicated servers"""
    
    config = SingleGPUOpulenceConfig(
        model_name="codellama/CodeLlama-7b-Instruct-hf",
        exclude_gpu_0=False,  # Can use any GPU
        min_memory_gb=8.0,    # Higher requirement
        max_tokens=2048,      # Larger context
        auto_cleanup=True
    )
    
    return SingleGPUOpulenceCoordinator(config)


# Global coordinator instance for easy access
_global_coordinator = None

def get_global_coordinator() -> SingleGPUOpulenceCoordinator:
    """Get or create global coordinator instance"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = create_single_gpu_coordinator()
    return _global_coordinator


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
    return await coordinator.analyze_component(component_name, component_type)


async def quick_chat_query(query: str, conversation_history: List[Dict] = None) -> str:
    """Quick chat query using global coordinator"""
    coordinator = get_global_coordinator()
    enhancer = SingleGPUChatEnhancer(coordinator)
    return await enhancer.process_regular_chat_query(query, conversation_history)


def get_system_status() -> Dict[str, Any]:
    """Get system status using global coordinator"""
    coordinator = get_global_coordinator()
    return coordinator.get_health_status()


# Example Usage and Testing
async def run_example():
    """Complete example showing all functionality"""
    print("🚀 Starting Single GPU Coordinator Example")
    
    # Create coordinator with auto-cleanup
    with create_single_gpu_coordinator() as coordinator:
        print(f"✅ Coordinator ready on GPU {coordinator.selected_gpu}")
        
        # Process some files
        file_paths = [Path("example.cbl"), Path("data.csv")]
        print("📁 Processing files...")
        
        results = await coordinator.process_batch_files(file_paths, "auto")
        print(f"✅ Processed {results['files_processed']} files")
        print(f"   Successful: {results['successful_files']}")
        print(f"   Failed: {results['failed_files']}")
        
        # Analyze a component
        print("🔍 Analyzing component...")
        analysis = await coordinator.analyze_component("TRADE_DATE", "field")
        print(f"✅ Analysis status: {analysis['status']}")
        
        # Chat interaction
        print("💬 Processing chat query...")
        chat_result = await coordinator.process_chat_query("What is TRADE_DATE used for?")
        print(f"✅ Chat response: {chat_result.get('response', 'No response')[:100]}...")
        
        # Search patterns
        print("🔍 Searching code patterns...")
        search_results = await coordinator.search_code_patterns("date calculation", limit=5)
        print(f"✅ Found {search_results['total_found']} patterns")
        
        # Get system status
        print("📊 System status:")
        status = coordinator.get_health_status()
        print(f"   Status: {status['status']}")
        print(f"   GPU: {status['selected_gpu']}")
        print(f"   Agents: {status['active_agents']}")
        print(f"   Tasks completed: {status['stats']['tasks_completed']}")
        
    print("✅ Example completed, GPU automatically released")


def test_coordinator_basic():
    """Basic test of coordinator functionality"""
    print("Testing Single GPU Coordinator...")
    
    try:
        # Test creation
        coordinator = create_single_gpu_coordinator()
        print(f"✅ Created coordinator on GPU {coordinator.selected_gpu}")
        
        # Test agent creation
        agent = coordinator.get_agent("code_parser")
        print("✅ Created code parser agent")
        
        # Test status
        status = coordinator.get_health_status()
        print(f"✅ Status: {status['status']}")
        
        # Test cleanup
        coordinator.cleanup()
        print("✅ Cleanup successful")
        
        # Test shutdown
        coordinator.shutdown()
        print("✅ Shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic test
    if test_coordinator_basic():
        print("\n🎉 All tests passed!")
        
        # Run full example
        print("\n" + "="*50)
        asyncio.run(run_example())
    else:
        print("\n❌ Tests failed!")


# Advanced Features for Production Use
class ProductionCoordinatorManager:
    """Production-ready coordinator manager with monitoring and recovery"""
    
    def __init__(self, config: SingleGPUOpulenceConfig = None):
        self.config = config or SingleGPUOpulenceConfig()
        self.coordinator = None
        self.health_check_interval = 60  # seconds
        self.max_restart_attempts = 3
        self.restart_count = 0
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start coordinator with health monitoring"""
        try:
            self.coordinator = SingleGPUOpulenceCoordinator(self.config)
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            self.logger.info("✅ Production coordinator started")
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
                
                # Check GPU status
                status = self.coordinator.get_health_status()
                
                if status['status'] != 'healthy':
                    self.logger.warning(f"⚠️ Unhealthy status: {status['status']}")
                    
                    if self.restart_count < self.max_restart_attempts:
                        await self._restart_coordinator()
                    else:
                        self.logger.error("❌ Max restart attempts reached")
                        break
                
            except Exception as e:
                self.logger.error(f"❌ Health check failed: {e}")
    
    async def _restart_coordinator(self):
        """Restart coordinator"""
        try:
            self.logger.info("🔄 Restarting coordinator...")
            
            if self.coordinator:
                self.coordinator.shutdown()
            
            self.coordinator = SingleGPUOpulenceCoordinator(self.config)
            self.restart_count += 1
            
            self.logger.info(f"✅ Coordinator restarted (attempt {self.restart_count})")
            
        except Exception as e:
            self.logger.error(f"❌ Restart failed: {e}")
    
    def stop(self):
        """Stop coordinator"""
        if self.coordinator:
            self.coordinator.shutdown()
            self.coordinator = None


# Export main classes and functions
__all__ = [
    'SingleGPUOpulenceCoordinator',
    'SingleGPUOpulenceConfig', 
    'SingleGPUChatEnhancer',
    'create_single_gpu_coordinator',
    'create_shared_server_coordinator',
    'create_dedicated_server_coordinator',
    'get_global_coordinator',
    'quick_file_processing',
    'quick_component_analysis',
    'quick_chat_query',
    'get_system_status',
    'ProductionCoordinatorManager'
]