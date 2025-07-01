"""
Opulence - Complete Mainframe Analysis Module
Single comprehensive module for COBOL/JCL/Copybook analysis with chat interface
Includes: File parsing, vector indexing, lineage analysis, chat interface
Uses: CodeLlama for analysis + CodeBERT for vectors + Streamlit UI
"""

import asyncio
import sqlite3
import json
import re
import os
import time
import uuid
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime as dt
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd

# Core ML/AI imports
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import faiss

# UI imports
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Disable warnings and external connections
import warnings
warnings.filterwarnings("ignore")
os.environ.update({
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1'
})

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""
    program_name: str
    chunk_id: str
    chunk_type: str
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int
    file_type: str


@dataclass
class FieldLineage:
    """Represents field lineage information"""
    field_name: str
    source_programs: List[str]
    target_programs: List[str]
    operations: List[str]
    lifecycle_stage: str
    usage_count: int


@dataclass
class FileLifecycle:
    """Represents complete file lifecycle"""
    file_name: str
    file_type: str
    creation_points: List[str]
    read_operations: List[str]
    update_operations: List[str]
    dependencies: List[str]
    business_purpose: str


class OpulenceMainframeAnalyzer:
    """Complete mainframe analysis system in single module"""
    
    def __init__(self, 
                 model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
                 embedding_model_path: str = "./models/microsoft-codebert-base",
                 gpu_id: int = 0,
                 db_path: str = "opulence_complete.db"):
        
        self.model_name = model_name
        self.embedding_model_path = embedding_model_path
        self.gpu_id = gpu_id
        self.db_path = db_path
        
        # Initialize core components
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.llm_engine = None
        self.tokenizer = None
        self.embedding_model = None
        self.vector_index = None
        self.vector_dim = 768
        
        # In-memory storage for fast access
        self.parsed_files = {}
        self.field_lineage_cache = {}
        self.lifecycle_cache = {}
        
        # Initialize database and components
        self._init_database()
        self._init_models()
        
        logger.info(f"✅ Opulence Analyzer initialized on {self.device}")
    
    def safe_json_loads(self, json_str: str) -> Dict[str, Any]:
        """Safely load JSON string"""
        try:
            return json.loads(json_str) if json_str else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _init_database(self):
        """Initialize SQLite database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            -- File metadata table
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT,
                file_hash TEXT,
                processing_status TEXT DEFAULT 'pending',
                total_lines INTEGER,
                total_chunks INTEGER,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP
            );
            
            -- Program chunks table
            CREATE TABLE IF NOT EXISTS program_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                file_type TEXT,
                line_start INTEGER,
                line_end INTEGER,
                embedding_vector TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(program_name, chunk_id)
            );
            
            -- Field lineage table
            CREATE TABLE IF NOT EXISTS field_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT NOT NULL,
                program_name TEXT NOT NULL,
                operation_type TEXT,
                operation_context TEXT,
                chunk_id TEXT,
                confidence_score REAL DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- File lifecycle table
            CREATE TABLE IF NOT EXISTS file_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                lifecycle_stage TEXT NOT NULL,
                program_name TEXT,
                operation_details TEXT,
                business_purpose TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Chat history table
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                system_response TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Vector embeddings table
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                embedding_vector TEXT,
                faiss_id INTEGER,
                similarity_threshold REAL DEFAULT 0.7,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES program_chunks (id)
            );
            
            -- CICS analysis table
            CREATE TABLE IF NOT EXISTS cics_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT NOT NULL,
                transaction_type TEXT,
                analysis_data TEXT NOT NULL,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- DB2 analysis table
            CREATE TABLE IF NOT EXISTS db2_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT NOT NULL,
                access_pattern TEXT,
                analysis_data TEXT NOT NULL,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Complete lifecycle analysis table
            CREATE TABLE IF NOT EXISTS complete_lifecycle_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT NOT NULL,
                system_type TEXT,
                analysis_data TEXT NOT NULL,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Complete file lifecycle table
            CREATE TABLE IF NOT EXISTS complete_file_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                lifecycle_stages TEXT,
                data_journey TEXT,
                db2_integration TEXT,
                lifecycle_report TEXT,
                lifecycle_completeness REAL,
                data_flow_complexity TEXT,
                integration_score REAL,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Enhanced chat history table
            CREATE TABLE IF NOT EXISTS enhanced_chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                system_response TEXT,
                query_intent TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- File lifecycle chat history table
            CREATE TABLE IF NOT EXISTS file_lifecycle_chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                system_response TEXT,
                query_intent TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_chunks_program ON program_chunks(program_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON program_chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_lineage_field ON field_lineage(field_name);
            CREATE INDEX IF NOT EXISTS idx_lineage_program ON field_lineage(program_name);
            CREATE INDEX IF NOT EXISTS idx_lifecycle_file ON file_lifecycle(file_name);
            CREATE INDEX IF NOT EXISTS idx_cics_program ON cics_analysis(program_name);
            CREATE INDEX IF NOT EXISTS idx_db2_program ON db2_analysis(program_name);
            CREATE INDEX IF NOT EXISTS idx_complete_program ON complete_lifecycle_analysis(program_name);
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
    
    def _init_models(self):
        """Initialize LLM and embedding models"""
        try:
            # Initialize FAISS index
            self.vector_index = faiss.IndexFlatIP(self.vector_dim)
            
            # Initialize embedding model
            if Path(self.embedding_model_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.embedding_model_path, local_files_only=True
                )
                self.embedding_model = AutoModel.from_pretrained(
                    self.embedding_model_path, local_files_only=True
                )
                self.embedding_model.to(self.device)
                self.embedding_model.eval()
                logger.info("✅ Embedding model loaded")
            else:
                logger.warning("⚠️ Embedding model not found, will use simplified embeddings")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
    
    async def _init_llm_engine(self):
        """Initialize LLM engine on demand"""
        if self.llm_engine is None:
            try:
                engine_args = AsyncEngineArgs(
                    model=self.model_name,
                    gpu_memory_utilization=0.8,
                    max_model_len=2048,
                    tensor_parallel_size=1
                )
                self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                logger.info("✅ LLM engine initialized")
            except Exception as e:
                logger.error(f"❌ LLM initialization failed: {e}")
                raise
    
    # =================== FILE PARSING SECTION ===================
    
    async def load_and_parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse COBOL/JCL/Copybook file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Calculate file hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Determine file type
            file_type = self._detect_file_type(content, file_path.suffix)
            
            # Check if already processed
            if self._is_file_already_processed(file_path.name, file_hash):
                return await self._get_cached_file_analysis(file_path.name)
            
            # Parse content into chunks
            chunks = await self._parse_file_content(content, file_path.name, file_type)
            
            # Store in database
            await self._store_file_and_chunks(file_path, file_type, file_hash, chunks)
            
            # Create embeddings
            await self._create_embeddings_for_chunks(chunks)
            
            # Analyze lifecycle and lineage
            lifecycle_analysis = await self._analyze_file_lifecycle(file_path.name, chunks)
            lineage_analysis = await self._analyze_field_lineage(chunks)
            
            result = {
                "status": "success",
                "file_name": file_path.name,
                "file_type": file_type,
                "total_lines": len(content.split('\n')),
                "total_chunks": len(chunks),
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "lifecycle_analysis": lifecycle_analysis,
                "lineage_analysis": lineage_analysis,
                "processing_time": time.time()
            }
            
            # Cache result
            self.parsed_files[file_path.name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ File parsing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _detect_file_type(self, content: str, file_suffix: str) -> str:
        """Detect file type from content and suffix"""
        content_upper = content.upper()
        
        if 'IDENTIFICATION DIVISION' in content_upper or 'PROGRAM-ID' in content_upper:
            return "cobol"
        elif content.startswith('//') and 'JOB' in content_upper:
            return "jcl"
        elif file_suffix.lower() in ['.cpy', '.copy']:
            return "copybook"
        elif 'PIC ' in content_upper or 'PICTURE ' in content_upper:
            return "copybook"
        else:
            return "unknown"
    
    def _is_file_already_processed(self, file_name: str, file_hash: str) -> bool:
        """Check if file is already processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM file_metadata 
            WHERE file_name = ? AND file_hash = ? AND processing_status = 'completed'
        """, (file_name, file_hash))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    async def _parse_file_content(self, content: str, file_name: str, file_type: str) -> List[CodeChunk]:
        """Parse file content into logical chunks"""
        chunks = []
        lines = content.split('\n')
        
        if file_type == "cobol":
            chunks.extend(await self._parse_cobol_content(lines, file_name))
        elif file_type == "jcl":
            chunks.extend(await self._parse_jcl_content(lines, file_name))
        elif file_type == "copybook":
            chunks.extend(await self._parse_copybook_content(lines, file_name))
        else:
            # Generic parsing
            chunks.append(CodeChunk(
                program_name=file_name,
                chunk_id="full_content",
                chunk_type="unknown",
                content=content,
                metadata={"total_lines": len(lines)},
                line_start=1,
                line_end=len(lines),
                file_type=file_type
            ))
        
        return chunks
    
    async def _parse_cobol_content(self, lines: List[str], file_name: str) -> List[CodeChunk]:
        """Parse COBOL program into logical chunks"""
        chunks = []
        current_chunk = []
        current_chunk_type = "unknown"
        chunk_start_line = 1
        chunk_counter = 1
        
        for i, line in enumerate(lines, 1):
            line_upper = line.strip().upper()
            
            # Detect division boundaries
            if 'IDENTIFICATION DIVISION' in line_upper:
                if current_chunk:
                    chunks.append(await self._create_cobol_chunk(
                        current_chunk, file_name, current_chunk_type, 
                        chunk_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = [line]
                current_chunk_type = "identification_division"
                chunk_start_line = i
                
            elif 'ENVIRONMENT DIVISION' in line_upper:
                if current_chunk:
                    chunks.append(await self._create_cobol_chunk(
                        current_chunk, file_name, current_chunk_type,
                        chunk_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = [line]
                current_chunk_type = "environment_division"
                chunk_start_line = i
                
            elif 'DATA DIVISION' in line_upper:
                if current_chunk:
                    chunks.append(await self._create_cobol_chunk(
                        current_chunk, file_name, current_chunk_type,
                        chunk_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = [line]
                current_chunk_type = "data_division"
                chunk_start_line = i
                
            elif 'WORKING-STORAGE SECTION' in line_upper:
                if current_chunk and current_chunk_type != "working_storage":
                    chunks.append(await self._create_cobol_chunk(
                        current_chunk, file_name, current_chunk_type,
                        chunk_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = [line]
                current_chunk_type = "working_storage"
                chunk_start_line = i
                
            elif 'PROCEDURE DIVISION' in line_upper:
                if current_chunk:
                    chunks.append(await self._create_cobol_chunk(
                        current_chunk, file_name, current_chunk_type,
                        chunk_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = [line]
                current_chunk_type = "procedure_division"
                chunk_start_line = i
                
            elif line_upper.endswith('.') and len(line_upper.split()) <= 3 and current_chunk_type == "procedure_division":
                # Paragraph definition
                if current_chunk and len(current_chunk) > 1:
                    chunks.append(await self._create_cobol_chunk(
                        current_chunk, file_name, current_chunk_type,
                        chunk_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = [line]
                current_chunk_type = "paragraph"
                chunk_start_line = i
                
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(await self._create_cobol_chunk(
                current_chunk, file_name, current_chunk_type,
                chunk_start_line, len(lines), chunk_counter
            ))
        
        return chunks
    
    async def _create_cobol_chunk(self, lines: List[str], file_name: str, 
                                chunk_type: str, start_line: int, end_line: int, 
                                chunk_number: int) -> CodeChunk:
        """Create COBOL chunk with metadata"""
        content = '\n'.join(lines)
        
        # Extract metadata based on chunk type
        metadata = await self._extract_cobol_metadata(content, chunk_type)
        metadata.update({
            "chunk_number": chunk_number,
            "line_count": len(lines),
            "chunk_size": len(content)
        })
        
        return CodeChunk(
            program_name=file_name.replace('.cbl', '').replace('.cob', ''),
            chunk_id=f"{chunk_type}_{chunk_number:03d}",
            chunk_type=chunk_type,
            content=content,
            metadata=metadata,
            line_start=start_line,
            line_end=end_line,
            file_type="cobol"
        )
    
    async def _extract_cobol_metadata(self, content: str, chunk_type: str) -> Dict[str, Any]:
        """Extract metadata from COBOL chunk using LLM analysis"""
        await self._init_llm_engine()
        
        metadata = {"chunk_type": chunk_type}
        
        try:
            # Use LLM for comprehensive metadata extraction
            metadata_prompt = f"""
            Analyze this COBOL {chunk_type} code and extract detailed metadata:
            
            {content[:1500]}
            
            Extract and return JSON with:
            1. field_names: List of all field/variable names
            2. operations: List of COBOL operations (MOVE, COMPUTE, etc.)
            3. file_operations: List of file operations (READ, WRITE, etc.)
            4. cics_operations: List of CICS commands (EXEC CICS, etc.)
            5. db2_operations: List of DB2 operations (EXEC SQL, etc.)
            6. called_programs: List of programs called
            7. performed_paragraphs: List of paragraphs performed
            8. business_logic: Brief description of business purpose
            9. data_structures: Complex data structures identified
            10. error_handling: Error handling patterns found
            11. transaction_type: Type of transaction (online, batch, etc.)
            12. database_tables: Tables accessed
            13. cics_resources: CICS resources used (maps, files, etc.)
            14. complexity_indicators: Indicators of code complexity
            
            Return valid JSON only.
            """
            
            sampling_params = SamplingParams(temperature=0.1, max_tokens=800)
            request_id = str(uuid.uuid4())
            
            try:
                async for result in self.llm_engine.generate(metadata_prompt, sampling_params, request_id=request_id):
                    response_text = result.outputs[0].text.strip()
                    
                    # Extract JSON from response
                    if '{' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        llm_metadata = json.loads(response_text[json_start:json_end])
                        metadata.update(llm_metadata)
                    break
            except Exception as e:
                logger.warning(f"⚠️ LLM metadata extraction failed: {e}")
                # Fallback to regex extraction
                metadata.update(await self._fallback_metadata_extraction(content, chunk_type))
        
        except Exception as e:
            logger.warning(f"⚠️ Metadata extraction failed: {e}")
            metadata.update(await self._fallback_metadata_extraction(content, chunk_type))
        
        return metadata
    async def _fallback_metadata_extraction(self, content: str, chunk_type: str) -> Dict[str, Any]:
        """Fallback regex-based metadata extraction"""
        metadata = {}
        
        try:
            # Basic field extraction
            field_pattern = re.compile(r'^\s*\d+\s+([A-Z0-9\-_]+)', re.MULTILINE | re.IGNORECASE)
            fields = field_pattern.findall(content)
            metadata["field_names"] = list(set(fields))
            
            # COBOL operations
            operations = []
            for keyword in ['MOVE', 'COMPUTE', 'PERFORM', 'IF', 'READ', 'WRITE', 'CALL', 'ADD', 'SUBTRACT']:
                if keyword in content.upper():
                    operations.append(keyword)
            metadata["operations"] = operations
            
            # CICS operations
            cics_ops = []
            cics_pattern = re.compile(r'EXEC\s+CICS\s+(\w+)', re.IGNORECASE)
            cics_commands = cics_pattern.findall(content)
            metadata["cics_operations"] = list(set(cics_commands))
            
            # DB2 operations
            db2_ops = []
            db2_pattern = re.compile(r'EXEC\s+SQL\s+(\w+)', re.IGNORECASE)
            sql_commands = db2_pattern.findall(content)
            metadata["db2_operations"] = list(set(sql_commands))
            
            # Database tables
            table_pattern = re.compile(r'FROM\s+([A-Z0-9_]+)|INTO\s+([A-Z0-9_]+)|UPDATE\s+([A-Z0-9_]+)', re.IGNORECASE)
            table_matches = table_pattern.findall(content)
            tables = []
            for match in table_matches:
                tables.extend([t for t in match if t])
            metadata["database_tables"] = list(set(tables))
            
            # File operations
            file_ops = []
            for op in ['OPEN', 'CLOSE', 'READ', 'WRITE', 'REWRITE', 'DELETE']:
                if op in content.upper():
                    file_ops.append(op)
            metadata["file_operations"] = file_ops
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback metadata extraction failed: {e}")
        
        return metadata
    
    async def _parse_jcl_content(self, lines: List[str], file_name: str) -> List[CodeChunk]:
        """Parse JCL into logical chunks"""
        chunks = []
        current_step = []
        step_name = "UNKNOWN"
        chunk_counter = 1
        step_start_line = 1
        
        for i, line in enumerate(lines, 1):
            if line.startswith('//') and ' EXEC ' in line.upper():
                # New job step
                if current_step:
                    chunks.append(await self._create_jcl_chunk(
                        current_step, file_name, step_name,
                        step_start_line, i-1, chunk_counter
                    ))
                    chunk_counter += 1
                
                # Extract step name
                step_match = re.match(r'//(\w+)\s+EXEC', line, re.IGNORECASE)
                step_name = step_match.group(1) if step_match else f"STEP_{chunk_counter}"
                current_step = [line]
                step_start_line = i
                
            elif line.startswith('//') or line.startswith(' '):
                current_step.append(line)
            else:
                if line.strip():  # Non-empty line
                    current_step.append(line)
        
        # Add final step
        if current_step:
            chunks.append(await self._create_jcl_chunk(
                current_step, file_name, step_name,
                step_start_line, len(lines), chunk_counter
            ))
        
        return chunks
    
    async def _create_jcl_chunk(self, lines: List[str], file_name: str,
                              step_name: str, start_line: int, end_line: int,
                              chunk_number: int) -> CodeChunk:
        """Create JCL chunk with metadata"""
        content = '\n'.join(lines)
        
        metadata = {
            "step_name": step_name,
            "chunk_number": chunk_number,
            "line_count": len(lines)
        }
        
        # Extract JCL metadata
        metadata.update(await self._extract_jcl_metadata(content))
        
        return CodeChunk(
            program_name=file_name.replace('.jcl', ''),
            chunk_id=f"step_{step_name}_{chunk_number:03d}",
            chunk_type="job_step",
            content=content,
            metadata=metadata,
            line_start=start_line,
            line_end=end_line,
            file_type="jcl"
        )
    
    async def _extract_jcl_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from JCL chunk"""
        metadata = {}
        
        try:
            # Extract DD statements
            dd_pattern = re.compile(r'//(\w+)\s+DD', re.IGNORECASE)
            dd_names = dd_pattern.findall(content)
            metadata["dd_statements"] = dd_names
            
            # Extract dataset names
            dsn_pattern = re.compile(r'DSN=([^\s,]+)', re.IGNORECASE)
            datasets = dsn_pattern.findall(content)
            metadata["datasets"] = datasets
            
            # Extract programs called
            pgm_pattern = re.compile(r'PGM=([^\s,]+)', re.IGNORECASE)
            programs = pgm_pattern.findall(content)
            metadata["programs_executed"] = programs
            
        except Exception as e:
            logger.warning(f"⚠️ JCL metadata extraction failed: {e}")
        
        return metadata
    
    async def _parse_copybook_content(self, lines: List[str], file_name: str) -> List[CodeChunk]:
        """Parse copybook into field definitions"""
        chunks = []
        current_group = []
        group_level = None
        chunk_counter = 1
        group_start_line = 1
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for level numbers
            level_match = re.match(r'^\s*(\d+)\s+', line)
            if level_match:
                level = int(level_match.group(1))
                
                # Start new group for 01 level
                if level == 1:
                    if current_group:
                        chunks.append(await self._create_copybook_chunk(
                            current_group, file_name, group_level,
                            group_start_line, i-1, chunk_counter
                        ))
                        chunk_counter += 1
                    
                    current_group = [line]
                    group_level = level
                    group_start_line = i
                else:
                    current_group.append(line)
            else:
                if line_stripped:
                    current_group.append(line)
        
        # Add final group
        if current_group:
            chunks.append(await self._create_copybook_chunk(
                current_group, file_name, group_level,
                group_start_line, len(lines), chunk_counter
            ))
        
        return chunks
    
    async def _create_copybook_chunk(self, lines: List[str], file_name: str,
                                   group_level: int, start_line: int, end_line: int,
                                   chunk_number: int) -> CodeChunk:
        """Create copybook chunk with metadata"""
        content = '\n'.join(lines)
        
        metadata = await self._extract_copybook_metadata(content)
        metadata.update({
            "group_level": group_level,
            "chunk_number": chunk_number,
            "line_count": len(lines)
        })
        
        return CodeChunk(
            program_name=file_name.replace('.cpy', '').replace('.copy', ''),
            chunk_id=f"group_{group_level}_{chunk_number:03d}",
            chunk_type="data_structure",
            content=content,
            metadata=metadata,
            line_start=start_line,
            line_end=end_line,
            file_type="copybook"
        )
    
    async def _extract_copybook_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from copybook chunk"""
        metadata = {}
        
        try:
            # Extract all field definitions
            field_pattern = re.compile(r'^\s*(\d+)\s+([A-Z0-9\-_]+)', re.MULTILINE | re.IGNORECASE)
            fields = field_pattern.findall(content)
            metadata["fields"] = [{"level": int(level), "name": name} for level, name in fields]
            metadata["field_names"] = [name for level, name in fields]
            
            # Extract PIC clauses with field mapping
            pic_pattern = re.compile(r'^\s*\d+\s+([A-Z0-9\-_]+).*?PIC\s+([X9VS\(\)\+\-\$,\.]+)', 
                                   re.MULTILINE | re.IGNORECASE)
            pic_mappings = pic_pattern.findall(content)
            metadata["pic_mappings"] = {field: pic for field, pic in pic_mappings}
            
        except Exception as e:
            logger.warning(f"⚠️ Copybook metadata extraction failed: {e}")
        
        return metadata
    
    def _chunk_to_dict(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Convert CodeChunk to dictionary"""
        return {
            "program_name": chunk.program_name,
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "file_type": chunk.file_type
        }
    
    # =================== STORAGE SECTION ===================
    
    async def _store_file_and_chunks(self, file_path: Path, file_type: str, 
                                   file_hash: str, chunks: List[CodeChunk]):
        """Store file metadata and chunks in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            # Store file metadata
            cursor.execute("""
                INSERT OR REPLACE INTO file_metadata 
                (file_name, file_type, file_path, file_hash, processing_status, 
                 total_lines, total_chunks, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path.name, file_type, str(file_path), file_hash, 'completed',
                sum(chunk.line_end - chunk.line_start + 1 for chunk in chunks),
                len(chunks), dt.now().isoformat()
            ))
            
            # Store chunks
            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO program_chunks 
                    (program_name, chunk_id, chunk_type, content, metadata, 
                     file_type, line_start, line_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.program_name, chunk.chunk_id, chunk.chunk_type,
                    chunk.content, json.dumps(chunk.metadata), chunk.file_type,
                    chunk.line_start, chunk.line_end
                ))
            
            cursor.execute("COMMIT")
            logger.info(f"✅ Stored {len(chunks)} chunks for {file_path.name}")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"❌ Storage failed: {e}")
            raise
        finally:
            conn.close()
    
    # =================== VECTOR EMBEDDINGS SECTION ===================
    
    async def _create_embeddings_for_chunks(self, chunks: List[CodeChunk]):
        """Create vector embeddings for chunks"""
        if not self.embedding_model:
            logger.warning("⚠️ No embedding model available")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for chunk in chunks:
                # Prepare text for embedding
                text_to_embed = self._prepare_text_for_embedding(chunk)
                
                # Generate embedding
                embedding = await self._generate_embedding(text_to_embed)
                
                if embedding is not None:
                    # Add to FAISS index
                    faiss_id = self.vector_index.ntotal
                    self.vector_index.add(embedding.reshape(1, -1).astype('float32'))
                    
                    # Store embedding in database
                    cursor.execute("""
                        SELECT id FROM program_chunks 
                        WHERE program_name = ? AND chunk_id = ?
                    """, (chunk.program_name, chunk.chunk_id))
                    
                    chunk_db_id = cursor.fetchone()
                    if chunk_db_id:
                        cursor.execute("""
                            INSERT OR REPLACE INTO vector_embeddings 
                            (chunk_id, embedding_vector, faiss_id)
                            VALUES (?, ?, ?)
                        """, (chunk_db_id[0], json.dumps(embedding.tolist()), faiss_id))
            
            conn.commit()
            logger.info(f"✅ Created embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
        finally:
            conn.close()
    
    def _prepare_text_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare chunk text for embedding"""
        text_parts = [chunk.content.strip()]
        
        # Add metadata context
        if chunk.metadata:
            if 'field_names' in chunk.metadata:
                fields = chunk.metadata['field_names'][:10]  # Limit to avoid token overflow
                text_parts.append(f"Fields: {', '.join(fields)}")
            
            if 'operations' in chunk.metadata:
                ops = chunk.metadata['operations'][:5]
                text_parts.append(f"Operations: {', '.join(ops)}")
            
            if 'called_programs' in chunk.metadata:
                programs = chunk.metadata['called_programs'][:3]
                text_parts.append(f"Calls: {', '.join(programs)}")
        
        return " | ".join(text_parts)
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.flatten()
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return None
    
    # =================== LINEAGE ANALYSIS SECTION ===================
    
    async def _analyze_field_lineage(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Analyze field lineage across chunks"""
        field_lineage = defaultdict(list)
        field_operations = defaultdict(list)
        
        try:
            for chunk in chunks:
                # Extract field references and operations
                field_refs = await self._extract_field_references(chunk)
                
                for field_ref in field_refs:
                    field_name = field_ref['field_name']
                    operation = field_ref['operation']
                    context = field_ref['context']
                    
                    field_lineage[field_name].append({
                        'program': chunk.program_name,
                        'chunk_id': chunk.chunk_id,
                        'operation': operation,
                        'context': context,
                        'chunk_type': chunk.chunk_type
                    })
                    
                    field_operations[field_name].append(operation)
                    
                    # Store in database
                    await self._store_field_lineage(
                        field_name, chunk.program_name, operation, 
                        context, chunk.chunk_id
                    )
            
            # Analyze lineage patterns
            lineage_summary = {}
            for field_name, references in field_lineage.items():
                lineage_summary[field_name] = {
                    'total_references': len(references),
                    'programs_involved': list(set(ref['program'] for ref in references)),
                    'operations': list(set(field_operations[field_name])),
                    'lifecycle_stage': self._determine_field_lifecycle_stage(field_operations[field_name]),
                    'usage_pattern': self._analyze_field_usage_pattern(references),
                    'references': references
                }
            
            return {
                'field_lineage': lineage_summary,
                'total_fields_analyzed': len(field_lineage),
                'fields_with_multiple_refs': len([f for f, refs in field_lineage.items() if len(refs) > 1])
            }
            
        except Exception as e:
            logger.error(f"❌ Field lineage analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_field_references(self, chunk: CodeChunk) -> List[Dict[str, Any]]:
        """Extract field references from chunk content"""
        field_refs = []
        content = chunk.content.upper()
        
        try:
            # Define patterns for different operations
            patterns = {
                'READ': [r'MOVE\s+(\w+)\s+TO', r'IF\s+(\w+)', r'COMPUTE.*?(\w+)'],
                'WRITE': [r'MOVE\s+.*?\s+TO\s+(\w+)', r'COMPUTE\s+(\w+)'],
                'UPDATE': [r'ADD\s+.*?\s+TO\s+(\w+)', r'SUBTRACT\s+.*?\s+FROM\s+(\w+)'],
                'DEFINE': [r'^\s*\d+\s+(\w+)', r'PIC.*?(\w+)']
            }
            
            for operation, operation_patterns in patterns.items():
                for pattern in operation_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        
                        # Filter out common COBOL keywords
                        if self._is_valid_field_name(match):
                            field_refs.append({
                                'field_name': match,
                                'operation': operation,
                                'context': self._extract_field_context(content, match),
                                'confidence': 0.8
                            })
            
            # Also extract from metadata if available
            if chunk.metadata and 'field_names' in chunk.metadata:
                for field_name in chunk.metadata['field_names']:
                    if not any(ref['field_name'] == field_name for ref in field_refs):
                        field_refs.append({
                            'field_name': field_name,
                            'operation': 'DEFINE' if chunk.chunk_type == 'working_storage' else 'REFERENCE',
                            'context': 'Field definition',
                            'confidence': 0.9
                        })
            
        except Exception as e:
            logger.warning(f"⚠️ Field reference extraction failed: {e}")
        
        return field_refs
    
    def _is_valid_field_name(self, name: str) -> bool:
        """Check if name is a valid field name"""
        if not name or len(name) < 2:
            return False
        
        # Exclude common COBOL keywords
        keywords = {
            'IF', 'THEN', 'ELSE', 'END', 'MOVE', 'TO', 'FROM', 'COMPUTE',
            'PERFORM', 'UNTIL', 'VARYING', 'BY', 'CALL', 'USING', 'RETURNING',
            'PIC', 'PICTURE', 'VALUE', 'OCCURS', 'REDEFINES', 'DEPENDING'
        }
        
        return name.upper() not in keywords and name.replace('-', '').replace('_', '').isalnum()
    
    def _extract_field_context(self, content: str, field_name: str) -> str:
        """Extract context around field usage"""
        lines = content.split('\n')
        for line in lines:
            if field_name.upper() in line.upper():
                return line.strip()[:100]  # Return first 100 chars of context
        return "No context found"
    
    def _determine_field_lifecycle_stage(self, operations: List[str]) -> str:
        """Determine field lifecycle stage based on operations"""
        if 'DEFINE' in operations:
            if any(op in operations for op in ['WRITE', 'UPDATE']):
                return 'ACTIVE'
            else:
                return 'DEFINED'
        elif any(op in operations for op in ['READ', 'REFERENCE']):
            return 'USED'
        else:
            return 'UNKNOWN'
    
    def _analyze_field_usage_pattern(self, references: List[Dict[str, Any]]) -> str:
        """Analyze field usage pattern"""
        operations = [ref['operation'] for ref in references]
        
        if 'DEFINE' in operations and any(op in operations for op in ['READ', 'WRITE']):
            return 'ACTIVE_FIELD'
        elif operations.count('READ') > operations.count('WRITE'):
            return 'READ_MOSTLY'
        elif operations.count('WRITE') > operations.count('READ'):
            return 'WRITE_MOSTLY'
        else:
            return 'BALANCED_USAGE'
    
    async def _store_field_lineage(self, field_name: str, program_name: str, 
                                 operation: str, context: str, chunk_id: str):
        """Store field lineage in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO field_lineage 
                (field_name, program_name, operation_type, operation_context, chunk_id)
                VALUES (?, ?, ?, ?, ?)
            """, (field_name, program_name, operation, context, chunk_id))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ Field lineage storage failed: {e}")
        finally:
            conn.close()
    
    # =================== LIFECYCLE ANALYSIS SECTION ===================
    
    async def _analyze_file_lifecycle(self, file_name: str, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Analyze complete file lifecycle"""
        lifecycle_stages = {
            'creation': [],
            'read_operations': [],
            'update_operations': [],
            'dependencies': [],
            'business_purpose': ''
        }
        
        try:
            for chunk in chunks:
                # Analyze operations in each chunk
                operations = await self._extract_lifecycle_operations(chunk)
                
                for operation in operations:
                    stage = self._map_operation_to_lifecycle_stage(operation['type'])
                    if stage:
                        lifecycle_stages[stage].append({
                            'program': chunk.program_name,
                            'chunk_id': chunk.chunk_id,
                            'operation': operation['type'],
                            'details': operation['details'],
                            'context': operation.get('context', '')
                        })
                
                # Extract dependencies
                dependencies = await self._extract_chunk_dependencies(chunk)
                lifecycle_stages['dependencies'].extend(dependencies)
            
            # Remove duplicates from dependencies
            lifecycle_stages['dependencies'] = list(set(lifecycle_stages['dependencies']))
            
            # Determine business purpose using LLM
            lifecycle_stages['business_purpose'] = await self._determine_business_purpose(chunks)
            
            # Store lifecycle information
            await self._store_file_lifecycle(file_name, lifecycle_stages)
            
            return lifecycle_stages
            
        except Exception as e:
            logger.error(f"❌ Lifecycle analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_lifecycle_operations(self, chunk: CodeChunk) -> List[Dict[str, Any]]:
        """Extract lifecycle operations from chunk"""
        operations = []
        content = chunk.content.upper()
        
        # File operations
        file_op_patterns = {
            'CREATE': [r'OPEN\s+OUTPUT', r'WRITE\s+.*INVALID'],
            'READ': [r'OPEN\s+INPUT', r'READ\s+\w+', r'SELECT\s+.*FROM'],
            'UPDATE': [r'REWRITE', r'UPDATE\s+.*SET'],
            'DELETE': [r'DELETE\s+\w+', r'DELETE\s+.*WHERE']
        }
        
        for op_type, patterns in file_op_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    operations.append({
                        'type': op_type,
                        'details': match,
                        'context': self._extract_operation_context(content, match)
                    })
        
        return operations
    
    def _map_operation_to_lifecycle_stage(self, operation_type: str) -> Optional[str]:
        """Map operation type to lifecycle stage"""
        mapping = {
            'CREATE': 'creation',
            'READ': 'read_operations',
            'UPDATE': 'update_operations',
            'DELETE': 'update_operations'
        }
        return mapping.get(operation_type)
    
    def _extract_operation_context(self, content: str, operation: str) -> str:
        """Extract context around operation"""
        lines = content.split('\n')
        for line in lines:
            if operation in line:
                return line.strip()[:100]
        return "No context found"
    
    async def _extract_chunk_dependencies(self, chunk: CodeChunk) -> List[str]:
        """Extract dependencies from chunk"""
        dependencies = []
        
        # Extract from metadata if available
        if chunk.metadata:
            for key in ['called_programs', 'datasets', 'files']:
                if key in chunk.metadata:
                    dependencies.extend(chunk.metadata[key])
        
        # Extract from content
        content = chunk.content.upper()
        
        # COBOL CALL statements
        call_pattern = re.compile(r'CALL\s+[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
        called_programs = call_pattern.findall(content)
        dependencies.extend(called_programs)
        
        # JCL DSN references
        dsn_pattern = re.compile(r'DSN=([^\s,]+)', re.IGNORECASE)
        datasets = dsn_pattern.findall(content)
        dependencies.extend(datasets)
        
        return dependencies
    
    async def _determine_business_purpose(self, chunks: List[CodeChunk]) -> str:
        """Determine business purpose using LLM analysis"""
        try:
            await self._init_llm_engine()
            
            # Prepare content for analysis
            combined_content = ""
            for chunk in chunks[:3]:  # Analyze first 3 chunks to avoid token limit
                combined_content += f"\n{chunk.chunk_type}: {chunk.content[:200]}"
            
            prompt = f"""
            Analyze this mainframe code and determine its business purpose:
            
            {combined_content[:1000]}...
            
            Provide a concise business purpose description (2-3 sentences maximum).
            Focus on what business function this code performs.
            """
            
            sampling_params = SamplingParams(temperature=0.2, max_tokens=150)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            logger.warning(f"⚠️ Business purpose analysis failed: {e}")
            return "Business purpose analysis not available"
    
    async def _store_file_lifecycle(self, file_name: str, lifecycle_stages: Dict[str, Any]):
        """Store file lifecycle information in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for stage_name, stage_data in lifecycle_stages.items():
                if stage_name == 'business_purpose':
                    cursor.execute("""
                        INSERT INTO file_lifecycle 
                        (file_name, lifecycle_stage, business_purpose)
                        VALUES (?, ?, ?)
                    """, (file_name, 'business_purpose', stage_data))
                elif isinstance(stage_data, list):
                    for item in stage_data:
                        if isinstance(item, dict):
                            cursor.execute("""
                                INSERT INTO file_lifecycle 
                                (file_name, lifecycle_stage, program_name, operation_details)
                                VALUES (?, ?, ?, ?)
                            """, (file_name, stage_name, 
                                 item.get('program', ''), 
                                 json.dumps(item)))
                        else:
                            cursor.execute("""
                                INSERT INTO file_lifecycle 
                                (file_name, lifecycle_stage, operation_details)
                                VALUES (?, ?, ?)
                            """, (file_name, stage_name, str(item)))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ Lifecycle storage failed: {e}")
        finally:
            conn.close()
    
    # =================== ENHANCED CICS/DB2 ANALYSIS SECTION ===================
    
    async def analyze_cics_transaction_lifecycle(self, program_name: str) -> Dict[str, Any]:
        """Analyze complete CICS transaction lifecycle using LLM"""
        try:
            await self._init_llm_engine()
            
            # Get program chunks
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_type, content, metadata 
                FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
            """, (program_name, f"%{program_name}%"))
            
            chunks = cursor.fetchall()
            conn.close()
            
            if not chunks:
                return {'status': 'not_found', 'program': program_name}
            
            # Combine chunks for comprehensive analysis
            combined_content = ""
            cics_operations = []
            all_metadata = {}
            
            for chunk_type, content, metadata_str in chunks:
                combined_content += f"\n{chunk_type}:\n{content}\n"
                
                if metadata_str:
                    metadata = self.safe_json_loads(metadata_str)
                    if 'cics_operations' in metadata:
                        cics_operations.extend(metadata['cics_operations'])
                    all_metadata.update(metadata)
            
            # LLM-based CICS lifecycle analysis
            cics_analysis_prompt = f"""
            Analyze this COBOL program for complete CICS transaction lifecycle:
            
            Program: {program_name}
            CICS Operations Found: {list(set(cics_operations))}
            
            Code:
            {combined_content[:3000]}...
            
            Provide comprehensive CICS transaction lifecycle analysis in JSON format:
            {{
                "transaction_type": "online/batch/mixed",
                "transaction_flow": [
                    {{
                        "stage": "initialization",
                        "description": "What happens at start",
                        "cics_commands": ["RECEIVE", "GETMAIN"],
                        "business_purpose": "Purpose of this stage"
                    }}
                ],
                "cics_resources_used": {{
                    "maps": ["map1", "map2"],
                    "files": ["file1", "file2"], 
                    "queues": ["queue1"],
                    "programs": ["prog1", "prog2"],
                    "terminals": true/false
                }},
                "data_flow": {{
                    "input_sources": ["terminal", "files", "databases"],
                    "output_destinations": ["terminal", "files", "databases"],
                    "intermediate_storage": ["working storage", "temp files"]
                }},
                "error_handling": {{
                    "error_conditions": ["condition1", "condition2"],
                    "recovery_mechanisms": ["mechanism1", "mechanism2"],
                    "user_notification": "how errors are shown to user"
                }},
                "performance_characteristics": {{
                    "response_time_factors": ["factor1", "factor2"],
                    "resource_usage": "high/medium/low",
                    "scalability_notes": "notes about scalability"
                }},
                "security_aspects": {{
                    "authentication": "method",
                    "authorization": "access controls",
                    "data_protection": "sensitive data handling"
                }},
                "business_function": "Complete description of business purpose",
                "transaction_completion": {{
                    "success_path": "Normal completion steps",
                    "cleanup_required": "Resources to clean up",
                    "commit_rollback": "Transaction integrity handling"
                }},
                "recommendations": ["improvement1", "improvement2"]
            }}
            """
            
            sampling_params = SamplingParams(temperature=0.2, max_tokens=1500)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(cics_analysis_prompt, sampling_params, request_id=request_id):
                response_text = result.outputs[0].text.strip()
                
                try:
                    if '{' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        cics_lifecycle = json.loads(response_text[json_start:json_end])
                        
                        # Store CICS analysis
                        await self._store_cics_analysis(program_name, cics_lifecycle)
                        
                        return {
                            'status': 'success',
                            'program_name': program_name,
                            'cics_lifecycle': cics_lifecycle,
                            'raw_operations': list(set(cics_operations)),
                            'analysis_timestamp': dt.now().isoformat()
                        }
                except Exception as e:
                    logger.error(f"❌ Failed to parse CICS analysis: {e}")
                break
            
            # Fallback analysis
            return await self._fallback_cics_analysis(program_name, cics_operations, all_metadata)
            
        except Exception as e:
            logger.error(f"❌ CICS lifecycle analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}

    async def analyze_db2_data_lifecycle(self, program_name: str) -> Dict[str, Any]:
        """Analyze complete DB2 data lifecycle using LLM"""
        try:
            await self._init_llm_engine()
            
            # Get program chunks with DB2 operations
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_type, content, metadata 
                FROM program_chunks 
                WHERE (program_name = ? OR program_name LIKE ?)
                AND (content LIKE '%EXEC SQL%' OR metadata LIKE '%db2%')
            """, (program_name, f"%{program_name}%"))
            
            chunks = cursor.fetchall()
            conn.close()
            
            if not chunks:
                return {'status': 'no_db2_operations', 'program': program_name}
            
            # Extract SQL operations and tables
            sql_operations = []
            tables_accessed = []
            combined_sql_content = ""
            
            for chunk_type, content, metadata_str in chunks:
                # Extract EXEC SQL blocks
                sql_pattern = re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE)
                sql_blocks = sql_pattern.findall(content)
                
                for sql_block in sql_blocks:
                    combined_sql_content += f"\n{sql_block.strip()}\n"
                    
                    # Extract operation type
                    sql_upper = sql_block.upper().strip()
                    if sql_upper.startswith('SELECT'):
                        sql_operations.append('SELECT')
                    elif sql_upper.startswith('INSERT'):
                        sql_operations.append('INSERT')
                    elif sql_upper.startswith('UPDATE'):
                        sql_operations.append('UPDATE')
                    elif sql_upper.startswith('DELETE'):
                        sql_operations.append('DELETE')
                
                if metadata_str:
                    metadata = self.safe_json_loads(metadata_str)
                    if 'database_tables' in metadata:
                        tables_accessed.extend(metadata['database_tables'])
            
            # LLM-based DB2 lifecycle analysis
            db2_analysis_prompt = f"""
            Analyze this COBOL program for complete DB2 data lifecycle:
            
            Program: {program_name}
            SQL Operations: {list(set(sql_operations))}
            Tables: {list(set(tables_accessed))}
            
            SQL Code:
            {combined_sql_content[:2000]}...
            
            Provide comprehensive DB2 data lifecycle analysis in JSON format:
            {{
                "data_access_pattern": "read-only/read-write/write-heavy/mixed",
                "database_lifecycle": [
                    {{
                        "stage": "data_retrieval",
                        "operations": ["SELECT statements"],
                        "tables_involved": ["table1", "table2"],
                        "business_purpose": "Why this data is needed",
                        "performance_impact": "Expected query performance"
                    }}
                ],
                "table_operations": {{
                    "table1": {{
                        "operations": ["SELECT", "UPDATE"],
                        "access_frequency": "high/medium/low",
                        "data_volume": "estimate",
                        "business_criticality": "high/medium/low"
                    }}
                }},
                "data_transformations": [
                    {{
                        "source": "input source",
                        "target": "output target", 
                        "transformation_logic": "what happens to data",
                        "validation_rules": "data validation applied"
                    }}
                ],
                "transaction_management": {{
                    "commit_strategy": "when commits happen",
                    "rollback_scenarios": "when rollbacks occur",
                    "isolation_level": "transaction isolation",
                    "deadlock_prevention": "deadlock handling"
                }},
                "data_integrity": {{
                    "constraints_enforced": ["constraint1", "constraint2"],
                    "referential_integrity": "how foreign keys handled",
                    "data_validation": "validation rules applied",
                    "audit_trail": "change tracking"
                }},
                "performance_considerations": {{
                    "index_usage": "how indexes are used",
                    "query_optimization": "optimization techniques",
                    "resource_usage": "CPU/IO impact",
                    "scalability_factors": "factors affecting scale"
                }},
                "security_aspects": {{
                    "access_controls": "who can access what",
                    "sensitive_data": "sensitive data handling",
                    "encryption": "data encryption used",
                    "audit_logging": "access audit requirements"
                }},
                "data_quality": {{
                    "validation_rules": "data quality checks",
                    "error_handling": "how errors are handled",
                    "data_cleansing": "data cleaning processes",
                    "monitoring": "data quality monitoring"
                }},
                "business_impact": "How this data processing affects business",
                "recommendations": ["optimization1", "improvement2"]
            }}
            """
            
            sampling_params = SamplingParams(temperature=0.2, max_tokens=1500)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(db2_analysis_prompt, sampling_params, request_id=request_id):
                response_text = result.outputs[0].text.strip()
                
                try:
                    if '{' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        db2_lifecycle = json.loads(response_text[json_start:json_end])
                        
                        # Store DB2 analysis
                        await self._store_db2_analysis(program_name, db2_lifecycle)
                        
                        return {
                            'status': 'success',
                            'program_name': program_name,
                            'db2_lifecycle': db2_lifecycle,
                            'sql_operations': list(set(sql_operations)),
                            'tables_accessed': list(set(tables_accessed)),
                            'analysis_timestamp': dt.now().isoformat()
                        }
                except Exception as e:
                    logger.error(f"❌ Failed to parse DB2 analysis: {e}")
                break
            
            # Fallback analysis
            return await self._fallback_db2_analysis(program_name, sql_operations, tables_accessed)
            
        except Exception as e:
            logger.error(f"❌ DB2 lifecycle analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _store_cics_analysis(self, program_name: str, analysis: Dict[str, Any]):
        """Store CICS analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO cics_analysis 
                (program_name, transaction_type, analysis_data)
                VALUES (?, ?, ?)
            """, (
                program_name,
                analysis.get('transaction_type', 'unknown'),
                json.dumps(analysis)
            ))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ CICS analysis storage failed: {e}")
        finally:
            conn.close()
    
    async def _store_db2_analysis(self, program_name: str, analysis: Dict[str, Any]):
        """Store DB2 analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO db2_analysis 
                (program_name, access_pattern, analysis_data)
                VALUES (?, ?, ?)
            """, (
                program_name,
                analysis.get('data_access_pattern', 'unknown'),
                json.dumps(analysis)
            ))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ DB2 analysis storage failed: {e}")
        finally:
            conn.close()
    
    async def _store_complete_lifecycle_analysis(self, program_name: str, analysis: Dict[str, Any]):
        """Store complete lifecycle analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO complete_lifecycle_analysis 
                (program_name, system_type, analysis_data)
                VALUES (?, ?, ?)
            """, (
                program_name,
                analysis.get('system_overview', {}).get('program_type', 'unknown'),
                json.dumps(analysis)
            ))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ Complete lifecycle analysis storage failed: {e}")
        finally:
            conn.close()
    
    async def _fallback_cics_analysis(self, program_name: str, cics_operations: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback CICS analysis when LLM fails"""
        return {
            'status': 'success',
            'program_name': program_name,
            'cics_lifecycle': {
                'transaction_type': 'online' if cics_operations else 'batch',
                'cics_operations_found': cics_operations,
                'analysis_method': 'fallback_regex',
                'note': 'Limited analysis due to LLM processing failure'
            }
        }
    
    async def _fallback_db2_analysis(self, program_name: str, sql_operations: List[str], tables: List[str]) -> Dict[str, Any]:
        """Fallback DB2 analysis when LLM fails"""
        return {
            'status': 'success',
            'program_name': program_name,
            'db2_lifecycle': {
                'data_access_pattern': 'read-write' if any(op in ['INSERT', 'UPDATE', 'DELETE'] for op in sql_operations) else 'read-only',
                'sql_operations_found': sql_operations,
                'tables_accessed': tables,
                'analysis_method': 'fallback_regex',
                'note': 'Limited analysis due to LLM processing failure'
            }
        }
    
    async def _analyze_field_lineage_for_program(self, program_name: str) -> Dict[str, Any]:
        """Get field lineage analysis for specific program"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT field_name, operation_type, operation_context, COUNT(*) as usage_count
                FROM field_lineage 
                WHERE program_name = ? OR program_name LIKE ?
                GROUP BY field_name, operation_type
            """, (program_name, f"%{program_name}%"))
            
            lineage_data = cursor.fetchall()
            
            field_summary = {}
            for field, operation, context, count in lineage_data:
                if field not in field_summary:
                    field_summary[field] = {
                        'operations': [],
                        'total_usage': 0,
                        'contexts': []
                    }
                
                field_summary[field]['operations'].append(operation)
                field_summary[field]['total_usage'] += count
                field_summary[field]['contexts'].append(context)
            
            return {
                'program_name': program_name,
                'fields_analyzed': len(field_summary),
                'field_details': field_summary
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Field lineage analysis failed: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    # =================== COMPLETE FILE LIFECYCLE TRACKING ===================
    
    async def analyze_complete_file_lifecycle(self, file_name: str) -> Dict[str, Any]:
        """Analyze complete file lifecycle from creation to purge including DB2 operations"""
        try:
            await self._init_llm_engine()
            
            # Get all references to this file across the entire system
            file_references = await self._find_all_file_references(file_name)
            
            if not file_references:
                return {
                    'status': 'not_found',
                    'file_name': file_name,
                    'message': 'No references found for this file'
                }
            
            # Analyze complete lifecycle stages
            lifecycle_stages = await self._analyze_file_lifecycle_stages(file_name, file_references)
            
            # Track data movement and transformations
            data_journey = await self._track_file_data_journey(file_name, file_references)
            
            # Analyze database integration
            db2_integration = await self._analyze_file_db2_integration(file_name, file_references)
            
            # Generate comprehensive lifecycle report
            lifecycle_report = await self._generate_complete_file_lifecycle_report(
                file_name, lifecycle_stages, data_journey, db2_integration, file_references
            )
            
            # Store complete lifecycle analysis
            await self._store_complete_file_lifecycle(file_name, {
                'lifecycle_stages': lifecycle_stages,
                'data_journey': data_journey,
                'db2_integration': db2_integration,
                'lifecycle_report': lifecycle_report
            })
            
            return {
                'status': 'success',
                'file_name': file_name,
                'complete_lifecycle': {
                    'lifecycle_stages': lifecycle_stages,
                    'data_journey': data_journey,
                    'db2_integration': db2_integration,
                    'lifecycle_report': lifecycle_report
                },
                'total_references': len(file_references),
                'analysis_timestamp': dt.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Complete file lifecycle analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _find_all_file_references(self, file_name: str) -> List[Dict[str, Any]]:
        """Find all references to a file across programs, JCL, and copybooks"""
        references = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Search in all program chunks for file references
            search_patterns = [
                f"%{file_name}%",
                f"%{file_name.upper()}%",
                f"%{file_name.lower()}%",
                f"%{file_name.replace('-', '_')}%",
                f"%{file_name.replace('_', '-')}%"
            ]
            
            for pattern in search_patterns:
                cursor.execute("""
                    SELECT program_name, chunk_id, chunk_type, content, metadata, file_type
                    FROM program_chunks
                    WHERE content LIKE ? OR metadata LIKE ?
                    ORDER BY program_name, chunk_id
                """, (pattern, pattern))
                
                results = cursor.fetchall()
                
                for program_name, chunk_id, chunk_type, content, metadata_str, file_type in results:
                    # Avoid duplicates
                    if any(ref.get('chunk_id') == chunk_id for ref in references):
                        continue
                    
                    metadata = self.safe_json_loads(metadata_str)
                    
                    # Analyze the specific file operation
                    file_operation = await self._analyze_file_operation_in_chunk(
                        file_name, content, chunk_type, program_name, metadata
                    )
                    
                    if file_operation:
                        references.append({
                            'program_name': program_name,
                            'chunk_id': chunk_id,
                            'chunk_type': chunk_type,
                            'file_type': file_type,
                            'content_snippet': content[:300] + "..." if len(content) > 300 else content,
                            'metadata': metadata,
                            'file_operation': file_operation
                        })
            
            # Search in file metadata table
            cursor.execute("""
                SELECT file_name, file_type, file_path, processing_status, total_lines
                FROM file_metadata
                WHERE file_name LIKE ? OR file_name LIKE ? OR file_name LIKE ?
            """, (f"%{file_name}%", f"%{file_name.upper()}%", f"%{file_name.lower()}%"))
            
            for file_name_found, file_type, file_path, status, total_lines in cursor.fetchall():
                references.append({
                    'reference_type': 'file_metadata',
                    'file_name': file_name_found,
                    'file_type': file_type,
                    'file_path': file_path,
                    'processing_status': status,
                    'total_lines': total_lines,
                    'file_operation': {
                        'operation_type': 'DEFINITION',
                        'description': 'File definition in metadata',
                        'confidence': 1.0
                    }
                })
            
        except Exception as e:
            logger.error(f"❌ Error finding file references: {e}")
        finally:
            conn.close()
        
        logger.info(f"Found {len(references)} references for file {file_name}")
        return references
    
    async def _analyze_file_operation_in_chunk(self, file_name: str, content: str, 
                                             chunk_type: str, program_name: str, 
                                             metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze specific file operation in a chunk using LLM"""
        try:
            await self._init_llm_engine()
            
            # Prepare content for analysis
            content_preview = content[:800] if len(content) > 800 else content
            
            file_operation_prompt = f"""
            Analyze how file "{file_name}" is used in this {chunk_type} from program {program_name}:
            
            Content:
            {content_preview}
            
            Metadata: {json.dumps(metadata, indent=2)[:200]}...
            
            Determine the file operation and return JSON:
            {{
                "operation_type": "CREATE/READ/WRITE/UPDATE/DELETE/COPY/ARCHIVE/PURGE/BACKUP/RESTORE",
                "operation_subtype": "specific type like OPEN_INPUT, WRITE_AFTER, etc.",
                "file_access_mode": "INPUT/OUTPUT/I-O/EXTEND",
                "business_purpose": "why this operation is performed",
                "data_transformation": "what happens to the data",
                "timing": "when this operation occurs (startup/runtime/shutdown/batch)",
                "frequency": "how often (daily/weekly/monthly/on-demand)",
                "dependency_files": ["other files this operation depends on"],
                "target_destinations": ["where data goes after this operation"],
                "error_handling": "how errors are handled",
                "volume_characteristics": "data volume handled",
                "retention_period": "how long data is kept",
                "compliance_requirements": "regulatory or business requirements",
                "performance_impact": "impact on system performance",
                "db2_integration": "if data moves to/from DB2 tables",
                "cics_integration": "if CICS transaction involved",
                "confidence": 0.9
            }}
            """
            
            sampling_params = SamplingParams(temperature=0.1, max_tokens=600)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(file_operation_prompt, sampling_params, request_id=request_id):
                response_text = result.outputs[0].text.strip()
                
                try:
                    if '{' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        operation_data = json.loads(response_text[json_start:json_end])
                        return operation_data
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse file operation analysis: {e}")
                break
            
            # Fallback analysis
            return await self._fallback_file_operation_analysis(file_name, content, chunk_type)
            
        except Exception as e:
            logger.warning(f"⚠️ File operation analysis failed: {e}")
            return await self._fallback_file_operation_analysis(file_name, content, chunk_type)
    
    async def _fallback_file_operation_analysis(self, file_name: str, content: str, chunk_type: str) -> Dict[str, Any]:
        """Fallback file operation analysis using regex"""
        content_upper = content.upper()
        
        # Determine operation type
        if 'OPEN OUTPUT' in content_upper or 'WRITE' in content_upper:
            operation_type = 'CREATE' if 'OPEN OUTPUT' in content_upper else 'WRITE'
        elif 'OPEN INPUT' in content_upper or 'READ' in content_upper:
            operation_type = 'READ'
        elif 'REWRITE' in content_upper:
            operation_type = 'UPDATE'
        elif 'DELETE' in content_upper:
            operation_type = 'DELETE'
        elif 'COPY' in content_upper:
            operation_type = 'COPY'
        else:
            operation_type = 'REFERENCE'
        
        return {
            'operation_type': operation_type,
            'operation_subtype': f"{operation_type.lower()}_operation",
            'file_access_mode': 'unknown',
            'business_purpose': f"File {operation_type.lower()} operation",
            'confidence': 0.6,
            'analysis_method': 'fallback_regex'
        }
    
    async def _store_complete_file_lifecycle(self, file_name: str, lifecycle_data: Dict[str, Any]):
        """Store complete file lifecycle analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO complete_file_lifecycle 
                (file_name, lifecycle_stages, data_journey, db2_integration, 
                 lifecycle_report, lifecycle_completeness, data_flow_complexity, integration_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_name,
                json.dumps(lifecycle_data.get('lifecycle_stages', {})),
                json.dumps(lifecycle_data.get('data_journey', {})),
                json.dumps(lifecycle_data.get('db2_integration', {})),
                lifecycle_data.get('lifecycle_report', ''),
                lifecycle_data.get('lifecycle_stages', {}).get('lifecycle_completeness', 0.0),
                lifecycle_data.get('data_journey', {}).get('data_flow_complexity', 'Unknown'),
                lifecycle_data.get('data_journey', {}).get('integration_score', 0.0)
            ))
            
            conn.commit()
            logger.info(f"✅ Stored complete file lifecycle for {file_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Complete file lifecycle storage failed: {e}")
        finally:
            conn.close()
    
    async def _store_cics_analysis(self, program_name: str, analysis: Dict[str, Any]):
        """Store CICS analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO cics_analysis 
                (program_name, transaction_type, analysis_data)
                VALUES (?, ?, ?)
            """, (
                program_name,
                analysis.get('transaction_type', 'unknown'),
                json.dumps(analysis)
            ))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ CICS analysis storage failed: {e}")
        finally:
            conn.close()
    
    async def _store_db2_analysis(self, program_name: str, analysis: Dict[str, Any]):
        """Store DB2 analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO db2_analysis 
                (program_name, access_pattern, analysis_data)
                VALUES (?, ?, ?)
            """, (
                program_name,
                analysis.get('data_access_pattern', 'unknown'),
                json.dumps(analysis)
            ))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ DB2 analysis storage failed: {e}")
        finally:
            conn.close()
    
    async def _store_complete_lifecycle_analysis(self, program_name: str, analysis: Dict[str, Any]):
        """Store complete lifecycle analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO complete_lifecycle_analysis 
                (program_name, system_type, analysis_data)
                VALUES (?, ?, ?)
            """, (
                program_name,
                analysis.get('system_overview', {}).get('program_type', 'unknown'),
                json.dumps(analysis)
            ))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ Complete lifecycle analysis storage failed: {e}")
        finally:
            conn.close()
    
    async def _fallback_cics_analysis(self, program_name: str, cics_operations: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback CICS analysis when LLM fails"""
        return {
            'status': 'success',
            'program_name': program_name,
            'cics_lifecycle': {
                'transaction_type': 'online' if cics_operations else 'batch',
                'cics_operations_found': cics_operations,
                'analysis_method': 'fallback_regex',
                'note': 'Limited analysis due to LLM processing failure'
            }
        }
    
    async def _fallback_db2_analysis(self, program_name: str, sql_operations: List[str], tables: List[str]) -> Dict[str, Any]:
        """Fallback DB2 analysis when LLM fails"""
        return {
            'status': 'success',
            'program_name': program_name,
            'db2_lifecycle': {
                'data_access_pattern': 'read-write' if any(op in ['INSERT', 'UPDATE', 'DELETE'] for op in sql_operations) else 'read-only',
                'sql_operations_found': sql_operations,
                'tables_accessed': tables,
                'analysis_method': 'fallback_regex',
                'note': 'Limited analysis due to LLM processing failure'
            }
        }
    
    async def _analyze_field_lineage_for_program(self, program_name: str) -> Dict[str, Any]:
        """Get field lineage analysis for specific program"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT field_name, operation_type, operation_context, COUNT(*) as usage_count
                FROM field_lineage 
                WHERE program_name = ? OR program_name LIKE ?
                GROUP BY field_name, operation_type
            """, (program_name, f"%{program_name}%"))
            
            lineage_data = cursor.fetchall()
            
            field_summary = {}
            for field, operation, context, count in lineage_data:
                if field not in field_summary:
                    field_summary[field] = {
                        'operations': [],
                        'total_usage': 0,
                        'contexts': []
                    }
                
                field_summary[field]['operations'].append(operation)
                field_summary[field]['total_usage'] += count
                field_summary[field]['contexts'].append(context)
            
            return {
                'program_name': program_name,
                'fields_analyzed': len(field_summary),
                'field_details': field_summary
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Field lineage analysis failed: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    async def analyze_complete_system_lifecycle(self, program_name: str) -> Dict[str, Any]:
        """Analyze complete system lifecycle including COBOL, CICS, and DB2"""
        try:
            await self._init_llm_engine()
            
            # Get all analysis components
            cobol_analysis = await self.get_component_analysis(program_name)
            cics_analysis = await self.analyze_cics_transaction_lifecycle(program_name)
            db2_analysis = await self.analyze_db2_data_lifecycle(program_name)
            field_lineage = await self._analyze_field_lineage_for_program(program_name)
            
            # Combine all analysis for comprehensive lifecycle
            combined_prompt = f"""
            Provide a complete system lifecycle analysis for mainframe program: {program_name}
            
            COBOL Analysis: {json.dumps(cobol_analysis, default=str)[:1000]}...
            CICS Analysis: {json.dumps(cics_analysis, default=str)[:1000]}...
            DB2 Analysis: {json.dumps(db2_analysis, default=str)[:1000]}...
            
            Generate comprehensive system lifecycle in JSON format:
            {{
                "system_overview": {{
                    "program_type": "online/batch/mixed",
                    "primary_function": "main business function",
                    "integration_level": "how integrated with other systems",
                    "criticality": "business criticality level"
                }},
                "complete_lifecycle": [
                    {{
                        "phase": "initialization",
                        "description": "System startup and initialization",
                        "components_involved": ["COBOL", "CICS", "DB2"],
                        "activities": ["activity1", "activity2"],
                        "duration": "estimated time",
                        "resources_required": ["resource1", "resource2"]
                    }}
                ],
                "data_journey": {{
                    "input_phase": {{
                        "sources": ["terminal", "files", "queues"],
                        "validation": "input validation process",
                        "transformation": "initial data transformation"
                    }},
                    "processing_phase": {{
                        "business_logic": "core business processing",
                        "calculations": "calculations performed",
                        "decisions": "decision points in processing"
                    }},
                    "persistence_phase": {{
                        "database_updates": "how data is saved",
                        "file_outputs": "file generation process",
                        "transaction_completion": "transaction finalization"
                    }},
                    "output_phase": {{
                        "user_interface": "user output generation",
                        "reports": "report generation",
                        "notifications": "notification mechanisms"
                    }}
                }},
                "integration_points": {{
                    "upstream_systems": ["system1", "system2"],
                    "downstream_systems": ["system3", "system4"],
                    "shared_resources": ["database", "files", "queues"],
                    "synchronization_points": ["point1", "point2"]
                }},
                "error_and_recovery": {{
                    "error_scenarios": ["scenario1", "scenario2"],
                    "recovery_procedures": ["procedure1", "procedure2"],
                    "rollback_mechanisms": ["mechanism1", "mechanism2"],
                    "user_notification": "how users are informed of issues"
                }},
                "performance_profile": {{
                    "response_time_targets": "expected response times",
                    "throughput_capacity": "transaction volume capacity",
                    "resource_utilization": "CPU, memory, I/O usage",
                    "bottleneck_points": ["potential bottleneck1", "bottleneck2"]
                }},
                "maintenance_lifecycle": {{
                    "deployment_process": "how changes are deployed",
                    "testing_requirements": "testing needed for changes",
                    "monitoring_points": "what needs to be monitored",
                    "backup_recovery": "backup and recovery procedures"
                }},
                "business_impact_analysis": {{
                    "business_processes_affected": ["process1", "process2"],
                    "user_communities": ["users1", "users2"],
                    "financial_impact": "financial implications",
                    "compliance_requirements": ["requirement1", "requirement2"]
                }},
                "optimization_opportunities": [
                    {{
                        "area": "performance",
                        "recommendation": "specific improvement",
                        "estimated_impact": "expected benefit",
                        "implementation_effort": "effort required"
                    }}
                ],
                "risk_assessment": {{
                    "technical_risks": ["risk1", "risk2"],
                    "business_risks": ["risk3", "risk4"],
                    "mitigation_strategies": ["strategy1", "strategy2"],
                    "contingency_plans": ["plan1", "plan2"]
                }}
            }}
            """
            
            sampling_params = SamplingParams(temperature=0.2, max_tokens=2000)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(combined_prompt, sampling_params, request_id=request_id):
                response_text = result.outputs[0].text.strip()
                
                try:
                    if '{' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        complete_lifecycle = json.loads(response_text[json_start:json_end])
                        
                        # Store complete analysis
                        await self._store_complete_lifecycle_analysis(program_name, complete_lifecycle)
                        
                        return {
                            'status': 'success',
                            'program_name': program_name,
                            'complete_lifecycle': complete_lifecycle,
                            'component_analyses': {
                                'cobol': cobol_analysis,
                                'cics': cics_analysis,
                                'db2': db2_analysis
                            },
                            'analysis_timestamp': dt.now().isoformat()
                        }
                except Exception as e:
                    logger.error(f"❌ Failed to parse complete lifecycle analysis: {e}")
                break
            
            # Return individual analyses if LLM synthesis fails
            return {
                'status': 'partial_success',
                'program_name': program_name,
                'component_analyses': {
                    'cobol': cobol_analysis,
                    'cics': cics_analysis,
                    'db2': db2_analysis
                },
                'note': 'Individual analyses completed, comprehensive synthesis failed'
            }
            
        except Exception as e:
            logger.error(f"❌ Complete system lifecycle analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
        
    # =================== CHAT INTERFACE SECTION ===================
    
    async def chat_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process chat query with context-aware responses"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            await self._init_llm_engine()
            
            # Analyze query intent
            query_intent = self._analyze_query_intent(query)
            
            # Get relevant context
            context = await self._get_query_context(query, query_intent)
            
            # Generate response based on intent
            if query_intent == 'component_analysis':
                response = await self._handle_component_analysis_query(query, context)
            elif query_intent == 'field_lineage':
                response = await self._handle_field_lineage_query(query, context)
            elif query_intent == 'file_lifecycle':
                response = await self._handle_file_lifecycle_query(query, context)
            elif query_intent == 'code_search':
                response = await self._handle_code_search_query(query, context)
            elif query_intent == 'cics_lifecycle':
                response = await self._handle_cics_lifecycle_query(query, context)
            elif query_intent == 'db2_lifecycle':
                response = await self._handle_db2_lifecycle_query(query, context)
            elif query_intent == 'complete_lifecycle':
                response = await self._handle_complete_lifecycle_query(query, context)
            else:
                response = await self._handle_general_query(query, context)
            
            # Store chat history
            await self._store_chat_history(session_id, query, response, context)
            
            return {
                'response': response,
                'intent': query_intent,
                'session_id': session_id,
                'context_used': len(str(context)) if context else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Chat query failed: {e}")
            return {
                'response': f"I encountered an error: {str(e)}",
                'intent': 'error',
                'session_id': session_id
            }
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze user query intent"""
        query_lower = query.lower()
        
        # CICS-specific intents
        if any(word in query_lower for word in ['cics', 'transaction', 'terminal', 'map', 'commarea']):
            return 'cics_lifecycle'
        
        # DB2-specific intents
        elif any(word in query_lower for word in ['db2', 'sql', 'database', 'table', 'query', 'cursor']):
            return 'db2_lifecycle'
        
        # Complete lifecycle
        elif any(word in query_lower for word in ['complete lifecycle', 'full lifecycle', 'system lifecycle', 'end to end']):
            return 'complete_lifecycle'
        
        # Component analysis intent
        elif any(word in query_lower for word in ['analyze', 'analysis', 'what does', 'explain']):
            return 'component_analysis'
        
        # Field lineage intent
        elif any(word in query_lower for word in ['lineage', 'trace', 'flow', 'where is', 'used by']):
            return 'field_lineage'
        
        # File lifecycle intent
        elif any(word in query_lower for word in ['lifecycle', 'created', 'updated', 'dependencies']):
            return 'file_lifecycle'
        
        # Code search intent
        elif any(word in query_lower for word in ['find', 'search', 'show me', 'locate']):
            return 'code_search'
        
        # Default to general
        else:
            return 'general'
    
    async def _get_query_context(self, query: str, intent: str) -> Dict[str, Any]:
        """Get relevant context for query"""
        context = {}
        
        try:
            # Extract component names from query
            potential_components = self._extract_component_names_from_query(query)
            
            if potential_components:
                component_name = potential_components[0]
                
                # Get component data based on intent
                if intent == 'component_analysis':
                    context = await self._get_component_analysis_context(component_name)
                elif intent == 'field_lineage':
                    context = await self._get_field_lineage_context(component_name)
                elif intent == 'file_lifecycle':
                    context = await self._get_file_lifecycle_context(component_name)
                elif intent == 'code_search':
                    context = await self._get_search_context(query)
                elif intent == 'cics_lifecycle':
                    context = await self._get_cics_context(component_name)
                elif intent == 'db2_lifecycle':
                    context = await self._get_db2_context(component_name)
                elif intent == 'complete_lifecycle':
                    context = await self._get_complete_lifecycle_context(component_name)
            
        except Exception as e:
            logger.warning(f"⚠️ Context retrieval failed: {e}")
        
        return context
    
    def _extract_component_names_from_query(self, query: str) -> List[str]:
        """Extract potential component names from query"""
        # Look for uppercase words (typical mainframe naming)
        uppercase_words = re.findall(r'\b[A-Z][A-Z0-9_-]{2,}\b', query)
        
        # Look for quoted components
        quoted_components = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        
        components = uppercase_words + quoted_components
        return list(set(components))
    
    async def _get_component_analysis_context(self, component_name: str) -> Dict[str, Any]:
        """Get component analysis context"""
        context = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get component chunks
            cursor.execute("""
                SELECT chunk_type, content, metadata 
                FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
                LIMIT 10
            """, (component_name, f"%{component_name}%"))
            
            chunks = cursor.fetchall()
            context['chunks'] = [
                {
                    'chunk_type': chunk[0],
                    'content': chunk[1][:200] + "..." if len(chunk[1]) > 200 else chunk[1],
                    'metadata': self.safe_json_loads(chunk[2])
                }
                for chunk in chunks
            ]
            
            # Get field lineage
            cursor.execute("""
                SELECT field_name, operation_type, COUNT(*) as usage_count
                FROM field_lineage 
                WHERE program_name = ? OR program_name LIKE ?
                GROUP BY field_name, operation_type
                LIMIT 20
            """, (component_name, f"%{component_name}%"))
            
            lineage = cursor.fetchall()
            context['field_usage'] = [
                {'field': row[0], 'operation': row[1], 'count': row[2]}
                for row in lineage
            ]
            
        except Exception as e:
            logger.warning(f"⚠️ Component context retrieval failed: {e}")
        finally:
            conn.close()
        
        return context
    
    async def _get_field_lineage_context(self, field_name: str) -> Dict[str, Any]:
        """Get field lineage context"""
        context = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT program_name, operation_type, operation_context, chunk_id
                FROM field_lineage 
                WHERE field_name = ? OR field_name LIKE ?
                ORDER BY created_timestamp DESC
                LIMIT 20
            """, (field_name, f"%{field_name}%"))
            
            lineage_data = cursor.fetchall()
            context['lineage'] = [
                {
                    'program': row[0],
                    'operation': row[1],
                    'context': row[2],
                    'chunk_id': row[3]
                }
                for row in lineage_data
            ]
            
        except Exception as e:
            logger.warning(f"⚠️ Field lineage context retrieval failed: {e}")
        finally:
            conn.close()
        
        return context
    
    async def _get_file_lifecycle_context(self, file_name: str) -> Dict[str, Any]:
        """Get file lifecycle context"""
        context = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT lifecycle_stage, program_name, operation_details, business_purpose
                FROM file_lifecycle 
                WHERE file_name = ? OR file_name LIKE ?
                ORDER BY created_timestamp DESC
                LIMIT 20
            """, (file_name, f"%{file_name}%"))
            
            lifecycle_data = cursor.fetchall()
            context['lifecycle'] = [
                {
                    'stage': row[0],
                    'program': row[1],
                    'details': row[2],
                    'business_purpose': row[3]
                }
                for row in lifecycle_data
            ]
            
        except Exception as e:
            logger.warning(f"⚠️ File lifecycle context retrieval failed: {e}")
        finally:
            conn.close()
        
        return context
    
    async def _get_search_context(self, query: str) -> Dict[str, Any]:
        """Get search context using vector similarity"""
        context = {}
        
        try:
            if self.vector_index.ntotal > 0:
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                
                if query_embedding is not None:
                    # Search similar chunks
                    k = min(5, self.vector_index.ntotal)
                    scores, indices = self.vector_index.search(
                        query_embedding.reshape(1, -1).astype('float32'), k
                    )
                    
                    # Get chunk data
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    search_results = []
                    for score, idx in zip(scores[0], indices[0]):
                        if idx != -1:
                            cursor.execute("""
                                SELECT pc.program_name, pc.chunk_type, pc.content, pc.metadata
                                FROM program_chunks pc
                                JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                                WHERE ve.faiss_id = ?
                            """, (int(idx),))
                            
                            result = cursor.fetchone()
                            if result:
                                search_results.append({
                                    'program': result[0],
                                    'chunk_type': result[1],
                                    'content': result[2][:200] + "..." if len(result[2]) > 200 else result[2],
                                    'metadata': self.safe_json_loads(result[3]),
                                    'similarity': float(score)
                                })
                    
                    context['search_results'] = search_results
                    conn.close()
                    
        except Exception as e:
            logger.warning(f"⚠️ Search context retrieval failed: {e}")
        
        return context
    
    async def _get_cics_context(self, component_name: str) -> Dict[str, Any]:
        """Get CICS-specific context"""
        context = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if CICS analysis table exists and get data
            cursor.execute("""
                SELECT transaction_type, analysis_data
                FROM cics_analysis 
                WHERE program_name = ? OR program_name LIKE ?
                ORDER BY created_timestamp DESC
                LIMIT 1
            """, (component_name, f"%{component_name}%"))
            
            result = cursor.fetchone()
            if result:
                context['cics_analysis'] = {
                    'transaction_type': result[0],
                    'analysis_data': self.safe_json_loads(result[1])
                }
            
            # Get CICS operations from metadata
            cursor.execute("""
                SELECT metadata FROM program_chunks 
                WHERE (program_name = ? OR program_name LIKE ?)
                AND metadata LIKE '%cics%'
            """, (component_name, f"%{component_name}%"))
            
            cics_operations = []
            for (metadata_str,) in cursor.fetchall():
                if metadata_str:
                    metadata = self.safe_json_loads(metadata_str)
                    if 'cics_operations' in metadata:
                        cics_operations.extend(metadata['cics_operations'])
            
            if cics_operations:
                context['cics_operations'] = list(set(cics_operations))
            
        except Exception as e:
            logger.warning(f"⚠️ CICS context retrieval failed: {e}")
        finally:
            conn.close()
        
        return context
    
    async def _get_db2_context(self, component_name: str) -> Dict[str, Any]:
        """Get DB2-specific context"""
        context = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if DB2 analysis table exists and get data
            cursor.execute("""
                SELECT access_pattern, analysis_data
                FROM db2_analysis 
                WHERE program_name = ? OR program_name LIKE ?
                ORDER BY created_timestamp DESC
                LIMIT 1
            """, (component_name, f"%{component_name}%"))
            
            result = cursor.fetchone()
            if result:
                context['db2_analysis'] = {
                    'access_pattern': result[0],
                    'analysis_data': self.safe_json_loads(result[1])
                }
            
            # Get DB2 operations from metadata
            cursor.execute("""
                SELECT metadata FROM program_chunks 
                WHERE (program_name = ? OR program_name LIKE ?)
                AND (metadata LIKE '%db2%' OR content LIKE '%EXEC SQL%')
            """, (component_name, f"%{component_name}%"))
            
            db2_operations = []
            tables_accessed = []
            
            for (metadata_str,) in cursor.fetchall():
                if metadata_str:
                    metadata = self.safe_json_loads(metadata_str)
                    if 'db2_operations' in metadata:
                        db2_operations.extend(metadata['db2_operations'])
                    if 'database_tables' in metadata:
                        tables_accessed.extend(metadata['database_tables'])
            
            if db2_operations:
                context['db2_operations'] = list(set(db2_operations))
            if tables_accessed:
                context['database_tables'] = list(set(tables_accessed))
            
        except Exception as e:
            logger.warning(f"⚠️ DB2 context retrieval failed: {e}")
        finally:
            conn.close()
        
        return context
    
    async def _get_complete_lifecycle_context(self, component_name: str) -> Dict[str, Any]:
        """Get complete lifecycle context"""
        context = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if complete analysis table exists and get data
            cursor.execute("""
                SELECT system_type, analysis_data
                FROM complete_lifecycle_analysis 
                WHERE program_name = ? OR program_name LIKE ?
                ORDER BY created_timestamp DESC
                LIMIT 1
            """, (component_name, f"%{component_name}%"))
            
            result = cursor.fetchone()
            if result:
                context['complete_lifecycle'] = {
                    'system_type': result[0],
                    'analysis_data': self.safe_json_loads(result[1])
                }
            
        except Exception as e:
            logger.warning(f"⚠️ Complete lifecycle context retrieval failed: {e}")
        finally:
            conn.close()
        
        return context
    
    async def _handle_component_analysis_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle component analysis query"""
        try:
            component_info = ""
            if 'chunks' in context and context['chunks']:
                chunk_types = list(set(chunk['chunk_type'] for chunk in context['chunks']))
                component_info = f"Component has {len(context['chunks'])} chunks of types: {', '.join(chunk_types)}"
                
                # Add field information
                if 'field_usage' in context and context['field_usage']:
                    fields = list(set(item['field'] for item in context['field_usage']))
                    component_info += f"\nFields involved: {', '.join(fields[:10])}"
            
            prompt = f"""
            Answer this user question about a mainframe component: {query}
            
            Component Information:
            {component_info}
            
            Provide a helpful, detailed response about the component's functionality and purpose.
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found information about the component but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_field_lineage_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle field lineage query"""
        try:
            lineage_info = ""
            if 'lineage' in context and context['lineage']:
                programs = list(set(item['program'] for item in context['lineage']))
                operations = list(set(item['operation'] for item in context['lineage']))
                lineage_info = f"Field is used in {len(programs)} programs with operations: {', '.join(operations)}"
            
            prompt = f"""
            Answer this user question about field lineage: {query}
            
            Lineage Information:
            {lineage_info}
            
            Provide details about how and where this field is used across the codebase.
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found lineage information but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_file_lifecycle_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle file lifecycle query"""
        try:
            lifecycle_info = ""
            if 'lifecycle' in context and context['lifecycle']:
                stages = list(set(item['stage'] for item in context['lifecycle']))
                lifecycle_info = f"File has lifecycle stages: {', '.join(stages)}"
                
                # Add business purpose if available
                business_purpose = next((item['business_purpose'] for item in context['lifecycle'] 
                                       if item['business_purpose']), None)
                if business_purpose:
                    lifecycle_info += f"\nBusiness Purpose: {business_purpose}"
            
            prompt = f"""
            Answer this user question about file lifecycle: {query}
            
            Lifecycle Information:
            {lifecycle_info}
            
            Provide details about the file's creation, usage, and dependencies.
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found lifecycle information but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_code_search_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle code search query"""
        try:
            search_info = ""
            if 'search_results' in context and context['search_results']:
                programs = list(set(result['program'] for result in context['search_results']))
                search_info = f"Found {len(context['search_results'])} relevant code sections in programs: {', '.join(programs)}"
            
            prompt = f"""
            Answer this user search query: {query}
            
            Search Results:
            {search_info}
            
            Summarize what was found and explain how it relates to the user's question.
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found search results but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_cics_lifecycle_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle CICS lifecycle queries"""
        try:
            cics_info = ""
            if 'cics_analysis' in context:
                analysis = context['cics_analysis']['analysis_data']
                transaction_type = analysis.get('transaction_type', 'unknown')
                resources = analysis.get('cics_resources_used', {})
                cics_info = f"CICS Transaction Type: {transaction_type}\n"
                cics_info += f"Resources Used: {', '.join([f'{k}: {v}' for k, v in resources.items() if v])}\n"
            
            if 'cics_operations' in context:
                cics_info += f"CICS Operations: {', '.join(context['cics_operations'])}\n"
            
            prompt = f"""
            Answer this CICS transaction lifecycle question: {query}
            
            CICS Information:
            {cics_info}
            
            Provide detailed information about CICS transaction flow, resource usage, and lifecycle stages.
            Include performance considerations and best practices.
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=400)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found CICS information but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_db2_lifecycle_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle DB2 lifecycle queries"""
        try:
            db2_info = ""
            if 'db2_analysis' in context:
                analysis = context['db2_analysis']['analysis_data']
                access_pattern = analysis.get('data_access_pattern', 'unknown')
                tables = analysis.get('table_operations', {})
                db2_info = f"DB2 Access Pattern: {access_pattern}\n"
                db2_info += f"Tables Accessed: {', '.join(tables.keys())}\n"
            
            if 'db2_operations' in context:
                db2_info += f"SQL Operations: {', '.join(context['db2_operations'])}\n"
            
            if 'database_tables' in context:
                db2_info += f"Database Tables: {', '.join(context['database_tables'])}\n"
            
            prompt = f"""
            Answer this DB2 data lifecycle question: {query}
            
            DB2 Information:
            {db2_info}
            
            Provide detailed information about data flow, SQL operations, transaction management, and performance.
            Include data integrity and optimization recommendations.
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=400)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found DB2 information but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_complete_lifecycle_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle complete system lifecycle queries"""
        try:
            lifecycle_info = ""
            if 'complete_lifecycle' in context:
                analysis = context['complete_lifecycle']['analysis_data']
                system_overview = analysis.get('system_overview', {})
                lifecycle_info = f"System Type: {system_overview.get('program_type', 'unknown')}\n"
                lifecycle_info += f"Primary Function: {system_overview.get('primary_function', 'not specified')}\n"
                lifecycle_info += f"Criticality: {system_overview.get('criticality', 'unknown')}\n"
            
            prompt = f"""
            Answer this complete system lifecycle question: {query}
            
            System Lifecycle Information:
            {lifecycle_info}
            
            Provide comprehensive information about the complete system lifecycle including:
            - COBOL program flow
            - CICS transaction processing
            - DB2 data operations
            - Integration points
            - Performance characteristics
            - Business impact
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=500)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I found system lifecycle information but couldn't generate a detailed response: {str(e)}"
    
    async def _handle_general_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle general query"""
        try:
            prompt = f"""
            You are an expert mainframe analysis assistant with deep knowledge of COBOL, CICS, and DB2.
            Answer this question: {query}
            
            Provide helpful information about:
            - COBOL program development and best practices
            - CICS transaction processing and lifecycle
            - DB2 database operations and optimization
            - Integration between COBOL, CICS, and DB2
            - Performance tuning and troubleshooting
            - Mainframe development lifecycle
            
            If you need specific component information, ask the user to specify the component name.
            """
            
            sampling_params = SamplingParams(temperature=0.4, max_tokens=350)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
                
        except Exception as e:
            return f"I can help with mainframe COBOL/CICS/DB2 questions. Could you please rephrase your question? Error: {str(e)}"
    
    async def _store_chat_history(self, session_id: str, query: str, response: str, context: Dict[str, Any]):
        """Store chat history in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO chat_history (session_id, user_query, system_response, context_data)
                VALUES (?, ?, ?, ?)
            """, (session_id, query, response, json.dumps(context)))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ Chat history storage failed: {e}")
        finally:
            conn.close()
    
    # =================== ANALYSIS AND REPORTING SECTION ===================
    
    async def get_component_analysis(self, component_name: str) -> Dict[str, Any]:
        """Get comprehensive component analysis"""
        try:
            # Check if component exists
            if component_name not in self.parsed_files:
                # Try to load from database
                analysis = await self._load_component_from_db(component_name)
                if not analysis:
                    return {'status': 'not_found', 'component': component_name}
            
            # Get detailed analysis
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get chunks
            cursor.execute("""
                SELECT chunk_type, content, metadata, line_start, line_end
                FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            chunks = cursor.fetchall()
            
            # Get field lineage
            cursor.execute("""
                SELECT field_name, operation_type, COUNT(*) as usage_count
                FROM field_lineage 
                WHERE program_name = ? OR program_name LIKE ?
                GROUP BY field_name, operation_type
            """, (component_name, f"%{component_name}%"))
            
            field_usage = cursor.fetchall()
            
            # Get lifecycle information
            cursor.execute("""
                SELECT lifecycle_stage, operation_details, business_purpose
                FROM file_lifecycle 
                WHERE file_name = ? OR file_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            lifecycle = cursor.fetchall()
            
            conn.close()
            
            # Analyze complexity and generate insights
            complexity_analysis = await self._analyze_component_complexity(chunks)
            dead_field_analysis = await self._analyze_dead_fields(field_usage)
            
            return {
                'status': 'success',
                'component_name': component_name,
                'total_chunks': len(chunks),
                'chunk_breakdown': self._get_chunk_breakdown(chunks),
                'field_usage_summary': self._get_field_usage_summary(field_usage),
                'lifecycle_summary': self._get_lifecycle_summary(lifecycle),
                'complexity_analysis': complexity_analysis,
                'dead_field_analysis': dead_field_analysis,
                'recommendations': await self._generate_component_recommendations(chunks, field_usage)
            }
            
        except Exception as e:
            logger.error(f"❌ Component analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _load_component_from_db(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Load component analysis from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT file_name, file_type, total_chunks
                FROM file_metadata 
                WHERE file_name = ? OR file_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            result = cursor.fetchone()
            if result:
                return {
                    'file_name': result[0],
                    'file_type': result[1],
                    'total_chunks': result[2]
                }
        except Exception as e:
            logger.warning(f"⚠️ Component loading failed: {e}")
        finally:
            conn.close()
        
        return None
    
    def _get_chunk_breakdown(self, chunks: List[tuple]) -> Dict[str, int]:
        """Get breakdown of chunk types"""
        breakdown = defaultdict(int)
        for chunk in chunks:
            chunk_type = chunk[0]
            breakdown[chunk_type] += 1
        return dict(breakdown)
    
    def _get_field_usage_summary(self, field_usage: List[tuple]) -> Dict[str, Any]:
        """Get field usage summary"""
        total_fields = len(set(row[0] for row in field_usage))
        operations = defaultdict(int)
        
        for field, operation, count in field_usage:
            operations[operation] += count
        
        return {
            'total_unique_fields': total_fields,
            'total_operations': sum(operations.values()),
            'operation_breakdown': dict(operations)
        }
    
    def _get_lifecycle_summary(self, lifecycle: List[tuple]) -> Dict[str, Any]:
        """Get lifecycle summary"""
        stages = defaultdict(int)
        business_purposes = []
        
        for stage, details, purpose in lifecycle:
            stages[stage] += 1
            if purpose and purpose not in business_purposes:
                business_purposes.append(purpose)
        
        return {
            'lifecycle_stages': dict(stages),
            'business_purposes': business_purposes
        }
    
    async def _analyze_component_complexity(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Analyze component complexity"""
        try:
            total_lines = sum(chunk[4] - chunk[3] + 1 for chunk in chunks if chunk[3] and chunk[4])
            total_chunks = len(chunks)
            
            # Analyze content complexity
            complexity_scores = []
            for chunk in chunks:
                content = chunk[1]
                metadata = self.safe_json_loads(chunk[2]) if chunk[2] else {}
                
                # Calculate complexity based on operations and nesting
                operations = metadata.get('operations', [])
                complexity = len(operations) + content.count('IF') + content.count('PERFORM')
                complexity_scores.append(complexity)
            
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
            
            return {
                'total_lines': total_lines,
                'total_chunks': total_chunks,
                'avg_lines_per_chunk': total_lines / total_chunks if total_chunks > 0 else 0,
                'avg_complexity_score': avg_complexity,
                'complexity_rating': self._get_complexity_rating(avg_complexity),
                'high_complexity_chunks': len([s for s in complexity_scores if s > 10])
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Complexity analysis failed: {e}")
            return {'error': str(e)}
    
    def _get_complexity_rating(self, score: float) -> str:
        """Get complexity rating based on score"""
        if score < 3:
            return "Low"
        elif score < 7:
            return "Medium"
        elif score < 12:
            return "High"
        else:
            return "Very High"
    
    async def _analyze_dead_fields(self, field_usage: List[tuple]) -> Dict[str, Any]:
        """Analyze potentially dead/unused fields"""
        field_operations = defaultdict(list)
        
        for field, operation, count in field_usage:
            field_operations[field].append((operation, count))
        
        dead_fields = []
        read_only_fields = []
        active_fields = []
        
        for field, operations in field_operations.items():
            op_types = [op[0] for op in operations]
            total_usage = sum(op[1] for op in operations)
            
            if total_usage == 0:
                dead_fields.append(field)
            elif 'WRITE' not in op_types and 'UPDATE' not in op_types:
                if 'READ' in op_types or 'REFERENCE' in op_types:
                    read_only_fields.append(field)
                else:
                    dead_fields.append(field)
            else:
                active_fields.append(field)
        
        return {
            'total_fields_analyzed': len(field_operations),
            'dead_fields': dead_fields,
            'read_only_fields': read_only_fields,
            'active_fields': active_fields,
            'dead_field_count': len(dead_fields),
            'read_only_count': len(read_only_fields),
            'active_field_count': len(active_fields)
        }
    
    async def _generate_component_recommendations(self, chunks: List[tuple], 
                                                field_usage: List[tuple]) -> List[str]:
        """Generate recommendations for component improvement"""
        recommendations = []
        
        try:
            # Analyze chunk structure
            chunk_types = [chunk[0] for chunk in chunks]
            
            if chunk_types.count('procedure_division') > 5:
                recommendations.append("Consider breaking down large procedure divisions into smaller, focused modules")
            
            if 'working_storage' not in chunk_types:
                recommendations.append("No working storage section found - verify data definitions")
            
            # Analyze field usage
            field_ops = defaultdict(list)
            for field, operation, count in field_usage:
                field_ops[field].append(operation)
            
            unused_fields = [field for field, ops in field_ops.items() if 'READ' not in ops and 'REFERENCE' not in ops]
            if len(unused_fields) > 5:
                recommendations.append(f"Found {len(unused_fields)} potentially unused fields - consider cleanup")
            
            # Analyze complexity
            total_lines = sum(chunk[4] - chunk[3] + 1 for chunk in chunks if chunk[3] and chunk[4])
            if total_lines > 1000:
                recommendations.append("Large component detected - consider modularization for better maintainability")
            
            # Generic recommendations
            if not recommendations:
                recommendations.append("Component structure appears well-organized")
                recommendations.append("Consider adding comprehensive error handling if not present")
                recommendations.append("Ensure proper documentation for business logic sections")
            
        except Exception as e:
            logger.warning(f"⚠️ Recommendation generation failed: {e}")
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    async def _get_cached_file_analysis(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Get cached file analysis from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT file_type, total_chunks, processing_status
                FROM file_metadata 
                WHERE file_name = ?
            """, (file_name,))
            
            result = cursor.fetchone()
            if result and result[2] == 'completed':
                # Get chunks
                cursor.execute("""
                    SELECT chunk_type, content, metadata
                    FROM program_chunks 
                    WHERE program_name = ?
                """, (file_name.split('.')[0],))
                
                chunks = cursor.fetchall()
                
                return {
                    'status': 'success',
                    'file_name': file_name,
                    'file_type': result[0],
                    'total_chunks': result[1],
                    'chunks': [
                        {
                            'chunk_type': chunk[0],
                            'content': chunk[1],
                            'metadata': self.safe_json_loads(chunk[2])
                        }
                        for chunk in chunks
                    ],
                    'cached': True
                }
        except Exception as e:
            logger.warning(f"⚠️ Cached analysis retrieval failed: {e}")
        finally:
            conn.close()
        
        return None
    
    
    async def find_field_usage(self, field_name: str) -> Dict[str, Any]:
        """Find all usage of a specific field"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT program_name, operation_type, operation_context, chunk_id
                FROM field_lineage 
                WHERE field_name = ? OR field_name LIKE ?
                ORDER BY created_timestamp DESC
            """, (field_name, f"%{field_name}%"))
            
            usage_data = cursor.fetchall()
            
            # Organize by operation type
            usage_by_operation = defaultdict(list)
            programs_involved = set()
            
            for program, operation, context, chunk_id in usage_data:
                usage_by_operation[operation].append({
                    'program': program,
                    'context': context,
                    'chunk_id': chunk_id
                })
                programs_involved.add(program)
            
            return {
                'field_name': field_name,
                'total_references': len(usage_data),
                'programs_involved': list(programs_involved),
                'usage_by_operation': dict(usage_by_operation),
                'lifecycle_stage': self._determine_field_lifecycle_stage(
                    list(usage_by_operation.keys())
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Field usage search failed: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # File statistics
            cursor.execute("""
                SELECT file_type, COUNT(*) as count, 
                       SUM(total_chunks) as total_chunks
                FROM file_metadata 
                WHERE processing_status = 'completed'
                GROUP BY file_type
            """)
            file_stats = cursor.fetchall()
            
            # Field statistics
            cursor.execute("""
                SELECT COUNT(DISTINCT field_name) as unique_fields,
                       COUNT(*) as total_references
                FROM field_lineage
            """)
            field_stats = cursor.fetchone()
            
            # Chunk statistics
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count
                FROM program_chunks
                GROUP BY chunk_type
            """)
            chunk_stats = cursor.fetchall()
            
            # Vector index statistics
            cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
            vector_count = cursor.fetchone()[0]
            
            # CICS analysis statistics
            cursor.execute("SELECT COUNT(*) FROM cics_analysis")
            cics_count = cursor.fetchone()[0]
            
            # DB2 analysis statistics
            cursor.execute("SELECT COUNT(*) FROM db2_analysis")
            db2_count = cursor.fetchone()[0]
            
            # Complete lifecycle statistics
            cursor.execute("SELECT COUNT(*) FROM complete_lifecycle_analysis")
            lifecycle_count = cursor.fetchone()[0]
            
            return {
                'file_statistics': [
                    {'file_type': row[0], 'count': row[1], 'total_chunks': row[2]}
                    for row in file_stats
                ],
                'field_statistics': {
                    'unique_fields': field_stats[0] if field_stats else 0,
                    'total_references': field_stats[1] if field_stats else 0
                },
                'chunk_statistics': [
                    {'chunk_type': row[0], 'count': row[1]}
                    for row in chunk_stats
                ],
                'vector_statistics': {
                    'embeddings_created': vector_count,
                    'search_index_size': self.vector_index.ntotal
                },
                'enhanced_statistics': {
                    'cics_analyses': cics_count,
                    'db2_analyses': db2_count,
                    'complete_lifecycle_analyses': lifecycle_count
                },
                'system_status': 'operational'
            }
            
        except Exception as e:
            logger.error(f"❌ System overview failed: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    # =================== ENHANCED FILE LIFECYCLE METHODS ===================
    
    async def _analyze_file_lifecycle_stages(self, file_name: str, references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze complete file lifecycle stages"""
        lifecycle_stages = {
            'creation': [],
            'population': [],
            'active_usage': [],
            'updates': [],
            'archival': [],
            'purge': [],
            'backup_restore': [],
            'migration': [],
            'integration': []
        }
        
        # Categorize operations into lifecycle stages
        for ref in references:
            operation = ref.get('file_operation', {})
            operation_type = operation.get('operation_type', 'UNKNOWN')
            
            stage_mapping = {
                'CREATE': 'creation',
                'WRITE': 'population',
                'READ': 'active_usage',
                'UPDATE': 'updates',
                'REWRITE': 'updates',
                'DELETE': 'updates',
                'COPY': 'migration',
                'ARCHIVE': 'archival',
                'PURGE': 'purge',
                'BACKUP': 'backup_restore',
                'RESTORE': 'backup_restore'
            }
            
            stage = stage_mapping.get(operation_type, 'active_usage')
            
            lifecycle_stages[stage].append({
                'program': ref.get('program_name', 'Unknown'),
                'chunk_id': ref.get('chunk_id', 'Unknown'),
                'operation_details': operation,
                'timing': operation.get('timing', 'unknown'),
                'frequency': operation.get('frequency', 'unknown'),
                'business_purpose': operation.get('business_purpose', 'Unknown')
            })
        
        # Analyze lifecycle completeness and generate insights
        lifecycle_insights = await self._generate_lifecycle_insights(file_name, lifecycle_stages)
        
        return {
            'stages': lifecycle_stages,
            'insights': lifecycle_insights,
            'stage_summary': {stage: len(operations) for stage, operations in lifecycle_stages.items()},
            'lifecycle_completeness': self._calculate_lifecycle_completeness(lifecycle_stages)
        }
    
    async def _track_file_data_journey(self, file_name: str, references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track complete data journey of a file through the system"""
        data_journey = {
            'data_sources': [],
            'processing_stages': [],
            'data_destinations': [],
            'transformation_points': [],
            'quality_checkpoints': [],
            'integration_points': []
        }
        
        # Analyze each reference for data movement
        for ref in references:
            operation = ref.get('file_operation', {})
            
            # Track data sources
            if operation.get('operation_type') in ['CREATE', 'WRITE']:
                dependency_files = operation.get('dependency_files', [])
                for dep_file in dependency_files:
                    data_journey['data_sources'].append({
                        'source': dep_file,
                        'program': ref.get('program_name', 'Unknown'),
                        'transformation': operation.get('data_transformation', 'Unknown')
                    })
            
            # Track processing stages
            if operation.get('operation_type') in ['READ', 'UPDATE']:
                data_journey['processing_stages'].append({
                    'stage': f"{ref.get('program_name', 'Unknown')}.{ref.get('chunk_id', 'Unknown')}",
                    'operation': operation.get('operation_type', 'Unknown'),
                    'business_purpose': operation.get('business_purpose', 'Unknown'),
                    'timing': operation.get('timing', 'Unknown')
                })
            
            # Track data destinations
            target_destinations = operation.get('target_destinations', [])
            for destination in target_destinations:
                data_journey['data_destinations'].append({
                    'destination': destination,
                    'program': ref.get('program_name', 'Unknown'),
                    'purpose': operation.get('business_purpose', 'Unknown')
                })
            
            # Track transformation points
            if operation.get('data_transformation'):
                data_journey['transformation_points'].append({
                    'program': ref.get('program_name', 'Unknown'),
                    'transformation': operation.get('data_transformation', 'Unknown'),
                    'operation_type': operation.get('operation_type', 'Unknown')
                })
            
            # Track DB2 integration points
            if operation.get('db2_integration'):
                data_journey['integration_points'].append({
                    'integration_type': 'DB2',
                    'program': ref.get('program_name', 'Unknown'),
                    'details': operation.get('db2_integration', 'Unknown'),
                    'operation': operation.get('operation_type', 'Unknown')
                })
            
            # Track CICS integration points
            if operation.get('cics_integration'):
                data_journey['integration_points'].append({
                    'integration_type': 'CICS',
                    'program': ref.get('program_name', 'Unknown'),
                    'details': operation.get('cics_integration', 'Unknown'),
                    'operation': operation.get('operation_type', 'Unknown')
                })
        
        # Generate data journey insights
        journey_insights = await self._generate_data_journey_insights(file_name, data_journey)
        
        return {
            'journey_map': data_journey,
            'insights': journey_insights,
            'data_flow_complexity': self._calculate_data_flow_complexity(data_journey),
            'integration_score': self._calculate_integration_score(data_journey)
        }
    
    async def _analyze_file_db2_integration(self, file_name: str, references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze file integration with DB2 databases"""
        db2_integration = {
            'file_to_db2_operations': [],
            'db2_to_file_operations': [],
            'synchronization_points': [],
            'data_consistency_checks': [],
            'performance_considerations': [],
            'backup_recovery_strategy': []
        }
        
        # Analyze DB2 integration from references
        for ref in references:
            operation = ref.get('file_operation', {})
            
            if operation.get('db2_integration'):
                # Analyze the specific DB2 integration
                db2_details = await self._analyze_specific_db2_integration(
                    file_name, ref, operation
                )
                
                if 'file_to_db2' in operation.get('db2_integration', '').lower():
                    db2_integration['file_to_db2_operations'].append(db2_details)
                elif 'db2_to_file' in operation.get('db2_integration', '').lower():
                    db2_integration['db2_to_file_operations'].append(db2_details)
                else:
                    # Determine direction based on operation type
                    if operation.get('operation_type') in ['READ']:
                        db2_integration['db2_to_file_operations'].append(db2_details)
                    else:
                        db2_integration['file_to_db2_operations'].append(db2_details)
        
        # Generate DB2 integration insights
        integration_insights = await self._generate_db2_integration_insights(file_name, db2_integration)
        
        return {
            'integration_details': db2_integration,
            'insights': integration_insights,
            'integration_complexity': self._calculate_db2_integration_complexity(db2_integration),
            'data_synchronization_risk': self._assess_data_synchronization_risk(db2_integration)
        }
    
    async def _generate_complete_file_lifecycle_report(self, file_name: str, lifecycle_stages: Dict[str, Any],
                                                     data_journey: Dict[str, Any], db2_integration: Dict[str, Any],
                                                     references: List[Dict[str, Any]]) -> str:
        """Generate comprehensive file lifecycle report using LLM"""
        try:
            await self._init_llm_engine()
            
            # Prepare summary data for LLM
            report_data = {
                'file_name': file_name,
                'total_references': len(references),
                'lifecycle_stages': {stage: len(ops) for stage, ops in lifecycle_stages['stages'].items()},
                'data_sources': len(data_journey['journey_map']['data_sources']),
                'data_destinations': len(data_journey['journey_map']['data_destinations']),
                'db2_integrations': len(db2_integration['integration_details']['file_to_db2_operations']) + 
                                  len(db2_integration['integration_details']['db2_to_file_operations']),
                'lifecycle_completeness': lifecycle_stages['lifecycle_completeness']
            }
            
            lifecycle_report_prompt = f"""
            Generate a comprehensive file lifecycle report for: {file_name}
            
            Report Summary Data:
            {json.dumps(report_data, indent=2)}
            
            Lifecycle Stages Found: {list(lifecycle_stages['stages'].keys())}
            Data Integration Points: {len(data_journey['journey_map']['integration_points'])}
            
            Generate a detailed executive report covering:
            
            1. EXECUTIVE SUMMARY
            - File importance and business criticality
            - Overall lifecycle health assessment
            - Key findings and recommendations
            
            2. COMPLETE LIFECYCLE OVERVIEW
            - File creation and initialization process
            - Data population and validation stages
            - Active usage patterns and frequency
            - Update and maintenance procedures
            - Archival and retention policies
            - Purge and cleanup processes
            
            3. DATA JOURNEY ANALYSIS
            - Data sources and input mechanisms
            - Processing and transformation stages
            - Output destinations and consumers
            - Integration with external systems
            
            4. DB2 INTEGRATION ASSESSMENT
            - File-to-database synchronization
            - Database-to-file data flows
            - Data consistency and integrity measures
            - Performance optimization opportunities
            
            5. BUSINESS IMPACT ANALYSIS
            - Business processes dependent on this file
            - Impact of file unavailability
            - Compliance and regulatory considerations
            - Data governance requirements
            
            6. RISK ASSESSMENT
            - Data loss risks and mitigation
            - Performance bottlenecks
            - Integration failure scenarios
            - Recovery time objectives
            
            7. OPTIMIZATION RECOMMENDATIONS
            - Performance improvement opportunities
            - Process automation possibilities
            - Cost reduction strategies
            - Technology modernization options
            
            8. OPERATIONAL PROCEDURES
            - Monitoring and alerting requirements
            - Backup and recovery procedures
            - Disaster recovery considerations
            - Change management processes
            
            Format as a professional technical report (800 words maximum).
            """
            
            sampling_params = SamplingParams(temperature=0.2, max_tokens=1500)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(lifecycle_report_prompt, sampling_params, request_id=request_id):
                response_text = result.outputs[0].text.strip()
                return response_text
            
        except Exception as e:
            logger.warning(f"⚠️ Lifecycle report generation failed: {e}")
            return f"""
            COMPLETE FILE LIFECYCLE REPORT - {file_name}
            
            EXECUTIVE SUMMARY:
            File {file_name} has been analyzed across {len(references)} references in the system.
            Lifecycle completeness: {lifecycle_stages['lifecycle_completeness']:.2f}
            
            LIFECYCLE STAGES IDENTIFIED:
            {', '.join([f"{stage}: {count}" for stage, count in report_data['lifecycle_stages'].items() if count > 0])}
            
            DATA INTEGRATION:
            - Data sources: {report_data['data_sources']}
            - Data destinations: {report_data['data_destinations']}
            - DB2 integrations: {report_data['db2_integrations']}
            
            RECOMMENDATIONS:
            Complete lifecycle analysis shows this file requires comprehensive monitoring and management.
            """
    
    def _calculate_lifecycle_completeness(self, lifecycle_stages: Dict[str, Any]) -> float:
        """Calculate lifecycle completeness score"""
        stages = lifecycle_stages['stages']
        stage_weights = {
            'creation': 0.2,
            'population': 0.15,
            'active_usage': 0.25,
            'updates': 0.15,
            'archival': 0.1,
            'purge': 0.05,
            'backup_restore': 0.05,
            'migration': 0.03,
            'integration': 0.02
        }
        
        score = 0.0
        for stage, weight in stage_weights.items():
            if stages.get(stage):
                score += weight
        
        return score
    
    def _calculate_data_flow_complexity(self, data_journey: Dict[str, Any]) -> str:
        """Calculate data flow complexity"""
        journey_map = data_journey['journey_map']
        
        total_points = (
            len(journey_map['data_sources']) +
            len(journey_map['processing_stages']) +
            len(journey_map['data_destinations']) +
            len(journey_map['transformation_points']) +
            len(journey_map['integration_points'])
        )
        
        if total_points <= 5:
            return "Low"
        elif total_points <= 15:
            return "Medium"
        elif total_points <= 30:
            return "High"
        else:
            return "Very High"
    
    def _calculate_integration_score(self, data_journey: Dict[str, Any]) -> float:
        """Calculate integration score based on external system connections"""
        integration_points = data_journey['journey_map']['integration_points']
        
        if not integration_points:
            return 0.0
        
        # Score based on integration types and complexity
        score = 0.0
        for integration in integration_points:
            if integration.get('integration_type') == 'DB2':
                score += 0.5
            elif integration.get('integration_type') == 'CICS':
                score += 0.3
            else:
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_db2_integration_complexity(self, db2_integration: Dict[str, Any]) -> str:
        """Calculate DB2 integration complexity"""
        integration_details = db2_integration['integration_details']
        
        total_operations = (
            len(integration_details['file_to_db2_operations']) +
            len(integration_details['db2_to_file_operations'])
        )
        
        if total_operations == 0:
            return "None"
        elif total_operations <= 2:
            return "Low"
        elif total_operations <= 5:
            return "Medium"
        else:
            return "High"
    
    def _assess_data_synchronization_risk(self, db2_integration: Dict[str, Any]) -> str:
        """Assess data synchronization risk"""
        integration_details = db2_integration['integration_details']
        
        # Check for bidirectional operations
        file_to_db2 = len(integration_details['file_to_db2_operations'])
        db2_to_file = len(integration_details['db2_to_file_operations'])
        
        if file_to_db2 > 0 and db2_to_file > 0:
            return "High - Bidirectional sync"
        elif file_to_db2 > 2 or db2_to_file > 2:
            return "Medium - Multiple operations"
        elif file_to_db2 > 0 or db2_to_file > 0:
            return "Low - Simple integration"
        else:
            return "None"
    
    async def _analyze_specific_db2_integration(self, file_name: str, reference: Dict[str, Any], 
                                              operation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific DB2 integration operation using LLM"""
        try:
            await self._init_llm_engine()
            
            content = reference.get('content_snippet', '')
            program_name = reference.get('program_name', 'Unknown')
            
            db2_integration_prompt = f"""
            Analyze the DB2 integration for file "{file_name}" in program {program_name}:
            
            Code Content:
            {content}
            
            Operation Details: {json.dumps(operation, indent=2)}
            
            Analyze the DB2 integration and return JSON:
            {{
                "integration_type": "ETL/Synchronization/Backup/Archive/Real-time",
                "data_direction": "file_to_db2/db2_to_file/bidirectional",
                "db2_tables_involved": ["table1", "table2"],
                "sql_operations": ["SELECT", "INSERT", "UPDATE"],
                "data_transformation_rules": ["rule1", "rule2"],
                "synchronization_frequency": "real-time/hourly/daily/weekly",
                "data_validation_checks": ["check1", "check2"],
                "error_handling_strategy": "how errors are handled",
                "performance_optimization": "optimization techniques used",
                "data_volume_estimate": "estimated data volume",
                "business_criticality": "high/medium/low",
                "compliance_requirements": ["requirement1", "requirement2"],
                "rollback_strategy": "how rollbacks are handled",
                "monitoring_requirements": ["monitoring1", "monitoring2"],
                "integration_dependencies": ["dependency1", "dependency2"]
            }}
            """
            
            sampling_params = SamplingParams(temperature=0.1, max_tokens=700)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(db2_integration_prompt, sampling_params, request_id=request_id):
                response_text = result.outputs[0].text.strip()
                
                try:
                    if '{' in response_text:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        integration_details = json.loads(response_text[json_start:json_end])
                        integration_details['program'] = program_name
                        integration_details['analysis_confidence'] = 0.9
                        return integration_details
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse DB2 integration analysis: {e}")
                break
            
        except Exception as e:
            logger.warning(f"⚠️ DB2 integration analysis failed: {e}")
        
        # Fallback analysis
        return {
            'integration_type': 'Unknown',
            'data_direction': 'unknown',
            'program': reference.get('program_name', 'Unknown'),
            'analysis_confidence': 0.5,
            'note': 'Limited analysis due to LLM processing failure'
        }
    
    async def _generate_lifecycle_insights(self, file_name: str, lifecycle_stages: Dict[str, Any]) -> str:
        """Generate lifecycle insights using LLM"""
        try:
            await self._init_llm_engine()
            
            stage_summary = {stage: len(ops) for stage, ops in lifecycle_stages.items()}
            
            insights_prompt = f"""
            Generate lifecycle insights for file: {file_name}
            
            Lifecycle Stage Summary: {json.dumps(stage_summary, indent=2)}
            
            Provide key insights about:
            1. Lifecycle completeness and gaps
            2. Operational risks and concerns
            3. Optimization opportunities
            4. Compliance and governance issues
            5. Business impact assessment
            
            Format as concise bullet points (150 words max).
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(insights_prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Lifecycle insights generation failed: {e}")
            return f"File {file_name} lifecycle analysis: {sum(stage_summary.values())} total operations across {len([s for s in stage_summary.values() if s > 0])} lifecycle stages."
    
    async def _generate_data_journey_insights(self, file_name: str, data_journey: Dict[str, Any]) -> str:
        """Generate data journey insights using LLM"""
        try:
            await self._init_llm_engine()
            
            journey_map = data_journey['journey_map']
            journey_summary = {
                'data_sources': len(journey_map['data_sources']),
                'processing_stages': len(journey_map['processing_stages']),
                'data_destinations': len(journey_map['data_destinations']),
                'transformation_points': len(journey_map['transformation_points']),
                'integration_points': len(journey_map['integration_points'])
            }
            
            journey_insights_prompt = f"""
            Generate data journey insights for file: {file_name}
            
            Data Journey Summary: {json.dumps(journey_summary, indent=2)}
            
            Provide insights about:
            1. Data flow complexity and efficiency
            2. Integration points and dependencies
            3. Transformation quality and consistency
            4. Performance bottlenecks
            5. Data governance implications
            
            Format as concise analysis (150 words max).
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(journey_insights_prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Data journey insights generation failed: {e}")
            return f"Data journey for {file_name}: {sum(journey_summary.values())} total data flow points identified."
    
    async def _generate_db2_integration_insights(self, file_name: str, db2_integration: Dict[str, Any]) -> str:
        """Generate DB2 integration insights using LLM"""
        try:
            await self._init_llm_engine()
            
            integration_details = db2_integration['integration_details']
            integration_summary = {
                'file_to_db2_operations': len(integration_details['file_to_db2_operations']),
                'db2_to_file_operations': len(integration_details['db2_to_file_operations']),
                'integration_complexity': db2_integration.get('integration_complexity', 'Unknown'),
                'synchronization_risk': db2_integration.get('data_synchronization_risk', 'Unknown')
            }
            
            db2_insights_prompt = f"""
            Generate DB2 integration insights for file: {file_name}
            
            Integration Summary: {json.dumps(integration_summary, indent=2)}
            
            Provide insights about:
            1. Integration architecture and patterns
            2. Data synchronization challenges
            3. Performance and scalability considerations
            4. Data consistency and integrity risks
            5. Operational monitoring requirements
            
            Format as actionable insights (150 words max).
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=300)
            request_id = str(uuid.uuid4())
            
            async for result in self.llm_engine.generate(db2_insights_prompt, sampling_params, request_id=request_id):
                return result.outputs[0].text.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ DB2 integration insights generation failed: {e}")
            return f"DB2 integration for {file_name}: {sum([integration_summary['file_to_db2_operations'], integration_summary['db2_to_file_operations']])} integration operations identified."# =================== SEARCH AND DISCOVERY SECTION ===================
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search across all parsed code"""
        try:
            if self.vector_index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            if query_embedding is None:
                return []
            
            # Search FAISS index
            k = min(limit, self.vector_index.ntotal)
            scores, indices = self.vector_index.search(
                query_embedding.reshape(1, -1).astype('float32'), k
            )
            
            # Retrieve chunk data
            results = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    cursor.execute("""
                        SELECT pc.program_name, pc.chunk_id, pc.chunk_type, 
                               pc.content, pc.metadata, pc.file_type
                        FROM program_chunks pc
                        JOIN vector_embeddings ve ON pc.id = ve.chunk_id
                        WHERE ve.faiss_id = ?
                    """, (int(idx),))
                    
                    result = cursor.fetchone()
                    if result:
                         results.append({
                            'program_name': result[0],
                            'chunk_id': result[1],
                            'chunk_type': result[2],
                            'content': result[3],
                            'metadata': self.safe_json_loads(result[4]),
                            'file_type': result[5],
                            'similarity_score': float(score)
                        })
            
            conn.close()
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    

    async def _display_complete_lifecycle_analysis(self, analysis: Dict[str, Any]):
        """Display complete system lifecycle analysis"""
        lifecycle_data = analysis.get('complete_lifecycle', {})
        
        # System Overview
        if 'system_overview' in lifecycle_data:
            st.subheader("🏢 System Overview")
            overview = lifecycle_data['system_overview']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Program Type", overview.get('program_type', 'N/A'))
            with col2:
                st.metric("Integration Level", overview.get('integration_level', 'N/A'))
            with col3:
                st.metric("Criticality", overview.get('criticality', 'N/A'))
            with col4:
                st.metric("Analysis Type", "Complete Lifecycle")
            
            if overview.get('primary_function'):
                st.write(f"**Primary Function:** {overview['primary_function']}")
        
        # Lifecycle Phases
        if 'complete_lifecycle' in lifecycle_data:
            st.subheader("📊 Lifecycle Phases")
            phases = lifecycle_data['complete_lifecycle']
            
            # Create timeline visualization
            if phases:
                phase_data = pd.DataFrame([
                    {
                        'Phase': phase.get('phase', 'Unknown'),
                        'Duration': phase.get('duration', 'Unknown'),
                        'Components': ', '.join(phase.get('components_involved', [])),
                        'Activities': len(phase.get('activities', []))
                    }
                    for phase in phases
                ])
                
                st.dataframe(phase_data, use_container_width=True)
    
    async def _display_cics_lifecycle_analysis(self, analysis: Dict[str, Any]):
        """Display CICS transaction lifecycle analysis"""
        cics_data = analysis.get('cics_lifecycle', {})
        
        # Transaction Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transaction Type", cics_data.get('transaction_type', 'N/A'))
        with col2:
            st.metric("CICS Operations", len(analysis.get('raw_operations', [])))
        with col3:
            st.metric("Analysis Type", "CICS Transaction")
        
        # Transaction Flow
        if 'transaction_flow' in cics_data:
            st.subheader("🔄 Transaction Flow")
            flow = cics_data['transaction_flow']
            
            flow_data = []
            for stage in flow:
                flow_data.append({
                    'Stage': stage.get('stage', 'Unknown'),
                    'Description': stage.get('description', 'N/A'),
                    'CICS Commands': ', '.join(stage.get('cics_commands', [])),
                    'Business Purpose': stage.get('business_purpose', 'N/A')
                })
            
            if flow_data:
                flow_df = pd.DataFrame(flow_data)
                st.dataframe(flow_df, use_container_width=True)
    
    async def _display_db2_lifecycle_analysis(self, analysis: Dict[str, Any]):
        """Display DB2 data lifecycle analysis"""
        db2_data = analysis.get('db2_lifecycle', {})
        
        # DB2 Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Access Pattern", db2_data.get('data_access_pattern', 'N/A'))
        with col2:
            st.metric("SQL Operations", len(analysis.get('sql_operations', [])))
        with col3:
            st.metric("Tables Accessed", len(analysis.get('tables_accessed', [])))
        
        # Database Lifecycle
        if 'database_lifecycle' in db2_data:
            st.subheader("🗄️ Database Lifecycle")
            lifecycle = db2_data['database_lifecycle']
            
            lifecycle_data = []
            for stage in lifecycle:
                lifecycle_data.append({
                    'Stage': stage.get('stage', 'Unknown'),
                    'Operations': ', '.join(stage.get('operations', [])),
                    'Tables': ', '.join(stage.get('tables_involved', [])),
                    'Business Purpose': stage.get('business_purpose', 'N/A'),
                    'Performance Impact': stage.get('performance_impact', 'N/A')
                })
            
            if lifecycle_data:
                lifecycle_df = pd.DataFrame(lifecycle_data)
                st.dataframe(lifecycle_df, use_container_width=True)
    
    async def _display_all_component_analyses(self, analysis: Dict[str, Any]):
        """Display all component analyses"""
        analyses = analysis.get('component_analyses', {})
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["COBOL Analysis", "CICS Analysis", "DB2 Analysis"])
        
        with tab1:
            cobol_analysis = analyses.get('cobol', {})
            if cobol_analysis.get('status') == 'success':
                st.subheader("📄 COBOL Component Analysis")
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Chunks", cobol_analysis.get('total_chunks', 0))
                with col2:
                    complexity = cobol_analysis.get('complexity_analysis', {})
                    st.metric("Complexity Rating", complexity.get('complexity_rating', 'N/A'))
                with col3:
                    field_summary = cobol_analysis.get('field_usage_summary', {})
                    st.metric("Unique Fields", field_summary.get('total_unique_fields', 0))
                with col4:
                    dead_fields = cobol_analysis.get('dead_field_analysis', {})
                    st.metric("Dead Fields", dead_fields.get('dead_field_count', 0))
                
                # Recommendations
                if cobol_analysis.get('recommendations'):
                    st.subheader("💡 COBOL Recommendations")
                    for i, recommendation in enumerate(cobol_analysis['recommendations'], 1):
                        st.write(f"{i}. {recommendation}")
            else:
                st.warning("⚠️ COBOL analysis not available")
        
        with tab2:
            cics_analysis = analyses.get('cics', {})
            if cics_analysis.get('status') == 'success':
                await self._display_cics_lifecycle_analysis(cics_analysis)
            else:
                st.warning("⚠️ CICS analysis not available or no CICS operations found")
        
        with tab3:
            db2_analysis = analyses.get('db2', {})
            if db2_analysis.get('status') == 'success':
                await self._display_db2_lifecycle_analysis(db2_analysis)
            else:
                st.warning("⚠️ DB2 analysis not available or no SQL operations found")
    
    async def field_lineage_page(self):
        """Field lineage analysis page"""
        st.header("🔗 Field Lineage Analysis")
        
        field_name = st.text_input("Enter field name to trace:")
        
        if field_name and st.button("Trace Field Lineage"):
            with st.spinner(f"Tracing lineage for {field_name}..."):
                lineage = await self.analyzer.find_field_usage(field_name)
            
            if 'error' not in lineage:
                st.success(f"✅ Lineage traced for {field_name}")
                
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total References", lineage['total_references'])
                with col2:
                    st.metric("Programs Involved", len(lineage['programs_involved']))
                with col3:
                    st.metric("Lifecycle Stage", lineage['lifecycle_stage'])
                
                # Programs involved
                st.subheader("📋 Programs Using This Field")
                programs_df = pd.DataFrame(lineage['programs_involved'], columns=['Program Name'])
                st.dataframe(programs_df, use_container_width=True)
                
                # Usage by operation
                if lineage.get('usage_by_operation'):
                    st.subheader("🔄 Operations Breakdown")
                    
                    operation_counts = {
                        op: len(usages) 
                        for op, usages in lineage['usage_by_operation'].items()
                    }
                    
                    fig = px.pie(
                        values=list(operation_counts.values()), 
                        names=list(operation_counts.keys()),
                        title="Field Operations Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error(f"❌ Field lineage failed: {lineage['error']}")
    
    async def code_search_page(self):
        """Code search page"""
        st.header("🔍 Semantic Code Search")
        
        search_query = st.text_input("Enter search query (natural language or code patterns):")
        search_limit = st.slider("Number of results", 1, 20, 10)
        
        if search_query and st.button("Search Code"):
            with st.spinner(f"Searching for: {search_query}..."):
                results = await self.analyzer.semantic_search(search_query, search_limit)
            
            if results:
                st.success(f"✅ Found {len(results)} relevant code sections")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"#{i} - {result['program_name']} ({result['chunk_type']}) - Similarity: {result['similarity_score']:.3f}"):
                        
                        # Show metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Program:** {result['program_name']}")
                        with col2:
                            st.write(f"**Type:** {result['chunk_type']}")
                        with col3:
                            st.write(f"**File Type:** {result['file_type']}")
                        
                        # Show content
                        st.code(result['content'], language='cobol')
            
            else:
                st.warning("⚠️ No relevant code sections found. Try different search terms.")
    
    async def chat_interface_page(self):
        """Chat interface page"""
        st.header("💬 Enhanced AI Chat Interface")
        st.markdown("**Ask about COBOL, CICS transactions, DB2 operations, and complete system lifecycles**")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_query = st.text_input("Ask me about your mainframe system:")
        
        if user_query and st.button("Send"):
            # Add user query to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Get AI response
            with st.spinner("Analyzing with enhanced CICS/DB2 support..."):
                response = await self.analyzer.chat_query(user_query)
            
            # Add AI response to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response['response'],
                "intent": response.get('intent', 'general'),
                "enhanced": True
            })
        
        # Display chat history
        st.subheader("💬 Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                # Enhanced intent emojis
                intent_emoji = {
                    'cics_lifecycle': '🔄',
                    'db2_lifecycle': '🗄️',
                    'complete_lifecycle': '🏢',
                    'component_analysis': '🔍',
                    'field_lineage': '🔗',
                    'code_search': '🔍',
                    'general': '💡'
                }.get(message.get('intent', 'general'), '💡')
                
                enhanced_badge = "🚀" if message.get('enhanced') else ""
                st.write(f"**Assistant {intent_emoji}{enhanced_badge}:** {message['content']}")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    async def system_overview_page(self):
        """System overview page"""
        st.header("📊 System Overview")
        
        with st.spinner("Loading system statistics..."):
            overview = await self.analyzer.get_system_overview()
        
        if 'error' not in overview:
            # File statistics
            st.subheader("📁 File Statistics")
            if overview.get('file_statistics'):
                file_df = pd.DataFrame(overview['file_statistics'])
                
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.pie(file_df, values='count', names='file_type', 
                                title="Files by Type")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.bar(file_df, x='file_type', y='total_chunks', 
                                title="Chunks by File Type")
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Field statistics
            st.subheader("🔗 Field Statistics")
            field_stats = overview.get('field_statistics', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Fields", field_stats.get('unique_fields', 0))
            with col2:
                st.metric("Total References", field_stats.get('total_references', 0))
            
            # System status
            st.subheader("⚙️ System Status")
            status = overview.get('system_status', 'unknown')
            if status == 'operational':
                st.success("✅ System is operational")
            else:
                st.warning(f"⚠️ System status: {status}")
        
        else:
            st.error(f"❌ Failed to load system overview: {overview['error']}")


# =================== UTILITY FUNCTIONS ===================

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('opulence_complete.log'),
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'torch', 'transformers', 'vllm', 'faiss-cpu', 'streamlit', 
        'plotly', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def create_sample_data():
    """Create sample COBOL/JCL files for testing"""
    sample_cobol = '''
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SAMPLE-PROG.
       
       ENVIRONMENT DIVISION.
       
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-CUSTOMER-RECORD.
           05  WS-CUSTOMER-ID      PIC 9(8).
           05  WS-CUSTOMER-NAME    PIC X(30).
           05  WS-ACCOUNT-BALANCE  PIC 9(10)V99.
       
       01  WS-COMMAREA             PIC X(100).
       01  WS-RESP                 PIC 9(8) COMP.
       
       PROCEDURE DIVISION.
       MAIN-LOGIC.
           PERFORM CICS-INITIALIZATION
           PERFORM DB2-OPERATIONS
           PERFORM BUSINESS-PROCESSING
           PERFORM CICS-TERMINATION
           STOP RUN.
       
       CICS-INITIALIZATION.
           EXEC CICS RECEIVE
               INTO(WS-COMMAREA)
               LENGTH(100)
               RESP(WS-RESP)
           END-EXEC.
           
           IF WS-RESP NOT = DFHRESP(NORMAL)
               PERFORM ERROR-HANDLING
           END-IF.
       
       DB2-OPERATIONS.
           EXEC SQL
               SELECT CUSTOMER_ID, CUSTOMER_NAME, ACCOUNT_BALANCE
               INTO :WS-CUSTOMER-ID, :WS-CUSTOMER-NAME, :WS-ACCOUNT-BALANCE
               FROM CUSTOMER_TABLE
               WHERE CUSTOMER_ID = :WS-CUSTOMER-ID
           END-EXEC.
           
           IF SQLCODE NOT = 0
               PERFORM SQL-ERROR-HANDLING
           END-IF.
           
           EXEC SQL
               UPDATE ACCOUNT_TABLE
               SET LAST_ACCESS = CURRENT_TIMESTAMP
               WHERE CUSTOMER_ID = :WS-CUSTOMER-ID
           END-EXEC.
       
       BUSINESS-PROCESSING.
           COMPUTE WS-ACCOUNT-BALANCE = WS-ACCOUNT-BALANCE * 1.02
           DISPLAY "Enhanced processing for: " WS-CUSTOMER-NAME.
       
       CICS-TERMINATION.
           EXEC CICS SEND
               FROM(WS-COMMAREA)
               LENGTH(100)
               RESP(WS-RESP)
           END-EXEC.
           
           EXEC CICS RETURN
           END-EXEC.
       
       ERROR-HANDLING.
           DISPLAY "CICS Error occurred: " WS-RESP.
           
       SQL-ERROR-HANDLING.
           DISPLAY "SQL Error occurred: " SQLCODE.
    '''
    
    sample_jcl = '''
//SAMPLEJOB JOB (ACCT),'SAMPLE JOB',CLASS=A,MSGCLASS=X
//STEP1    EXEC PGM=SAMPLE-PROG,REGION=8M
//STEPLIB  DD   DSN=CICS.LOADLIB,DISP=SHR
//         DD   DSN=DB2.LOADLIB,DISP=SHR
//SYSOUT   DD   SYSOUT=*
//CUSTFILE DD   DSN=PROD.CUSTOMER.DATA,DISP=SHR
//ACCTFILE DD   DSN=PROD.ACCOUNT.DATA,DISP=SHR
//OUTFILE  DD   DSN=PROD.OUTPUT,
//              DISP=(NEW,CATLG,DELETE),
//              SPACE=(TRK,(100,10))
//STEP2    EXEC PGM=DSNTEP2
//SYSTSIN  DD   *
  RUN PROGRAM(SAMPLE-PROG) PLAN(CUSTPLAN)
/*
//SYSPRINT DD   SYSOUT=*
//SYSTSPDB DD   DSN=DB2.CUSTOMER.DATABASE,DISP=SHR
    '''
    
    sample_copybook = '''
       01  ENHANCED-CUSTOMER-RECORD.
           05  CUST-ID             PIC 9(8).
           05  CUST-NAME           PIC X(30).
           05  CUST-ADDRESS.
               10  CUST-STREET     PIC X(25).
               10  CUST-CITY       PIC X(20).
               10  CUST-ZIP        PIC 9(5).
           05  CUST-PHONE          PIC X(12).
           05  CUST-BALANCE        PIC 9(10)V99.
           05  CUST-LAST-ACCESS    PIC X(26).
           05  CUST-TRANSACTION-TYPE PIC X(4).
           05  CUST-CICS-TERMINAL  PIC X(8).
           05  CUST-STATUS         PIC X(1).
               88  CUST-ACTIVE     VALUE 'A'.
               88  CUST-INACTIVE   VALUE 'I'.
               88  CUST-SUSPENDED  VALUE 'S'.
    '''
    
    # Create sample files
    Path("samples").mkdir(exist_ok=True)
    
    with open("samples/enhanced_sample.cbl", "w") as f:
        f.write(sample_cobol)
    
    with open("samples/enhanced_sample.jcl", "w") as f:
        f.write(sample_jcl)
    
    with open("samples/enhanced_customer.cpy", "w") as f:
        f.write(sample_copybook)
    
    print("✅ Enhanced sample files created in 'samples' directory")
    print("   - enhanced_sample.cbl (COBOL with CICS and DB2)")
    print("   - enhanced_sample.jcl (JCL with DB2 operations)")
    print("   - enhanced_customer.cpy (Enhanced copybook)")


# =================== CLI INTERFACE ===================

def cli_interface():
    """Command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Opulence Mainframe Analyzer CLI")
    parser.add_argument("--files", nargs="+", help="Files to process")
    parser.add_argument("--analyze", help="Component to analyze")
    parser.add_argument("--cics-lifecycle", help="Analyze CICS transaction lifecycle")
    parser.add_argument("--db2-lifecycle", help="Analyze DB2 data lifecycle") 
    parser.add_argument("--complete-lifecycle", help="Analyze complete system lifecycle")
    parser.add_argument("--field", help="Field to trace lineage")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--chat", help="Chat query with enhanced CICS/DB2 support")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--create-samples", action="store_true", help="Create sample files")
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_data()
        return
    
    # Initialize analyzer
    analyzer = OpulenceMainframeAnalyzer()
    
    async def process_cli():
        results = {}
        
        if args.files:
            print(f"Processing {len(args.files)} files with enhanced CICS/DB2 analysis...")
            for file_path in args.files:
                result = await analyzer.load_and_parse_file(file_path)
                results[file_path] = result
                print(f"✅ Processed {file_path}: {result['status']}")
        
        if args.analyze:
            print(f"Enhanced component analysis: {args.analyze}")
            result = await analyzer.get_component_analysis(args.analyze)
            results[f"analysis_{args.analyze}"] = result
        
        if args.cics_lifecycle:
            print(f"CICS transaction lifecycle analysis: {args.cics_lifecycle}")
            result = await analyzer.analyze_cics_transaction_lifecycle(args.cics_lifecycle)
            results[f"cics_lifecycle_{args.cics_lifecycle}"] = result
        
        if args.db2_lifecycle:
            print(f"DB2 data lifecycle analysis: {args.db2_lifecycle}")
            result = await analyzer.analyze_db2_data_lifecycle(args.db2_lifecycle)
            results[f"db2_lifecycle_{args.db2_lifecycle}"] = result
        
        if args.complete_lifecycle:
            print(f"Complete system lifecycle analysis: {args.complete_lifecycle}")
            result = await analyzer.analyze_complete_system_lifecycle(args.complete_lifecycle)
            results[f"complete_lifecycle_{args.complete_lifecycle}"] = result
        
        if args.field:
            print(f"Enhanced field lineage tracing: {args.field}")
            result = await analyzer.find_field_usage(args.field)
            results[f"lineage_{args.field}"] = result
        
        if args.search:
            print(f"Enhanced semantic search: {args.search}")
            result = await analyzer.semantic_search(args.search)
            results[f"search_{args.search}"] = result
        
        if args.chat:
            print(f"Enhanced chat query: {args.chat}")
            result = await analyzer.chat_query(args.chat)
            results[f"chat_{args.chat}"] = result
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✅ Enhanced results saved to {args.output}")
        else:
            print("\n📋 Enhanced Results:")
            print(json.dumps(results, indent=2, default=str))
    
    # Run enhanced async CLI
    asyncio.run(process_cli())


# =================== MAIN APPLICATION SECTION ===================

async def main():
    """Main application entry point"""
    
    # Initialize analyzer
    @st.cache_resource
    def get_analyzer():
        return OpulenceMainframeAnalyzer(
            model_name="codellama/CodeLlama-7b-Instruct-hf",
            embedding_model_path="./models/microsoft-codebert-base",
            gpu_id=0
        )
    
    analyzer = get_analyzer()
    
    # Initialize UI
    ui = OpulenceStreamlitUI(analyzer)
    
    # Run UI
    await ui.run()


# =================== ENTRY POINT ===================

if __name__ == "__main__":
    import sys
    
    # Setup logging
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Enhanced CLI mode
        cli_interface()
    else:
        # Enhanced Streamlit UI mode
        print("🚀 Starting Enhanced Opulence Mainframe Analyzer...")
        print("📋 Enhanced with CICS Transaction Lifecycle & DB2 Data Lifecycle Analysis")
        print("📋 Navigate to the Streamlit URL to access the enhanced web interface")
        
        # Run enhanced Streamlit app
        if 'streamlit' in sys.modules:
            asyncio.run(main())
        else:
            print("❌ Streamlit not available. Install with: pip install streamlit")
            print("Then run with: streamlit run opulence_complete_analyzer.py")


# =================== EXPORT CLASSES ===================

__all__ = [
    'OpulenceMainframeAnalyzer',
    'OpulenceStreamlitUI', 
    'CodeChunk',
    'FieldLineage',
    'FileLifecycle',
    'cli_interface',
    'create_sample_data'
]