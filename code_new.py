"""
Agent 1: Complete Enhanced Code Parser & Chunker - BUSINESS LOGIC FIXED VERSION
Handles COBOL, JCL, CICS, BMS, and Copybook parsing with intelligent chunking and proper business rules
"""

import re
import asyncio
import sqlite3
import json
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
from datetime import datetime as dt
import logging
from enum import Enum

import torch
from vllm import AsyncLLMEngine, SamplingParams

# Business Rule Enums and Classes
class COBOLDivision(Enum):
    IDENTIFICATION = 1
    ENVIRONMENT = 2
    DATA = 3
    PROCEDURE = 4

class DataItemType(Enum):
    GROUP = "01-49"
    RENAMES = "66"
    INDEPENDENT = "77"
    CONDITION = "88"

class TransactionState:
    def __init__(self):
        self.input_received = False
        self.map_loaded = False
        self.file_opened = {}
        self.error_handlers = {}
        
    def set_input_received(self):
        self.input_received = True
        
    def set_map_loaded(self, mapset, map_name):
        self.map_loaded = True
        self.current_map = (mapset, map_name)

@dataclass
class BusinessRuleViolation(Exception):
    rule: str
    context: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO

@dataclass
class CodeChunk:
    """Represents a parsed code chunk with business context"""
    program_name: str
    chunk_id: str
    chunk_type: str  # paragraph, perform, job_step, proc, sql_block, section, cics_command
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int
    business_context: Dict[str, Any] = None  # NEW: Business-specific context

@dataclass
class ControlFlowPath:
    """Represents a control flow execution path"""
    path_id: str
    entry_point: str
    exit_points: List[str]
    conditions: List[str]
    called_paragraphs: List[str]
    data_accessed: List[str]

class CompleteEnhancedCodeParserAgent:
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
                 gpu_id: int = None, coordinator=None):
        self._engine = None  # Cached engine reference (starts as None)
        
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        # Thread safety
        self._engine_lock = asyncio.Lock()
        self._engine_created = False
        self._using_coordinator_llm = False
        self._processed_files = set()  # Duplicate prevention
        self._engine_loaded = False
        self._using_shared_engine = False
        
        # Business Rule Validators
        self.business_validators = {
            'cobol': COBOLBusinessValidator(),
            'jcl': JCLBusinessValidator(),
            'cics': CICSBusinessValidator(),
            'bms': BMSBusinessValidator()
        }
        
        # ENHANCED COBOL PATTERNS with business context
        self.cobol_patterns = {
            # Basic identification with stricter boundaries
            'program_id': re.compile(r'^\s*PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'author': re.compile(r'^\s*AUTHOR\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'date_written': re.compile(r'^\s*DATE-WRITTEN\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            'date_compiled': re.compile(r'^\s*DATE-COMPILED\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            
            # Divisions with proper boundaries and order enforcement
            'identification_division': re.compile(r'^\s*IDENTIFICATION\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'environment_division': re.compile(r'^\s*ENVIRONMENT\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'data_division': re.compile(r'^\s*DATA\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'procedure_division': re.compile(r'^\s*PROCEDURE\s+DIVISION(?:\s+USING\s+([^\.]+))?\s*\.', re.IGNORECASE | re.MULTILINE),
            
            # Sections with proper hierarchy
            'working_storage': re.compile(r'^\s*WORKING-STORAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'file_section': re.compile(r'^\s*FILE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'linkage_section': re.compile(r'^\s*LINKAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'local_storage': re.compile(r'^\s*LOCAL-STORAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'^\s*([A-Z0-9][A-Z0-9-]*)\s+SECTION\s*\.\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Paragraphs with better boundary detection
            'paragraph': re.compile(r'^\s*([A-Z0-9][A-Z0-9-]*)\s*\.\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PERFORM patterns with business logic
            'perform_simple': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s*(?:\.|$)', re.IGNORECASE),
            'perform_until': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?UNTIL\s+(.*?)(?=\s+|$)', re.IGNORECASE | re.DOTALL),
            'perform_varying': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+(.*?)\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*)', re.MULTILINE | re.IGNORECASE),
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+IS\s+([X9AV\(\)S\+\-\.,/Z*]+)|PIC(?:TURE)?\s+([X9AV\(\)S\+\-\.,/Z*]+)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER)', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # File operations with enhanced context
            'file_control': re.compile(r'^\s*FILE-CONTROL\s*\.', re.IGNORECASE | re.MULTILINE),
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Enhanced SQL blocks with host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR', re.IGNORECASE),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Enhanced COPY statements
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+IN\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
        }
        
        # Enhanced JCL patterns with execution context
        self.jcl_patterns = {
            'job_card': re.compile(r'^//(\S+)\s+JOB\s+', re.MULTILINE),
            'job_step': re.compile(r'^//(\S+)\s+EXEC\s+', re.MULTILINE),
            'dd_statement': re.compile(r'^//(\S+)\s+DD\s+', re.MULTILINE),
            'proc_call': re.compile(r'EXEC\s+(\S+)', re.IGNORECASE),
            'dataset': re.compile(r'DSN=([^,\s]+)', re.IGNORECASE),
            'proc_definition': re.compile(r'^//(\S+)\s+PROC', re.MULTILINE),
            'pend_statement': re.compile(r'^//\s+PEND', re.MULTILINE),
            'set_statement': re.compile(r'^//\s+SET\s+([^=]+)=([^\s,]+)', re.MULTILINE),
            'if_statement': re.compile(r'^//\s+IF\s+(.*?)\s+THEN', re.MULTILINE),
            'endif_statement': re.compile(r'^//\s+ENDIF', re.MULTILINE),
            'cond_parameter': re.compile(r'COND=\(([^)]+)\)', re.IGNORECASE),
            'restart_parameter': re.compile(r'RESTART=([A-Z0-9]+)', re.IGNORECASE),
            'return_code_check': re.compile(r'\bRC\s*(=|EQ|NE|GT|LT|GE|LE)\s*(\d+)', re.IGNORECASE),
        }
        
        # Enhanced CICS patterns with transaction context
        self.cics_patterns = {
            # Terminal operations with parameter validation
            'cics_send_map': re.compile(r'EXEC\s+CICS\s+SEND\s+MAP\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive_map': re.compile(r'EXEC\s+CICS\s+RECEIVE\s+MAP\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_send_text': re.compile(r'EXEC\s+CICS\s+SEND\s+TEXT\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive': re.compile(r'EXEC\s+CICS\s+RECEIVE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # File operations with enhanced context
            'cics_read': re.compile(r'EXEC\s+CICS\s+READ\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_write': re.compile(r'EXEC\s+CICS\s+WRITE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_rewrite': re.compile(r'EXEC\s+CICS\s+REWRITE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_delete': re.compile(r'EXEC\s+CICS\s+DELETE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Program control with flow analysis
            'cics_link': re.compile(r'EXEC\s+CICS\s+LINK\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_xctl': re.compile(r'EXEC\s+CICS\s+XCTL\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_return': re.compile(r'EXEC\s+CICS\s+RETURN\s*(?:\((.*?)\))?\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Error handling with context tracking
            'cics_handle_condition': re.compile(r'EXEC\s+CICS\s+HANDLE\s+CONDITION\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_handle_aid': re.compile(r'EXEC\s+CICS\s+HANDLE\s+AID\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_resp': re.compile(r'RESP\(([A-Z][A-Z0-9-]*)\)', re.IGNORECASE),
            'cics_nohandle': re.compile(r'\bNOHANDLE\b', re.IGNORECASE),
        }
        
        # Enhanced BMS patterns
        self.bms_patterns = {
            'bms_mapset': re.compile(r'(\w+)\s+DFHMSD\s+(.*?)(?=\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_map': re.compile(r'(\w+)\s+DFHMDI\s+(.*?)(?=\w+\s+DFHMDI|\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_field': re.compile(r'(\w+)\s+DFHMDF\s+(.*?)(?=\w+\s+DFHMDF|\w+\s+DFHMDI|\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_mapset_end': re.compile(r'\s+DFHMSD\s+TYPE=FINAL', re.IGNORECASE),
            'bms_pos': re.compile(r'POS=\((\d+),(\d+)\)', re.IGNORECASE),
            'bms_length': re.compile(r'LENGTH=(\d+)', re.IGNORECASE),
        }
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database with enhanced schema including business rules"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced table with business context
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS program_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    business_context TEXT,
                    embedding_id TEXT,
                    file_hash TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    line_start INTEGER,
                    line_end INTEGER,
                    UNIQUE(program_name, chunk_id)
                )
            """)
            
            # Business rule violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS business_rule_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    line_number INTEGER,
                    context TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Control flow analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS control_flow_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    path_id TEXT NOT NULL,
                    entry_point TEXT,
                    exit_points TEXT,
                    conditions TEXT,
                    called_paragraphs TEXT,
                    data_accessed TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_name ON program_chunks(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON program_chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON program_chunks(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_business_rules ON business_rule_violations(program_name, rule_type)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")

    async def get_engine(self):
        """Get LLM engine with lazy loading and sharing - UNCHANGED"""
        if self._engine is None and self.coordinator:
            async with self._engine_lock:
                if self._engine is None:
                    try:
                        assigned_gpu = self.coordinator.agent_gpu_assignments.get("code_parser")
                        if assigned_gpu is not None:
                            self._engine = await self.coordinator.get_shared_llm_engine(assigned_gpu)
                            self.gpu_id = assigned_gpu
                            self._using_shared_engine = True
                            self._engine_loaded = True
                            self.logger.info(f"✅ CodeParser using shared engine on GPU {assigned_gpu}")
                        else:
                            raise ValueError("No GPU assigned for code_parser agent type")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to get shared engine: {e}")
                        raise
        
        return self._engine

    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM - UNCHANGED"""
        try:
            engine = await self.get_engine()
            if engine is None:
                self.logger.warning("No LLM engine available")
                return ""
            
            request_id = str(uuid.uuid4())
            
            try:
                result_generator = engine.generate(
                    prompt, sampling_params, request_id=request_id
                )
                
                async for result in result_generator:
                    if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                        return result.outputs[0].text.strip()
                    break
                    
            except TypeError as e:
                if "request_id" in str(e):
                    result_generator = engine.generate(prompt, sampling_params)
                    async for result in result_generator:
                        if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                            return result.outputs[0].text.strip()
                        break
                        
            return ""
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""

    def _generate_file_hash(self, content: str, file_path: Path) -> str:
        """Generate unique hash for file content and metadata - UNCHANGED"""
        hash_input = f"{file_path.name}:{file_path.stat().st_mtime}:{len(content)}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _is_duplicate_file(self, file_path: Path, content: str) -> bool:
        """Check if file has already been processed - UNCHANGED"""
        file_hash = self._generate_file_hash(content, file_path)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks 
                WHERE file_hash = ?
            """, (file_hash,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Duplicate check failed: {str(e)}")
            return False

    def _extract_program_name(self, content: str, file_path: Path) -> str:
        """Extract program name more robustly from content or filename - UNCHANGED"""
        try:
            program_match = self.cobol_patterns['program_id'].search(content)
            if program_match:
                return program_match.group(1).strip()
            
            job_match = self.jcl_patterns['job_card'].search(content)
            if job_match:
                return job_match.group(1).strip()
            
            if isinstance(file_path, str):
                file_path = Path(file_path)
            filename = file_path.name
            
            for ext in ['.cbl', '.cob', '.jcl', '.copy', '.cpy', '.bms']:
                if filename.lower().endswith(ext):
                    return filename[:-len(ext)]
            
            return file_path.stem
            
        except Exception as e:
            self.logger.error(f"Error extracting program name: {str(e)}")
            if isinstance(file_path, (str, Path)):
                return Path(file_path).stem or "UNKNOWN_PROGRAM"
            return file_path.stem or "UNKNOWN_PROGRAM"

    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results - UNCHANGED"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'enhanced_code_parser'
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_loaded_lazily'] = self._engine_loaded
        return result

    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Enhanced file type detection with proper business rule ordering"""
        content_upper = content.upper()
        
        # Order matters - check most specific patterns first
        
        # BMS detection (most specific)
        if self._is_bms_file(content_upper):
            return 'bms'
        
        # Heavy CICS detection (before general COBOL)
        if self._is_heavy_cics_program(content_upper):
            return 'cics'
        
        # JCL detection (specific format)
        if self._is_jcl_file(content, suffix):
            return 'jcl'
        
        # COBOL detection (most common, check after specific types)
        if self._is_cobol_program(content_upper):
            return 'cobol'
        
        # Copybook detection (specific content patterns)
        if self._is_copybook(content_upper, suffix):
            return 'copybook'
        
        # Extension-based fallback
        suffix_lower = suffix.lower()
        if suffix_lower in ['.cbl', '.cob']:
            return 'cobol'
        elif suffix_lower == '.jcl':
            return 'jcl'
        elif suffix_lower in ['.cpy', '.copy']:
            return 'copybook'
        elif suffix_lower == '.bms':
            return 'bms'
        
        return 'unknown'

    def _is_bms_file(self, content_upper: str) -> bool:
        """Check if file is BMS mapset"""
        return any(marker in content_upper for marker in ['DFHMSD', 'DFHMDI', 'DFHMDF'])

    def _is_heavy_cics_program(self, content_upper: str) -> bool:
        """Check if file is CICS-heavy program"""
        cics_count = content_upper.count('EXEC CICS')
        total_lines = content_upper.count('\n') + 1
        # More than 10% of lines contain CICS commands
        return cics_count > max(5, total_lines * 0.1)

    def _is_jcl_file(self, content: str, suffix: str) -> bool:
        """Check if file is JCL"""
        if suffix.lower() == '.jcl':
            return True
        return (content.strip().startswith('//') and 
                any(marker in content.upper() for marker in ['JOB', 'EXEC', 'DD']))

    def _is_cobol_program(self, content_upper: str) -> bool:
        """Check if file is COBOL program"""
        return any(marker in content_upper for marker in [
            'IDENTIFICATION DIVISION', 'PROGRAM-ID', 'WORKING-STORAGE SECTION',
            'PROCEDURE DIVISION', 'DATA DIVISION'
        ])

    def _is_copybook(self, content_upper: str, suffix: str) -> bool:
        """Check if file is copybook"""
        if suffix.lower() in ['.cpy', '.copy']:
            return True
        # Copybooks often have data definitions without divisions
        has_data_items = 'PIC' in content_upper and ('01 ' in content_upper or '05 ' in content_upper)
        no_divisions = not any(div in content_upper for div in ['IDENTIFICATION DIVISION', 'PROCEDURE DIVISION'])
        return has_data_items and no_divisions and len(content_upper.split('\n')) < 500

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single code file with enhanced business rule validation"""
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            if not file_path.exists():
                return self._add_processing_info({
                    "status": "error",
                    "file_name": str(file_path.name),
                    "error": "File not found"
                })
            
            # Enhanced file reading with multiple encoding attempts
            content = await self._read_file_with_encoding(file_path)
            if content is None:
                return self._add_processing_info({
                    "status": "error",
                    "file_name": str(file_path.name),
                    "error": "Unable to decode file"
                })

            if not content.strip():
                return self._add_processing_info({
                    "status": "error",
                    "file_name": str(file_path.name),
                    "error": "File is empty"
                })

            # Check for duplicates
            if self._is_duplicate_file(file_path, content):
                return self._add_processing_info({
                    "status": "skipped",
                    "file_name": str(file_path.name),
                    "message": "File already processed (duplicate detected)"
                })
            
            file_type = self._detect_file_type(content, file_path.suffix)
            self.logger.info(f"Detected file type: {file_type}")
            
            # Business rule validation before parsing
            business_violations = []
            if file_type in self.business_validators:
                violations = await self.business_validators[file_type].validate_structure(content)
                business_violations.extend(violations)
            
            # Parse based on file type with business context
            if file_type == 'cobol':
                chunks = await self._parse_cobol_with_business_rules(content, str(file_path.name))
            elif file_type == 'jcl':
                chunks = await self._parse_jcl_with_business_rules(content, str(file_path.name))
            elif file_type == 'copybook':
                chunks = await self._parse_copybook_with_business_rules(content, str(file_path.name))
            elif file_type == 'bms':
                chunks = await self._parse_bms_with_business_rules(content, str(file_path.name))
            elif file_type == 'cics':
                chunks = await self._parse_cics_with_business_rules(content, str(file_path.name))
            else:
                chunks = await self._parse_generic(content, str(file_path.name))
            
            self.logger.info(f"Generated {len(chunks)} chunks")
            
            if not chunks:
                return self._add_processing_info({
                    "status": "warning",
                    "file_name": str(file_path.name),
                    "file_type": file_type,
                    "chunks_created": 0,
                    "message": "No chunks were created from this file"
                })
            
            # Store business violations
            if business_violations:
                await self._store_business_violations(business_violations, self._extract_program_name(content, file_path))
            
            # Add file hash to all chunks
            file_hash = self._generate_file_hash(content, file_path)
            for chunk in chunks:
                chunk.metadata['file_hash'] = file_hash
            
            # Store chunks with verification
            await self._store_chunks_enhanced(chunks, file_hash)
            
            # Verify chunks were stored
            stored_chunks = await self._verify_chunks_stored(self._extract_program_name(content, file_path))
            
            # Generate enhanced metadata with business context
            metadata = await self._generate_metadata_enhanced(chunks, file_type, business_violations)

            # Generate control flow analysis
            if file_type in ['cobol', 'cics']:
                control_flow = await self._analyze_control_flow(chunks)
                await self._store_control_flow_analysis(control_flow, self._extract_program_name(content, file_path))
            
            result = {
                "status": "success",
                "file_name": str(file_path.name),
                "file_type": file_type,
                "chunks_created": len(chunks),
                "chunks_verified": stored_chunks,
                "business_violations": len(business_violations),
                "metadata": metadata,
                "processing_timestamp": dt.now().isoformat(),
                "file_hash": file_hash
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {str(e)}")
            return self._add_processing_info({
                "status": "error",
                "file_name": file_path.name,
                "error": str(e)
            })

    async def _read_file_with_encoding(self, file_path: Path) -> Optional[str]:
        """Enhanced file reading with multiple encoding attempts"""
        encodings = ['utf-8', 'cp1252', 'latin1', 'ascii', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                self.logger.debug(f"Successfully read file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error reading file with {encoding}: {e}")
                continue
        
        return None

    async def _parse_cobol_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete COBOL parsing with business rule enforcement"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # Validate COBOL structure first
        structure_validation = await self._validate_cobol_structure(content, program_name)
        
        # Parse divisions with hierarchy validation
        division_chunks = await self._parse_cobol_divisions_with_validation(content, program_name)
        chunks.extend(division_chunks)
        
        # Parse sections with proper context
        section_chunks = await self._parse_cobol_sections_with_context(content, program_name)
        chunks.extend(section_chunks)
        
        # Parse data items with level validation
        data_chunks = await self._parse_data_items_with_business_rules(content, program_name)
        chunks.extend(data_chunks)
        
        # Parse procedure division with control flow analysis
        procedure_chunks = await self._parse_procedure_division_with_flow(content, program_name)
        chunks.extend(procedure_chunks)
        
        # Parse SQL blocks with host variable validation
        sql_chunks = await self._parse_sql_blocks_with_host_variables(content, program_name)
        chunks.extend(sql_chunks)
        
        # Parse CICS commands with transaction context
        cics_chunks = await self._parse_cics_with_transaction_context(content, program_name)
        chunks.extend(cics_chunks)
        
        return chunks

    async def _validate_cobol_structure(self, content: str, program_name: str) -> Dict[str, Any]:
        """Validate COBOL program structure according to business rules"""
        violations = []
        divisions_found = {}
        
        # Check for required divisions
        required_divisions = ['IDENTIFICATION', 'PROCEDURE']
        division_order = ['IDENTIFICATION', 'ENVIRONMENT', 'DATA', 'PROCEDURE']
        
        for division in division_order:
            pattern_name = f'{division.lower()}_division'
            if pattern_name in self.cobol_patterns:
                match = self.cobol_patterns[pattern_name].search(content)
                if match:
                    divisions_found[division] = {
                        'position': match.start(),
                        'line': content[:match.start()].count('\n') + 1
                    }
        
        # Validate required divisions
        for req_div in required_divisions:
            if req_div not in divisions_found:
                violations.append(BusinessRuleViolation(
                    rule=f"MISSING_REQUIRED_DIVISION",
                    context=f"Missing required {req_div} DIVISION",
                    severity="ERROR"
                ))
        
        # Validate division order
        found_order = [d for d in division_order if d in divisions_found]
        for i in range(len(found_order) - 1):
            current = found_order[i]
            next_div = found_order[i + 1]
            
            if divisions_found[current]['position'] > divisions_found[next_div]['position']:
                violations.append(BusinessRuleViolation(
                    rule="DIVISION_ORDER_VIOLATION",
                    context=f"{current} DIVISION appears after {next_div} DIVISION",
                    severity="ERROR"
                ))
        
        return {
            'divisions_found': divisions_found,
            'violations': violations,
            'structure_valid': len([v for v in violations if v.severity == "ERROR"]) == 0
        }

    async def _parse_cobol_divisions_with_validation(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL divisions with proper business validation"""
        chunks = []
        
        division_patterns = {
            'identification_division': self.cobol_patterns['identification_division'],
            'environment_division': self.cobol_patterns['environment_division'],
            'data_division': self.cobol_patterns['data_division'],
            'procedure_division': self.cobol_patterns['procedure_division']
        }
        
        division_positions = {}
        for div_name, pattern in division_patterns.items():
            match = pattern.search(content)
            if match:
                division_positions[div_name] = {
                    'start': match.start(),
                    'match': match
                }
        
        # Sort divisions by position
        sorted_divisions = sorted(division_positions.items(), key=lambda x: x[1]['start'])
        
        for i, (div_name, div_info) in enumerate(sorted_divisions):
            start_pos = div_info['start']
            
            # Find end position (next division or end of content)
            if i + 1 < len(sorted_divisions):
                end_pos = sorted_divisions[i + 1][1]['start']
            else:
                end_pos = len(content)
            
            div_content = content[start_pos:end_pos].strip()
            
            # Enhanced business context analysis
            business_context = await self._analyze_division_business_context(div_content, div_name)
            
            # LLM analysis for deeper insights
            metadata = await self._analyze_division_with_llm(div_content, div_name)
            metadata.update(business_context)
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{div_name.upper()}",
                chunk_type="division",
                content=div_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _analyze_division_business_context(self, content: str, division_name: str) -> Dict[str, Any]:
        """Analyze division-specific business context"""
        context = {
            'division_type': division_name,
            'business_purpose': '',
            'dependencies': [],
            'compliance_requirements': []
        }
        
        if 'identification' in division_name:
            context.update({
                'business_purpose': 'Program identification and documentation',
                'program_metadata': self._extract_identification_metadata(content),
                'compliance_requirements': ['PROGRAM-ID required', 'Documentation standards']
            })
        elif 'environment' in division_name:
            context.update({
                'business_purpose': 'System and file environment configuration',
                'file_assignments': self._extract_file_assignments(content),
                'system_dependencies': self._extract_system_dependencies(content)
            })
        elif 'data' in division_name:
            context.update({
                'business_purpose': 'Data structure definitions and storage allocation',
                'data_structures': self._extract_data_structures_summary(content),
                'memory_requirements': self._estimate_memory_requirements(content)
            })
        elif 'procedure' in division_name:
            context.update({
                'business_purpose': 'Business logic implementation and program flow',
                'entry_points': self._extract_entry_points(content),
                'business_functions': self._extract_business_functions(content)
            })
        
        return context

    async def _parse_data_items_with_business_rules(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse data items with proper level validation and business rules"""
        chunks = []
        
        # Find data sections
        data_sections = {
            'working_storage': self.cobol_patterns['working_storage'],
            'file_section': self.cobol_patterns['file_section'],
            'linkage_section': self.cobol_patterns['linkage_section'],
            'local_storage': self.cobol_patterns['local_storage']
        }
        
        for section_name, pattern in data_sections.items():
            section_match = pattern.search(content)
            if not section_match:
                continue
            
            # Extract section content
            section_start = section_match.end()
            section_end = self._find_section_end(content, section_start, list(data_sections.keys()) + ['procedure_division'])
            section_content = content[section_start:section_end]
            
            # Parse data items with business validation
            data_items = await self._parse_section_data_items(section_content, program_name, section_name)
            chunks.extend(data_items)
        
        return chunks

    async def _parse_section_data_items(self, section_content: str, program_name: str, section_name: str) -> List[CodeChunk]:
        """Parse data items within a section with level validation"""
        chunks = []
        level_stack = []  # Track level hierarchy
        
        data_matches = list(self.cobol_patterns['data_item'].finditer(section_content))
        
        for match in data_matches:
            level = int(match.group(1))
            name = match.group(2)
            definition = match.group(3)
            
            # Skip comment lines
            if match.group(0).strip().startswith('*'):
                continue
            
            # Validate data item according to business rules
            validation_result = self._validate_data_item_business_rules(level, name, definition, level_stack)
            
            if validation_result['valid']:
                # Update level stack
                level_stack = self._update_level_stack(level_stack, level, name)
                
                # Extract business metadata
                business_context = await self._analyze_data_item_business_context(level, name, definition, section_name)
                
                # Standard metadata
                metadata = {
                    'level': level,
                    'field_name': name,
                    'section': section_name,
                    'pic_clause': self._extract_pic_clause(definition),
                    'usage': self._extract_usage_clause(definition),
                    'value': self._extract_value_clause(definition),
                    'occurs': self._extract_occurs_info(definition),
                    'redefines': self._extract_redefines_info(definition),
                    'level_hierarchy': level_stack.copy(),
                    'data_type': self._determine_data_type_enhanced(definition),
                    'business_validation': validation_result
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_DATA_{name}_{level}",
                    chunk_type="data_item",
                    content=match.group(0),
                    metadata=metadata,
                    business_context=business_context,
                    line_start=section_content[:match.start()].count('\n'),
                    line_end=section_content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    def _validate_data_item_business_rules(self, level: int, name: str, definition: str, level_stack: List) -> Dict[str, Any]:
        """Validate data item according to COBOL business rules"""
        violations = []
        valid = True
        
        # Level number validation
        if level == 66:  # RENAMES
            if 'RENAMES' not in definition.upper():
                violations.append("Level 66 must have RENAMES clause")
                valid = False
        elif level == 77:  # Independent item
            if level_stack and any(item['level'] < 77 for item in level_stack):
                violations.append("Level 77 cannot be subordinate to other items")
                valid = False
        elif level == 88:  # Condition name
            if not level_stack:
                violations.append("Level 88 must be subordinate to another item")
                valid = False
            elif 'VALUE' not in definition.upper():
                violations.append("Level 88 must have VALUE clause")
                valid = False
        
        # Hierarchy validation
        if level_stack and 1 <= level <= 49:
            last_level = level_stack[-1]['level'] if level_stack else 0
            if level <= last_level and level != 1:
                # Check if this is a valid sibling or parent level
                valid_levels = self._get_valid_next_levels(level_stack)
                if level not in valid_levels:
                    violations.append(f"Invalid level hierarchy: {level} after {last_level}")
                    valid = False
        
        # PIC clause validation
        if 1 <= level <= 49 and level != 66 and level != 77:
            has_pic = self._extract_pic_clause(definition) is not None
            has_subordinates = False  # Would need to check following items
            
            if not has_pic and not has_subordinates and 'REDEFINES' not in definition.upper():
                # Elementary items must have PIC unless they're group items
                pass  # Would need more context to validate properly
        
        return {
            'valid': valid,
            'violations': violations,
            'level_type': self._classify_level_type(level)
        }

    def _update_level_stack(self, level_stack: List, current_level: int, current_name: str) -> List:
        """Update the level stack maintaining proper hierarchy"""
        # Remove levels that are not parents of current level
        while level_stack and level_stack[-1]['level'] >= current_level:
            level_stack.pop()
        
        # Add current level
        level_stack.append({
            'level': current_level,
            'name': current_name
        })
        
        return level_stack

    def _get_valid_next_levels(self, level_stack: List) -> List[int]:
        """Get valid level numbers that can follow the current hierarchy"""
        if not level_stack:
            return [1, 77]  # Can start with 01 or 77
        
        last_level = level_stack[-1]['level']
        
        if last_level == 77:
            return [1, 77]  # After 77, only 01 or another 77
        elif last_level == 88:
            # After 88, can have sibling 88 or return to parent levels
            parent_levels = [item['level'] for item in level_stack[:-1]]
            return [88] + parent_levels + [level for level in range(1, 49) if level > max(parent_levels) if parent_levels]
        else:
            # Normal hierarchy: can have subordinate, sibling, or parent levels
            valid = []
            # Subordinate levels
            valid.extend(range(last_level + 1, 50))
            # Sibling and parent levels
            valid.extend([item['level'] for item in level_stack])
            # 77 and 88 are always possible
            valid.extend([77, 88])
            return list(set(valid))

    def _classify_level_type(self, level: int) -> str:
        """Classify the type of level number"""
        if 1 <= level <= 49:
            return "group_or_elementary"
        elif level == 66:
            return "renames"
        elif level == 77:
            return "independent"
        elif level == 88:
            return "condition_name"
        else:
            return "invalid"

    async def _analyze_data_item_business_context(self, level: int, name: str, definition: str, section_name: str) -> Dict[str, Any]:
        """Analyze business context of data item"""
        context = {
            'data_category': self._categorize_data_item(name, definition),
            'business_domain': self._infer_business_domain(name),
            'usage_pattern': self._analyze_usage_pattern(definition),
            'validation_rules': self._extract_validation_rules(definition),
            'security_classification': self._classify_security_level(name, definition)
        }
        
        return context

    def _categorize_data_item(self, name: str, definition: str) -> str:
        """Categorize data item by business purpose"""
        name_upper = name.upper()
        def_upper = definition.upper()
        
        # Common business data patterns
        if any(pattern in name_upper for pattern in ['AMOUNT', 'AMT', 'TOTAL', 'SUM']):
            return 'financial'
        elif any(pattern in name_upper for pattern in ['DATE', 'TIME', 'TIMESTAMP']):
            return 'temporal'
        elif any(pattern in name_upper for pattern in ['NAME', 'ADDR', 'ADDRESS', 'PHONE']):
            return 'personal_data'
        elif any(pattern in name_upper for pattern in ['ID', 'KEY', 'NBR', 'NUMBER']):
            return 'identifier'
        elif any(pattern in name_upper for pattern in ['STATUS', 'FLAG', 'IND', 'INDICATOR']):
            return 'control'
        elif any(pattern in name_upper for pattern in ['CTR', 'COUNTER', 'CNT', 'COUNT']):
            return 'counter'
        elif 'FILLER' in name_upper:
            return 'filler'
        else:
            return 'business_data'

    async def _parse_procedure_division_with_flow(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse procedure division with control flow analysis"""
        chunks = []
        
        # Find procedure division
        proc_match = self.cobol_patterns['procedure_division'].search(content)
        if not proc_match:
            return chunks
        
        proc_start = proc_match.end()
        proc_content = content[proc_start:]
        
        # Parse paragraphs with enhanced context
        paragraph_chunks = await self._parse_paragraphs_with_business_context(proc_content, program_name, proc_start)
        chunks.extend(paragraph_chunks)
        
        # Parse PERFORM statements with flow analysis
        perform_chunks = await self._parse_perform_statements_with_flow(proc_content, program_name, proc_start)
        chunks.extend(perform_chunks)
        
        # Parse control structures with nesting analysis
        control_chunks = await self._parse_control_structures_with_nesting(proc_content, program_name, proc_start)
        chunks.extend(control_chunks)
        
        return chunks

    async def _parse_paragraphs_with_business_context(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse paragraphs with enhanced business context"""
        chunks = []
        
        # Find paragraphs (excluding sections)
        paragraph_matches = []
        for match in self.cobol_patterns['paragraph'].finditer(content):
            para_name = match.group(1)
            if not para_name.endswith('SECTION'):
                paragraph_matches.append(match)
        
        for i, match in enumerate(paragraph_matches):
            para_name = match.group(1)
            para_start = match.start()
            
            # Find paragraph end
            if i + 1 < len(paragraph_matches):
                para_end = paragraph_matches[i + 1].start()
            else:
                para_end = len(content)
            
            para_content = content[para_start:para_end].strip()
            
            # Enhanced business context analysis
            business_context = await self._analyze_paragraph_business_context(para_content, para_name)
            
            # LLM analysis for deeper insights
            metadata = await self._analyze_paragraph_with_llm(para_content)
            metadata.update({
                'paragraph_name': para_name,
                'business_function': business_context.get('business_function', 'Unknown'),
                'complexity_score': self._calculate_paragraph_complexity(para_content),
                'performance_indicators': self._analyze_performance_indicators(para_content)
            })
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{para_name}",
                chunk_type="paragraph",
                content=para_content,
                metadata=metadata,
                business_context=business_context,
                line_start=offset + para_start,
                line_end=offset + para_end
            )
            chunks.append(chunk)
        
        return chunks

    async def _analyze_paragraph_business_context(self, content: str, para_name: str) -> Dict[str, Any]:
        """Analyze business context of a paragraph"""
        context = {
            'business_function': self._infer_business_function(para_name, content),
            'data_operations': self._extract_data_operations(content),
            'file_operations': self._extract_file_operations_detailed(content),
            'control_flow': self._extract_control_flow_elements(content),
            'error_handling': self._extract_error_handling_detailed(content),
            'performance_impact': self._assess_performance_impact(content),
            'business_rules_implemented': self._identify_business_rules(content)
        }
        
        return context

    def _infer_business_function(self, para_name: str, content: str) -> str:
        """Infer business function from paragraph name and content"""
        name_upper = para_name.upper()
        content_upper = content.upper()
        
        # Common business function patterns
        if any(pattern in name_upper for pattern in ['INIT', 'INITIAL', 'START']):
            return 'initialization'
        elif any(pattern in name_upper for pattern in ['READ', 'GET', 'FETCH']):
            return 'data_retrieval'
        elif any(pattern in name_upper for pattern in ['WRITE', 'UPDATE', 'SAVE']):
            return 'data_modification'
        elif any(pattern in name_upper for pattern in ['CALC', 'COMPUTE', 'TOTAL']):
            return 'calculation'
        elif any(pattern in name_upper for pattern in ['VALID', 'CHECK', 'EDIT']):
            return 'validation'
        elif any(pattern in name_upper for pattern in ['PRINT', 'DISPLAY', 'SHOW']):
            return 'output'
        elif any(pattern in name_upper for pattern in ['END', 'EXIT', 'TERM']):
            return 'termination'
        elif any(pattern in name_upper for pattern in ['ERROR', 'EXCEPT']):
            return 'error_handling'
        else:
            # Analyze content for clues
            if 'PERFORM' in content_upper and 'UNTIL' in content_upper:
                return 'loop_processing'
            elif 'IF' in content_upper:
                return 'conditional_processing'
            elif 'MOVE' in content_upper:
                return 'data_movement'
            else:
                return 'business_logic'

    async def _parse_perform_statements_with_flow(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse PERFORM statements with enhanced flow analysis"""
        chunks = []
        
        # Enhanced PERFORM patterns with proper business logic
        perform_patterns = {
            'perform_simple': self.cobol_patterns['perform_simple'],
            'perform_until': self.cobol_patterns['perform_until'],
            'perform_varying': self.cobol_patterns['perform_varying'],
            'perform_thru': self.cobol_patterns['perform_thru'],
            'perform_times': self.cobol_patterns['perform_times'],
            'perform_inline': self.cobol_patterns['perform_inline']
        }
        
        for perform_type, pattern in perform_patterns.items():
            matches = pattern.finditer(content)
            
            for i, match in enumerate(matches):
                # Extract PERFORM statement content
                perform_content = self._extract_complete_perform_statement(content, match.start())
                
                # Analyze business context
                business_context = await self._analyze_perform_business_context(perform_content, perform_type)
                
                # Enhanced metadata with flow analysis
                metadata = {
                    'perform_type': perform_type,
                    'target_paragraph': self._extract_target_paragraph(match),
                    'loop_analysis': self._analyze_loop_characteristics(perform_content, perform_type),
                    'execution_pattern': self._determine_execution_pattern(perform_type, perform_content),
                    'performance_implications': self._assess_perform_performance(perform_content, perform_type)
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_PERFORM_{perform_type}_{i+1}",
                    chunk_type="perform_statement",
                    content=perform_content,
                    metadata=metadata,
                    business_context=business_context,
                    line_start=offset + match.start(),
                    line_end=offset + match.start() + len(perform_content)
                )
                chunks.append(chunk)
        
        return chunks

    async def _analyze_perform_business_context(self, content: str, perform_type: str) -> Dict[str, Any]:
        """Analyze business context of PERFORM statement"""
        context = {
            'execution_model': self._classify_execution_model(perform_type),
            'business_purpose': self._infer_perform_purpose(content, perform_type),
            'data_impact': self._analyze_perform_data_impact(content),
            'control_complexity': self._measure_control_complexity(content, perform_type),
            'maintainability_score': self._assess_perform_maintainability(content, perform_type)
        }
        
        return context

    def _classify_execution_model(self, perform_type: str) -> str:
        """Classify the execution model of PERFORM statement"""
        if 'simple' in perform_type:
            return 'single_execution'
        elif 'until' in perform_type:
            return 'conditional_loop'
        elif 'varying' in perform_type:
            return 'counted_loop'
        elif 'times' in perform_type:
            return 'fixed_iteration'
        elif 'thru' in perform_type:
            return 'range_execution'
        elif 'inline' in perform_type:
            return 'inline_block'
        else:
            return 'unknown'

    async def _parse_sql_blocks_with_host_variables(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse SQL blocks with proper host variable validation"""
        chunks = []
        
        sql_matches = self.cobol_patterns['sql_block'].finditer(content)
        
        for i, match in enumerate(sql_matches):
            sql_content = match.group(0)
            sql_inner = match.group(1).strip()
            
            # Extract and validate host variables
            host_variables = self._extract_host_variables(sql_inner)
            host_var_validation = await self._validate_host_variables(host_variables, content)
            
            # Comprehensive SQL analysis with business context
            business_context = await self._analyze_sql_business_context(sql_inner, host_variables)
            
            # Enhanced metadata with host variable context
            metadata = await self._analyze_sql_comprehensive(sql_inner)
            metadata.update({
                'host_variables': host_variables,
                'host_variable_validation': host_var_validation,
                'sql_complexity': self._calculate_sql_complexity(sql_inner),
                'performance_indicators': self._analyze_sql_performance(sql_inner)
            })
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_SQL_BLOCK_{i+1}",
                chunk_type="sql_block",
                content=sql_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _extract_host_variables(self, sql_content: str) -> List[Dict[str, str]]:
        """Extract host variables from SQL content"""
        host_vars = []
        
        host_var_matches = self.cobol_patterns['sql_host_var'].finditer(sql_content)
        
        for match in host_var_matches:
            var_name = match.group(1)
            var_context = self._determine_host_var_context(sql_content, match.start())
            
            host_vars.append({
                'name': var_name,
                'cobol_name': var_name.replace('-', '_'),  # SQL uses underscores
                'context': var_context,
                'position': match.start()
            })
        
        return host_vars

    async def _validate_host_variables(self, host_variables: List[Dict], cobol_content: str) -> Dict[str, Any]:
        """Validate that host variables exist in COBOL working storage"""
        validation_results = {
            'valid_variables': [],
            'missing_variables': [],
            'type_mismatches': [],
            'all_valid': True
        }
        
        # Extract COBOL data definitions for validation
        cobol_data_items = self._extract_cobol_data_definitions(cobol_content)
        
        for host_var in host_variables:
            cobol_name = host_var['cobol_name'].upper()
            
            if cobol_name in cobol_data_items:
                validation_results['valid_variables'].append({
                    'host_var': host_var['name'],
                    'cobol_definition': cobol_data_items[cobol_name]
                })
            else:
                validation_results['missing_variables'].append(host_var['name'])
                validation_results['all_valid'] = False
        
        return validation_results

    async def _parse_cics_with_transaction_context(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse CICS commands with proper transaction state tracking"""
        chunks = []
        transaction_state = TransactionState()
        
        # Parse CICS commands in sequence to maintain transaction context
        cics_commands = []
        
        for command_type, pattern in self.cics_patterns.items():
            matches = pattern.finditer(content)
            
            for match in matches:
                cics_commands.append({
                    'type': command_type,
                    'match': match,
                    'position': match.start(),
                    'content': match.group(0),
                    'params': match.group(1) if match.groups() else ""
                })
        
        # Sort commands by position to maintain execution order
        cics_commands.sort(key=lambda x: x['position'])
        
        # Process commands with transaction state validation
        for i, cmd in enumerate(cics_commands):
            # Validate command in transaction context
            context_validation = await self._validate_cics_transaction_context(cmd, transaction_state)
            
            # Update transaction state
            self._update_transaction_state(transaction_state, cmd)
            
            # Analyze business context
            business_context = await self._analyze_cics_business_context(cmd, transaction_state)
            
            # Enhanced metadata
            metadata = await self._analyze_cics_command_comprehensive(cmd['type'], cmd['params'], cmd['content'])
            metadata.update({
                'transaction_sequence': i + 1,
                'context_validation': context_validation,
                'transaction_state': {
                    'input_received': transaction_state.input_received,
                    'map_loaded': transaction_state.map_loaded,
                    'files_open': list(transaction_state.file_opened.keys())
                }
            })
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_CICS_{cmd['type']}_{i+1}",
                chunk_type="cics_command",
                content=cmd['content'],
                metadata=metadata,
                business_context=business_context,
                line_start=content[:cmd['position']].count('\n'),
                line_end=content[:cmd['position'] + len(cmd['content'])].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _validate_cics_transaction_context(self, cmd: Dict, transaction_state: TransactionState) -> Dict[str, Any]:
        """Validate CICS command in current transaction context"""
        violations = []
        warnings = []
        
        cmd_type = cmd['type']
        cmd_params = cmd['params']
        
        # Business rule validations
        if cmd_type.startswith('cics_send') and not transaction_state.input_received:
            warnings.append("SEND operation without prior RECEIVE - potential transaction flow issue")
        
        if cmd_type == 'cics_send_map':
            if not self._has_required_params(cmd_params, ['MAP', 'MAPSET']):
                violations.append("SEND MAP requires both MAP and MAPSET parameters")
        
        if cmd_type == 'cics_receive_map':
            if not self._has_required_params(cmd_params, ['MAP', 'MAPSET']):
                violations.append("RECEIVE MAP requires both MAP and MAPSET parameters")
        
        # File operation validations
        if cmd_type in ['cics_read', 'cics_write', 'cics_rewrite', 'cics_delete']:
            if not self._has_required_params(cmd_params, ['FILE']):
                violations.append(f"{cmd_type} requires FILE parameter")
        
        # Error handling validations
        has_resp = 'RESP(' in cmd_params.upper()
        has_nohandle = 'NOHANDLE' in cmd_params.upper()
        
        if not has_resp and not has_nohandle:
            warnings.append("No error handling specified (RESP or NOHANDLE)")
        
        if has_resp and has_nohandle:
            violations.append("RESP and NOHANDLE are mutually exclusive")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'error_handling_present': has_resp or has_nohandle
        }

    def _update_transaction_state(self, state: TransactionState, cmd: Dict):
        """Update transaction state based on CICS command"""
        cmd_type = cmd['type']
        cmd_params = cmd['params']
        
        if cmd_type == 'cics_receive' or cmd_type == 'cics_receive_map':
            state.set_input_received()
        
        if cmd_type == 'cics_send_map':
            mapset = self._extract_parameter_value(cmd_params, 'MAPSET')
            map_name = self._extract_parameter_value(cmd_params, 'MAP')
            if mapset and map_name:
                state.set_map_loaded(mapset, map_name)
        
        if cmd_type in ['cics_read', 'cics_write']:
            file_name = self._extract_parameter_value(cmd_params, 'FILE')
            if file_name:
                state.file_opened[file_name] = True

    # Business Validator Classes
    async def _parse_jcl_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse JCL with business rule validation"""
        chunks = []
        job_name = self._extract_program_name(content, Path(filename))
        
        # Validate JCL structure
        jcl_validation = await self._validate_jcl_structure(content, job_name)
        
        # Parse with execution flow analysis
        job_chunks = await self._parse_jcl_job_with_flow(content, job_name)
        chunks.extend(job_chunks)
        
        # Parse steps with dependency analysis
        step_chunks = await self._parse_jcl_steps_with_dependencies(content, job_name)
        chunks.extend(step_chunks)
        
        return chunks

    async def _validate_jcl_structure(self, content: str, job_name: str) -> Dict[str, Any]:
        """Validate JCL structure according to business rules"""
        violations = []
        
        # Check for required JOB card
        job_match = self.jcl_patterns['job_card'].search(content)
        if not job_match:
            violations.append(BusinessRuleViolation(
                rule="MISSING_JOB_CARD",
                context="JCL must start with JOB card",
                severity="ERROR"
            ))
        
        # Check for at least one EXEC step
        exec_matches = list(self.jcl_patterns['job_step'].finditer(content))
        if not exec_matches:
            violations.append(BusinessRuleViolation(
                rule="NO_EXEC_STEPS",
                context="JCL must have at least one EXEC statement",
                severity="ERROR"
            ))
        
        # Validate step dependencies and return code logic
        step_dependencies = self._analyze_jcl_step_dependencies(content)
        
        return {
            'structure_valid': len([v for v in violations if v.severity == "ERROR"]) == 0,
            'violations': violations,
            'step_dependencies': step_dependencies
        }

    async def _parse_jcl_steps_with_dependencies(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL steps with dependency and execution flow analysis"""
        chunks = []
        
        step_matches = list(self.jcl_patterns['job_step'].finditer(content))
        
        for i, match in enumerate(step_matches):
            step_name = match.group(1)
            
            # Extract complete step content
            step_content = self._extract_jcl_step_complete(content, match.start(), step_matches, i)
            
            # Analyze step dependencies and execution conditions
            business_context = await self._analyze_jcl_step_business_context(step_content, step_name, i)
            
            # Enhanced metadata with execution flow
            metadata = await self._analyze_jcl_step_comprehensive(step_content, step_name)
            metadata.update({
                'step_sequence': i + 1,
                'execution_dependencies': business_context.get('dependencies', []),
                'conditional_execution': business_context.get('conditional_logic', {}),
                'resource_requirements': business_context.get('resources', {})
            })
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_STEP_{step_name}",
                chunk_type="jcl_step",
                content=step_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(step_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    # Additional business context analysis methods
    def _analyze_jcl_step_dependencies(self, content: str) -> Dict[str, Any]:
        """Analyze JCL step execution dependencies"""
        dependencies = {
            'conditional_steps': [],
            'restart_points': [],
            'return_code_checks': []
        }
        
        # Find COND parameters
        cond_matches = self.jcl_patterns['cond_parameter'].finditer(content)
        for match in cond_matches:
            dependencies['conditional_steps'].append({
                'condition': match.group(1),
                'position': match.start()
            })
        
        # Find RESTART parameters
        restart_matches = self.jcl_patterns['restart_parameter'].finditer(content)
        for match in restart_matches:
            dependencies['restart_points'].append(match.group(1))
        
        return dependencies

    async def _store_business_violations(self, violations: List[BusinessRuleViolation], program_name: str):
        """Store business rule violations in database"""
        if not violations:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for violation in violations:
                cursor.execute("""
                    INSERT INTO business_rule_violations 
                    (program_name, rule_type, rule_name, severity, description, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    type(violation).__name__,
                    violation.rule,
                    violation.severity,
                    str(violation),
                    violation.context
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(violations)} business rule violations for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store business violations: {str(e)}")

    async def _analyze_control_flow(self, chunks: List[CodeChunk]) -> List[ControlFlowPath]:
        """Analyze control flow paths through the program"""
        control_paths = []
        
        # Extract paragraphs and PERFORM statements
        paragraphs = [c for c in chunks if c.chunk_type == "paragraph"]
        performs = [c for c in chunks if c.chunk_type == "perform_statement"]
        
        # Build control flow graph
        for paragraph in paragraphs:
            path = ControlFlowPath(
                path_id=f"{paragraph.program_name}_{paragraph.chunk_id}_PATH",
                entry_point=paragraph.metadata.get('paragraph_name', ''),
                exit_points=[],
                conditions=[],
                called_paragraphs=[],
                data_accessed=[]
            )
            
            # Find PERFORM statements that call this paragraph
            calling_performs = [p for p in performs 
                             if paragraph.metadata.get('paragraph_name', '') in p.content]
            
            # Analyze data access patterns
            path.data_accessed = paragraph.metadata.get('field_names', [])
            
            # Extract conditions from IF statements
            if_statements = re.findall(r'IF\s+(.*?)(?:\s+THEN|\s|$)', paragraph.content, re.IGNORECASE)
            path.conditions = if_statements
            
            control_paths.append(path)
        
        return control_paths

    async def _store_control_flow_analysis(self, control_paths: List[ControlFlowPath], program_name: str):
        """Store control flow analysis in database"""
        if not control_paths:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for path in control_paths:
                cursor.execute("""
                    INSERT INTO control_flow_paths 
                    (program_name, path_id, entry_point, exit_points, conditions, 
                     called_paragraphs, data_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    path.path_id,
                    path.entry_point,
                    json.dumps(path.exit_points),
                    json.dumps(path.conditions),
                    json.dumps(path.called_paragraphs),
                    json.dumps(path.data_accessed)
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(control_paths)} control flow paths for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store control flow analysis: {str(e)}")

    # Enhanced storage with business context
    async def _store_chunks_enhanced(self, chunks: List[CodeChunk], file_hash: str):
        """Store chunks with enhanced business context - KEEPING LLM CALLS UNCHANGED"""
        if not chunks:
            self.logger.warning("No chunks to store")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    stored_count = 0
                    
                    for chunk in chunks:
                        try:
                            program_name = str(chunk.program_name)
                            chunk_id = str(chunk.chunk_id)
                            chunk_type = str(chunk.chunk_type)
                            content = str(chunk.content)
                            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
                            business_context_json = json.dumps(chunk.business_context) if chunk.business_context else "{}"
                            embedding_id = hashlib.md5(content.encode()).hexdigest()
                            
                            cursor.execute("""
                                SELECT id FROM program_chunks 
                                WHERE program_name = ? AND chunk_id = ?
                            """, (program_name, chunk_id))
                            
                            existing = cursor.fetchone()
                            
                            if existing:
                                cursor.execute("""
                                    UPDATE program_chunks 
                                    SET content = ?, metadata = ?, business_context = ?, 
                                        file_hash = ?, line_start = ?, line_end = ?, 
                                        created_timestamp = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                """, (
                                    content, metadata_json, business_context_json,
                                    str(file_hash), int(chunk.line_start), 
                                    int(chunk.line_end), existing[0]
                                ))
                            else:
                                cursor.execute("""
                                    INSERT INTO program_chunks 
                                    (program_name, chunk_id, chunk_type, content, metadata, 
                                     business_context, embedding_id, file_hash, line_start, line_end)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    program_name, chunk_id, chunk_type, content, 
                                    metadata_json, business_context_json, embedding_id,
                                    str(file_hash), int(chunk.line_start), int(chunk.line_end)
                                ))
                            
                            stored_count += 1
                            
                        except sqlite3.Error as e:
                            self.logger.error(f"Failed to store chunk {chunk.chunk_id}: {str(e)}")
                            continue
                    
                    cursor.execute("COMMIT")
                    self.logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks")
                    
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    self.logger.error(f"Transaction failed, rolled back: {str(e)}")
                    raise e
                    
        except Exception as e:
            self.logger.error(f"Database operation failed: {str(e)}")
            raise e

    # ALL LLM ANALYSIS METHODS REMAIN UNCHANGED TO PRESERVE LLM CALLS
    async def _analyze_division_with_llm(self, content: str, division_name: str) -> Dict[str, Any]:
        """Analyze COBOL division with LLM - UNCHANGED"""
        prompt = f"""
        Analyze this COBOL {division_name}:
        
        {content[:800]}...
        
        Extract key information:
        1. Main purpose and functionality
        2. Key elements defined
        3. Dependencies and relationships
        4. Configuration or setup details
        
        Return as JSON:
        {{
            "purpose": "main purpose",
            "key_elements": ["element1", "element2"],
            "dependencies": ["dep1", "dep2"],
            "configuration": "setup details"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Division analysis failed: {str(e)}")
        
        return {
            "purpose": f"{division_name} processing",
            "key_elements": [],
            "dependencies": [],
            "configuration": "Standard COBOL division"
        }

    async def _analyze_paragraph_with_llm(self, content: str) -> Dict[str, Any]:
        """Analyze paragraph with LLM - UNCHANGED"""
        prompt = f"""
        Analyze this COBOL paragraph:
        
        {content[:600]}...
        
        Extract:
        1. Field names referenced
        2. File operations performed
        3. Database operations (SQL statements)
        4. Called paragraphs (PERFORM statements)
        5. Main purpose/operation
        6. Error handling used
        
        Return as JSON:
        {{
            "field_names": ["field1", "field2"],
            "file_operations": ["READ FILE1", "WRITE FILE2"],
            "sql_operations": ["SELECT", "UPDATE"],
            "called_paragraphs": ["PARA1", "PARA2"],
            "main_purpose": "description",
            "error_handling": ["pattern1", "pattern2"]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=500)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Paragraph analysis failed: {str(e)}")
        
        return {
            "field_names": self._extract_field_names(content),
            "file_operations": self._extract_file_operations(content),
            "sql_operations": self._extract_sql_operations(content),
            "called_paragraphs": self._extract_perform_statements(content),
            "main_purpose": "Code processing",
            "error_handling": self._extract_error_handling_patterns(content)
        }

    async def _analyze_sql_comprehensive(self, sql_content: str) -> Dict[str, Any]:
        """Comprehensive SQL analysis - UNCHANGED"""
        prompt = f"""
        Perform comprehensive analysis of this SQL statement:
        
        {sql_content}
        
        Extract and analyze:
        1. SQL operation type and subtype
        2. All tables accessed with their roles
        3. All columns with operations
        4. Join conditions and types
        5. WHERE clause analysis
        6. Subqueries and their purposes
        7. Functions and expressions
        8. Performance considerations
        9. Business logic implemented
        
        Return as JSON:
        {{
            "operation_type": "SELECT|INSERT|UPDATE|DELETE|etc",
            "operation_subtype": "simple|complex|bulk|etc",
            "tables": [
                {{"name": "table1", "role": "source|target|lookup", "alias": "t1"}}
            ],
            "columns": [
                {{"name": "col1", "table": "table1", "operation": "select|update|filter"}}
            ],
            "joins": [
                {{"type": "INNER|LEFT|RIGHT", "condition": "condition", "tables": ["t1", "t2"]}}
            ],
            "conditions": [
                {{"type": "WHERE|HAVING", "expression": "expression", "purpose": "filter"}}
            ],
            "subqueries": [
                {{"type": "EXISTS|IN|SCALAR", "purpose": "purpose"}}
            ],
            "functions": ["func1", "func2"],
            "performance_notes": "performance analysis",
            "business_logic": "business rules"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=800)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"SQL analysis failed: {str(e)}")
        
        return {
            "operation_type": self._extract_sql_type(sql_content),
            "tables": [{"name": table, "role": "unknown", "alias": None} 
                      for table in self._extract_table_names(sql_content)],
            "columns": [],
            "joins": [],
            "conditions": [],
            "subqueries": [],
            "functions": [],
            "performance_notes": "Analysis not available",
            "business_logic": "SQL operation"
        }

    async def _analyze_cics_command_comprehensive(self, command_type: str, params: str, content: str) -> Dict[str, Any]:
        """Comprehensive CICS command analysis - UNCHANGED"""
        prompt = f"""
        Analyze this CICS command:
        
        Command Type: {command_type}
        Parameters: {params}
        Full Command: {content}
        
        Provide analysis:
        1. Command category and purpose
        2. Key parameters and their values
        3. Resource accessed (file, map, program, etc.)
        4. Transaction flow impact
        5. Error conditions handled
        6. Performance implications
        
        Return as JSON:
        {{
            "category": "terminal|file|program_control|storage|etc",
            "purpose": "purpose description",
            "key_parameters": {{"param1": "value1", "param2": "value2"}},
            "resource_accessed": "resource name",
            "flow_impact": "impact description",
            "error_conditions": ["cond1", "cond2"],
            "performance_impact": "impact analysis"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"CICS command analysis failed: {str(e)}")
        
        return {
            "category": self._categorize_cics_command(command_type),
            "purpose": f"{command_type} operation",
            "key_parameters": self._extract_cics_parameters(params),
            "resource_accessed": self._extract_cics_resource(params),
            "flow_impact": "CICS transaction processing",
            "error_conditions": [],
            "performance_impact": "Standard CICS overhead"
        }

    async def _analyze_jcl_step_comprehensive(self, content: str, step_name: str) -> Dict[str, Any]:
        """Comprehensive JCL step analysis - UNCHANGED"""
        prompt = f"""
        Analyze this JCL step: {step_name}
        
        {content}
        
        Extract comprehensive information:
        1. Program being executed
        2. Input datasets and their purposes
        3. Output datasets and their purposes
        4. Parameters and their meanings
        5. Conditional execution logic
        6. Step dependencies
        7. Resource requirements
        
        Return as JSON:
        {{
            "program": "program_name",
            "input_datasets": [
                {{"dsn": "dataset1", "purpose": "input data", "disposition": "SHR"}}
            ],
            "output_datasets": [
                {{"dsn": "dataset2", "purpose": "output", "disposition": "NEW"}}
            ],
            "parameters": {{"parm1": "value1"}},
            "conditional_logic": "condition description",
            "dependencies": ["step1", "step2"],
            "resources": {{"region": "size", "time": "limit"}}
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=600)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"JCL step analysis failed: {str(e)}")
        
        return {
            "program": self._extract_exec_program(content),
            "input_datasets": self._extract_input_datasets_detailed(content),
            "output_datasets": self._extract_output_datasets_detailed(content),
            "parameters": self._extract_parameters_detailed(content),
            "conditional_logic": "None identified",
            "dependencies": [],
            "resources": {}
        }

    # Enhanced metadata generation with business violations
    async def _generate_metadata_enhanced(self, chunks: List[CodeChunk], file_type: str, business_violations: List = None) -> Dict[str, Any]:
        """Generate enhanced metadata with business context"""
        metadata = {
            "total_chunks": len(chunks),
            "file_type": file_type,
            "chunk_types": {},
            "complexity_metrics": {},
            "business_violations": len(business_violations) if business_violations else 0,
            "processing_timestamp": dt.now().isoformat()
        }
        
        # Count chunk types
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            metadata["chunk_types"][chunk_type] = metadata["chunk_types"].get(chunk_type, 0) + 1
        
        # Enhanced complexity metrics with business context
        if file_type == 'cobol':
            metadata["complexity_metrics"] = {
                "total_paragraphs": metadata["chunk_types"].get("paragraph", 0),
                "total_sql_blocks": metadata["chunk_types"].get("sql_block", 0),
                "total_cics_commands": metadata["chunk_types"].get("cics_command", 0),
                "total_file_operations": metadata["chunk_types"].get("file_operation", 0),
                "total_data_items": metadata["chunk_types"].get("data_item", 0),
                "has_complex_logic": any(chunk.chunk_type in ["if_statement", "evaluate_statement", "perform_statement"] for chunk in chunks),
                "business_complexity_score": self._calculate_business_complexity_score(chunks),
                "maintainability_score": self._calculate_maintainability_score(chunks)
            }
        elif file_type == 'jcl':
            metadata["complexity_metrics"] = {
                "total_steps": metadata["chunk_types"].get("jcl_step", 0),
                "total_dd_statements": metadata["chunk_types"].get("jcl_dd_statement", 0),
                "has_procedures": metadata["chunk_types"].get("jcl_procedure", 0) > 0,
                "has_conditional_logic": metadata["chunk_types"].get("jcl_conditional", 0) > 0,
                "execution_complexity": self._calculate_jcl_execution_complexity(chunks)
            }
        
        # Business context aggregation
        business_functions = set()
        data_categories = set()
        performance_indicators = []
        
        for chunk in chunks:
            if chunk.business_context:
                if 'business_function' in chunk.business_context:
                    business_functions.add(chunk.business_context['business_function'])
                if 'data_category' in chunk.business_context:
                    data_categories.add(chunk.business_context['data_category'])
                if 'performance_impact' in chunk.business_context:
                    performance_indicators.append(chunk.business_context['performance_impact'])
        
        metadata.update({
            "business_functions": list(business_functions),
            "data_categories": list(data_categories),
            "performance_summary": self._summarize_performance_indicators(performance_indicators)
        })
        
        return metadata

    def _calculate_business_complexity_score(self, chunks: List[CodeChunk]) -> int:
        """Calculate business complexity score based on business context"""
        score = 0
        
        for chunk in chunks:
            # Base complexity from chunk type
            if chunk.chunk_type == "paragraph":
                score += 2
            elif chunk.chunk_type in ["if_statement", "evaluate_statement"]:
                score += 3
            elif chunk.chunk_type == "perform_statement":
                score += 2
            elif chunk.chunk_type == "sql_block":
                score += 4
            elif chunk.chunk_type == "cics_command":
                score += 2
            
            # Additional complexity from business context
            if chunk.business_context:
                if chunk.business_context.get('control_complexity', 0) > 5:
                    score += 2
                if chunk.business_context.get('business_function') == 'error_handling':
                    score += 1
                if 'financial' in chunk.business_context.get('data_category', ''):
                    score += 1
        
        return min(score, 100)

    # Helper methods for business analysis
    def _extract_identification_metadata(self, content: str) -> Dict[str, str]:
        """Extract identification division metadata"""
        metadata = {}
        
        patterns = {
            'author': self.cobol_patterns['author'],
            'date_written': self.cobol_patterns['date_written'],
            'date_compiled': self.cobol_patterns['date_compiled']
        }
        
        for key, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                metadata[key] = match.group(1).strip()
        
        return metadata

    def _extract_file_assignments(self, content: str) -> List[Dict[str, str]]:
        """Extract file assignments from environment division"""
        assignments = []
        
        select_matches = self.cobol_patterns['select_statement'].finditer(content)
        for match in select_matches:
            assignments.append({
                'logical_file': match.group(1),
                'physical_file': match.group(2)
            })
        
        return assignments

    def _extract_data_structures_summary(self, content: str) -> Dict[str, Any]:
        """Extract summary of data structures"""
        summary = {
            'record_layouts': 0,
            'table_structures': 0,
            'redefines_count': 0,
            'total_fields': 0
        }
        
        # Count 01 level items (records)
        record_pattern = re.compile(r'^\s*01\s+', re.MULTILINE | re.IGNORECASE)
        summary['record_layouts'] = len(record_pattern.findall(content))
        
        # Count OCCURS clauses (tables)
        occurs_matches = self.cobol_patterns['occurs_clause'].findall(content)
        summary['table_structures'] = len(occurs_matches)
        
        # Count REDEFINES
        redefines_matches = self.cobol_patterns['redefines'].findall(content)
        summary['redefines_count'] = len(redefines_matches)
        
        # Count total data items
        data_matches = self.cobol_patterns['data_item'].findall(content)
        summary['total_fields'] = len(data_matches)
        
        return summary

    def _extract_entry_points(self, content: str) -> List[str]:
        """Extract procedure division entry points"""
        entry_points = []
        
        # USING clause in PROCEDURE DIVISION
        proc_match = self.cobol_patterns['procedure_division'].search(content)
        if proc_match and proc_match.group(1):
            using_params = proc_match.group(1).strip().split()
            entry_points.extend(using_params)
        
        # First paragraph after PROCEDURE DIVISION
        paragraphs = self.cobol_patterns['paragraph'].findall(content)
        if paragraphs:
            entry_points.append(paragraphs[0])
        
        return entry_points

    def _extract_business_functions(self, content: str) -> List[str]:
        """Extract business functions from procedure division"""
        functions = set()
        
        # Analyze paragraph names for business functions
        paragraphs = self.cobol_patterns['paragraph'].findall(content)
        for para in paragraphs:
            function = self._infer_business_function(para, "")
            functions.add(function)
        
        return list(functions)

    def _find_section_end(self, content: str, start_pos: int, section_names: List[str]) -> int:
        """Find the end of a section"""
        end_pos = len(content)
        
        for section_name in section_names:
            if section_name in self.cobol_patterns:
                pattern = self.cobol_patterns[section_name]
                match = pattern.search(content, start_pos)
                if match and match.start() < end_pos:
                    end_pos = match.start()
        
        return end_pos

    def _extract_complete_perform_statement(self, content: str, start_pos: int) -> str:
        """Extract complete PERFORM statement handling all variations"""
        lines = content[start_pos:].split('\n')
        perform_lines = []
        
        for line in lines:
            perform_lines.append(line)
            
            # Check for end of PERFORM
            if 'END-PERFORM' in line.upper():
                break
            elif line.strip().endswith('.') and len(perform_lines) > 1:
                break
            elif (len(perform_lines) == 1 and 
                  not any(keyword in line.upper() for keyword in ['UNTIL', 'VARYING', 'TIMES', 'THRU', 'THROUGH'])):
                break
        
        return '\n'.join(perform_lines)

    def _extract_target_paragraph(self, match) -> str:
        """Extract target paragraph from PERFORM match"""
        content = match.group(0)
        
        # Simple PERFORM
        simple_match = re.search(r'PERFORM\s+([A-Z0-9][A-Z0-9-]*)', content, re.IGNORECASE)
        if simple_match:
            return simple_match.group(1)
        
        return "UNKNOWN"

    def _analyze_loop_characteristics(self, content: str, perform_type: str) -> Dict[str, Any]:
        """Analyze loop characteristics of PERFORM statement"""
        characteristics = {
            'is_loop': False,
            'loop_type': 'none',
            'complexity': 'low'
        }
        
        if 'until' in perform_type:
            characteristics.update({
                'is_loop': True,
                'loop_type': 'conditional',
                'complexity': 'medium'
            })
        elif 'varying' in perform_type:
            characteristics.update({
                'is_loop': True,
                'loop_type': 'iterative',
                'complexity': 'high'
            })
        elif 'times' in perform_type:
            characteristics.update({
                'is_loop': True,
                'loop_type': 'counted',
                'complexity': 'low'
            })
        
        return characteristics

    def _determine_execution_pattern(self, perform_type: str, content: str) -> str:
        """Determine execution pattern of PERFORM"""
        if 'inline' in perform_type:
            return 'inline_execution'
        elif 'thru' in perform_type:
            return 'range_execution'
        elif any(keyword in content.upper() for keyword in ['UNTIL', 'VARYING', 'TIMES']):
            return 'iterative_execution'
        else:
            return 'single_execution'

    def _assess_perform_performance(self, content: str, perform_type: str) -> Dict[str, str]:
        """Assess performance implications of PERFORM statement"""
        implications = {
            'cpu_impact': 'low',
            'memory_impact': 'low',
            'io_impact': 'unknown'
        }
        
        if 'varying' in perform_type or 'until' in perform_type:
            implications['cpu_impact'] = 'medium'
            
        if 'thru' in perform_type:
            implications['cpu_impact'] = 'high'  # Multiple paragraph execution
            
        # Check for I/O operations in content
        if any(op in content.upper() for op in ['READ', 'WRITE', 'EXEC SQL']):
            implications['io_impact'] = 'high'
        
        return implications

    def _measure_control_complexity(self, content: str, perform_type: str) -> int:
        """Measure control complexity of PERFORM statement"""
        complexity = 1  # Base complexity
        
        # Add complexity for loop types
        if 'until' in perform_type:
            complexity += 2
        elif 'varying' in perform_type:
            complexity += 3
        elif 'times' in perform_type:
            complexity += 1
        
        # Add complexity for nested control structures
        if_count = content.upper().count('IF ')
        evaluate_count = content.upper().count('EVALUATE ')
        nested_perform_count = content.upper().count('PERFORM ') - 1  # Exclude this PERFORM
        
        complexity += if_count + (evaluate_count * 2) + nested_perform_count
        
        return complexity

    def _assess_perform_maintainability(self, content: str, perform_type: str) -> int:
        """Assess maintainability score of PERFORM statement (1-10)"""
        score = 8  # Start with good score
        
        # Reduce score for complex types
        if 'varying' in perform_type:
            score -= 2
        elif 'until' in perform_type:
            score -= 1
        
        # Reduce score for long content
        if len(content) > 500:
            score -= 1
        if len(content) > 1000:
            score -= 1
        
        # Reduce score for complex nesting
        nesting_level = max(content.upper().count('IF '), content.upper().count('EVALUATE '))
        if nesting_level > 3:
            score -= 2
        elif nesting_level > 1:
            score -= 1
        
        return max(1, score)

    def _determine_host_var_context(self, sql_content: str, position: int) -> str:
        """Determine context of host variable usage"""
        before_text = sql_content[:position].upper()
        after_text = sql_content[position:position+50].upper()
        
        if 'INTO' in before_text[-20:]:
            return 'output'
        elif any(clause in before_text[-50:] for clause in ['WHERE', 'SET', 'VALUES']):
            return 'input'
        elif 'FROM' in before_text[-20:]:
            return 'input'
        else:
            return 'unknown'

    def _extract_cobol_data_definitions(self, content: str) -> Dict[str, Dict]:
        """Extract COBOL data definitions for host variable validation"""
        data_items = {}
        
        matches = self.cobol_patterns['data_item'].finditer(content)
        for match in matches:
            level = match.group(1)
            name = match.group(2).upper()
            definition = match.group(3)
            
            data_items[name] = {
                'level': level,
                'definition': definition,
                'pic': self._extract_pic_clause(definition),
                'usage': self._extract_usage_clause(definition)
            }
        
        return data_items

    def _has_required_params(self, params: str, required: List[str]) -> bool:
        """Check if CICS command has required parameters"""
        params_upper = params.upper()
        return all(param in params_upper for param in required)

    def _extract_parameter_value(self, params: str, param_name: str) -> Optional[str]:
        """Extract parameter value from CICS command parameters"""
        pattern = re.compile(f'{param_name}\\(([^)]+)\\)', re.IGNORECASE)
        match = pattern.search(params)
        return match.group(1) if match else None

    def _extract_jcl_step_complete(self, content: str, start_pos: int, all_matches: List, current_index: int) -> str:
        """Extract complete JCL step including all DD statements"""
        if current_index + 1 < len(all_matches):
            end_pos = all_matches[current_index + 1].start()
        else:
            end_pos = len(content)
        
        return content[start_pos:end_pos].strip()

    async def _analyze_jcl_step_business_context(self, content: str, step_name: str, sequence: int) -> Dict[str, Any]:
        """Analyze business context of JCL step"""
        context = {
            'step_purpose': self._infer_jcl_step_purpose(step_name, content),
            'execution_sequence': sequence + 1,
            'dependencies': self._extract_step_dependencies(content),
            'conditional_logic': self._extract_conditional_logic(content),
            'resources': self._extract_resource_requirements(content),
            'data_flow': self._analyze_jcl_data_flow(content)
        }
        
        return context

    def _infer_jcl_step_purpose(self, step_name: str, content: str) -> str:
        """Infer business purpose of JCL step"""
        name_upper = step_name.upper()
        content_upper = content.upper()
        
        if any(pattern in name_upper for pattern in ['SORT', 'MERGE']):
            return 'data_sorting'
        elif any(pattern in name_upper for pattern in ['COPY', 'BACKUP']):
            return 'data_backup'
        elif any(pattern in name_upper for pattern in ['LOAD', 'UNLOAD']):
            return 'data_transfer'
        elif 'PROC' in content_upper:
            return 'procedure_execution'
        elif 'PGM=' in content_upper:
            return 'program_execution'
        else:
            return 'batch_processing'

    def _extract_step_dependencies(self, content: str) -> List[str]:
        """Extract step dependencies from JCL content"""
        dependencies = []
        
        # Check for COND parameters that reference other steps
        cond_matches = re.finditer(r'COND=\([^)]*,([A-Z0-9]+)\)', content, re.IGNORECASE)
        for match in cond_matches:
            dependencies.append(match.group(1))
        
        return dependencies

    def _extract_conditional_logic(self, content: str) -> Dict[str, Any]:
        """Extract conditional execution logic from JCL step"""
        logic = {
            'has_conditions': False,
            'conditions': [],
            'execution_type': 'unconditional'
        }
        
        # Check for COND parameter
        cond_match = self.jcl_patterns['cond_parameter'].search(content)
        if cond_match:
            logic.update({
                'has_conditions': True,
                'conditions': [cond_match.group(1)],
                'execution_type': 'conditional'
            })
        
        return logic

    def _extract_resource_requirements(self, content: str) -> Dict[str, str]:
        """Extract resource requirements from JCL step"""
        resources = {}
        
        # Extract common resource parameters
        resource_patterns = {
            'REGION': re.compile(r'REGION=([^,\s]+)', re.IGNORECASE),
            'TIME': re.compile(r'TIME=([^,\s]+)', re.IGNORECASE),
            'CLASS': re.compile(r'CLASS=([^,\s]+)', re.IGNORECASE)
        }
        
        for resource, pattern in resource_patterns.items():
            match = pattern.search(content)
            if match:
                resources[resource] = match.group(1)
        
        return resources

    def _analyze_jcl_data_flow(self, content: str) -> Dict[str, List[str]]:
        """Analyze data flow in JCL step"""
        data_flow = {
            'input_datasets': [],
            'output_datasets': [],
            'temporary_datasets': []
        }
        
        # Find DD statements and classify datasets
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            if 'DSN=' in dd_line.upper():
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_line, re.IGNORECASE)
                if dsn_match:
                    dsn = dsn_match.group(1)
                    
                    if 'DISP=(NEW' in dd_line.upper():
                        data_flow['output_datasets'].append(dsn)
                    elif 'DISP=(OLD' in dd_line.upper() or 'DISP=SHR' in dd_line.upper():
                        data_flow['input_datasets'].append(dsn)
                    elif '&&' in dsn:  # Temporary dataset
                        data_flow['temporary_datasets'].append(dsn)
        
        return data_flow

    def _calculate_jcl_execution_complexity(self, chunks: List[CodeChunk]) -> int:
        """Calculate JCL execution complexity"""
        complexity = 0
        
        step_count = len([c for c in chunks if c.chunk_type == 'jcl_step'])
        complexity += step_count * 2
        
        conditional_steps = len([c for c in chunks if c.business_context and 
                               c.business_context.get('conditional_logic', {}).get('has_conditions', False)])
        complexity += conditional_steps * 3
        
        return min(complexity, 100)

    def _summarize_performance_indicators(self, indicators: List) -> Dict[str, Any]:
        """Summarize performance indicators across chunks"""
        summary = {
            'high_impact_operations': 0,
            'io_operations': 0,
            'cpu_intensive_operations': 0,
            'overall_performance_risk': 'low'
        }
        
        for indicator in indicators:
            if isinstance(indicator, dict):
                if indicator.get('cpu_impact') == 'high':
                    summary['cpu_intensive_operations'] += 1
                if indicator.get('io_impact') == 'high':
                    summary['io_operations'] += 1
        
        # Determine overall risk
        total_high_impact = summary['cpu_intensive_operations'] + summary['io_operations']
        if total_high_impact > 10:
            summary['overall_performance_risk'] = 'high'
        elif total_high_impact > 5:
            summary['overall_performance_risk'] = 'medium'
        
        return summary

    def _calculate_maintainability_score(self, chunks: List[CodeChunk]) -> int:
        """Calculate overall maintainability score"""
        scores = []
        
        for chunk in chunks:
            if chunk.business_context and 'maintainability_score' in chunk.business_context:
                scores.append(chunk.business_context['maintainability_score'])
            elif chunk.metadata and 'maintainability_score' in chunk.metadata:
                scores.append(chunk.metadata['maintainability_score'])
        
        return int(sum(scores) / len(scores)) if scores else 7

    # Keep all remaining helper methods unchanged for regex extraction
    def _extract_field_names(self, content: str) -> List[str]:
        """Extract field names from COBOL code - UNCHANGED"""
        fields = set()
        
        field_pattern = re.compile(r'\b\d+\s+([A-Z][A-Z0-9-]*)\s+PIC', re.IGNORECASE)
        fields.update(match.group(1) for match in field_pattern.finditer(content))
        
        move_pattern = re.compile(r'\bMOVE\s+([A-Z][A-Z0-9-]*)\s+TO\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in move_pattern.finditer(content):
            fields.add(match.group(1))
            fields.add(match.group(2))
        
        if_pattern = re.compile(r'\bIF\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        fields.update(match.group(1) for match in if_pattern.finditer(content))
        
        return list(fields)

    def _extract_file_operations(self, content: str) -> List[str]:
        """Extract file operations - UNCHANGED"""
        ops = []
        
        file_op_patterns = [
            (r'\b(OPEN)\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*)', lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
            (r'\b(READ|WRITE|REWRITE|DELETE|CLOSE)\s+([A-Z][A-Z0-9-]*)', lambda m: f"{m.group(1)} {m.group(2)}")
        ]
        
        for pattern, formatter in file_op_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                ops.append(formatter(match))
        
        return ops

    def _extract_sql_operations(self, content: str) -> List[str]:
        """Extract SQL operations - UNCHANGED"""
        ops = set()
        sql_pattern = re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|GRANT|REVOKE)\b', re.IGNORECASE)
        ops.update(match.group(1).upper() for match in sql_pattern.finditer(content))
        return list(ops)

    def _extract_perform_statements(self, content: str) -> List[str]:
        """Extract PERFORM statements - UNCHANGED"""
        performs = set()
        
        simple_pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        performs.update(match.group(1) for match in simple_pattern.finditer(content))
        
        thru_pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in thru_pattern.finditer(content):
            performs.add(f"{match.group(1)} THRU {match.group(2)}")
        
        return list(performs)

    def _extract_error_handling_patterns(self, content: str) -> List[str]:
        """Extract error handling patterns - UNCHANGED"""
        patterns = []
        content_upper = content.upper()
        
        error_indicators = [
            ('ON SIZE ERROR', 'SIZE_ERROR_HANDLING'),
            ('AT END', 'END_OF_FILE_HANDLING'),
            ('INVALID KEY', 'INVALID_KEY_HANDLING'),
            ('NOT ON SIZE ERROR', 'NO_SIZE_ERROR_PATH'),
            ('NOT AT END', 'NOT_AT_END_PATH'),
            ('NOT INVALID KEY', 'VALID_KEY_PATH'),
            ('OVERFLOW', 'OVERFLOW_HANDLING'),
            ('EXCEPTION', 'EXCEPTION_HANDLING')
        ]
        
        for indicator, pattern_name in error_indicators:
            if indicator in content_upper:
                patterns.append(pattern_name)
        
        return patterns

    # Keep all other helper methods for backward compatibility
    def _extract_sql_type(self, sql_content: str) -> str:
        """Extract SQL operation type - UNCHANGED"""
        sql_upper = sql_content.upper().strip()
        sql_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'GRANT', 'REVOKE']
        
        for sql_type in sql_types:
            if sql_upper.startswith(sql_type):
                return sql_type
        
        return 'UNKNOWN'

    def _extract_table_names(self, sql_content: str) -> List[str]:
        """Extract table names from SQL - UNCHANGED"""
        tables = set()
        
        patterns = [
            re.compile(r'\bFROM\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            re.compile(r'\bUPDATE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            re.compile(r'\bINSERT\s+INTO\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            re.compile(r'\bJOIN\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        ]
        
        for pattern in patterns:
            tables.update(match.group(1) for match in pattern.finditer(sql_content))
        
        return list(tables)

    def _categorize_cics_command(self, command_type: str) -> str:
        """Categorize CICS command - UNCHANGED"""
        terminal_commands = ['cics_send_map', 'cics_receive_map', 'cics_send_text', 'cics_receive']
        file_commands = ['cics_read', 'cics_write', 'cics_rewrite', 'cics_delete', 'cics_startbr', 'cics_readnext']
        program_commands = ['cics_link', 'cics_xctl', 'cics_load', 'cics_return']
        storage_commands = ['cics_getmain', 'cics_freemain']
        
        if command_type in terminal_commands:
            return "terminal"
        elif command_type in file_commands:
            return "file"
        elif command_type in program_commands:
            return "program_control"
        elif command_type in storage_commands:
            return "storage"
        else:
            return "other"

    def _extract_cics_parameters(self, params: str) -> Dict[str, str]:
        """Extract CICS command parameters - UNCHANGED"""
        param_dict = {}
        
        param_patterns = {
            'FILE': re.compile(r'FILE\(([^)]+)\)', re.IGNORECASE),
            'MAP': re.compile(r'MAP\(([^)]+)\)', re.IGNORECASE),
            'MAPSET': re.compile(r'MAPSET\(([^)]+)\)', re.IGNORECASE),
            'PROGRAM': re.compile(r'PROGRAM\(([^)]+)\)', re.IGNORECASE),
            'LENGTH': re.compile(r'LENGTH\(([^)]+)\)', re.IGNORECASE),
            'INTO': re.compile(r'INTO\(([^)]+)\)', re.IGNORECASE),
            'FROM': re.compile(r'FROM\(([^)]+)\)', re.IGNORECASE),
            'RIDFLD': re.compile(r'RIDFLD\(([^)]+)\)', re.IGNORECASE)
        }
        
        for param_name, pattern in param_patterns.items():
            match = pattern.search(params)
            if match:
                param_dict[param_name] = match.group(1)
        
        return param_dict

    def _extract_cics_resource(self, params: str) -> str:
        """Extract main resource from CICS parameters - UNCHANGED"""
        resource_patterns = ['FILE', 'MAP', 'PROGRAM', 'QUEUE']
        
        for resource_type in resource_patterns:
            pattern = re.compile(f'{resource_type}\\(([^)]+)\\)', re.IGNORECASE)
            match = pattern.search(params)
            if match:
                return match.group(1)
        
        return "UNKNOWN"

    def _extract_exec_program(self, content: str) -> str:
        """Extract program name from EXEC statement - UNCHANGED"""
        exec_patterns = [
            re.compile(r'EXEC\s+PGM=([A-Z0-9]+)', re.IGNORECASE),
            re.compile(r'EXEC\s+([A-Z0-9]+)', re.IGNORECASE)
        ]
        
        for pattern in exec_patterns:
            match = pattern.search(content)
            if match:
                return match.group(1)
        
        return "UNKNOWN"

    def _extract_input_datasets_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed input dataset information - UNCHANGED"""
        datasets = []
        
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            if any(indicator in dd_line.upper() for indicator in ['DSN=', 'DISP=SHR', 'DISP=OLD']):
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_line, re.IGNORECASE)
                disp_match = re.search(r'DISP=\(([^,)]+)', dd_line, re.IGNORECASE)
                
                datasets.append({
                    "dd_name": dd_name,
                    "dsn": dsn_match.group(1) if dsn_match else "UNKNOWN",
                    "disposition": disp_match.group(1) if disp_match else "UNKNOWN",
                    "purpose": "input_data"
                })
        
        return datasets

    def _extract_output_datasets_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed output dataset information - UNCHANGED"""
        datasets = []
        
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            if any(indicator in dd_line.upper() for indicator in ['DISP=(NEW', 'DISP=(,CATLG', 'DISP=(,DELETE']):
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_line, re.IGNORECASE)
                disp_match = re.search(r'DISP=\(([^,)]+)', dd_line, re.IGNORECASE)
                
                datasets.append({
                    "dd_name": dd_name,
                    "dsn": dsn_match.group(1) if dsn_match else "UNKNOWN",
                    "disposition": disp_match.group(1) if disp_match else "UNKNOWN",
                    "purpose": "output_data"
                })
        
        return datasets

    def _extract_parameters_detailed(self, content: str) -> Dict[str, str]:
        """Extract detailed parameters - UNCHANGED"""
        params = {}
        
        param_patterns = {
            'PARM': re.compile(r'PARM=([^,\s]+)', re.IGNORECASE),
            'REGION': re.compile(r'REGION=([^,\s]+)', re.IGNORECASE),
            'TIME': re.compile(r'TIME=([^,\s]+)', re.IGNORECASE)
        }
        
        for param_name, pattern in param_patterns.items():
            match = pattern.search(content)
            if match:
                params[param_name] = match.group(1)
        
        return params

    # Additional helper methods for enhanced business analysis
    def _extract_pic_clause(self, definition: str) -> Optional[str]:
        """Extract PIC clause from field definition - ENHANCED"""
        match = self.cobol_patterns['pic_clause'].search(definition)
        if match:
            return match.group(1) or match.group(2)
        return None

    def _extract_usage_clause(self, definition: str) -> str:
        """Extract USAGE clause from field definition - ENHANCED"""
        match = self.cobol_patterns['usage_clause'].search(definition)
        return match.group(1) if match else "DISPLAY"

    def _extract_value_clause(self, definition: str) -> Optional[str]:
        """Extract VALUE clause from field definition - UNCHANGED"""
        match = self.cobol_patterns['value_clause'].search(definition)
        return match.group(1) if match else None

    def _extract_occurs_info(self, definition: str) -> Optional[Dict[str, Any]]:
        """Extract OCCURS clause information - ENHANCED"""
        match = self.cobol_patterns['occurs_clause'].search(definition)
        if match:
            min_occurs = int(match.group(1))
            max_occurs = int(match.group(2)) if match.group(2) else min_occurs
            depending_field = match.group(3) if match.group(3) else None
            indexed_field = match.group(4) if match.group(4) else None
            
            return {
                "min_occurs": min_occurs,
                "max_occurs": max_occurs,
                "is_variable": max_occurs != min_occurs or depending_field is not None,
                "depending_on": depending_field,
                "indexed_by": indexed_field
            }
        return None

    def _extract_redefines_info(self, definition: str) -> Optional[str]:
        """Extract REDEFINES information - UNCHANGED"""
        match = self.cobol_patterns['redefines'].search(definition)
        return match.group(1) if match else None

    def _determine_data_type_enhanced(self, definition: str) -> str:
        """Enhanced data type determination"""
        pic_clause = self._extract_pic_clause(definition)
        usage = self._extract_usage_clause(definition)
        
        if not pic_clause:
            return "group"
        
        pic_upper = pic_clause.upper()
        
        # Enhanced type detection
        if '9' in pic_upper:
            if 'V' in pic_upper or '.' in pic_upper:
                if usage in ['COMP-3', 'PACKED-DECIMAL']:
                    return "packed_decimal"
                elif usage in ['COMP', 'COMP-4', 'BINARY']:
                    return "binary_decimal"
                else:
                    return "display_decimal"
            else:
                if usage in ['COMP', 'COMP-4', 'BINARY']:
                    return "binary_integer"
                elif usage == 'COMP-3':
                    return "packed_integer"
                else:
                    return "display_integer"
        elif 'X' in pic_upper:
            return "alphanumeric"
        elif 'A' in pic_upper:
            return "alphabetic"
        elif 'N' in pic_upper:
            return "national"
        elif 'S' in pic_upper and '9' in pic_upper:
            return "signed_numeric"
        else:
            return "special"

    def _infer_business_domain(self, name: str) -> str:
        """Infer business domain from field name"""
        name_upper = name.upper()
        
        # Financial domain indicators
        if any(pattern in name_upper for pattern in [
            'AMOUNT', 'AMT', 'BALANCE', 'BAL', 'RATE', 'INTEREST', 'PRINCIPAL',
            'PAYMENT', 'PMT', 'CHARGE', 'FEE', 'COST', 'PRICE', 'VALUE', 'VAL'
        ]):
            return 'financial'
        
        # Customer domain indicators
        if any(pattern in name_upper for pattern in [
            'CUSTOMER', 'CUST', 'CLIENT', 'MEMBER', 'ACCOUNT', 'ACCT'
        ]):
            return 'customer'
        
        # Product domain indicators
        if any(pattern in name_upper for pattern in [
            'PRODUCT', 'PROD', 'ITEM', 'SERVICE', 'POLICY', 'CONTRACT'
        ]):
            return 'product'
        
        # Transaction domain indicators
        if any(pattern in name_upper for pattern in [
            'TRANSACTION', 'TRANS', 'TXN', 'POSTING', 'ENTRY'
        ]):
            return 'transaction'
        
        # Date/Time domain indicators
        if any(pattern in name_upper for pattern in [
            'DATE', 'TIME', 'TIMESTAMP', 'YEAR', 'MONTH', 'DAY'
        ]):
            return 'temporal'
        
        # Control/Status domain indicators
        if any(pattern in name_upper for pattern in [
            'STATUS', 'FLAG', 'INDICATOR', 'IND', 'CODE', 'TYPE'
        ]):
            return 'control'
        
        return 'general'

    def _analyze_usage_pattern(self, definition: str) -> str:
        """Analyze usage pattern of data item"""
        definition_upper = definition.upper()
        
        if 'VALUE' in definition_upper:
            if any(val in definition_upper for val in ['SPACE', 'ZERO', 'LOW-VALUE', 'HIGH-VALUE']):
                return 'initialized_constant'
            else:
                return 'initialized_variable'
        elif 'OCCURS' in definition_upper:
            return 'array_table'
        elif 'REDEFINES' in definition_upper:
            return 'overlay_structure'
        elif any(usage in definition_upper for usage in ['COMP', 'BINARY', 'PACKED']):
            return 'computational'
        else:
            return 'standard_storage'

    def _extract_validation_rules(self, definition: str) -> List[str]:
        """Extract validation rules from data definition"""
        rules = []
        definition_upper = definition.upper()
        
        # Picture clause validations
        pic_clause = self._extract_pic_clause(definition)
        if pic_clause:
            if '9' in pic_clause:
                rules.append('numeric_only')
            if 'A' in pic_clause:
                rules.append('alphabetic_only')
            if 'X' in pic_clause:
                rules.append('alphanumeric')
        
        # Value clause validations
        value_clause = self._extract_value_clause(definition)
        if value_clause:
            rules.append('default_value_assigned')
        
        # OCCURS validations
        occurs_info = self._extract_occurs_info(definition)
        if occurs_info:
            rules.append('array_bounds_check')
            if occurs_info.get('depending_on'):
                rules.append('variable_length_validation')
        
        return rules

    def _classify_security_level(self, name: str, definition: str) -> str:
        """Classify security level of data item"""
        name_upper = name.upper()
        definition_upper = definition.upper()
        
        # High security indicators
        if any(pattern in name_upper for pattern in [
            'SSN', 'SOCIAL', 'PASSWORD', 'PIN', 'ACCOUNT', 'CREDIT', 'DEBIT'
        ]):
            return 'high'
        
        # Medium security indicators
        if any(pattern in name_upper for pattern in [
            'NAME', 'ADDRESS', 'PHONE', 'EMAIL', 'SALARY', 'INCOME'
        ]):
            return 'medium'
        
        # Check for PII patterns in definition
        if 'PIC' in definition_upper:
            pic = self._extract_pic_clause(definition)
            if pic and 'X' in pic:
                # Text fields might contain sensitive data
                return 'medium'
        
        return 'low'

    def _extract_data_operations(self, content: str) -> List[Dict[str, str]]:
        """Extract data operations from paragraph content"""
        operations = []
        
        # MOVE operations
        move_pattern = re.compile(r'MOVE\s+([A-Z0-9-]+)\s+TO\s+([A-Z0-9-]+)', re.IGNORECASE)
        for match in move_pattern.finditer(content):
            operations.append({
                'operation': 'MOVE',
                'source': match.group(1),
                'target': match.group(2)
            })
        
        # Arithmetic operations
        arithmetic_patterns = {
            'ADD': re.compile(r'ADD\s+([A-Z0-9-]+)\s+TO\s+([A-Z0-9-]+)', re.IGNORECASE),
            'SUBTRACT': re.compile(r'SUBTRACT\s+([A-Z0-9-]+)\s+FROM\s+([A-Z0-9-]+)', re.IGNORECASE),
            'MULTIPLY': re.compile(r'MULTIPLY\s+([A-Z0-9-]+)\s+BY\s+([A-Z0-9-]+)', re.IGNORECASE),
            'DIVIDE': re.compile(r'DIVIDE\s+([A-Z0-9-]+)\s+(?:BY|INTO)\s+([A-Z0-9-]+)', re.IGNORECASE)
        }
        
        for op_name, pattern in arithmetic_patterns.items():
            for match in pattern.finditer(content):
                operations.append({
                    'operation': op_name,
                    'operand1': match.group(1),
                    'operand2': match.group(2)
                })
        
        return operations

    def _extract_file_operations_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed file operations"""
        operations = []
        
        file_patterns = {
            'OPEN': re.compile(r'OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z0-9-]+)', re.IGNORECASE),
            'READ': re.compile(r'READ\s+([A-Z0-9-]+)', re.IGNORECASE),
            'WRITE': re.compile(r'WRITE\s+([A-Z0-9-]+)', re.IGNORECASE),
            'REWRITE': re.compile(r'REWRITE\s+([A-Z0-9-]+)', re.IGNORECASE),
            'DELETE': re.compile(r'DELETE\s+([A-Z0-9-]+)', re.IGNORECASE),
            'CLOSE': re.compile(r'CLOSE\s+([A-Z0-9-]+)', re.IGNORECASE)
        }
        
        for op_name, pattern in file_patterns.items():
            for match in pattern.finditer(content):
                if op_name == 'OPEN':
                    operations.append({
                        'operation': op_name,
                        'file': match.group(2),
                        'mode': match.group(1)
                    })
                else:
                    operations.append({
                        'operation': op_name,
                        'file': match.group(1),
                        'mode': ''
                    })
        
        return operations

    def _extract_control_flow_elements(self, content: str) -> List[Dict[str, str]]:
        """Extract control flow elements from content"""
        elements = []
        
        # IF statements
        if_pattern = re.compile(r'IF\s+([^\.]+)', re.IGNORECASE)
        for match in if_pattern.finditer(content):
            elements.append({
                'type': 'IF',
                'condition': match.group(1).strip()
            })
        
        # PERFORM statements
        perform_pattern = re.compile(r'PERFORM\s+([A-Z0-9-]+)', re.IGNORECASE)
        for match in perform_pattern.finditer(content):
            elements.append({
                'type': 'PERFORM',
                'target': match.group(1)
            })
        
        # EVALUATE statements
        evaluate_pattern = re.compile(r'EVALUATE\s+([^\.]+)', re.IGNORECASE)
        for match in evaluate_pattern.finditer(content):
            elements.append({
                'type': 'EVALUATE',
                'expression': match.group(1).strip()
            })
        
        return elements

    def _extract_error_handling_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed error handling information"""
        error_handling = []
        
        error_patterns = {
            'ON SIZE ERROR': re.compile(r'ON\s+SIZE\s+ERROR\s*([^\.]*)', re.IGNORECASE),
            'AT END': re.compile(r'AT\s+END\s*([^\.]*)', re.IGNORECASE),
            'INVALID KEY': re.compile(r'INVALID\s+KEY\s*([^\.]*)', re.IGNORECASE),
            'NOT ON SIZE ERROR': re.compile(r'NOT\s+ON\s+SIZE\s+ERROR\s*([^\.]*)', re.IGNORECASE),
            'NOT AT END': re.compile(r'NOT\s+AT\s+END\s*([^\.]*)', re.IGNORECASE),
            'NOT INVALID KEY': re.compile(r'NOT\s+INVALID\s+KEY\s*([^\.]*)', re.IGNORECASE)
        }
        
        for error_type, pattern in error_patterns.items():
            for match in pattern.finditer(content):
                error_handling.append({
                    'type': error_type,
                    'action': match.group(1).strip() if match.group(1) else 'IMPLICIT'
                })
        
        return error_handling

    def _assess_performance_impact(self, content: str) -> Dict[str, str]:
        """Assess performance impact of paragraph"""
        impact = {
            'cpu_intensive': 'low',
            'io_operations': 'low',
            'memory_usage': 'low',
            'overall_impact': 'low'
        }
        
        content_upper = content.upper()
        
        # Check for CPU-intensive operations
        if any(op in content_upper for op in ['PERFORM', 'UNTIL', 'VARYING', 'COMPUTE']):
            impact['cpu_intensive'] = 'medium'
        
        if content_upper.count('PERFORM') > 5:
            impact['cpu_intensive'] = 'high'
        
        # Check for I/O operations
        io_operations = ['READ', 'WRITE', 'REWRITE', 'DELETE', 'EXEC SQL']
        io_count = sum(content_upper.count(op) for op in io_operations)
        
        if io_count > 0:
            impact['io_operations'] = 'medium'
        if io_count > 3:
            impact['io_operations'] = 'high'
        
        # Check for memory-intensive operations
        if any(op in content_upper for op in ['OCCURS', 'TABLE', 'ARRAY']):
            impact['memory_usage'] = 'medium'
        
        # Determine overall impact
        high_impacts = [v for v in impact.values() if v == 'high']
        medium_impacts = [v for v in impact.values() if v == 'medium']
        
        if high_impacts:
            impact['overall_impact'] = 'high'
        elif len(medium_impacts) >= 2:
            impact['overall_impact'] = 'medium'
        
        return impact

    def _identify_business_rules(self, content: str) -> List[str]:
        """Identify business rules implemented in paragraph"""
        rules = []
        content_upper = content.upper()
        
        # Validation rules
        if any(pattern in content_upper for pattern in ['IF', 'WHEN', 'EVALUATE']):
            rules.append('conditional_validation')
        
        # Range checks
        if any(pattern in content_upper for pattern in ['GREATER', 'LESS', 'EQUAL', '>', '<', '=']):
            rules.append('range_validation')
        
        # Required field checks
        if any(pattern in content_upper for pattern in ['SPACE', 'ZERO', 'EMPTY']):
            rules.append('required_field_validation')
        
        # Business calculations
        if any(pattern in content_upper for pattern in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'COMPUTE']):
            rules.append('business_calculation')
        
        # Data transformation
        if 'MOVE' in content_upper:
            rules.append('data_transformation')
        
        # Error processing
        if any(pattern in content_upper for pattern in ['ERROR', 'EXCEPTION', 'INVALID']):
            rules.append('error_processing')
        
        return rules

    def _calculate_paragraph_complexity(self, content: str) -> int:
        """Calculate complexity score for paragraph"""
        complexity = 1  # Base complexity
        content_upper = content.upper()
        
        # Add complexity for control structures
        complexity += content_upper.count('IF ')
        complexity += content_upper.count('EVALUATE ') * 2
        complexity += content_upper.count('PERFORM ')
        complexity += content_upper.count('WHEN ')
        
        # Add complexity for file operations
        complexity += content_upper.count('READ ')
        complexity += content_upper.count('WRITE ')
        complexity += content_upper.count('EXEC SQL') * 2
        
        # Add complexity for error handling
        complexity += content_upper.count('ON SIZE ERROR')
        complexity += content_upper.count('AT END')
        complexity += content_upper.count('INVALID KEY')
        
        return min(complexity, 20)  # Cap at 20

    def _analyze_performance_indicators(self, content: str) -> Dict[str, Any]:
        """Analyze performance indicators in paragraph"""
        indicators = {
            'loop_operations': 0,
            'file_io_operations': 0,
            'sql_operations': 0,
            'arithmetic_operations': 0,
            'performance_risk': 'low'
        }
        
        content_upper = content.upper()
        
        # Count different operation types
        indicators['loop_operations'] = content_upper.count('PERFORM UNTIL') + content_upper.count('PERFORM VARYING')
        indicators['file_io_operations'] = (content_upper.count('READ ') + content_upper.count('WRITE ') + 
                                          content_upper.count('REWRITE ') + content_upper.count('DELETE '))
        indicators['sql_operations'] = content_upper.count('EXEC SQL')
        indicators['arithmetic_operations'] = (content_upper.count('ADD ') + content_upper.count('SUBTRACT ') +
                                             content_upper.count('MULTIPLY ') + content_upper.count('DIVIDE ') +
                                             content_upper.count('COMPUTE '))
        
        # Assess overall performance risk
        total_operations = (indicators['loop_operations'] * 3 + 
                          indicators['file_io_operations'] * 2 +
                          indicators['sql_operations'] * 3 +
                          indicators['arithmetic_operations'])
        
        if total_operations > 10:
            indicators['performance_risk'] = 'high'
        elif total_operations > 5:
            indicators['performance_risk'] = 'medium'
        
        return indicators

    async def _analyze_sql_business_context(self, sql_content: str, host_variables: List[Dict]) -> Dict[str, Any]:
        """Analyze business context of SQL block"""
        context = {
            'sql_purpose': self._classify_sql_purpose(sql_content),
            'data_access_pattern': self._analyze_sql_access_pattern(sql_content),
            'business_entity': self._identify_business_entity(sql_content),
            'transaction_impact': self._assess_sql_transaction_impact(sql_content),
            'host_variable_flow': self._analyze_host_variable_flow(host_variables)
        }
        
        return context

    def _classify_sql_purpose(self, sql_content: str) -> str:
        """Classify the business purpose of SQL statement"""
        sql_upper = sql_content.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            if 'COUNT(' in sql_upper:
                return 'data_validation'
            elif 'SUM(' in sql_upper or 'AVG(' in sql_upper:
                return 'aggregation'
            else:
                return 'data_retrieval'
        elif sql_upper.startswith('INSERT'):
            return 'data_creation'
        elif sql_upper.startswith('UPDATE'):
            return 'data_modification'
        elif sql_upper.startswith('DELETE'):
            return 'data_removal'
        else:
            return 'database_operation'

    def _analyze_sql_access_pattern(self, sql_content: str) -> str:
        """Analyze SQL data access pattern"""
        sql_upper = sql_content.upper()
        
        if 'JOIN' in sql_upper:
            return 'relational_access'
        elif 'WHERE' in sql_upper and ('=' in sql_upper or 'IN' in sql_upper):
            return 'indexed_access'
        elif 'WHERE' in sql_upper:
            return 'filtered_access'
        elif 'ORDER BY' in sql_upper:
            return 'sorted_access'
        else:
            return 'sequential_access'

    def _identify_business_entity(self, sql_content: str) -> str:
        """Identify primary business entity in SQL"""
        sql_upper = sql_content.upper()
        
        # Extract table names and classify
        tables = self._extract_table_names(sql_content)
        
        for table in tables:
            table_upper = table.upper()
            if any(pattern in table_upper for pattern in ['CUSTOMER', 'CLIENT', 'MEMBER']):
                return 'customer'
            elif any(pattern in table_upper for pattern in ['ACCOUNT', 'ACCT']):
                return 'account'
            elif any(pattern in table_upper for pattern in ['PRODUCT', 'ITEM', 'SERVICE']):
                return 'product'
            elif any(pattern in table_upper for pattern in ['TRANSACTION', 'TRANS', 'TXN']):
                return 'transaction'
            elif any(pattern in table_upper for pattern in ['ORDER', 'INVOICE', 'PAYMENT']):
                return 'order_management'
        
        return 'general'

    def _assess_sql_transaction_impact(self, sql_content: str) -> str:
        """Assess transaction impact of SQL statement"""
        sql_upper = sql_content.upper()
        
        if any(stmt in sql_upper for stmt in ['INSERT', 'UPDATE', 'DELETE']):
            if 'WHERE' not in sql_upper:
                return 'high'  # Mass updates without WHERE clause
            else:
                return 'medium'
        elif sql_upper.startswith('SELECT'):
            if any(func in sql_upper for func in ['COUNT(', 'SUM(', 'MAX(', 'MIN(']):
                return 'medium'  # Aggregation queries
            else:
                return 'low'
        else:
            return 'low'

    def _analyze_host_variable_flow(self, host_variables: List[Dict]) -> Dict[str, List[str]]:
        """Analyze host variable data flow"""
        flow = {
            'input_variables': [],
            'output_variables': [],
            'bidirectional_variables': []
        }
        
        for var in host_variables:
            context = var.get('context', 'unknown')
            var_name = var.get('name', '')
            
            if context == 'input':
                flow['input_variables'].append(var_name)
            elif context == 'output':
                flow['output_variables'].append(var_name)
            else:
                flow['bidirectional_variables'].append(var_name)
        
        return flow

    def _calculate_sql_complexity(self, sql_content: str) -> int:
        """Calculate SQL complexity score"""
        complexity = 1
        sql_upper = sql_content.upper()
        
        # Add complexity for joins
        complexity += sql_upper.count('JOIN') * 2
        
        # Add complexity for subqueries
        complexity += sql_upper.count('SELECT') - 1  # Exclude main SELECT
        
        # Add complexity for aggregation functions
        complexity += sql_upper.count('GROUP BY') * 2
        complexity += sql_upper.count('HAVING') * 2
        
        # Add complexity for functions
        functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'SUBSTR(', 'DECODE(']
        complexity += sum(sql_upper.count(func) for func in functions)
        
        return min(complexity, 20)

    def _analyze_sql_performance(self, sql_content: str) -> Dict[str, Any]:
        """Analyze SQL performance indicators"""
        indicators = {
            'join_count': 0,
            'subquery_count': 0,
            'function_count': 0,
            'has_where_clause': False,
            'has_order_by': False,
            'performance_risk': 'low'
        }
        
        sql_upper = sql_content.upper()
        
        indicators['join_count'] = sql_upper.count('JOIN')
        indicators['subquery_count'] = sql_upper.count('SELECT') - 1
        indicators['has_where_clause'] = 'WHERE' in sql_upper
        indicators['has_order_by'] = 'ORDER BY' in sql_upper
        
        # Count function usage
        functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'SUBSTR(', 'DECODE(']
        indicators['function_count'] = sum(sql_upper.count(func) for func in functions)
        
        # Assess performance risk
        risk_score = 0
        if indicators['join_count'] > 3:
            risk_score += 2
        if indicators['subquery_count'] > 2:
            risk_score += 2
        if not indicators['has_where_clause'] and any(stmt in sql_upper for stmt in ['UPDATE', 'DELETE']):
            risk_score += 3
        if indicators['function_count'] > 5:
            risk_score += 1
        
        if risk_score >= 5:
            indicators['performance_risk'] = 'high'
        elif risk_score >= 2:
            indicators['performance_risk'] = 'medium'
        
        return indicators

    async def _analyze_cics_business_context(self, cmd: Dict, transaction_state: TransactionState) -> Dict[str, Any]:
        """Analyze business context of CICS command"""
        context = {
            'transaction_purpose': self._classify_cics_transaction_purpose(cmd),
            'resource_usage': self._analyze_cics_resource_usage(cmd),
            'user_interaction': self._classify_user_interaction(cmd),
            'data_flow': self._analyze_cics_data_flow(cmd),
            'error_strategy': self._analyze_cics_error_strategy(cmd)
        }
        
        return context

    def _classify_cics_transaction_purpose(self, cmd: Dict) -> str:
        """Classify the business purpose of CICS command"""
        cmd_type = cmd['type']
        
        if cmd_type in ['cics_receive_map', 'cics_receive']:
            return 'user_input'
        elif cmd_type in ['cics_send_map', 'cics_send_text']:
            return 'user_output'
        elif cmd_type in ['cics_read', 'cics_write', 'cics_rewrite', 'cics_delete']:
            return 'data_access'
        elif cmd_type in ['cics_link', 'cics_xctl']:
            return 'program_control'
        elif cmd_type == 'cics_return':
            return 'transaction_termination'
        else:
            return 'system_operation'

    def _analyze_cics_resource_usage(self, cmd: Dict) -> Dict[str, str]:
        """Analyze resource usage of CICS command"""
        usage = {
            'resource_type': 'unknown',
            'resource_name': 'unknown',
            'access_mode': 'unknown'
        }
        
        cmd_params = cmd['params']
        
        # Determine resource type and name
        if 'FILE(' in cmd_params.upper():
            usage['resource_type'] = 'file'
            usage['resource_name'] = self._extract_parameter_value(cmd_params, 'FILE') or 'unknown'
        elif 'MAP(' in cmd_params.upper():
            usage['resource_type'] = 'map'
            usage['resource_name'] = self._extract_parameter_value(cmd_params, 'MAP') or 'unknown'
        elif 'PROGRAM(' in cmd_params.upper():
            usage['resource_type'] = 'program'
            usage['resource_name'] = self._extract_parameter_value(cmd_params, 'PROGRAM') or 'unknown'
        elif 'QUEUE(' in cmd_params.upper():
            usage['resource_type'] = 'queue'
            usage['resource_name'] = self._extract_parameter_value(cmd_params, 'QUEUE') or 'unknown'
        
        # Determine access mode
        cmd_type = cmd['type']
        if 'read' in cmd_type:
            usage['access_mode'] = 'read'
        elif any(op in cmd_type for op in ['write', 'rewrite', 'delete']):
            usage['access_mode'] = 'write'
        elif 'send' in cmd_type:
            usage['access_mode'] = 'output'
        elif 'receive' in cmd_type:
            usage['access_mode'] = 'input'
        
        return usage

    def _classify_user_interaction(self, cmd: Dict) -> str:
        """Classify type of user interaction"""
        cmd_type = cmd['type']
        
        if cmd_type == 'cics_receive_map':
            return 'structured_input'
        elif cmd_type == 'cics_receive':
            return 'free_form_input'
        elif cmd_type == 'cics_send_map':
            return 'structured_output'
        elif cmd_type == 'cics_send_text':
            return 'text_output'
        else:
            return 'no_interaction'

    def _analyze_cics_data_flow(self, cmd: Dict) -> Dict[str, Any]:
        """Analyze data flow in CICS command"""
        flow = {
            'direction': 'unknown',
            'data_areas': [],
            'length_specified': False
        }
        
        cmd_params = cmd['params']
        cmd_type = cmd['type']
        
        # Determine data flow direction
        if any(direction in cmd_type for direction in ['receive', 'read']):
            flow['direction'] = 'inbound'
        elif any(direction in cmd_type for direction in ['send', 'write']):
            flow['direction'] = 'outbound'
        
        # Extract data areas
        data_area_patterns = ['INTO(', 'FROM(', 'SET(', 'RIDFLD(']
        for pattern in data_area_patterns:
            if pattern in cmd_params.upper():
                param_name = pattern[:-1]  # Remove the '('
                area = self._extract_parameter_value(cmd_params, param_name)
                if area:
                    flow['data_areas'].append(area)
        
        # Check if length is specified
        flow['length_specified'] = 'LENGTH(' in cmd_params.upper()
        
        return flow

    def _analyze_cics_error_strategy(self, cmd: Dict) -> Dict[str, Any]:
        """Analyze error handling strategy in CICS command"""
        strategy = {
            'error_handling_type': 'none',
            'specific_conditions': [],
            'recovery_mechanism': 'unknown'
        }
        
        cmd_params = cmd['params']
        
        if 'RESP(' in cmd_params.upper():
            strategy['error_handling_type'] = 'response_code'
            strategy['recovery_mechanism'] = 'programmatic'
        elif 'NOHANDLE' in cmd_params.upper():
            strategy['error_handling_type'] = 'no_handle'
            strategy['recovery_mechanism'] = 'ignore'
        else:
            strategy['error_handling_type'] = 'default_handle'
            strategy['recovery_mechanism'] = 'system_default'
        
        return strategy

    # Business Validator Classes Implementation
    class COBOLBusinessValidator:
        """Business rule validator for COBOL programs"""
        
        async def validate_structure(self, content: str) -> List[BusinessRuleViolation]:
            violations = []
            
            # Check division order
            divisions = self._find_divisions(content)
            expected_order = ['IDENTIFICATION', 'ENVIRONMENT', 'DATA', 'PROCEDURE']
            
            if not self._validate_division_order(divisions, expected_order):
                violations.append(BusinessRuleViolation(
                    rule="DIVISION_ORDER",
                    context="COBOL divisions must appear in correct order",
                    severity="ERROR"
                ))
            
            # Check required divisions
            required = ['IDENTIFICATION', 'PROCEDURE']
            for req in required:
                if req not in divisions:
                    violations.append(BusinessRuleViolation(
                        rule="MISSING_DIVISION",
                        context=f"Missing required {req} DIVISION",
                        severity="ERROR"
                    ))
            
            return violations
        
        def _find_divisions(self, content: str) -> List[str]:
            division_patterns = {
                'IDENTIFICATION': r'IDENTIFICATION\s+DIVISION',
                'ENVIRONMENT': r'ENVIRONMENT\s+DIVISION',
                'DATA': r'DATA\s+DIVISION',
                'PROCEDURE': r'PROCEDURE\s+DIVISION'
            }
            
            found = []
            for div_name, pattern in division_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    found.append(div_name)
            
            return found
        
        def _validate_division_order(self, found_divisions: List[str], expected_order: List[str]) -> bool:
            filtered_expected = [div for div in expected_order if div in found_divisions]
            return found_divisions == filtered_expected

    class JCLBusinessValidator:
        """Business rule validator for JCL"""
        
        async def validate_structure(self, content: str) -> List[BusinessRuleViolation]:
            violations = []
            
            # Check for JOB card
            if not re.search(r'^//\w+\s+JOB\s', content, re.MULTILINE):
                violations.append(BusinessRuleViolation(
                    rule="MISSING_JOB_CARD",
                    context="JCL must start with a JOB card",
                    severity="ERROR"
                ))
            
            # Check for at least one EXEC step
            if not re.search(r'^//\w+\s+EXEC\s', content, re.MULTILINE):
                violations.append(BusinessRuleViolation(
                    rule="NO_EXEC_STEPS",
                    context="JCL must have at least one EXEC step",
                    severity="ERROR"
                ))
            
            return violations

    class CICSBusinessValidator:
        """Business rule validator for CICS programs"""
        
        async def validate_structure(self, content: str) -> List[BusinessRuleViolation]:
            violations = []
            
            # Check for proper CICS command structure
            cics_commands = re.findall(r'EXEC\s+CICS\s+(\w+)', content, re.IGNORECASE)
            
            if not cics_commands:
                violations.append(BusinessRuleViolation(
                    rule="NO_CICS_COMMANDS",
                    context="CICS program must contain CICS commands",
                    severity="WARNING"
                ))
            
            # Check for transaction termination
            has_return = bool(re.search(r'EXEC\s+CICS\s+RETURN', content, re.IGNORECASE))
            if not has_return and cics_commands:
                violations.append(BusinessRuleViolation(
                    rule="MISSING_RETURN",
                    context="CICS program should have EXEC CICS RETURN",
                    severity="WARNING"
                ))
            
            return violations

    class BMSBusinessValidator:
        """Business rule validator for BMS mapsets"""
        
        async def validate_structure(self, content: str) -> List[BusinessRuleViolation]:
            violations = []
            
            # Check for mapset definition
            if not re.search(r'\w+\s+DFHMSD', content, re.IGNORECASE):
                violations.append(BusinessRuleViolation(
                    rule="MISSING_MAPSET",
                    context="BMS file must contain mapset definition",
                    severity="ERROR"
                ))
            
            # Check for map definition
            if not re.search(r'\w+\s+DFHMDI', content, re.IGNORECASE):
                violations.append(BusinessRuleViolation(
                    rule="MISSING_MAP",
                    context="BMS mapset must contain at least one map",
                    severity="WARNING"
                ))
            
            return violations

    # Additional parsing methods for completeness
    async def _parse_bms_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse BMS with business rule validation"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # Parse mapset definition
        mapset_chunk = await self._parse_bms_mapset_definition(content, program_name)
        if mapset_chunk:
            chunks.append(mapset_chunk)
        
        # Parse individual maps
        map_chunks = await self._parse_bms_maps(content, program_name)
        chunks.extend(map_chunks)
        
        # Parse fields
        field_chunks = await self._parse_bms_fields(content, program_name)
        chunks.extend(field_chunks)
        
        return chunks

    async def _parse_bms_mapset_definition(self, content: str, program_name: str) -> Optional[CodeChunk]:
        """Parse BMS mapset definition"""
        mapset_match = self.bms_patterns['bms_mapset'].search(content)
        if not mapset_match:
            return None
        
        mapset_name = mapset_match.group(1)
        mapset_definition = mapset_match.group(2)
        
        business_context = {
            'mapset_purpose': 'screen_definition',
            'terminal_interaction': 'user_interface',
            'data_presentation': 'structured_forms'
        }
        
        metadata = {
            "mapset_name": mapset_name,
            "bms_type": "mapset_definition",
            "attributes": self._extract_bms_attributes(mapset_definition),
            "maps_count": len(self.bms_patterns['bms_map'].findall(content))
        }
        
        return CodeChunk(
            program_name=program_name,
            chunk_id=f"{program_name}_MAPSET_{mapset_name}",
            chunk_type="bms_mapset",
            content=mapset_match.group(0),
            metadata=metadata,
            business_context=business_context,
            line_start=content[:mapset_match.start()].count('\n'),
            line_end=content[:mapset_match.end()].count('\n')
        )

    async def _parse_bms_maps(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse BMS individual maps"""
        chunks = []
        
        map_matches = self.bms_patterns['bms_map'].finditer(content)
        
        for match in map_matches:
            map_name = match.group(1)
            map_definition = match.group(2)
            
            business_context = {
                'screen_purpose': self._classify_screen_purpose(map_name),
                'user_interaction_type': 'form_based',
                'data_entry_pattern': 'structured_input'
            }
            
            metadata = {
                "map_name": map_name,
                "bms_type": "map_definition",
                "attributes": self._extract_bms_attributes(map_definition),
                "field_count": self._count_bms_fields_in_map(content, match.end())
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_MAP_{map_name}",
                chunk_type="bms_map",
                content=match.group(0),
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_bms_fields(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse BMS field definitions"""
        chunks = []
        
        field_matches = self.bms_patterns['bms_field'].finditer(content)
        
        for match in field_matches:
            field_name = match.group(1)
            field_definition = match.group(2)
            
            business_context = {
                'field_purpose': self._classify_field_purpose(field_name),
                'data_validation': self._analyze_field_validation(field_definition),
                'user_interaction': 'input' if self._is_input_field(field_definition) else 'output'
            }
            
            metadata = {
                "field_name": field_name,
                "bms_type": "field_definition",
                "attributes": self._extract_bms_attributes(field_definition),
                "is_input_field": self._is_input_field(field_definition),
                "is_output_field": self._is_output_field(field_definition)
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_FIELD_{field_name}",
                chunk_type="bms_field",
                content=match.group(0),
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _classify_screen_purpose(self, map_name: str) -> str:
        """Classify the business purpose of a BMS map"""
        name_upper = map_name.upper()
        
        if any(pattern in name_upper for pattern in ['MENU', 'MAIN', 'INDEX']):
            return 'navigation'
        elif any(pattern in name_upper for pattern in ['INQUIRY', 'DISPLAY', 'VIEW']):
            return 'data_display'
        elif any(pattern in name_upper for pattern in ['UPDATE', 'MODIFY', 'CHANGE']):
            return 'data_modification'
        elif any(pattern in name_upper for pattern in ['ADD', 'CREATE', 'NEW']):
            return 'data_entry'
        elif any(pattern in name_upper for pattern in ['DELETE', 'REMOVE']):
            return 'data_deletion'
        else:
            return 'general_processing'

    def _classify_field_purpose(self, field_name: str) -> str:
        """Classify the business purpose of a BMS field"""
        name_upper = field_name.upper()
        
        if any(pattern in name_upper for pattern in ['MSG', 'MESSAGE', 'ERROR']):
            return 'message_display'
        elif any(pattern in name_upper for pattern in ['TITLE', 'HEADER', 'LITERAL']):
            return 'label'
        elif any(pattern in name_upper for pattern in ['INPUT', 'ENTRY', 'KEY']):
            return 'data_input'
        elif any(pattern in name_upper for pattern in ['OUTPUT', 'DISPLAY', 'SHOW']):
            return 'data_output'
        elif any(pattern in name_upper for pattern in ['PF', 'FUNCTION', 'CMD']):
            return 'function_key'
        else:
            return 'data_field'

    def _analyze_field_validation(self, field_definition: str) -> List[str]:
        """Analyze validation rules for BMS field"""
        validations = []
        def_upper = field_definition.upper()
        
        if 'PICIN=' in def_upper:
            validations.append('input_format_validation')
        
        if 'PICOUT=' in def_upper:
            validations.append('output_format_validation')
        
        if 'ATTRB=' in def_upper:
            if 'UNPROT' in def_upper:
                validations.append('input_allowed')
            if 'PROT' in def_upper:
                validations.append('protected_field')
            if 'NUM' in def_upper:
                validations.append('numeric_only')
        
        return validations

    async def _parse_copybook_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse copybook with business rule validation"""
        copybook_name = self._extract_program_name(content, Path(filename))
        
        # Parse overall structure
        structure_chunk = await self._parse_copybook_structure(content, copybook_name)
        chunks = [structure_chunk] if structure_chunk else []
        
        # Parse individual fields with enhanced business context
        field_chunks = await self._parse_copybook_fields_enhanced(content, copybook_name)
        chunks.extend(field_chunks)
        
        # Parse record layouts
        record_chunks = await self._parse_copybook_records(content, copybook_name)
        chunks.extend(record_chunks)
        
        return chunks

    async def _parse_copybook_structure(self, content: str, copybook_name: str) -> Optional[CodeChunk]:
        """Parse overall copybook structure"""
        business_context = {
            'copybook_purpose': 'data_definition',
            'reusability': 'shared_structure',
            'maintenance_impact': 'high'
        }
        
        field_analysis = await self._analyze_fields_comprehensive(content)
        
        metadata = {
            "copybook_type": "data_structure",
            "field_analysis": field_analysis,
            "total_fields": field_analysis["statistics"]["total_fields"],
            "record_layouts": self._extract_record_layouts(content),
            "occurs_tables": self._extract_occurs_tables(content),
            "redefines_structures": self._extract_redefines_structures(content)
        }
        
        return CodeChunk(
            program_name=copybook_name,
            chunk_id=f"{copybook_name}_STRUCTURE",
            chunk_type="copybook_structure",
            content=content,
            metadata=metadata,
            business_context=business_context,
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )

    async def _parse_copybook_fields_enhanced(self, content: str, copybook_name: str) -> List[CodeChunk]:
        """Parse copybook fields with enhanced business context"""
        chunks = []
        
        data_matches = self.cobol_patterns['data_item'].finditer(content)
        
        for match in data_matches:
            level = match.group(1)
            name = match.group(2)
            definition = match.group(3)
            
            if match.group(0).strip().startswith('*'):
                continue
            
            if int(level) <= 49:
                business_context = {
                    'field_category': self._categorize_data_item(name, definition),
                    'business_domain': self._infer_business_domain(name),
                    'reuse_pattern': 'copybook_member',
                    'impact_scope': 'multiple_programs'
                }
                
                metadata = {
                    "level": int(level),
                    "field_name": name,
                    "definition": definition,
                    "pic_clause": self._extract_pic_clause(definition),
                    "usage": self._extract_usage_clause(definition),
                    "occurs": self._extract_occurs_info(definition),
                    "redefines": self._extract_redefines_info(definition)
                }
                
                chunk = CodeChunk(
                    program_name=copybook_name,
                    chunk_id=f"{copybook_name}_FIELD_{name}",
                    chunk_type="copybook_field",
                    content=match.group(0),
                    metadata=metadata,
                    business_context=business_context,
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_copybook_records(self, content: str, copybook_name: str) -> List[CodeChunk]:
        """Parse record-level structures in copybook"""
        chunks = []
        
        record_pattern = re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*)(.*?)(?=^\s*01\s|\Z)', 
                                  re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(record_pattern.finditer(content)):
            record_name = match.group(1)
            record_content = match.group(0)
            
            business_context = {
                'record_purpose': self._classify_record_purpose(record_name),
                'data_organization': 'hierarchical_structure',
                'usage_pattern': 'shared_definition'
            }
            
            metadata = await self._analyze_copybook_record(record_content, record_name)
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_RECORD_{record_name}",
                chunk_type="copybook_record",
                content=record_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _classify_record_purpose(self, record_name: str) -> str:
        """Classify the business purpose of a copybook record"""
        name_upper = record_name.upper()
        
        if any(pattern in name_upper for pattern in ['CUSTOMER', 'CLIENT']):
            return 'customer_data'
        elif any(pattern in name_upper for pattern in ['ACCOUNT', 'ACCT']):
            return 'account_data'
        elif any(pattern in name_upper for pattern in ['TRANSACTION', 'TRANS']):
            return 'transaction_data'
        elif any(pattern in name_upper for pattern in ['PRODUCT', 'ITEM']):
            return 'product_data'
        elif any(pattern in name_upper for pattern in ['HEADER', 'HDR']):
            return 'header_information'
        elif any(pattern in name_upper for pattern in ['DETAIL', 'DTL']):
            return 'detail_information'
        elif any(pattern in name_upper for pattern in ['TOTAL', 'SUM']):
            return 'summary_information'
        else:
            return 'business_data'

    async def _analyze_copybook_record(self, content: str, record_name: str) -> Dict[str, Any]:
        """Analyze copybook record structure"""
        metadata = {
            "record_name": record_name,
            "fields": self._extract_record_fields(content),
            "total_length": self._calculate_record_length(content),
            "purpose": f"Data record {record_name}",
            "complexity": self._assess_record_complexity(content)
        }
        
        return metadata

    def _assess_record_complexity(self, content: str) -> str:
        """Assess complexity of record structure"""
        field_count = len(self.cobol_patterns['data_item'].findall(content))
        occurs_count = len(self.cobol_patterns['occurs_clause'].findall(content))
        redefines_count = len(self.cobol_patterns['redefines'].findall(content))
        
        complexity_score = field_count + (occurs_count * 2) + (redefines_count * 3)
        
        if complexity_score > 20:
            return 'high'
        elif complexity_score > 10:
            return 'medium'
        else:
            return 'low'

    async def _parse_generic(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse unknown file type generically"""
        program_name = self._extract_program_name(content, Path(filename))
        
        chunks = []
        
        if 'EXEC' in content.upper():
            exec_chunks = await self._parse_generic_exec_statements(content, program_name)
            chunks.extend(exec_chunks)
        
        if not chunks:
            business_context = {
                'file_classification': 'unknown',
                'processing_approach': 'generic',
                'analysis_confidence': 'low'
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_GENERIC",
                chunk_type="generic",
                content=content,
                metadata={"file_type": "unknown", "analysis": "Generic file processing"},
                business_context=business_context,
                line_start=0,
                line_end=len(content.split('\n')) - 1
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_generic_exec_statements(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse generic EXEC statements in unknown files"""
        chunks = []
        
        exec_pattern = re.compile(r'EXEC\s+(\w+)\s+(.*?)(?=EXEC|\Z)', re.IGNORECASE | re.DOTALL)
        
        for i, match in enumerate(exec_pattern.finditer(content)):
            exec_type = match.group(1)
            exec_content = match.group(2)
            
            business_context = {
                'execution_type': exec_type.lower(),
                'command_purpose': 'system_operation',
                'integration_point': 'external_system'
            }
            
            metadata = {
                "exec_type": exec_type,
                "content_preview": exec_content[:100],
                "statement_type": "generic_exec"
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_EXEC_{exec_type}_{i+1}",
                chunk_type="generic_exec",
                content=match.group(0),
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    # Keep remaining helper methods unchanged for backward compatibility
    def _extract_record_layouts(self, content: str) -> List[Dict[str, Any]]:
        """Extract record layout information - UNCHANGED"""
        layouts = []
        
        record_pattern = re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*)(.*?)')
                                           
    # Enhanced metadata generation with business violations
    async def _generate_metadata_enhanced(self, chunks: List[CodeChunk], file_type: str, business_violations: List = None) -> Dict[str, Any]:
        """Generate enhanced metadata with business context"""
        metadata = {
            "total_chunks": len(chunks),
            "file_type": file_type,
            "chunk_types": {},
            "complexity_metrics": {},
            "business_violations": len(business_violations) if business_violations else 0,
            "processing_timestamp": dt.now().isoformat()
        }
        
        # Count chunk types
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            metadata["chunk_types"][chunk_type] = metadata["chunk_types"].get(chunk_type, 0) + 1
        
        # Enhanced complexity metrics with business context
        if file_type == 'cobol':
            metadata["complexity_metrics"] = {
                "total_paragraphs": metadata["chunk_types"].get("paragraph", 0),
                "total_sql_blocks": metadata["chunk_types"].get("sql_block", 0),
                "total_cics_commands": metadata["chunk_types"].get("cics_command", 0),
                "total_file_operations": metadata["chunk_types"].get("file_operation", 0),
                "total_data_items": metadata["chunk_types"].get("data_item", 0),
                "has_complex_logic": any(chunk.chunk_type in ["if_statement", "evaluate_statement", "perform_statement"] for chunk in chunks),
                "business_complexity_score": self._calculate_business_complexity_score(chunks),
                "maintainability_score": self._calculate_maintainability_score(chunks)
            }
        elif file_type == 'jcl':
            metadata["complexity_metrics"] = {
                "total_steps": metadata["chunk_types"].get("jcl_step", 0),
                "total_dd_statements": metadata["chunk_types"].get("jcl_dd_statement", 0),
                "has_procedures": metadata["chunk_types"].get("jcl_procedure", 0) > 0,
                "has_conditional_logic": metadata["chunk_types"].get("jcl_conditional", 0) > 0,
                "execution_complexity": self._calculate_jcl_execution_complexity(chunks)
            }
        
        # Business context aggregation
        business_functions = set()
        data_categories = set()
        performance_indicators = []
        
        for chunk in chunks:
            if chunk.business_context:
                if 'business_function' in chunk.business_context:
                    business_functions.add(chunk.business_context['business_function'])
                if 'data_category' in chunk.business_context:
                    data_categories.add(chunk.business_context['data_category'])
                if 'performance_impact' in chunk.business_context:
                    performance_indicators.append(chunk.business_context['performance_impact'])
        
        metadata.update({
            "business_functions": list(business_functions),
            "data_categories": list(data_categories),
            "performance_summary": self._summarize_performance_indicators(performance_indicators)
        })
        
        return metadata

    def _calculate_business_complexity_score(self, chunks: List[CodeChunk]) -> int:
        """Calculate business complexity score based on business context"""
        score = 0
        
        for chunk in chunks:
            # Base complexity from chunk type
            if chunk.chunk_type == "paragraph":
                score += 2
            elif chunk.chunk_type in ["if_statement", "evaluate_statement"]:
                score += 3
            elif chunk.chunk_type == "perform_statement":
                score += 2
            elif chunk.chunk_type == "sql_block":
                score += 4
            elif chunk.chunk_type == "cics_command":
                score += 2
            
            # Additional complexity from business context
            if chunk.business_context:
                if chunk.business_context.get('control_complexity', 0) > 5:
                    score += 2
                if chunk.business_context.get('business_function') == 'error_handling':
                    score += 1
                if 'financial' in chunk.business_context.get('data_category', ''):
                    score += 1
        
        return min(score, 100)

    # Helper methods for business analysis
    def _extract_identification_metadata(self, content: str) -> Dict[str, str]:
        """Extract identification division metadata"""
        metadata = {}
        
        patterns = {
            'author': self.cobol_patterns['author'],
            'date_written': self.cobol_patterns['date_written'],
            'date_compiled': self.cobol_patterns['date_compiled']
        }
        
        for key, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                metadata[key] = match.group(1).strip()
        
        return metadata

    def _extract_file_assignments(self, content: str) -> List[Dict[str, str]]:
        """Extract file assignments from environment division"""
        assignments = []
        
        select_matches = self.cobol_patterns['select_statement'].finditer(content)
        for match in select_matches:
            assignments.append({
                'logical_file': match.group(1),
                'physical_file': match.group(2)
            })
        
        return assignments

    def _extract_data_structures_summary(self, content: str) -> Dict[str, Any]:
        """Extract summary of data structures"""
        summary = {
            'record_layouts': 0,
            'table_structures': 0,
            'redefines_count': 0,
            'total_fields': 0
        }
        
        # Count 01 level items (records)
        record_pattern = re.compile(r'^\s*01\s+', re.MULTILINE | re.IGNORECASE)
        summary['record_layouts'] = len(record_pattern.findall(content))
        
        # Count OCCURS clauses (tables)
        occurs_matches = self.cobol_patterns['occurs_clause'].findall(content)
        summary['table_structures'] = len(occurs_matches)
        
        # Count REDEFINES
        redefines_matches = self.cobol_patterns['redefines'].findall(content)
        summary['redefines_count'] = len(redefines_matches)
        
        # Count total data items
        data_matches = self.cobol_patterns['data_item'].findall(content)
        summary['total_fields'] = len(data_matches)
        
        return summary

    def _extract_entry_points(self, content: str) -> List[str]:
        """Extract procedure division entry points"""
        entry_points = []
        
        # USING clause in PROCEDURE DIVISION
        proc_match = self.cobol_patterns['procedure_division'].search(content)
        if proc_match and proc_match.group(1):
            using_params = proc_match.group(1).strip().split()
            entry_points.extend(using_params)
        
        # First paragraph after PROCEDURE DIVISION
        paragraphs = self.cobol_patterns['paragraph'].findall(content)
        if paragraphs:
            entry_points.append(paragraphs[0])
        
        return entry_points

    def _extract_business_functions(self, content: str) -> List[str]:
        """Extract business functions from procedure division"""
        functions = set()
        
        # Analyze paragraph names for business functions
        paragraphs = self.cobol_patterns['paragraph'].findall(content)
        for para in paragraphs:
            function = self._infer_business_function(para, "")
            functions.add(function)
        
        return list(functions)

    def _find_section_end(self, content: str, start_pos: int, section_names: List[str]) -> int:
        """Find the end of a section"""
        end_pos = len(content)
        
        for section_name in section_names:
            if section_name in self.cobol_patterns:
                pattern = self.cobol_patterns[section_name]
                match = pattern.search(content, start_pos)
                if match and match.start() < end_pos:
                    end_pos = match.start()
        
        return end_pos

    def _extract_complete_perform_statement(self, content: str, start_pos: int) -> str:
        """Extract complete PERFORM statement handling all variations"""
        lines = content[start_pos:].split('\n')
        perform_lines = []
        
        for line in lines:
            perform_lines.append(line)
            
            # Check for end of PERFORM
            if 'END-PERFORM' in line.upper():
                break
            elif line.strip().endswith('.') and len(perform_lines) > 1:
                break
            elif (len(perform_lines) == 1 and 
                  not any(keyword in line.upper() for keyword in ['UNTIL', 'VARYING', 'TIMES', 'THRU', 'THROUGH'])):
                break
        
        return '\n'.join(perform_lines)

    def _extract_target_paragraph(self, match) -> str:
        """Extract target paragraph from PERFORM match"""
        content = match.group(0)
        
        # Simple PERFORM
        simple_match = re.search(r'PERFORM\s+([A-Z0-9][A-Z0-9-]*)', content, re.IGNORECASE)
        if simple_match:
            return simple_match.group(1)
        
        return "UNKNOWN"

    def _analyze_loop_characteristics(self, content: str, perform_type: str) -> Dict[str, Any]:
        """Analyze loop characteristics of PERFORM statement"""
        characteristics = {
            'is_loop': False,
            'loop_type': 'none',
            'complexity': 'low'
        }
        
        if 'until' in perform_type:
            characteristics.update({
                'is_loop': True,
                'loop_type': 'conditional',
                'complexity': 'medium'
            })
        elif 'varying' in perform_type:
            characteristics.update({
                'is_loop': True,
                'loop_type': 'iterative',
                'complexity': 'high'
            })
        elif 'times' in perform_type:
            characteristics.update({
                'is_loop': True,
                'loop_type': 'counted',
                'complexity': 'low'
            })
        
        return characteristics

    def _determine_execution_pattern(self, perform_type: str, content: str) -> str:
        """Determine execution pattern of PERFORM"""
        if 'inline' in perform_type:
            return 'inline_execution'
        elif 'thru' in perform_type:
            return 'range_execution'
        elif any(keyword in content.upper() for keyword in ['UNTIL', 'VARYING', 'TIMES']):
            return 'iterative_execution'
        else:
            return 'single_execution'

    def _assess_perform_performance(self, content: str, perform_type: str) -> Dict[str, str]:
        """Assess performance implications of PERFORM statement"""
        implications = {
            'cpu_impact': 'low',
            'memory_impact': 'low',
            'io_impact': 'unknown'
        }
        
        if 'varying' in perform_type or 'until' in perform_type:
            implications['cpu_impact'] = 'medium'
            
        if 'thru' in perform_type:
            implications['cpu_impact'] = 'high'  # Multiple paragraph execution
            
        # Check for I/O operations in content
        if any(op in content.upper() for op in ['READ', 'WRITE', 'EXEC SQL']):
            implications['io_impact'] = 'high'
        
        return implications

    def _measure_control_complexity(self, content: str, perform_type: str) -> int:
        """Measure control complexity of PERFORM statement"""
        complexity = 1  # Base complexity
        
        # Add complexity for loop types
        if 'until' in perform_type:
            complexity += 2
        elif 'varying' in perform_type:
            complexity += 3
        elif 'times' in perform_type:
            complexity += 1
        
        # Add complexity for nested control structures
        if_count = content.upper().count('IF ')
        evaluate_count = content.upper().count('EVALUATE ')
        nested_perform_count = content.upper().count('PERFORM ') - 1  # Exclude this PERFORM
        
        complexity += if_count + (evaluate_count * 2) + nested_perform_count
        
        return complexity

    def _assess_perform_maintainability(self, content: str, perform_type: str) -> int:
        """Assess maintainability score of PERFORM statement (1-10)"""
        score = 8  # Start with good score
        
        # Reduce score for complex types
        if 'varying' in perform_type:
            score -= 2
        elif 'until' in perform_type:
            score -= 1
        
        # Reduce score for long content
        if len(content) > 500:
            score -= 1
        if len(content) > 1000:
            score -= 1
        
        # Reduce score for complex nesting
        nesting_level = max(content.upper().count('IF '), content.upper().count('EVALUATE '))
        if nesting_level > 3:
            score -= 2
        elif nesting_level > 1:
            score -= 1
        
        return max(1, score)

    def _determine_host_var_context(self, sql_content: str, position: int) -> str:
        """Determine context of host variable usage"""
        before_text = sql_content[:position].upper()
        after_text = sql_content[position:position+50].upper()
        
        if 'INTO' in before_text[-20:]:
            return 'output'
        elif any(clause in before_text[-50:] for clause in ['WHERE', 'SET', 'VALUES']):
            return 'input'
        elif 'FROM' in before_text[-20:]:
            return 'input'
        else:
            return 'unknown'

    def _extract_cobol_data_definitions(self, content: str) -> Dict[str, Dict]:
        """Extract COBOL data definitions for host variable validation"""
        data_items = {}
        
        matches = self.cobol_patterns['data_item'].finditer(content)
        for match in matches:
            level = match.group(1)
            name = match.group(2).upper()
            definition = match.group(3)
            
            data_items[name] = {
                'level': level,
                'definition': definition,
                'pic': self._extract_pic_clause(definition),
                'usage': self._extract_usage_clause(definition)
            }
        
        return data_items

    def _has_required_params(self, params: str, required: List[str]) -> bool:
        """Check if CICS command has required parameters"""
        params_upper = params.upper()
        return all(param in params_upper for param in required)

    def _extract_parameter_value(self, params: str, param_name: str) -> Optional[str]:
        """Extract parameter value from CICS command parameters"""
        pattern = re.compile(f'{param_name}\\(([^)]+)\\)', re.IGNORECASE)
        match = pattern.search(params)
        return match.group(1) if match else None

    def _extract_jcl_step_complete(self, content: str, start_pos: int, all_matches: List, current_index: int) -> str:
        """Extract complete JCL step including all DD statements"""
        if current_index + 1 < len(all_matches):
            end_pos = all_matches[current_index + 1].start()
        else:
            end_pos = len(content)
        
        return content[start_pos:end_pos].strip()

    async def _analyze_jcl_step_business_context(self, content: str, step_name: str, sequence: int) -> Dict[str, Any]:
        """Analyze business context of JCL step"""
        context = {
            'step_purpose': self._infer_jcl_step_purpose(step_name, content),
            'execution_sequence': sequence + 1,
            'dependencies': self._extract_step_dependencies(content),
            'conditional_logic': self._extract_conditional_logic(content),
            'resources': self._extract_resource_requirements(content),
            'data_flow': self._analyze_jcl_data_flow(content)
        }
        
        return context

    def _infer_jcl_step_purpose(self, step_name: str, content: str) -> str:
        """Infer business purpose of JCL step"""
        name_upper = step_name.upper()
        content_upper = content.upper()
        
        if any(pattern in name_upper for pattern in ['SORT', 'MERGE']):
            return 'data_sorting'
        elif any(pattern in name_upper for pattern in ['COPY', 'BACKUP']):
            return 'data_backup'
        elif any(pattern in name_upper for pattern in ['LOAD', 'UNLOAD']):
            return 'data_transfer'
        elif 'PROC' in content_upper:
            return 'procedure_execution'
        elif 'PGM=' in content_upper:
            return 'program_execution'
        else:
            return 'batch_processing'

    def _extract_step_dependencies(self, content: str) -> List[str]:
        """Extract step dependencies from JCL content"""
        dependencies = []
        
        # Check for COND parameters that reference other steps
        cond_matches = re.finditer(r'COND=\([^)]*,([A-Z0-9]+)\)', content, re.IGNORECASE)
        for match in cond_matches:
            dependencies.append(match.group(1))
        
        return dependencies

    def _extract_conditional_logic(self, content: str) -> Dict[str, Any]:
        """Extract conditional execution logic from JCL step"""
        logic = {
            'has_conditions': False,
            'conditions': [],
            'execution_type': 'unconditional'
        }
        
        # Check for COND parameter
        cond_match = self.jcl_patterns['cond_parameter'].search(content)
        if cond_match:
            logic.update({
                'has_conditions': True,
                'conditions': [cond_match.group(1)],
                'execution_type': 'conditional'
            })
        
        return logic

    def _extract_resource_requirements(self, content: str) -> Dict[str, str]:
        """Extract resource requirements from JCL step"""
        resources = {}
        
        # Extract common resource parameters
        resource_patterns = {
            'REGION': re.compile(r'REGION=([^,\s]+)', re.IGNORECASE),
            'TIME': re.compile(r'TIME=([^,\s]+)', re.IGNORECASE),
            'CLASS': re.compile(r'CLASS=([^,\s]+)', re.IGNORECASE)
        }
        
        for resource, pattern in resource_patterns.items():
            match = pattern.search(content)
            if match:
                resources[resource] = match.group(1)
        
        return resources

    def _analyze_jcl_data_flow(self, content: str) -> Dict[str, List[str]]:
        """Analyze data flow in JCL step"""
        data_flow = {
            'input_datasets': [],
            'output_datasets': [],
            'temporary_datasets': []
        }
        
        # Find DD statements and classify datasets
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            if 'DSN=' in dd_line.upper():
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_line, re.IGNORECASE)
                if dsn_match:
                    dsn = dsn_match.group(1)
                    
                    if 'DISP=(NEW' in dd_line.upper():
                        data_flow['output_datasets'].append(dsn)
                    elif 'DISP=(OLD' in dd_line.upper() or 'DISP=SHR' in dd_line.upper():
                        data_flow['input_datasets'].append(dsn)
                    elif '&&' in dsn:  # Temporary dataset
                        data_flow['temporary_datasets'].append(dsn)
        
        return data_flow

    def _calculate_jcl_execution_complexity(self, chunks: List[CodeChunk]) -> int:
        """Calculate JCL execution complexity"""
        complexity = 0
        
        step_count = len([c for c in chunks if c.chunk_type == 'jcl_step'])
        complexity += step_count * 2
        
        conditional_steps = len([c for c in chunks if c.business_context and 
                               c.business_context.get('conditional_logic', {}).get('has_conditions', False)])
        complexity += conditional_steps * 3
        
        return min(complexity, 100)

    def _summarize_performance_indicators(self, indicators: List) -> Dict[str, Any]:
        """Summarize performance indicators across chunks"""
        summary = {
            'high_impact_operations': 0,
            'io_operations': 0,
            'cpu_intensive_operations': 0,
            'overall_performance_risk': 'low'
        }
        
        for indicator in indicators:
            if isinstance(indicator, dict):
                if indicator.get('cpu_impact') == 'high':
                    summary['cpu_intensive_operations'] += 1
                if indicator.get('io_impact') == 'high':
                    summary['io_operations'] += 1
        
        # Determine overall risk
        total_high_impact = summary['cpu_intensive_operations'] + summary['io_operations']
        if total_high_impact > 10:
            summary['overall_performance_risk'] = 'high'
        elif total_high_impact > 5:
            summary['overall_performance_risk'] = 'medium'
        
        return summary

    def _calculate_maintainability_score(self, chunks: List[CodeChunk]) -> int:
        """Calculate overall maintainability score"""
        scores = []
        
        for chunk in chunks:
            if chunk.business_context and 'maintainability_score' in chunk.business_context:
                scores.append(chunk.business_context['maintainability_score'])
            elif chunk.metadata and 'maintainability_score' in chunk.metadata:
                scores.append(chunk.metadata['maintainability_score'])
        
        return int(sum(scores) / len(scores)) if scores else 7

    # Keep all remaining helper methods unchanged for regex extraction
    def _extract_field_names(self, content: str) -> List[str]:
        """Extract field names from COBOL code - UNCHANGED"""
        fields = set()
        
        field_pattern = re.compile(r'\b\d+\s+([A-Z][A-Z0-9-]*)\s+PIC', re.IGNORECASE)
        fields.update(match.group(1) for match in field_pattern.finditer(content))
        
        move_pattern = re.compile(r'\bMOVE\s+([A-Z][A-Z0-9-]*)\s+TO\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in move_pattern.finditer(content):
            fields.add(match.group(1))
            fields.add(match.group(2))
        
        if_pattern = re.compile(r'\bIF\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        fields.update(match.group(1) for match in if_pattern.finditer(content))
        
        return list(fields)

    def _extract_file_operations(self, content: str) -> List[str]:
        """Extract file operations - UNCHANGED"""
        ops = []
        
        file_op_patterns = [
            (r'\b(OPEN)\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*)', lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
            (r'\b(READ|WRITE|REWRITE|DELETE|CLOSE)\s+([A-Z][A-Z0-9-]*)', lambda m: f"{m.group(1)} {m.group(2)}")
        ]
        
        for pattern, formatter in file_op_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                ops.append(formatter(match))
        
        return ops

    def _extract_sql_operations(self, content: str) -> List[str]:
        """Extract SQL operations - UNCHANGED"""
        ops = set()
        sql_pattern = re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|GRANT|REVOKE)\b', re.IGNORECASE)
        ops.update(match.group(1).upper() for match in sql_pattern.finditer(content))
        return list(ops)

    def _extract_perform_statements(self, content: str) -> List[str]:
        """Extract PERFORM statements - UNCHANGED"""
        performs = set()
        
        simple_pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        performs.update(match.group(1) for match in simple_pattern.finditer(content))
        
        thru_pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in thru_pattern.finditer(content):
            performs.add(f"{match.group(1)} THRU {match.group(2)}")
        
        return list(performs)

    def _extract_error_handling_patterns(self, content: str) -> List[str]:
        """Extract error handling patterns - UNCHANGED"""
        patterns = []
        content_upper = content.upper()
        
        error_indicators = [
            ('ON SIZE ERROR', 'SIZE_ERROR_HANDLING'),
            ('AT END', 'END_OF_FILE_HANDLING'),
            ('INVALID KEY', 'INVALID_KEY_HANDLING'),
            ('NOT ON SIZE ERROR', 'NO_SIZE_ERROR_PATH'),
            ('NOT AT END', 'NOT_AT_END_PATH'),
            ('NOT INVALID KEY', 'VALID_KEY_PATH'),
            ('OVERFLOW', 'OVERFLOW_HANDLING'),
            ('EXCEPTION', 'EXCEPTION_HANDLING')
        ]
        
        for indicator, pattern_name in error_indicators:
            if indicator in content_upper:
                patterns.append(pattern_name)
        
        return patterns

    # Keep all other helper methods for backward compatibility
    def _extract_sql_type(self, sql_content: str) -> str:
        """Extract SQL operation type - UNCHANGED"""
        sql_upper = sql_content.upper().strip()
        sql_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'GRANT', 'REVOKE']
        
        for sql_type in sql_types:
            if sql_upper.startswith(sql_type):
                return sql_type
        
        return 'UNKNOWN'

    def _extract_table_names(self, sql_content: str) -> List[str]:
        """Extract table names from SQL - UNCHANGED"""
        tables = set()
        
        patterns = [
            re.compile(r'\bFROM\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            re.compile(r'\bUPDATE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            re.compile(r'\bINSERT\s+INTO\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            re.compile(r'\bJOIN\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        ]
        
        for pattern in patterns:
            tables.update(match.group(1) for match in pattern.finditer(sql_content))
        
        return list(tables)

    def _categorize_cics_command(self, command_type: str) -> str:
        """Categorize CICS command - UNCHANGED"""
        terminal_commands = ['cics_send_map', 'cics_receive_map', 'cics_send_text', 'cics_receive']
        file_commands = ['cics_read', 'cics_write', 'cics_rewrite', 'cics_delete', 'cics_startbr', 'cics_readnext']
        program_commands = ['cics_link', 'cics_xctl', 'cics_load', 'cics_return']
        storage_commands = ['cics_getmain', 'cics_freemain']
        
        if command_type in terminal_commands:
            return "terminal"
        elif command_type in file_commands:
            return "file"
        elif command_type in program_commands:
            return "program_control"
        elif command_type in storage_commands:
            return "storage"
        else:
            return "other"

    def _extract_cics_parameters(self, params: str) -> Dict[str, str]:
        """Extract CICS command parameters - UNCHANGED"""
        param_dict = {}
        
        param_patterns = {
            'FILE': re.compile(r'FILE\(([^)]+)\)', re.IGNORECASE),
            'MAP': re.compile(r'MAP\(([^)]+)\)', re.IGNORECASE),
            'MAPSET': re.compile(r'MAPSET\(([^)]+)\)', re.IGNORECASE),
            'PROGRAM': re.compile(r'PROGRAM\(([^)]+)\)', re.IGNORECASE),
            'LENGTH': re.compile(r'LENGTH\(([^)]+)\)', re.IGNORECASE),
            'INTO': re.compile(r'INTO\(([^)]+)\)', re.IGNORECASE),
            'FROM': re.compile(r'FROM\(([^)]+)\)', re.IGNORECASE),
            'RIDFLD': re.compile(r'RIDFLD\(([^)]+)\)', re.IGNORECASE)
        }
        
        for param_name, pattern in param_patterns.items():
            match = pattern.search(params)
            if match:
                param_dict[param_name] = match.group(1)
        
        return param_dict

    def _extract_cics_resource(self, params: str) -> str:
        """Extract main resource from CICS parameters - UNCHANGED"""
        resource_patterns = ['FILE', 'MAP', 'PROGRAM', 'QUEUE']
        
        for resource_type in resource_patterns:
            pattern = re.compile(f'{resource_type}\\(([^)]+)\\)', re.IGNORECASE)
            match = pattern.search(params)
            if match:
                return match.group(1)
        
        return "UNKNOWN"

    def _extract_exec_program(self, content: str) -> str:
        """Extract program name from EXEC statement - UNCHANGED"""
        exec_patterns = [
            re.compile(r'EXEC\s+PGM=([A-Z0-9]+)', re.IGNORECASE),
            re.compile(r'EXEC\s+([A-Z0-9]+)', re.IGNORECASE)
        ]
        
        for pattern in exec_patterns:
            match = pattern.search(content)
            if match:
                return match.group(1)
        
        return "UNKNOWN"

    def _extract_input_datasets_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed input dataset information - UNCHANGED"""
        datasets = []
        
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            if any(indicator in dd_line.upper() for indicator in ['DSN=', 'DISP=SHR', 'DISP=OLD']):
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_line, re.IGNORECASE)
                disp_match = re.search(r'DISP=\(([^,)]+)', dd_line, re.IGNORECASE)
                
                datasets.append({
                    "dd_name": dd_name,
                    "dsn": dsn_match.group(1) if dsn_match else "UNKNOWN",
                    "disposition": disp_match.group(1) if disp_match else "UNKNOWN",
                    "purpose": "input_data"
                })
        
        return datasets

    def _extract_output_datasets_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed output dataset information - UNCHANGED"""
        datasets = []
        
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            if any(indicator in dd_line.upper() for indicator in ['DISP=(NEW', 'DISP=(,CATLG', 'DISP=(,DELETE']):
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_line, re.IGNORECASE)
                disp_match = re.search(r'DISP=\(([^,)]+)', dd_line, re.IGNORECASE)
                
                datasets.append({
                    "dd_name": dd_name,
                    "dsn": dsn_match.group(1) if dsn_match else "UNKNOWN",
                    "disposition": disp_match.group(1) if disp_match else "UNKNOWN",
                    "purpose": "output_data"
                })
        
        return datasets

    def _extract_parameters_detailed(self, content: str) -> Dict[str, str]:
        """Extract detailed parameters - UNCHANGED"""
        params = {}
        
        param_patterns = {
            'PARM': re.compile(r'PARM=([^,\s]+)', re.IGNORECASE),
            'REGION': re.compile(r'REGION=([^,\s]+)', re.IGNORECASE),
            'TIME': re.compile(r'TIME=([^,\s]+)', re.IGNORECASE)
        }
        
        for param_name, pattern in param_patterns.items():
            match = pattern.search(content)
            if match:
                params[param_name] = match.group(1)
        
        return params

    # Additional helper methods for enhanced business analysis
    def _extract_pic_clause(self, definition: str) -> Optional[str]:
        """Extract PIC clause from field definition - ENHANCED"""
        match = self.cobol_patterns['pic_clause'].search(definition)
        if match:
            return match.group(1) or match.group(2)
        return None

    def _extract_usage_clause(self, definition: str) -> str:
        """Extract USAGE clause from field definition - ENHANCED"""
        match = self.cobol_patterns['usage_clause'].search(definition)
        return match.group(1) if match else "DISPLAY"

    def _extract_value_clause(self, definition: str) -> Optional[str]:
        """Extract VALUE clause from field definition - UNCHANGED"""
        match = self.cobol_patterns['value_clause'].search(definition)
        return match.group(1) if match else None

    def _extract_occurs_info(self, definition: str) -> Optional[Dict[str, Any]]:
        """Extract OCCURS clause information - ENHANCED"""
        match = self.cobol_patterns['occurs_clause'].search(definition)
        if match:
            min_occurs = int(match.group(1))
            max_occurs = int(match.group(2)) if match.group(2) else min_occurs
            depending_field = match.group(3) if match.group(3) else None
            indexed_field = match.group(4) if match.group(4) else None
            
            return {
                "min_occurs": min_occurs,
                "max_occurs": max_occurs,
                "is_variable": max_occurs != min_occurs or depending_field is not None,
                "depending_on": depending_field,
                "indexed_by": indexed_field
            }
        return None

    def _extract_redefines_info(self, definition: str) -> Optional[str]:
        """Extract REDEFINES information - UNCHANGED"""
        match = self.cobol_patterns['redefines'].search(definition)
        return match.group(1) if match else None

    def _determine_data_type_enhanced(self, definition: str) -> str:
        """Enhanced data type determination"""
        pic_clause = self._extract_pic_clause(definition)
        usage = self._extract_usage_clause(definition)
        
        if not pic_clause:
            return "group"
        
        pic_upper = pic_clause.upper()
        
        # Enhanced type detection
        if '9' in pic_upper:
            if 'V' in pic_upper or '.' in pic_upper:
                if usage in ['COMP-3', 'PACKED-DECIMAL']:
                    return "packed_decimal"
                elif usage in ['COMP', 'COMP-4', 'BINARY']:
                    return "binary_decimal"
                else:
                    return "display_decimal"
            else:
                if usage in ['COMP', 'COMP-4', 'BINARY']:
                    return "binary_integer"
                elif usage == 'COMP-3':
                    return "packed_integer"
                else:
                    return "display_integer"
        elif 'X' in pic_upper:
            return "alphanumeric"
        elif 'A' in pic_upper:
            return "alphabetic"
        elif 'N' in pic_upper:
            return "national"
        elif 'S' in pic_upper and '9' in pic_upper:
            return "signed_numeric"
        else:
            return "special"

    def _infer_business_domain(self, name: str) -> str:
        """Infer business domain from field name"""
        name_upper = name.upper()
        
        # Financial domain indicators
        if any(pattern in name_upper for pattern in [
            'AMOUNT', 'AMT', 'BALANCE', 'BAL', 'RATE', 'INTEREST', 'PRINCIPAL',
            'PAYMENT', 'PMT', 'CHARGE', 'FEE', 'COST', 'PRICE', 'VALUE', 'VAL'
        ]):
            return 'financial'
        
        # Customer domain indicators
        if any(pattern in name_upper for pattern in [
            'CUSTOMER', 'CUST', 'CLIENT', 'MEMBER', 'ACCOUNT', 'ACCT'
        ]):
            return 'customer'
        
        # Product domain indicators
        if any(pattern in name_upper for pattern in [
            'PRODUCT', 'PROD', 'ITEM', 'SERVICE', 'POLICY', 'CONTRACT'
        ]):
            return 'product'
        
        # Transaction domain indicators
        if any(pattern in name_upper for pattern in [
            'TRANSACTION', 'TRANS', 'TXN', 'POSTING', 'ENTRY'
        ]):
            return 'transaction'
        
        # Date/Time domain indicators
        if any(pattern in name_upper for pattern in [
            'DATE', 'TIME', 'TIMESTAMP', 'YEAR', 'MONTH', 'DAY'
        ]):
            return 'temporal'
        
        # Control/Status domain indicators
        if any(pattern in name_upper for pattern in [
            'STATUS', 'FLAG', 'INDICATOR', 'IND', 'CODE', 'TYPE'
        ]):
            return 'control'
        
        return 'general'

    def _analyze_usage_pattern(self, definition: str) -> str:
        """Analyze usage pattern of data item"""
        definition_upper = definition.upper()
        
        if 'VALUE' in definition_upper:
            if any(val in definition_upper for val in ['SPACE', 'ZERO', 'LOW-VALUE', 'HIGH-VALUE']):
                return 'initialized_constant'
            else:
                return 'initialized_variable'
        elif 'OCCURS' in definition_upper:
            return 'array_table'
        elif 'REDEFINES' in definition_upper:
            return 'overlay_structure'
        elif any(usage in definition_upper for usage in ['COMP', 'BINARY', 'PACKED']):
            return 'computational'
        else:
            return 'standard_storage'

    def _extract_validation_rules(self, definition: str) -> List[str]:
        """Extract validation rules from data definition"""
        rules = []
        definition_upper = definition.upper()
        
        # Picture clause validations
        pic_clause = self._extract_pic_clause(definition)
        if pic_clause:
            if '9' in pic_clause:
                rules.append('numeric_only')
            if 'A' in pic_clause:
                rules.append('alphabetic_only')
            if 'X' in pic_clause:
                rules.append('alphanumeric')
        
        # Value clause validations
        value_clause = self._extract_value_clause(definition)
        if value_clause:
            rules.append('default_value_assigned')
        
        # OCCURS validations
        occurs_info = self._extract_occurs_info(definition)
        if occurs_info:
            rules.append('array_bounds_check')
            if occurs_info.get('depending_on'):
                rules.append('variable_length_validation')
        
        return rules

    def _classify_security_level(self, name: str, definition: str) -> str:
        """Classify security level of data item"""
        name_upper = name.upper()
        definition_upper = definition.upper()
        
        # High security indicators
        if any(pattern in name_upper for pattern in [
            'SSN', 'SOCIAL', 'PASSWORD', 'PIN', 'ACCOUNT', 'CREDIT', 'DEBIT'
        ]):
            return 'high'
        
        # Medium security indicators
        if any(pattern in name_upper for pattern in [
            'NAME', 'ADDRESS', 'PHONE', 'EMAIL', 'SALARY', 'INCOME'
        ]):
            return 'medium'
        
        # Check for PII patterns in definition
        if 'PIC' in definition_upper:
            pic = self._extract_pic_clause(definition)
            if pic and 'X' in pic:
                # Text fields might contain sensitive data
                return 'medium'
        
        return 'low'

    def _extract_data_operations(self, content: str) -> List[Dict[str, str]]:
        """Extract data operations from paragraph content"""
        operations = []
        
        # MOVE operations
        move_pattern = re.compile(r'MOVE\s+([A-Z0-9-]+)\s+TO\s+([A-Z0-9-]+)', re.IGNORECASE)
        for match in move_pattern.finditer(content):
            operations.append({
                'operation': 'MOVE',
                'source': match.group(1),
                'target': match.group(2)
            })
        
        # Arithmetic operations
        arithmetic_patterns = {
            'ADD': re.compile(r'ADD\s+([A-Z0-9-]+)\s+TO\s+([A-Z0-9-]+)', re.IGNORECASE),
            'SUBTRACT': re.compile(r'SUBTRACT\s+([A-Z0-9-]+)\s+FROM\s+([A-Z0-9-]+)', re.IGNORECASE),
            'MULTIPLY': re.compile(r'MULTIPLY\s+([A-Z0-9-]+)\s+BY\s+([A-Z0-9-]+)', re.IGNORECASE),
            'DIVIDE': re.compile(r'DIVIDE\s+([A-Z0-9-]+)\s+(?:BY|INTO)\s+([A-Z0-9-]+)', re.IGNORECASE)
        }
        
        for op_name, pattern in arithmetic_patterns.items():
            for match in pattern.finditer(content):
                operations.append({
                    'operation': op_name,
                    'operand1': match.group(1),
                    'operand2': match.group(2)
                })
        
        return operations

    def _extract_file_operations_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed file operations"""
        operations = []
        
        file_patterns = {
            'OPEN': re.compile(r'OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z0-9-]+)', re.IGNORECASE),
            'READ': re.compile(r'READ\s+([A-Z0-9-]+)', re.IGNORECASE),
            'WRITE': re.compile(r'WRITE\s+([A-Z0-9-]+)', re.IGNORECASE),
            'REWRITE': re.compile(r'REWRITE\s+([A-Z0-9-]+)', re.IGNORECASE),
            'DELETE': re.compile(r'DELETE\s+([A-Z0-9-]+)', re.IGNORECASE),
            'CLOSE': re.compile(r'CLOSE\s+([A-Z0-9-]+)', re.IGNORECASE)
        }
        
        for op_name, pattern in file_patterns.items():
            for match in pattern.finditer(content):
                if op_name == 'OPEN':
                    operations.append({
                        'operation': op_name,
                        'file': match.group(2),
                        'mode': match.group(1)
                    })
                else:
                    operations.append({
                        'operation': op_name,
                        'file': match.group(1),
                        'mode': ''
                    })
        
        return operations

    def _extract_control_flow_elements(self, content: str) -> List[Dict[str, str]]:
        """Extract control flow elements from content"""
        elements = []
        
        # IF statements
        if_pattern = re.compile(r'IF\s+([^\.]+)', re.IGNORECASE)
        for match in if_pattern.finditer(content):
            elements.append({
                'type': 'IF',
                'condition': match.group(1).strip()
            })
        
        # PERFORM statements
        perform_pattern = re.compile(r'PERFORM\s+([A-Z0-9-]+)', re.IGNORECASE)
        for match in perform_pattern.finditer(content):
            elements.append({
                'type': 'PERFORM',
                'target': match.group(1)
            })
        
        # EVALUATE statements
        evaluate_pattern = re.compile(r'EVALUATE\s+([^\.]+)', re.IGNORECASE)
        for match in evaluate_pattern.finditer(content):
            elements.append({
                'type': 'EVALUATE',
                'expression': match.group(1).strip()
            })
        
        return elements

    def _extract_error_handling_detailed(self, content: str) -> List[Dict[str, str]]:
        """Extract detailed error handling information"""
        error_handling = []
        
        error_patterns = {
            'ON SIZE ERROR': re.compile(r'ON\s+SIZE\s+ERROR\s*([^\.]*)', re.IGNORECASE),
            'AT END': re.compile(r'AT\s+END\s*([^\.]*)', re.IGNORECASE),
            'INVALID KEY': re.compile(r'INVALID\s+KEY\s*([^\.]*)', re.IGNORECASE),
            'NOT ON SIZE ERROR': re.compile(r'NOT\s+ON\s+SIZE\s+ERROR\s*([^\.]*)', re.IGNORECASE),
            'NOT AT END': re.compile(r'NOT\s+AT\s+END\s*([^\.]*)', re.IGNORECASE),
            'NOT INVALID KEY': re.compile(r'NOT\s+INVALID\s+KEY\s*([^\.]*)', re.IGNORECASE)
        }
        
        for error_type, pattern in error_patterns.items():
            for match in pattern.finditer(content):
                error_handling.append({
                    'type': error_type,
                    'action': match.group(1).strip() if match.group(1) else 'IMPLICIT'
                })
        
        return error_handling

    def _assess_performance_impact(self, content: str) -> Dict[str, str]:
        """Assess performance impact of paragraph"""
        impact = {
            'cpu_intensive': 'low',
            'io_operations': 'low',
            'memory_usage': 'low',
            'overall_impact': 'low'
        }
        
        content_upper = content.upper()
        
        # Check for CPU-intensive operations
        if any(op in content_upper for op in ['PERFORM', 'UNTIL', 'VARYING', 'COMPUTE']):
            impact['cpu_intensive'] = 'medium'
        
        if content_upper.count('PERFORM') > 5:
            impact['cpu_intensive'] = 'high'
        
        # Check for I/O operations
        io_operations = ['READ', 'WRITE', 'REWRITE', 'DELETE', 'EXEC SQL']
        io_count = sum(content_upper.count(op) for op in io_operations)
        
        if io_count > 0:
            impact['io_operations'] = 'medium'
        if io_count > 3:
            impact['io_operations'] = 'high'
        
        # Check for memory-intensive operations
        if any(op in content_upper for op in ['OCCURS', 'TABLE', 'ARRAY']):
            impact['memory_usage'] = 'medium'
        
        # Determine overall impact
        high_impacts = [v for v in impact.values() if v == 'high']
        medium_impacts = [v for v in impact.values() if v == 'medium']
        
        if high_impacts:
            impact['overall_impact'] = 'high'
        elif len(medium_impacts) >= 2:
            impact['overall_impact'] = 'medium'
        
        return impact

    def _identify_business_rules(self, content: str) -> List[str]:
        """Identify business rules implemented in paragraph"""
        rules = []
        content_upper = content.upper()
        
        # Validation rules
        if any(pattern in content_upper for pattern in ['IF', 'WHEN', 'EVALUATE']):
            rules.append('conditional_validation')
        
        # Range checks
        if any(pattern in content_upper for pattern in ['GREATER', 'LESS', 'EQUAL', '>', '<', '=']):
            rules.append('range_validation')
        
        # Required field checks
        if any(pattern in content_upper for pattern in ['SPACE', 'ZERO', 'EMPTY']):
            rules.append('required_field_validation')
        
        # Business calculations
        if any(pattern in content_upper for pattern in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'COMPUTE']):
            rules.append('business_calculation')
        
        # Data transformation
        if 'MOVE' in content_upper:
            rules.append('data_transformation')
        
        # Error processing
        if any(pattern in content_upper for pattern in ['ERROR', 'EXCEPTION', 'INVALID']):
            rules.append('error_processing')
        
        return rules

    def _calculate_paragraph_complexity(self, content: str) -> int:
        """Calculate complexity score for paragraph"""
        complexity = 1  # Base complexity
        content_upper = content.upper()
        
        # Add complexity for control structures
        complexity += content_upper.count('IF ')
        complexity += content_upper.count('EVALUATE ') * 2
        complexity += content_upper.count('PERFORM ')
        complexity += content_upper.count('WHEN ')
        
        # Add complexity for file operations
        complexity += content_upper.count('READ ')
        complexity += content_upper.count('WRITE ')
        complexity += content_upper.count('EXEC SQL') * 2
        
        # Add complexity for error handling
        complexity += content_upper.count('ON SIZE ERROR')
        complexity += content_upper.count('AT END')
        complexity += content_upper.count('INVALID KEY')
        
        return min(complexity, 20)  # Cap at 20

    def _analyze_performance_indicators(self, content: str) -> Dict[str, Any]:
        """Analyze performance indicators in paragraph"""
        indicators = {
            'loop_operations': 0,
            'file_io_operations': 0,
            'sql_operations': 0,
            'arithmetic_operations': 0,
            'performance_risk': 'low'
        }
        
        content_upper = content.upper()
        
        # Count different operation types
        indicators['loop_operations'] = content_upper.count('PERFORM UNTIL') + content_upper.count('PERFORM VARYING')
        indicators['file_io_operations'] = (content_upper.count('READ ') + content_upper.count('WRITE ') + 
                                          content_upper.count('REWRITE ') + content_upper.count('DELETE '))
        indicators['sql_operations'] = content_upper.count('EXEC SQL')
        indicators['arithmetic_operations'] = (content_upper.count('ADD ') + content_upper.count('SUBTRACT ') +
                                             content_upper.count('MULTIPLY ') + content_upper.count('DIVIDE ') +
                                             content_upper.count('COMPUTE '))
        
        # Assess overall performance risk
        total_operations = (indicators['loop_operations'] * 3 + 
                          indicators['file_io_operations'] * 2 +
                          indicators['sql_operations'] * 3 +
                          indicators['arithmetic_operations'])
        
        if total_operations > 10:
            indicators['performance_risk'] = 'high'
        elif total_operations > 5:
            indicators['performance_risk'] = 'medium'
        
        return indicators

    async def _analyze_sql_business_context(self, sql_content: str, host_variables: List[Dict]) -> Dict[str, Any]:
        """Analyze business context of SQL block"""
        context = {
            'sql_purpose': self._classify_sql_purpose(sql_content),
            'data_access_pattern': self._analyze_sql_access_pattern(sql_content),
            'business_entity': self._identify_business_entity(sql_content),
            'transaction_impact': self._assess_sql_transaction_impact(sql_content),
            'host_variable_flow': self._analyze_host_variable_flow(host_variables)
        }
        
        return context

    def _classify_sql_purpose(self, sql_content: str) -> str:
        """Classify the business purpose of SQL statement"""
        sql_upper = sql_content.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            if 'COUNT(' in sql_upper:
                return 'data_validation'
            elif 'SUM(' in sql_upper or 'AVG(' in sql_upper:
                return 'aggregation'
            else:
                return 'data_retrieval'
        elif sql_upper.startswith('INSERT'):
            return 'data_creation'
        elif sql_upper.startswith('UPDATE'):
            return 'data_modification'
        elif sql_upper.startswith('DELETE'):
            return 'data_removal'
        else:
            return 'database_operation'

    def _analyze_sql_access_pattern(self, sql_content: str) -> str:
        """Analyze SQL data access pattern"""
        sql_upper = sql_content.upper()
        
        if 'JOIN' in sql_upper:
            return 'relational_access'
        elif 'WHERE' in sql_upper and ('=' in sql_upper or 'IN' in sql_upper):
            return 'indexed_access'
        elif 'WHERE' in sql_upper:
            return 'filtered_access'
        elif 'ORDER BY' in sql_upper:
            return 'sorted_access'
        else:
            return 'sequential_access'

    def _identify_business_entity(self, sql_content: str) -> str:
        """Identify primary business entity in SQL"""
        sql_upper = sql_content.upper()
        
        # Extract table names and classify
        tables = self._extract_table_names(sql_content)
        
        for table in tables:
            table_upper = table.upper()
            if any(pattern in table_upper for pattern in ['CUSTOMER', 'CLIENT', 'MEMBER']):
                return 'customer'
            elif any(pattern in table_upper for pattern in ['ACCOUNT', 'ACCT']):
                return 'account'
            elif any(pattern in table_upper for pattern in ['PRODUCT', 'ITEM', 'SERVICE']):
                return 'product'
            elif any(pattern in table_upper for pattern in ['TRANSACTION', 'TRANS', 'TXN']):
                return 'transaction'
            elif any(pattern in table_upper for pattern in ['ORDER', 'INVOICE', 'PAYMENT']):
                return 'order_management'
        
        return 'general'

    def _assess_sql_transaction_impact(self, sql_content: str) -> str:
        """Assess transaction impact of SQL statement"""
        sql_upper = sql_content.upper()
        
        if any(stmt in sql_upper for stmt in ['INSERT', 'UPDATE', 'DELETE']):
            if 'WHERE' not in sql_upper:
                return 'high'  # Mass updates without WHERE clause
            else:
                return 'medium'
        elif sql_upper.startswith('SELECT'):
            if any(func in sql_upper for func in ['COUNT(', 'SUM(', 'MAX(', 'MIN(']):
                return 'medium'  # Aggregation queries
            else:
                return 'low'
        else:
            return 'low'

    def _analyze_host_variable_flow(self, host_variables: List[Dict]) -> Dict[str, List[str]]:
        """Analyze host variable data flow"""
        flow = {
            'input_variables': [],
            'output_variables': [],
            'bidirectional_variables': []
        }
        
        for var in host_variables:
            context = var.get('context', 'unknown')
            var_name = var.get('name', '')
            
            if context == 'input':
                flow['input_variables'].append(var_name)
            elif context == 'output':
                flow['output_variables'].append(var_name)
            else:
                flow['bidirectional_variables'].append(var_name)
        
        return flow

    def _calculate_sql_complexity(self, sql_content: str) -> int:
        """Calculate SQL complexity score"""
        complexity = 1
        sql_upper = sql_content.upper()
        
        # Add complexity for joins
        complexity += sql_upper.count('JOIN') * 2
        
        # Add complexity for subqueries
        complexity += sql_upper.count('SELECT') - 1  # Exclude main SELECT
        
        # Add complexity for aggregation functions
        complexity += sql_upper.count('GROUP BY') * 2
        complexity += sql_upper.count('HAVING') * 2
        
        # Add complexity for functions
        functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'SUBSTR(', 'DECODE(']
        complexity += sum(sql_upper.count(func) for func in functions)
        
        return min(complexity, 20)

    def _analyze_sql_performance(self, sql_content: str) -> Dict[str, Any]:
        """Analyze SQL performance indicators"""
        indicators = {
            'join_count': 0,
            'subquery_count': 0,
            'function_count': 0,
            'has_where_clause': False,
            'has_order_by': False,
            'performance_risk': 'low'
        }
        
        sql_upper = sql_content.upper()
        
        indicators['join_count'] = sql_upper.count('JOIN')
        indicators['subquery_count'] = sql_upper.count('SELECT') - 1
        indicators['has_where_clause'] = 'WHERE' in sql_upper
        indicators['has_"""
Agent 1: Complete Enhanced Code Parser & Chunker - BUSINESS LOGIC FIXED VERSION
Handles COBOL, JCL, CICS, BMS, and Copybook parsing with intelligent chunking and proper business rules
"""

import re
import asyncio
import sqlite3
import json
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
from datetime import datetime as dt
import logging
from enum import Enum

import torch
from vllm import AsyncLLMEngine, SamplingParams

# Business Rule Enums and Classes
class COBOLDivision(Enum):
    IDENTIFICATION = 1
    ENVIRONMENT = 2
    DATA = 3
    PROCEDURE = 4

class DataItemType(Enum):
    GROUP = "01-49"
    RENAMES = "66"
    INDEPENDENT = "77"
    CONDITION = "88"

class TransactionState:
    def __init__(self):
        self.input_received = False
        self.map_loaded = False
        self.file_opened = {}
        self.error_handlers = {}
        
    def set_input_received(self):
        self.input_received = True
        
    def set_map_loaded(self, mapset, map_name):
        self.map_loaded = True
        self.current_map = (mapset, map_name)

@dataclass
class BusinessRuleViolation(Exception):
    rule: str
    context: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO

@dataclass
class CodeChunk:
    """Represents a parsed code chunk with business context"""
    program_name: str
    chunk_id: str
    chunk_type: str  # paragraph, perform, job_step, proc, sql_block, section, cics_command
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int
    business_context: Dict[str, Any] = None  # NEW: Business-specific context

@dataclass
class ControlFlowPath:
    """Represents a control flow execution path"""
    path_id: str
    entry_point: str
    exit_points: List[str]
    conditions: List[str]
    called_paragraphs: List[str]
    data_accessed: List[str]

class CompleteEnhancedCodeParserAgent:
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
                 gpu_id: int = None, coordinator=None):
        self._engine = None  # Cached engine reference (starts as None)
        
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        # Thread safety
        self._engine_lock = asyncio.Lock()
        self._engine_created = False
        self._using_coordinator_llm = False
        self._processed_files = set()  # Duplicate prevention
        self._engine_loaded = False
        self._using_shared_engine = False
        
        # Business Rule Validators
        self.business_validators = {
            'cobol': COBOLBusinessValidator(),
            'jcl': JCLBusinessValidator(),
            'cics': CICSBusinessValidator(),
            'bms': BMSBusinessValidator()
        }
        
        # ENHANCED COBOL PATTERNS with business context
        self.cobol_patterns = {
            # Basic identification with stricter boundaries
            'program_id': re.compile(r'^\s*PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'author': re.compile(r'^\s*AUTHOR\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'date_written': re.compile(r'^\s*DATE-WRITTEN\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            'date_compiled': re.compile(r'^\s*DATE-COMPILED\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            
            # Divisions with proper boundaries and order enforcement
            'identification_division': re.compile(r'^\s*IDENTIFICATION\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'environment_division': re.compile(r'^\s*ENVIRONMENT\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'data_division': re.compile(r'^\s*DATA\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'procedure_division': re.compile(r'^\s*PROCEDURE\s+DIVISION(?:\s+USING\s+([^\.]+))?\s*\.', re.IGNORECASE | re.MULTILINE),
            
            # Sections with proper hierarchy
            'working_storage': re.compile(r'^\s*WORKING-STORAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'file_section': re.compile(r'^\s*FILE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'linkage_section': re.compile(r'^\s*LINKAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'local_storage': re.compile(r'^\s*LOCAL-STORAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'^\s*([A-Z0-9][A-Z0-9-]*)\s+SECTION\s*\.\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Paragraphs with better boundary detection
            'paragraph': re.compile(r'^\s*([A-Z0-9][A-Z0-9-]*)\s*\.\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PERFORM patterns with business logic
            'perform_simple': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s*(?:\.|$)', re.IGNORECASE),
            'perform_until': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?UNTIL\s+(.*?)(?=\s+|$)', re.IGNORECASE | re.DOTALL),
            'perform_varying': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+(.*?)\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*)', re.MULTILINE | re.IGNORECASE),
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+IS\s+([X9AV\(\)S\+\-\.,/Z*]+)|PIC(?:TURE)?\s+([X9AV\(\)S\+\-\.,/Z*]+)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER)', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # File operations with enhanced context
            'file_control': re.compile(r'^\s*FILE-CONTROL\s*\.', re.IGNORECASE | re.MULTILINE),
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Enhanced SQL blocks with host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR', re.IGNORECASE),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Enhanced COPY statements
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+IN\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
        }
        
        # Enhanced JCL patterns with execution context
        self.jcl_patterns = {
            'job_card': re.compile(r'^//(\S+)\s+JOB\s+', re.MULTILINE),
            'job_step': re.compile(r'^//(\S+)\s+EXEC\s+', re.MULTILINE),
            'dd_statement': re.compile(r'^//(\S+)\s+DD\s+', re.MULTILINE),
            'proc_call': re.compile(r'EXEC\s+(\S+)', re.IGNORECASE),
            'dataset': re.compile(r'DSN=([^,\s]+)', re.IGNORECASE),
            'proc_definition': re.compile(r'^//(\S+)\s+PROC', re.MULTILINE),
            'pend_statement': re.compile(r'^//\s+PEND', re.MULTILINE),
            'set_statement': re.compile(r'^//\s+SET\s+([^=]+)=([^\s,]+)', re.MULTILINE),
            'if_statement': re.compile(r'^//\s+IF\s+(.*?)\s+THEN', re.MULTILINE),
            'endif_statement': re.compile(r'^//\s+ENDIF', re.MULTILINE),
            'cond_parameter': re.compile(r'COND=\(([^)]+)\)', re.IGNORECASE),
            'restart_parameter': re.compile(r'RESTART=([A-Z0-9]+)', re.IGNORECASE),
            'return_code_check': re.compile(r'\bRC\s*(=|EQ|NE|GT|LT|GE|LE)\s*(\d+)', re.IGNORECASE),
        }
        
        # Enhanced CICS patterns with transaction context
        self.cics_patterns = {
            # Terminal operations with parameter validation
            'cics_send_map': re.compile(r'EXEC\s+CICS\s+SEND\s+MAP\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive_map': re.compile(r'EXEC\s+CICS\s+RECEIVE\s+MAP\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_send_text': re.compile(r'EXEC\s+CICS\s+SEND\s+TEXT\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive': re.compile(r'EXEC\s+CICS\s+RECEIVE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # File operations with enhanced context
            'cics_read': re.compile(r'EXEC\s+CICS\s+READ\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_write': re.compile(r'EXEC\s+CICS\s+WRITE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_rewrite': re.compile(r'EXEC\s+CICS\s+REWRITE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_delete': re.compile(r'EXEC\s+CICS\s+DELETE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Program control with flow analysis
            'cics_link': re.compile(r'EXEC\s+CICS\s+LINK\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_xctl': re.compile(r'EXEC\s+CICS\s+XCTL\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_return': re.compile(r'EXEC\s+CICS\s+RETURN\s*(?:\((.*?)\))?\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Error handling with context tracking
            'cics_handle_condition': re.compile(r'EXEC\s+CICS\s+HANDLE\s+CONDITION\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_handle_aid': re.compile(r'EXEC\s+CICS\s+HANDLE\s+AID\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_resp': re.compile(r'RESP\(([A-Z][A-Z0-9-]*)\)', re.IGNORECASE),
            'cics_nohandle': re.compile(r'\bNOHANDLE\b', re.IGNORECASE),
        }
        
        # Enhanced BMS patterns
        self.bms_patterns = {
            'bms_mapset': re.compile(r'(\w+)\s+DFHMSD\s+(.*?)(?=\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_map': re.compile(r'(\w+)\s+DFHMDI\s+(.*?)(?=\w+\s+DFHMDI|\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_field': re.compile(r'(\w+)\s+DFHMDF\s+(.*?)(?=\w+\s+DFHMDF|\w+\s+DFHMDI|\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_mapset_end': re.compile(r'\s+DFHMSD\s+TYPE=FINAL', re.IGNORECASE),
            'bms_pos': re.compile(r'POS=\((\d+),(\d+)\)', re.IGNORECASE),
            'bms_length': re.compile(r'LENGTH=(\d+)', re.IGNORECASE),
        }
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database with enhanced schema including business rules"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced table with business context
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS program_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    business_context TEXT,
                    embedding_id TEXT,
                    file_hash TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    line_start INTEGER,
                    line_end INTEGER,
                    UNIQUE(program_name, chunk_id)
                )
            """)
            
            # Business rule violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS business_rule_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    line_number INTEGER,
                    context TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Control flow analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS control_flow_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    path_id TEXT NOT NULL,
                    entry_point TEXT,
                    exit_points TEXT,
                    conditions TEXT,
                    called_paragraphs TEXT,
                    data_accessed TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_name ON program_chunks(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON program_chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON program_chunks(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_business_rules ON business_rule_violations(program_name, rule_type)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")

    async def get_engine(self):
        """Get LLM engine with lazy loading and sharing - UNCHANGED"""
        if self._engine is None and self.coordinator:
            async with self._engine_lock:
                if self._engine is None:
                    try:
                        assigned_gpu = self.coordinator.agent_gpu_assignments.get("code_parser")
                        if assigned_gpu is not None:
                            self._engine = await self.coordinator.get_shared_llm_engine(assigned_gpu)
                            self.gpu_id = assigned_gpu
                            self._using_shared_engine = True
                            self._engine_loaded = True
                            self.logger.info(f"✅ CodeParser using shared engine on GPU {assigned_gpu}")
                        else:
                            raise ValueError("No GPU assigned for code_parser agent type")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to get shared engine: {e}")
                        raise
        
        return self._engine

    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM - UNCHANGED"""
        try:
            engine = await self.get_engine()
            if engine is None:
                self.logger.warning("No LLM engine available")
                return ""
            
            request_id = str(uuid.uuid4())
            
            try:
                result_generator = engine.generate(
                    prompt, sampling_params, request_id=request_id
                )
                
                async for result in result_generator:
                    if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                        return result.outputs[0].text.strip()
                    break
                    
            except TypeError as e:
                if "request_id" in str(e):
                    result_generator = engine.generate(prompt, sampling_params)
                    async for result in result_generator:
                        if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                            return result.outputs[0].text.strip()
                        break
                        
            return ""
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""

    def _generate_file_hash(self, content: str, file_path: Path) -> str:
        """Generate unique hash for file content and metadata - UNCHANGED"""
        hash_input = f"{file_path.name}:{file_path.stat().st_mtime}:{len(content)}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _is_duplicate_file(self, file_path: Path, content: str) -> bool:
        """Check if file has already been processed - UNCHANGED"""
        file_hash = self._generate_file_hash(content, file_path)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks 
                WHERE file_hash = ?
            """, (file_hash,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"Duplicate check failed: {str(e)}")
            return False

    def _extract_program_name(self, content: str, file_path: Path) -> str:
        """Extract program name more robustly from content or filename - UNCHANGED"""
        try:
            program_match = self.cobol_patterns['program_id'].search(content)
            if program_match:
                return program_match.group(1).strip()
            
            job_match = self.jcl_patterns['job_card'].search(content)
            if job_match:
                return job_match.group(1).strip()
            
            if isinstance(file_path, str):
                file_path = Path(file_path)
            filename = file_path.name
            
            for ext in ['.cbl', '.cob', '.jcl', '.copy', '.cpy', '.bms']:
                if filename.lower().endswith(ext):
                    return filename[:-len(ext)]
            
            return file_path.stem
            
        except Exception as e:
            self.logger.error(f"Error extracting program name: {str(e)}")
            if isinstance(file_path, (str, Path)):
                return Path(file_path).stem or "UNKNOWN_PROGRAM"
            return file_path.stem or "UNKNOWN_PROGRAM"

    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results - UNCHANGED"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'enhanced_code_parser'
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_loaded_lazily'] = self._engine_loaded
        return result

    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Enhanced file type detection with proper business rule ordering"""
        content_upper = content.upper()
        
        # Order matters - check most specific patterns first
        
        # BMS detection (most specific)
        if self._is_bms_file(content_upper):
            return 'bms'
        
        # Heavy CICS detection (before general COBOL)
        if self._is_heavy_cics_program(content_upper):
            return 'cics'
        
        # JCL detection (specific format)
        if self._is_jcl_file(content, suffix):
            return 'jcl'
        
        # COBOL detection (most common, check after specific types)
        if self._is_cobol_program(content_upper):
            return 'cobol'
        
        # Copybook detection (specific content patterns)
        if self._is_copybook(content_upper, suffix):
            return 'copybook'
        
        # Extension-based fallback
        suffix_lower = suffix.lower()
        if suffix_lower in ['.cbl', '.cob']:
            return 'cobol'
        elif suffix_lower == '.jcl':
            return 'jcl'
        elif suffix_lower in ['.cpy', '.copy']:
            return 'copybook'
        elif suffix_lower == '.bms':
            return 'bms'
        
        return 'unknown'

    def _is_bms_file(self, content_upper: str) -> bool:
        """Check if file is BMS mapset"""
        return any(marker in content_upper for marker in ['DFHMSD', 'DFHMDI', 'DFHMDF'])

    def _is_heavy_cics_program(self, content_upper: str) -> bool:
        """Check if file is CICS-heavy program"""
        cics_count = content_upper.count('EXEC CICS')
        total_lines = content_upper.count('\n') + 1
        # More than 10% of lines contain CICS commands
        return cics_count > max(5, total_lines * 0.1)

    def _is_jcl_file(self, content: str, suffix: str) -> bool:
        """Check if file is JCL"""
        if suffix.lower() == '.jcl':
            return True
        return (content.strip().startswith('//') and 
                any(marker in content.upper() for marker in ['JOB', 'EXEC', 'DD']))

    def _is_cobol_program(self, content_upper: str) -> bool:
        """Check if file is COBOL program"""
        return any(marker in content_upper for marker in [
            'IDENTIFICATION DIVISION', 'PROGRAM-ID', 'WORKING-STORAGE SECTION',
            'PROCEDURE DIVISION', 'DATA DIVISION'
        ])

    def _is_copybook(self, content_upper: str, suffix: str) -> bool:
        """Check if file is copybook"""
        if suffix.lower() in ['.cpy', '.copy']:
            return True
        # Copybooks often have data definitions without divisions
        has_data_items = 'PIC' in content_upper and ('01 ' in content_upper or '05 ' in content_upper)
        no_divisions = not any(div in content_upper for div in ['IDENTIFICATION DIVISION', 'PROCEDURE DIVISION'])
        return has_data_items and no_divisions and len(content_upper.split('\n')) < 500

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single code file with enhanced business rule validation"""
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            if not file_path.exists():
                return self._add_processing_info({
                    "status": "error",
                    "file_name": str(file_path.name),
                    "error": "File not found"
                })
            
            # Enhanced file reading with multiple encoding attempts
            content = await self._read_file_with_encoding(file_path)
            if content is None:
                return self._add_processing_info({
                    "status": "error",
                    "file_name": str(file_path.name),
                    "error": "Unable to decode file"
                })

            if not content.strip():
                return self._add_processing_info({
                    "status": "error",
                    "file_name": str(file_path.name),
                    "error": "File is empty"
                })

            # Check for duplicates
            if self._is_duplicate_file(file_path, content):
                return self._add_processing_info({
                    "status": "skipped",
                    "file_name": str(file_path.name),
                    "message": "File already processed (duplicate detected)"
                })
            
            file_type = self._detect_file_type(content, file_path.suffix)
            self.logger.info(f"Detected file type: {file_type}")
            
            # Business rule validation before parsing
            business_violations = []
            if file_type in self.business_validators:
                violations = await self.business_validators[file_type].validate_structure(content)
                business_violations.extend(violations)
            
            # Parse based on file type with business context
            if file_type == 'cobol':
                chunks = await self._parse_cobol_with_business_rules(content, str(file_path.name))
            elif file_type == 'jcl':
                chunks = await self._parse_jcl_with_business_rules(content, str(file_path.name))
            elif file_type == 'copybook':
                chunks = await self._parse_copybook_with_business_rules(content, str(file_path.name))
            elif file_type == 'bms':
                chunks = await self._parse_bms_with_business_rules(content, str(file_path.name))
            elif file_type == 'cics':
                chunks = await self._parse_cics_with_business_rules(content, str(file_path.name))
            else:
                chunks = await self._parse_generic(content, str(file_path.name))
            
            self.logger.info(f"Generated {len(chunks)} chunks")
            
            if not chunks:
                return self._add_processing_info({
                    "status": "warning",
                    "file_name": str(file_path.name),
                    "file_type": file_type,
                    "chunks_created": 0,
                    "message": "No chunks were created from this file"
                })
            
            # Store business violations
            if business_violations:
                await self._store_business_violations(business_violations, self._extract_program_name(content, file_path))
            
            # Add file hash to all chunks
            file_hash = self._generate_file_hash(content, file_path)
            for chunk in chunks:
                chunk.metadata['file_hash'] = file_hash
            
            # Store chunks with verification
            await self._store_chunks_enhanced(chunks, file_hash)
            
            # Verify chunks were stored
            stored_chunks = await self._verify_chunks_stored(self._extract_program_name(content, file_path))
            
            # Generate enhanced metadata with business context
            metadata = await self._generate_metadata_enhanced(chunks, file_type, business_violations)

            # Generate control flow analysis
            if file_type in ['cobol', 'cics']:
                control_flow = await self._analyze_control_flow(chunks)
                await self._store_control_flow_analysis(control_flow, self._extract_program_name(content, file_path))
            
            result = {
                "status": "success",
                "file_name": str(file_path.name),
                "file_type": file_type,
                "chunks_created": len(chunks),
                "chunks_verified": stored_chunks,
                "business_violations": len(business_violations),
                "metadata": metadata,
                "processing_timestamp": dt.now().isoformat(),
                "file_hash": file_hash
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {str(e)}")
            return self._add_processing_info({
                "status": "error",
                "file_name": file_path.name,
                "error": str(e)
            })

    async def _read_file_with_encoding(self, file_path: Path) -> Optional[str]:
        """Enhanced file reading with multiple encoding attempts"""
        encodings = ['utf-8', 'cp1252', 'latin1', 'ascii', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                self.logger.debug(f"Successfully read file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error reading file with {encoding}: {e}")
                continue
        
        return None

    async def _parse_cobol_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete COBOL parsing with business rule enforcement"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # Validate COBOL structure first
        structure_validation = await self._validate_cobol_structure(content, program_name)
        
        # Parse divisions with hierarchy validation
        division_chunks = await self._parse_cobol_divisions_with_validation(content, program_name)
        chunks.extend(division_chunks)
        
        # Parse sections with proper context
        section_chunks = await self._parse_cobol_sections_with_context(content, program_name)
        chunks.extend(section_chunks)
        
        # Parse data items with level validation
        data_chunks = await self._parse_data_items_with_business_rules(content, program_name)
        chunks.extend(data_chunks)
        
        # Parse procedure division with control flow analysis
        procedure_chunks = await self._parse_procedure_division_with_flow(content, program_name)
        chunks.extend(procedure_chunks)
        
        # Parse SQL blocks with host variable validation
        sql_chunks = await self._parse_sql_blocks_with_host_variables(content, program_name)
        chunks.extend(sql_chunks)
        
        # Parse CICS commands with transaction context
        cics_chunks = await self._parse_cics_with_transaction_context(content, program_name)
        chunks.extend(cics_chunks)
        
        return chunks

    async def _validate_cobol_structure(self, content: str, program_name: str) -> Dict[str, Any]:
        """Validate COBOL program structure according to business rules"""
        violations = []
        divisions_found = {}
        
        # Check for required divisions
        required_divisions = ['IDENTIFICATION', 'PROCEDURE']
        division_order = ['IDENTIFICATION', 'ENVIRONMENT', 'DATA', 'PROCEDURE']
        
        for division in division_order:
            pattern_name = f'{division.lower()}_division'
            if pattern_name in self.cobol_patterns:
                match = self.cobol_patterns[pattern_name].search(content)
                if match:
                    divisions_found[division] = {
                        'position': match.start(),
                        'line': content[:match.start()].count('\n') + 1
                    }
        
        # Validate required divisions
        for req_div in required_divisions:
            if req_div not in divisions_found:
                violations.append(BusinessRuleViolation(
                    rule=f"MISSING_REQUIRED_DIVISION",
                    context=f"Missing required {req_div} DIVISION",
                    severity="ERROR"
                ))
        
        # Validate division order
        found_order = [d for d in division_order if d in divisions_found]
        for i in range(len(found_order) - 1):
            current = found_order[i]
            next_div = found_order[i + 1]
            
            if divisions_found[current]['position'] > divisions_found[next_div]['position']:
                violations.append(BusinessRuleViolation(
                    rule="DIVISION_ORDER_VIOLATION",
                    context=f"{current} DIVISION appears after {next_div} DIVISION",
                    severity="ERROR"
                ))
        
        return {
            'divisions_found': divisions_found,
            'violations': violations,
            'structure_valid': len([v for v in violations if v.severity == "ERROR"]) == 0
        }

    async def _parse_cobol_divisions_with_validation(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL divisions with proper business validation"""
        chunks = []
        
        division_patterns = {
            'identification_division': self.cobol_patterns['identification_division'],
            'environment_division': self.cobol_patterns['environment_division'],
            'data_division': self.cobol_patterns['data_division'],
            'procedure_division': self.cobol_patterns['procedure_division']
        }
        
        division_positions = {}
        for div_name, pattern in division_patterns.items():
            match = pattern.search(content)
            if match:
                division_positions[div_name] = {
                    'start': match.start(),
                    'match': match
                }
        
        # Sort divisions by position
        sorted_divisions = sorted(division_positions.items(), key=lambda x: x[1]['start'])
        
        for i, (div_name, div_info) in enumerate(sorted_divisions):
            start_pos = div_info['start']
            
            # Find end position (next division or end of content)
            if i + 1 < len(sorted_divisions):
                end_pos = sorted_divisions[i + 1][1]['start']
            else:
                end_pos = len(content)
            
            div_content = content[start_pos:end_pos].strip()
            
            # Enhanced business context analysis
            business_context = await self._analyze_division_business_context(div_content, div_name)
            
            # LLM analysis for deeper insights
            metadata = await self._analyze_division_with_llm(div_content, div_name)
            metadata.update(business_context)
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{div_name.upper()}",
                chunk_type="division",
                content=div_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _analyze_division_business_context(self, content: str, division_name: str) -> Dict[str, Any]:
        """Analyze division-specific business context"""
        context = {
            'division_type': division_name,
            'business_purpose': '',
            'dependencies': [],
            'compliance_requirements': []
        }
        
        if 'identification' in division_name:
            context.update({
                'business_purpose': 'Program identification and documentation',
                'program_metadata': self._extract_identification_metadata(content),
                'compliance_requirements': ['PROGRAM-ID required', 'Documentation standards']
            })
        elif 'environment' in division_name:
            context.update({
                'business_purpose': 'System and file environment configuration',
                'file_assignments': self._extract_file_assignments(content),
                'system_dependencies': self._extract_system_dependencies(content)
            })
        elif 'data' in division_name:
            context.update({
                'business_purpose': 'Data structure definitions and storage allocation',
                'data_structures': self._extract_data_structures_summary(content),
                'memory_requirements': self._estimate_memory_requirements(content)
            })
        elif 'procedure' in division_name:
            context.update({
                'business_purpose': 'Business logic implementation and program flow',
                'entry_points': self._extract_entry_points(content),
                'business_functions': self._extract_business_functions(content)
            })
        
        return context

    async def _parse_data_items_with_business_rules(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse data items with proper level validation and business rules"""
        chunks = []
        
        # Find data sections
        data_sections = {
            'working_storage': self.cobol_patterns['working_storage'],
            'file_section': self.cobol_patterns['file_section'],
            'linkage_section': self.cobol_patterns['linkage_section'],
            'local_storage': self.cobol_patterns['local_storage']
        }
        
        for section_name, pattern in data_sections.items():
            section_match = pattern.search(content)
            if not section_match:
                continue
            
            # Extract section content
            section_start = section_match.end()
            section_end = self._find_section_end(content, section_start, list(data_sections.keys()) + ['procedure_division'])
            section_content = content[section_start:section_end]
            
            # Parse data items with business validation
            data_items = await self._parse_section_data_items(section_content, program_name, section_name)
            chunks.extend(data_items)
        
        return chunks

    async def _parse_section_data_items(self, section_content: str, program_name: str, section_name: str) -> List[CodeChunk]:
        """Parse data items within a section with level validation"""
        chunks = []
        level_stack = []  # Track level hierarchy
        
        data_matches = list(self.cobol_patterns['data_item'].finditer(section_content))
        
        for match in data_matches:
            level = int(match.group(1))
            name = match.group(2)
            definition = match.group(3)
            
            # Skip comment lines
            if match.group(0).strip().startswith('*'):
                continue
            
            # Validate data item according to business rules
            validation_result = self._validate_data_item_business_rules(level, name, definition, level_stack)
            
            if validation_result['valid']:
                # Update level stack
                level_stack = self._update_level_stack(level_stack, level, name)
                
                # Extract business metadata
                business_context = await self._analyze_data_item_business_context(level, name, definition, section_name)
                
                # Standard metadata
                metadata = {
                    'level': level,
                    'field_name': name,
                    'section': section_name,
                    'pic_clause': self._extract_pic_clause(definition),
                    'usage': self._extract_usage_clause(definition),
                    'value': self._extract_value_clause(definition),
                    'occurs': self._extract_occurs_info(definition),
                    'redefines': self._extract_redefines_info(definition),
                    'level_hierarchy': level_stack.copy(),
                    'data_type': self._determine_data_type_enhanced(definition),
                    'business_validation': validation_result
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_DATA_{name}_{level}",
                    chunk_type="data_item",
                    content=match.group(0),
                    metadata=metadata,
                    business_context=business_context,
                    line_start=section_content[:match.start()].count('\n'),
                    line_end=section_content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    def _validate_data_item_business_rules(self, level: int, name: str, definition: str, level_stack: List) -> Dict[str, Any]:
        """Validate data item according to COBOL business rules"""
        violations = []
        valid = True
        
        # Level number validation
        if level == 66:  # RENAMES
            if 'RENAMES' not in definition.upper():
                violations.append("Level 66 must have RENAMES clause")
                valid = False
        elif level == 77:  # Independent item
            if level_stack and any(item['level'] < 77 for item in level_stack):
                violations.append("Level 77 cannot be subordinate to other items")
                valid = False
        elif level == 88:  # Condition name
            if not level_stack:
                violations.append("Level 88 must be subordinate to another item")
                valid = False
            elif 'VALUE' not in definition.upper():
                violations.append("Level 88 must have VALUE clause")
                valid = False
        
        # Hierarchy validation
        if level_stack and 1 <= level <= 49:
            last_level = level_stack[-1]['level'] if level_stack else 0
            if level <= last_level and level != 1:
                # Check if this is a valid sibling or parent level
                valid_levels = self._get_valid_next_levels(level_stack)
                if level not in valid_levels:
                    violations.append(f"Invalid level hierarchy: {level} after {last_level}")
                    valid = False
        
        # PIC clause validation
        if 1 <= level <= 49 and level != 66 and level != 77:
            has_pic = self._extract_pic_clause(definition) is not None
            has_subordinates = False  # Would need to check following items
            
            if not has_pic and not has_subordinates and 'REDEFINES' not in definition.upper():
                # Elementary items must have PIC unless they're group items
                pass  # Would need more context to validate properly
        
        return {
            'valid': valid,
            'violations': violations,
            'level_type': self._classify_level_type(level)
        }

    def _update_level_stack(self, level_stack: List, current_level: int, current_name: str) -> List:
        """Update the level stack maintaining proper hierarchy"""
        # Remove levels that are not parents of current level
        while level_stack and level_stack[-1]['level'] >= current_level:
            level_stack.pop()
        
        # Add current level
        level_stack.append({
            'level': current_level,
            'name': current_name
        })
        
        return level_stack

    def _get_valid_next_levels(self, level_stack: List) -> List[int]:
        """Get valid level numbers that can follow the current hierarchy"""
        if not level_stack:
            return [1, 77]  # Can start with 01 or 77
        
        last_level = level_stack[-1]['level']
        
        if last_level == 77:
            return [1, 77]  # After 77, only 01 or another 77
        elif last_level == 88:
            # After 88, can have sibling 88 or return to parent levels
            parent_levels = [item['level'] for item in level_stack[:-1]]
            return [88] + parent_levels + [level for level in range(1, 49) if level > max(parent_levels) if parent_levels]
        else:
            # Normal hierarchy: can have subordinate, sibling, or parent levels
            valid = []
            # Subordinate levels
            valid.extend(range(last_level + 1, 50))
            # Sibling and parent levels
            valid.extend([item['level'] for item in level_stack])
            # 77 and 88 are always possible
            valid.extend([77, 88])
            return list(set(valid))

    def _classify_level_type(self, level: int) -> str:
        """Classify the type of level number"""
        if 1 <= level <= 49:
            return "group_or_elementary"
        elif level == 66:
            return "renames"
        elif level == 77:
            return "independent"
        elif level == 88:
            return "condition_name"
        else:
            return "invalid"

    async def _analyze_data_item_business_context(self, level: int, name: str, definition: str, section_name: str) -> Dict[str, Any]:
        """Analyze business context of data item"""
        context = {
            'data_category': self._categorize_data_item(name, definition),
            'business_domain': self._infer_business_domain(name),
            'usage_pattern': self._analyze_usage_pattern(definition),
            'validation_rules': self._extract_validation_rules(definition),
            'security_classification': self._classify_security_level(name, definition)
        }
        
        return context

    def _categorize_data_item(self, name: str, definition: str) -> str:
        """Categorize data item by business purpose"""
        name_upper = name.upper()
        def_upper = definition.upper()
        
        # Common business data patterns
        if any(pattern in name_upper for pattern in ['AMOUNT', 'AMT', 'TOTAL', 'SUM']):
            return 'financial'
        elif any(pattern in name_upper for pattern in ['DATE', 'TIME', 'TIMESTAMP']):
            return 'temporal'
        elif any(pattern in name_upper for pattern in ['NAME', 'ADDR', 'ADDRESS', 'PHONE']):
            return 'personal_data'
        elif any(pattern in name_upper for pattern in ['ID', 'KEY', 'NBR', 'NUMBER']):
            return 'identifier'
        elif any(pattern in name_upper for pattern in ['STATUS', 'FLAG', 'IND', 'INDICATOR']):
            return 'control'
        elif any(pattern in name_upper for pattern in ['CTR', 'COUNTER', 'CNT', 'COUNT']):
            return 'counter'
        elif 'FILLER' in name_upper:
            return 'filler'
        else:
            return 'business_data'

    async def _parse_procedure_division_with_flow(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse procedure division with control flow analysis"""
        chunks = []
        
        # Find procedure division
        proc_match = self.cobol_patterns['procedure_division'].search(content)
        if not proc_match:
            return chunks
        
        proc_start = proc_match.end()
        proc_content = content[proc_start:]
        
        # Parse paragraphs with enhanced context
        paragraph_chunks = await self._parse_paragraphs_with_business_context(proc_content, program_name, proc_start)
        chunks.extend(paragraph_chunks)
        
        # Parse PERFORM statements with flow analysis
        perform_chunks = await self._parse_perform_statements_with_flow(proc_content, program_name, proc_start)
        chunks.extend(perform_chunks)
        
        # Parse control structures with nesting analysis
        control_chunks = await self._parse_control_structures_with_nesting(proc_content, program_name, proc_start)
        chunks.extend(control_chunks)
        
        return chunks

    async def _parse_paragraphs_with_business_context(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse paragraphs with enhanced business context"""
        chunks = []
        
        # Find paragraphs (excluding sections)
        paragraph_matches = []
        for match in self.cobol_patterns['paragraph'].finditer(content):
            para_name = match.group(1)
            if not para_name.endswith('SECTION'):
                paragraph_matches.append(match)
        
        for i, match in enumerate(paragraph_matches):
            para_name = match.group(1)
            para_start = match.start()
            
            # Find paragraph end
            if i + 1 < len(paragraph_matches):
                para_end = paragraph_matches[i + 1].start()
            else:
                para_end = len(content)
            
            para_content = content[para_start:para_end].strip()
            
            # Enhanced business context analysis
            business_context = await self._analyze_paragraph_business_context(para_content, para_name)
            
            # LLM analysis for deeper insights
            metadata = await self._analyze_paragraph_with_llm(para_content)
            metadata.update({
                'paragraph_name': para_name,
                'business_function': business_context.get('business_function', 'Unknown'),
                'complexity_score': self._calculate_paragraph_complexity(para_content),
                'performance_indicators': self._analyze_performance_indicators(para_content)
            })
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{para_name}",
                chunk_type="paragraph",
                content=para_content,
                metadata=metadata,
                business_context=business_context,
                line_start=offset + para_start,
                line_end=offset + para_end
            )
            chunks.append(chunk)
        
        return chunks

    async def _analyze_paragraph_business_context(self, content: str, para_name: str) -> Dict[str, Any]:
        """Analyze business context of a paragraph"""
        context = {
            'business_function': self._infer_business_function(para_name, content),
            'data_operations': self._extract_data_operations(content),
            'file_operations': self._extract_file_operations_detailed(content),
            'control_flow': self._extract_control_flow_elements(content),
            'error_handling': self._extract_error_handling_detailed(content),
            'performance_impact': self._assess_performance_impact(content),
            'business_rules_implemented': self._identify_business_rules(content)
        }
        
        return context

    def _infer_business_function(self, para_name: str, content: str) -> str:
        """Infer business function from paragraph name and content"""
        name_upper = para_name.upper()
        content_upper = content.upper()
        
        # Common business function patterns
        if any(pattern in name_upper for pattern in ['INIT', 'INITIAL', 'START']):
            return 'initialization'
        elif any(pattern in name_upper for pattern in ['READ', 'GET', 'FETCH']):
            return 'data_retrieval'
        elif any(pattern in name_upper for pattern in ['WRITE', 'UPDATE', 'SAVE']):
            return 'data_modification'
        elif any(pattern in name_upper for pattern in ['CALC', 'COMPUTE', 'TOTAL']):
            return 'calculation'
        elif any(pattern in name_upper for pattern in ['VALID', 'CHECK', 'EDIT']):
            return 'validation'
        elif any(pattern in name_upper for pattern in ['PRINT', 'DISPLAY', 'SHOW']):
            return 'output'
        elif any(pattern in name_upper for pattern in ['END', 'EXIT', 'TERM']):
            return 'termination'
        elif any(pattern in name_upper for pattern in ['ERROR', 'EXCEPT']):
            return 'error_handling'
        else:
            # Analyze content for clues
            if 'PERFORM' in content_upper and 'UNTIL' in content_upper:
                return 'loop_processing'
            elif 'IF' in content_upper:
                return 'conditional_processing'
            elif 'MOVE' in content_upper:
                return 'data_movement'
            else:
                return 'business_logic'

    async def _parse_perform_statements_with_flow(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse PERFORM statements with enhanced flow analysis"""
        chunks = []
        
        # Enhanced PERFORM patterns with proper business logic
        perform_patterns = {
            'perform_simple': self.cobol_patterns['perform_simple'],
            'perform_until': self.cobol_patterns['perform_until'],
            'perform_varying': self.cobol_patterns['perform_varying'],
            'perform_thru': self.cobol_patterns['perform_thru'],
            'perform_times': self.cobol_patterns['perform_times'],
            'perform_inline': self.cobol_patterns['perform_inline']
        }
        
        for perform_type, pattern in perform_patterns.items():
            matches = pattern.finditer(content)
            
            for i, match in enumerate(matches):
                # Extract PERFORM statement content
                perform_content = self._extract_complete_perform_statement(content, match.start())
                
                # Analyze business context
                business_context = await self._analyze_perform_business_context(perform_content, perform_type)
                
                # Enhanced metadata with flow analysis
                metadata = {
                    'perform_type': perform_type,
                    'target_paragraph': self._extract_target_paragraph(match),
                    'loop_analysis': self._analyze_loop_characteristics(perform_content, perform_type),
                    'execution_pattern': self._determine_execution_pattern(perform_type, perform_content),
                    'performance_implications': self._assess_perform_performance(perform_content, perform_type)
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_PERFORM_{perform_type}_{i+1}",
                    chunk_type="perform_statement",
                    content=perform_content,
                    metadata=metadata,
                    business_context=business_context,
                    line_start=offset + match.start(),
                    line_end=offset + match.start() + len(perform_content)
                )
                chunks.append(chunk)
        
        return chunks

    async def _analyze_perform_business_context(self, content: str, perform_type: str) -> Dict[str, Any]:
        """Analyze business context of PERFORM statement"""
        context = {
            'execution_model': self._classify_execution_model(perform_type),
            'business_purpose': self._infer_perform_purpose(content, perform_type),
            'data_impact': self._analyze_perform_data_impact(content),
            'control_complexity': self._measure_control_complexity(content, perform_type),
            'maintainability_score': self._assess_perform_maintainability(content, perform_type)
        }
        
        return context

    def _classify_execution_model(self, perform_type: str) -> str:
        """Classify the execution model of PERFORM statement"""
        if 'simple' in perform_type:
            return 'single_execution'
        elif 'until' in perform_type:
            return 'conditional_loop'
        elif 'varying' in perform_type:
            return 'counted_loop'
        elif 'times' in perform_type:
            return 'fixed_iteration'
        elif 'thru' in perform_type:
            return 'range_execution'
        elif 'inline' in perform_type:
            return 'inline_block'
        else:
            return 'unknown'

    async def _parse_sql_blocks_with_host_variables(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse SQL blocks with proper host variable validation"""
        chunks = []
        
        sql_matches = self.cobol_patterns['sql_block'].finditer(content)
        
        for i, match in enumerate(sql_matches):
            sql_content = match.group(0)
            sql_inner = match.group(1).strip()
            
            # Extract and validate host variables
            host_variables = self._extract_host_variables(sql_inner)
            host_var_validation = await self._validate_host_variables(host_variables, content)
            
            # Comprehensive SQL analysis with business context
            business_context = await self._analyze_sql_business_context(sql_inner, host_variables)
            
            # Enhanced metadata with host variable context
            metadata = await self._analyze_sql_comprehensive(sql_inner)
            metadata.update({
                'host_variables': host_variables,
                'host_variable_validation': host_var_validation,
                'sql_complexity': self._calculate_sql_complexity(sql_inner),
                'performance_indicators': self._analyze_sql_performance(sql_inner)
            })
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_SQL_BLOCK_{i+1}",
                chunk_type="sql_block",
                content=sql_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _extract_host_variables(self, sql_content: str) -> List[Dict[str, str]]:
        """Extract host variables from SQL content"""
        host_vars = []
        
        host_var_matches = self.cobol_patterns['sql_host_var'].finditer(sql_content)
        
        for match in host_var_matches:
            var_name = match.group(1)
            var_context = self._determine_host_var_context(sql_content, match.start())
            
            host_vars.append({
                'name': var_name,
                'cobol_name': var_name.replace('-', '_'),  # SQL uses underscores
                'context': var_context,
                'position': match.start()
            })
        
        return host_vars

    async def _validate_host_variables(self, host_variables: List[Dict], cobol_content: str) -> Dict[str, Any]:
        """Validate that host variables exist in COBOL working storage"""
        validation_results = {
            'valid_variables': [],
            'missing_variables': [],
            'type_mismatches': [],
            'all_valid': True
        }
        
        # Extract COBOL data definitions for validation
        cobol_data_items = self._extract_cobol_data_definitions(cobol_content)
        
        for host_var in host_variables:
            cobol_name = host_var['cobol_name'].upper()
            
            if cobol_name in cobol_data_items:
                validation_results['valid_variables'].append({
                    'host_var': host_var['name'],
                    'cobol_definition': cobol_data_items[cobol_name]
                })
            else:
                validation_results['missing_variables'].append(host_var['name'])
                validation_results['all_valid'] = False
        
        return validation_results

    async def _parse_cics_with_transaction_context(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse CICS commands with proper transaction state tracking"""
        chunks = []
        transaction_state = TransactionState()
        
        # Parse CICS commands in sequence to maintain transaction context
        cics_commands = []
        
        for command_type, pattern in self.cics_patterns.items():
            matches = pattern.finditer(content)
            
            for match in matches:
                cics_commands.append({
                    'type': command_type,
                    'match': match,
                    'position': match.start(),
                    'content': match.group(0),
                    'params': match.group(1) if match.groups() else ""
                })
        
        # Sort commands by position to maintain execution order
        cics_commands.sort(key=lambda x: x['position'])
        
        # Process commands with transaction state validation
        for i, cmd in enumerate(cics_commands):
            # Validate command in transaction context
            context_validation = await self._validate_cics_transaction_context(cmd, transaction_state)
            
            # Update transaction state
            self._update_transaction_state(transaction_state, cmd)
            
            # Analyze business context
            business_context = await self._analyze_cics_business_context(cmd, transaction_state)
            
            # Enhanced metadata
            metadata = await self._analyze_cics_command_comprehensive(cmd['type'], cmd['params'], cmd['content'])
            metadata.update({
                'transaction_sequence': i + 1,
                'context_validation': context_validation,
                'transaction_state': {
                    'input_received': transaction_state.input_received,
                    'map_loaded': transaction_state.map_loaded,
                    'files_open': list(transaction_state.file_opened.keys())
                }
            })
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_CICS_{cmd['type']}_{i+1}",
                chunk_type="cics_command",
                content=cmd['content'],
                metadata=metadata,
                business_context=business_context,
                line_start=content[:cmd['position']].count('\n'),
                line_end=content[:cmd['position'] + len(cmd['content'])].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _validate_cics_transaction_context(self, cmd: Dict, transaction_state: TransactionState) -> Dict[str, Any]:
        """Validate CICS command in current transaction context"""
        violations = []
        warnings = []
        
        cmd_type = cmd['type']
        cmd_params = cmd['params']
        
        # Business rule validations
        if cmd_type.startswith('cics_send') and not transaction_state.input_received:
            warnings.append("SEND operation without prior RECEIVE - potential transaction flow issue")
        
        if cmd_type == 'cics_send_map':
            if not self._has_required_params(cmd_params, ['MAP', 'MAPSET']):
                violations.append("SEND MAP requires both MAP and MAPSET parameters")
        
        if cmd_type == 'cics_receive_map':
            if not self._has_required_params(cmd_params, ['MAP', 'MAPSET']):
                violations.append("RECEIVE MAP requires both MAP and MAPSET parameters")
        
        # File operation validations
        if cmd_type in ['cics_read', 'cics_write', 'cics_rewrite', 'cics_delete']:
            if not self._has_required_params(cmd_params, ['FILE']):
                violations.append(f"{cmd_type} requires FILE parameter")
        
        # Error handling validations
        has_resp = 'RESP(' in cmd_params.upper()
        has_nohandle = 'NOHANDLE' in cmd_params.upper()
        
        if not has_resp and not has_nohandle:
            warnings.append("No error handling specified (RESP or NOHANDLE)")
        
        if has_resp and has_nohandle:
            violations.append("RESP and NOHANDLE are mutually exclusive")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'error_handling_present': has_resp or has_nohandle
        }

    def _update_transaction_state(self, state: TransactionState, cmd: Dict):
        """Update transaction state based on CICS command"""
        cmd_type = cmd['type']
        cmd_params = cmd['params']
        
        if cmd_type == 'cics_receive' or cmd_type == 'cics_receive_map':
            state.set_input_received()
        
        if cmd_type == 'cics_send_map':
            mapset = self._extract_parameter_value(cmd_params, 'MAPSET')
            map_name = self._extract_parameter_value(cmd_params, 'MAP')
            if mapset and map_name:
                state.set_map_loaded(mapset, map_name)
        
        if cmd_type in ['cics_read', 'cics_write']:
            file_name = self._extract_parameter_value(cmd_params, 'FILE')
            if file_name:
                state.file_opened[file_name] = True

    # Business Validator Classes
    async def _parse_jcl_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse JCL with business rule validation"""
        chunks = []
        job_name = self._extract_program_name(content, Path(filename))
        
        # Validate JCL structure
        jcl_validation = await self._validate_jcl_structure(content, job_name)
        
        # Parse with execution flow analysis
        job_chunks = await self._parse_jcl_job_with_flow(content, job_name)
        chunks.extend(job_chunks)
        
        # Parse steps with dependency analysis
        step_chunks = await self._parse_jcl_steps_with_dependencies(content, job_name)
        chunks.extend(step_chunks)
        
        return chunks

    async def _validate_jcl_structure(self, content: str, job_name: str) -> Dict[str, Any]:
        """Validate JCL structure according to business rules"""
        violations = []
        
        # Check for required JOB card
        job_match = self.jcl_patterns['job_card'].search(content)
        if not job_match:
            violations.append(BusinessRuleViolation(
                rule="MISSING_JOB_CARD",
                context="JCL must start with JOB card",
                severity="ERROR"
            ))
        
        # Check for at least one EXEC step
        exec_matches = list(self.jcl_patterns['job_step'].finditer(content))
        if not exec_matches:
            violations.append(BusinessRuleViolation(
                rule="NO_EXEC_STEPS",
                context="JCL must have at least one EXEC statement",
                severity="ERROR"
            ))
        
        # Validate step dependencies and return code logic
        step_dependencies = self._analyze_jcl_step_dependencies(content)
        
        return {
            'structure_valid': len([v for v in violations if v.severity == "ERROR"]) == 0,
            'violations': violations,
            'step_dependencies': step_dependencies
        }

    async def _parse_jcl_steps_with_dependencies(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL steps with dependency and execution flow analysis"""
        chunks = []
        
        step_matches = list(self.jcl_patterns['job_step'].finditer(content))
        
        for i, match in enumerate(step_matches):
            step_name = match.group(1)
            
            # Extract complete step content
            step_content = self._extract_jcl_step_complete(content, match.start(), step_matches, i)
            
            # Analyze step dependencies and execution conditions
            business_context = await self._analyze_jcl_step_business_context(step_content, step_name, i)
            
            # Enhanced metadata with execution flow
            metadata = await self._analyze_jcl_step_comprehensive(step_content, step_name)
            metadata.update({
                'step_sequence': i + 1,
                'execution_dependencies': business_context.get('dependencies', []),
                'conditional_execution': business_context.get('conditional_logic', {}),
                'resource_requirements': business_context.get('resources', {})
            })
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_STEP_{step_name}",
                chunk_type="jcl_step",
                content=step_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(step_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    # Additional business context analysis methods
    def _analyze_jcl_step_dependencies(self, content: str) -> Dict[str, Any]:
        """Analyze JCL step execution dependencies"""
        dependencies = {
            'conditional_steps': [],
            'restart_points': [],
            'return_code_checks': []
        }
        
        # Find COND parameters
        cond_matches = self.jcl_patterns['cond_parameter'].finditer(content)
        for match in cond_matches:
            dependencies['conditional_steps'].append({
                'condition': match.group(1),
                'position': match.start()
            })
        
        # Find RESTART parameters
        restart_matches = self.jcl_patterns['restart_parameter'].finditer(content)
        for match in restart_matches:
            dependencies['restart_points'].append(match.group(1))
        
        return dependencies

    async def _store_business_violations(self, violations: List[BusinessRuleViolation], program_name: str):
        """Store business rule violations in database"""
        if not violations:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for violation in violations:
                cursor.execute("""
                    INSERT INTO business_rule_violations 
                    (program_name, rule_type, rule_name, severity, description, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    type(violation).__name__,
                    violation.rule,
                    violation.severity,
                    str(violation),
                    violation.context
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(violations)} business rule violations for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store business violations: {str(e)}")

    async def _analyze_control_flow(self, chunks: List[CodeChunk]) -> List[ControlFlowPath]:
        """Analyze control flow paths through the program"""
        control_paths = []
        
        # Extract paragraphs and PERFORM statements
        paragraphs = [c for c in chunks if c.chunk_type == "paragraph"]
        performs = [c for c in chunks if c.chunk_type == "perform_statement"]
        
        # Build control flow graph
        for paragraph in paragraphs:
            path = ControlFlowPath(
                path_id=f"{paragraph.program_name}_{paragraph.chunk_id}_PATH",
                entry_point=paragraph.metadata.get('paragraph_name', ''),
                exit_points=[],
                conditions=[],
                called_paragraphs=[],
                data_accessed=[]
            )
            
            # Find PERFORM statements that call this paragraph
            calling_performs = [p for p in performs 
                             if paragraph.metadata.get('paragraph_name', '') in p.content]
            
            # Analyze data access patterns
            path.data_accessed = paragraph.metadata.get('field_names', [])
            
            # Extract conditions from IF statements
            if_statements = re.findall(r'IF\s+(.*?)(?:\s+THEN|\s|$)', paragraph.content, re.IGNORECASE)
            path.conditions = if_statements
            
            control_paths.append(path)
        
        return control_paths

    async def _store_control_flow_analysis(self, control_paths: List[ControlFlowPath], program_name: str):
        """Store control flow analysis in database"""
        if not control_paths:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for path in control_paths:
                cursor.execute("""
                    INSERT INTO control_flow_paths 
                    (program_name, path_id, entry_point, exit_points, conditions, 
                     called_paragraphs, data_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    path.path_id,
                    path.entry_point,
                    json.dumps(path.exit_points),
                    json.dumps(path.conditions),
                    json.dumps(path.called_paragraphs),
                    json.dumps(path.data_accessed)
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(control_paths)} control flow paths for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store control flow analysis: {str(e)}")

    # Enhanced storage with business context
    async def _store_chunks_enhanced(self, chunks: List[CodeChunk], file_hash: str):
        """Store chunks with enhanced business context - KEEPING LLM CALLS UNCHANGED"""
        if not chunks:
            self.logger.warning("No chunks to store")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    stored_count = 0
                    
                    for chunk in chunks:
                        try:
                            program_name = str(chunk.program_name)
                            chunk_id = str(chunk.chunk_id)
                            chunk_type = str(chunk.chunk_type)
                            content = str(chunk.content)
                            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
                            business_context_json = json.dumps(chunk.business_context) if chunk.business_context else "{}"
                            embedding_id = hashlib.md5(content.encode()).hexdigest()
                            
                            cursor.execute("""
                                SELECT id FROM program_chunks 
                                WHERE program_name = ? AND chunk_id = ?
                            """, (program_name, chunk_id))
                            
                            existing = cursor.fetchone()
                            
                            if existing:
                                cursor.execute("""
                                    UPDATE program_chunks 
                                    SET content = ?, metadata = ?, business_context = ?, 
                                        file_hash = ?, line_start = ?, line_end = ?, 
                                        created_timestamp = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                """, (
                                    content, metadata_json, business_context_json,
                                    str(file_hash), int(chunk.line_start), 
                                    int(chunk.line_end), existing[0]
                                ))
                            else:
                                cursor.execute("""
                                    INSERT INTO program_chunks 
                                    (program_name, chunk_id, chunk_type, content, metadata, 
                                     business_context, embedding_id, file_hash, line_start, line_end)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    program_name, chunk_id, chunk_type, content, 
                                    metadata_json, business_context_json, embedding_id,
                                    str(file_hash), int(chunk.line_start), int(chunk.line_end)
                                ))
                            
                            stored_count += 1
                            
                        except sqlite3.Error as e:
                            self.logger.error(f"Failed to store chunk {chunk.chunk_id}: {str(e)}")
                            continue
                    
                    cursor.execute("COMMIT")
                    self.logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks")
                    
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    self.logger.error(f"Transaction failed, rolled back: {str(e)}")
                    raise e
                    
        except Exception as e:
            self.logger.error(f"Database operation failed: {str(e)}")
            raise e

    # ALL LLM ANALYSIS METHODS REMAIN UNCHANGED TO PRESERVE LLM CALLS
    async def _analyze_division_with_llm(self, content: str, division_name: str) -> Dict[str, Any]:
        """Analyze COBOL division with LLM - UNCHANGED"""
        prompt = f"""
        Analyze this COBOL {division_name}:
        
        {content[:800]}...
        
        Extract key information:
        1. Main purpose and functionality
        2. Key elements defined
        3. Dependencies and relationships
        4. Configuration or setup details
        
        Return as JSON:
        {{
            "purpose": "main purpose",
            "key_elements": ["element1", "element2"],
            "dependencies": ["dep1", "dep2"],
            "configuration": "setup details"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Division analysis failed: {str(e)}")
        
        return {
            "purpose": f"{division_name} processing",
            "key_elements": [],
            "dependencies": [],
            "configuration": "Standard COBOL division"
        }

    async def _analyze_paragraph_with_llm(self, content: str) -> Dict[str, Any]:
        """Analyze paragraph with LLM - UNCHANGED"""
        prompt = f"""
        Analyze this COBOL paragraph:
        
        {content[:600]}...
        
        Extract:
        1. Field names referenced
        2. File operations performed
        3. Database operations (SQL statements)
        4. Called paragraphs (PERFORM statements)
        5. Main purpose/operation
        6. Error handling used
        
        Return as JSON:
        {{
            "field_names": ["field1", "field2"],
            "file_operations": ["READ FILE1", "WRITE FILE2"],
            "sql_operations": ["SELECT", "UPDATE"],
            "called_paragraphs": ["PARA1", "PARA2"],
            "main_purpose": "description",
            "error_handling": ["pattern1", "pattern2"]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=500)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Paragraph analysis failed: {str(e)}")
        
        return {
            "field_names": self._extract_field_names(content),
            "file_operations": self._extract_file_operations(content),
            "sql_operations": self._extract_sql_operations(content),
            "called_paragraphs": self._extract_perform_statements(content),
            "main_purpose": "Code processing",
            "error_handling": self._extract_error_handling_patterns(content)
        }

    async def _analyze_sql_comprehensive(self, sql_content: str) -> Dict[str, Any]:
        """Comprehensive SQL analysis - UNCHANGED"""
        prompt = f"""
        Perform comprehensive analysis of this SQL statement:
        
        {sql_content}
        
        Extract and analyze:
        1. SQL operation type and subtype
        2. All tables accessed with their roles
        3. All columns with operations
        4. Join conditions and types
        5. WHERE clause analysis
        6. Subqueries and their purposes
        7. Functions and expressions
        8. Performance considerations
        9. Business logic implemented
        
        Return as JSON:
        {{
            "operation_type": "SELECT|INSERT|UPDATE|DELETE|etc",
            "operation_subtype": "simple|complex|bulk|etc",
            "tables": [
                {{"name": "table1", "role": "source|target|lookup", "alias": "t1"}}
            ],
            "columns": [
                {{"name": "col1", "table": "table1", "operation": "select|update|filter"}}
            ],
            "joins": [
                {{"type": "INNER|LEFT|RIGHT", "condition": "condition", "tables": ["t1", "t2"]}}
            ],
            "conditions": [
                {{"type": "WHERE|HAVING", "expression": "expression", "purpose": "filter"}}
            ],
            "subqueries": [
                {{"type": "EXISTS|IN|SCALAR", "purpose": "purpose"}}
            ],
            "functions": ["func1", "func2"],
            "performance_notes": "performance analysis",
            "business_logic": "business rules"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=800)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"SQL analysis failed: {str(e)}")
        
        return {
            "operation_type": self._extract_sql_type(sql_content),
            "tables": [{"name": table, "role": "unknown", "alias": None} 
                      for table in self._extract_table_names(sql_content)],
            "columns": [],
            "joins": [],
            "conditions": [],
            "subqueries": [],
            "functions": [],
            "performance_notes": "Analysis not available",
            "business_logic": "SQL operation"
        }

    async def _analyze_cics_command_comprehensive(self, command_type: str, params: str, content: str) -> Dict[str, Any]:
        """Comprehensive CICS command analysis - UNCHANGED"""
        prompt = f"""
        Analyze this CICS command:
        
        Command Type: {command_type}
        Parameters: {params}
        Full Command: {content}
        
        Provide analysis:
        1. Command category and purpose
        2. Key parameters and their values
        3. Resource accessed (file, map, program, etc.)
        4. Transaction flow impact
        5. Error conditions handled
        6. Performance implications
        
        Return as JSON:
        {{
            "category": "terminal|file|program_control|storage|etc",
            "purpose": "purpose description",
            "key_parameters": {{"param1": "value1", "param2": "value2"}},
            "resource_accessed": "resource name",
            "flow_impact": "impact description",
            "error_conditions": ["cond1", "cond2"],
            "performance_impact": "impact analysis"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"CICS command analysis failed: {str(e)}")
        
            return {
                "category": self._categorize_cics_command(command_type),
                "purpose": f"{command_type} operation",
                "key_parameters": self._extract_cics_parameters(params),
                "resource_accessed": self._extract_cics_resource(params),
                "flow_impact": "CICS transaction processing",
                "error_conditions": [],
                "performance_impact": "Standard CICS overhead"
            }