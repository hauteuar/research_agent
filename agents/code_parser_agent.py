"""
Agent 1: Complete Enhanced Code Parser & Chunker - FIXED VERSION
Handles COBOL, JCL, CICS, BMS, and Copybook parsing with intelligent chunking
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
from datetime import datetime, time
import logging

import torch
from vllm import AsyncLLMEngine, SamplingParams

def _ensure_airgap_environment(self):
    """Ensure no external connections are possible"""
    import os
    
    # Set environment variables to disable external connections
    os.environ.update({
        'NO_PROXY': '*',
        'DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
        'REQUESTS_CA_BUNDLE': '',
        'CURL_CA_BUNDLE': ''
    })
    
    # Disable requests library
    try:
        import requests
        def blocked_request(*args, **kwargs):
            raise requests.exceptions.ConnectionError("External connections disabled")
        
        requests.get = blocked_request
        requests.post = blocked_request
        requests.request = blocked_request
    except ImportError:
        pass


@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""
    program_name: str
    chunk_id: str
    chunk_type: str  # paragraph, perform, job_step, proc, sql_block, section, cics_command
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int

class CompleteEnhancedCodeParserAgent:
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
                 gpu_id: int = None, coordinator=None):
        self.llm_engine = llm_engine
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._engine_lock = asyncio.Lock()
        self._engine_created = False
        self._using_coordinator_llm = False
        self._processed_files = set()  # Duplicate prevention
        self._request_semaphore = asyncio.Semaphore(1)  # Prevent request flooding
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum 1 second between LLM requests
        
        self.processing_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "errors_encountered": 0,
            "llm_calls_made": 0,
            "fallback_used": 0
        }
        
        # Initialize database first
        self._init_database()
        
        # COMPLETE COBOL PATTERNS
        self.cobol_patterns = {
            # Basic identification
            'program_id': re.compile(r'PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'author': re.compile(r'AUTHOR\s*\.\s*(.*?)\.', re.IGNORECASE | re.DOTALL),
            'date_written': re.compile(r'DATE-WRITTEN\s*\.\s*(.*?)\.', re.IGNORECASE),
            'date_compiled': re.compile(r'DATE-COMPILED\s*\.\s*(.*?)\.', re.IGNORECASE),
            
            # Divisions and sections
            'identification_division': re.compile(r'IDENTIFICATION\s+DIVISION', re.IGNORECASE),
            'environment_division': re.compile(r'ENVIRONMENT\s+DIVISION', re.IGNORECASE),
            'data_division': re.compile(r'DATA\s+DIVISION', re.IGNORECASE),
            'procedure_division': re.compile(r'PROCEDURE\s+DIVISION', re.IGNORECASE),
            'working_storage': re.compile(r'WORKING-STORAGE\s+SECTION', re.IGNORECASE),
            'file_section': re.compile(r'FILE\s+SECTION', re.IGNORECASE),
            'linkage_section': re.compile(r'LINKAGE\s+SECTION', re.IGNORECASE),
            'local_storage': re.compile(r'LOCAL-STORAGE\s+SECTION', re.IGNORECASE),
            'section': re.compile(r'^([A-Z0-9][A-Z0-9-]*)\s+SECTION\s*\.\s*$', re.MULTILINE),
            
            # Paragraphs and control structures
            'paragraph': re.compile(r'^([A-Z0-9][A-Z0-9-]*)\s*\.\s*$', re.MULTILINE),
            'perform': re.compile(r'PERFORM\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_until': re.compile(r'PERFORM\s+.*?\s+UNTIL\s+(.*?)(?=\s+|$)', re.IGNORECASE | re.DOTALL),
            'perform_varying': re.compile(r'PERFORM\s+.*?\s+VARYING\s+(.*?)(?=\s+|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'PERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            
            # Control flow
            'if_statement': re.compile(r'IF\s+(.*?)(?=\s+THEN|\s|$)', re.IGNORECASE),
            'evaluate': re.compile(r'EVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'when_clause': re.compile(r'WHEN\s+([^\.]+)', re.IGNORECASE),
            'when_other': re.compile(r'WHEN\s+OTHER', re.IGNORECASE),
            'end_if': re.compile(r'END-IF', re.IGNORECASE),
            'end_evaluate': re.compile(r'END-EVALUATE', re.IGNORECASE),
            
            # Data definitions
            'data_item': re.compile(r'^(\s*)(\d+)\s+([A-Z][A-Z0-9-]*)\s+(.*?)\.?\s*$', re.MULTILINE),
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+([X9AV\(\)S\+\-\.,/]+)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX)', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES', re.IGNORECASE),
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'depending_on': re.compile(r'DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'indexed_by': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # File operations
            'file_control': re.compile(r'FILE-CONTROL\s*\.', re.IGNORECASE),
            'select_statement': re.compile(r'SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)', re.IGNORECASE),
            'fd_statement': re.compile(r'FD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'open_statement': re.compile(r'OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'close_statement': re.compile(r'CLOSE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'read_statement': re.compile(r'READ\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'write_statement': re.compile(r'WRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'rewrite_statement': re.compile(r'REWRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'delete_statement': re.compile(r'DELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # SQL blocks
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare': re.compile(r'EXEC\s+SQL\s+DECLARE\s+(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            
            # COPY statements
            'copy_statement': re.compile(r'COPY\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_replacing': re.compile(r'COPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'COPY\s+([A-Z][A-Z0-9-]*)\s+IN\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Math operations
            'add_statement': re.compile(r'ADD\s+(.*?)\s+TO\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'subtract_statement': re.compile(r'SUBTRACT\s+(.*?)\s+FROM\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'multiply_statement': re.compile(r'MULTIPLY\s+(.*?)\s+BY\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'divide_statement': re.compile(r'DIVIDE\s+(.*?)\s+(?:BY|INTO)\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'compute_statement': re.compile(r'COMPUTE\s+(.*?)\s*=\s*(.*?)(?=\s|$)', re.IGNORECASE),
            
            # String operations
            'move_statement': re.compile(r'MOVE\s+(.*?)\s+TO\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'string_statement': re.compile(r'STRING\s+(.*?)(?=\s+END-STRING|$)', re.IGNORECASE | re.DOTALL),
            'unstring_statement': re.compile(r'UNSTRING\s+(.*?)(?=\s+END-UNSTRING|$)', re.IGNORECASE | re.DOTALL),
            'inspect_statement': re.compile(r'INSPECT\s+(.*?)(?=\s|$)', re.IGNORECASE),
            
            # Call statements
            'call_statement': re.compile(r'CALL\s+([\'"]?[A-Z0-9][A-Z0-9-]*[\'"]?)', re.IGNORECASE),
            'invoke_statement': re.compile(r'INVOKE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Error handling
            'on_size_error': re.compile(r'ON\s+SIZE\s+ERROR', re.IGNORECASE),
            'not_on_size_error': re.compile(r'NOT\s+ON\s+SIZE\s+ERROR', re.IGNORECASE),
            'at_end': re.compile(r'AT\s+END', re.IGNORECASE),
            'not_at_end': re.compile(r'NOT\s+AT\s+END', re.IGNORECASE),
            'invalid_key': re.compile(r'INVALID\s+KEY', re.IGNORECASE),
            'not_invalid_key': re.compile(r'NOT\s+INVALID\s+KEY', re.IGNORECASE),

            'cics_transaction_flow': re.compile(r'EXEC\s+CICS\s+(SEND|RECEIVE|RETURN)', re.IGNORECASE),
            'cics_error_handling': re.compile(r'EXEC\s+CICS\s+HANDLE\s+(CONDITION|AID|ABEND)', re.IGNORECASE),
        }
        
        # COMPLETE JCL PATTERNS
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
            'include_statement': re.compile(r'^//\s+INCLUDE\s+MEMBER=([A-Z0-9]+)', re.MULTILINE),
            'jcllib_statement': re.compile(r'^//\s+JCLLIB\s+ORDER=\((.*?)\)', re.MULTILINE),
            'output_statement': re.compile(r'^//\s+OUTPUT\s+(.*?)$', re.MULTILINE),
        }
        
        # COMPLETE CICS PATTERNS
        self.cics_patterns = {
            # Terminal operations
            'cics_send_map': re.compile(r'EXEC\s+CICS\s+SEND\s+MAP\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive_map': re.compile(r'EXEC\s+CICS\s+RECEIVE\s+MAP\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_send_text': re.compile(r'EXEC\s+CICS\s+SEND\s+TEXT\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_send_page': re.compile(r'EXEC\s+CICS\s+SEND\s+PAGE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive': re.compile(r'EXEC\s+CICS\s+RECEIVE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # File operations
            'cics_read': re.compile(r'EXEC\s+CICS\s+READ\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_write': re.compile(r'EXEC\s+CICS\s+WRITE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_rewrite': re.compile(r'EXEC\s+CICS\s+REWRITE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_delete': re.compile(r'EXEC\s+CICS\s+DELETE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_startbr': re.compile(r'EXEC\s+CICS\s+STARTBR\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readnext': re.compile(r'EXEC\s+CICS\s+READNEXT\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readprev': re.compile(r'EXEC\s+CICS\s+READPREV\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_endbr': re.compile(r'EXEC\s+CICS\s+ENDBR\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_unlock': re.compile(r'EXEC\s+CICS\s+UNLOCK\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Program control
            'cics_link': re.compile(r'EXEC\s+CICS\s+LINK\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_xctl': re.compile(r'EXEC\s+CICS\s+XCTL\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_load': re.compile(r'EXEC\s+CICS\s+LOAD\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_release': re.compile(r'EXEC\s+CICS\s+RELEASE\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_return': re.compile(r'EXEC\s+CICS\s+RETURN\s*(?:\((.*?)\))?\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_abend': re.compile(r'EXEC\s+CICS\s+ABEND\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Storage operations
            'cics_getmain': re.compile(r'EXEC\s+CICS\s+GETMAIN\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_freemain': re.compile(r'EXEC\s+CICS\s+FREEMAIN\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Task control
            'cics_suspend': re.compile(r'EXEC\s+CICS\s+SUSPEND\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_resume': re.compile(r'EXEC\s+CICS\s+RESUME\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_start': re.compile(r'EXEC\s+CICS\s+START\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_cancel': re.compile(r'EXEC\s+CICS\s+CANCEL\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Interval control
            'cics_delay': re.compile(r'EXEC\s+CICS\s+DELAY\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_post': re.compile(r'EXEC\s+CICS\s+POST\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_wait': re.compile(r'EXEC\s+CICS\s+WAIT\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Error handling
            'cics_handle_condition': re.compile(r'EXEC\s+CICS\s+HANDLE\s+CONDITION\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_handle_aid': re.compile(r'EXEC\s+CICS\s+HANDLE\s+AID\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_handle_abend': re.compile(r'EXEC\s+CICS\s+HANDLE\s+ABEND\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_push_handle': re.compile(r'EXEC\s+CICS\s+PUSH\s+HANDLE\s*END-EXEC', re.IGNORECASE),
            'cics_pop_handle': re.compile(r'EXEC\s+CICS\s+POP\s+HANDLE\s*END-EXEC', re.IGNORECASE),
            
            # Syncpoint operations
            'cics_syncpoint': re.compile(r'EXEC\s+CICS\s+SYNCPOINT\s*(?:\((.*?)\))?\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_syncpoint_rollback': re.compile(r'EXEC\s+CICS\s+SYNCPOINT\s+ROLLBACK\s*END-EXEC', re.IGNORECASE),
            
            # Temporary storage
            'cics_writeq_ts': re.compile(r'EXEC\s+CICS\s+WRITEQ\s+TS\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readq_ts': re.compile(r'EXEC\s+CICS\s+READQ\s+TS\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_deleteq_ts': re.compile(r'EXEC\s+CICS\s+DELETEQ\s+TS\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Transient data
            'cics_writeq_td': re.compile(r'EXEC\s+CICS\s+WRITEQ\s+TD\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readq_td': re.compile(r'EXEC\s+CICS\s+READQ\s+TD\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_deleteq_td': re.compile(r'EXEC\s+CICS\s+DELETEQ\s+TD\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
        }
        
        # COMPLETE BMS PATTERNS
        self.bms_patterns = {
            'bms_mapset': re.compile(r'(\w+)\s+DFHMSD\s+(.*?)(?=\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_map': re.compile(r'(\w+)\s+DFHMDI\s+(.*?)(?=\w+\s+DFHMDI|\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_field': re.compile(r'(\w+)\s+DFHMDF\s+(.*?)(?=\w+\s+DFHMDF|\w+\s+DFHMDI|\w+\s+DFHMSD|$)', re.IGNORECASE | re.DOTALL),
            'bms_mapset_end': re.compile(r'\s+DFHMSD\s+TYPE=FINAL', re.IGNORECASE),
        }
        
        # Initialize database
        self._init_database()
        self._disable_external_connections()
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processing_stats": self.processing_stats.copy(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "llm_available": self._llm_available,
            "initialized": self._initialized,
            "gpu_used": self.gpu_id,
            "timestamp": datetime.now().isoformat()
        }

    def _disable_external_connections(self):
        """Completely disable external connections for airgap mode"""
        import os
        import socket
        
        # Set environment variables
        os.environ.update({
            'NO_PROXY': '*',
            'DISABLE_TELEMETRY': '1',
            'TOKENIZERS_PARALLELISM': 'false',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_OFFLINE': '1',
            'REQUESTS_CA_BUNDLE': '',
            'CURL_CA_BUNDLE': ''
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

    def _init_database(self):
        """Initialize database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced table with duplicate prevention
            cursor.execute("""
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
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_name ON program_chunks(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON program_chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON program_chunks(file_hash)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")

    def _generate_file_hash(self, content: str, file_path: Path) -> str:
        """Generate unique hash for file content and metadata"""
        hash_input = f"{file_path.name}:{file_path.stat().st_mtime}:{len(content)}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _is_duplicate_file(self, file_path: Path, content: str) -> bool:
        """Check if file has already been processed"""
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

    async def _ensure_llm_engine(self):
        """Ensure LLM engine is available with thread safety"""
        async with self._engine_lock:
            if self.llm_engine is not None:
                return
            
            # Try coordinator first
            if self.coordinator is not None:
                try:
                    existing_engines = list(self.coordinator.llm_engine_pool.keys())
                    
                    for engine_key in existing_engines:
                        gpu_id = int(engine_key.split('_')[1])
                        
                        try:
                            from gpu_force_fix import GPUForcer
                            memory_info = GPUForcer.check_gpu_memory(gpu_id)
                            free_gb = memory_info.get('free_gb', 0)
                            
                            if free_gb >= 1.0:
                                self.llm_engine = self.coordinator.llm_engine_pool[engine_key]
                                self.gpu_id = gpu_id
                                self._using_coordinator_llm = True
                                self.logger.info(f"CodeParser SHARING coordinator's LLM on GPU {gpu_id}")
                                return
                        except Exception as e:
                            self.logger.warning(f"Error checking GPU {gpu_id} for sharing: {e}")
                            continue
                    
                    best_gpu = await self.coordinator.get_available_gpu_for_agent("code_parser")
                    if best_gpu is not None:
                        engine = await self.coordinator.get_or_create_llm_engine(best_gpu)
                        self.llm_engine = engine
                        self.gpu_id = best_gpu
                        self._using_coordinator_llm = True
                        self.logger.info(f"CodeParser using coordinator's NEW LLM on GPU {best_gpu}")
                        return
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get LLM from coordinator: {e}")
            
            # Try global coordinator as fallback
            if not self._engine_created:
                try:
                    from opulence_coordinator import get_dynamic_coordinator
                    global_coordinator = get_dynamic_coordinator()
                    
                    existing_engines = list(global_coordinator.llm_engine_pool.keys())
                    for engine_key in existing_engines:
                        gpu_id = int(engine_key.split('_')[1])
                        try:
                            from gpu_force_fix import GPUForcer
                            memory_info = GPUForcer.check_gpu_memory(gpu_id)
                            free_gb = memory_info.get('free_gb', 0)
                            
                            if free_gb >= 1.0:
                                self.llm_engine = global_coordinator.llm_engine_pool[engine_key]
                                self.gpu_id = gpu_id
                                self._using_coordinator_llm = True
                                self.logger.info(f"CodeParser SHARING global coordinator's LLM on GPU {gpu_id}")
                                return
                        except Exception as e:
                            continue
                    
                    best_gpu = await global_coordinator.get_available_gpu_for_agent("code_parser")
                    if best_gpu is not None:
                        engine = await global_coordinator.get_or_create_llm_engine(best_gpu)
                        self.llm_engine = engine
                        self.gpu_id = best_gpu
                        self._using_coordinator_llm = True
                        self.logger.info(f"CodeParser using global coordinator's NEW LLM on GPU {best_gpu}")
                        return
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get LLM from global coordinator: {e}")
            
            if not self._engine_created:
                await self._create_fallback_llm_engine()

    async def _create_fallback_llm_engine(self):
        """Create own LLM engine as last resort"""
        try:
            from gpu_force_fix import GPUForcer
            
            best_gpu = None
            best_memory = 0
            
            for gpu_id in range(4):
                try:
                    memory_info = GPUForcer.check_gpu_memory(gpu_id)
                    free_gb = memory_info.get('free_gb', 0)
                    if free_gb > best_memory:
                        best_memory = free_gb
                        best_gpu = gpu_id
                except:
                    continue
            
            if best_gpu is None or best_memory < 0.5:
                raise RuntimeError(f"No GPU found with sufficient memory. Best: {best_memory:.1f}GB")
            
            self.logger.warning(f"CodeParser creating FALLBACK LLM on GPU {best_gpu} with {best_memory:.1f}GB")
            
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                GPUForcer.force_gpu_environment(best_gpu)
                
                engine_args = GPUForcer.create_vllm_engine_args(
                    "microsoft/DialoGPT-small",
                    1024
                )
                engine_args.gpu_memory_utilization = 0.2
                
                from vllm import AsyncLLMEngine
                self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.gpu_id = best_gpu
                self._engine_created = True
                self._using_coordinator_llm = False
                
                self.logger.info(f"✅ CodeParser fallback LLM created on GPU {best_gpu}")
                
            finally:
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                    
        except Exception as e:
            self.logger.error(f"Failed to create fallback LLM engine: {str(e)}")
            raise

    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM - FIXED async generator handling"""
        
        # Rate limiting to prevent request flooding
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        async with self._request_semaphore:  # Prevent concurrent requests
            try:
                if self.llm_engine is None:
                    self.logger.warning("LLM engine not available, using fallback")
                    self.processing_stats["fallback_used"] += 1
                    return ""
                
                request_id = str(uuid.uuid4())
                self._last_request_time = time.time()
                
                try:
                    # FIXED: Use async for instead of await
                    result_generator = self.llm_engine.generate(
                        prompt, sampling_params, request_id=request_id
                    )
                    
                    # Properly iterate through async generator
                    async for result in result_generator:
                        if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                            self.processing_stats["llm_calls_made"] += 1
                            return result.outputs[0].text.strip()
                        break  # Take first result only
                        
                except TypeError as e:
                    if "request_id" in str(e):
                        # Fallback to old API
                        result_generator = self.llm_engine.generate(prompt, sampling_params)
                        async for result in result_generator:
                            if result and hasattr(result, 'outputs') and len(result.outputs) > 0:
                                self.processing_stats["llm_calls_made"] += 1
                                return result.outputs[0].text.strip()
                            break
                    else:
                        raise e
                
                return ""
                
            except Exception as e:
                self.logger.error(f"LLM generation failed: {str(e)}")
                self.processing_stats["fallback_used"] += 1
                return ""

        
    def _extract_program_name(self, content: str, file_path: Path) -> str:
        """Extract program name more robustly from content or filename"""
        try:
            # For COBOL files, try PROGRAM-ID first
            program_match = self.cobol_patterns['program_id'].search(content)
            if program_match:
                return program_match.group(1).strip()
            
            # For JCL files, try JOB name
            job_match = self.jcl_patterns['job_card'].search(content)
            if job_match:
                return job_match.group(1).strip()
            
            # Fallback to filename
            if isinstance(file_path, str):
                file_path = Path(file_path)
            filename = file_path.name
            
            # Remove common extensions
            for ext in ['.cbl', '.cob', '.jcl', '.copy', '.cpy', '.bms']:
                if filename.lower().endswith(ext):
                    return filename[:-len(ext)]
            
            # Remove any temp directory prefixes and just use the base name
            return file_path.stem
            
        except Exception as e:
            self.logger.error(f"Error extracting program name: {str(e)}")
            # Ultimate fallback
            if isinstance(file_path, (str, Path)):
                return Path(file_path).stem or "UNKNOWN_PROGRAM"
            return file_path.stem or "UNKNOWN_PROGRAM"
        
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'enhanced_code_parser'
            result['using_coordinator_llm'] = self._using_coordinator_llm
        return result

    async def process_file(self, file_path) -> Dict[str, Any]:
        """Process file with enhanced error handling and type safety"""
        try:
            # FIXED: Ensure proper type handling
            if isinstance(file_path, str):
                file_path = Path(file_path)
            elif not isinstance(file_path, Path):
                return self._create_error_result(
                    f"Invalid file path type: {type(file_path)}", 
                    str(file_path)
                )
            
            # Validate file exists
            if not file_path.exists():
                return self._create_error_result("File does not exist", str(file_path.name))
            
            if not file_path.is_file():
                return self._create_error_result("Path is not a file", str(file_path.name))
            
            # FIXED: Safe file reading with multiple encodings
            content = None
            encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    self.logger.debug(f"Successfully read file with {encoding} encoding")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None:
                return self._create_error_result("Could not read file with any encoding", str(file_path.name))
            
            if not content.strip():
                return self._create_error_result("File is empty or contains only whitespace", str(file_path.name))
            
            # FIXED: Ensure all strings are properly converted
            file_name_str = str(file_path.name)
            content_str = str(content)
            
            # Check for duplicates
            file_hash = self._generate_file_hash(content_str, file_path)
            if self._is_duplicate_file(file_path, content_str):
                return self._create_success_result(
                    file_name_str, "skipped", 0, 
                    message="File already processed (duplicate detected)"
                )
            
            # Detect file type
            file_type = self._detect_file_type(content_str, str(file_path.suffix))
            
            # Parse based on file type with fallback
            chunks = []
            try:
                if file_type == 'cobol':
                    chunks = await self._parse_cobol_safe(content_str, file_name_str)
                elif file_type == 'jcl':
                    chunks = await self._parse_jcl_safe(content_str, file_name_str)
                elif file_type == 'copybook':
                    chunks = await self._parse_copybook_complete(content, str(file_path.name))
                elif file_type == 'bms':
                    chunks = await self._parse_bms_complete(content, str(file_path.name))
                elif file_type == 'cics':
                    chunks = await self._parse_cics_complete(content, str(file_path.name))
                elif file_type == 'sql':
                    chunks = await self._parse_sql_blocks_complete(content_str, file_name_str)   
                else:
                    chunks = await self._parse_generic_safe(content_str, file_name_str)
            except Exception as parse_error:
                self.logger.error(f"Parsing failed for {file_name_str}: {str(parse_error)}")
                # Create fallback chunk
                chunks = [self._create_fallback_chunk(content_str, file_name_str, file_type)]
            
            if not chunks:
                # Create fallback chunk if no chunks were created
                chunks = [self._create_fallback_chunk(content_str, file_name_str, file_type)]
            
            # Store chunks safely
            try:
                await self._store_chunks_safe(chunks, file_hash)
                self.processing_stats["files_processed"] += 1
                self.processing_stats["chunks_created"] += len(chunks)
            except Exception as store_error:
                self.logger.error(f"Failed to store chunks: {str(store_error)}")
                return self._create_error_result(f"Storage failed: {str(store_error)}", file_name_str)
            
            return self._create_success_result(
                file_name_str, "success", len(chunks),
                file_type=file_type, file_hash=file_hash
            )
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.processing_stats["errors_encountered"] += 1
            
            file_name_safe = str(file_path.name) if hasattr(file_path, 'name') else str(file_path)
            self.logger.error(f"Processing failed for {file_name_safe}: {str(e)}")
            
            return self._create_error_result(str(e), file_name_safe)

    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Detect the type of mainframe file with enhanced detection"""
        content_upper = content.upper()
        
        # COBOL detection
        if any(marker in content_upper for marker in ['IDENTIFICATION DIVISION', 'PROGRAM-ID', 'WORKING-STORAGE SECTION']):
            return 'cobol'
        
        # JCL detection
        if content.strip().startswith('//') and any(marker in content_upper for marker in ['JOB', 'EXEC', 'DD']):
            return 'jcl'
        
        # BMS detection
        if any(marker in content_upper for marker in ['DFHMSD', 'DFHMDI', 'DFHMDF']):
            return 'bms'
        
        # CICS detection (heavy CICS usage)
        cics_count = content_upper.count('EXEC CICS')
        if cics_count > 5:  # More than 5 CICS commands suggests CICS program
            return 'cics'
        
        # Copybook detection
        if 'COPY' in content_upper and len(content.split('\n')) < 200:
            return 'copybook'
        
        # Extension-based detection
        suffix_lower = suffix.lower()
        if suffix_lower in ['.cbl', '.cob']:
            return 'cobol'
        elif suffix_lower == '.jcl':
            return 'jcl'
        elif suffix_lower in ['.cpy', '.copy']:
            return 'copybook'
        elif suffix_lower == '.bms':
            return 'bms'
        else:
            return 'unknown'

    def _create_error_result(self, error_msg: str, file_name: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "status": "error",
            "file_name": str(file_name),
            "error": str(error_msg),
            "processing_timestamp": datetime.now().isoformat(),
            "agent_type": "enhanced_code_parser",
            "gpu_used": self.gpu_id
        }

    def _create_success_result(self, file_name: str, status: str, chunks_count: int, 
                            **kwargs) -> Dict[str, Any]:
        """Create standardized success result"""
        result = {
            "status": str(status),
            "file_name": str(file_name),
            "chunks_created": int(chunks_count),
            "processing_timestamp": datetime.now().isoformat(),
            "agent_type": "enhanced_code_parser",
            "gpu_used": self.gpu_id
        }
        result.update(kwargs)
        return result

    def _create_fallback_chunk(self, content: str, file_name: str, file_type: str) -> CodeChunk:
        """Create fallback chunk when parsing fails"""
        program_name = self._extract_program_name_safe(content, file_name)
        
        return CodeChunk(
            program_name=str(program_name),
            chunk_id=f"{program_name}_FULL_FILE",
            chunk_type="full_file_fallback",
            content=str(content)[:2000],  # Limit content size
            metadata={
                "fallback": True,
                "file_type": str(file_type),
                "reason": "Parsing failed, created fallback chunk",
                "content_length": len(content)
            },
            line_start=0,
            line_end=len(content.split('\n'))
        )

    def _extract_program_name_safe(self, content: str, file_name: str) -> str:
        """Safely extract program name with fallbacks"""
        try:
            # Try PROGRAM-ID first
            if hasattr(self, 'cobol_patterns') and 'program_id' in self.cobol_patterns:
                program_match = self.cobol_patterns['program_id'].search(content)
                if program_match:
                    return str(program_match.group(1)).strip()
            
            # Try JOB name
            if hasattr(self, 'jcl_patterns') and 'job_card' in self.jcl_patterns:
                job_match = self.jcl_patterns['job_card'].search(content)
                if job_match:
                    return str(job_match.group(1)).strip()
            
            # Fallback to filename
            if isinstance(file_name, (str, Path)):
                name = str(Path(file_name).stem)
                # Remove common extensions
                for ext in ['.cbl', '.cob', '.jcl', '.copy', '.cpy']:
                    if name.lower().endswith(ext.lower()):
                        name = name[:-len(ext)]
                return name or "UNKNOWN_PROGRAM"
            
            return "UNKNOWN_PROGRAM"
            
        except Exception as e:
            self.logger.warning(f"Program name extraction failed: {str(e)}")
            return "UNKNOWN_PROGRAM"

    async def _parse_cobol_complete(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete COBOL parsing with all structures"""
        await self._ensure_llm_engine()
        
        chunks = []
        lines = content.split('\n')
        
        # Extract program name
        program_name = self._extract_program_name(content, Path(filename))
        self.logger.info(f"Extracted program name: '{program_name}' from file: '{filename}'")
        
        # Parse divisions
        division_chunks = await self._parse_cobol_divisions(content, program_name)
        chunks.extend(division_chunks)
        
        # Parse sections within divisions
        section_chunks = await self._parse_cobol_sections(content, program_name)
        chunks.extend(section_chunks)
        
        # Parse procedure division paragraphs
        paragraph_chunks = await self._parse_cobol_paragraphs(content, program_name)
        chunks.extend(paragraph_chunks)
        
        # Parse control structures
        control_chunks = await self._parse_cobol_control_structures(content, program_name)
        chunks.extend(control_chunks)
        
        # Parse SQL blocks
        sql_chunks = await self._parse_sql_blocks_complete(content, program_name)
        chunks.extend(sql_chunks)
        
        # Parse COPY statements
        copy_chunks = await self._parse_copy_statements_complete(content, program_name)
        chunks.extend(copy_chunks)
        
        # Parse CICS commands if present
        cics_chunks = await self._parse_cics_in_cobol(content, program_name)
        chunks.extend(cics_chunks)
        
        # Parse file operations
        file_chunks = await self._parse_file_operations(content, program_name)
        chunks.extend(file_chunks)
        
        # Parse data structures
        data_chunks = await self._parse_data_structures(content, program_name)
        chunks.extend(data_chunks)
        
        return chunks

    async def _parse_cobol_divisions(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL divisions"""
        chunks = []
        
        division_patterns = {
            'identification_division': self.cobol_patterns['identification_division'],
            'environment_division': self.cobol_patterns['environment_division'],
            'data_division': self.cobol_patterns['data_division'],
            'procedure_division': self.cobol_patterns['procedure_division']
        }
        
        for div_name, pattern in division_patterns.items():
            match = pattern.search(content)
            if match:
                # Find division content
                start_pos = match.start()
                next_div_pos = len(content)
                
                # Find next division
                for other_name, other_pattern in division_patterns.items():
                    if other_name != div_name:
                        other_match = other_pattern.search(content, start_pos + 1)
                        if other_match and other_match.start() < next_div_pos:
                            next_div_pos = other_match.start()
                
                div_content = content[start_pos:next_div_pos].strip()
                
                metadata = await self._analyze_division_with_llm(div_content, div_name)
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_{div_name.upper()}",
                    chunk_type="division",
                    content=div_content,
                    metadata=metadata,
                    line_start=content[:start_pos].count('\n'),
                    line_end=content[:next_div_pos].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_cobol_sections(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL sections"""
        chunks = []
        
        section_patterns = {
            'working_storage': self.cobol_patterns['working_storage'],
            'file_section': self.cobol_patterns['file_section'],
            'linkage_section': self.cobol_patterns['linkage_section'],
            'local_storage': self.cobol_patterns['local_storage']
        }
        
        for sect_name, pattern in section_patterns.items():
            match = pattern.search(content)
            if match:
                # Find section content
                start_pos = match.start()
                next_sect_pos = len(content)
                
                # Find next section or division
                all_patterns = {**section_patterns, **{
                    'procedure_division': self.cobol_patterns['procedure_division']
                }}
                
                for other_name, other_pattern in all_patterns.items():
                    if other_name != sect_name:
                        other_match = other_pattern.search(content, start_pos + 1)
                        if other_match and other_match.start() < next_sect_pos:
                            next_sect_pos = other_match.start()
                
                sect_content = content[start_pos:next_sect_pos].strip()
                
                if sect_name in ['working_storage', 'file_section', 'linkage_section', 'local_storage']:
                    metadata = await self._analyze_data_section_with_llm(sect_content, sect_name)
                else:
                    metadata = await self._analyze_section_with_llm(sect_content, sect_name)
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_{sect_name.upper()}_SECTION",
                    chunk_type="section",
                    content=sect_content,
                    metadata=metadata,
                    line_start=content[:start_pos].count('\n'),
                    line_end=content[:next_sect_pos].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_cobol_paragraphs(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL procedure division paragraphs"""
        chunks = []
        
        # Find procedure division
        proc_match = self.cobol_patterns['procedure_division'].search(content)
        if not proc_match:
            return chunks
        
        proc_start = proc_match.end()
        proc_content = content[proc_start:]
        
        # Find paragraphs
        paragraph_matches = list(self.cobol_patterns['paragraph'].finditer(proc_content))
        
        for i, match in enumerate(paragraph_matches):
            para_name = match.group(1)
            para_start = match.start()
            
            # Find paragraph end (next paragraph or end of content)
            if i + 1 < len(paragraph_matches):
                para_end = paragraph_matches[i + 1].start()
            else:
                para_end = len(proc_content)
            
            para_content = proc_content[para_start:para_end].strip()
            
            # Skip sections (they end with SECTION)
            if para_name.endswith('SECTION'):
                continue
            
            metadata = await self._analyze_paragraph_with_llm(para_content)
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{para_name}",
                chunk_type="paragraph",
                content=para_content,
                metadata=metadata,
                line_start=content[:proc_start + para_start].count('\n'),
                line_end=content[:proc_start + para_end].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_cobol_control_structures(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL control structures"""
        chunks = []
        
        # Parse PERFORM statements
        perform_chunks = await self._parse_perform_statements_complete(content, program_name)
        chunks.extend(perform_chunks)
        
        # Parse IF statements
        if_chunks = await self._parse_if_statements_complete(content, program_name)
        chunks.extend(if_chunks)
        
        # Parse EVALUATE statements
        eval_chunks = await self._parse_evaluate_statements_complete(content, program_name)
        chunks.extend(eval_chunks)
        
        return chunks

    async def _parse_perform_statements_complete(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse all types of PERFORM statements"""
        chunks = []
        
        # PERFORM UNTIL
        until_matches = self.cobol_patterns['perform_until'].finditer(content)
        for i, match in enumerate(until_matches):
            metadata = {
                "perform_type": "until",
                "condition": match.group(1).strip(),
                "control_structure": "loop"
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_PERFORM_UNTIL_{i+1}",
                chunk_type="perform_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # PERFORM VARYING
        varying_matches = self.cobol_patterns['perform_varying'].finditer(content)
        for i, match in enumerate(varying_matches):
            metadata = {
                "perform_type": "varying",
                "varying_clause": match.group(1).strip(),
                "control_structure": "iterative_loop"
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_PERFORM_VARYING_{i+1}",
                chunk_type="perform_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # PERFORM THRU
        thru_matches = self.cobol_patterns['perform_thru'].finditer(content)
        for i, match in enumerate(thru_matches):
            metadata = {
                "perform_type": "thru",
                "start_paragraph": match.group(1),
                "end_paragraph": match.group(2),
                "control_structure": "paragraph_range"
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_PERFORM_THRU_{i+1}",
                chunk_type="perform_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # Simple PERFORM
        simple_matches = self.cobol_patterns['perform'].finditer(content)
        for i, match in enumerate(simple_matches):
            # Skip if already captured by other PERFORM types
            if not any(other_match.start() <= match.start() <= other_match.end() 
                      for other_pattern in [self.cobol_patterns['perform_until'], 
                                          self.cobol_patterns['perform_varying'],
                                          self.cobol_patterns['perform_thru']]
                      for other_match in other_pattern.finditer(content)):
                
                metadata = {
                    "perform_type": "simple",
                    "target_paragraph": match.group(1),
                    "control_structure": "procedure_call"
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_PERFORM_SIMPLE_{i+1}",
                    chunk_type="perform_statement",
                    content=match.group(0),
                    metadata=metadata,
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_if_statements_complete(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse complete IF statements with ELSE and END-IF"""
        chunks = []
        
        if_matches = self.cobol_patterns['if_statement'].finditer(content)
        for i, match in enumerate(if_matches):
            # Extract complete IF statement
            if_start = match.start()
            if_content = self._extract_complete_if_statement(content, if_start)
            
            metadata = {
                "condition": match.group(1).strip(),
                "has_else": "ELSE" in if_content.upper(),
                "nested_ifs": if_content.upper().count("IF") - 1,
                "has_end_if": "END-IF" in if_content.upper()
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_IF_STMT_{i+1}",
                chunk_type="if_statement",
                content=if_content,
                metadata=metadata,
                line_start=content[:if_start].count('\n'),
                line_end=content[:if_start + len(if_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_evaluate_statements_complete(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse complete EVALUATE statements"""
        chunks = []
        
        eval_matches = self.cobol_patterns['evaluate'].finditer(content)
        for i, match in enumerate(eval_matches):
            # Extract complete EVALUATE statement
            eval_start = match.start()
            eval_content = self._extract_complete_evaluate_statement(content, eval_start)
            
            # Analyze WHEN clauses
            when_matches = self.cobol_patterns['when_clause'].findall(eval_content)
            has_other = self.cobol_patterns['when_other'].search(eval_content) is not None
            
            metadata = {
                "expression": match.group(1).strip(),
                "when_clauses": when_matches,
                "when_count": len(when_matches),
                "has_when_other": has_other
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_EVALUATE_{i+1}",
                chunk_type="evaluate_statement",
                content=eval_content,
                metadata=metadata,
                line_start=content[:eval_start].count('\n'),
                line_end=content[:eval_start + len(eval_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _extract_complete_if_statement(self, content: str, start_pos: int) -> str:
        """Extract complete IF statement including all nested structures"""
        lines = content[start_pos:].split('\n')
        if_content = []
        if_level = 0
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if_content.append(line)
            
            # Count IF levels
            if 'IF ' in line_upper and not line_upper.startswith('*'):
                if_level += 1
            elif 'END-IF' in line_upper:
                if_level -= 1
                if if_level == 0:
                    break
            # Handle implicit END-IF (next paragraph or period)
            elif if_level == 1 and (line.strip().endswith('.') and 
                                   not any(keyword in line_upper for keyword in ['ELSE', 'END-IF'])):
                break
        
        return '\n'.join(if_content)

    def _extract_complete_evaluate_statement(self, content: str, start_pos: int) -> str:
        """Extract complete EVALUATE statement"""
        lines = content[start_pos:].split('\n')
        eval_content = []
        eval_level = 0
        
        for line in lines:
            line_upper = line.upper().strip()
            
            eval_content.append(line)
            
            if 'EVALUATE ' in line_upper and not line_upper.startswith('*'):
                eval_level += 1
            elif 'END-EVALUATE' in line_upper:
                eval_level -= 1
                if eval_level == 0:
                    break
        
        return '\n'.join(eval_content)

    async def _parse_sql_blocks_complete(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse all SQL-related structures"""
        chunks = []
        
        # Regular SQL blocks
        sql_matches = self.cobol_patterns['sql_block'].finditer(content)
        for i, match in enumerate(sql_matches):
            sql_content = match.group(0)
            sql_inner = match.group(1).strip()
            
            metadata = await self._analyze_sql_comprehensive(sql_inner)
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_SQL_BLOCK_{i+1}",
                chunk_type="sql_block",
                content=sql_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # SQL INCLUDE statements
        include_matches = self.cobol_patterns['sql_include'].finditer(content)
        for i, match in enumerate(include_matches):
            metadata = {
                "sql_type": "include",
                "included_member": match.group(1),
                "purpose": "SQL copybook inclusion"
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_SQL_INCLUDE_{i+1}",
                chunk_type="sql_include",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # SQL DECLARE statements
        declare_matches = self.cobol_patterns['sql_declare'].finditer(content)
        for i, match in enumerate(declare_matches):
            metadata = {
                "sql_type": "declare",
                "declaration_content": match.group(1).strip(),
                "purpose": "SQL variable/cursor declaration"
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_SQL_DECLARE_{i+1}",
                chunk_type="sql_declare",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_copy_statements_complete(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse all COPY statement variations"""
        chunks = []
        
        # Simple COPY statements
        copy_matches = self.cobol_patterns['copy_statement'].finditer(content)
        for i, match in enumerate(copy_matches):
            copybook_name = match.group(1)
            
            metadata = {
                "copy_type": "simple",
                "copybook_name": copybook_name,
                "replacement": None,
                "library": None
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_COPY_{copybook_name}",
                chunk_type="copy_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # COPY REPLACING statements
        replacing_matches = self.cobol_patterns['copy_replacing'].finditer(content)
        for i, match in enumerate(replacing_matches):
            copybook_name = match.group(1)
            replacing_clause = match.group(2)
            
            metadata = {
                "copy_type": "replacing",
                "copybook_name": copybook_name,
                "replacement": replacing_clause,
                "library": None
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_COPY_REPLACING_{copybook_name}",
                chunk_type="copy_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # COPY IN statements
        copy_in_matches = self.cobol_patterns['copy_in'].finditer(content)
        for i, match in enumerate(copy_in_matches):
            copybook_name = match.group(1)
            library_name = match.group(2)
            
            metadata = {
                "copy_type": "in_library",
                "copybook_name": copybook_name,
                "replacement": None,
                "library": library_name
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_COPY_IN_{copybook_name}_{library_name}",
                chunk_type="copy_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_cics_in_cobol(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse CICS commands embedded in COBOL"""
        chunks = []
        
        # Parse all CICS command types
        for command_type, pattern in self.cics_patterns.items():
            matches = pattern.finditer(content)
            
            for i, match in enumerate(matches):
                cics_content = match.group(0)
                params = match.group(1) if match.groups() else ""
                
                metadata = await self._analyze_cics_command(command_type, params, cics_content)
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_{command_type.upper()}_{i+1}",
                    chunk_type="cics_command",
                    content=cics_content,
                    metadata=metadata,
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_file_operations(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse file operations"""
        chunks = []
        
        file_patterns = {
            'select_statement': self.cobol_patterns['select_statement'],
            'fd_statement': self.cobol_patterns['fd_statement'],
            'open_statement': self.cobol_patterns['open_statement'],
            'close_statement': self.cobol_patterns['close_statement'],
            'read_statement': self.cobol_patterns['read_statement'],
            'write_statement': self.cobol_patterns['write_statement'],
            'rewrite_statement': self.cobol_patterns['rewrite_statement'],
            'delete_statement': self.cobol_patterns['delete_statement']
        }
        
        for op_type, pattern in file_patterns.items():
            matches = pattern.finditer(content)
            
            for i, match in enumerate(matches):
                file_name = match.group(1) if match.groups() else "UNKNOWN"
                operation_mode = match.group(1) if op_type == 'open_statement' and len(match.groups()) > 1 else None
                
                metadata = {
                    "operation_type": op_type,
                    "file_name": file_name,
                    "operation_mode": operation_mode,
                    "is_file_control": op_type in ['select_statement', 'fd_statement'],
                    "is_file_operation": op_type in ['open_statement', 'close_statement', 'read_statement', 
                                                   'write_statement', 'rewrite_statement', 'delete_statement']
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_{op_type.upper()}_{file_name}_{i+1}",
                    chunk_type="file_operation",
                    content=match.group(0),
                    metadata=metadata,
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _parse_cics_transaction_flow(self, content: str, program_name: str) -> Optional[CodeChunk]:
        """Analyze CICS transaction flow"""
        # Extract CICS command sequence
        cics_commands = []
        
        for pattern_name, pattern in self.cics_patterns.items():
            if pattern_name.startswith('cics_'):
                matches = pattern.finditer(content)
                for match in matches:
                    cics_commands.append({
                        "command": pattern_name,
                        "content": match.group(0),
                        "line": content[:match.start()].count('\n') + 1
                    })
        
        if not cics_commands:
            return None
        
        # Sort by line number to get execution sequence
        cics_commands.sort(key=lambda x: x["line"])
        
        metadata = {
            "transaction_type": "standard",
            "command_sequence": [cmd["command"] for cmd in cics_commands],
            "total_commands": len(cics_commands)
        }
        
        return CodeChunk(
            program_name=program_name,
            chunk_id=f"{program_name}_TRANSACTION_FLOW",
            chunk_type="cics_transaction_flow",
            content='\n'.join([cmd["content"] for cmd in cics_commands]),
            metadata=metadata,
            line_start=min([cmd["line"] for cmd in cics_commands]),
            line_end=max([cmd["line"] for cmd in cics_commands])
        )

    async def _parse_cics_error_handling(self, content: str, program_name: str) -> Optional[CodeChunk]:
        """Analyze CICS error handling patterns"""
        error_patterns = []
        
        # Find HANDLE CONDITION statements
        if 'cics_handle_condition' in self.cics_patterns:
            handle_matches = self.cics_patterns['cics_handle_condition'].finditer(content)
            for match in handle_matches:
                error_patterns.append({
                    "type": "handle_condition",
                    "content": match.group(0)
                })
        
        if not error_patterns:
            return None
        
        metadata = {
            "error_handling_type": "cics_error_handling",
            "patterns": error_patterns,
            "handles_conditions": len(error_patterns)
        }
        
        return CodeChunk(
            program_name=program_name,
            chunk_id=f"{program_name}_ERROR_HANDLING",
            chunk_type="cics_error_handling",
            content='\n'.join([p["content"] for p in error_patterns]),
            metadata=metadata,
            line_start=0,
            line_end=len(content.split('\n'))
        )


    async def _parse_data_structures(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse data structure definitions"""
        chunks = []
        
        # Find working storage and other data sections
        data_sections = ['WORKING-STORAGE SECTION', 'FILE SECTION', 'LINKAGE SECTION', 'LOCAL-STORAGE SECTION']
        
        for section_name in data_sections:
            section_pattern = re.compile(section_name, re.IGNORECASE)
            section_match = section_pattern.search(content)
            
            if section_match:
                # Extract section content
                start_pos = section_match.end()
                
                # Find end of section (next section or procedure division)
                end_patterns = data_sections + ['PROCEDURE DIVISION']
                end_pos = len(content)
                
                for end_pattern_name in end_patterns:
                    if end_pattern_name != section_name:
                        end_pattern = re.compile(end_pattern_name, re.IGNORECASE)
                        end_match = end_pattern.search(content, start_pos)
                        if end_match and end_match.start() < end_pos:
                            end_pos = end_match.start()
                
                section_content = content[start_pos:end_pos]
                
                # Parse individual data items
                data_items = await self._parse_data_items(section_content, program_name, section_name)
                chunks.extend(data_items)
        
        return chunks

    async def _parse_data_items(self, section_content: str, program_name: str, section_name: str) -> List[CodeChunk]:
        """Parse individual data items in a section"""
        chunks = []
        
        data_matches = self.cobol_patterns['data_item'].finditer(section_content)
        
        for i, match in enumerate(data_matches):
            level = match.group(2)
            name = match.group(3)
            definition = match.group(4)
            
            # Skip if it's a comment line
            if match.group(1).strip().startswith('*'):
                continue
            
            # Parse field attributes
            pic_clause = self._extract_pic_clause(definition)
            usage_clause = self._extract_usage_clause(definition)
            value_clause = self._extract_value_clause(definition)
            occurs_info = self._extract_occurs_info(definition)
            redefines_info = self._extract_redefines_info(definition)
            
            metadata = {
                "level": int(level),
                "field_name": name,
                "pic_clause": pic_clause,
                "usage": usage_clause,
                "value": value_clause,
                "occurs": occurs_info,
                "redefines": redefines_info,
                "section": section_name,
                "data_type": self._determine_data_type(pic_clause, usage_clause),
                "length": self._calculate_field_length(pic_clause, usage_clause),
                "is_group_item": pic_clause is None and occurs_info is None,
                "is_elementary": pic_clause is not None
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_DATA_{name}_{level}",
                chunk_type="data_item",
                content=match.group(0),
                metadata=metadata,
                line_start=section_content[:match.start()].count('\n'),
                line_end=section_content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_complete(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete JCL parsing with all elements"""
        chunks = []
        
        job_name = self._extract_program_name(content, Path(filename))
        lines = content.split('\n')
        
        # Parse job card
        job_chunk = await self._parse_jcl_job_card(content, job_name)
        if job_chunk:
            chunks.append(job_chunk)
        
        # Parse job steps
        step_chunks = await self._parse_jcl_steps(content, job_name)
        chunks.extend(step_chunks)
        
        # Parse DD statements
        dd_chunks = await self._parse_jcl_dd_statements(content, job_name)
        chunks.extend(dd_chunks)
        
        # Parse procedures
        proc_chunks = await self._parse_jcl_procedures(content, job_name)
        chunks.extend(proc_chunks)
        
        # Parse conditional logic
        cond_chunks = await self._parse_jcl_conditional_logic(content, job_name)
        chunks.extend(cond_chunks)
        
        return chunks

    async def _parse_jcl_job_card(self, content: str, job_name: str) -> Optional[CodeChunk]:
        """Parse JCL job card"""
        job_match = self.jcl_patterns['job_card'].search(content)
        if not job_match:
            return None
        
        # Extract complete job card (may span multiple lines)
        job_card_content = self._extract_complete_jcl_statement(content, job_match.start(), '//')
        
        metadata = await self._analyze_jcl_job_card(job_card_content)
        
        return CodeChunk(
            program_name=job_name,
            chunk_id=f"{job_name}_JOB_CARD",
            chunk_type="jcl_job_card",
            content=job_card_content,
            metadata=metadata,
            line_start=content[:job_match.start()].count('\n'),
            line_end=content[:job_match.start() + len(job_card_content)].count('\n')
        )

    async def _parse_jcl_steps(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL job steps"""
        chunks = []
        
        step_matches = self.jcl_patterns['job_step'].finditer(content)
        
        for match in step_matches:
            step_name = match.group(1)
            
            # Extract complete step (until next step or end)
            step_content = self._extract_jcl_step_content(content, match.start())
            
            metadata = await self._analyze_jcl_step_comprehensive(step_content, step_name)
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_STEP_{step_name}",
                chunk_type="jcl_step",
                content=step_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(step_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_dd_statements(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL DD statements"""
        chunks = []
        
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for i, match in enumerate(dd_matches):
            dd_name = match.group(1)
            
            # Extract complete DD statement
            dd_content = self._extract_complete_jcl_statement(content, match.start(), '//')
            
            metadata = await self._analyze_jcl_dd_statement(dd_content, dd_name)
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_DD_{dd_name}",
                chunk_type="jcl_dd_statement",
                content=dd_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(dd_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_procedures(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL procedures"""
        chunks = []
        
        proc_matches = self.jcl_patterns['proc_definition'].finditer(content)
        
        for match in proc_matches:
            proc_name = match.group(1)
            
            # Find PEND statement
            pend_match = self.jcl_patterns['pend_statement'].search(content, match.end())
            if pend_match:
                proc_content = content[match.start():pend_match.end()]
            else:
                # Procedure without PEND (rest of file)
                proc_content = content[match.start():]
            
            metadata = await self._analyze_jcl_procedure(proc_content, proc_name)
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_PROC_{proc_name}",
                chunk_type="jcl_procedure",
                content=proc_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(proc_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_conditional_logic(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL conditional logic"""
        chunks = []
        
        # Parse IF statements
        if_matches = self.jcl_patterns['if_statement'].finditer(content)
        
        for i, match in enumerate(if_matches):
            condition = match.group(1)
            
            # Find corresponding ENDIF
            endif_match = self.jcl_patterns['endif_statement'].search(content, match.end())
            if endif_match:
                if_content = content[match.start():endif_match.end()]
            else:
                if_content = match.group(0)
            
            metadata = {
                "condition": condition,
                "has_endif": endif_match is not None,
                "conditional_type": "if_then"
            }
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_IF_COND_{i+1}",
                chunk_type="jcl_conditional",
                content=if_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(if_content)].count('\n')
            )
            chunks.append(chunk)
        
        # Parse SET statements
        set_matches = self.jcl_patterns['set_statement'].finditer(content)
        
        for i, match in enumerate(set_matches):
            variable = match.group(1)
            value = match.group(2)
            
            metadata = {
                "variable": variable,
                "value": value,
                "statement_type": "set"
            }
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_SET_{variable}",
                chunk_type="jcl_set_statement",
                content=match.group(0),
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_copybook_complete(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete copybook parsing"""
        copybook_name = self._extract_program_name(content, Path(filename))
        
        # Parse overall structure
        structure_chunk = await self._parse_copybook_structure(content, copybook_name)
        chunks = [structure_chunk] if structure_chunk else []
        
        # Parse individual fields
        field_chunks = await self._parse_copybook_fields(content, copybook_name)
        chunks.extend(field_chunks)
        
        # Parse record layouts
        record_chunks = await self._parse_copybook_records(content, copybook_name)
        chunks.extend(record_chunks)
        
        return chunks

    async def _parse_copybook_structure(self, content: str, copybook_name: str) -> Optional[CodeChunk]:
        """Parse overall copybook structure"""
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
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )

    async def _parse_copybook_fields(self, content: str, copybook_name: str) -> List[CodeChunk]:
        """Parse individual copybook fields"""
        chunks = []
        
        data_matches = self.cobol_patterns['data_item'].finditer(content)
        
        for match in data_matches:
            level = match.group(2)
            name = match.group(3)
            definition = match.group(4)
            
            # Skip comment lines
            if match.group(1).strip().startswith('*'):
                continue
            
            # Only process significant levels (01, 05, 10, etc.)
            if int(level) <= 49:
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
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_copybook_records(self, content: str, copybook_name: str) -> List[CodeChunk]:
        """Parse record-level structures in copybook"""
        chunks = []
        
        # Find 01-level items (records)
        record_pattern = re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*)(.*?)(?=^\s*01\s|\Z)', 
                                  re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(record_pattern.finditer(content)):
            record_name = match.group(1)
            record_content = match.group(0)
            
            metadata = await self._analyze_copybook_record(record_content, record_name)
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_RECORD_{record_name}",
                chunk_type="copybook_record",
                content=record_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_bms_complete(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete BMS parsing"""
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
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_cics_complete(self, content: str, filename: str) -> List[CodeChunk]:
        """Complete CICS program parsing"""
        chunks = []
        
        program_name = self._extract_program_name(content, Path(filename))
        
        # Parse CICS commands
        command_chunks = await self._parse_cics_commands_complete(content, program_name)
        chunks.extend(command_chunks)
        
        # Parse transaction flow
        flow_chunk = await self._parse_cics_transaction_flow(content, program_name)
        if flow_chunk:
            chunks.append(flow_chunk)
        
        # Parse error handling
        error_chunk = await self._parse_cics_error_handling(content, program_name)
        if error_chunk:
            chunks.append(error_chunk)
        
        return chunks

    async def _parse_cics_commands_complete(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse all CICS commands comprehensively"""
        chunks = []
        
        for command_type, pattern in self.cics_patterns.items():
            matches = pattern.finditer(content)
            
            for i, match in enumerate(matches):
                cics_content = match.group(0)
                params = match.group(1) if match.groups() else ""
                
                metadata = await self._analyze_cics_command_comprehensive(command_type, params, cics_content)
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_{command_type.upper()}_{i+1}",
                    chunk_type="cics_command",
                    content=cics_content,
                    metadata=metadata,
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_generic(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse unknown file type generically"""
        program_name = self._extract_program_name(content, Path(filename))
        
        # Try to identify patterns even in unknown files
        chunks = []
        
        # Look for common mainframe patterns
        if 'EXEC' in content.upper():
            # Might contain embedded commands
            exec_chunks = await self._parse_generic_exec_statements(content, program_name)
            chunks.extend(exec_chunks)
        
        # Default generic chunk
        if not chunks:
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_GENERIC",
                chunk_type="generic",
                content=content,
                metadata={"file_type": "unknown", "analysis": "Generic file processing"},
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
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    # Helper method implementations
    def _extract_complete_jcl_statement(self, content: str, start_pos: int, delimiter: str) -> str:
        """Extract complete JCL statement handling continuations"""
        lines = content[start_pos:].split('\n')
        statement_lines = []
        
        for line in lines:
            statement_lines.append(line)
            
            # Check for continuation
            if len(line) >= 72 and line[71:72] not in [' ', '']:
                continue  # Continued line
            else:
                # Check if next line is continuation
                next_line_idx = len(statement_lines)
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx]
                    if len(next_line) >= 16 and next_line[0:3] == '// ':
                        continue  # Next line is continuation
                break
        
        return '\n'.join(statement_lines)

    def _extract_jcl_step_content(self, content: str, start_pos: int) -> str:
        """Extract complete JCL step content"""
        lines = content[start_pos:].split('\n')
        step_lines = []
        
        in_step = True
        for line in lines:
            if in_step:
                step_lines.append(line)
                
                # Check if this starts a new step
                if line.startswith('//') and ' EXEC ' in line.upper() and len(step_lines) > 1:
                    # This is the start of next step, remove it
                    step_lines.pop()
                    break
                elif line.startswith('//') and line.endswith(' JOB '):
                    # This is a new job, end step
                    step_lines.pop()
                    break
        
        return '\n'.join(step_lines)

    def _extract_pic_clause(self, definition: str) -> Optional[str]:
        """Extract PIC clause from field definition"""
        match = self.cobol_patterns['pic_clause'].search(definition)
        return match.group(1) if match else None

    def _extract_usage_clause(self, definition: str) -> str:
        """Extract USAGE clause from field definition"""
        match = self.cobol_patterns['usage_clause'].search(definition)
        return match.group(1) if match else "DISPLAY"

    def _extract_value_clause(self, definition: str) -> Optional[str]:
        """Extract VALUE clause from field definition"""
        match = self.cobol_patterns['value_clause'].search(definition)
        return match.group(1) if match else None

    def _extract_occurs_info(self, definition: str) -> Optional[Dict[str, Any]]:
        """Extract OCCURS clause information"""
        match = self.cobol_patterns['occurs_clause'].search(definition)
        if match:
            min_occurs = int(match.group(1))
            max_occurs = int(match.group(2)) if match.group(2) else min_occurs
            
            # Check for DEPENDING ON
            depending_match = self.cobol_patterns['depending_on'].search(definition)
            depending_field = depending_match.group(1) if depending_match else None
            
            # Check for INDEXED BY
            indexed_match = self.cobol_patterns['indexed_by'].search(definition)
            indexed_field = indexed_match.group(1) if indexed_match else None
            
            return {
                "min_occurs": min_occurs,
                "max_occurs": max_occurs,
                "is_variable": max_occurs != min_occurs or depending_field is not None,
                "depending_on": depending_field,
                "indexed_by": indexed_field
            }
        return None

    def _extract_redefines_info(self, definition: str) -> Optional[str]:
        """Extract REDEFINES information"""
        match = self.cobol_patterns['redefines'].search(definition)
        return match.group(1) if match else None

    def _determine_data_type(self, pic_clause: str, usage: str) -> str:
        """Determine data type from PIC clause and usage"""
        if not pic_clause:
            return "group"
        
        pic_upper = pic_clause.upper()
        
        if '9' in pic_upper:
            if 'V' in pic_upper or '.' in pic_upper:
                return "decimal"
            else:
                return "integer"
        elif 'X' in pic_upper:
            return "alphanumeric"
        elif 'A' in pic_upper:
            return "alphabetic"
        elif 'N' in pic_upper:
            return "national"
        elif 'S' in pic_upper:
            return "signed_numeric"
        else:
            return "special"

    def _calculate_field_length(self, pic_clause: str, usage: str) -> int:
        """Calculate field length based on PIC clause and usage"""
        if not pic_clause:
            return 0
        
        import re
        
        # Handle explicit lengths like PIC X(10) or PIC 9(5)
        explicit_match = re.search(r'[X9AN]\((\d+)\)', pic_clause.upper())
        if explicit_match:
            base_length = int(explicit_match.group(1))
        else:
            # Count character repetitions like PIC XXX or PIC 999
            base_length = len(re.findall(r'[X9AN]', pic_clause.upper()))
        
        # Adjust for usage
        if usage == 'COMP-3' or usage == 'PACKED-DECIMAL':
            return (base_length + 1) // 2  # Packed decimal
        elif usage in ['COMP', 'COMP-4', 'BINARY']:
            if base_length <= 4:
                return 2  # Half word
            elif base_length <= 9:
                return 4  # Full word
            else:
                return 8  # Double word
        elif usage == 'COMP-1':
            return 4  # Single precision floating point
        elif usage == 'COMP-2':
            return 8  # Double precision floating point
        elif usage == 'INDEX':
            return 4  # Index data item
        
        return base_length

    def _extract_record_layouts(self, content: str) -> List[Dict[str, Any]]:
        """Extract record layout information"""
        layouts = []
        
        # Find 01 level items (record definitions)
        record_pattern = re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*)(.*?)$', re.MULTILINE | re.IGNORECASE)

        
        for match in record_pattern.finditer(content):
            name = match.group(1)
            definition = match.group(2)
            
            layouts.append({
                "record_name": name,
                "definition": definition.strip(),
                "has_occurs": "OCCURS" in definition.upper(),
                "is_redefines": "REDEFINES" in definition.upper(),
                "has_value": "VALUE" in definition.upper()
            })
        
        return layouts

    def _extract_occurs_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract OCCURS table information"""
        tables = []
        
        occurs_matches = self.cobol_patterns['occurs_clause'].finditer(content)
        
        for match in occurs_matches:
            # Find the field name associated with this OCCURS
            before_occurs = content[:match.start()]
            lines = before_occurs.split('\n')
            last_line = lines[-1] if lines else ""
            
            # Extract field name from the line
            field_match = re.search(r'\d+\s+([A-Z][A-Z0-9-]*)', last_line)
            field_name = field_match.group(1) if field_match else "UNKNOWN"
            
            min_occurs = int(match.group(1))
            max_occurs = int(match.group(2)) if match.group(2) else min_occurs
            
            tables.append({
                "table_name": field_name,
                "min_occurs": min_occurs,
                "max_occurs": max_occurs,
                "is_variable": max_occurs != min_occurs,
                "occurs_clause": match.group(0)
            })
        
        return tables

    def _extract_redefines_structures(self, content: str) -> List[Dict[str, Any]]:
        """Extract REDEFINES structure information"""
        redefines = []
        
        redefines_matches = self.cobol_patterns['redefines'].finditer(content)
        
        for match in redefines_matches:
            # Find the field name that does the redefining
            before_redefines = content[:match.start()]
            lines = before_redefines.split('\n')
            last_line = lines[-1] if lines else ""
            
            field_match = re.search(r'\d+\s+([A-Z][A-Z0-9-]*)', last_line)
            redefining_field = field_match.group(1) if field_match else "UNKNOWN"
            
            redefines.append({
                "redefining_field": redefining_field,
                "redefined_field": match.group(1),
                "purpose": "Memory overlay structure"
            })
        
        return redefines

    def _extract_bms_attributes(self, bms_definition: str) -> Dict[str, str]:
        """Extract BMS attributes from definition"""
        attributes = {}
        
        # Common BMS attribute patterns
        attr_patterns = {
            'POS': re.compile(r'POS=\((\d+),(\d+)\)', re.IGNORECASE),
            'SIZE': re.compile(r'SIZE=\((\d+),(\d+)\)', re.IGNORECASE),
            'LENGTH': re.compile(r'LENGTH=(\d+)', re.IGNORECASE),
            'ATTRB': re.compile(r'ATTRB=\(([^)]+)\)', re.IGNORECASE),
            'INITIAL': re.compile(r'INITIAL=([\'"][^\']*[\'"])', re.IGNORECASE),
            'PICIN': re.compile(r'PICIN=([\'"][^\']*[\'"])', re.IGNORECASE),
            'PICOUT': re.compile(r'PICOUT=([\'"][^\']*[\'"])', re.IGNORECASE),
            'JUSTIFY': re.compile(r'JUSTIFY=(\w+)', re.IGNORECASE),
            'COLOR': re.compile(r'COLOR=(\w+)', re.IGNORECASE),
            'HILIGHT': re.compile(r'HILIGHT=(\w+)', re.IGNORECASE)
        }
        
        for attr_name, pattern in attr_patterns.items():
            match = pattern.search(bms_definition)
            if match:
                if attr_name in ['POS', 'SIZE']:
                    attributes[attr_name] = f"({match.group(1)},{match.group(2)})"
                else:
                    attributes[attr_name] = match.group(1)
        
        return attributes

    def _count_bms_fields_in_map(self, content: str, map_start: int) -> int:
        """Count BMS fields in a map"""
        remaining_content = content[map_start:]
        
        # Find next map or mapset end
        next_map = remaining_content.find('DFHMDI')
        next_mapset = remaining_content.find('DFHMSD')
        
        if next_map != -1 and (next_mapset == -1 or next_map < next_mapset):
            map_content = remaining_content[:next_map]
        elif next_mapset != -1:
            map_content = remaining_content[:next_mapset]
        else:
            map_content = remaining_content
        
        return len(self.bms_patterns['bms_field'].findall(map_content))

    def _is_input_field(self, field_definition: str) -> bool:
        """Check if BMS field is input capable"""
        input_indicators = ['PICIN', 'FSET', 'CURSOR', 'IC']
        return any(indicator in field_definition.upper() for indicator in input_indicators)

    def _is_output_field(self, field_definition: str) -> bool:
        """Check if BMS field is output capable"""
        output_indicators = ['PICOUT', 'INITIAL']
        return any(indicator in field_definition.upper() for indicator in output_indicators)

    # LLM Analysis Methods
    async def _analyze_division_with_llm(self, content: str, division_name: str) -> Dict[str, Any]:
        """Analyze COBOL division with LLM"""
        await self._ensure_llm_engine()
        
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

    async def _analyze_data_section_with_llm(self, content: str, section_name: str) -> Dict[str, Any]:
        """Analyze data section with comprehensive field analysis"""
        await self._ensure_llm_engine()
        
        # Extract field information
        field_analysis = await self._analyze_fields_comprehensive(content)
        
        prompt = f"""
        Analyze this COBOL data section: {section_name}
        
        {content[:800]}...
        
        Provide analysis of:
        1. Record structures and layouts
        2. Key data elements and their purposes
        3. Relationships between fields
        4. Data validation and constraints
        5. Business domain represented
        
        Return as JSON:
        {{
            "record_structures": ["structure1", "structure2"],
            "key_elements": ["element1", "element2"],
            "relationships": ["rel1", "rel2"],
            "validations": ["val1", "val2"],
            "business_domain": "domain description"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=500)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                llm_analysis = json.loads(response_text[json_start:json_end])
                
                # Combine with field analysis
                llm_analysis['field_analysis'] = field_analysis
                return llm_analysis
        except Exception as e:
            self.logger.warning(f"Data section analysis failed: {str(e)}")
        
        return {
            "record_structures": [],
            "key_elements": [],
            "relationships": [],
            "validations": [],
            "business_domain": f"{section_name} data",
            "field_analysis": field_analysis
        }

    async def _analyze_section_with_llm(self, content: str, section_name: str) -> Dict[str, Any]:
        """Analyze general section with LLM"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this COBOL section: {section_name}
        
        {content[:500]}...
        
        Extract:
        1. Primary purpose of this section
        2. Main operations performed
        3. Error handling patterns
        4. Business logic implemented
        
        Return as JSON:
        {{
            "purpose": "description",
            "operations": ["op1", "op2"],
            "error_handling": ["pattern1", "pattern2"],
            "business_logic": "description"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=400)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Section analysis failed: {str(e)}")
        
        return {
            "purpose": f"Section {section_name} processing",
            "operations": [],
            "error_handling": [],
            "business_logic": "Business processing section"
        }

    async def _analyze_paragraph_with_llm(self, content: str) -> Dict[str, Any]:
        """Analyze paragraph with LLM"""
        await self._ensure_llm_engine()
        
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
        
        # Fallback to regex extraction
        return {
            "field_names": self._extract_field_names(content),
            "file_operations": self._extract_file_operations(content),
            "sql_operations": self._extract_sql_operations(content),
            "called_paragraphs": self._extract_perform_statements(content),
            "main_purpose": "Code processing",
            "error_handling": self._extract_error_handling_patterns(content)
        }

    async def _analyze_sql_comprehensive(self, sql_content: str) -> Dict[str, Any]:
        """Comprehensive SQL analysis"""
        await self._ensure_llm_engine()
        
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
        
        # Fallback analysis
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

    async def _analyze_cics_command(self, command_type: str, params: str, content: str) -> Dict[str, Any]:
        """Analyze CICS command"""
        return await self._analyze_cics_command_comprehensive(command_type, params, content)

    async def _analyze_cics_command_comprehensive(self, command_type: str, params: str, content: str) -> Dict[str, Any]:
        """Comprehensive CICS command analysis"""
        await self._ensure_llm_engine()
        
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
        
        # Fallback analysis
        return {
            "category": self._categorize_cics_command(command_type),
            "purpose": f"{command_type} operation",
            "key_parameters": self._extract_cics_parameters(params),
            "resource_accessed": self._extract_cics_resource(params),
            "flow_impact": "CICS transaction processing",
            "error_conditions": [],
            "performance_impact": "Standard CICS overhead"
        }

    def _categorize_cics_command(self, command_type: str) -> str:
        """Categorize CICS command"""
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
        """Extract CICS command parameters"""
        param_dict = {}
        
        # Common CICS parameter patterns
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
        """Extract main resource from CICS parameters"""
        resource_patterns = ['FILE', 'MAP', 'PROGRAM', 'QUEUE']
        
        for resource_type in resource_patterns:
            pattern = re.compile(f'{resource_type}\\(([^)]+)\\)', re.IGNORECASE)
            match = pattern.search(params)
            if match:
                return match.group(1)
        
        return "UNKNOWN"

    async def _analyze_jcl_job_card(self, content: str) -> Dict[str, Any]:
        """Analyze JCL job card"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this JCL job card:
        
        {content}
        
        Extract:
        1. Job name and class
        2. User and accounting information
        3. Priority and time limits
        4. Output specifications
        5. Special parameters
        
        Return as JSON:
        {{
            "job_name": "name",
            "job_class": "class",
            "user": "userid",
            "accounting": "account info",
            "priority": "priority",
            "time_limit": "time",
            "output_class": "class",
            "special_params": ["param1", "param2"]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=300)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"JCL job card analysis failed: {str(e)}")
        
        return {
            "job_name": self._extract_jcl_job_name(content),
            "job_class": "UNKNOWN",
            "user": "UNKNOWN",
            "accounting": "UNKNOWN",
            "priority": "UNKNOWN",
            "time_limit": "UNKNOWN",
            "output_class": "UNKNOWN",
            "special_params": []
        }

    async def _analyze_jcl_step_comprehensive(self, content: str, step_name: str) -> Dict[str, Any]:
        """Comprehensive JCL step analysis"""
        await self._ensure_llm_engine()
        
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

    async def _analyze_jcl_dd_statement(self, content: str, dd_name: str) -> Dict[str, Any]:
        """Analyze JCL DD statement"""
        await self._ensure_llm_engine()
        
        dataset_info = self._extract_dd_dataset_info(content)
        
        return {
            "dd_name": dd_name,
            "dataset_info": dataset_info,
            "purpose": self._determine_dd_purpose(content, dd_name),
            "disposition": self._extract_dd_disposition(content),
            "space_allocation": self._extract_dd_space(content),
            "dcb_info": self._extract_dd_dcb(content)
        }

    async def _analyze_jcl_procedure(self, content: str, proc_name: str) -> Dict[str, Any]:
        """Analyze JCL procedure"""
        return {
            "procedure_name": proc_name,
            "parameters": self._extract_proc_parameters(content),
            "steps": self._extract_proc_steps(content),
            "purpose": f"JCL procedure {proc_name}"
        }

    async def _analyze_copybook_record(self, content: str, record_name: str) -> Dict[str, Any]:
        """Analyze copybook record structure"""
        return {
            "record_name": record_name,
            "fields": self._extract_record_fields(content),
            "total_length": self._calculate_record_length(content),
            "purpose": f"Data record {record_name}"
        }

    async def _analyze_fields_comprehensive(self, content: str) -> Dict[str, Any]:
        """Comprehensive field analysis"""
        fields = []
        field_stats = {
            "total_fields": 0,
            "numeric_fields": 0,
            "alphanumeric_fields": 0,
            "computational_fields": 0,
            "table_fields": 0,
            "redefines_fields": 0,
            "group_items": 0,
            "elementary_items": 0
        }
        
        data_matches = self.cobol_patterns['data_item'].finditer(content)
        
        for match in data_matches:
            level = int(match.group(2))
            name = match.group(3)
            definition = match.group(4)
            
            # Skip comment lines
            if match.group(1).strip().startswith('*'):
                continue
            
            pic_clause = self._extract_pic_clause(definition)
            usage = self._extract_usage_clause(definition)
            value = self._extract_value_clause(definition)
            occurs = self._extract_occurs_info(definition)
            redefines = self._extract_redefines_info(definition)
            
            field_info = {
                "level": level,
                "name": name,
                "pic_clause": pic_clause,
                "usage": usage,
                "value": value,
                "occurs": occurs,
                "redefines": redefines,
                "data_type": self._determine_data_type(pic_clause, usage),
                "length": self._calculate_field_length(pic_clause, usage),
                "is_group": pic_clause is None,
                "is_elementary": pic_clause is not None
            }
            
            fields.append(field_info)
            
            # Update statistics
            field_stats["total_fields"] += 1
            
            if pic_clause:
                field_stats["elementary_items"] += 1
                if '9' in pic_clause:
                    field_stats["numeric_fields"] += 1
                elif 'X' in pic_clause:
                    field_stats["alphanumeric_fields"] += 1
            else:
                field_stats["group_items"] += 1
            
            if usage in ['COMP', 'COMP-3', 'COMP-4', 'BINARY', 'PACKED-DECIMAL']:
                field_stats["computational_fields"] += 1
            
            if occurs:
                field_stats["table_fields"] += 1
            
            if redefines:
                field_stats["redefines_fields"] += 1
        
        return {
            "fields": fields,
            "statistics": field_stats
        }

    # Helper extraction methods
    def _extract_field_names(self, content: str) -> List[str]:
        """Extract field names from COBOL code"""
        fields = set()
        
        # Field definitions
        field_pattern = re.compile(r'\b\d+\s+([A-Z][A-Z0-9-]*)\s+PIC', re.IGNORECASE)
        fields.update(match.group(1) for match in field_pattern.finditer(content))
        
        # Field references in MOVE statements
        move_pattern = re.compile(r'\bMOVE\s+([A-Z][A-Z0-9-]*)\s+TO\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in move_pattern.finditer(content):
            fields.add(match.group(1))
            fields.add(match.group(2))
        
        # Field references in IF statements
        if_pattern = re.compile(r'\bIF\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        fields.update(match.group(1) for match in if_pattern.finditer(content))
        
        return list(fields)

    def _extract_file_operations(self, content: str) -> List[str]:
        """Extract file operations"""
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
        """Extract SQL operations"""
        ops = set()
        sql_pattern = re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|GRANT|REVOKE)\b', re.IGNORECASE)
        ops.update(match.group(1).upper() for match in sql_pattern.finditer(content))
        return list(ops)

    def _extract_perform_statements(self, content: str) -> List[str]:
        """Extract PERFORM statements"""
        performs = set()
        
        # Simple PERFORM
        simple_pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        performs.update(match.group(1) for match in simple_pattern.finditer(content))
        
        # PERFORM THRU
        thru_pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in thru_pattern.finditer(content):
            performs.add(f"{match.group(1)} THRU {match.group(2)}")
        
        return list(performs)

    def _extract_error_handling_patterns(self, content: str) -> List[str]:
        """Extract error handling patterns"""
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

    def _extract_sql_type(self, sql_content: str) -> str:
        """Extract SQL operation type"""
        sql_upper = sql_content.upper().strip()
        
        sql_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'GRANT', 'REVOKE']
        
        for sql_type in sql_types:
            if sql_upper.startswith(sql_type):
                return sql_type
        
        return 'UNKNOWN'

    def _extract_table_names(self, sql_content: str) -> List[str]:
        """Extract table names from SQL"""
        tables = set()
        
        # FROM clause
        from_pattern = re.compile(r'\bFROM\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        tables.update(match.group(1) for match in from_pattern.finditer(sql_content))
        
        # UPDATE table
        update_pattern = re.compile(r'\bUPDATE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        tables.update(match.group(1) for match in update_pattern.finditer(sql_content))
        
        # INSERT INTO
        insert_pattern = re.compile(r'\bINSERT\s+INTO\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        tables.update(match.group(1) for match in insert_pattern.finditer(sql_content))
        
        # JOIN clauses
        join_pattern = re.compile(r'\bJOIN\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        tables.update(match.group(1) for match in join_pattern.finditer(sql_content))
        
        return list(tables)

    # JCL helper methods
    def _extract_jcl_job_name(self, content: str) -> str:
        """Extract job name from JCL"""
        job_match = self.jcl_patterns['job_card'].search(content)
        return job_match.group(1) if job_match else "UNKNOWN"

    def _extract_exec_program(self, content: str) -> str:
        """Extract program name from EXEC statement"""
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
        """Extract detailed input dataset information"""
        datasets = []
        
        # Find DD statements
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            # Check if it's an input dataset
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
        """Extract detailed output dataset information"""
        datasets = []
        
        dd_matches = self.jcl_patterns['dd_statement'].finditer(content)
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_line = content[match.start():content.find('\n', match.start())]
            
            # Check if it's an output dataset
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
        """Extract detailed parameters"""
        params = {}
        
        # PARM parameters
        parm_match = re.search(r'PARM=([^,\s]+)', content, re.IGNORECASE)
        if parm_match:
            params['PARM'] = parm_match.group(1)
        
        # REGION parameter
        region_match = re.search(r'REGION=([^,\s]+)', content, re.IGNORECASE)
        if region_match:
            params['REGION'] = region_match.group(1)
        
        # TIME parameter
        time_match = re.search(r'TIME=([^,\s]+)', content, re.IGNORECASE)
        if time_match:
            params['TIME'] = time_match.group(1)
        
        return params

    def _extract_dd_dataset_info(self, content: str) -> Dict[str, str]:
        """Extract DD dataset information"""
        info = {}
        
        patterns = {
            'DSN': re.compile(r'DSN=([^,\s]+)', re.IGNORECASE),
            'DISP': re.compile(r'DISP=\(([^)]+)\)', re.IGNORECASE),
            'SPACE': re.compile(r'SPACE=\(([^)]+)\)', re.IGNORECASE),
            'DCB': re.compile(r'DCB=\(([^)]+)\)', re.IGNORECASE),
            'UNIT': re.compile(r'UNIT=([^,\s]+)', re.IGNORECASE),
            'VOL': re.compile(r'VOL=([^,\s]+)', re.IGNORECASE)
        }
        
        for param, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                info[param] = match.group(1)
        
        return info

    def _determine_dd_purpose(self, content: str, dd_name: str) -> str:
        """Determine DD statement purpose"""
        content_upper = content.upper()
        
        if any(name in dd_name.upper() for name in ['SYSIN', 'INPUT', 'IN']):
            return "input"
        elif any(name in dd_name.upper() for name in ['SYSOUT', 'OUTPUT', 'OUT', 'PRINT', 'LIST']):
            return "output"
        elif 'SYSLIB' in dd_name.upper():
            return "library"
        elif 'SYSDUMP' in dd_name.upper():
            return "dump"
        elif 'TEMP' in dd_name.upper() or 'WORK' in dd_name.upper():
            return "temporary"
        else:
            return "data"

    def _extract_dd_disposition(self, content: str) -> str:
        """Extract DD disposition"""
        disp_match = re.search(r'DISP=\(([^)]+)\)', content, re.IGNORECASE)
        return disp_match.group(1) if disp_match else "UNKNOWN"

    def _extract_dd_space(self, content: str) -> Dict[str, str]:
        """Extract DD space allocation"""
        space_match = re.search(r'SPACE=\(([^)]+)\)', content, re.IGNORECASE)
        if space_match:
            space_params = space_match.group(1).split(',')
            return {
                "unit": space_params[0] if len(space_params) > 0 else "UNKNOWN",
                "primary": space_params[1] if len(space_params) > 1 else "UNKNOWN",
                "secondary": space_params[2] if len(space_params) > 2 else "UNKNOWN"
            }
        return {}

    def _extract_dd_dcb(self, content: str) -> Dict[str, str]:
        """Extract DD DCB information"""
        dcb_match = re.search(r'DCB=\(([^)]+)\)', content, re.IGNORECASE)
        if dcb_match:
            dcb_params = {}
            dcb_string = dcb_match.group(1)
            
            # Parse DCB parameters
            dcb_patterns = {
                'RECFM': re.compile(r'RECFM=([^,)]+)', re.IGNORECASE),
                'LRECL': re.compile(r'LRECL=([^,)]+)', re.IGNORECASE),
                'BLKSIZE': re.compile(r'BLKSIZE=([^,)]+)', re.IGNORECASE),
                'DSORG': re.compile(r'DSORG=([^,)]+)', re.IGNORECASE)
            }
            
            for param, pattern in dcb_patterns.items():
                match = pattern.search(dcb_string)
                if match:
                    dcb_params[param] = match.group(1)
            
            return dcb_params
        return {}

    def _extract_proc_parameters(self, content: str) -> List[str]:
        """Extract procedure parameters"""
        params = []
        
        # Look for symbolic parameters
        param_pattern = re.compile(r'&([A-Z0-9]+)', re.IGNORECASE)
        params.extend(match.group(1) for match in param_pattern.finditer(content))
        
        return list(set(params))

    def _extract_proc_steps(self, content: str) -> List[str]:
        """Extract procedure step names"""
        steps = []
        
        step_matches = self.jcl_patterns['job_step'].finditer(content)
        steps.extend(match.group(1) for match in step_matches)
        
        return steps

    def _extract_record_fields(self, content: str) -> List[Dict[str, Any]]:
        """Extract fields from record"""
        fields = []
        
        data_matches = self.cobol_patterns['data_item'].finditer(content)
        
        for match in data_matches:
            level = int(match.group(2))
            name = match.group(3)
            definition = match.group(4)
            
            if level > 1:  # Skip 01 level (record itself)
                fields.append({
                    "level": level,
                    "name": name,
                    "pic": self._extract_pic_clause(definition),
                    "usage": self._extract_usage_clause(definition)
                })
        
        return fields

    def _calculate_record_length(self, content: str) -> int:
        """Calculate total record length"""
        total_length = 0
        
        data_matches = self.cobol_patterns['data_item'].finditer(content)
        
        for match in data_matches:
            level = int(match.group(2))
            definition = match.group(4)
            
            if level > 1:  # Skip group items for length calculation
                pic = self._extract_pic_clause(definition)
                usage = self._extract_usage_clause(definition)
                
                if pic:  # Only elementary items have length
                    field_length = self._calculate_field_length(pic, usage)
                    occurs_info = self._extract_occurs_info(definition)
                    
                    if occurs_info:
                        field_length *= occurs_info['max_occurs']
                    
                    total_length += field_length
        
        return total_length

    # Enhanced storage methods
    async def _store_chunks_enhanced(self, chunks: List[CodeChunk], file_hash: str):
        """Store chunks with enhanced duplicate handling"""
        if not chunks:
            self.logger.warning("No chunks to store")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            
                # Start transaction
                cursor.execute("BEGIN TRANSACTION")
            
                try:
                    stored_count = 0
                    skipped_count = 0
                
                    for chunk in chunks:
                        try:
                            # Ensure all values are properly converted to strings/appropriate types
                            program_name = str(chunk.program_name)
                            chunk_id = str(chunk.chunk_id)
                            chunk_type = str(chunk.chunk_type)
                            content = str(chunk.content)
                            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
                            embedding_id = hashlib.md5(content.encode()).hexdigest()
                            
                            # Check for existing chunk with same ID
                            cursor.execute("""
                            SELECT id FROM program_chunks 
                            WHERE program_name = ? AND chunk_id = ?
                            """, (program_name, chunk_id))
                        
                            existing = cursor.fetchone()
                        
                            if existing:
                                # Update existing chunk
                                cursor.execute("""
                                    UPDATE program_chunks 
                                    SET content = ?, metadata = ?, file_hash = ?, 
                                    line_start = ?, line_end = ?, created_timestamp = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                """, (
                                    content,
                                    metadata_json,
                                    str(file_hash),
                                    int(chunk.line_start),
                                    int(chunk.line_end),
                                    existing[0]
                                ))
                                self.logger.debug(f"Updated existing chunk: {chunk_id}")
                            else:
                            # Insert new chunk
                                cursor.execute("""
                                    INSERT INTO program_chunks 
                                    (program_name, chunk_id, chunk_type, content, metadata, 
                                    embedding_id, file_hash, line_start, line_end)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    program_name,
                                    chunk_id,
                                    chunk_type,
                                    content,
                                    metadata_json,
                                    embedding_id,
                                    str(file_hash),
                                    int(chunk.line_start),
                                    int(chunk.line_end)
                                ))
                                self.logger.debug(f"Inserted new chunk: {chunk_id}")
                        
                            stored_count += 1
                        
                        except sqlite3.Error as e:
                            self.logger.error(f"Failed to store chunk {chunk.chunk_id}: {str(e)}")
                            skipped_count += 1
                            continue
                
                # Commit transaction
                    cursor.execute("COMMIT")
                
                    self.logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks, skipped {skipped_count}")
                
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    self.logger.error(f"Transaction failed, rolled back: {str(e)}")
                    raise e
                
        except Exception as e:
            self.logger.error(f"Database operation failed: {str(e)}")
            raise e

    async def _verify_chunks_stored(self, program_name: str) -> int:
        """Verify chunks were stored"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks 
                WHERE program_name = ?
            """, (program_name,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to verify chunks for {program_name}: {str(e)}")
            return 0

    async def _generate_metadata_enhanced(self, chunks: List[CodeChunk], file_type: str) -> Dict[str, Any]:
        """Generate enhanced metadata"""
        metadata = {
            "total_chunks": len(chunks),
            "file_type": file_type,
            "chunk_types": {},
            "complexity_metrics": {},
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Count chunk types
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            metadata["chunk_types"][chunk_type] = metadata["chunk_types"].get(chunk_type, 0) + 1
        
        # Calculate complexity metrics
        if file_type == 'cobol':
            metadata["complexity_metrics"] = {
                "total_paragraphs": metadata["chunk_types"].get("paragraph", 0),
                "total_sql_blocks": metadata["chunk_types"].get("sql_block", 0),
                "total_cics_commands": metadata["chunk_types"].get("cics_command", 0),
                "total_file_operations": metadata["chunk_types"].get("file_operation", 0),
                "has_complex_logic": any(chunk.chunk_type in ["if_statement", "evaluate_statement", "perform_statement"] for chunk in chunks)
            }
        elif file_type == 'jcl':
            metadata["complexity_metrics"] = {
                "total_steps": metadata["chunk_types"].get("jcl_step", 0),
                "total_dd_statements": metadata["chunk_types"].get("jcl_dd_statement", 0),
                "has_procedures": metadata["chunk_types"].get("jcl_procedure", 0) > 0,
                "has_conditional_logic": metadata["chunk_types"].get("jcl_conditional", 0) > 0
            }
        
        # Extract aggregated information
        all_fields = set()
        all_files = set()
        all_operations = set()
        
        for chunk in chunks:
            if 'field_names' in chunk.metadata:
                if isinstance(chunk.metadata['field_names'], list):
                    all_fields.update(chunk.metadata['field_names'])
            if 'file_operations' in chunk.metadata:
                if isinstance(chunk.metadata['file_operations'], list):
                    all_operations.update(chunk.metadata['file_operations'])
            if 'tables' in chunk.metadata:
                if isinstance(chunk.metadata['tables'], list):
                    all_files.update(table['name'] if isinstance(table, dict) else str(table) 
                                   for table in chunk.metadata['tables'])
        
        metadata.update({
            "all_fields": list(all_fields),
            "all_files": list(all_files), 
            "all_operations": list(all_operations)
        })
        
        return metadata

    # Analysis methods for external use
    async def analyze_program(self, program_name: str) -> Dict[str, Any]:
        """Analyze a complete program"""
        await self._ensure_llm_engine()
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_type, content, metadata FROM program_chunks 
                WHERE program_name = ?
                ORDER BY chunk_type, chunk_id
            """, (program_name,))
            
            chunks = cursor.fetchall()
            conn.close()
            
            if not chunks:
                return {"error": f"Program {program_name} not found"}
            
            # Determine program type
            chunk_types = [chunk[0] for chunk in chunks]
            if any('jcl' in ct for ct in chunk_types):
                return await self._analyze_jcl_program(program_name, chunks)
            elif any('cics' in ct for ct in chunk_types):
                return await self._analyze_cics_program(program_name, chunks)
            elif any('bms' in ct for ct in chunk_types):
                return await self._analyze_bms_program(program_name, chunks)
            else:
                return await self._analyze_cobol_program(program_name, chunks)
                
        except Exception as e:
            self.logger.error(f"Program analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _analyze_cobol_program(self, program_name: str, chunks: List[Tuple]) -> Dict[str, Any]:
        """Analyze COBOL program comprehensively"""
        # Combine all chunk content for LLM analysis
        program_content = '\n'.join([chunk[1] for chunk in chunks[:10]])  # Limit for LLM
        
        prompt = f"""
        Analyze this complete COBOL program: {program_name}
        
        {program_content}
        
        Provide comprehensive analysis:
        1. Program structure and organization
        2. Main business logic and purpose
        3. Data processing patterns
        4. File and database operations
        5. Error handling approach
        6. Performance considerations
        7. Maintainability assessment
        
        Format as detailed technical analysis.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1200)
        
        try:
            analysis_text = await self._generate_with_llm(prompt, sampling_params)
        except Exception as e:
            analysis_text = f"Analysis failed: {str(e)}"
        
        # Compile statistics
        chunk_stats = {}
        for chunk_type, content, metadata_str in chunks:
            chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
        
        return {
            "program_name": program_name,
            "program_type": "COBOL",
            "total_chunks": len(chunks),
            "chunk_statistics": chunk_stats,
            "analysis": analysis_text,
            "complexity_score": self._calculate_complexity_score(chunks),
            "recommendations": self._generate_recommendations(chunks)
        }
     
    async def _analyze_jcl_program(self, program_name: str, chunks: List[Tuple]) -> Dict[str, Any]:
        """Analyze JCL program"""
        return {
            "program_name": program_name,
            "program_type": "JCL",
            "analysis": "JCL job flow analysis",
            "total_chunks": len(chunks)
        }

    async def _analyze_cics_program(self, program_name: str, chunks: List[Tuple]) -> Dict[str, Any]:
        """Analyze CICS program"""
        return {
            "program_name": program_name,
            "program_type": "CICS",
            "analysis": "CICS transaction program analysis", 
            "total_chunks": len(chunks)
        }

    async def _analyze_bms_program(self, program_name: str, chunks: List[Tuple]) -> Dict[str, Any]:
        """Analyze BMS program"""
        return {
            "program_name": program_name,
            "program_type": "BMS",
            "analysis": "BMS screen definition analysis",
            "total_chunks": len(chunks)
        }

    def _calculate_complexity_score(self, chunks: List[Tuple]) -> int:
        """Calculate program complexity score"""
        score = 0
        
        for chunk_type, content, metadata_str in chunks:
            if chunk_type == "paragraph":
                score += 2
            elif chunk_type in ["if_statement", "evaluate_statement"]:
                score += 3
            elif chunk_type == "perform_statement":
                score += 2
            elif chunk_type == "sql_block":
                score += 4
            elif chunk_type == "cics_command":
                score += 2
            elif chunk_type == "file_operation":
                score += 1
        
        return min(score, 100)  # Cap at 100
    
    async def _generate_field_lineage(self, program_name: str, chunks: List) -> List[Dict]:
        """Generate field lineage data from parsed chunks"""
        lineage_records = []
        
        for chunk in chunks:
            try:
                if isinstance(chunk, CodeChunk):
                    content = chunk.content
                    metadata = chunk.metadata or {}
                    chunk_id = chunk.chunk_id
                elif isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    metadata = chunk.get('metadata', {})
                    chunk_id = chunk.get('chunk_id', '')
                else:  # It's a tuple or other format
                    # Handle tuple format safely
                    if len(chunk) >= 5:
                        content = str(chunk[4]) if chunk[4] else ''
                        metadata = {}
                        chunk_id = str(chunk[2]) if chunk[2] else ''
                    else:
                        continue  # Skip malformed chunk
                        
                # Extract field operations from content
                field_operations = self._extract_field_operations(content)
                
                for field_op in field_operations:
                    lineage_record = {
                        'field_name': str(field_op.get('field_name', '')),
                        'program_name': str(program_name),
                        'paragraph': str(chunk_id),
                        'operation': str(field_op.get('operation', '')),
                        'source_file': str(field_op.get('source_file', '')),
                        'last_used': datetime.now().isoformat(),
                        'read_in': str(program_name) if field_op.get('operation') == 'READ' else '',
                        'updated_in': str(program_name) if field_op.get('operation') in ['WRITE', 'UPDATE'] else '',
                        'purged_in': str(program_name) if field_op.get('operation') == 'DELETE' else ''
                    }
                    lineage_records.append(lineage_record)
            except Exception as e:
                self.logger.error(f"Error processing chunk for lineage: {str(e)}")
                continue
        
        return lineage_records
    def _extract_field_operations(self, content: str) -> List[Dict]:
        """Extract field operations from COBOL content"""
        operations = []
        
        # Common COBOL field operation patterns
        patterns = {
            'READ': [r'READ\s+(\w+)', r'INTO\s+(\w+)'],
            'WRITE': [r'WRITE\s+(\w+)', r'MOVE\s+.+\s+TO\s+(\w+)'],
            'UPDATE': [r'REWRITE\s+(\w+)', r'ADD\s+.+\s+TO\s+(\w+)'],
            'DELETE': [r'DELETE\s+(\w+)']
        }
        
        for operation, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    field_name = match if isinstance(match, str) else match[0]
                    operations.append({
                        'field_name': field_name,
                        'operation': operation,
                        'source_file': ''  # Could be enhanced to detect file names
                    })
        
        return operations

    # Add this method to store lineage data
    async def _store_field_lineage(self, lineage_records: List[Dict]):
        """Store field lineage records in database"""
        if not lineage_records:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure field_lineage table exists
        cursor.execute("""
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
            )
        """)
        
        # Insert lineage records
        for record in lineage_records:
            cursor.execute("""
                INSERT INTO field_lineage 
                (field_name, program_name, paragraph, operation, source_file, 
                last_used, read_in, updated_in, purged_in)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['field_name'], record['program_name'], record['paragraph'],
                record['operation'], record['source_file'], record['last_used'],
                record['read_in'], record['updated_in'], record['purged_in']
            ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Stored {len(lineage_records)} field lineage records")

    def _generate_recommendations(self, chunks: List[Tuple]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        chunk_types = [chunk[0] for chunk in chunks]
        
        if chunk_types.count("paragraph") > 20:
            recommendations.append("Consider breaking down large programs into smaller modules")
        
        if chunk_types.count("sql_block") > 10:
            recommendations.append("Review SQL performance and consider optimization")
        
        if "error_handling" not in str([chunk[2] for chunk in chunks]):
            recommendations.append("Add comprehensive error handling")
        
        return recommendations

    # Cleanup and maintenance
    # Version 6 Changes - Starting from cleanup_duplicates method
    
    async def cleanup_duplicates(self) -> Dict[str, int]:
        """Clean up duplicate chunks"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find duplicates based on program_name and chunk_id
            cursor.execute("""
                SELECT program_name, chunk_id, COUNT(*) as count
                FROM program_chunks 
                GROUP BY program_name, chunk_id 
                HAVING COUNT(*) > 1
            """)
            
            duplicates = cursor.fetchall()
            removed_count = 0
            
            for program_name, chunk_id, count in duplicates:
                # Keep the most recent one, delete others
                cursor.execute("""
                    DELETE FROM program_chunks 
                    WHERE program_name = ? AND chunk_id = ? 
                    AND id NOT IN (
                        SELECT id FROM program_chunks 
                        WHERE program_name = ? AND chunk_id = ? 
                        ORDER BY created_timestamp DESC 
                        LIMIT 1
                    )
                """, (program_name, chunk_id, program_name, chunk_id))
                
                removed_count += count - 1
            
            conn.commit()
            conn.close()
            
            return {
                "duplicates_found": len(duplicates),
                "chunks_removed": removed_count
            }
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            return {"error": str(e)}

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total chunks
            cursor.execute("SELECT COUNT(*) FROM program_chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Chunks by type
            cursor.execute("""
                SELECT chunk_type, COUNT(*) 
                FROM program_chunks 
                GROUP BY chunk_type 
                ORDER BY COUNT(*) DESC
            """)
            chunks_by_type = dict(cursor.fetchall())
            
            # Programs by type
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN chunk_type LIKE '%jcl%' THEN 'JCL'
                        WHEN chunk_type LIKE '%cics%' THEN 'CICS'
                        WHEN chunk_type LIKE '%bms%' THEN 'BMS'
                        WHEN chunk_type LIKE '%sql%' THEN 'SQL'
                        ELSE 'COBOL'
                    END as program_type,
                    COUNT(DISTINCT program_name) as program_count
                FROM program_chunks 
                GROUP BY program_type
            """)
            programs_by_type = dict(cursor.fetchall())
            
            # Recent processing
            cursor.execute("""
                SELECT DATE(created_timestamp) as date, COUNT(*) as chunks_processed
                FROM program_chunks 
                WHERE created_timestamp >= datetime('now', '-7 days')
                GROUP BY DATE(created_timestamp)
                ORDER BY date DESC
            """)
            recent_activity = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_chunks": total_chunks,
                "chunks_by_type": chunks_by_type,
                "programs_by_type": programs_by_type,
                "recent_activity": recent_activity,
                "gpu_used": self.gpu_id,
                "using_coordinator_llm": self._using_coordinator_llm
            }
            
        except Exception as e:
            self.logger.error(f"Statistics retrieval failed: {str(e)}")
            return {"error": str(e)}

    # NEW: Batch processing methods
    async def process_directory(self, directory_path: Path, 
                               file_extensions: List[str] = None) -> Dict[str, Any]:
        """Process all files in a directory"""
        if file_extensions is None:
            file_extensions = ['.cbl', '.cob', '.jcl', '.cpy', '.copy', '.bms']
        
        results = {
            "processed_files": [],
            "failed_files": [],
            "skipped_files": [],
            "total_chunks": 0,
            "processing_summary": {}
        }
        
        try:
            # Find all matching files
            files_to_process = []
            for ext in file_extensions:
                files_to_process.extend(directory_path.glob(f"**/*{ext}"))
            
            self.logger.info(f"Found {len(files_to_process)} files to process")
            
            # Process files in batches to manage memory
            batch_size = 10
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                
                batch_tasks = [self.process_file(file_path) for file_path in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for file_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results["failed_files"].append({
                            "file": str(file_path),
                            "error": str(result)
                        })
                    elif result.get("status") == "success":
                        results["processed_files"].append(result)
                        results["total_chunks"] += result.get("chunks_created", 0)
                    elif result.get("status") == "skipped":
                        results["skipped_files"].append(result)
                    else:
                        results["failed_files"].append(result)
                
                # Log progress
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(files_to_process)-1)//batch_size + 1}")
            
            # Generate summary
            results["processing_summary"] = {
                "total_files_found": len(files_to_process),
                "successfully_processed": len(results["processed_files"]),
                "failed": len(results["failed_files"]),
                "skipped": len(results["skipped_files"]),
                "total_chunks_created": results["total_chunks"],
                "processing_timestamp": datetime.now().isoformat()
            }
            
            return self._add_processing_info(results)
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {str(e)}")
            return self._add_processing_info({
                "error": str(e),
                "processing_summary": results.get("processing_summary", {})
            })

    async def reprocess_program(self, program_name: str) -> Dict[str, Any]:
        """Reprocess a specific program (useful for updates)"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find original file information
            cursor.execute("""
                SELECT DISTINCT file_hash, metadata
                FROM program_chunks 
                WHERE program_name = ?
                LIMIT 1
            """, (program_name,))
            
            result = cursor.fetchone()
            if not result:
                return {"error": f"Program {program_name} not found"}
            
            # Delete existing chunks
            cursor.execute("""
                DELETE FROM program_chunks 
                WHERE program_name = ?
            """, (program_name,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Deleted existing chunks for {program_name}")
            
            # Note: File would need to be reprocessed from original source
            # This is a placeholder for the reprocessing logic
            
            return {
                "status": "chunks_deleted",
                "program_name": program_name,
                "message": "Program chunks deleted. Reprocess from original file."
            }
            
        except Exception as e:
            self.logger.error(f"Reprocess failed for {program_name}: {str(e)}")
            return {"error": str(e)}

    # NEW: Search and retrieval methods
    async def search_chunks(self, 
                           program_name: str = None,
                           chunk_type: str = None,
                           content_search: str = None,
                           limit: int = 100) -> Dict[str, Any]:
        """Search for chunks with various criteria"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build dynamic query
            where_clauses = []
            params = []
            
            if program_name:
                where_clauses.append("program_name LIKE ?")
                params.append(f"%{program_name}%")
            
            if chunk_type:
                where_clauses.append("chunk_type = ?")
                params.append(chunk_type)
            
            if content_search:
                where_clauses.append("content LIKE ?")
                params.append(f"%{content_search}%")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                SELECT program_name, chunk_id, chunk_type, content, metadata,
                       line_start, line_end, created_timestamp
                FROM program_chunks 
                WHERE {where_clause}
                ORDER BY program_name, chunk_id
                LIMIT ?
            """
            
            params.append(limit)
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            conn.close()
            
            chunks = []
            for row in rows:
                chunks.append({
                    "program_name": row[0],
                    "chunk_id": row[1], 
                    "chunk_type": row[2],
                    "content": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "line_start": row[5],
                    "line_end": row[6],
                    "created_timestamp": row[7]
                })
            
            return {
                "total_found": len(chunks),
                "chunks": chunks,
                "search_criteria": {
                    "program_name": program_name,
                    "chunk_type": chunk_type, 
                    "content_search": content_search,
                    "limit": limit
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return {"error": str(e)}

    async def get_program_overview(self, program_name: str) -> Dict[str, Any]:
        """Get comprehensive overview of a program"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic program info
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count,
                       MIN(line_start) as first_line,
                       MAX(line_end) as last_line
                FROM program_chunks 
                WHERE program_name = ?
                GROUP BY chunk_type
                ORDER BY count DESC
            """, (program_name,))
            
            chunk_summary = []
            for row in cursor.fetchall():
                chunk_summary.append({
                    "chunk_type": row[0],
                    "count": row[1],
                    "first_line": row[2],
                    "last_line": row[3]
                })
            
            if not chunk_summary:
                conn.close()
                return {"error": f"Program {program_name} not found"}
            
            # Get file info
            cursor.execute("""
                SELECT file_hash, created_timestamp, metadata
                FROM program_chunks 
                WHERE program_name = ?
                ORDER BY created_timestamp DESC
                LIMIT 1
            """, (program_name,))
            
            file_info = cursor.fetchone()
            
            # Get complex chunks for analysis
            cursor.execute("""
                SELECT chunk_type, content, metadata
                FROM program_chunks 
                WHERE program_name = ? 
                AND chunk_type IN ('sql_block', 'cics_command', 'paragraph', 'if_statement')
                ORDER BY chunk_type, line_start
            """, (program_name,))
            
            complex_chunks = cursor.fetchall()
            conn.close()
            
            # Analyze complexity
            complexity_analysis = {
                "total_chunks": sum(cs["count"] for cs in chunk_summary),
                "chunk_types": len(chunk_summary),
                "has_sql": any(cs["chunk_type"] == "sql_block" for cs in chunk_summary),
                "has_cics": any("cics" in cs["chunk_type"] for cs in chunk_summary),
                "has_complex_logic": any(cs["chunk_type"] in ["if_statement", "evaluate_statement"] for cs in chunk_summary),
                "estimated_lines": max((cs.get("last_line", 0) for cs in chunk_summary), default=0)
            }
            
            return {
                "program_name": program_name,
                "chunk_summary": chunk_summary,
                "complexity_analysis": complexity_analysis,
                "file_info": {
                    "file_hash": file_info[0] if file_info else None,
                    "last_processed": file_info[1] if file_info else None,
                    "metadata": json.loads(file_info[2]) if file_info and file_info[2] else {}
                },
                "sample_complex_chunks": len(complex_chunks)
            }
            
        except Exception as e:
            self.logger.error(f"Program overview failed: {str(e)}")
            return {"error": str(e)}

    # NEW: Export methods
    async def export_program_analysis(self, program_name: str, 
                                    export_format: str = "json") -> Dict[str, Any]:
        """Export complete program analysis"""
        try:
            # Get comprehensive program data
            overview = await self.get_program_overview(program_name)
            if "error" in overview:
                return overview
            
            analysis = await self.analyze_program(program_name)
            chunks = await self.search_chunks(program_name=program_name, limit=1000)
            
            export_data = {
                "program_name": program_name,
                "export_timestamp": datetime.now().isoformat(),
                "overview": overview,
                "analysis": analysis,
                "all_chunks": chunks.get("chunks", []),
                "export_format": export_format
            }
            
            if export_format == "json":
                return {
                    "status": "success",
                    "data": export_data,
                    "size_kb": len(json.dumps(export_data)) / 1024
                }
            else:
                return {"error": f"Unsupported export format: {export_format}"}
                
        except Exception as e:
            self.logger.error(f"Export failed for {program_name}: {str(e)}")
            return {"error": str(e)}

    # NEW: Database maintenance methods
    async def vacuum_database(self) -> Dict[str, Any]:
        """Vacuum database to reclaim space and optimize"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # Get size before
            cursor = conn.cursor()
            cursor.execute("PRAGMA page_count")
            pages_before = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            size_before = pages_before * page_size
            
            # Vacuum
            cursor.execute("VACUUM")
            
            # Get size after
            cursor.execute("PRAGMA page_count")
            pages_after = cursor.fetchone()[0]
            size_after = pages_after * page_size
            
            conn.close()
            
            return {
                "status": "success",
                "size_before_mb": size_before / (1024 * 1024),
                "size_after_mb": size_after / (1024 * 1024),
                "space_saved_mb": (size_before - size_after) / (1024 * 1024),
                "pages_before": pages_before,
                "pages_after": pages_after
            }
            
        except Exception as e:
            self.logger.error(f"Database vacuum failed: {str(e)}")
            return {"error": str(e)}

    async def backup_database(self, backup_path: str) -> Dict[str, Any]:
        """Create database backup"""
        try:
            import sqlite3
            import shutil
            
            # Simple file copy backup
            shutil.copy2(self.db_path, backup_path)
            
            # Verify backup
            backup_size = Path(backup_path).stat().st_size
            original_size = Path(self.db_path).stat().st_size
            
            return {
                "status": "success",
                "backup_path": backup_path,
                "original_size_mb": original_size / (1024 * 1024),
                "backup_size_mb": backup_size / (1024 * 1024),
                "backup_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {str(e)}")
            return {"error": str(e)}

# Export the main class
CodeParserAgent = CompleteEnhancedCodeParserAgent