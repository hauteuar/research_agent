"""
COMPLETE LLM CodeParser Agent with Full Database Storage
Part 1: Core Classes, Data Structures, and Initialization
"""

import json
import asyncio
import sqlite3
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
from datetime import datetime as dt
import logging

@dataclass
class CodeSection:
    """Represents a logical section of COBOL code"""
    section_type: str
    name: str
    content: str
    start_line: int
    end_line: int
    depends_on: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)

@dataclass
class DependencyAnalysis:
    """Analysis of program dependencies"""
    program_name: str
    required_copybooks: Set[str]
    called_programs: Set[str]
    accessed_files: Set[str]
    missing_copybooks: Set[str]
    missing_programs: Set[str]
    confidence_score: float

class CompleteLLMCodeParser:
    """
    Complete LLM-based code parser with full database integration
    Handles all mainframe patterns: COBOL, CICS, SQL, MQ, JCL, etc.
    """
    
    def __init__(self, coordinator, db_path: str = None, llm_engine = None, gpu_id: int = 0):
        self.coordinator = coordinator
        self.db_path = db_path or "opulence_data.db"
        self.logger = coordinator.logger if hasattr(coordinator, 'logger') else self._setup_logger()
        
        # LLM parameters
        self.base_llm_params = {
            "max_tokens": 2048,
            "temperature": 0.05,
            "top_p": 0.9
        }
        
        # Chunking parameters - Very aggressive limits to prevent 4888 token chunks
        self.max_tokens_per_chunk = 1800  # Much smaller to prevent oversized chunks
        self.min_chunk_lines = 30         # Reasonable minimum
        self.overlap_lines = 15           # Moderate overlap
        self.max_chunk_size = 150         # Much smaller line limit
        self.chars_per_token = 2.8        # More conservative estimate
        self.prompt_overhead_tokens = 800 # Conservative prompt overhead
        
        # Statistics tracking
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "llm_calls": 0,
            "total_relationships": 0,
            "dependency_checks": 0,
            "missing_dependencies": 0,
            "cache_hits": 0,
            "processing_time": 0.0
        }
        
        # Initialize complete database schema
        self._init_complete_database()

    def _setup_logger(self):
        """Setup basic logger if coordinator doesn't provide one"""
        logger = logging.getLogger('CompleteLLMCodeParser')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _init_complete_database(self):
        """Initialize complete database schema with all required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ==================== CORE CODEPARSER TABLES ====================
            
            # Program chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS program_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    file_hash TEXT,
                    confidence_score REAL DEFAULT 1.0,
                    line_start INTEGER,
                    line_end INTEGER,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, chunk_id)
                )
            """)
            
            # Program relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS program_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calling_program TEXT NOT NULL,
                    called_program TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    call_location TEXT,
                    line_number INTEGER,
                    call_statement TEXT,
                    parameters TEXT,
                    replacing_clause TEXT DEFAULT '',
                    conditional_call INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT DEFAULT 'llm',
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(calling_program, called_program, call_type, line_number)
                )
            """)
            
            # File access relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_access_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    logical_file_name TEXT NOT NULL,
                    physical_file_name TEXT DEFAULT '',
                    access_type TEXT NOT NULL,
                    access_mode TEXT DEFAULT '',
                    line_number INTEGER,
                    access_statement TEXT,
                    record_format TEXT DEFAULT '',
                    validation_status TEXT DEFAULT 'llm_validated',
                    false_positive_filtered INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT DEFAULT 'llm',
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, logical_file_name, access_type, line_number)
                )
            """)
            
            # Copybook relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS copybook_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    copybook_name TEXT NOT NULL,
                    copy_location TEXT DEFAULT 'UNKNOWN',
                    line_number INTEGER,
                    copy_statement TEXT,
                    replacing_clause TEXT DEFAULT '',
                    usage_context TEXT DEFAULT 'llm_extracted',
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT DEFAULT 'llm',
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, copybook_name, copy_location, line_number)
                )
            """)
            
            # Field definitions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_definitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    level_number INTEGER,
                    data_type TEXT,
                    picture_clause TEXT,
                    value_clause TEXT,
                    usage_clause TEXT,
                    line_number INTEGER,
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT DEFAULT 'llm',
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(field_name, source_name, line_number)
                )
            """)
            
            # SQL analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sql_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    sql_type TEXT NOT NULL,
                    tables_accessed TEXT,
                    operation_type TEXT,
                    line_number INTEGER,
                    sql_statement TEXT,
                    database_objects TEXT DEFAULT '',
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT DEFAULT 'llm',
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, operation_type, line_number)
                )
            """)
            
            # MQ operations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mq_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    queue_name TEXT,
                    queue_manager TEXT DEFAULT '',
                    line_number INTEGER,
                    operation_statement TEXT,
                    operation_details TEXT DEFAULT '',
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT DEFAULT 'llm',
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, operation_type, queue_name, line_number)
                )
            """)
            
            # Continue with remaining tables...
            self._create_remaining_tables(cursor)
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Complete database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _create_remaining_tables(self, cursor):
        """Create remaining database tables"""
        
        # WebSphere/XML operations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS websphere_xml_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                service_name TEXT,
                endpoint_url TEXT DEFAULT '',
                line_number INTEGER,
                operation_statement TEXT,
                service_details TEXT DEFAULT '',
                confidence_score REAL DEFAULT 1.0,
                extraction_method TEXT DEFAULT 'llm',
                created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(program_name, operation_type, service_name, line_number)
            )
        """)
        
        # JCL operations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jcl_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name TEXT NOT NULL,
                step_name TEXT,
                operation_type TEXT NOT NULL,
                program_name TEXT,
                dataset_name TEXT DEFAULT '',
                line_number INTEGER,
                operation_statement TEXT,
                step_details TEXT DEFAULT '',
                confidence_score REAL DEFAULT 1.0,
                extraction_method TEXT DEFAULT 'llm',
                created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(job_name, step_name, operation_type, line_number)
            )
        """)
        
        # Additional tables for lineage and metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS field_cross_reference (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT,
                qualified_name TEXT,
                source_type TEXT,
                source_name TEXT,
                definition_location TEXT,
                data_type TEXT,
                picture_clause TEXT,
                usage_clause TEXT,
                level_number INTEGER,
                parent_field TEXT,
                occurs_info TEXT,
                business_domain TEXT,
                confidence_score REAL DEFAULT 1.0,
                extraction_method TEXT DEFAULT 'llm',
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(field_name, source_name, definition_location)
            )
        """)
        
        # Dependency analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dependency_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT UNIQUE NOT NULL,
                required_copybooks TEXT,
                called_programs TEXT,
                accessed_files TEXT,
                missing_copybooks TEXT,
                missing_programs TEXT,
                confidence_score REAL DEFAULT 1.0,
                analysis_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                extraction_method TEXT DEFAULT 'llm'
            )
        """)
        
        # LLM analysis cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL UNIQUE,
                analysis_type TEXT NOT NULL,
                analysis_result TEXT,
                confidence_score REAL DEFAULT 1.0,
                model_version TEXT DEFAULT 'codellama',
                created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # File metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE NOT NULL,
                file_type TEXT,
                table_name TEXT,
                fields TEXT,
                source_type TEXT,
                last_modified TIMESTAMP,
                processing_status TEXT DEFAULT 'processed',
                extraction_method TEXT DEFAULT 'llm',
                confidence_score REAL DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create performance indexes
        self._create_indexes(cursor)

    def _create_indexes(self, cursor):
        """Create all indexes for performance optimization"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_prog_chunks_name ON program_chunks(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_prog_rel_calling ON program_relationships(calling_program)",
            "CREATE INDEX IF NOT EXISTS idx_prog_rel_called ON program_relationships(called_program)",
            "CREATE INDEX IF NOT EXISTS idx_file_rel_program ON file_access_relationships(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_file_rel_file ON file_access_relationships(logical_file_name)",
            "CREATE INDEX IF NOT EXISTS idx_copy_rel_program ON copybook_relationships(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_copy_rel_copybook ON copybook_relationships(copybook_name)",
            "CREATE INDEX IF NOT EXISTS idx_field_def_source ON field_definitions(source_name)",
            "CREATE INDEX IF NOT EXISTS idx_field_def_field ON field_definitions(field_name)",
            "CREATE INDEX IF NOT EXISTS idx_sql_program ON sql_analysis(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_mq_program ON mq_operations(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_websphere_program ON websphere_xml_operations(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_jcl_job ON jcl_operations(job_name)",
            "CREATE INDEX IF NOT EXISTS idx_field_xref_field ON field_cross_reference(field_name)",
            "CREATE INDEX IF NOT EXISTS idx_field_xref_source ON field_cross_reference(source_name)",
            "CREATE INDEX IF NOT EXISTS idx_dependency_program ON dependency_analysis(program_name)",
            "CREATE INDEX IF NOT EXISTS idx_cache_hash ON llm_analysis_cache(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_file_metadata_name ON file_metadata(file_name)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    """
COMPLETE LLM CodeParser Agent with Full Database Storage
Part 2: Prompt Creation and Intelligent Code Chunking
"""

    def create_focused_analysis_prompt(self, code_section: CodeSection, 
                                     file_type: str, program_name: str) -> str:
        """Create focused prompt optimized for 4096 context window"""
        
        # Analyze what's actually in the code to determine focus
        content_upper = code_section.content.upper()
        
        # Determine primary focus based on content
        has_cics = 'EXEC CICS' in content_upper
        has_sql = 'EXEC SQL' in content_upper  
        has_copy = 'COPY ' in content_upper
        has_call = 'CALL ' in content_upper
        has_fields = 'PIC ' in content_upper and ('05 ' in content_upper or '01 ' in content_upper)
        has_file_ops = any(op in content_upper for op in ['READ ', 'WRITE ', 'OPEN ', 'CLOSE '])
        has_mq = any(mq in content_upper for mq in ['MQOPEN', 'MQPUT', 'MQGET', 'MQCLOSE'])
        
        # Create comprehensive but focused prompt for 4096 context
        if has_cics:
            return self._create_cics_comprehensive_prompt(code_section, program_name)
        elif has_sql:
            return self._create_sql_comprehensive_prompt(code_section, program_name)
        elif has_copy:
            return self._create_copy_comprehensive_prompt(code_section, program_name)
        elif has_call:
            return self._create_call_comprehensive_prompt(code_section, program_name)
        elif has_fields:
            return self._create_field_comprehensive_prompt(code_section, program_name)
        elif has_file_ops:
            return self._create_file_comprehensive_prompt(code_section, program_name)
        elif has_mq:
            return self._create_mq_comprehensive_prompt(code_section, program_name)
        else:
            return self._create_multi_pattern_prompt(code_section, program_name)

    def _create_sql_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter SQL analysis prompt to prevent echo-back"""
        return f"""Find SQL operations in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "sql_operations": [
    {{"type": "SQL_SELECT", "target": "table_name", "line_number": 123, "confidence": 0.9}}
  ],
  "field_definitions": [
    {{"type": "FIELD_DEFINITION", "target": "field_name", "line_number": 67, "picture_clause": "X(10)", "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_cics_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter CICS analysis prompt to prevent echo-back"""
        return f"""Find CICS operations in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "cics_operations": [
    {{"type": "CICS_READ", "target": "dataset_name", "line_number": 123, "confidence": 0.9}}
  ],
  "field_definitions": [
    {{"type": "FIELD_DEFINITION", "target": "field_name", "line_number": 67, "picture_clause": "X(10)", "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_copy_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter COPY analysis prompt to prevent echo-back"""
        return f"""Find COPY statements in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "copybook_includes": [
    {{"type": "COPY", "target": "copybook_name", "line_number": 123, "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_call_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter CALL analysis prompt to prevent echo-back"""
        return f"""Find CALL statements in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "program_calls": [
    {{"type": "CALL", "target": "program_name", "line_number": 123, "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_field_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter field analysis prompt to prevent echo-back"""
        return f"""Find field definitions in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "field_definitions": [
    {{"type": "FIELD_DEFINITION", "target": "field_name", "line_number": 67, "picture_clause": "X(10)", "level_number": 5, "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_file_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter file operations prompt to prevent echo-back"""
        return f"""Find file operations in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "file_operations": [
    {{"type": "READ", "target": "file_name", "line_number": 123, "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_mq_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter MQ operations prompt to prevent echo-back"""
        return f"""Find MQ operations in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "mq_operations": [
    {{"type": "MQOPEN", "target": "queue_name", "line_number": 123, "confidence": 0.9}}
  ]
}}

Return only JSON, no explanations."""

    def _create_multi_pattern_prompt(self, section: CodeSection, program_name: str) -> str:
        """Much shorter multi-pattern analysis prompt to prevent echo-back"""
        return f"""Find ALL patterns in this COBOL code. Return only JSON:

{section.content}

Required JSON format:
{{
  "cics_operations": [],
  "copybook_includes": [],
  "program_calls": [],
  "sql_operations": [],
  "field_definitions": [],
  "file_operations": []
}}

Fill arrays with findings. Return only JSON, no explanations."""

    def _create_sql_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive SQL analysis prompt with strict JSON requirements"""
        return f"""Analyze this COBOL code for SQL operations. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "sql_operations": [
    {{
      "type": "SQL_SELECT",
      "target": "CUSTOMER",
      "line_number": 200,
      "statement": "EXEC SQL SELECT...",
      "confidence": 0.96
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "WS-ID",
      "line_number": 67,
      "statement": "05 WS-ID PIC X(10)",
      "picture_clause": "X(10)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Use empty arrays [] for categories with no findings."""

    def _create_copy_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive COPY analysis prompt with strict JSON requirements"""
        return f"""Analyze this COBOL code for COPY statements. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "copybook_includes": [
    {{
      "type": "COPY",
      "target": "TMS06ASU",
      "line_number": 123,
      "statement": "COPY TMS06ASU",
      "location": "WORKING-STORAGE",
      "confidence": 0.99
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "WS-NAME",
      "line_number": 67,
      "statement": "05 WS-NAME PIC X(30)",
      "picture_clause": "X(30)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Use empty arrays [] for categories with no findings."""

    def _create_call_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive CALL analysis prompt with strict JSON requirements"""
        return f"""Analyze this COBOL code for program calls. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "program_calls": [
    {{
      "type": "CALL",
      "target": "SUBPROG",
      "line_number": 156,
      "statement": "CALL 'SUBPROG' USING WS-PARMS",
      "confidence": 0.97
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "WS-PARMS",
      "line_number": 67,
      "statement": "05 WS-PARMS PIC X(100)",
      "picture_clause": "X(100)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Use empty arrays [] for categories with no findings."""

    def _create_field_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive field analysis prompt with strict JSON requirements"""
        return f"""Analyze this COBOL code for field definitions. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "WS-CUSTOMER-NAME",
      "line_number": 67,
      "statement": "05 WS-CUSTOMER-NAME PIC X(30) VALUE SPACES",
      "picture_clause": "X(30)",
      "value_clause": "SPACES",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Use empty arrays [] if no field definitions found."""

    def _create_file_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive file operations prompt with strict JSON requirements"""
        return f"""Analyze this COBOL code for file operations. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "file_operations": [
    {{
      "type": "READ",
      "target": "CUSTOMER-FILE",
      "line_number": 123,
      "statement": "READ CUSTOMER-FILE INTO WS-CUSTOMER-REC",
      "confidence": 0.95
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "WS-CUSTOMER-REC",
      "line_number": 67,
      "statement": "01 WS-CUSTOMER-REC PIC X(100)",
      "picture_clause": "X(100)",
      "level_number": 1,
      "confidence": 0.98
    }}
  ]
}}

Use empty arrays [] for categories with no findings."""

    def _create_mq_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive MQ operations prompt with strict JSON requirements"""
        return f"""Analyze this COBOL code for MQ operations. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "mq_operations": [
    {{
      "type": "MQOPEN",
      "target": "QUEUE1",
      "line_number": 300,
      "statement": "CALL 'MQOPEN' USING HCONN, OBJDESC",
      "confidence": 0.94
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "HCONN",
      "line_number": 67,
      "statement": "05 HCONN PIC S9(9) COMP",
      "picture_clause": "S9(9)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Use empty arrays [] for categories with no findings."""

    def _create_multi_pattern_prompt(self, section: CodeSection, program_name: str) -> str:
        """Multi-pattern analysis with strict JSON requirements"""
        return f"""Analyze this COBOL code for ALL patterns. Return ONLY valid JSON, no explanations.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

CRITICAL: Return ONLY valid JSON in this EXACT format. No additional text:

{{
  "file_operations": [],
  "cics_operations": [],
  "copybook_includes": [],
  "program_calls": [],
  "sql_operations": [],
  "field_definitions": []
}}

Fill arrays only with actual findings. Example format for findings:
- "type": "CICS_READ", "target": "TMS92ASO", "line_number": 123, "confidence": 0.95
- "type": "COPY", "target": "TMS06ASU", "line_number": 45, "confidence": 0.99
- "type": "FIELD_DEFINITION", "target": "WS-NAME", "picture_clause": "X(30)", "level_number": 5

Use empty arrays [] for categories with no findings."""

    def _create_sql_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive SQL analysis prompt for 4096 context"""
        return f"""Analyze this COBOL code section for SQL operations and related patterns.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL SQL operations and any related patterns. Return JSON:

{{
  "sql_operations": [
    {{
      "type": "SQL_SELECT|SQL_INSERT|SQL_UPDATE|SQL_DELETE|SQL_CALL",
      "target": "table_or_procedure_name",
      "line_number": 200,
      "statement": "EXEC SQL SELECT * FROM CUSTOMER WHERE ID = :WS-ID",
      "database_objects": ["CUSTOMER"],
      "confidence": 0.96
    }}
  ],
  "stored_procedures": [
    {{
      "type": "CREATE_PROCEDURE|EXEC_PROCEDURE|CALL_PROCEDURE",
      "target": "procedure_name",
      "line_number": 500,
      "statement": "EXEC SQL CALL MYPROC(:PARM1, :PARM2)",
      "parameters": ":PARM1, :PARM2",
      "confidence": 0.93
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "05 WS-ID PIC X(10)",
      "picture_clause": "X(10)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Focus on SQL operations but include field definitions that are used in SQL statements."""

    def _create_copy_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive COPY analysis prompt for 4096 context"""
        return f"""Analyze this COBOL code section for COPY statements and related patterns.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL COPY statements and any related field definitions. Return JSON:

{{
  "copybook_includes": [
    {{
      "type": "COPY",
      "target": "copybook_name",
      "line_number": 123,
      "statement": "COPY TMS06ASU REPLACING ==TAG== BY ==PROD==",
      "location": "WORKING-STORAGE",
      "replacing_clause": "==TAG== BY ==PROD==",
      "confidence": 0.99
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "05 WS-CUSTOMER-NAME PIC X(30)",
      "picture_clause": "X(30)",
      "level_number": 5,
      "business_domain": "CUSTOMER",
      "confidence": 0.98
    }}
  ]
}}

Focus on COPY statements and include any field definitions that appear in the same section."""

    def _create_call_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive CALL analysis prompt for 4096 context"""
        return f"""Analyze this COBOL code section for program calls and related patterns.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL program calls and any related patterns. Return JSON:

{{
  "program_calls": [
    {{
      "type": "CALL",
      "target": "program_name",
      "line_number": 156,
      "statement": "CALL 'SUBPROG' USING WS-PARMS",
      "parameters": "WS-PARMS",
      "confidence": 0.97
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "05 WS-PARMS PIC X(100)",
      "picture_clause": "X(100)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Focus on CALL statements and include parameter field definitions."""

    def _create_field_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive field analysis prompt for 4096 context"""
        return f"""Analyze this COBOL code section for field definitions and data structures.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL field definitions and data structures. Return JSON:

{{
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "05 WS-CUSTOMER-NAME PIC X(30) VALUE SPACES",
      "data_type": "CHARACTER",
      "picture_clause": "X(30)",
      "value_clause": "SPACES",
      "level_number": 5,
      "business_domain": "CUSTOMER",
      "confidence": 0.98
    }}
  ]
}}

Include ALL field definitions with PIC clauses, VALUE clauses, and 88-level condition names."""

    def _create_file_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive file operations prompt for 4096 context"""
        return f"""Analyze this COBOL code section for file operations and related patterns.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL file operations and related patterns. Return JSON:

{{
  "file_operations": [
    {{
      "type": "READ|WRITE|OPEN|CLOSE|REWRITE",
      "target": "file_name",
      "line_number": 123,
      "statement": "READ CUSTOMER-FILE INTO WS-CUSTOMER-REC",
      "access_mode": "INPUT",
      "confidence": 0.95
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "01 WS-CUSTOMER-REC PIC X(100)",
      "picture_clause": "X(100)",
      "level_number": 1,
      "confidence": 0.98
    }}
  ]
}}

Focus on file I/O operations and include related record definitions."""

    def _create_mq_comprehensive_prompt(self, section: CodeSection, program_name: str) -> str:
        """Comprehensive MQ operations prompt for 4096 context"""
        return f"""Analyze this COBOL code section for MQ operations and related patterns.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL MQ operations and related patterns. Return JSON:

{{
  "mq_operations": [
    {{
      "type": "MQOPEN|MQPUT|MQGET|MQCLOSE",
      "target": "queue_name",
      "line_number": 300,
      "statement": "CALL 'MQOPEN' USING HCONN, OBJDESC",
      "queue_manager": "QM1",
      "confidence": 0.94
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "05 HCONN PIC S9(9) COMP",
      "picture_clause": "S9(9)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Focus on MQ operations and include related MQ field definitions."""

    def _create_multi_pattern_prompt(self, section: CodeSection, program_name: str) -> str:
        """Multi-pattern analysis for mixed content with 4096 context"""
        return f"""Analyze this COBOL code section for ALL types of patterns and relationships.

PROGRAM: {program_name}
SECTION: {section.name} (Lines {section.start_line}-{section.end_line})

CODE:
{section.content}

Extract ALL patterns found. Return JSON with any categories that have findings:

{{
  "file_operations": [
    {{
      "type": "READ|WRITE|OPEN|CLOSE|REWRITE",
      "target": "file_name",
      "line_number": 123,
      "statement": "READ CUSTOMER-FILE INTO WS-CUSTOMER-REC",
      "confidence": 0.95
    }}
  ],
  "cics_operations": [
    {{
      "type": "CICS_READ|CICS_WRITE|CICS_LINK|CICS_XCTL",
      "target": "dataset_or_program_name",
      "line_number": 124,
      "statement": "EXEC CICS READ DATASET('TMS92ASO')",
      "confidence": 0.98
    }}
  ],
  "copybook_includes": [
    {{
      "type": "COPY",
      "target": "copybook_name",
      "line_number": 45,
      "statement": "COPY TMS06ASU",
      "location": "WORKING-STORAGE",
      "confidence": 0.99
    }}
  ],
  "program_calls": [
    {{
      "type": "CALL",
      "target": "program_name",
      "line_number": 156,
      "statement": "CALL 'SUBPROG' USING WS-PARMS",
      "confidence": 0.97
    }}
  ],
  "sql_operations": [
    {{
      "type": "SQL_SELECT|SQL_INSERT|SQL_UPDATE|SQL_DELETE",
      "target": "table_name",
      "line_number": 200,
      "statement": "EXEC SQL SELECT * FROM CUSTOMER",
      "confidence": 0.96
    }}
  ],
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 67,
      "statement": "05 WS-CUSTOMER-NAME PIC X(30)",
      "picture_clause": "X(30)",
      "level_number": 5,
      "confidence": 0.98
    }}
  ]
}}

Only include categories that have actual findings. Be thorough but precise."""

    def _create_cics_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for CICS operations only"""
        return f"""Extract CICS operations from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all EXEC CICS statements. Return only JSON:
{{
  "cics_operations": [
    {{
      "type": "CICS_READ|CICS_WRITE|CICS_LINK|CICS_XCTL|CICS_START|CICS_RETURN",
      "target": "dataset_or_program_name",
      "line_number": 123,
      "statement": "EXEC CICS READ DATASET('TMS92ASO')",
      "confidence": 0.95
    }}
  ]
}}

Only include actual CICS operations found. Be precise."""

    def _create_sql_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for SQL operations only"""
        return f"""Extract SQL operations from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all EXEC SQL statements. Return only JSON:
{{
  "sql_operations": [
    {{
      "type": "SQL_SELECT|SQL_INSERT|SQL_UPDATE|SQL_DELETE|SQL_CALL",
      "target": "table_or_procedure_name",
      "line_number": 123,
      "statement": "EXEC SQL SELECT * FROM CUSTOMER",
      "confidence": 0.95
    }}
  ]
}}

Only include actual SQL operations found. Be precise."""

    def _create_copybook_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for COPY statements only"""
        return f"""Extract COPY statements from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all COPY statements. Return only JSON:
{{
  "copybook_includes": [
    {{
      "type": "COPY",
      "target": "copybook_name",
      "line_number": 123,
      "statement": "COPY TMS06ASU",
      "location": "WORKING-STORAGE|FILE-SECTION|LINKAGE-SECTION",
      "confidence": 0.95
    }}
  ]
}}

Only include actual COPY statements found. Be precise."""

    def _create_call_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for CALL statements only"""
        return f"""Extract CALL statements from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all CALL statements. Return only JSON:
{{
  "program_calls": [
    {{
      "type": "CALL",
      "target": "program_name",
      "line_number": 123,
      "statement": "CALL 'SUBPROG' USING WS-PARMS",
      "confidence": 0.95
    }}
  ]
}}

Only include actual CALL statements found. Be precise."""

    def _create_field_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for field definitions only"""
        return f"""Extract field definitions from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all field definitions with PIC clauses. Return only JSON:
{{
  "field_definitions": [
    {{
      "type": "FIELD_DEFINITION",
      "target": "field_name",
      "line_number": 123,
      "statement": "05 WS-CUSTOMER-NAME PIC X(30)",
      "picture_clause": "X(30)",
      "level_number": 5,
      "confidence": 0.95
    }}
  ]
}}

Only include actual field definitions found. Be precise."""

    def _create_file_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for file operations only"""
        return f"""Extract file operations from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all file I/O operations. Return only JSON:
{{
  "file_operations": [
    {{
      "type": "READ|WRITE|OPEN|CLOSE|REWRITE",
      "target": "file_name",
      "line_number": 123,
      "statement": "READ CUSTOMER-FILE INTO WS-CUSTOMER-REC",
      "confidence": 0.95
    }}
  ]
}}

Only include actual file operations found. Be precise."""

    def _create_mq_focused_prompt(self, section: CodeSection, program_name: str) -> str:
        """Focused prompt for MQ operations only"""
        return f"""Extract MQ operations from this COBOL code.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Find all MQ operations. Return only JSON:
{{
  "mq_operations": [
    {{
      "type": "MQOPEN|MQPUT|MQGET|MQCLOSE",
      "target": "queue_name",
      "line_number": 123,
      "statement": "CALL 'MQOPEN' USING HCONN, OBJDESC",
      "confidence": 0.95
    }}
  ]
}}

Only include actual MQ operations found. Be precise."""

    def _create_minimal_prompt(self, section: CodeSection, program_name: str) -> str:
        """Minimal prompt for general analysis"""
        return f"""Analyze this COBOL code section for any relationships.

PROGRAM: {program_name}
SECTION: {section.name}

CODE:
{section.content}

Return JSON with any relationships found:
{{
  "relationships": [
    {{
      "type": "relationship_type",
      "target": "target_name",
      "line_number": 123,
      "confidence": 0.95
    }}
  ]
}}

Be precise and only include what you actually find."""

    def intelligent_chunk_cobol(self, content: str, program_name: str) -> List[CodeSection]:
        """Intelligently chunk COBOL code based on structure"""
        sections = []
        lines = content.split('\n')
        
        # Find COBOL boundaries (divisions, sections, paragraphs)
        division_boundaries = self._find_cobol_boundaries(lines)
        
        if not division_boundaries:
            return self._fallback_line_chunking(content, program_name)
        
        # Create sections based on COBOL structure
        for i, (boundary_type, name, start_line, end_line) in enumerate(division_boundaries):
            section_content = '\n'.join(lines[start_line:end_line])
            estimated_tokens = self._estimate_tokens(section_content)
            
            # Check if section fits within limits (accounting for prompt overhead)
            if estimated_tokens + self.prompt_overhead_tokens <= self.max_tokens_per_chunk:
                sections.append(CodeSection(
                    section_type=boundary_type,
                    name=name,
                    content=section_content,
                    start_line=start_line + 1,
                    end_line=end_line
                ))
            else:
                # Split large sections
                sub_sections = self._split_large_section(
                    section_content, boundary_type, name, start_line
                )
                sections.extend(sub_sections)
        
        if not sections:
            sections = self._fallback_line_chunking(content, program_name)
        
        # Validate all sections are within token limits
        validated_sections = []
        for section in sections:
            if self._validate_chunk_size(section):
                validated_sections.append(section)
            else:
                # Force split oversized chunks
                split_chunks = self._force_split_chunk(section, program_name)
                validated_sections.extend(split_chunks)
        
        self.logger.info(f"Created {len(validated_sections)} validated sections for {program_name}")
        return validated_sections

    def _find_cobol_boundaries(self, lines: List[str]) -> List[Tuple[str, str, int, int]]:
        """Find COBOL division and section boundaries"""
        boundaries = []
        current_start = 0
        
        division_pattern = re.compile(r'^\s*(\w+)\s+DIVISION\s*\.', re.IGNORECASE)
        section_pattern = re.compile(r'^\s*(\w+(?:-\w+)*)\s+SECTION\s*\.', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            if not line.strip() or line.strip().startswith('*'):
                continue
            
            # Check for divisions
            div_match = division_pattern.match(line)
            if div_match:
                if boundaries and current_start < i:
                    boundaries[-1] = (*boundaries[-1][:3], i)
                boundaries.append(('DIVISION', div_match.group(1), i, len(lines)))
                current_start = i
                continue
            
            # Check for sections
            sec_match = section_pattern.match(line)
            if sec_match and boundaries:
                if current_start < i:
                    boundaries[-1] = (*boundaries[-1][:3], i)
                boundaries.append(('SECTION', sec_match.group(1), i, len(lines)))
                current_start = i
        
        return boundaries

    def _split_large_section(self, content: str, section_type: str, 
                           section_name: str, start_line_offset: int) -> List[CodeSection]:
        """Split large sections into smaller chunks"""
        sections = []
        lines = content.split('\n')
        
        # Use very conservative token limit for content
        max_content_tokens = 1000  # Much smaller to prevent oversized chunks
        
        # For PROCEDURE DIVISION, split by paragraphs
        if section_type == 'DIVISION' and 'PROCEDURE' in section_name.upper():
            paragraph_starts = []
            
            for i, line in enumerate(lines):
                if (line.strip() and 
                    not line.strip().startswith('*') and 
                    line.strip().endswith('.') and 
                    len(line.strip().split()) == 1):
                    paragraph_starts.append(i)
            
            # Create paragraph chunks
            for i, para_start in enumerate(paragraph_starts):
                para_end = paragraph_starts[i + 1] if i + 1 < len(paragraph_starts) else len(lines)
                para_content = '\n'.join(lines[para_start:para_end])
                
                # Check if paragraph fits within conservative limits
                para_tokens = self._estimate_tokens(para_content)
                if para_tokens <= max_content_tokens:
                    para_name = lines[para_start].strip().rstrip('.')
                    new_section = CodeSection(
                        section_type='PARAGRAPH',
                        name=para_name,
                        content=para_content,
                        start_line=start_line_offset + para_start + 1,
                        end_line=start_line_offset + para_end
                    )
                    
                    # Double-check validation
                    if self._validate_chunk_size(new_section):
                        sections.append(new_section)
                    else:
                        # Paragraph still too large, split by lines
                        line_chunks = self._split_by_lines(para_content, para_start + start_line_offset)
                        sections.extend(line_chunks)
                else:
                    # Paragraph too large, split by lines
                    line_chunks = self._split_by_lines(para_content, para_start + start_line_offset)
                    sections.extend(line_chunks)
        else:
            # Split by lines for other divisions
            line_chunks = self._split_by_lines(content, start_line_offset)
            sections.extend(line_chunks)
        
        return sections

    def _split_by_lines(self, content: str, start_line_offset: int) -> List[CodeSection]:
        """Split content by lines when structure-based splitting isn't possible"""
        sections = []
        lines = content.split('\n')
        
        chunk_start = 0
        chunk_num = 1
        max_content_tokens = 1000  # Very conservative content limit
        
        while chunk_start < len(lines):
            chunk_lines = []
            current_tokens = 0
            
            # Build chunk line by line, checking tokens aggressively
            for i in range(chunk_start, len(lines)):
                line = lines[i]
                line_tokens = self._estimate_tokens(line)
                
                # Check if adding this line would exceed very conservative content limit
                if current_tokens + line_tokens > max_content_tokens and chunk_lines:
                    break
                
                chunk_lines.append(line)
                current_tokens += line_tokens
                
                # Also check line count limit (much smaller chunks)
                if len(chunk_lines) >= 60:  # Very small chunks
                    break
            
            if chunk_lines:
                chunk_content = '\n'.join(chunk_lines)
                new_section = CodeSection(
                    section_type='CHUNK',
                    name=f'CHUNK-{chunk_num}',
                    content=chunk_content,
                    start_line=start_line_offset + chunk_start + 1,
                    end_line=start_line_offset + chunk_start + len(chunk_lines)
                )
                
                # Validate the chunk before adding
                if self._validate_chunk_size(new_section):
                    sections.append(new_section)
                else:
                    # If still too large, make it even smaller
                    self.logger.warning(f"Line-based chunk still too large, making smaller")
                    half_lines = chunk_lines[:len(chunk_lines)//2]
                    if half_lines:
                        half_content = '\n'.join(half_lines)
                        half_section = CodeSection(
                            section_type='EMERGENCY',
                            name=f'EMERGENCY-{chunk_num}',
                            content=half_content,
                            start_line=start_line_offset + chunk_start + 1,
                            end_line=start_line_offset + chunk_start + len(half_lines)
                        )
                        sections.append(half_section)
                        chunk_start += len(half_lines)
                    else:
                        # Single line emergency
                        if chunk_start < len(lines):
                            single_line = lines[chunk_start]
                            single_section = CodeSection(
                                section_type='SINGLE_LINE',
                                name=f'SINGLE-{chunk_num}',
                                content=single_line,
                                start_line=start_line_offset + chunk_start + 1,
                                end_line=start_line_offset + chunk_start + 1
                            )
                            sections.append(single_section)
                            chunk_start += 1
                
                chunk_start += len(chunk_lines) - self.overlap_lines
                chunk_num += 1
                
                if chunk_start >= len(lines) - self.overlap_lines:
                    break
            else:
                # Emergency: single line chunk
                if chunk_start < len(lines):
                    single_line = lines[chunk_start]
                    sections.append(CodeSection(
                        section_type='EMERGENCY',
                        name=f'EMERGENCY-{chunk_num}',
                        content=single_line,
                        start_line=start_line_offset + chunk_start + 1,
                        end_line=start_line_offset + chunk_start + 1
                    ))
                    chunk_start += 1
                    chunk_num += 1
                else:
                    break
        
        return sections

    def _fallback_line_chunking(self, content: str, program_name: str) -> List[CodeSection]:
        """Fallback chunking when structure detection fails"""
        sections = []
        lines = content.split('\n')
        
        chunk_size = self.max_chunk_size
        chunk_num = 1
        
        for i in range(0, len(lines), chunk_size - self.overlap_lines):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            sections.append(CodeSection(
                section_type='LINES',
                name=f'{program_name}-CHUNK-{chunk_num}',
                content=chunk_content,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines))
            ))
            
            chunk_num += 1
        
        return sections

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content with safety buffer"""
        # Remove excessive whitespace for better estimation
        content = re.sub(r'\s+', ' ', content.strip())
        
        # More conservative token estimation
        estimated = len(content) / self.chars_per_token
        
        # Add 10% safety buffer
        return int(estimated * 1.1)

    def _validate_chunk_size(self, section: CodeSection) -> bool:
        """Validate that chunk is within token limits including prompt overhead"""
        content_tokens = self._estimate_tokens(section.content)
        total_tokens = content_tokens + self.prompt_overhead_tokens
        
        # Use much more conservative limit to prevent 4888 token chunks
        max_safe_tokens = 2800  # Very conservative limit
        
        if total_tokens > max_safe_tokens:
            self.logger.warning(f"Chunk {section.name} exceeds safe limit: {total_tokens} tokens (limit: {max_safe_tokens})")
            return False
        
        return True

    def _force_split_chunk(self, section: CodeSection, program_name: str) -> List[CodeSection]:
        """Force split a chunk that's too large into smaller pieces"""
        lines = section.content.split('\n')
        sections = []
        chunk_num = 1
        
        # Use very conservative content limit to prevent oversized chunks
        max_content_tokens = 1200  # Very small to ensure no 4888 token chunks
        
        i = 0
        while i < len(lines):
            chunk_lines = []
            current_tokens = 0
            
            # Build chunk line by line, checking tokens aggressively
            while i < len(lines):
                line = lines[i]
                line_tokens = self._estimate_tokens(line)
                
                # Check if adding this line would exceed very conservative content limit
                if current_tokens + line_tokens > max_content_tokens and chunk_lines:
                    break
                
                chunk_lines.append(line)
                current_tokens += line_tokens
                i += 1
                
                # Also check line count limit (much smaller)
                if len(chunk_lines) >= 80:  # Much smaller chunks
                    break
            
            if chunk_lines:
                chunk_content = '\n'.join(chunk_lines)
                sections.append(CodeSection(
                    section_type=section.section_type,
                    name=f"{section.name}-SPLIT-{chunk_num}",
                    content=chunk_content,
                    start_line=section.start_line + i - len(chunk_lines),
                    end_line=section.start_line + i - 1
                ))
                chunk_num += 1
                
                # Double-check that this chunk is actually safe
                if not self._validate_chunk_size(sections[-1]):
                    self.logger.error(f"Force split still created oversized chunk: {sections[-1].name}")
                    # Remove the oversized chunk and try smaller
                    sections.pop()
                    # Reduce chunk size even more and retry
                    i -= len(chunk_lines)
                    chunk_lines = chunk_lines[:len(chunk_lines)//2]  # Half the size
                    if chunk_lines:
                        chunk_content = '\n'.join(chunk_lines)
                        sections.append(CodeSection(
                            section_type=section.section_type,
                            name=f"{section.name}-EMERGENCY-{chunk_num}",
                            content=chunk_content,
                            start_line=section.start_line + i,
                            end_line=section.start_line + i + len(chunk_lines) - 1
                        ))
                        i += len(chunk_lines)
                    chunk_num += 1
        
        return sections

    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Enhanced file type detection"""
        content_upper = content.upper()
        
        if 'CREATE PROCEDURE' in content_upper:
            return 'db2_procedure'
        elif 'IDENTIFICATION DIVISION' in content_upper or 'PROGRAM-ID' in content_upper:
            if 'EXEC CICS' in content_upper:
                return 'cics'
            elif 'EXEC SQL' in content_upper:
                return 'cobol_sql'
            elif any(mq in content_upper for mq in ['MQOPEN', 'MQPUT', 'MQGET']):
                return 'cobol_mq'
            else:
                return 'cobol'
        elif 'PIC' in content_upper and 'IDENTIFICATION DIVISION' not in content_upper:
            return 'copybook'
        elif content.strip().startswith('//'):
            return 'jcl'
        elif '<' in content and '>' in content:
            return 'xml'
        
        # Extension-based fallback
        suffix_map = {
            '.cbl': 'cobol', '.cob': 'cobol', '.cpy': 'copybook', '.copy': 'copybook',
            '.jcl': 'jcl', '.sql': 'db2_procedure', '.db2': 'db2_procedure',
            '.xml': 'xml', '.xsd': 'xml'
        }
        
        return suffix_map.get(suffix.lower(), 'unknown')

    def _extract_program_name(self, content: str, file_path: Path) -> str:
        """Extract program name from content"""
        # COBOL PROGRAM-ID
        match = re.search(r'PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # JCL job name
        match = re.search(r'^//(\w+)\s+JOB\s', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # DB2 procedure name
        match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z][A-Z0-9_]*)', 
                         content, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            return name.split('.')[-1] if '.' in name else name
        
        # Fallback to filename
        return file_path.stem

    async def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read file with encoding detection"""
        encodings = ['utf-8', 'cp1252', 'latin1', 'ascii', 'cp037']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    return f.read()
            except Exception:
                continue
        
        return None
    
    """
COMPLETE LLM CodeParser Agent with Full Database Storage
Part 3: Main Processing Engine and LLM Analysis
"""

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Complete file processing with full database storage"""
        start_time = dt.now()
        
        try:
            self.logger.info(f"🧠 Processing file with complete LLM analysis: {file_path}")
            self.stats["files_processed"] += 1
            
            # Read and prepare content
            content = await self._read_file_safely(file_path)
            if not content:
                return {"status": "error", "error": "Could not read file"}
            
            file_type = self._detect_file_type(content, file_path.suffix)
            program_name = self._extract_program_name(content, file_path)
            
            self.logger.info(f"📋 Analyzing {file_type} program: {program_name}")
            
            # Intelligent chunking
            sections = self.intelligent_chunk_cobol(content, program_name)
            self.stats["chunks_created"] += len(sections)
            
            # Analyze each section
            all_analyses = []
            
            for i, section in enumerate(sections):
                self.logger.info(f"🔍 Analyzing section {i+1}/{len(sections)}: {section.name}")
                
                analysis = await self._analyze_section_with_llm(section, file_type, program_name)
                
                if analysis and "error" not in analysis:
                    all_analyses.append(analysis)
                else:
                    self.logger.warning(f"Section analysis failed: {section.name}")
            
            # Merge all analyses
            merged_analysis = self._merge_section_analyses(all_analyses, program_name)
            
            # Dependency analysis
            dependency_analysis = await self._analyze_dependencies(merged_analysis, program_name)
            
            # COMPLETE DATABASE STORAGE
            await self._store_complete_analysis_results(
                merged_analysis, dependency_analysis, program_name, file_path, file_type
            )
            
            processing_time = (dt.now() - start_time).total_seconds()
            self.stats["processing_time"] += processing_time
            
            return {
                "status": "success",
                "file_name": str(file_path.name),
                "program_name": program_name,
                "file_type": file_type,
                "sections_processed": len(all_analyses),
                "relationships_stored": {
                    "file_operations": len(merged_analysis.get("file_operations", [])),
                    "cics_operations": len(merged_analysis.get("cics_operations", [])),
                    "copybook_includes": len(merged_analysis.get("copybook_includes", [])),
                    "program_calls": len(merged_analysis.get("program_calls", [])),
                    "sql_operations": len(merged_analysis.get("sql_operations", [])),
                    "mq_operations": len(merged_analysis.get("mq_operations", [])),
                    "websphere_xml": len(merged_analysis.get("websphere_xml", [])),
                    "field_definitions": len(merged_analysis.get("field_definitions", [])),
                    "jcl_operations": len(merged_analysis.get("jcl_operations", [])),
                    "stored_procedures": len(merged_analysis.get("stored_procedures", []))
                },
                "total_relationships": merged_analysis.get("total_relationships", 0),
                "dependency_analysis": {
                    "required_copybooks": len(dependency_analysis.required_copybooks),
                    "called_programs": len(dependency_analysis.called_programs),
                    "accessed_files": len(dependency_analysis.accessed_files),
                    "missing_copybooks": list(dependency_analysis.missing_copybooks),
                    "missing_programs": list(dependency_analysis.missing_programs),
                    "confidence_score": dependency_analysis.confidence_score
                },
                "processing_time_seconds": processing_time,
                "database_storage_complete": True,
                "extraction_method": "llm_intelligent_chunking"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Complete processing failed for {file_path}: {e}")
            return {"status": "error", "file_name": str(file_path.name), "error": str(e)}

    async def _analyze_section_with_llm(self, section: CodeSection, file_type: str, 
                                      program_name: str) -> Optional[Dict[str, Any]]:
        """Analyze a single section with LLM"""
        
        # Critical: Validate chunk size before LLM call
        if not self._validate_chunk_size(section):
            self.logger.error(f"🚨 Skipping oversized chunk: {section.name}")
            return {"error": "Chunk too large for model", "section": section.name}
        
        # Check cache
        section_hash = hashlib.sha256(section.content.encode()).hexdigest()
        cached_result = await self._check_analysis_cache(section_hash)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Create prompt
        prompt = self.create_comprehensive_analysis_prompt(section, file_type, program_name)
        
        # Final token check with actual prompt
        prompt_tokens = self._estimate_tokens(prompt)
        if prompt_tokens > 2048:
            self.logger.error(f"🚨 Prompt too large: {prompt_tokens} tokens for {section.name}")
            return {"error": "Prompt exceeds model limit", "section": section.name}
        
        try:
            self.stats["llm_calls"] += 1
            self.logger.info(f"📤 Sending to LLM: {section.name} ({prompt_tokens} tokens)")
            
            response = await self.coordinator.call_model_api(
                prompt=prompt,
                params=self.base_llm_params
            )
            
            analysis = self._parse_comprehensive_response(response, section)
            
            # Cache successful analysis
            if analysis and "error" not in analysis:
                await self._cache_analysis_result(section_hash, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed for section {section.name}: {e}")
            return {"error": str(e), "section": section.name}

    async def _analyze_section_with_llm(self, section: CodeSection, file_type: str, 
                                      program_name: str) -> Optional[Dict[str, Any]]:
        """Analyze a single section with LLM using 4096 context window"""
        
        # Critical: Validate chunk size before LLM call
        if not self._validate_chunk_size(section):
            self.logger.error(f"🚨 Skipping oversized chunk: {section.name}")
            return {"error": "Chunk too large for model", "section": section.name}
        
        # Check cache
        section_hash = hashlib.sha256(section.content.encode()).hexdigest()
        cached_result = await self._check_analysis_cache(section_hash)
        if cached_result:
            self.stats["cache_hits"] += 1
            return cached_result
        
        # Create focused prompt based on content
        prompt = self.create_focused_analysis_prompt(section, file_type, program_name)
        
        # Final token check with actual prompt for 4096 context
        prompt_tokens = self._estimate_tokens(prompt)
        if prompt_tokens > 3800:  # Safe limit for 4096 context
            self.logger.error(f"🚨 Prompt too large: {prompt_tokens} tokens for {section.name}")
            return {"error": "Prompt exceeds model limit", "section": section.name}
        
        try:
            self.stats["llm_calls"] += 1
            self.logger.info(f"📤 Sending comprehensive prompt to LLM: {section.name} ({prompt_tokens} tokens)")
            
            response = await self.coordinator.call_model_api(
                prompt=prompt,
                params=self.base_llm_params
            )
            
            analysis = self._parse_focused_response(response, section)
            
            # Cache successful analysis
            if analysis and "error" not in analysis:
                await self._cache_analysis_result(section_hash, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ LLM analysis failed for section {section.name}: {e}")
            return {"error": str(e), "section": section.name}

    def _parse_focused_response(self, response: Dict[str, Any], 
                               section: CodeSection) -> Dict[str, Any]:
        """Parse focused LLM response with robust JSON cleaning"""
        try:
            text_content = self._extract_response_text(response)
            if not text_content:
                self.logger.warning(f"Empty response for {section.name}, using fallback")
                return self._fallback_pattern_extraction("", section)
            
            self.logger.debug(f"Raw response for {section.name}: {text_content[:200]}...")
            
            # STEP 1: Aggressive cleaning - remove everything before first {
            first_brace = text_content.find('{')
            if first_brace == -1:
                self.logger.warning(f"No JSON found in response for {section.name}, using fallback")
                return self._fallback_pattern_extraction(text_content, section)
            
            # STEP 2: Extract from first { to last }
            last_brace = text_content.rfind('}') + 1
            if last_brace <= first_brace:
                self.logger.warning(f"Malformed JSON brackets for {section.name}, using fallback")
                return self._fallback_pattern_extraction(text_content, section)
            
            json_str = text_content[first_brace:last_brace].strip()
            self.logger.debug(f"Extracted JSON for {section.name}: {json_str[:200]}...")
            
            # STEP 3: Comprehensive JSON cleaning
            cleaned_json = self._comprehensive_json_cleaning(json_str)
            
            try:
                analysis = json.loads(cleaned_json)
                self.logger.info(f"✅ Successfully parsed JSON for {section.name}")
                
                # Normalize the response
                normalized = self._normalize_focused_response(analysis, section)
                return normalized
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON decode failed for {section.name} at column {e.colno}: {str(e)}")
                self.logger.debug(f"Failed JSON: {cleaned_json[:500]}...")
                
                # STEP 4: Try multiple repair strategies
                repaired_json = self._multi_strategy_json_repair(cleaned_json, section)
                if repaired_json:
                    try:
                        analysis = json.loads(repaired_json)
                        self.logger.info(f"✅ Repaired and parsed JSON for {section.name}")
                        return self._normalize_focused_response(analysis, section)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Even repaired JSON failed for {section.name}")
                
                # STEP 5: Final fallback to pattern extraction
                self.logger.warning(f"Using pattern fallback for {section.name}")
                return self._fallback_pattern_extraction("", section)
            
        except Exception as e:
            self.logger.error(f"Response parsing error for {section.name}: {e}")
            return self._fallback_pattern_extraction("", section)

    def _comprehensive_json_cleaning(self, json_str: str) -> str:
        """Comprehensive JSON cleaning to fix common LLM JSON issues"""
        
        # Remove any text before the first { and after the last }
        json_str = json_str.strip()
        
        # Remove common LLM prefixes/suffixes
        prefixes_to_remove = [
            "Here is the JSON:",
            "Here's the JSON:",
            "The JSON is:",
            "JSON:",
            "```json",
            "```",
            "Response:",
            "Analysis:",
        ]
        
        for prefix in prefixes_to_remove:
            if json_str.startswith(prefix):
                json_str = json_str[len(prefix):].strip()
        
        # Remove common suffixes
        suffixes_to_remove = [
            "```",
            "End of JSON",
            "That's the analysis",
            "This completes",
        ]
        
        for suffix in suffixes_to_remove:
            if suffix in json_str:
                json_str = json_str.split(suffix)[0].strip()
        
        # Clean up common JSON formatting issues
        
        # 1. Fix trailing commas before } or ]
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 2. Fix missing commas between array elements
        json_str = re.sub(r'}\s*{', r'},{', json_str)
        
        # 3. Fix missing commas between object properties
        json_str = re.sub(r'"\s*"([^"]*)":', r'","\1":', json_str)
        
        # 4. Fix unquoted keys
        json_str = re.sub(r'(\w+)(\s*):', r'"\1"\2:', json_str)
        
        # 5. Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # 6. Fix escaped quotes issues
        json_str = json_str.replace('\\"', '"')
        
        # 7. Remove any trailing text after the final }
        last_brace = json_str.rfind('}')
        if last_brace != -1:
            json_str = json_str[:last_brace + 1]
        
        # 8. Fix common field name issues
        json_str = re.sub(r'"type"(\s*)([^,}\]]+)', r'"type"\1"\2"', json_str)
        json_str = re.sub(r'"target"(\s*)([^,}\]]+)', r'"target"\1"\2"', json_str)
        
        return json_str

    def _multi_strategy_json_repair(self, json_str: str, section: CodeSection) -> Optional[str]:
        """Try multiple strategies to repair malformed JSON"""
        
        strategies = [
            self._repair_strategy_1_brackets,
            self._repair_strategy_2_quotes,
            self._repair_strategy_3_minimal,
            self._repair_strategy_4_reconstruct,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                repaired = strategy(json_str)
                if repaired:
                    # Test if it's valid JSON
                    json.loads(repaired)
                    self.logger.info(f"✅ JSON repair strategy {i+1} succeeded for {section.name}")
                    return repaired
            except (json.JSONDecodeError, Exception) as e:
                self.logger.debug(f"Repair strategy {i+1} failed for {section.name}: {e}")
                continue
        
        return None

    def _repair_strategy_1_brackets(self, json_str: str) -> str:
        """Strategy 1: Fix bracket matching issues"""
        
        # Count brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Add missing closing brackets
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            json_str = '{' * (close_braces - open_braces) + json_str
        
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            json_str = '[' * (close_brackets - open_brackets) + json_str
        
        return json_str

    def _repair_strategy_2_quotes(self, json_str: str) -> str:
        """Strategy 2: Fix quote matching issues"""
        
        # Fix unmatched quotes
        quote_count = json_str.count('"')
        if quote_count % 2 != 0:
            json_str += '"'
        
        # Fix common quote escaping issues
        json_str = re.sub(r'([^\\])"([^",:}\]]*)"([^",:}\]]*)"', r'\1"\2\3"', json_str)
        
        return json_str

    def _repair_strategy_3_minimal(self, json_str: str) -> str:
        """Strategy 3: Create minimal valid JSON structure"""
        
        # If all else fails, create a minimal structure
        if not json_str.strip().startswith('{'):
            json_str = '{' + json_str
        
        if not json_str.strip().endswith('}'):
            json_str = json_str + '}'
        
        # Ensure it has at least empty arrays for required fields
        minimal_structure = {
            "file_operations": [],
            "cics_operations": [],
            "copybook_includes": [],
            "program_calls": [],
            "sql_operations": [],
            "field_definitions": []
        }
        
        try:
            # Try to merge with minimal structure
            parsed = json.loads(json_str)
            for key in minimal_structure:
                if key not in parsed:
                    parsed[key] = []
            return json.dumps(parsed)
        except:
            return json.dumps(minimal_structure)

    def _repair_strategy_4_reconstruct(self, json_str: str) -> str:
        """Strategy 4: Reconstruct JSON from partial data"""
        
        # Extract any recognizable patterns and reconstruct
        result = {
            "file_operations": [],
            "cics_operations": [],
            "copybook_includes": [],
            "program_calls": [],
            "sql_operations": [],
            "field_definitions": []
        }
        
        # Look for CICS patterns
        cics_matches = re.findall(r'"type":\s*"CICS_[^"]*"[^}]*"target":\s*"([^"]*)"', json_str)
        for match in cics_matches:
            result["cics_operations"].append({
                "type": "CICS_READ",
                "target": match,
                "line_number": 1,
                "confidence": 0.7
            })
        
        # Look for COPY patterns
        copy_matches = re.findall(r'"type":\s*"COPY"[^}]*"target":\s*"([^"]*)"', json_str)
        for match in copy_matches:
            result["copybook_includes"].append({
                "type": "COPY",
                "target": match,
                "line_number": 1,
                "confidence": 0.7
            })
        
        # Look for field patterns
        field_matches = re.findall(r'"target":\s*"([^"]*)"[^}]*"picture_clause":\s*"([^"]*)"', json_str)
        for field_name, pic_clause in field_matches:
            result["field_definitions"].append({
                "type": "FIELD_DEFINITION",
                "target": field_name,
                "line_number": 1,
                "picture_clause": pic_clause,
                "confidence": 0.7
            })
        
        return json.dumps(result)

    def _normalize_focused_response(self, analysis: Dict[str, Any], 
                                   section: CodeSection) -> Dict[str, Any]:
        """Normalize focused response to standard format"""
        
        # Initialize standard format
        normalized = {
            "file_operations": [],
            "cics_operations": [],
            "copybook_includes": [],
            "program_calls": [],
            "sql_operations": [],
            "mq_operations": [],
            "websphere_xml": [],
            "field_definitions": [],
            "jcl_operations": [],
            "stored_procedures": [],
            "section_info": {
                "section_type": section.section_type,
                "section_name": section.name,
                "start_line": section.start_line,
                "end_line": section.end_line,
                "focused_analysis": True
            }
        }
        
        # Copy over any found relationships
        for key in normalized.keys():
            if key in analysis and isinstance(analysis[key], list):
                normalized[key] = analysis[key]
        
        # Handle generic "relationships" key from minimal prompt
        if "relationships" in analysis:
            for rel in analysis["relationships"]:
                rel_type = rel.get("type", "").lower()
                
                if "cics" in rel_type:
                    normalized["cics_operations"].append(rel)
                elif "copy" in rel_type:
                    normalized["copybook_includes"].append(rel)
                elif "call" in rel_type:
                    normalized["program_calls"].append(rel)
                elif "sql" in rel_type:
                    normalized["sql_operations"].append(rel)
                elif "field" in rel_type or "pic" in rel_type:
                    normalized["field_definitions"].append(rel)
                elif "file" in rel_type or "read" in rel_type or "write" in rel_type:
                    normalized["file_operations"].append(rel)
                elif "mq" in rel_type:
                    normalized["mq_operations"].append(rel)
        
        # Validate line numbers are relative to section start
        for category in normalized:
            if isinstance(normalized[category], list):
                for item in normalized[category]:
                    if isinstance(item, dict) and "line_number" in item:
                        # Ensure line number is absolute, not relative to section
                        if item["line_number"] < section.start_line:
                            item["line_number"] += section.start_line
        
        return normalized

    def _fallback_pattern_extraction(self, text: str, section: CodeSection) -> Dict[str, Any]:
        """Enhanced fallback extraction for when JSON parsing fails"""
        
        analysis = {
            "file_operations": [], "cics_operations": [], "copybook_includes": [],
            "program_calls": [], "sql_operations": [], "field_definitions": [],
            "mq_operations": [], "websphere_xml": [], "jcl_operations": [],
            "stored_procedures": [],
            "section_info": {
                "section_type": section.section_type, 
                "section_name": section.name, 
                "fallback_parsing": True
            }
        }
        
        # Use the original code content for pattern extraction, not the LLM response
        code_lines = section.content.split('\n')
        
        for i, line in enumerate(code_lines):
            line_clean = line.strip().upper()
            actual_line_num = section.start_line + i
            
            # Extract CICS operations
            if 'EXEC CICS' in line_clean:
                if 'DATASET(' in line_clean or 'FILE(' in line_clean:
                    # Extract dataset/file name
                    for pattern in [r"DATASET\s*\(\s*['\"]([^'\"]+)['\"]", r"FILE\s*\(\s*['\"]([^'\"]+)['\"]"]:
                        match = re.search(pattern, line_clean)
                        if match:
                            op_type = 'CICS_READ' if 'READ' in line_clean else 'CICS_WRITE'
                            analysis["cics_operations"].append({
                                "type": op_type,
                                "target": match.group(1),
                                "line_number": actual_line_num,
                                "statement": line.strip(),
                                "confidence": 0.7
                            })
                            break
                
                elif 'LINK' in line_clean or 'XCTL' in line_clean:
                    # Extract program name
                    match = re.search(r"PROGRAM\s*\(\s*['\"]([^'\"]+)['\"]", line_clean)
                    if match:
                        op_type = 'CICS_LINK' if 'LINK' in line_clean else 'CICS_XCTL'
                        analysis["cics_operations"].append({
                            "type": op_type,
                            "target": match.group(1),
                            "line_number": actual_line_num,
                            "statement": line.strip(),
                            "confidence": 0.8
                        })
            
            # Extract COPY statements
            elif line_clean.startswith('COPY ') or ' COPY ' in line_clean:
                match = re.search(r'COPY\s+([A-Z][A-Z0-9-]*)', line_clean)
                if match:
                    analysis["copybook_includes"].append({
                        "type": "COPY",
                        "target": match.group(1),
                        "line_number": actual_line_num,
                        "statement": line.strip(),
                        "location": "UNKNOWN",
                        "confidence": 0.9
                    })
            
            # Extract CALL statements
            elif 'CALL ' in line_clean and not line_clean.strip().startswith('*'):
                match = re.search(r"CALL\s+['\"]([^'\"]+)['\"]", line_clean)
                if match:
                    analysis["program_calls"].append({
                        "type": "CALL",
                        "target": match.group(1),
                        "line_number": actual_line_num,
                        "statement": line.strip(),
                        "confidence": 0.8
                    })
            
            # Extract SQL operations
            elif 'EXEC SQL' in line_clean:
                sql_type = 'SQL_SELECT'
                if 'SELECT' in line_clean:
                    sql_type = 'SQL_SELECT'
                elif 'INSERT' in line_clean:
                    sql_type = 'SQL_INSERT'
                elif 'UPDATE' in line_clean:
                    sql_type = 'SQL_UPDATE'
                elif 'DELETE' in line_clean:
                    sql_type = 'SQL_DELETE'
                
                # Try to extract table name
                table_match = re.search(r'FROM\s+([A-Z][A-Z0-9_]*)', line_clean)
                target = table_match.group(1) if table_match else "UNKNOWN_TABLE"
                
                analysis["sql_operations"].append({
                    "type": sql_type,
                    "target": target,
                    "line_number": actual_line_num,
                    "statement": line.strip(),
                    "confidence": 0.8
                })
            
            # Extract field definitions
            elif re.match(r'^\s*\d+\s+\w+.*PIC\s+', line_clean):
                match = re.search(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+.*PIC\s+([X9V()]+)', line_clean)
                if match:
                    analysis["field_definitions"].append({
                        "type": "FIELD_DEFINITION",
                        "target": match.group(2),
                        "line_number": actual_line_num,
                        "statement": line.strip(),
                        "level_number": int(match.group(1)),
                        "picture_clause": match.group(3),
                        "confidence": 0.9
                    })
        
        return analysis

    def _repair_and_parse_json(self, json_str: str, section: CodeSection) -> Dict[str, Any]:
        """Attempt to repair malformed JSON"""
        try:
            # Common JSON repairs
            repaired = json_str
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)  # Remove trailing commas
            repaired = re.sub(r'(\w+):', r'"\1":', repaired)    # Quote unquoted keys
            repaired = repaired.replace("'", '"')               # Fix single quotes
            
            analysis = json.loads(repaired)
            analysis['section_info'] = {
                'section_type': section.section_type,
                'section_name': section.name,
                'repaired_json': True
            }
            return analysis
            
        except Exception:
            # Return minimal structure if repair fails
            return {
                "file_operations": [], "cics_operations": [], "copybook_includes": [],
                "program_calls": [], "sql_operations": [], "mq_operations": [],
                "websphere_xml": [], "field_definitions": [], "jcl_operations": [],
                "stored_procedures": [],
                "section_info": {"section_type": section.section_type, "section_name": section.name, "parsing_failed": True}
            }

    def _merge_section_analyses(self, all_analyses: List[Dict[str, Any]], 
                              program_name: str) -> Dict[str, Any]:
        """Merge analyses from all sections with deduplication"""
        
        merged = {
            "program_name": program_name,
            "file_operations": [], "cics_operations": [], "copybook_includes": [],
            "program_calls": [], "sql_operations": [], "mq_operations": [],
            "websphere_xml": [], "field_definitions": [], "jcl_operations": [],
            "stored_procedures": [],
            "sections_analyzed": len(all_analyses),
            "total_relationships": 0
        }
        
        # Collect all relationships
        for analysis in all_analyses:
            for category in merged.keys():
                if category in analysis and isinstance(analysis[category], list):
                    merged[category].extend(analysis[category])
        
        # Deduplicate each category
        for category in merged:
            if isinstance(merged[category], list):
                merged[category] = self._deduplicate_relationships(merged[category])
        
        # Calculate totals
        relationship_counts = sum(
            len(merged[cat]) for cat in merged 
            if isinstance(merged[cat], list) and cat not in ['sections_analyzed']
        )
        merged["total_relationships"] = relationship_counts
        self.stats["total_relationships"] += relationship_counts
        
        return merged

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships"""
        seen = {}
        
        for rel in relationships:
            key = f"{rel.get('type', '')}:{rel.get('target', '')}:{rel.get('line_number', 0)}"
            
            if (key not in seen or 
                rel.get('confidence', 0) > seen[key].get('confidence', 0)):
                seen[key] = rel
        
        return list(seen.values())

    def _fallback_comprehensive_extraction(self, text: str, section: CodeSection) -> Dict[str, Any]:
        """This method is no longer used - replaced by _fallback_pattern_extraction"""
        return self._fallback_pattern_extraction(text, section)

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract text from LLM response and detect prompt echo-back"""
        # Handle different response formats from your model server
        text_content = ""
        
        if isinstance(response, dict):
            # Try common response keys
            for key in ['text', 'response', 'content', 'generated_text', 'output']:
                if key in response and response[key]:
                    text_content = str(response[key]).strip()
                    break
            
            # Handle nested response structures
            if not text_content and 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if isinstance(choice, dict):
                    for key in ['text', 'message', 'content']:
                        if key in choice and choice[key]:
                            if isinstance(choice[key], dict) and 'content' in choice[key]:
                                text_content = str(choice[key]['content']).strip()
                            else:
                                text_content = str(choice[key]).strip()
                            break
            
            # If response is a simple dict with direct content
            if not text_content and len(response) == 1:
                text_content = str(list(response.values())[0]).strip()
        
        # Fallback to string conversion
        if not text_content:
            text_content = str(response).strip() if response else ""
        
        # CRITICAL: Detect if LLM is echoing back our prompt
        prompt_indicators = [
            "Find SQL operations in this COBOL code",
            "Find CICS operations in this COBOL code", 
            "Find COPY statements in this COBOL code",
            "Find CALL statements in this COBOL code",
            "Find field definitions in this COBOL code",
            "Find file operations in this COBOL code",
            "Find MQ operations in this COBOL code",
            "Find ALL patterns in this COBOL code",
            "Required JSON format:",
            "Return only JSON, no explanations",
            "Extract ALL SQL operations and any related patterns"
        ]
        
        is_prompt_echo = any(indicator in text_content for indicator in prompt_indicators)
        
        if is_prompt_echo:
            self.logger.warning(f"🚨 DETECTED PROMPT ECHO-BACK! LLM returned prompt instead of analysis")
            self.logger.debug(f"Echo response preview: {text_content[:200]}...")
            
            # Try to extract any JSON that might be after the prompt echo
            json_start = text_content.find('{')
            if json_start > 200:  # If JSON is far into the response, might be actual response after echo
                potential_json = text_content[json_start:]
                self.logger.info(f"🔍 Attempting to extract JSON from echo response...")
                return potential_json
            else:
                # This is mostly just echo, return empty to trigger fallback
                return ""
        
        return text_content

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract text from LLM response"""
        return (
            response.get('text') or 
            response.get('response') or 
            response.get('content') or
            str(response.get('choices', [{}])[0].get('text', ''))
        )

    async def _check_analysis_cache(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Check analysis cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT analysis_result FROM llm_analysis_cache 
                WHERE content_hash = ? AND analysis_type = 'complete_llm_analysis'
            """, (content_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return None
            
        except Exception:
            return None

    async def _cache_analysis_result(self, content_hash: str, analysis: Dict[str, Any]):
        """Cache analysis result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO llm_analysis_cache 
                (content_hash, analysis_type, analysis_result, confidence_score, model_version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content_hash, 'complete_llm_analysis',
                json.dumps(analysis), 0.9, 'codellama'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to cache analysis: {e}")

    """
COMPLETE LLM CodeParser Agent with Full Database Storage
Part 4: Complete Database Storage Engine - All Relationship Types
"""

    async def _store_complete_analysis_results(self, merged_analysis: Dict[str, Any], 
                                             dependency_analysis: DependencyAnalysis,
                                             program_name: str, file_path: Path, file_type: str):
        """COMPLETE database storage - all relationship types in proper tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. Store file operations in file_access_relationships table
            for file_op in merged_analysis.get('file_operations', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO file_access_relationships 
                    (program_name, logical_file_name, physical_file_name, access_type, 
                     access_mode, line_number, access_statement, validation_status, 
                     confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, file_op.get('target', ''), '', file_op.get('type', ''),
                    file_op.get('access_mode', ''), file_op.get('line_number', 0),
                    file_op.get('statement', ''), 'llm_validated', 
                    file_op.get('confidence', 0.8), 'llm'
                ))
            
            # 2. Store CICS operations (split between file operations and program calls)
            for cics_op in merged_analysis.get('cics_operations', []):
                if cics_op.get('type') in ['CICS_LINK', 'CICS_XCTL']:
                    # Store as program relationship
                    cursor.execute("""
                        INSERT OR IGNORE INTO program_relationships 
                        (calling_program, called_program, call_type, line_number, call_statement, 
                         confidence_score, extraction_method)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        program_name, cics_op.get('target', ''), cics_op.get('type', ''),
                        cics_op.get('line_number', 0), cics_op.get('statement', ''),
                        cics_op.get('confidence', 0.8), 'llm'
                    ))
                else:
                    # Store as file operation (CICS READ/WRITE DATASET/FILE)
                    cursor.execute("""
                        INSERT OR IGNORE INTO file_access_relationships 
                        (program_name, logical_file_name, access_type, line_number, 
                         access_statement, validation_status, confidence_score, extraction_method)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        program_name, cics_op.get('target', ''), cics_op.get('type', ''),
                        cics_op.get('line_number', 0), cics_op.get('statement', ''),
                        'llm_cics_validated', cics_op.get('confidence', 0.8), 'llm'
                    ))
            
            # 3. Store copybook relationships in copybook_relationships table
            for copybook in merged_analysis.get('copybook_includes', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO copybook_relationships 
                    (program_name, copybook_name, copy_location, line_number, copy_statement, 
                     replacing_clause, usage_context, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, copybook.get('target', ''), 
                    copybook.get('location', 'UNKNOWN'), copybook.get('line_number', 0),
                    copybook.get('statement', ''), copybook.get('replacing_clause', ''),
                    'llm_extracted', copybook.get('confidence', 0.8), 'llm'
                ))
            
            # 4. Store program calls in program_relationships table
            for prog_call in merged_analysis.get('program_calls', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO program_relationships 
                    (calling_program, called_program, call_type, line_number, call_statement, 
                     parameters, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, prog_call.get('target', ''), prog_call.get('type', ''),
                    prog_call.get('line_number', 0), prog_call.get('statement', ''),
                    prog_call.get('parameters', ''), prog_call.get('confidence', 0.8), 'llm'
                ))
            
            # 5. Store SQL operations in sql_analysis table
            for sql_op in merged_analysis.get('sql_operations', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO sql_analysis 
                    (program_name, sql_type, tables_accessed, operation_type, line_number, 
                     sql_statement, database_objects, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, 'embedded_sql', sql_op.get('target', ''),
                    sql_op.get('type', ''), sql_op.get('line_number', 0),
                    sql_op.get('statement', ''), 
                    json.dumps(sql_op.get('database_objects', [])),
                    sql_op.get('confidence', 0.8), 'llm'
                ))
            
            # 6. Store MQ operations in mq_operations table
            for mq_op in merged_analysis.get('mq_operations', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO mq_operations 
                    (program_name, operation_type, queue_name, queue_manager, line_number, 
                     operation_statement, operation_details, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, mq_op.get('type', ''), mq_op.get('target', ''),
                    mq_op.get('queue_manager', ''), mq_op.get('line_number', 0),
                    mq_op.get('statement', ''), mq_op.get('queue_details', ''),
                    mq_op.get('confidence', 0.8), 'llm'
                ))
            
            # 7. Store WebSphere/XML operations in websphere_xml_operations table
            for ws_op in merged_analysis.get('websphere_xml', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO websphere_xml_operations 
                    (program_name, operation_type, service_name, endpoint_url, line_number, 
                     operation_statement, service_details, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, ws_op.get('type', ''), ws_op.get('target', ''),
                    '', ws_op.get('line_number', 0), ws_op.get('statement', ''),
                    ws_op.get('service_details', ''), ws_op.get('confidence', 0.8), 'llm'
                ))
            
            # 8. Store field definitions in both field_definitions and field_cross_reference tables
            for field_def in merged_analysis.get('field_definitions', []):
                # field_definitions table
                cursor.execute("""
                    INSERT OR IGNORE INTO field_definitions 
                    (field_name, source_name, source_type, level_number, data_type, 
                     picture_clause, value_clause, line_number, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    field_def.get('target', ''), program_name, file_type,
                    field_def.get('level_number', 0), field_def.get('data_type', 'UNKNOWN'),
                    field_def.get('picture_clause', ''), field_def.get('value_clause', ''),
                    field_def.get('line_number', 0), field_def.get('confidence', 0.8), 'llm'
                ))
                
                # field_cross_reference table (for lineage)
                cursor.execute("""
                    INSERT OR IGNORE INTO field_cross_reference 
                    (field_name, qualified_name, source_name, source_type, definition_location,
                     data_type, picture_clause, usage_clause, level_number, business_domain,
                     confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    field_def.get('target', ''), f"{program_name}.{field_def.get('target', '')}",
                    program_name, file_type, 'WORKING_STORAGE',
                    field_def.get('data_type', 'UNKNOWN'), 
                    field_def.get('picture_clause', ''), field_def.get('value_clause', ''),
                    field_def.get('level_number', 0), field_def.get('business_domain', 'GENERAL'),
                    field_def.get('confidence', 0.8), 'llm'
                ))
            
            # 9. Store JCL operations in jcl_operations table
            for jcl_op in merged_analysis.get('jcl_operations', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO jcl_operations 
                    (job_name, step_name, operation_type, program_name, dataset_name, 
                     line_number, operation_statement, step_details, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, 'STEP01', jcl_op.get('type', ''), jcl_op.get('target', ''),
                    '', jcl_op.get('line_number', 0), jcl_op.get('statement', ''),
                    jcl_op.get('step_details', ''), jcl_op.get('confidence', 0.8), 'llm'
                ))
            
            # 10. Store stored procedures (also in sql_analysis table with special type)
            for proc_op in merged_analysis.get('stored_procedures', []):
                cursor.execute("""
                    INSERT OR IGNORE INTO sql_analysis 
                    (program_name, sql_type, tables_accessed, operation_type, line_number, 
                     sql_statement, database_objects, confidence_score, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name, 'stored_procedure', proc_op.get('target', ''),
                    proc_op.get('type', ''), proc_op.get('line_number', 0),
                    proc_op.get('statement', ''), proc_op.get('parameters', ''),
                    proc_op.get('confidence', 0.8), 'llm'
                ))
            
            # 11. Store dependency analysis in dependency_analysis table
            cursor.execute("""
                INSERT OR REPLACE INTO dependency_analysis 
                (program_name, required_copybooks, called_programs, accessed_files,
                 missing_copybooks, missing_programs, confidence_score, extraction_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_name,
                json.dumps(list(dependency_analysis.required_copybooks)),
                json.dumps(list(dependency_analysis.called_programs)),
                json.dumps(list(dependency_analysis.accessed_files)),
                json.dumps(list(dependency_analysis.missing_copybooks)),
                json.dumps(list(dependency_analysis.missing_programs)),
                dependency_analysis.confidence_score,
                'llm'
            ))
            
            # 12. Store file metadata in file_metadata table
            field_names = [field.get('target', '') for field in merged_analysis.get('field_definitions', [])]
            cursor.execute("""
                INSERT OR REPLACE INTO file_metadata 
                (file_name, file_type, table_name, fields, source_type, processing_status,
                 extraction_method, confidence_score, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path.name, file_type, program_name, json.dumps(field_names),
                file_type, 'processed', 'llm', 0.9, dt.now().isoformat()
            ))
            
            # 13. Store program chunks (for completeness)
            file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()
            cursor.execute("""
                INSERT OR REPLACE INTO program_chunks 
                (program_name, chunk_id, chunk_type, content, metadata, 
                 file_hash, confidence_score, line_start, line_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_name, f"{program_name}_MAIN", f"{file_type}_main",
                "LLM analyzed content", json.dumps({
                    "extraction_method": "llm",
                    "sections_analyzed": merged_analysis.get("sections_analyzed", 0),
                    "total_relationships": merged_analysis.get("total_relationships", 0)
                }),
                file_hash, 0.9, 1, 100
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ COMPLETE database storage for {program_name} - all tables populated")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store complete analysis results: {e}")
            raise

    async def _analyze_dependencies(self, merged_analysis: Dict[str, Any], 
                                  program_name: str) -> DependencyAnalysis:
        """Analyze program dependencies and check for missing components"""
        
        self.stats["dependency_checks"] += 1
        
        # Extract required dependencies
        required_copybooks = set()
        called_programs = set()
        accessed_files = set()
        
        # From copybook includes
        for copybook in merged_analysis.get('copybook_includes', []):
            if copybook.get('target'):
                required_copybooks.add(copybook['target'])
        
        # From program calls
        for call in merged_analysis.get('program_calls', []):
            if call.get('target'):
                called_programs.add(call['target'])
        
        # From CICS operations
        for cics_op in merged_analysis.get('cics_operations', []):
            if cics_op.get('type') in ['CICS_LINK', 'CICS_XCTL'] and cics_op.get('target'):
                called_programs.add(cics_op['target'])
            elif cics_op.get('target'):
                accessed_files.add(cics_op['target'])
        
        # From file operations
        for file_op in merged_analysis.get('file_operations', []):
            if file_op.get('target'):
                accessed_files.add(file_op['target'])
        
        # Check what's missing in database
        missing_copybooks = await self._check_missing_copybooks(required_copybooks)
        missing_programs = await self._check_missing_programs(called_programs)
        
        self.stats["missing_dependencies"] += len(missing_copybooks) + len(missing_programs)
        
        # Calculate confidence
        total_dependencies = len(required_copybooks) + len(called_programs)
        missing_count = len(missing_copybooks) + len(missing_programs)
        confidence_score = 1.0 - (missing_count / max(1, total_dependencies))
        
        return DependencyAnalysis(
            program_name=program_name,
            required_copybooks=required_copybooks,
            called_programs=called_programs,
            accessed_files=accessed_files,
            missing_copybooks=missing_copybooks,
            missing_programs=missing_programs,
            confidence_score=confidence_score
        )

    async def _check_missing_copybooks(self, required_copybooks: Set[str]) -> Set[str]:
        """Check which copybooks are missing from database"""
        if not required_copybooks:
            return set()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in required_copybooks])
            
            # Check copybook_relationships table
            cursor.execute(f"""
                SELECT DISTINCT copybook_name 
                FROM copybook_relationships 
                WHERE copybook_name IN ({placeholders})
            """, list(required_copybooks))
            existing_copybooks = {row[0] for row in cursor.fetchall()}
            
            # Check program_chunks table (copybooks can be programs)
            cursor.execute(f"""
                SELECT DISTINCT program_name 
                FROM program_chunks 
                WHERE program_name IN ({placeholders})
            """, list(required_copybooks))
            existing_as_programs = {row[0] for row in cursor.fetchall()}
            
            # Check file_metadata table
            cursor.execute(f"""
                SELECT DISTINCT file_name 
                FROM file_metadata 
                WHERE file_name IN ({placeholders}) AND file_type = 'copybook'
            """, list(required_copybooks))
            existing_in_metadata = {row[0] for row in cursor.fetchall()}
            
            conn.close()
            
            all_existing = existing_copybooks | existing_as_programs | existing_in_metadata
            missing = required_copybooks - all_existing
            
            return missing
            
        except Exception as e:
            self.logger.error(f"Failed to check missing copybooks: {e}")
            return required_copybooks

    async def _check_missing_programs(self, called_programs: Set[str]) -> Set[str]:
        """Check which called programs are missing from database"""
        if not called_programs:
            return set()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in called_programs])
            
            # Check program_chunks table
            cursor.execute(f"""
                SELECT DISTINCT program_name 
                FROM program_chunks 
                WHERE program_name IN ({placeholders})
            """, list(called_programs))
            existing_programs = {row[0] for row in cursor.fetchall()}
            
            # Check program_relationships as targets
            cursor.execute(f"""
                SELECT DISTINCT called_program 
                FROM program_relationships 
                WHERE called_program IN ({placeholders})
            """, list(called_programs))
            existing_as_targets = {row[0] for row in cursor.fetchall()}
            
            conn.close()
            
            all_existing = existing_programs | existing_as_targets
            missing = called_programs - all_existing
            
            return missing
            
        except Exception as e:
            self.logger.error(f"Failed to check missing programs: {e}")
            return called_programs
    
    """
COMPLETE LLM CodeParser Agent with Full Database Storage
Part 5: Reporting, Statistics, and Utility Methods
"""

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            **self.stats,
            "cache_hit_rate": (self.stats["cache_hits"] / max(1, self.stats["llm_calls"])) * 100,
            "avg_relationships_per_file": self.stats["total_relationships"] / max(1, self.stats["files_processed"]),
            "avg_chunks_per_file": self.stats["chunks_created"] / max(1, self.stats["files_processed"]),
            "avg_processing_time": self.stats["processing_time"] / max(1, self.stats["files_processed"]),
            "dependency_completeness": 1.0 - (self.stats["missing_dependencies"] / max(1, self.stats["dependency_checks"]))
        }

    async def get_missing_dependencies_report(self, program_name: str) -> Dict[str, Any]:
        """Get missing dependencies report for a specific program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT required_copybooks, called_programs, accessed_files,
                       missing_copybooks, missing_programs, confidence_score
                FROM dependency_analysis 
                WHERE program_name = ?
            """, (program_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "program": program_name,
                    "required_copybooks": json.loads(result[0]) if result[0] else [],
                    "called_programs": json.loads(result[1]) if result[1] else [],
                    "accessed_files": json.loads(result[2]) if result[2] else [],
                    "missing_copybooks": json.loads(result[3]) if result[3] else [],
                    "missing_programs": json.loads(result[4]) if result[4] else [],
                    "confidence_score": result[5] or 0.0
                }
            else:
                return {"error": "No dependency analysis found", "program": program_name}
                
        except Exception as e:
            return {"error": str(e), "program": program_name}

    async def get_database_summary(self) -> Dict[str, Any]:
        """Get summary of what's stored in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            summary = {}
            
            # Count records in each table
            tables = [
                'program_chunks', 'program_relationships', 'file_access_relationships',
                'copybook_relationships', 'field_definitions', 'sql_analysis',
                'mq_operations', 'websphere_xml_operations', 'jcl_operations',
                'field_cross_reference', 'dependency_analysis', 'file_metadata'
            ]
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    summary[table] = count
                except Exception:
                    summary[table] = 0
            
            # Get extraction method breakdown
            cursor.execute("""
                SELECT extraction_method, COUNT(*) 
                FROM file_access_relationships 
                GROUP BY extraction_method
            """)
            summary["file_access_by_method"] = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT extraction_method, COUNT(*) 
                FROM copybook_relationships 
                GROUP BY extraction_method
            """)
            summary["copybook_by_method"] = dict(cursor.fetchall())
            
            # Get programs with missing dependencies
            cursor.execute("""
                SELECT program_name, missing_copybooks, missing_programs 
                FROM dependency_analysis 
                WHERE missing_copybooks != '[]' OR missing_programs != '[]'
            """)
            
            missing_deps = []
            for row in cursor.fetchall():
                prog_name, missing_copybooks, missing_programs = row
                missing_cb = json.loads(missing_copybooks) if missing_copybooks else []
                missing_prog = json.loads(missing_programs) if missing_programs else []
                
                if missing_cb or missing_prog:
                    missing_deps.append({
                        "program": prog_name,
                        "missing_copybooks": missing_cb,
                        "missing_programs": missing_prog
                    })
            
            summary["programs_with_missing_dependencies"] = missing_deps
            summary["total_programs_with_missing_deps"] = len(missing_deps)
            
            conn.close()
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}

    def verify_database_storage(self, program_name: str) -> Dict[str, Any]:
        """Verify that all relationship types are properly stored for a program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            verification = {"program": program_name, "storage_verification": {}}
            
            # Check each table for the program
            
            # 1. File access relationships
            cursor.execute("SELECT COUNT(*) FROM file_access_relationships WHERE program_name = ?", (program_name,))
            verification["storage_verification"]["file_access_relationships"] = cursor.fetchone()[0]
            
            # 2. Copybook relationships
            cursor.execute("SELECT COUNT(*) FROM copybook_relationships WHERE program_name = ?", (program_name,))
            verification["storage_verification"]["copybook_relationships"] = cursor.fetchone()[0]
            
            # 3. Program relationships (as caller)
            cursor.execute("SELECT COUNT(*) FROM program_relationships WHERE calling_program = ?", (program_name,))
            verification["storage_verification"]["program_relationships"] = cursor.fetchone()[0]
            
            # 4. SQL analysis
            cursor.execute("SELECT COUNT(*) FROM sql_analysis WHERE program_name = ?", (program_name,))
            verification["storage_verification"]["sql_analysis"] = cursor.fetchone()[0]
            
            # 5. MQ operations
            cursor.execute("SELECT COUNT(*) FROM mq_operations WHERE program_name = ?", (program_name,))
            verification["storage_verification"]["mq_operations"] = cursor.fetchone()[0]
            
            # 6. Field definitions
            cursor.execute("SELECT COUNT(*) FROM field_definitions WHERE source_name = ?", (program_name,))
            verification["storage_verification"]["field_definitions"] = cursor.fetchone()[0]
            
            # 7. Field cross reference
            cursor.execute("SELECT COUNT(*) FROM field_cross_reference WHERE source_name = ?", (program_name,))
            verification["storage_verification"]["field_cross_reference"] = cursor.fetchone()[0]
            
            # 8. Dependency analysis
            cursor.execute("SELECT COUNT(*) FROM dependency_analysis WHERE program_name = ?", (program_name,))
            verification["storage_verification"]["dependency_analysis"] = cursor.fetchone()[0]
            
            # 9. File metadata
            cursor.execute("SELECT COUNT(*) FROM file_metadata WHERE table_name = ?", (program_name,))
            verification["storage_verification"]["file_metadata"] = cursor.fetchone()[0]
            
            # 10. Program chunks
            cursor.execute("SELECT COUNT(*) FROM program_chunks WHERE program_name = ?", (program_name,))
            verification["storage_verification"]["program_chunks"] = cursor.fetchone()[0]
            
            # Calculate total relationships stored
            total_relationships = sum([
                verification["storage_verification"]["file_access_relationships"],
                verification["storage_verification"]["copybook_relationships"],
                verification["storage_verification"]["program_relationships"],
                verification["storage_verification"]["sql_analysis"],
                verification["storage_verification"]["mq_operations"],
                verification["storage_verification"]["field_definitions"]
            ])
            
            verification["total_relationships_stored"] = total_relationships
            verification["storage_complete"] = total_relationships > 0
            
            conn.close()
            
            return verification
            
        except Exception as e:
            return {"error": str(e), "program": program_name}

    async def get_program_analysis_report(self, program_name: str) -> Dict[str, Any]:
        """Get comprehensive analysis report for a specific program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            report = {"program": program_name, "analysis": {}}
            
            # Get file operations
            cursor.execute("""
                SELECT logical_file_name, access_type, line_number, access_statement, confidence_score
                FROM file_access_relationships WHERE program_name = ?
                ORDER BY line_number
            """, (program_name,))
            report["analysis"]["file_operations"] = [
                {
                    "file": row[0], "access_type": row[1], "line": row[2],
                    "statement": row[3], "confidence": row[4]
                } for row in cursor.fetchall()
            ]
            
            # Get copybook includes
            cursor.execute("""
                SELECT copybook_name, copy_location, line_number, copy_statement, confidence_score
                FROM copybook_relationships WHERE program_name = ?
                ORDER BY line_number
            """, (program_name,))
            report["analysis"]["copybook_includes"] = [
                {
                    "copybook": row[0], "location": row[1], "line": row[2],
                    "statement": row[3], "confidence": row[4]
                } for row in cursor.fetchall()
            ]
            
            # Get program calls
            cursor.execute("""
                SELECT called_program, call_type, line_number, call_statement, confidence_score
                FROM program_relationships WHERE calling_program = ?
                ORDER BY line_number
            """, (program_name,))
            report["analysis"]["program_calls"] = [
                {
                    "called_program": row[0], "call_type": row[1], "line": row[2],
                    "statement": row[3], "confidence": row[4]
                } for row in cursor.fetchall()
            ]
            
            # Get SQL operations
            cursor.execute("""
                SELECT tables_accessed, operation_type, line_number, sql_statement, confidence_score
                FROM sql_analysis WHERE program_name = ?
                ORDER BY line_number
            """, (program_name,))
            report["analysis"]["sql_operations"] = [
                {
                    "tables": row[0], "operation": row[1], "line": row[2],
                    "statement": row[3], "confidence": row[4]
                } for row in cursor.fetchall()
            ]
            
            # Get field definitions
            cursor.execute("""
                SELECT field_name, level_number, data_type, picture_clause, line_number, confidence_score
                FROM field_definitions WHERE source_name = ?
                ORDER BY line_number
            """, (program_name,))
            report["analysis"]["field_definitions"] = [
                {
                    "field": row[0], "level": row[1], "type": row[2],
                    "picture": row[3], "line": row[4], "confidence": row[5]
                } for row in cursor.fetchall()
            ]
            
            # Get dependency analysis
            cursor.execute("""
                SELECT required_copybooks, called_programs, missing_copybooks, missing_programs, confidence_score
                FROM dependency_analysis WHERE program_name = ?
            """, (program_name,))
            dep_result = cursor.fetchone()
            if dep_result:
                report["analysis"]["dependencies"] = {
                    "required_copybooks": json.loads(dep_result[0]) if dep_result[0] else [],
                    "called_programs": json.loads(dep_result[1]) if dep_result[1] else [],
                    "missing_copybooks": json.loads(dep_result[2]) if dep_result[2] else [],
                    "missing_programs": json.loads(dep_result[3]) if dep_result[3] else [],
                    "confidence_score": dep_result[4] or 0.0
                }
            
            conn.close()
            
            # Calculate summary stats
            report["summary"] = {
                "total_file_operations": len(report["analysis"]["file_operations"]),
                "total_copybook_includes": len(report["analysis"]["copybook_includes"]),
                "total_program_calls": len(report["analysis"]["program_calls"]),
                "total_sql_operations": len(report["analysis"]["sql_operations"]),
                "total_field_definitions": len(report["analysis"]["field_definitions"]),
                "has_missing_dependencies": (
                    len(report["analysis"].get("dependencies", {}).get("missing_copybooks", [])) > 0 or
                    len(report["analysis"].get("dependencies", {}).get("missing_programs", [])) > 0
                )
            }
            
            return report
            
        except Exception as e:
            return {"error": str(e), "program": program_name}

    async def export_analysis_results(self, output_format: str = "json") -> Dict[str, Any]:
        """Export complete analysis results in various formats"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if output_format.lower() == "json":
                # Export as structured JSON
                export_data = {
                    "export_timestamp": dt.now().isoformat(),
                    "database_summary": await self.get_database_summary(),
                    "processing_statistics": self.get_comprehensive_stats(),
                    "programs": {}
                }
                
                # Get all programs
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT program_name FROM program_chunks")
                programs = [row[0] for row in cursor.fetchall()]
                
                # Get analysis for each program
                for program in programs:
                    program_analysis = await self.get_program_analysis_report(program)
                    export_data["programs"][program] = program_analysis
                
                conn.close()
                return {"status": "success", "format": "json", "data": export_data}
                
            elif output_format.lower() == "csv":
                # Export key relationships as CSV format
                csv_data = {
                    "file_access_relationships": [],
                    "copybook_relationships": [],
                    "program_relationships": []
                }
                
                cursor = conn.cursor()
                
                # File access relationships
                cursor.execute("""
                    SELECT program_name, logical_file_name, access_type, line_number, confidence_score
                    FROM file_access_relationships
                    ORDER BY program_name, line_number
                """)
                csv_data["file_access_relationships"] = [
                    {"program": row[0], "file": row[1], "access": row[2], "line": row[3], "confidence": row[4]}
                    for row in cursor.fetchall()
                ]
                
                # Copybook relationships
                cursor.execute("""
                    SELECT program_name, copybook_name, copy_location, line_number, confidence_score
                    FROM copybook_relationships
                    ORDER BY program_name, line_number
                """)
                csv_data["copybook_relationships"] = [
                    {"program": row[0], "copybook": row[1], "location": row[2], "line": row[3], "confidence": row[4]}
                    for row in cursor.fetchall()
                ]
                
                # Program relationships
                cursor.execute("""
                    SELECT calling_program, called_program, call_type, line_number, confidence_score
                    FROM program_relationships
                    ORDER BY calling_program, line_number
                """)
                csv_data["program_relationships"] = [
                    {"caller": row[0], "called": row[1], "type": row[2], "line": row[3], "confidence": row[4]}
                    for row in cursor.fetchall()
                ]
                
                conn.close()
                return {"status": "success", "format": "csv", "data": csv_data}
                
            else:
                return {"status": "error", "error": f"Unsupported format: {output_format}"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Export the main class
__all__ = ['CodeParserAgent']