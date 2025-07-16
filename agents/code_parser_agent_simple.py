"""
Simplified Enhanced Code Parser Agent - Part 1: Core Parsing with Reliable Patterns
Focus on correct basic parsing first, then add LLM enhancement
"""
import tiktoken
import re
import asyncio
import sqlite3
import json
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
from datetime import datetime as dt
import logging
from enum import Enum
from contextlib import asynccontextmanager
import copy

# Import the base agent
from agents.base_agent_api import BaseOpulenceAgent, SamplingParams

# Simple data classes
@dataclass
class CodeChunk:
    """Simplified code chunk representation"""
    program_name: str
    chunk_id: str
    chunk_type: str
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int
    confidence_score: float = 1.0

@dataclass
class RelationshipRecord:
    """Generic relationship record"""
    source_name: str
    target_name: str
    relationship_type: str
    location: str
    line_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class CodeParserAgent(BaseOpulenceAgent):
    """
    Simplified Code Parser Agent - Reliable patterns first, LLM enhancement second
    """
    
    def __init__(self, coordinator, agent_type: str = "simplified_code_parser", 
                 db_path: str = "opulence_data.db", gpu_id: int = 0, **kwargs):
        super().__init__(coordinator, agent_type, db_path, gpu_id)
        
        # Agent-specific configuration for CodeLlama
        self.api_params.update({
            "max_tokens": 1500,
            "temperature": 0.1,
            "top_p": 0.9
        })
        
        # Context window management for CodeLlama
        self.max_context_tokens = 2048
        self.reserve_tokens = 300  # Reserve for prompt structure
        self.max_content_tokens = self.max_context_tokens - self.reserve_tokens
        
        # Initialize simplified patterns
        self._init_core_patterns()
        
        # Initialize database
        self._init_simplified_database()
        
        self.logger.info("🚀 Simplified Code Parser Agent initialized with CodeLlama integration")

    def _init_core_patterns(self):
        """Initialize simplified, reliable patterns"""
        
        # COBOL Core Patterns - Simplified and Reliable
        self.cobol_patterns = {
            # Basic identification
            'program_id': re.compile(r'PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            
            # File definitions - CRITICAL for relationships
            'select_assign': re.compile(r'SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([A-Z0-9-]+)', re.IGNORECASE),
            'fd_definition': re.compile(r'FD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # File operations - Simplified patterns
            'file_open': re.compile(r'OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'file_read': re.compile(r'READ\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'file_write': re.compile(r'WRITE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'file_close': re.compile(r'CLOSE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Program calls
            'cobol_call': re.compile(r'CALL\s+[\'"]([A-Z0-9][A-Z0-9-]*)[\'"]', re.IGNORECASE),
            
            # Copy statements
            'copy_statement': re.compile(r'COPY\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # SQL blocks
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            
            # Data items - Basic pattern
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)', re.MULTILINE | re.IGNORECASE),
        }
        
        # CICS Patterns - Core commands only
        self.cics_patterns = {
            'cics_link': re.compile(r'EXEC\s+CICS\s+LINK\s+PROGRAM\s*\(\s*([A-Z][A-Z0-9-]*)\s*\)', re.IGNORECASE),
            'cics_xctl': re.compile(r'EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*([A-Z][A-Z0-9-]*)\s*\)', re.IGNORECASE),
            'cics_read': re.compile(r'EXEC\s+CICS\s+READ\s+FILE\s*\(\s*([A-Z][A-Z0-9-]*)\s*\)', re.IGNORECASE),
            'cics_write': re.compile(r'EXEC\s+CICS\s+WRITE\s+FILE\s*\(\s*([A-Z][A-Z0-9-]*)\s*\)', re.IGNORECASE),
        }
        
        # JCL Patterns - Essential only
        self.jcl_patterns = {
            'job_card': re.compile(r'^//(\w+)\s+JOB\s', re.MULTILINE),
            'exec_pgm': re.compile(r'^//(\w+)\s+EXEC\s+PGM=([A-Z0-9]+)', re.MULTILINE | re.IGNORECASE),
            'exec_proc': re.compile(r'^//(\w+)\s+EXEC\s+PROC=([A-Z0-9]+)', re.MULTILINE | re.IGNORECASE),
            'exec_simple': re.compile(r'^//(\w+)\s+EXEC\s+([A-Z0-9]+)', re.MULTILINE | re.IGNORECASE),
            'dd_statement': re.compile(r'^//(\w+)\s+DD\s+DSN=([^,\s]+)', re.MULTILINE | re.IGNORECASE),
        }
        
        # MQ Patterns - API calls only
        self.mq_patterns = {
            'mq_call': re.compile(r'CALL\s+[\'"]MQ([A-Z]+)[\'"]', re.IGNORECASE),
            'mq_queue': re.compile(r'[\'"]([A-Z][A-Z0-9\._]*\.Q)[\'"]', re.IGNORECASE),
        }
        
        # DB2 Patterns - Basic SQL
        self.db2_patterns = {
            'create_procedure': re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'sql_table': re.compile(r'FROM\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'sql_insert': re.compile(r'INSERT\s+INTO\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'sql_update': re.compile(r'UPDATE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
        }

    def _init_simplified_database(self):
        """Initialize simplified database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Core chunks table
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
            
            # Program relationships - Fixed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS program_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calling_program TEXT NOT NULL,
                    called_program TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    call_location TEXT,
                    line_number INTEGER,
                    call_statement TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(calling_program, called_program, call_type, line_number)
                )
            """)
            
            # File relationships - Fixed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_access_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    logical_file_name TEXT NOT NULL,
                    physical_file_name TEXT,
                    access_type TEXT NOT NULL,
                    access_mode TEXT,
                    line_number INTEGER,
                    access_statement TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Copybook relationships - Fixed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS copybook_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    copybook_name TEXT NOT NULL,
                    copy_location TEXT,
                    line_number INTEGER,
                    copy_statement TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, copybook_name, copy_location)
                )
            """)
            
            # Field definitions - Simplified
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_definitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    level_number INTEGER,
                    data_type TEXT,
                    picture_clause TEXT,
                    line_number INTEGER,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # SQL analysis - Basic
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sql_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    sql_type TEXT NOT NULL,
                    tables_accessed TEXT,
                    operation_type TEXT,
                    line_number INTEGER,
                    sql_statement TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # LLM analysis cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    analysis_type TEXT NOT NULL,
                    analysis_result TEXT,
                    confidence_score REAL,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prog_chunks_name ON program_chunks(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prog_rel_calling ON program_relationships(calling_program)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_rel_program ON file_access_relationships(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_copy_rel_program ON copybook_relationships(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_def_source ON field_definitions(source_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sql_program ON sql_analysis(program_name)")
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Simplified database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Main file processing method - simplified approach"""
        try:
            self.logger.info(f"🔍 Processing file: {file_path}")
            
            # Read file content
            content = await self._read_file_safely(file_path)
            if not content:
                return {"status": "error", "error": "Could not read file"}
            
            # Detect file type
            file_type = self._detect_file_type_simple(content, file_path.suffix)
            program_name = self._extract_program_name_simple(content, file_path)
            
            self.logger.info(f"📋 File type: {file_type}, Program: {program_name}")
            
            # Extract relationships FIRST (reliable patterns)
            relationships = await self._extract_all_relationships(content, program_name, file_type)
            
            # Store relationships
            await self._store_all_relationships(relationships)
            
            # Create basic chunks
            chunks = await self._create_basic_chunks(content, program_name, file_type)
            
            # Enhanced analysis with LLM (if content is suitable)
            if len(content) > 100:  # Only for substantial content
                enhanced_analysis = await self._enhanced_llm_analysis(content, file_type, program_name)
                # Merge enhanced analysis into chunks
                chunks = self._merge_enhanced_analysis(chunks, enhanced_analysis)
            
            # Store chunks
            await self._store_chunks(chunks, file_path)
            
            return {
                "status": "success",
                "file_name": str(file_path.name),
                "file_type": file_type,
                "program_name": program_name,
                "chunks_created": len(chunks),
                "relationships_found": len(relationships),
                "processing_timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Processing failed for {file_path}: {str(e)}")
            return {
                "status": "error",
                "file_name": str(file_path.name),
                "error": str(e)
            }

    async def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read file with multiple encoding attempts"""
        encodings = ['utf-8', 'cp1252', 'latin1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    return f.read()
            except Exception:
                continue
        
        return None

    def _detect_file_type_simple(self, content: str, suffix: str) -> str:
        """Simplified file type detection"""
        content_upper = content.upper()
        
        # Check content patterns first
        if 'CREATE PROCEDURE' in content_upper or 'CREATE OR REPLACE PROCEDURE' in content_upper:
            return 'db2_procedure'
        elif 'DFHMSD' in content_upper or 'DFHMDI' in content_upper:
            return 'bms'
        elif content.strip().startswith('//') and ('JOB' in content_upper or 'EXEC' in content_upper):
            return 'jcl'
        elif 'IDENTIFICATION DIVISION' in content_upper or 'PROGRAM-ID' in content_upper:
            if 'EXEC CICS' in content_upper:
                return 'cics'
            elif 'EXEC SQL' in content_upper and 'CALL' in content_upper and 'MQ' in content_upper:
                return 'cobol_stored_procedure'
            elif 'CALL' in content_upper and any(mq in content_upper for mq in ['MQOPEN', 'MQPUT', 'MQGET']):
                return 'mq_program'
            else:
                return 'cobol'
        elif 'PIC' in content_upper and not 'IDENTIFICATION DIVISION' in content_upper:
            return 'copybook'
        
        # Fallback to extension
        suffix_map = {
            '.cbl': 'cobol', '.cob': 'cobol',
            '.jcl': 'jcl',
            '.cpy': 'copybook', '.copy': 'copybook',
            '.bms': 'bms',
            '.sql': 'db2_procedure', '.db2': 'db2_procedure'
        }
        
        return suffix_map.get(suffix.lower(), 'unknown')

    def _extract_program_name_simple(self, content: str, file_path: Path) -> str:
        """Extract program name reliably"""
        # Try COBOL PROGRAM-ID
        if match := self.cobol_patterns['program_id'].search(content):
            return match.group(1).strip()
        
        # Try JCL job name
        if match := self.jcl_patterns['job_card'].search(content):
            return match.group(1).strip()
        
        # Try DB2 procedure
        if match := self.db2_patterns['create_procedure'].search(content):
            name = match.group(1).strip()
            return name.split('.')[-1] if '.' in name else name
        
        # Use filename
        return file_path.stem

    async def _extract_all_relationships(self, content: str, program_name: str, file_type: str) -> List[RelationshipRecord]:
        """Extract all relationships using reliable patterns"""
        relationships = []
        
        # Program calls
        relationships.extend(self._extract_program_calls(content, program_name))
        
        # File relationships
        relationships.extend(self._extract_file_relationships(content, program_name))
        
        # Copybook relationships
        relationships.extend(self._extract_copybook_relationships(content, program_name))
        
        # SQL relationships
        relationships.extend(self._extract_sql_relationships(content, program_name))
        
        self.logger.info(f"📊 Extracted {len(relationships)} relationships")
        return relationships

    def _extract_program_calls(self, content: str, program_name: str) -> List[RelationshipRecord]:
        """Extract program call relationships"""
        relationships = []
        
        # COBOL CALL statements
        for match in self.cobol_patterns['cobol_call'].finditer(content):
            called_program = match.group(1).strip()
            line_num = content[:match.start()].count('\n') + 1
            
            relationships.append(RelationshipRecord(
                source_name=program_name,
                target_name=called_program,
                relationship_type='COBOL_CALL',
                location=self._find_paragraph(content, match.start()),
                line_number=line_num,
                metadata={'statement': match.group(0)}
            ))
        
        # CICS LINK/XCTL
        for pattern_name, pattern in self.cics_patterns.items():
            if 'link' in pattern_name or 'xctl' in pattern_name:
                for match in pattern.finditer(content):
                    called_program = match.group(1).strip()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    relationships.append(RelationshipRecord(
                        source_name=program_name,
                        target_name=called_program,
                        relationship_type=pattern_name.upper(),
                        location=self._find_paragraph(content, match.start()),
                        line_number=line_num,
                        metadata={'statement': match.group(0)}
                    ))
        
        # JCL EXEC statements
        for pattern_name, pattern in self.jcl_patterns.items():
            if 'exec' in pattern_name:
                for match in pattern.finditer(content):
                    step_name = match.group(1).strip()
                    program_name_exec = match.group(2).strip()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    relationships.append(RelationshipRecord(
                        source_name=program_name,
                        target_name=program_name_exec,
                        relationship_type='JCL_EXEC',
                        location=step_name,
                        line_number=line_num,
                        metadata={'statement': match.group(0)}
                    ))
        
        return relationships

    def _extract_file_relationships(self, content: str, program_name: str) -> List[RelationshipRecord]:
        """Extract file access relationships - FIXED"""
        relationships = []
        
        # File assignments (SELECT ... ASSIGN TO ...)
        for match in self.cobol_patterns['select_assign'].finditer(content):
            logical_file = match.group(1).strip()
            physical_file = match.group(2).strip()
            line_num = content[:match.start()].count('\n') + 1
            
            relationships.append(RelationshipRecord(
                source_name=program_name,
                target_name=logical_file,
                relationship_type='FILE_SELECT',
                location='FILE-CONTROL',
                line_number=line_num,
                metadata={
                    'logical_file': logical_file,
                    'physical_file': physical_file,
                    'statement': match.group(0)
                }
            ))
        
        # File operations
        file_ops = {
            'file_open': 'OPEN',
            'file_read': 'READ', 
            'file_write': 'WRITE',
            'file_close': 'CLOSE'
        }
        
        for pattern_name, op_type in file_ops.items():
            pattern = self.cobol_patterns[pattern_name]
            for match in pattern.finditer(content):
                if pattern_name == 'file_open':
                    file_mode = match.group(1).strip()
                    file_name = match.group(2).strip()
                else:
                    file_name = match.group(1).strip()
                    file_mode = ''
                
                line_num = content[:match.start()].count('\n') + 1
                
                relationships.append(RelationshipRecord(
                    source_name=program_name,
                    target_name=file_name,
                    relationship_type=f'FILE_{op_type}',
                    location=self._find_paragraph(content, match.start()),
                    line_number=line_num,
                    metadata={
                        'access_mode': file_mode,
                        'statement': match.group(0)
                    }
                ))
        
        # CICS file operations
        for pattern_name, pattern in self.cics_patterns.items():
            if 'read' in pattern_name or 'write' in pattern_name:
                for match in pattern.finditer(content):
                    file_name = match.group(1).strip()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    relationships.append(RelationshipRecord(
                        source_name=program_name,
                        target_name=file_name,
                        relationship_type=f'CICS_{pattern_name.split("_")[1].upper()}',
                        location=self._find_paragraph(content, match.start()),
                        line_number=line_num,
                        metadata={'statement': match.group(0)}
                    ))
        
        return relationships

    def _extract_copybook_relationships(self, content: str, program_name: str) -> List[RelationshipRecord]:
        """Extract copybook relationships"""
        relationships = []
        
        for match in self.cobol_patterns['copy_statement'].finditer(content):
            copybook_name = match.group(1).strip()
            line_num = content[:match.start()].count('\n') + 1
            
            relationships.append(RelationshipRecord(
                source_name=program_name,
                target_name=copybook_name,
                relationship_type='COPY',
                location=self._find_section(content, match.start()),
                line_number=line_num,
                metadata={'statement': match.group(0)}
            ))
        
        return relationships

    def _extract_sql_relationships(self, content: str, program_name: str) -> List[RelationshipRecord]:
        """Extract SQL table relationships"""
        relationships = []
        
        # Extract tables from SQL blocks
        for sql_match in self.cobol_patterns['sql_block'].finditer(content):
            sql_content = sql_match.group(1)
            line_num = content[:sql_match.start()].count('\n') + 1
            
            # Find tables in this SQL block
            for pattern_name, pattern in self.db2_patterns.items():
                for table_match in pattern.finditer(sql_content):
                    table_name = table_match.group(1).strip()
                    
                    relationships.append(RelationshipRecord(
                        source_name=program_name,
                        target_name=table_name,
                        relationship_type=f'SQL_{pattern_name.upper()}',
                        location=self._find_paragraph(content, sql_match.start()),
                        line_number=line_num,
                        metadata={
                            'sql_statement': sql_content[:100],
                            'operation': pattern_name
                        }
                    ))
        
        return relationships

    def _find_paragraph(self, content: str, position: int) -> str:
        """Find containing paragraph"""
        before_content = content[:position]
        lines = before_content.split('\n')
        
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('*') and line.endswith('.'):
                # Simple paragraph detection
                words = line.split()
                if len(words) >= 1 and words[0].replace('-', '').isalnum():
                    return words[0]
        
        return 'UNKNOWN'

    def _find_section(self, content: str, position: int) -> str:
        """Find containing section"""
        before_content = content[:position].upper()
        
        sections = [
            'WORKING-STORAGE SECTION',
            'FILE SECTION', 
            'LINKAGE SECTION',
            'PROCEDURE DIVISION'
        ]
        
        current_section = 'UNKNOWN'
        for section in sections:
            if section in before_content:
                current_section = section.replace(' SECTION', '').replace(' ', '-')
        
        return current_section

    async def _store_all_relationships(self, relationships: List[RelationshipRecord]):
        """Store all relationships in appropriate tables"""
        if not relationships:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rel in relationships:
                if rel.relationship_type in ['COBOL_CALL', 'CICS_LINK', 'CICS_XCTL', 'JCL_EXEC']:
                    # Program relationships
                    cursor.execute("""
                        INSERT OR IGNORE INTO program_relationships 
                        (calling_program, called_program, call_type, call_location, line_number, call_statement)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        rel.source_name, rel.target_name, rel.relationship_type,
                        rel.location, rel.line_number, rel.metadata.get('statement', '')
                    ))
                
                elif rel.relationship_type.startswith('FILE_') or rel.relationship_type.startswith('CICS_READ') or rel.relationship_type.startswith('CICS_WRITE'):
                    # File relationships
                    cursor.execute("""
                        INSERT OR IGNORE INTO file_access_relationships 
                        (program_name, logical_file_name, physical_file_name, access_type, access_mode, line_number, access_statement)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rel.source_name, rel.target_name, 
                        rel.metadata.get('physical_file', ''),
                        rel.relationship_type,
                        rel.metadata.get('access_mode', ''),
                        rel.line_number, rel.metadata.get('statement', '')
                    ))
                
                elif rel.relationship_type == 'COPY':
                    # Copybook relationships
                    cursor.execute("""
                        INSERT OR IGNORE INTO copybook_relationships 
                        (program_name, copybook_name, copy_location, line_number, copy_statement)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        rel.source_name, rel.target_name, rel.location,
                        rel.line_number, rel.metadata.get('statement', '')
                    ))
                
                elif rel.relationship_type.startswith('SQL_'):
                    # SQL relationships
                    cursor.execute("""
                        INSERT OR IGNORE INTO sql_analysis 
                        (program_name, sql_type, tables_accessed, operation_type, line_number, sql_statement)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        rel.source_name, 'embedded_sql', rel.target_name,
                        rel.relationship_type, rel.line_number,
                        rel.metadata.get('sql_statement', '')
                    ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Stored {len(relationships)} relationships")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store relationships: {str(e)}")

    async def _create_basic_chunks(self, content: str, program_name: str, file_type: str) -> List[CodeChunk]:
        """Create basic chunks for the content"""
        chunks = []
        
        # Create file-level chunk
        chunks.append(CodeChunk(
            program_name=program_name,
            chunk_id=f"{program_name}_MAIN",
            chunk_type=f"{file_type}_main",
            content=content[:min(1000, len(content))],  # First 1000 chars
            metadata={
                'file_type': file_type,
                'total_lines': content.count('\n') + 1,
                'content_size': len(content)
            },
            line_start=1,
            line_end=min(50, content.count('\n') + 1)
        ))
        
        # Create division/section chunks for COBOL
        if file_type in ['cobol', 'cics', 'cobol_stored_procedure']:
            chunks.extend(self._create_cobol_chunks(content, program_name))
        
        # Create JCL step chunks
        elif file_type == 'jcl':
            chunks.extend(self._create_jcl_chunks(content, program_name))
        
        # Create copybook chunks
        elif file_type == 'copybook':
            chunks.extend(self._create_copybook_chunks(content, program_name))
        
        return chunks

    def _create_cobol_chunks(self, content: str, program_name: str) -> List[CodeChunk]:
        """Create COBOL-specific chunks"""
        chunks = []
        
        # Find divisions
        divisions = [
            ('IDENTIFICATION DIVISION', 'identification'),
            ('ENVIRONMENT DIVISION', 'environment'),
            ('DATA DIVISION', 'data'),
            ('PROCEDURE DIVISION', 'procedure')
        ]
        
        for div_pattern, div_name in divisions:
            if div_pattern in content.upper():
                start_pos = content.upper().find(div_pattern)
                # Find next division or end
                end_pos = len(content)
                for next_div, _ in divisions:
                    next_pos = content.upper().find(next_div, start_pos + 1)
                    if next_pos != -1 and next_pos < end_pos:
                        end_pos = next_pos
                
                division_content = content[start_pos:end_pos]
                if len(division_content.strip()) > 10:
                    chunks.append(CodeChunk(
                        program_name=program_name,
                        chunk_id=f"{program_name}_{div_name.upper()}_DIV",
                        chunk_type=f"cobol_{div_name}_division",
                        content=division_content,
                        metadata={
                            'division_name': div_name,
                            'division_size': len(division_content.split('\n'))
                        },
                        line_start=content[:start_pos].count('\n') + 1,
                        line_end=content[:end_pos].count('\n') + 1
                    ))
        
        return chunks

    def _create_jcl_chunks(self, content: str, program_name: str) -> List[CodeChunk]:
        """Create JCL-specific chunks"""
        chunks = []
        
        # Find job steps
        for match in self.jcl_patterns['exec_pgm'].finditer(content):
            step_name = match.group(1)
            pgm_name = match.group(2)
            
            # Get step content (from this EXEC to next // or end)
            start_pos = match.start()
            next_step = content.find('\n//', start_pos + 1)
            end_pos = next_step if next_step != -1 else len(content)
            
            step_content = content[start_pos:end_pos]
            
            chunks.append(CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_STEP_{step_name}",
                chunk_type="jcl_step",
                content=step_content,
                metadata={
                    'step_name': step_name,
                    'program_executed': pgm_name
                },
                line_start=content[:start_pos].count('\n') + 1,
                line_end=content[:end_pos].count('\n') + 1
            ))
        
        return chunks

    def _create_copybook_chunks(self, content: str, program_name: str) -> List[CodeChunk]:
        """Create copybook-specific chunks"""
        chunks = []
        
        # Find 01 level items (record definitions)
        record_pattern = re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*)', re.MULTILINE | re.IGNORECASE)
        matches = list(record_pattern.finditer(content))
        
        for i, match in enumerate(matches):
            record_name = match.group(1)
            start_pos = match.start()
            
            # Find end (next 01 level or end of content)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            record_content = content[start_pos:end_pos]
            
            chunks.append(CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_RECORD_{record_name}",
                chunk_type="copybook_record",
                content=record_content,
                metadata={
                    'record_name': record_name,
                    'field_count': record_content.count('PIC')
                },
                line_start=content[:start_pos].count('\n') + 1,
                line_end=content[:end_pos].count('\n') + 1
            ))
        
        return chunks

    async def _enhanced_llm_analysis(self, content: str, file_type: str, program_name: str) -> Dict[str, Any]:
        """Enhanced analysis using LLM with chunking for CodeLlama 2048 context"""
        
        # Estimate token count (rough estimation: 1 token ≈ 4 characters)
        estimated_tokens = len(content) // 4
        
        if estimated_tokens <= self.max_content_tokens:
            # Content fits in one request
            return await self._single_llm_analysis(content, file_type, program_name)
        else:
            # Need to chunk the content
            return await self._chunked_llm_analysis(content, file_type, program_name)

    async def _single_llm_analysis(self, content: str, file_type: str, program_name: str) -> Dict[str, Any]:
        """Single LLM analysis call"""
        
        # Check cache first
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        cached_result = await self._check_llm_cache(content_hash, file_type)
        if cached_result:
            return cached_result
        
        # Build analysis prompt based on file type
        prompt = self._build_analysis_prompt(content, file_type, program_name)
        
        try:
            # Call LLM via coordinator
            response = await self.call_api(prompt, {
                "temperature": 0.1,
                "max_tokens": 1000
            })
            
            # Parse response
            analysis = self._parse_llm_response(response, file_type)
            
            # Cache result
            await self._cache_llm_result(content_hash, file_type, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ LLM analysis failed: {str(e)}")
            return {"error": str(e), "fallback": True}

    async def _chunked_llm_analysis(self, content: str, file_type: str, program_name: str) -> Dict[str, Any]:
        """Chunked LLM analysis for large content"""
        
        # Split content into chunks that fit in context window
        chunks = self._split_content_intelligently(content, file_type)
        
        self.logger.info(f"🔄 Analyzing {len(chunks)} chunks for {program_name}")
        
        chunk_analyses = []
        
        for i, chunk in enumerate(chunks):
            try:
                self.logger.info(f"📊 Analyzing chunk {i+1}/{len(chunks)}")
                
                # Analyze this chunk
                chunk_analysis = await self._single_llm_analysis(chunk, file_type, f"{program_name}_CHUNK_{i+1}")
                
                if chunk_analysis and not chunk_analysis.get('error'):
                    chunk_analyses.append(chunk_analysis)
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Chunk {i+1} analysis failed: {str(e)}")
                continue
        
        # Aggregate chunk analyses
        if chunk_analyses:
            return self._aggregate_chunk_analyses(chunk_analyses, file_type)
        else:
            return {"error": "All chunk analyses failed", "fallback": True}

    def _split_content_intelligently(self, content: str, file_type: str) -> List[str]:
        """Split content into intelligent chunks based on file type"""
        
        max_chunk_chars = self.max_content_tokens * 4  # Rough token-to-char conversion
        
        if file_type in ['cobol', 'cics', 'cobol_stored_procedure']:
            return self._split_cobol_content(content, max_chunk_chars)
        elif file_type == 'jcl':
            return self._split_jcl_content(content, max_chunk_chars)
        elif file_type == 'copybook':
            return self._split_copybook_content(content, max_chunk_chars)
        else:
            return self._split_generic_content(content, max_chunk_chars)

    def _split_cobol_content(self, content: str, max_chars: int) -> List[str]:
        """Split COBOL content at logical boundaries"""
        chunks = []
        
        # Try to split by divisions first
        divisions = []
        for div_pattern, div_name in [
            ('IDENTIFICATION DIVISION', 'identification'),
            ('ENVIRONMENT DIVISION', 'environment'), 
            ('DATA DIVISION', 'data'),
            ('PROCEDURE DIVISION', 'procedure')
        ]:
            pos = content.upper().find(div_pattern)
            if pos != -1:
                divisions.append((pos, div_pattern, div_name))
        
        divisions.sort()  # Sort by position
        
        if divisions:
            for i, (start_pos, div_pattern, div_name) in enumerate(divisions):
                # Find end of this division
                if i + 1 < len(divisions):
                    end_pos = divisions[i + 1][0]
                else:
                    end_pos = len(content)
                
                division_content = content[start_pos:end_pos]
                
                # If division is too large, split it further
                if len(division_content) > max_chars:
                    sub_chunks = self._split_by_paragraphs(division_content, max_chars)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(division_content)
        else:
            # Fallback to paragraph splitting
            chunks = self._split_by_paragraphs(content, max_chars)
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def _split_by_paragraphs(self, content: str, max_chars: int) -> List[str]:
        """Split content by paragraphs"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_chars and current_chunk:
                # Start new chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _split_jcl_content(self, content: str, max_chars: int) -> List[str]:
        """Split JCL content by job steps"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            
            # Start new chunk on new job step (//stepname EXEC)
            if (line.strip().startswith('//') and ' EXEC ' in line.upper() and 
                current_chunk and current_size > 0):
                
                if current_size > max_chars:
                    # Current chunk is too big, finalize it
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
            else:
                if current_size + line_size > max_chars and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _split_copybook_content(self, content: str, max_chars: int) -> List[str]:
        """Split copybook content by record definitions"""
        chunks = []
        
        # Split by 01 level items
        record_pattern = re.compile(r'^\s*01\s+', re.MULTILINE | re.IGNORECASE)
        splits = [m.start() for m in record_pattern.finditer(content)]
        
        if splits:
            for i, start_pos in enumerate(splits):
                end_pos = splits[i + 1] if i + 1 < len(splits) else len(content)
                record_content = content[start_pos:end_pos]
                
                if len(record_content) > max_chars:
                    # Further split large records
                    sub_chunks = self._split_by_paragraphs(record_content, max_chars)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(record_content)
        else:
            chunks = self._split_generic_content(content, max_chars)
        
        return chunks

    def _split_generic_content(self, content: str, max_chars: int) -> List[str]:
        """Generic content splitting"""
        chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            end_pos = min(current_pos + max_chars, len(content))
            
            # Try to break at line boundary
            if end_pos < len(content):
                last_newline = content.rfind('\n', current_pos, end_pos)
                if last_newline != -1 and last_newline > current_pos:
                    end_pos = last_newline
            
            chunk = content[current_pos:end_pos]
            if chunk.strip():
                chunks.append(chunk)
            
            current_pos = end_pos + 1
        
        return chunks

    def _build_analysis_prompt(self, content: str, file_type: str, program_name: str) -> str:
        """Build analysis prompt based on file type"""
        
        base_prompts = {
            'cobol': f"""
Analyze this COBOL program '{program_name}':

{content}

Provide analysis in JSON format:
{{
    "business_purpose": "description of main business function",
    "key_operations": ["operation1", "operation2"],
    "complexity": "low|medium|high",
    "data_processing": "description of data handling",
    "integration_points": ["files", "databases", "programs"],
    "maintainability": "good|fair|poor"
}}
""",
            
            'jcl': f"""
Analyze this JCL job '{program_name}':

{content}

Provide analysis in JSON format:
{{
    "job_purpose": "description of job function",
    "steps": ["step1", "step2"],
    "complexity": "low|medium|high", 
    "programs_executed": ["prog1", "prog2"],
    "data_flow": "description of data processing",
    "scheduling": "batch|online|daily"
}}
""",
            
            'copybook': f"""
Analyze this copybook '{program_name}':

{content}

Provide analysis in JSON format:
{{
    "data_purpose": "description of data structure",
    "business_domain": "finance|customer|inventory|etc",
    "complexity": "low|medium|high",
    "record_types": ["type1", "type2"],
    "usage_pattern": "input|output|working|shared"
}}
""",
            
            'cics': f"""
Analyze this CICS program '{program_name}':

{content}

Provide analysis in JSON format:
{{
    "transaction_purpose": "description of transaction",
    "user_interaction": "screen|batch|service",
    "complexity": "low|medium|high",
    "cics_resources": ["maps", "files", "queues"],
    "business_function": "description of business process"
}}
"""
        }
        
        return base_prompts.get(file_type, base_prompts['cobol'])

    def _parse_llm_response(self, response: str, file_type: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON"""
        try:
            # Try to find JSON in response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                # Parse JSON
                analysis = json.loads(json_str)
                analysis['llm_analysis'] = True
                analysis['confidence'] = 0.8
                return analysis
            else:
                # Fallback parsing
                return self._fallback_parse_response(response, file_type)
                
        except json.JSONDecodeError:
            return self._fallback_parse_response(response, file_type)

    def _fallback_parse_response(self, response: str, file_type: str) -> Dict[str, Any]:
        """Fallback response parsing when JSON fails"""
        analysis = {
            'llm_analysis': True,
            'confidence': 0.5,
            'raw_response': response[:200],
            'fallback_parsing': True
        }
        
        # Extract key information using simple patterns
        if 'complex' in response.lower():
            analysis['complexity'] = 'high'
        elif 'simple' in response.lower() or 'basic' in response.lower():
            analysis['complexity'] = 'low'
        else:
            analysis['complexity'] = 'medium'
        
        if file_type == 'cobol':
            analysis['business_purpose'] = 'data_processing'
        elif file_type == 'jcl':
            analysis['job_purpose'] = 'batch_processing'
        elif file_type == 'copybook':
            analysis['data_purpose'] = 'data_structure'
        
        return analysis

    def _aggregate_chunk_analyses(self, chunk_analyses: List[Dict[str, Any]], file_type: str) -> Dict[str, Any]:
        """Aggregate analyses from multiple chunks"""
        
        if not chunk_analyses:
            return {"error": "No analyses to aggregate"}
        
        if len(chunk_analyses) == 1:
            return chunk_analyses[0]
        
        # Aggregate common fields
        aggregated = {
            'llm_analysis': True,
            'confidence': sum(ca.get('confidence', 0.5) for ca in chunk_analyses) / len(chunk_analyses),
            'chunks_analyzed': len(chunk_analyses),
            'aggregated': True
        }
        
        # Aggregate complexity (take highest)
        complexities = [ca.get('complexity', 'medium') for ca in chunk_analyses]
        complexity_order = {'low': 1, 'medium': 2, 'high': 3}
        max_complexity = max(complexities, key=lambda x: complexity_order.get(x, 2))
        aggregated['complexity'] = max_complexity
        
        # Aggregate lists (combine unique items)
        list_fields = ['key_operations', 'steps', 'programs_executed', 'integration_points', 'cics_resources']
        for field in list_fields:
            all_items = []
            for ca in chunk_analyses:
                if field in ca and isinstance(ca[field], list):
                    all_items.extend(ca[field])
            if all_items:
                aggregated[field] = list(set(all_items))
        
        # Take first non-empty value for text fields
        text_fields = ['business_purpose', 'job_purpose', 'data_purpose', 'transaction_purpose']
        for field in text_fields:
            for ca in chunk_analyses:
                if field in ca and ca[field]:
                    aggregated[field] = ca[field]
                    break
        
        return aggregated

    async def _check_llm_cache(self, content_hash: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Check LLM analysis cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT analysis_result, confidence_score 
                FROM llm_analysis_cache 
                WHERE content_hash = ? AND analysis_type = ?
            """, (content_hash, analysis_type))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                self.logger.info(f"📋 Cache HIT for {analysis_type}")
                return {
                    'analysis': json.loads(result[0]),
                    'confidence': result[1],
                    'cached': True
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {str(e)}")
            return None

    async def _cache_llm_result(self, content_hash: str, analysis_type: str, analysis: Dict[str, Any]):
        """Cache LLM analysis result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO llm_analysis_cache 
                (content_hash, analysis_type, analysis_result, confidence_score)
                VALUES (?, ?, ?, ?)
            """, (
                content_hash, analysis_type, 
                json.dumps(analysis), analysis.get('confidence', 0.5)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")

    def _merge_enhanced_analysis(self, chunks: List[CodeChunk], enhanced_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Merge enhanced LLM analysis into chunks"""
        
        if enhanced_analysis.get('error') or enhanced_analysis.get('fallback'):
            return chunks  # Return chunks as-is if analysis failed
        
        # Add enhanced analysis to chunk metadata
        for chunk in chunks:
            if 'enhanced_analysis' not in chunk.metadata:
                chunk.metadata['enhanced_analysis'] = {}
            
            # Add relevant analysis fields
            for key, value in enhanced_analysis.items():
                if key not in ['chunks_analyzed', 'aggregated']:
                    chunk.metadata['enhanced_analysis'][key] = value
            
            # Adjust confidence score
            if enhanced_analysis.get('confidence'):
                chunk.confidence_score = min(chunk.confidence_score, enhanced_analysis['confidence'])
        
        return chunks

    async def _store_chunks(self, chunks: List[CodeChunk], file_path: Path):
        """Store chunks in database"""
        if not chunks:
            return
        
        try:
            file_hash = self._generate_file_hash(file_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO program_chunks 
                    (program_name, chunk_id, chunk_type, content, metadata, 
                     file_hash, confidence_score, line_start, line_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.program_name, chunk.chunk_id, chunk.chunk_type,
                    chunk.content, json.dumps(chunk.metadata), file_hash,
                    chunk.confidence_score, chunk.line_start, chunk.line_end
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Stored {len(chunks)} chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store chunks: {str(e)}")

    def _generate_file_hash(self, file_path: Path) -> str:
        """Generate hash for file"""
        try:
            stat_info = file_path.stat()
            hash_input = f"{file_path.name}:{stat_info.st_mtime}:{stat_info.st_size}"
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(file_path).encode()).hexdigest()

    def cleanup(self):
        """Cleanup method"""
        self.logger.info("🧹 Cleaning up Simplified Code Parser Agent...")
        super().cleanup()

    def get_version_info(self) -> Dict[str, str]:
        """Get version information"""
        return {
            "agent_name": "SimplifiedCodeParserAgent",
            "version": "1.0.0-Simplified-Fixed",
            "base_agent": "BaseOpulenceAgent", 
            "deployment_mode": "SIMPLIFIED_RELIABLE",
            "llm_model": "CodeLlama",
            "context_window": "2048 tokens",
            "chunking_strategy": "intelligent_content_aware",
            "parsing_approach": "reliable_patterns_first",
            "relationship_extraction": "pattern_based_reliable",
            "supported_file_types": [".cbl", ".cob", ".jcl", ".cpy", ".copy", ".bms", ".sql", ".db2"],
            "key_improvements": [
                "Fixed file relationship parsing",
                "Simplified reliable patterns", 
                "Proper CodeLlama integration",
                "Intelligent content chunking",
                "Cached LLM analysis",
                "Robust error handling",
                "Correct data storage"
            ]
        }

# Export the simplified class
__all__ = ['SimplifiedCodeParserAgent']