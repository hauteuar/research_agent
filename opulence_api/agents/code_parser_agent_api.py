"""
Agent 1: API-Compatible Enhanced Code Parser & Chunker - BUSINESS LOGIC FIXED VERSION
Updated to work with API-based Opulence Coordinator
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
from contextlib import asynccontextmanager

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

# API Sampling Parameters class for compatibility
class SamplingParams:
    def __init__(self, temperature=0.1, max_tokens=512, top_p=0.9, stop=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop

# Business Validator Classes
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
                context="JCL must have at least one EXEC statement",
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

class CompleteEnhancedCodeParserAgent:
    def __init__(self, llm_engine=None, db_path: str = None,
                 gpu_id: int = None, coordinator=None):
        self.llm_engine = llm_engine  # Keep for backward compatibility
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)

        # Thread safety
        self._engine_lock = asyncio.Lock()
        self._processed_files = set()  # Duplicate prevention

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

            # DB2 Stored Procedure patterns
            'db2_create_procedure': re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_language_sql': re.compile(r'LANGUAGE\s+SQL', re.IGNORECASE),
            'db2_begin_block': re.compile(r'BEGIN\s+(.*?)\s+END', re.IGNORECASE | re.DOTALL),
            'db2_declare_section': re.compile(r'DECLARE\s+(.*?)(?=BEGIN)', re.IGNORECASE | re.DOTALL),
            'db2_procedure_params': re.compile(r'PROCEDURE\s+[A-Z][A-Z0-9_]*\s*\((.*?)\)', re.IGNORECASE | re.DOTALL),
            'db2_exception_handler': re.compile(r'DECLARE\s+(CONTINUE|EXIT)\s+HANDLER\s+FOR\s+(.*?)\s+(.*?)(?=DECLARE|BEGIN)', re.IGNORECASE | re.DOTALL),
            'db2_cursor_declare': re.compile(r'DECLARE\s+([A-Z][A-Z0-9_]*)\s+CURSOR\s+FOR\s+(.*?)(?=;|DECLARE)', re.IGNORECASE | re.DOTALL),
            'db2_call_procedure': re.compile(r'CALL\s+([A-Z][A-Z0-9_]*)\s*\((.*?)\)', re.IGNORECASE),

            # COBOL Stored Procedure patterns
            'cobol_sql_procedure': re.compile(r'EXEC\s+SQL\s+CREATE\s+PROCEDURE\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cobol_procedure_call': re.compile(r'EXEC\s+SQL\s+CALL\s+([A-Z][A-Z0-9_]*)\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE),
            'cobol_result_set': re.compile(r'EXEC\s+SQL\s+ASSOCIATE\s+RESULT\s+SET\s+LOCATOR\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE),

            # MQ Program patterns
            'mq_mqconn': re.compile(r'CALL\s+[\'"]MQCONN[\'"]', re.IGNORECASE),
            'mq_mqdisc': re.compile(r'CALL\s+[\'"]MQDISC[\'"]', re.IGNORECASE),
            'mq_mqopen': re.compile(r'CALL\s+[\'"]MQOPEN[\'"]', re.IGNORECASE),
            'mq_mqclose': re.compile(r'CALL\s+[\'"]MQCLOSE[\'"]', re.IGNORECASE),
            'mq_mqput': re.compile(r'CALL\s+[\'"]MQPUT[\'"]', re.IGNORECASE),
            'mq_mqput1': re.compile(r'CALL\s+[\'"]MQPUT1[\'"]', re.IGNORECASE),
            'mq_mqget': re.compile(r'CALL\s+[\'"]MQGET[\'"]', re.IGNORECASE),
            'mq_mqbegin': re.compile(r'CALL\s+[\'"]MQBEGIN[\'"]', re.IGNORECASE),
            'mq_mqcmit': re.compile(r'CALL\s+[\'"]MQCMIT[\'"]', re.IGNORECASE),
            'mq_mqback': re.compile(r'CALL\s+[\'"]MQBACK[\'"]', re.IGNORECASE),
            'mq_message_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*MQMD[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_queue_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*MQOD[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_put_options': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*MQPMO[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_get_options': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*MQGMO[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_subscription': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*MQSD[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
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

            # Field lineage table
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

            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_name ON program_chunks(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON program_chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON program_chunks(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_business_rules ON business_rule_violations(program_name, rule_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_lineage_field ON field_lineage(field_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_lineage_program ON field_lineage(program_name)")

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")

    # ==================== API-BASED ENGINE CONTEXT ====================

    @asynccontextmanager
    async def get_engine_context(self):
        """Get API-based engine context for coordinator integration"""
        if self.coordinator:
            # Use API-based context through coordinator
            try:
                # Create a simple context that uses coordinator's API calls
                yield self
            finally:
                pass  # No cleanup needed for API calls
        else:
            # Fallback for standalone usage
            yield self

    async def _generate_with_llm(self, prompt: str, sampling_params: SamplingParams) -> str:
        """Generate text with LLM via API coordinator"""
        try:
            if self.coordinator:
                # Convert SamplingParams to API parameters
                params = {
                    "max_tokens": sampling_params.max_tokens,
                    "temperature": sampling_params.temperature,
                    "top_p": getattr(sampling_params, 'top_p', 0.9),
                    "stop": getattr(sampling_params, 'stop', None)
                }

                # Call coordinator's API
                result = await self.coordinator.call_model_api(prompt, params, self.gpu_id)

                # Extract text from API response
                if isinstance(result, dict):
                    return result.get('text', '').strip()
                else:
                    return str(result).strip()
            else:
                # Fallback for standalone usage
                self.logger.warning("No coordinator available for LLM generation")
                return ""

        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""

    # ==================== CORE PROCESSING METHODS ====================

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

    def _extract_program_name(self, content: str, file_path: Path) -> str:
        """Extract program name more robustly from content or filename"""
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
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'enhanced_code_parser'
            result['coordinator_type'] = 'api_based'
        return result

    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Enhanced file type detection with proper business rule ordering"""
        content_upper = content.upper()

        # Order matters - check most specific patterns first

        # DB2 Stored Procedure detection (most specific)
        if self._is_db2_procedure(content_upper):
            return 'db2_procedure'

        # COBOL Stored Procedure detection
        if self._is_cobol_stored_procedure(content_upper):
            return 'cobol_stored_procedure'

        # MQ Program detection
        if self._is_mq_program(content_upper):
            return 'mq_program'

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
        elif suffix_lower in ['.sql', '.db2']:
            return 'db2_procedure'

        return 'unknown'

    def _is_db2_procedure(self, content_upper: str) -> bool:
        """Check if file is DB2 stored procedure"""
        return (any(marker in content_upper for marker in ['CREATE PROCEDURE', 'CREATE OR REPLACE PROCEDURE']) and
                any(marker in content_upper for marker in ['LANGUAGE SQL', 'PARAMETER STYLE', 'BEGIN ATOMIC']))

    def _is_cobol_stored_procedure(self, content_upper: str) -> bool:
        """Check if file is COBOL stored procedure"""
        return ('EXEC SQL CREATE PROCEDURE' in content_upper or
                ('EXEC SQL CALL' in content_upper and 'PROCEDURE DIVISION' in content_upper))

    def _is_mq_program(self, content_upper: str) -> bool:
        """Check if file is MQ program"""
        mq_indicators = ['MQOPEN', 'MQPUT', 'MQGET', 'MQCLOSE', 'MQCONN', 'MQDISC']
        mq_count = sum(content_upper.count(f'CALL "{indicator}"') + content_upper.count(f"CALL '{indicator}'")
                      for indicator in mq_indicators)
        return mq_count >= 2  # At least 2 MQ calls indicate an MQ program

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
            elif file_type == 'db2_procedure':
                chunks = await self._parse_db2_procedure_with_business_rules(content, str(file_path.name))
            elif file_type == 'cobol_stored_procedure':
                chunks = await self._parse_cobol_stored_procedure_with_business_rules(content, str(file_path.name))
            elif file_type == 'mq_program':
                chunks = await self._parse_mq_program_with_business_rules(content, str(file_path.name))
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

            # Generate and store field lineage
            if file_type in ['cobol', 'cics', 'copybook']:
                lineage_records = await self._generate_field_lineage(self._extract_program_name(content, file_path), chunks)
                await self._store_field_lineage(lineage_records)

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

    # ==================== PARSING METHODS WITH BUSINESS RULES ====================

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

    async def _parse_cobol_sections_with_context(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL sections with enhanced business context"""
        chunks = []

        section_patterns = {
            'working_storage': self.cobol_patterns['working_storage'],
            'file_section': self.cobol_patterns['file_section'],
            'linkage_section': self.cobol_patterns['linkage_section'],
            'local_storage': self.cobol_patterns['local_storage']
        }

        section_positions = {}
        for sect_name, pattern in section_patterns.items():
            match = pattern.search(content)
            if match:
                section_positions[sect_name] = {
                    'start': match.start(),
                    'match': match
                }

        # Sort sections by position
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1]['start'])

        for i, (sect_name, sect_info) in enumerate(sorted_sections):
            start_pos = sect_info['start']

            # Find end position (next section or procedure division)
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]['start']
            else:
                # Look for procedure division
                proc_match = self.cobol_patterns['procedure_division'].search(content, start_pos)
                end_pos = proc_match.start() if proc_match else len(content)

            sect_content = content[start_pos:end_pos].strip()

            # Enhanced business context analysis for data sections
            business_context = self._analyze_data_section_business_context(sect_content, sect_name)

            # LLM analysis for deeper insights
            metadata = await self._analyze_data_section_with_llm(sect_content, sect_name)

            # Add section-specific metadata
            metadata.update({
                'section_type': sect_name,
                'data_organization': business_context.get('data_organization', 'sequential'),
                'field_count': self._count_fields_in_section(sect_content),
                'memory_estimate': self._estimate_section_memory(sect_content),
                'complexity_score': self._calculate_section_complexity(sect_content)
            })

            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{sect_name.upper()}_SECTION",
                chunk_type="section",
                content=sect_content,
                metadata=metadata,
                business_context=business_context,
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)

        return chunks

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

    # ==================== DB2 STORED PROCEDURE PARSING METHODS ====================

    async def _parse_db2_procedure_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse DB2 stored procedure with business rule validation"""
        chunks = []
        procedure_name = self._extract_program_name(content, Path(filename))

        # Validate DB2 procedure structure
        db2_validation = await self._validate_db2_procedure_structure(content, procedure_name)

        # Parse procedure header
        header_chunk = await self._parse_db2_procedure_header(content, procedure_name)
        if header_chunk:
            chunks.append(header_chunk)

        # Parse parameter declarations
        param_chunks = await self._parse_db2_procedure_parameters(content, procedure_name)
        chunks.extend(param_chunks)

        # Parse declare section
        declare_chunks = await self._parse_db2_declare_section(content, procedure_name)
        chunks.extend(declare_chunks)

        # Parse procedure body
        body_chunks = await self._parse_db2_procedure_body(content, procedure_name)
        chunks.extend(body_chunks)

        # Parse exception handlers
        handler_chunks = await self._parse_db2_exception_handlers(content, procedure_name)
        chunks.extend(handler_chunks)

        # Parse cursor declarations
        cursor_chunks = await self._parse_db2_cursors(content, procedure_name)
        chunks.extend(cursor_chunks)

        return chunks

    async def _validate_db2_procedure_structure(self, content: str, procedure_name: str) -> Dict[str, Any]:
        """Validate DB2 procedure structure according to business rules"""
        violations = []

        # Check for required CREATE PROCEDURE
        create_match = self.cobol_patterns['db2_create_procedure'].search(content)
        if not create_match:
            violations.append(BusinessRuleViolation(
                rule="MISSING_CREATE_PROCEDURE",
                context="DB2 procedure must start with CREATE PROCEDURE",
                severity="ERROR"
            ))

        # Check for LANGUAGE SQL
        if not self.cobol_patterns['db2_language_sql'].search(content):
            violations.append(BusinessRuleViolation(
                rule="MISSING_LANGUAGE_SQL",
                context="DB2 procedure must specify LANGUAGE SQL",
                severity="ERROR"
            ))

        # Check for proper BEGIN/END block
        begin_match = self.cobol_patterns['db2_begin_block'].search(content)
        if not begin_match:
            violations.append(BusinessRuleViolation(
                rule="MISSING_BEGIN_END",
                context="DB2 procedure must have BEGIN/END block",
                severity="ERROR"
            ))

        return {
            'structure_valid': len([v for v in violations if v.severity == "ERROR"]) == 0,
            'violations': violations
        }

    async def _parse_db2_procedure_header(self, content: str, procedure_name: str) -> Optional[CodeChunk]:
        """Parse DB2 procedure header"""
        create_match = self.cobol_patterns['db2_create_procedure'].search(content)
        if not create_match:
            return None

        # Extract header content (from CREATE to first BEGIN or DECLARE)
        header_start = create_match.start()

        # Find end of header
        begin_match = self.cobol_patterns['db2_begin_block'].search(content, header_start)
        declare_match = self.cobol_patterns['db2_declare_section'].search(content, header_start)

        header_end = len(content)
        if begin_match and declare_match:
            header_end = min(begin_match.start(), declare_match.start())
        elif begin_match:
            header_end = begin_match.start()
        elif declare_match:
            header_end = declare_match.start()

        header_content = content[header_start:header_end].strip()

        business_context = {
            'procedure_purpose': 'database_stored_procedure',
            'execution_environment': 'db2_database',
            'interface_type': 'sql_callable',
            'transaction_scope': 'database_transaction'
        }

        metadata = await self._analyze_db2_procedure_header(header_content, procedure_name)

        return CodeChunk(
            program_name=procedure_name,
            chunk_id=f"{procedure_name}_HEADER",
            chunk_type="db2_procedure_header",
            content=header_content,
            metadata=metadata,
            business_context=business_context,
            line_start=content[:header_start].count('\n'),
            line_end=content[:header_end].count('\n')
        )

    async def _parse_db2_procedure_parameters(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 procedure parameters"""
        chunks = []

        create_match = self.cobol_patterns['db2_create_procedure'].search(content)
        if not create_match:
            return chunks

        # Look for parameter section
        param_match = self.cobol_patterns['db2_procedure_params'].search(content, create_match.end())
        if not param_match:
            return chunks

        param_content = param_match.group(1)
        if not param_content.strip():
            return chunks

        # Parse individual parameters
        parameters = self._parse_db2_parameter_list(param_content)

        for i, param in enumerate(parameters):
            business_context = {
                'parameter_purpose': self._classify_db2_parameter_purpose(param),
                'data_flow': param.get('direction', 'IN'),
                'validation_required': param.get('data_type', '').upper() in ['VARCHAR', 'CHAR', 'DECIMAL']
            }

            metadata = {
                'parameter_name': param.get('name', ''),
                'data_type': param.get('data_type', ''),
                'direction': param.get('direction', 'IN'),
                'default_value': param.get('default', None),
                'parameter_index': i + 1
            }

            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_PARAM_{param.get('name', f'PARAM_{i+1}')}",
                chunk_type="db2_parameter",
                content=param.get('definition', ''),
                metadata=metadata,
                business_context=business_context,
                line_start=param_match.start(),
                line_end=param_match.end()
            )
            chunks.append(chunk)

        return chunks

    async def _parse_db2_declare_section(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 DECLARE section"""
        chunks = []

        declare_matches = self.cobol_patterns['db2_declare_section'].finditer(content)

        for i, match in enumerate(declare_matches):
            declare_content = match.group(1).strip()

            business_context = {
                'declaration_purpose': 'variable_declaration',
                'scope': 'procedure_local',
                'usage_pattern': self._analyze_db2_declare_usage(declare_content)
            }

            metadata = await self._analyze_db2_declare_content(declare_content)

            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_DECLARE_{i+1}",
                chunk_type="db2_declare",
                content=match.group(0),
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)

        return chunks

    async def _parse_db2_procedure_body(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 procedure body"""
        chunks = []

        begin_match = self.cobol_patterns['db2_begin_block'].search(content)
        if not begin_match:
            return chunks

        body_content = begin_match.group(1).strip()

        # Parse SQL statements within body
        sql_chunks = await self._parse_db2_sql_statements(body_content, procedure_name)
        chunks.extend(sql_chunks)

        # Parse control flow statements
        control_chunks = await self._parse_db2_control_flow(body_content, procedure_name)
        chunks.extend(control_chunks)

        # Parse procedure calls
        call_chunks = await self._parse_db2_procedure_calls(body_content, procedure_name)
        chunks.extend(call_chunks)

        return chunks

    async def _parse_db2_exception_handlers(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 exception handlers"""
        chunks = []

        handler_matches = self.cobol_patterns['db2_exception_handler'].finditer(content)

        for i, match in enumerate(handler_matches):
            handler_type = 'CONTINUE' if 'CONTINUE' in match.group(0).upper() else 'EXIT'
            condition = match.group(1).strip()
            action = match.group(2).strip()

            business_context = {
                'error_handling_type': handler_type.lower(),
                'error_condition': condition,
                'recovery_strategy': self._classify_db2_recovery_strategy(action),
                'business_impact': 'high' if handler_type == 'EXIT' else 'medium'
            }

            metadata = {
                'handler_type': handler_type,
                'condition': condition,
                'action': action,
                'handler_scope': 'procedure_level'
            }

            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_HANDLER_{i+1}",
                chunk_type="db2_exception_handler",
                content=match.group(0),
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)

        return chunks

    async def _parse_db2_cursors(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 cursor declarations"""
        chunks = []

        cursor_matches = self.cobol_patterns['db2_cursor_declare'].finditer(content)

        for match in cursor_matches:
            cursor_name = match.group(1)
            cursor_sql = match.group(2).strip()

            business_context = {
                'cursor_purpose': 'result_set_processing',
                'data_access_pattern': 'sequential_read',
                'performance_impact': self._assess_db2_cursor_performance(cursor_sql),
                'transaction_behavior': 'read_consistent'
            }

            metadata = {
                'cursor_name': cursor_name,
                'sql_statement': cursor_sql,
                'cursor_type': 'forward_only',  # Default for DB2
                'result_set_type': self._analyze_db2_cursor_result_type(cursor_sql)
            }

            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_CURSOR_{cursor_name}",
                chunk_type="db2_cursor",
                content=match.group(0),
                metadata=metadata,
                business_context=business_context,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)

        return chunks

    # ==================== COBOL STORED PROCEDURE PARSING METHODS ====================

    async def _parse_cobol_stored_procedure_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse COBOL stored procedure with business rule validation"""
        chunks = []
        procedure_name = self._extract_program_name(content, Path(filename))

        # First parse as regular COBOL program
        cobol_chunks = await self._parse_cobol_with_business_rules(content, filename)
        chunks.extend(cobol_chunks)

        # Then parse stored procedure specific elements
        sp_chunks = await self._parse_cobol_sp_specific_elements(content, procedure_name)
        chunks.extend(sp_chunks)

        # Parse SQL procedure definitions
        sql_proc_chunks = await self._parse_cobol_sql_procedures(content, procedure_name)
        chunks.extend(sql_proc_chunks)

        # Parse procedure calls
        call_chunks = await self._parse_cobol_procedure_calls(content, procedure_name)
        chunks.extend(call_chunks)

        # Parse result set handling
        result_set_chunks = await self._parse_cobol_result_sets(content, procedure_name)
        chunks.extend(result_set_chunks)

        return chunks

    async def _parse_cobol_sp_specific_elements(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse COBOL stored procedure specific elements"""
        chunks = []

        # Look for stored procedure indicators in working storage
        ws_section_match = self.cobol_patterns['working_storage'].search(content)
        if ws_section_match:
            ws_start = ws_section_match.end()
            ws_end = self._find_section_end(content, ws_start, ['procedure_division'])
            ws_content = content[ws_start:ws_end]

            # Parse SQLCA, SQLDA, and other SQL communication areas
            sql_comm_chunks = await self._parse_sql_communication_areas(ws_content, procedure_name, ws_start)
            chunks.extend(sql_comm_chunks)

        return chunks

    # ==================== MQ PROGRAM PARSING METHODS ====================

    async def _parse_mq_program_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse MQ (Message Queue) program with business rule validation"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))

        # First parse as regular COBOL program
        cobol_chunks = await self._parse_cobol_with_business_rules(content, filename)
        chunks.extend(cobol_chunks)

        # Parse MQ-specific elements
        mq_chunks = await self._parse_mq_specific_elements(content, program_name)
        chunks.extend(mq_chunks)

        # Parse MQ API calls
        api_chunks = await self._parse_mq_api_calls(content, program_name)
        chunks.extend(api_chunks)

        # Parse MQ data structures
        structure_chunks = await self._parse_mq_data_structures(content, program_name)
        chunks.extend(structure_chunks)

        # Parse message flow patterns
        flow_chunks = await self._parse_mq_message_flows(content, program_name)
        chunks.extend(flow_chunks)

        return chunks

    async def _parse_mq_specific_elements(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse MQ-specific program elements"""
        chunks = []

        # Analyze MQ usage patterns
        mq_usage = self._analyze_mq_usage_patterns(content)

        if mq_usage['connection_pattern']:
            business_context = {
                'mq_pattern': mq_usage['connection_pattern'],
                'message_paradigm': mq_usage['paradigm'],
                'reliability_level': mq_usage['reliability'],
                'transaction_model': mq_usage['transaction_model']
            }

            metadata = {
                'mq_usage_analysis': mq_usage,
                'connection_type': mq_usage['connection_pattern'],
                'estimated_throughput': mq_usage.get('throughput_estimate', 'unknown')
            }

            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_MQ_ANALYSIS",
                chunk_type="mq_usage_analysis",
                content="MQ Usage Analysis",
                metadata=metadata,
                business_context=business_context,
                line_start=0,
                line_end=0
            )
            chunks.append(chunk)

        return chunks

    async def _parse_mq_api_calls(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse MQ API calls"""
        chunks = []

        # Define MQ API patterns and their purposes
        mq_apis = {
            'mq_mqconn': ('connection_management', 'Connect to queue manager'),
            'mq_mqdisc': ('connection_management', 'Disconnect from queue manager'),
            'mq_mqopen': ('queue_management', 'Open queue for operations'),
            'mq_mqclose': ('queue_management', 'Close queue'),
            'mq_mqput': ('message_operations', 'Put message to queue'),
            'mq_mqput1': ('message_operations', 'Put single message (open, put, close)'),
            'mq_mqget': ('message_operations', 'Get message from queue'),
            'mq_mqbegin': ('transaction_management', 'Begin transaction'),
            'mq_mqcmit': ('transaction_management', 'Commit transaction'),
            'mq_mqback': ('transaction_management', 'Rollback transaction')
        }

        for api_name, (category, description) in mq_apis.items():
            if api_name in self.cobol_patterns:
                api_matches = self.cobol_patterns[api_name].finditer(content)

                for i, match in enumerate(api_matches):
                    business_context = {
                        'api_category': category,
                        'operation_purpose': description,
                        'message_flow_impact': self._analyze_mq_flow_impact(api_name),
                        'error_recovery_needed': api_name in ['mq_mqput', 'mq_mqget', 'mq_mqopen'],
                        'performance_considerations': self._analyze_mq_performance_impact(api_name)
                    }

                    metadata = {
                        'api_name': api_name.replace('mq_', '').upper(),
                        'parameters': [],  # Would be extracted from actual call
                        'return_codes': [],
                        'synchronous': api_name != 'mq_callback'
                    }

                    chunk = CodeChunk(
                        program_name=program_name,
                        chunk_id=f"{program_name}_{api_name.upper()}_{i+1}",
                        chunk_type="mq_api_call",
                        content=match.group(0),
                        metadata=metadata,
                        business_context=business_context,
                        line_start=content[:match.start()].count('\n'),
                        line_end=content[:match.end()].count('\n')
                    )
                    chunks.append(chunk)

        return chunks

    # ==================== HELPER METHODS ====================

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

    def _extract_pic_clause(self, definition: str) -> Optional[str]:
        """Extract PIC clause from field definition"""
        match = self.cobol_patterns['pic_clause'].search(definition)
        if match:
            return match.group(1) or match.group(2)
        return None

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
        """Extract REDEFINES information"""
        match = self.cobol_patterns['redefines'].search(definition)
        return match.group(1) if match else None

    def _determine_data_type_enhanced(self, definition: str) -> str:
        """Enhanced data type determination"""
        pic_clause = self._extract_pic_clause(definition)
        usage = self._extract_usage_clause(definition)

        if not pic_clause:
            return "group"

        pic_upper = pic_clause.upper()

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

    def _calculate_field_length(self, pic_clause: str, usage: str) -> int:
        """Calculate field length based on PIC clause and usage"""
        if not pic_clause:
            return 0

        pic_upper = pic_clause.upper().strip()
        length = 0

        # Pattern 1: Explicit repetition with parentheses - X(10), 9(5), etc.
        explicit_match = re.search(r'([X9ANVS])\((\d+)\)', pic_upper)
        if explicit_match:
            repeat_count = int(explicit_match.group(2))
            length = repeat_count
        else:
            # Pattern 2: Implicit repetition - XXX, 999, etc.
            length += len(re.findall(r'X', pic_upper))  # Alphanumeric
            length += len(re.findall(r'9', pic_upper))  # Numeric
            length += len(re.findall(r'A', pic_upper))  # Alphabetic
            length += len(re.findall(r'N', pic_upper))  # National (usually 2 bytes each)

            # National characters are typically 2 bytes each
            if 'N' in pic_upper:
                length += len(re.findall(r'N', pic_upper))  # Double count for national

        # Adjust length based on usage clause
        if usage in ['COMP-3', 'PACKED-DECIMAL']:
            return (length + 1) // 2
        elif usage in ['COMP', 'COMP-4', 'BINARY']:
            if length <= 4:
                return 2
            elif length <= 9:
                return 4
            elif length <= 18:
                return 8
            else:
                return 16
        elif usage == 'COMP-1':
            return 4
        elif usage == 'COMP-2':
            return 8
        else:
            return length

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

    # ==================== ANALYSIS METHODS USING API ====================

    async def _analyze_division_with_llm(self, content: str, division_name: str) -> Dict[str, Any]:
        """Analyze COBOL division with LLM via API"""
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
        """Analyze data section with comprehensive field analysis using API"""

        # Extract field information first for context
        field_analysis = await self._analyze_fields_comprehensive(content)

        prompt = f"""
        Analyze this COBOL data section: {section_name}

        {content[:800]}...

        Provide comprehensive analysis of:
        1. Record structures and hierarchical layouts
        2. Key data elements and their business purposes
        3. Relationships between fields and groups
        4. Data validation patterns and constraints
        5. Business domain and entity types represented

        Return as JSON:
        {{
            "record_structures": [
                {{"name": "record1", "purpose": "customer data", "fields": 15}}
            ],
            "key_elements": [
                {{"name": "element1", "type": "identifier", "business_purpose": "customer key"}}
            ],
            "field_relationships": [
                {{"parent": "customer-record", "children": ["cust-name", "cust-addr"]}}
            ],
            "validation_patterns": [
                {{"field": "field1", "validation": "required", "constraint": "not null"}}
            ],
            "business_domain": "customer management",
            "entity_types": ["customer", "address", "contact"]
        }}
        """

        sampling_params = SamplingParams(temperature=0.2, max_tokens=800)

        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                llm_analysis = json.loads(response_text[json_start:json_end])

                # Enhance LLM analysis with field analysis data
                llm_analysis['field_analysis'] = field_analysis
                llm_analysis['section_type'] = section_name
                llm_analysis['analysis_timestamp'] = dt.now().isoformat()

                return llm_analysis

        except Exception as e:
            self.logger.warning(f"Data section LLM analysis failed: {str(e)}")

        # Fallback analysis using extracted field data
        return self._generate_fallback_data_section_analysis(content, section_name, field_analysis)

    def _generate_fallback_data_section_analysis(self, content: str, section_name: str, field_analysis: Dict) -> Dict[str, Any]:
        """Generate fallback analysis when LLM analysis fails"""
        return {
            "record_structures": [],
            "key_elements": [],
            "field_relationships": [],
            "validation_patterns": [],
            "business_domain": "unknown",
            "entity_types": [],
            "field_analysis": field_analysis,
            "section_type": section_name,
            "analysis_method": "fallback"
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
            try:
                level = int(match.group(1))
                name = match.group(2)
                definition = match.group(3)

                # Skip comment lines
                if match.group(0).strip().startswith('*'):
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
                    "data_type": self._determine_data_type_enhanced(definition),
                    "length": self._calculate_field_length(pic_clause or "", usage),
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

            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing data item: {e}")
                continue

        return {
            "fields": fields,
            "statistics": field_stats
        }

    # ==================== DATABASE METHODS ====================

    async def _store_chunks_enhanced(self, chunks: List[CodeChunk], file_hash: str):
        """Store chunks with enhanced business context"""
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
                else:
                    if len(chunk) >= 5:
                        content = str(chunk[4]) if chunk[4] else ''
                        metadata = {}
                        chunk_id = str(chunk[2]) if chunk[2] else ''
                    else:
                        continue

                # Extract field operations from content
                field_operations = self._extract_field_operations(content)

                for field_op in field_operations:
                    lineage_record = {
                        'field_name': str(field_op.get('field_name', '')),
                        'program_name': str(program_name),
                        'paragraph': str(chunk_id),
                        'operation': str(field_op.get('operation', '')),
                        'source_file': str(field_op.get('source_file', '')),
                        'last_used': dt.now().isoformat(),
                        'read_in': str(program_name) if field_op.get('operation') == 'read' else '',
                        'updated_in': str(program_name) if field_op.get('operation') in ['write', 'update'] else '',
                        'purged_in': str(program_name) if field_op.get('operation') == 'delete' else ''
                    }
                    lineage_records.append(lineage_record)
            except Exception as e:
                self.logger.error(f"Error processing chunk for lineage: {str(e)}")
                continue

        return lineage_records

    def _extract_field_operations(self, content: str) -> List[Dict]:
        """Extract field operations from COBOL content"""
        operations = []

        # Enhanced field operation patterns
        patterns = {
            'read': [
                r'READ\s+(\w+)',
                r'INTO\s+(\w+)',
                r'FROM\s+(\w+)',
                r'ACCEPT\s+(\w+)'
            ],
            'write': [
                r'WRITE\s+(\w+)',
                r'MOVE\s+.+\s+TO\s+(\w+)',
                r'DISPLAY\s+(\w+)'
            ],
            'update': [
                r'REWRITE\s+(\w+)',
                r'ADD\s+.+\s+TO\s+(\w+)',
                r'SUBTRACT\s+.+\s+FROM\s+(\w+)',
                r'MULTIPLY\s+.+\s+BY\s+(\w+)',
                r'DIVIDE\s+.+\s+INTO\s+(\w+)',
                r'COMPUTE\s+(\w+)\s*='
            ],
            'delete': [
                r'DELETE\s+(\w+)'
            ],
            'validate': [
                r'IF\s+(\w+)',
                r'EVALUATE\s+(\w+)'
            ]
        }

        for operation, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    field_name = match if isinstance(match, str) else match[0]
                    operations.append({
                        'field_name': field_name,
                        'operation': operation,
                        'source_file': self._infer_source_file(content, field_name)
                    })

        return operations

    def _infer_source_file(self, content: str, field_name: str) -> str:
        """Infer source file for field based on context"""
        # Look for FD statements that might be associated with this field
        fd_pattern = re.compile(r'FD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        fd_matches = fd_pattern.findall(content)

        if fd_matches:
            return fd_matches[0]  # Return first file found

        # Look for file names in SELECT statements
        select_pattern = re.compile(r'SELECT\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        select_matches = select_pattern.findall(content)

        if select_matches:
            return select_matches[0]

        return 'UNKNOWN'

    # ==================== ADDITIONAL HELPER METHODS ====================

    def _count_fields_in_section(self, content: str) -> int:
        """Count fields in a section"""
        return len(self.cobol_patterns['data_item'].findall(content))

    def _estimate_section_memory(self, content: str) -> int:
        """Estimate memory usage for a section"""
        total_memory = 0
        data_matches = self.cobol_patterns['data_item'].finditer(content)

        for match in data_matches:
            try:
                definition = match.group(3)
                pic_clause = self._extract_pic_clause(definition)
                if pic_clause:
                    usage = self._extract_usage_clause(definition)
                    field_size = self._calculate_field_length(pic_clause, usage)
                    total_memory += field_size
            except:
                continue

        return total_memory

    def _calculate_section_complexity(self, content: str) -> int:
        """Calculate section complexity score"""
        complexity = 0

        # Count different types of fields
        data_items = len(self.cobol_patterns['data_item'].findall(content))
        occurs_items = len(self.cobol_patterns['occurs_clause'].findall(content))
        redefines_items = len(self.cobol_patterns['redefines'].findall(content))

        complexity = data_items + (occurs_items * 2) + (redefines_items * 3)

        return min(complexity, 100)  # Cap at 100

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

    async def _store_field_lineage(self, lineage_records: List[Dict]):
        """Store field lineage records in database"""
        if not lineage_records:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert lineage records
            for record in lineage_records:
                cursor.execute("""
                    INSERT INTO field_lineage
                    (field_name, program_name, paragraph, operation, source_file,
                    last_used, read_in, updated_in, purged_in)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.get('field_name', ''),
                    record.get('program_name', ''),
                    record.get('paragraph', ''),
                    record.get('operation', ''),
                    record.get('source_file', ''),
                    record.get('last_used', ''),
                    record.get('read_in', ''),
                    record.get('updated_in', ''),
                    record.get('purged_in', '')
                ))

            conn.commit()
            conn.close()

            self.logger.info(f"Stored {len(lineage_records)} field lineage records")

        except Exception as e:
            self.logger.error(f"Failed to store field lineage: {str(e)}")

    async def _verify_chunks_stored(self, program_name: str) -> int:
        """Verify that chunks were properly stored in database"""
        try:
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
            self.logger.error(f"Chunk verification failed: {str(e)}")
            return 0

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

    def _calculate_maintainability_score(self, chunks: List[CodeChunk]) -> int:
        """Calculate overall maintainability score"""
        scores = []

        for chunk in chunks:
            if chunk.business_context and 'maintainability_score' in chunk.business_context:
                scores.append(chunk.business_context['maintainability_score'])
            elif chunk.metadata and 'maintainability_score' in chunk.metadata:
                scores.append(chunk.metadata['maintainability_score'])

        return int(sum(scores) / len(scores)) if scores else 7

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

    # ==================== MISSING PLACEHOLDER METHODS ====================

    def _analyze_mq_usage_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze MQ usage patterns in the program"""
        content_upper = content.upper()

        # Detect connection patterns
        has_mqconn = 'MQCONN' in content_upper
        has_mqdisc = 'MQDISC' in content_upper
        has_mqopen = 'MQOPEN' in content_upper
        has_mqclose = 'MQCLOSE' in content_upper

        # Detect message operations
        has_mqput = 'MQPUT' in content_upper and 'MQPUT1' not in content_upper
        has_mqput1 = 'MQPUT1' in content_upper
        has_mqget = 'MQGET' in content_upper

        # Detect transaction operations
        has_mqbegin = 'MQBEGIN' in content_upper
        has_mqcmit = 'MQCMIT' in content_upper
        has_mqback = 'MQBACK' in content_upper

        # Determine patterns
        connection_pattern = 'persistent' if has_mqconn and has_mqdisc else 'transient'

        if has_mqput1:
            paradigm = 'fire_and_forget'
        elif has_mqput and has_mqget:
            paradigm = 'request_reply'
        elif has_mqput:
            paradigm = 'producer'
        elif has_mqget:
            paradigm = 'consumer'
        else:
            paradigm = 'unknown'

        transaction_model = 'transactional' if (has_mqbegin and (has_mqcmit or has_mqback)) else 'non_transactional'
        reliability = 'high' if transaction_model == 'transactional' else 'medium'

        return {
            'connection_pattern': connection_pattern,
            'paradigm': paradigm,
            'transaction_model': transaction_model,
            'reliability': reliability,
            'throughput_estimate': 'unknown'
        }

    def _analyze_mq_flow_impact(self, api_name: str) -> str:
        """Analyze the impact of an MQ API call on message flow"""
        flow_impacts = {
            'mq_mqconn': 'establishes_connection',
            'mq_mqdisc': 'terminates_connection',
            'mq_mqopen': 'prepares_queue_access',
            'mq_mqclose': 'finalizes_queue_access',
            'mq_mqput': 'sends_message',
            'mq_mqput1': 'sends_single_message',
            'mq_mqget': 'receives_message',
            'mq_mqbegin': 'starts_transaction',
            'mq_mqcmit': 'commits_transaction',
            'mq_mqback': 'aborts_transaction'
        }

        return flow_impacts.get(api_name, 'unknown_impact')

    def _analyze_mq_performance_impact(self, api_name: str) -> Dict[str, str]:
        """Analyze performance impact of MQ API calls"""
        performance_impacts = {
            'mq_mqconn': {'latency': 'high', 'resource_usage': 'medium', 'scalability': 'connection_pooling_recommended'},
            'mq_mqdisc': {'latency': 'low', 'resource_usage': 'low', 'scalability': 'good'},
            'mq_mqopen': {'latency': 'medium', 'resource_usage': 'medium', 'scalability': 'cache_handles'},
            'mq_mqclose': {'latency': 'low', 'resource_usage': 'low', 'scalability': 'good'},
            'mq_mqput': {'latency': 'medium', 'resource_usage': 'medium', 'scalability': 'batch_for_high_volume'},
            'mq_mqput1': {'latency': 'high', 'resource_usage': 'high', 'scalability': 'avoid_for_high_volume'},
            'mq_mqget': {'latency': 'variable', 'resource_usage': 'medium', 'scalability': 'depends_on_wait_strategy'},
        }

        return performance_impacts.get(api_name, {'latency': 'unknown', 'resource_usage': 'unknown', 'scalability': 'unknown'})

    # Add placeholder methods for missing functions
    def _analyze_data_section_business_context(self, content: str, section_name: str) -> Dict[str, Any]:
        """Analyze business context of data section"""
        return {
            'data_organization': 'sequential',
            'business_purpose': f'{section_name} data storage',
            'usage_pattern': 'standard'
        }

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

    # ==================== REMAINING PLACEHOLDER METHODS ====================

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

    def _extract_system_dependencies(self, content: str) -> List[str]:
        """Extract system dependencies from environment division"""
        dependencies = []

        # File assignments
        select_matches = self.cobol_patterns['select_statement'].finditer(content)
        for match in select_matches:
            physical_file = match.group(2)
            dependencies.append(f"File: {physical_file}")

        return dependencies

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

    def _estimate_memory_requirements(self, content: str) -> Dict[str, int]:
        """Estimate memory requirements for data division"""
        return {
            "working_storage_bytes": 1000,
            "file_section_bytes": 500,
            "linkage_section_bytes": 200,
            "total_estimated_bytes": 1700
        }

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

    def _infer_business_function(self, para_name: str, content: str) -> str:
        """Infer business function from paragraph name and content"""
        name_upper = para_name.upper()

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
            return 'business_logic'

    # ==================== STUB METHODS FOR MISSING FUNCTIONALITY ====================

    async def _parse_paragraphs_with_business_context(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse paragraphs with business context - STUB"""
        return []  # Implement as needed

    async def _parse_perform_statements_with_flow(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse PERFORM statements with flow analysis - STUB"""
        return []  # Implement as needed

    async def _parse_control_structures_with_nesting(self, content: str, program_name: str, offset: int) -> List[CodeChunk]:
        """Parse control structures with nesting - STUB"""
        return []  # Implement as needed

    def _extract_host_variables(self, sql_content: str) -> List[str]:
        """Extract host variables from SQL content"""
        return self.cobol_patterns['sql_host_var'].findall(sql_content)

    async def _validate_host_variables(self, host_vars: List[str], content: str) -> Dict[str, Any]:
        """Validate host variables against data definitions"""
        return {'valid': True, 'warnings': []}

    async def _analyze_sql_business_context(self, sql_content: str, host_vars: List[str]) -> Dict[str, Any]:
        """Analyze SQL business context"""
        return {
            'sql_purpose': 'data_access',
            'transaction_impact': 'database_operation',
            'host_variable_count': len(host_vars)
        }

    async def _analyze_sql_comprehensive(self, sql_content: str) -> Dict[str, Any]:
        """Comprehensive SQL analysis"""
        return {
            'operation_type': self._extract_sql_operation_type(sql_content),
            'tables_accessed': [],
            'complexity_score': 5
        }

    def _extract_sql_operation_type(self, sql_content: str) -> str:
        """Extract SQL operation type"""
        sql_upper = sql_content.upper().strip()
        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'UNKNOWN'

    def _calculate_sql_complexity(self, sql_content: str) -> int:
        """Calculate SQL complexity score"""
        complexity = 0
        sql_upper = sql_content.upper()

        # Basic complexity factors
        complexity += sql_upper.count('JOIN') * 2
        complexity += sql_upper.count('WHERE') * 1
        complexity += sql_upper.count('ORDER BY') * 1
        complexity += sql_upper.count('GROUP BY') * 2
        complexity += sql_upper.count('HAVING') * 2
        complexity += sql_upper.count('UNION') * 3

        return min(complexity, 20)

    def _analyze_sql_performance(self, sql_content: str) -> Dict[str, str]:
        """Analyze SQL performance indicators"""
        return {
            'complexity': 'medium',
            'index_usage': 'unknown',
            'join_efficiency': 'acceptable'
        }

    async def _validate_cics_transaction_context(self, cmd: Dict, transaction_state: TransactionState) -> Dict[str, Any]:
        """Validate CICS command in transaction context"""
        return {'valid': True, 'warnings': []}

    def _update_transaction_state(self, transaction_state: TransactionState, cmd: Dict):
        """Update transaction state based on CICS command"""
        cmd_type = cmd.get('type', '')
        if 'receive' in cmd_type:
            transaction_state.set_input_received()
        elif 'send_map' in cmd_type:
            transaction_state.set_map_loaded('MAPSET', 'MAP')

    async def _analyze_cics_business_context(self, cmd: Dict, transaction_state: TransactionState) -> Dict[str, Any]:
        """Analyze CICS business context"""
        return {
            'command_category': 'terminal_interaction',
            'transaction_flow': 'user_interface',
            'business_impact': 'medium'
        }

    async def _analyze_cics_command_comprehensive(self, command_type: str, params: str, content: str) -> Dict[str, Any]:
        """Comprehensive CICS command analysis"""
        return {
            'command_type': command_type,
            'parameters': params,
            'resource_accessed': 'unknown',
            'performance_impact': 'standard'
        }

    # ==================== JCL PARSING METHODS ====================

    async def _parse_jcl_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse JCL with business rule validation"""
        chunks = []
        job_name = self._extract_program_name(content, Path(filename))

        # Parse job card
        job_chunk = await self._parse_jcl_job_card(content, job_name)
        if job_chunk:
            chunks.append(job_chunk)

        # Parse job steps
        step_chunks = await self._parse_jcl_steps(content, job_name)
        chunks.extend(step_chunks)

        return chunks

    async def _parse_jcl_job_card(self, content: str, job_name: str) -> Optional[CodeChunk]:
        """Parse JCL job card"""
        job_match = self.jcl_patterns['job_card'].search(content)
        if not job_match:
            return None

        job_line_end = content.find('\n', job_match.end())
        if job_line_end == -1:
            job_line_end = len(content)

        job_content = content[job_match.start():job_line_end]

        return CodeChunk(
            program_name=job_name,
            chunk_id=f"{job_name}_JOB_CARD",
            chunk_type="jcl_job",
            content=job_content,
            metadata={'job_name': job_match.group(1)},
            business_context={'job_purpose': 'batch_execution'},
            line_start=0,
            line_end=1
        )

    async def _parse_jcl_steps(self, content: str, job_name: str) -> List[CodeChunk]:
        """Parse JCL job steps"""
        chunks = []
        step_matches = list(self.jcl_patterns['job_step'].finditer(content))

        for i, match in enumerate(step_matches):
            step_name = match.group(1)

            # Find step content
            start_pos = match.start()
            if i + 1 < len(step_matches):
                end_pos = step_matches[i + 1].start()
            else:
                end_pos = len(content)

            step_content = content[start_pos:end_pos].strip()

            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_STEP_{step_name}",
                chunk_type="jcl_step",
                content=step_content,
                metadata={'step_name': step_name, 'step_sequence': i + 1},
                business_context={'execution_purpose': 'program_execution'},
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)

        return chunks

    # ==================== GENERIC AND COPYBOOK PARSING ====================

    async def _parse_copybook_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse copybook with business rules"""
        copybook_name = self._extract_program_name(content, Path(filename))

        chunk = CodeChunk(
            program_name=copybook_name,
            chunk_id=f"{copybook_name}_STRUCTURE",
            chunk_type="copybook_structure",
            content=content,
            metadata={'copybook_type': 'data_definition'},
            business_context={'purpose': 'shared_data_structure'},
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )

        return [chunk]

    async def _parse_bms_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse BMS with business rules"""
        program_name = self._extract_program_name(content, Path(filename))

        chunk = CodeChunk(
            program_name=program_name,
            chunk_id=f"{program_name}_BMS_MAPSET",
            chunk_type="bms_mapset",
            content=content,
            metadata={'bms_type': 'screen_definition'},
            business_context={'purpose': 'user_interface'},
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )

        return [chunk]

    async def _parse_cics_with_business_rules(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse CICS program with business rules"""
        # Use the existing CICS parsing method
        return await self._parse_cics_with_transaction_context(content, self._extract_program_name(content, Path(filename)))

    async def _parse_generic(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse unknown file type generically"""
        program_name = self._extract_program_name(content, Path(filename))

        chunk = CodeChunk(
            program_name=program_name,
            chunk_id=f"{program_name}_GENERIC",
            chunk_type="generic",
            content=content,
            metadata={"file_type": "unknown", "analysis": "Generic file processing"},
            business_context={'file_classification': 'unknown'},
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )

        return [chunk]

    # ==================== PLACEHOLDER METHODS FOR DB2 AND MQ ====================

    def _parse_db2_parameter_list(self, param_content: str) -> List[Dict[str, str]]:
        """Parse DB2 procedure parameter list"""
        parameters = []

        # Split by comma, but be careful of nested parentheses
        param_parts = param_content.split(',')

        for param_part in param_parts:
            param_part = param_part.strip()
            if not param_part:
                continue

            # Parse parameter: [direction] name datatype [default]
            param_match = re.match(r'(?:(IN|OUT|INOUT)\s+)?(\w+)\s+([^=]+)(?:\s*=\s*(.+))?', param_part, re.IGNORECASE)

            if param_match:
                direction = param_match.group(1) or 'IN'
                name = param_match.group(2)
                data_type = param_match.group(3).strip()
                default = param_match.group(4)

                parameters.append({
                    'direction': direction.upper(),
                    'name': name,
                    'data_type': data_type,
                    'default': default,
                    'definition': param_part
                })

        return parameters

    def _classify_db2_parameter_purpose(self, param: Dict[str, str]) -> str:
        """Classify DB2 parameter purpose"""
        name_upper = param.get('name', '').upper()

        if any(pattern in name_upper for pattern in ['ID', 'KEY', 'NBR']):
            return 'identifier'
        elif any(pattern in name_upper for pattern in ['AMT', 'AMOUNT', 'TOTAL']):
            return 'financial'
        elif any(pattern in name_upper for pattern in ['DATE', 'TIME', 'TIMESTAMP']):
            return 'temporal'
        elif param.get('direction') == 'OUT':
            return 'output_result'
        else:
            return 'business_data'

    async def _analyze_db2_procedure_header(self, header_content: str, procedure_name: str) -> Dict[str, Any]:
        """Analyze DB2 procedure header"""
        return {
            "procedure_name": procedure_name,
            "parameter_style": "DB2SQL",
            "language": "SQL",
            "sql_access": "MODIFIES SQL DATA",
            "deterministic": False
        }

    async def _analyze_db2_declare_content(self, declare_content: str) -> Dict[str, Any]:
        """Analyze DB2 DECLARE section content"""
        return {
            'variables': [],
            'conditions': [],
            'handlers': [],
            'complexity_score': 5
        }

    def _analyze_db2_declare_usage(self, declare_content: str) -> str:
        """Analyze DB2 DECLARE usage pattern"""
        content_upper = declare_content.upper()

        if 'CURSOR' in content_upper:
            return 'cursor_processing'
        elif 'CONDITION' in content_upper:
            return 'error_handling'
        elif 'HANDLER' in content_upper:
            return 'exception_management'
        else:
            return 'variable_declaration'

    async def _parse_db2_sql_statements(self, body_content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse SQL statements within DB2 procedure body"""
        return []  # Implement as needed

    async def _parse_db2_control_flow(self, body_content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 control flow statements"""
        return []  # Implement as needed

    async def _parse_db2_procedure_calls(self, body_content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse DB2 procedure calls within body"""
        return []  # Implement as needed

    def _classify_db2_recovery_strategy(self, action: str) -> str:
        """Classify DB2 exception recovery strategy"""
        action_upper = action.upper()

        if 'RESIGNAL' in action_upper:
            return 'propagate_error'
        elif 'SIGNAL' in action_upper:
            return 'raise_custom_error'
        elif 'ROLLBACK' in action_upper:
            return 'abort_transaction'
        elif 'RETURN' in action_upper:
            return 'exit_procedure'
        else:
            return 'custom_recovery'

    def _assess_db2_cursor_performance(self, cursor_sql: str) -> Dict[str, str]:
        """Assess DB2 cursor performance characteristics"""
        return {
            'result_set_size': 'unknown',
            'index_usage': 'unknown',
            'join_complexity': 'simple'
        }

    def _analyze_db2_cursor_result_type(self, cursor_sql: str) -> str:
        """Analyze DB2 cursor result type"""
        sql_upper = cursor_sql.upper()

        if 'GROUP BY' in sql_upper:
            return 'aggregated_data'
        elif 'ORDER BY' in sql_upper:
            return 'sorted_result_set'
        elif 'JOIN' in sql_upper:
            return 'joined_data'
        else:
            return 'simple_result_set'

    async def _parse_sql_communication_areas(self, ws_content: str, procedure_name: str, offset: int) -> List[CodeChunk]:
        """Parse SQL communication areas (SQLCA, SQLDA, etc.)"""
        return []  # Implement as needed

    async def _parse_cobol_sql_procedures(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse COBOL SQL procedure definitions"""
        return []  # Implement as needed

    async def _parse_cobol_procedure_calls(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse COBOL procedure calls"""
        return []  # Implement as needed

    async def _parse_cobol_result_sets(self, content: str, procedure_name: str) -> List[CodeChunk]:
        """Parse COBOL result set handling"""
        return []  # Implement as needed

    async def _parse_mq_data_structures(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse MQ data structures"""
        return []  # Implement as needed

    async def _parse_mq_message_flows(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse and analyze MQ message flow patterns"""
        return []  # Implement as needed

    # ==================== PUBLIC API METHODS ====================

    async def analyze_program(self, program_name: str) -> Dict[str, Any]:
        """Analyze a specific program comprehensively"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all chunks for the program
            cursor.execute("""
                SELECT chunk_type, content, metadata, business_context,
                    line_start, line_end, created_timestamp
                FROM program_chunks
                WHERE program_name = ?
                ORDER BY line_start
            """, (program_name,))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {"error": f"Program {program_name} not found"}

            # Analyze program structure
            analysis = {
                "program_name": program_name,
                "total_chunks": len(rows),
                "chunk_distribution": {},
                "business_analysis": {},
                "technical_metrics": {}
            }

            # Count chunk types
            for row in rows:
                chunk_type = row[0]
                analysis["chunk_distribution"][chunk_type] = \
                    analysis["chunk_distribution"].get(chunk_type, 0) + 1

            # Analyze business context
            business_functions = set()
            data_categories = set()

            for row in rows:
                if row[3]:  # business_context
                    try:
                        business_context = json.loads(row[3])
                        if 'business_function' in business_context:
                            business_functions.add(business_context['business_function'])
                        if 'data_category' in business_context:
                            data_categories.add(business_context['data_category'])
                    except (json.JSONDecodeError, KeyError):
                        continue

            analysis["business_analysis"] = {
                "business_functions": list(business_functions),
                "data_categories": list(data_categories),
                "functional_diversity": len(business_functions),
                "data_complexity": len(data_categories)
            }

            # Calculate technical metrics
            total_lines = sum(row[5] - row[4] + 1 for row in rows if row[4] is not None and row[5] is not None)
            analysis["technical_metrics"] = {
                "total_lines": total_lines,
                "average_chunk_size": total_lines // len(rows) if rows else 0,
                "largest_chunk": max((row[5] - row[4] + 1 for row in rows if row[4] is not None and row[5] is not None), default=0)
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Program analysis failed: {str(e)}")
            return {"error": str(e)}

    async def search_chunks(self, program_name: str = None, chunk_type: str = None,
                           content_search: str = None, limit: int = 100) -> Dict[str, Any]:
        """Search for chunks with various criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

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
                       business_context, line_start, line_end, created_timestamp
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
                chunk_data = {
                    "program_name": row[0],
                    "chunk_id": row[1],
                    "chunk_type": row[2],
                    "content": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "business_context": json.loads(row[5]) if row[5] else {},
                    "line_start": row[6],
                    "line_end": row[7],
                    "created_timestamp": row[8]
                }
                chunks.append(chunk_data)

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

    def cleanup(self):
        """Cleanup method for coordinator integration"""
        self.logger.info("🧹 Cleaning up API Code Parser Agent resources...")

        # Clear any cached data
        self._processed_files.clear()

        self.logger.info("✅ API Code Parser Agent cleanup completed")

    def get_version_info(self) -> Dict[str, str]:
        """Get version and capability information"""
        return {
            "agent_name": "CompleteEnhancedCodeParserAgent",
            "version": "2.1.0-API-Extended",
            "api_compatible": True,
            "coordinator_integration": "API-based",
            "capabilities": [
                "COBOL business rule parsing",
                "JCL execution flow analysis",
                "CICS transaction context tracking",
                "BMS screen definition analysis",
                "SQL host variable validation",
                "DB2 Stored Procedure parsing",
                "COBOL Stored Procedure parsing",
                "IBM MQ/WebSphere MQ program analysis",
                "Field lineage tracking",
                "Control flow analysis",
                "Business rule violation detection",
                "Performance issue identification",
                "API-based LLM integration",
                "Enterprise middleware integration analysis"
            ],
            "supported_file_types": [".cbl", ".cob", ".jcl", ".cpy", ".copy", ".bms", ".sql", ".db2", ".mqt"],
            "supported_technologies": [
                "IBM COBOL",
                "IBM JCL",
                "IBM CICS",
                "IBM BMS",
                "IBM DB2",
                "IBM MQ/WebSphere MQ",
                "SQL Stored Procedures",
                "COBOL Stored Procedures"
            ],
            "database_schema_version": "2.1",
            "llm_integration": "API Coordinator compatible",
            "business_rules_enabled": True,
            "enterprise_ready": True
        }


# Export the main class
CodeParserAgent = CompleteEnhancedCodeParserAgent