"""
Enhanced Code Parser Agent - Part 1: Base Configuration and Imports
API-Compatible Enhanced Code Parser & Chunker with comprehensive business logic
Now inherits from BaseOpulenceAgent and uses LLM for complex pattern analysis
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

class CopybookLayoutType(Enum):
    SINGLE_RECORD = "single_record"
    MULTI_RECORD = "multi_record"
    CONDITIONAL_LAYOUT = "conditional_layout"
    REDEFINES_LAYOUT = "redefines_layout"
    OCCURS_LAYOUT = "occurs_layout"

class TransactionState:
    def __init__(self):
        self.input_received = False
        self.map_loaded = False
        self.file_opened = {}
        self.error_handlers = {}
        self.current_map = None
        
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
    line_number: Optional[int] = None

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
    business_context: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0  # LLM analysis confidence

@dataclass
class ControlFlowPath:
    """Represents a control flow execution path"""
    path_id: str
    entry_point: str
    exit_points: List[str]
    conditions: List[str]
    called_paragraphs: List[str]
    data_accessed: List[str]

@dataclass
class CopybookStructure:
    """Represents a copybook structure with layout information"""
    name: str
    layout_type: CopybookLayoutType
    record_layouts: List[Dict[str, Any]]
    field_hierarchy: Dict[str, Any]
    occurs_structures: List[Dict[str, Any]]
    redefines_structures: List[Dict[str, Any]]
    replacing_parameters: List[Dict[str, str]]
    business_domain: str
    complexity_score: int

@dataclass  
class MQStructure:
    """Represents IBM MQ program structure"""
    connection_type: str  # persistent, transient
    message_paradigm: str  # point_to_point, publish_subscribe, request_reply
    queue_operations: List[Dict[str, Any]]
    message_flow_patterns: List[str]
    transaction_scope: str
    error_handling_strategy: str
    performance_characteristics: Dict[str, Any]

@dataclass
class DB2ProcedureStructure:
    """Represents DB2 stored procedure structure"""
    procedure_name: str
    parameter_list: List[Dict[str, Any]]
    sql_operations: List[Dict[str, Any]]
    cursor_definitions: List[Dict[str, Any]]
    exception_handlers: List[Dict[str, Any]]
    transaction_control: Dict[str, Any]
    performance_hints: List[str]
    business_logic_complexity: int

# Business Validator Classes with enhanced LLM integration
class COBOLBusinessValidator:
    """Business rule validator for COBOL programs with LLM enhancement"""
    
    def __init__(self, llm_analyzer=None):
        self.llm_analyzer = llm_analyzer
    
    async def validate_structure(self, content: str) -> List[BusinessRuleViolation]:
        violations = []
        
        # Basic structural validation
        violations.extend(await self._validate_division_structure(content))
        violations.extend(await self._validate_data_division_rules(content))
        violations.extend(await self._validate_procedure_division_rules(content))
        
        # Enhanced LLM-based validation for complex patterns
        if self.llm_analyzer:
            llm_violations = await self._llm_validate_business_patterns(content)
            violations.extend(llm_violations)
        
        return violations
    
    async def _validate_division_structure(self, content: str) -> List[BusinessRuleViolation]:
        """Validate COBOL division structure"""
        violations = []
        divisions_found = {}
        
        # Division patterns
        division_patterns = {
            'IDENTIFICATION': r'^\s*IDENTIFICATION\s+DIVISION\s*\.',
            'ENVIRONMENT': r'^\s*ENVIRONMENT\s+DIVISION\s*\.',
            'DATA': r'^\s*DATA\s+DIVISION\s*\.',
            'PROCEDURE': r'^\s*PROCEDURE\s+DIVISION(?:\s+USING\s+[^\.]+)?\s*\.'
        }
        
        for div_name, pattern in division_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                divisions_found[div_name] = {
                    'position': match.start(),
                    'line': content[:match.start()].count('\n') + 1
                }
        
        # Check required divisions
        required_divisions = ['IDENTIFICATION', 'PROCEDURE']
        for req_div in required_divisions:
            if req_div not in divisions_found:
                violations.append(BusinessRuleViolation(
                    rule="MISSING_REQUIRED_DIVISION",
                    context=f"Missing required {req_div} DIVISION",
                    severity="ERROR",
                    line_number=1
                ))
        
        # Validate division order
        expected_order = ['IDENTIFICATION', 'ENVIRONMENT', 'DATA', 'PROCEDURE']
        found_order = [div for div in expected_order if div in divisions_found]
        
        for i in range(len(found_order) - 1):
            current = found_order[i]
            next_div = found_order[i + 1]
            
            if divisions_found[current]['position'] > divisions_found[next_div]['position']:
                violations.append(BusinessRuleViolation(
                    rule="DIVISION_ORDER_VIOLATION",
                    context=f"{current} DIVISION appears after {next_div} DIVISION",
                    severity="ERROR",
                    line_number=divisions_found[next_div]['line']
                ))
        
        return violations
    
    async def _validate_data_division_rules(self, content: str) -> List[BusinessRuleViolation]:
        """Validate data division business rules"""
        violations = []
        
        # Check for proper level number sequences
        data_items = re.findall(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)', content, re.MULTILINE | re.IGNORECASE)
        
        level_stack = []
        for level_str, name in data_items:
            try:
                level = int(level_str)
                
                # Validate level number ranges
                if level not in range(1, 50) and level not in [66, 77, 88]:
                    violations.append(BusinessRuleViolation(
                        rule="INVALID_LEVEL_NUMBER",
                        context=f"Invalid level number {level} for {name}",
                        severity="ERROR"
                    ))
                
                # Validate level 88 has a parent
                if level == 88:
                    if not level_stack or level_stack[-1] >= 88:
                        violations.append(BusinessRuleViolation(
                            rule="ORPHANED_CONDITION_NAME",
                            context=f"Level 88 item {name} has no valid parent",
                            severity="ERROR"
                        ))
                
                # Update level stack
                while level_stack and level_stack[-1] >= level:
                    level_stack.pop()
                level_stack.append(level)
                
            except ValueError:
                violations.append(BusinessRuleViolation(
                    rule="INVALID_LEVEL_FORMAT",
                    context=f"Invalid level number format: {level_str}",
                    severity="ERROR"
                ))
        
        return violations
    
    async def _validate_procedure_division_rules(self, content: str) -> List[BusinessRuleViolation]:
        """Validate procedure division business rules"""
        violations = []
        
        # Check for unreachable paragraphs
        paragraphs = re.findall(r'^\s*([A-Z0-9][A-Z0-9-]*)\s*\.\s*$', content, re.MULTILINE | re.IGNORECASE)
        perform_calls = re.findall(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)', content, re.IGNORECASE)
        
        unreachable_paragraphs = set(paragraphs) - set(perform_calls)
        if len(paragraphs) > 1:  # Exclude main paragraph
            for para in list(unreachable_paragraphs)[1:]:  # Skip first paragraph (main)
                violations.append(BusinessRuleViolation(
                    rule="UNREACHABLE_PARAGRAPH",
                    context=f"Paragraph {para} is never called",
                    severity="WARNING"
                ))
        
        return violations
    
    async def _llm_validate_business_patterns(self, content: str) -> List[BusinessRuleViolation]:
        """Use LLM to validate complex business patterns"""
        if not self.llm_analyzer:
            return []
        
        violations = []
        
        try:
            # LLM prompt for business pattern validation
            prompt = f"""
            Analyze this COBOL code for business rule violations and anti-patterns:
            
            {content[:2000]}...
            
            Check for:
            1. Inconsistent data naming conventions
            2. Missing error handling for critical operations
            3. Potential data integrity issues
            4. Performance anti-patterns
            5. Maintainability concerns
            
            Return findings as JSON array:
            [
                {{
                    "rule": "rule_name",
                    "context": "description",
                    "severity": "ERROR|WARNING|INFO",
                    "line_estimate": number
                }}
            ]
            """
            
            response = await self.llm_analyzer.call_api(prompt, {
                "temperature": 0.1,
                "max_tokens": 800
            })
            
            # Parse LLM response
            if '[' in response:
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                llm_findings = json.loads(response[json_start:json_end])
                
                for finding in llm_findings:
                    violations.append(BusinessRuleViolation(
                        rule=f"LLM_{finding.get('rule', 'UNKNOWN')}",
                        context=finding.get('context', ''),
                        severity=finding.get('severity', 'INFO'),
                        line_number=finding.get('line_estimate')
                    ))
        
        except Exception as e:
            logging.warning(f"LLM business validation failed: {e}")
        
        return violations

class JCLBusinessValidator:
    """Business rule validator for JCL with LLM enhancement"""
    
    def __init__(self, llm_analyzer=None):
        self.llm_analyzer = llm_analyzer
    
    async def validate_structure(self, content: str) -> List[BusinessRuleViolation]:
        violations = []
        
        # Check for JOB card
        if not re.search(r'^//\w+\s+JOB\s', content, re.MULTILINE):
            violations.append(BusinessRuleViolation(
                rule="MISSING_JOB_CARD",
                context="JCL must start with a JOB card",
                severity="ERROR",
                line_number=1
            ))
        
        # Check for at least one EXEC step
        if not re.search(r'^//\w+\s+EXEC\s', content, re.MULTILINE):
            violations.append(BusinessRuleViolation(
                rule="NO_EXEC_STEPS",
                context="JCL must have at least one EXEC statement",
                severity="ERROR"
            ))
        
        # Enhanced validation with LLM
        if self.llm_analyzer:
            llm_violations = await self._llm_validate_jcl_patterns(content)
            violations.extend(llm_violations)
        
        return violations
    
    async def _llm_validate_jcl_patterns(self, content: str) -> List[BusinessRuleViolation]:
        """Use LLM for complex JCL pattern validation"""
        if not self.llm_analyzer:
            return []
        
        violations = []
        
        try:
            prompt = f"""
            Analyze this JCL for execution flow issues and best practices:
            
            {content[:1500]}
            
            Look for:
            1. Missing or incorrect DD statements
            2. Step dependency issues
            3. Resource allocation problems
            4. Error handling gaps
            5. Performance bottlenecks
            
            Return as JSON array with rule violations.
            """
            
            response = await self.llm_analyzer.call_api(prompt, {
                "temperature": 0.1,
                "max_tokens": 600
            })
            
            # Process LLM response similar to COBOL validator
            # Implementation details...
            
        except Exception as e:
            logging.warning(f"LLM JCL validation failed: {e}")
        
        return violations

class CICSBusinessValidator:
    """Business rule validator for CICS programs"""
    
    def __init__(self, llm_analyzer=None):
        self.llm_analyzer = llm_analyzer
    
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
        
        # Validate command sequences with LLM
        if self.llm_analyzer and len(cics_commands) > 5:
            llm_violations = await self._llm_validate_cics_flow(content, cics_commands)
            violations.extend(llm_violations)
        
        return violations
    
    async def _llm_validate_cics_flow(self, content: str, commands: List[str]) -> List[BusinessRuleViolation]:
        """Use LLM to validate CICS transaction flow"""
        violations = []
        
        try:
            prompt = f"""
            Analyze this CICS transaction flow for proper command sequencing:
            
            Commands found: {', '.join(commands)}
            
            Code sample:
            {content[:1000]}
            
            Check for:
            1. Proper map handling sequence
            2. File operation safety
            3. Error condition handling
            4. Resource cleanup
            5. Transaction integrity
            
            Return violations as JSON array.
            """
            
            response = await self.llm_analyzer.call_api(prompt, {
                "temperature": 0.2,
                "max_tokens": 500
            })
            
            # Process response...
            
        except Exception as e:
            logging.warning(f"LLM CICS validation failed: {e}")
        
        return violations

class BMSBusinessValidator:
    """Business rule validator for BMS mapsets"""
    
    def __init__(self, llm_analyzer=None):
        self.llm_analyzer = llm_analyzer
    
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

class LLMContextManager:
    """Air-gapped LLM context window manager with intelligent chunking"""
    
    def __init__(self, max_tokens: int = 2048, reserve_tokens: int = 512):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.max_content_tokens = max_tokens - reserve_tokens
        self.tokenizer = None  # Always None in air-gapped environment
        self.logger = logging.getLogger(__name__)
        
        # Log that we're running in air-gapped mode
        self.logger.info("🔒 Running in air-gapped mode - using built-in token estimation")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Accurate token estimation for air-gapped environments
        Based on empirical analysis of various tokenizers
        """
        if not text:
            return 0
        
        # Enhanced character-based estimation with multiple factors
        # This is calibrated based on analysis of GPT tokenizers
        
        # Count different character types
        words = text.split()
        word_count = len(words)
        
        # Count characters by type
        alphanumeric = sum(1 for c in text if c.isalnum())
        whitespace = sum(1 for c in text if c.isspace())
        punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Special tokens and patterns
        newlines = text.count('\n')
        special_chars = sum(1 for c in text if c in '{}[]().,;:!?-_"\'')
        
        # Token estimation formula based on empirical data
        # Different content types have different token densities
        
        if self._is_code_content(text):
            # Code has more special characters and shorter tokens
            # Average ~3.2 characters per token for code
            base_tokens = len(text) / 3.2
        elif self._is_structured_data(text):
            # Structured data (JSON, XML, etc.) has predictable patterns
            # Average ~3.8 characters per token
            base_tokens = len(text) / 3.8
        else:
            # Natural language text
            # Average ~4.2 characters per token for English
            base_tokens = len(text) / 4.2
        
        # Adjustments based on content characteristics
        adjustments = 0
        
        # More punctuation = more tokens (each punct often separate)
        if punctuation / len(text) > 0.1:  # >10% punctuation
            adjustments += punctuation * 0.3
        
        # Many short words = more tokens
        if word_count > 0:
            avg_word_length = alphanumeric / word_count
            if avg_word_length < 4:  # Short words
                adjustments += word_count * 0.1
        
        # Newlines often create separate tokens
        adjustments += newlines * 0.5
        
        # Special programming tokens (common in code)
        programming_patterns = [
            '==', '!=', '<=', '>=', '->', '=>', '++', '--',
            '&&', '||', '**', '//', '/*', '*/', '<!--', '-->'
        ]
        for pattern in programming_patterns:
            adjustments += text.count(pattern) * 0.5
        
        # Calculate final estimate
        estimated_tokens = int(base_tokens + adjustments)
        
        # Apply bounds checking (minimum 1 token per 10 characters)
        min_tokens = max(1, len(text) // 10)
        max_tokens = len(text)  # Maximum 1 token per character
        
        return max(min_tokens, min(estimated_tokens, max_tokens))
    
    def _is_code_content(self, text: str) -> bool:
        """Detect if text is primarily code"""
        code_indicators = [
            '{', '}', '(', ')', '[', ']', ';', 
            'function', 'class', 'def', 'if', 'else', 'for', 'while',
            'IDENTIFICATION DIVISION', 'PROCEDURE DIVISION', 'DATA DIVISION',
            'EXEC CICS', 'EXEC SQL', 'PERFORM', 'MOVE', 'CALL',
            '//', 'EXEC', 'DD', 'JOB', 'STEP'
        ]
        
        text_upper = text.upper()
        indicator_count = sum(1 for indicator in code_indicators if indicator.upper() in text_upper)
        
        # If we find multiple code indicators, likely code
        return indicator_count >= 3 or any(indicator in text_upper for indicator in [
            'IDENTIFICATION DIVISION', 'PROCEDURE DIVISION', 'EXEC CICS', 'EXEC SQL'
        ])
    
    def _is_structured_data(self, text: str) -> bool:
        """Detect if text is structured data (JSON, XML, etc.)"""
        structured_indicators = [
            '{"', '"}', '":', '",', '[{', '}]',
            '<', '>', '<?xml', '</', '/>', '<!--',
            'PIC ', 'OCCURS ', 'REDEFINES ', '01 ', '05 '
        ]
        
        indicator_count = sum(1 for indicator in structured_indicators if indicator in text)
        return indicator_count >= 2
    
    def chunk_content_intelligently(self, content: str, chunk_type: str = 'code', overlap: int = 200) -> List[Dict[str, Any]]:
        """Intelligently chunk content based on type and structure"""
        if not content or not content.strip():
            return []
        
        estimated_tokens = self.estimate_tokens(content)
        
        if estimated_tokens <= self.max_content_tokens:
            return [{
                'content': content,
                'chunk_index': 0,
                'total_chunks': 1,
                'chunk_type': 'complete',
                'overlap_start': False,
                'overlap_end': False,
                'estimated_tokens': estimated_tokens
            }]
        
        # Choose appropriate chunking strategy
        try:
            if chunk_type == 'cobol':
                return self._chunk_cobol_intelligently(content, overlap)
            elif chunk_type == 'sql':
                return self._chunk_sql_intelligently(content, overlap)
            elif chunk_type == 'jcl':
                return self._chunk_jcl_intelligently(content, overlap)
            else:
                return self._chunk_generic_with_context(content, overlap)
        except Exception as e:
            self.logger.warning(f"Intelligent chunking failed: {e}, using simple chunking")
            return self._chunk_simple_safe(content)
    
    def _chunk_cobol_intelligently(self, content: str, overlap: int) -> List[Dict[str, Any]]:
        """Chunk COBOL code preserving logical boundaries"""
        chunks = []
        
        # COBOL division boundaries (most important)
        division_patterns = [
            (r'^\s*IDENTIFICATION\s+DIVISION', 'identification'),
            (r'^\s*ENVIRONMENT\s+DIVISION', 'environment'),
            (r'^\s*DATA\s+DIVISION', 'data'),
            (r'^\s*PROCEDURE\s+DIVISION', 'procedure')
        ]
        
        # COBOL section boundaries
        section_patterns = [
            (r'^\s*WORKING-STORAGE\s+SECTION', 'working-storage'),
            (r'^\s*FILE\s+SECTION', 'file'),
            (r'^\s*LINKAGE\s+SECTION', 'linkage'),
            (r'^\s*[A-Z][A-Z0-9-]*\s+SECTION', 'named-section')
        ]
        
        # Find all boundaries
        boundaries = []
        
        try:
            import re
            
            # Add division boundaries
            for pattern, div_type in division_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                    boundaries.append({
                        'position': match.start(),
                        'type': 'division',
                        'name': div_type,
                        'priority': 1  # Highest priority
                    })
            
            # Add section boundaries
            for pattern, sect_type in section_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                    boundaries.append({
                        'position': match.start(),
                        'type': 'section',
                        'name': sect_type,
                        'priority': 2
                    })
            
        except Exception as e:
            self.logger.warning(f"COBOL pattern matching failed: {e}")
            return self._chunk_by_lines(content, overlap, 'cobol')
        
        if not boundaries:
            return self._chunk_by_lines(content, overlap, 'cobol')
        
        # Sort by position
        boundaries.sort(key=lambda x: x['position'])
        
        return self._create_chunks_from_boundaries(content, boundaries, overlap, 'cobol_logical')
    
    def _chunk_sql_intelligently(self, content: str, overlap: int) -> List[Dict[str, Any]]:
        """Chunk SQL code preserving statement boundaries"""
        try:
            import re
            
            # SQL statement boundaries
            sql_patterns = [
                (r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE', 'create_procedure'),
                (r'BEGIN\s*(?:ATOMIC)?', 'begin_block'),
                (r'DECLARE\s+\w+\s+CURSOR', 'cursor_declare'),
                (r'DECLARE\s+\w+\s+HANDLER', 'handler_declare'),
                (r'IF\s+.*?\s+THEN', 'if_statement'),
                (r'WHILE\s+.*?\s+DO', 'while_loop'),
                (r'FOR\s+.*?\s+DO', 'for_loop')
            ]
            
            boundaries = []
            for pattern, stmt_type in sql_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    boundaries.append({
                        'position': match.start(),
                        'type': 'sql_statement',
                        'name': stmt_type,
                        'priority': 1
                    })
            
            boundaries.sort(key=lambda x: x['position'])
            return self._create_chunks_from_boundaries(content, boundaries, overlap, 'sql_logical')
            
        except Exception as e:
            self.logger.warning(f"SQL pattern matching failed: {e}")
            return self._chunk_by_lines(content, overlap, 'sql')
    
    def _chunk_jcl_intelligently(self, content: str, overlap: int) -> List[Dict[str, Any]]:
        """Chunk JCL preserving job step boundaries"""
        try:
            import re
            
            # JCL statement boundaries
            jcl_patterns = [
                (r'^//\w+\s+JOB\s', 'job_card'),
                (r'^//\w+\s+EXEC\s', 'exec_step'),
                (r'^//\w+\s+DD\s', 'dd_statement'),
                (r'^//\s+IF\s', 'if_statement'),
                (r'^//\s+ENDIF', 'endif_statement')
            ]
            
            boundaries = []
            for pattern, stmt_type in jcl_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                    boundaries.append({
                        'position': match.start(),
                        'type': 'jcl_statement',
                        'name': stmt_type,
                        'priority': 1
                    })
            
            boundaries.sort(key=lambda x: x['position'])
            return self._create_chunks_from_boundaries(content, boundaries, overlap, 'jcl_logical')
            
        except Exception as e:
            self.logger.warning(f"JCL pattern matching failed: {e}")
            return self._chunk_by_lines(content, overlap, 'jcl')
    
    def _create_chunks_from_boundaries(self, content: str, boundaries: List[Dict], overlap: int, chunk_type: str) -> List[Dict[str, Any]]:
        """Create chunks from boundary positions"""
        if not boundaries:
            return self._chunk_by_lines(content, overlap, chunk_type)
        
        chunks = []
        chunk_index = 0
        current_start = 0
        
        for i, boundary in enumerate(boundaries):
            # Look ahead to find good chunk end
            chunk_end = len(content)
            
            # Find next boundary for chunk end
            for j in range(i + 1, len(boundaries)):
                next_boundary = boundaries[j]
                potential_chunk = content[current_start:next_boundary['position']]
                
                if self.estimate_tokens(potential_chunk) > self.max_content_tokens:
                    # Current chunk would be too big, end at current boundary
                    chunk_end = boundary['position']
                    break
                else:
                    # Can include this boundary, keep looking
                    chunk_end = next_boundary['position']
            
            # Create chunk if we have content
            if current_start < chunk_end:
                chunk_content = content[current_start:chunk_end]
                
                if chunk_content.strip():
                    chunks.append({
                        'content': chunk_content,
                        'chunk_index': chunk_index,
                        'chunk_type': chunk_type,
                        'boundary_info': boundary,
                        'overlap_start': chunk_index > 0,
                        'overlap_end': chunk_end < len(content),
                        'estimated_tokens': self.estimate_tokens(chunk_content)
                    })
                    chunk_index += 1
                
                # Start next chunk with small overlap
                current_start = max(0, chunk_end - overlap)
        
        # Update total_chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks if chunks else self._chunk_by_lines(content, overlap, chunk_type)
    
    def _chunk_by_lines(self, content: str, overlap: int, chunk_type: str) -> List[Dict[str, Any]]:
        """Fallback chunking by lines when pattern matching fails"""
        chunks = []
        lines = content.split('\n')
        
        # Determine lines per chunk based on estimated token density
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 50
        estimated_tokens_per_line = self.estimate_tokens('x' * int(avg_line_length))
        lines_per_chunk = max(10, self.max_content_tokens // max(1, estimated_tokens_per_line))
        
        chunk_index = 0
        overlap_lines = overlap // int(avg_line_length) if avg_line_length > 0 else 5
        
        for i in range(0, len(lines), lines_per_chunk - overlap_lines):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():
                chunks.append({
                    'content': chunk_content,
                    'chunk_index': chunk_index,
                    'chunk_type': f'{chunk_type}_lines',
                    'overlap_start': i > 0,
                    'overlap_end': i + lines_per_chunk < len(lines),
                    'estimated_tokens': self.estimate_tokens(chunk_content),
                    'line_range': (i, i + len(chunk_lines))
                })
                chunk_index += 1
        
        # Update total_chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_simple_safe(self, content: str) -> List[Dict[str, Any]]:
        """Ultra-simple safe chunking as last resort"""
        chunks = []
        chunk_size = self.max_content_tokens * 3  # Assume ~3 chars per token
        overlap = 100  # Small fixed overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            
            if chunk_content.strip():
                chunks.append({
                    'content': chunk_content,
                    'chunk_index': len(chunks),
                    'total_chunks': 0,  # Will update below
                    'chunk_type': 'simple_safe',
                    'overlap_start': i > 0,
                    'overlap_end': i + chunk_size < len(content),
                    'estimated_tokens': self.estimate_tokens(chunk_content)
                })
        
        # Update total_chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_generic_with_context(self, content: str, overlap: int) -> List[Dict[str, Any]]:
        """Generic chunking with context preservation"""
        return self._chunk_by_lines(content, overlap, 'generic')
    
class CodeParserAgent(BaseOpulenceAgent):
    """
    Enhanced Code Parser Agent that inherits from BaseOpulenceAgent
    Provides comprehensive parsing for mainframe technologies with LLM integration
    """
    
    def __init__(self, coordinator, agent_type: str = "enhanced_code_parser", 
                 db_path: str = "opulence_data.db", gpu_id: int = 0, **kwargs):
        # Initialize base agent
        super().__init__(coordinator, agent_type, db_path, gpu_id)
        
        # Agent-specific configuration
        self.api_params.update({
            "max_tokens": 1500,  # Longer responses for complex analysis
            "temperature": 0.1,   # Low temperature for consistent parsing
            "top_p": 0.9
        })
        
        # Thread safety
        self._engine_lock = asyncio.Lock()
        self._processed_files = set()  # Duplicate prevention
        
        # Business Rule Validators with LLM integration
        self.business_validators = {
            'cobol': COBOLBusinessValidator(self),
            'jcl': JCLBusinessValidator(self),
            'cics': CICSBusinessValidator(self),
            'bms': BMSBusinessValidator(self)
        }
        
        # Initialize comprehensive pattern libraries
        self._init_cobol_patterns()
        self._init_jcl_patterns()
        self._init_cics_patterns()
        self._init_bms_patterns()
        self._init_sql_patterns()
        self._init_mq_patterns()
        self._init_db2_patterns()
        self._init_copybook_patterns()
        
        # Initialize database with enhanced schema
        self._init_enhanced_database()
        
        self.context_manager = LLMContextManager(max_tokens=2048, reserve_tokens=512)
        
        self.logger.info(f"🚀 Enhanced Code Parser Agent initialized (AIR-GAPPED) with API coordinator")
        self.logger.info(f"🔒 Running in air-gapped mode - no external network dependencies")


    def _init_cobol_patterns(self):
        """Initialize comprehensive COBOL pattern library"""
        self.cobol_patterns = {
            # Basic identification with stricter boundaries
            'program_id': re.compile(r'^\s*PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'author': re.compile(r'^\s*AUTHOR\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'date_written': re.compile(r'^\s*DATE-WRITTEN\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            'date_compiled': re.compile(r'^\s*DATE-COMPILED\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            'installation': re.compile(r'^\s*INSTALLATION\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            'security': re.compile(r'^\s*SECURITY\s*\.\s*(.*?)\.', re.IGNORECASE | re.MULTILINE),
            
            # Divisions with proper boundaries and order enforcement
            'identification_division': re.compile(r'^\s*IDENTIFICATION\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'environment_division': re.compile(r'^\s*ENVIRONMENT\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'data_division': re.compile(r'^\s*DATA\s+DIVISION\s*\.', re.IGNORECASE | re.MULTILINE),
            'procedure_division': re.compile(r'^\s*PROCEDURE\s+DIVISION(?:\s+USING\s+([^\.]+))?\s*\.', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced sections with proper hierarchy
            'configuration_section': re.compile(r'^\s*CONFIGURATION\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'input_output_section': re.compile(r'^\s*INPUT-OUTPUT\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'file_control': re.compile(r'^\s*FILE-CONTROL\s*\.', re.IGNORECASE | re.MULTILINE),
            'io_control': re.compile(r'^\s*I-O-CONTROL\s*\.', re.IGNORECASE | re.MULTILINE),
            'working_storage': re.compile(r'^\s*WORKING-STORAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'file_section': re.compile(r'^\s*FILE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'linkage_section': re.compile(r'^\s*LINKAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'local_storage': re.compile(r'^\s*LOCAL-STORAGE\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'report_section': re.compile(r'^\s*REPORT\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'screen_section': re.compile(r'^\s*SCREEN\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'communication_section': re.compile(r'^\s*COMMUNICATION\s+SECTION\s*\.', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
     
            
            # Enhanced paragraph detection
            'paragraph': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
       
            'paragraph_with_section': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
        
            # Comprehensive PERFORM patterns with enhanced business logic
            'perform_simple': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s*(?:\.|$)', re.IGNORECASE),
            'perform_until': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?UNTIL\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_varying': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),

            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
  
            'group_item': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?VARYING\s+(.*?)(?=\s+END-PERFORM|\.|$)', re.IGNORECASE | re.DOTALL),
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
        
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
     
            'perform_thru': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'perform_times': re.compile(r'\bPERFORM\s+(?:([A-Z0-9][A-Z0-9-]*)\s+)?(\d+)\s+TIMES', re.IGNORECASE),
            'perform_inline': re.compile(r'\bPERFORM\s*(.*?)\s*END-PERFORM', re.IGNORECASE | re.DOTALL),
            'perform_test': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+TEST\s+(BEFORE|AFTER)', re.IGNORECASE),
            'perform_with_debug': re.compile(r'\bPERFORM\s+.*?\s+WITH\s+DEBUGGING\s+MODE', re.IGNORECASE),
            
            # Enhanced control flow with proper nesting
            'if_statement': re.compile(r'\bIF\s+(.*?)(?=\s+THEN|\s+(?:NEXT|END)|$)', re.IGNORECASE),
            'if_then': re.compile(r'\bIF\s+.*?\s+THEN\b', re.IGNORECASE),
            'if_then_else': re.compile(r'\bIF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END-IF', re.IGNORECASE | re.DOTALL),
            'else_clause': re.compile(r'\bELSE\b', re.IGNORECASE),
            'end_if': re.compile(r'\bEND-IF\b', re.IGNORECASE),
            'evaluate': re.compile(r'\bEVALUATE\s+(.*?)(?=\s|$)', re.IGNORECASE),
            'evaluate_block': re.compile(r'\bEVALUATE\s+(.*?)\s+END-EVALUATE', re.IGNORECASE | re.DOTALL),
            'when_clause': re.compile(r'^\s*WHEN\s+([^\.]+)', re.IGNORECASE | re.MULTILINE),
            'when_other': re.compile(r'^\s*WHEN\s+OTHER', re.IGNORECASE | re.MULTILINE),
            'end_evaluate': re.compile(r'\bEND-EVALUATE\b', re.IGNORECASE),
            
            # Enhanced data definitions with comprehensive level validation
            'data_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*|FILLER)(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'data_item_with_level': re.compile(r'^\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|66|77|88)\s+([A-Z][A-Z0-9-]*|FILLER)', re.MULTILINE | re.IGNORECASE),
            'filler_item': re.compile(r'^\s*(\d+)\s+FILLER(?:\s+(.*?))?\.?\s*$', re.MULTILINE | re.IGNORECASE),
            'group_item': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s*\.?\s*$', re.MULTILINE | re.IGNORECASE),
            
            # Enhanced PIC and data type patterns
            'pic_clause': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', re.IGNORECASE),
            'pic_with_edit': re.compile(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+(?:\s*EDIT\s+[^\.]*)?)', re.IGNORECASE),
            'usage_clause': re.compile(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER|OBJECT\s+REFERENCE|FUNCTION-POINTER|PROCEDURE-POINTER)', re.IGNORECASE),
            'comp_usage': re.compile(r'\b(?:COMP|COMP-1|COMP-2|COMP-3|COMP-4|COMP-5|COMPUTATIONAL|COMPUTATIONAL-1|COMPUTATIONAL-2|COMPUTATIONAL-3|COMPUTATIONAL-4|COMPUTATIONAL-5|BINARY|PACKED-DECIMAL)\b', re.IGNORECASE),
            'value_clause': re.compile(r'VALUE\s+(?:IS\s+)?([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_all': re.compile(r'VALUE\s+(?:IS\s+)?ALL\s+([\'"][^\']*[\'"]|\S+)', re.IGNORECASE),
            'value_space': re.compile(r'VALUE\s+(?:IS\s+)?(?:SPACE|SPACES)', re.IGNORECASE),
            'value_zero': re.compile(r'VALUE\s+(?:IS\s+)?(?:ZERO|ZEROS|ZEROES)', re.IGNORECASE),
            'value_quote': re.compile(r'VALUE\s+(?:IS\s+)?(?:QUOTE|QUOTES)', re.IGNORECASE),
            'value_high_low': re.compile(r'VALUE\s+(?:IS\s+)?(?:HIGH-VALUE|HIGH-VALUES|LOW-VALUE|LOW-VALUES)', re.IGNORECASE),
            
            # Enhanced OCCURS patterns with comprehensive business rules
            'occurs_clause': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:ASCENDING|DESCENDING)?(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_depending': re.compile(r'OCCURS\s+\d+\s+TO\s+\d+\s+TIMES\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'occurs_indexed': re.compile(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'occurs_key': re.compile(r'(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            
            # Enhanced REDEFINES patterns with complex structures
            'redefines': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'renames': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)\s+(?:THROUGH|THRU)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'renames_simple': re.compile(r'^\s*66\s+([A-Z][A-Z0-9-]*)\s+RENAMES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Comprehensive file operations with enhanced context
            'select_statement': re.compile(r'^\s*SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)(?:\s+ORGANIZATION\s+(?:IS\s+)?([A-Z]+))?(?:\s+ACCESS\s+(?:MODE\s+)?(?:IS\s+)?([A-Z]+))?(?:\s+FILE\s+STATUS\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE | re.MULTILINE),
            'fd_statement': re.compile(r'^\s*FD\s+([A-Z][A-Z0-9-]*)(?:\s+(.*?))?(?=\n\s*\d|\nFD|\nSD|\nRD|$)', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'sd_statement': re.compile(r'^\s*SD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'rd_statement': re.compile(r'^\s*RD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'cd_statement': re.compile(r'^\s*CD\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Enhanced file operation statements
            'open_statement': re.compile(r'\bOPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'close_statement': re.compile(r'\bCLOSE\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'read_statement': re.compile(r'\bREAD\s+([A-Z][A-Z0-9-]*)(?:\s+(?:INTO\s+([A-Z][A-Z0-9-]*)|NEXT\s+RECORD|KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*)))?', re.IGNORECASE),
            'write_statement': re.compile(r'\bWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'rewrite_statement': re.compile(r'\bREWRITE\s+([A-Z][A-Z0-9-]*)(?:\s+FROM\s+([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            'delete_statement': re.compile(r'\bDELETE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'start_statement': re.compile(r'\bSTART\s+([A-Z][A-Z0-9-]*)(?:\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*))?', re.IGNORECASE),
            
            # Enhanced SQL blocks with comprehensive host variable detection
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'sql_host_var': re.compile(r':([A-Z][A-Z0-9-]*(?:-[A-Z0-9]+)*)', re.IGNORECASE),
            'sql_indicator_var': re.compile(r':([A-Z][A-Z0-9-]*)\s+INDICATOR\s+:([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_include': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_declare_cursor': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9-]*)\s+CURSOR(?:\s+WITH\s+(?:HOLD|RETURN))?(?:\s+FOR\s+(.*?))?(?=\s+END-EXEC)', re.IGNORECASE | re.DOTALL),
            'sql_whenever': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+(SQLWARNING|SQLERROR|NOT\s+FOUND)\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|GO\s+TO\s+[A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_connect': re.compile(r'EXEC\s+SQL\s+CONNECT\s+(?:TO\s+)?([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'sql_commit': re.compile(r'EXEC\s+SQL\s+COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'sql_rollback': re.compile(r'EXEC\s+SQL\s+ROLLBACK(?:\s+WORK)?', re.IGNORECASE),
            
            # Enhanced COPY statements with comprehensive replacement patterns
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)(?:\s+(?:OF|IN)\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?\.', re.IGNORECASE | re.DOTALL),
            'copy_replacing': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+REPLACING\s+(.*?)\.', re.IGNORECASE | re.DOTALL),
            'copy_in': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+(?:IN|OF)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE),
            'copy_suppress': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)\s+SUPPRESS', re.IGNORECASE),
            'replacing_clause': re.compile(r'==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_leading': re.compile(r'LEADING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            'replacing_trailing': re.compile(r'TRAILING\s+==([^=]*)==\s+BY\s+==([^=]*)==', re.IGNORECASE),
            
            # Enhanced error handling patterns
            'on_size_error': re.compile(r'\bON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'not_on_size_error': re.compile(r'\bNOT\s+ON\s+SIZE\s+ERROR\b', re.IGNORECASE),
            'at_end': re.compile(r'\bAT\s+END\b', re.IGNORECASE),
            'not_at_end': re.compile(r'\bNOT\s+AT\s+END\b', re.IGNORECASE),
            'invalid_key': re.compile(r'\bINVALID\s+KEY\b', re.IGNORECASE),
            'not_invalid_key': re.compile(r'\bNOT\s+INVALID\s+KEY\b', re.IGNORECASE),
            'on_overflow': re.compile(r'\bON\s+OVERFLOW\b', re.IGNORECASE),
            'not_on_overflow': re.compile(r'\bNOT\s+ON\s+OVERFLOW\b', re.IGNORECASE),
            'on_exception': re.compile(r'\bON\s+EXCEPTION\b', re.IGNORECASE),
            'not_on_exception': re.compile(r'\bNOT\s+ON\s+EXCEPTION\b', re.IGNORECASE),
        }

    def _init_copybook_patterns(self):
        """Initialize comprehensive copybook parsing patterns"""
        self.copybook_patterns = {
            # Multi-layout copybook patterns
            'layout_indicator': re.compile(r'^\s*88\s+([A-Z][A-Z0-9-]*-(?:LAYOUT|TYPE|FORMAT|REC))\s+VALUE', re.IGNORECASE | re.MULTILINE),
            'record_type_field': re.compile(r'^\s*\d+\s+([A-Z][A-Z0-9-]*-(?:TYPE|CODE|ID))\s+PIC', re.IGNORECASE | re.MULTILINE),
            'conditional_field': re.compile(r'^\s*\d+\s+([A-Z][A-Z0-9-]*)\s+(?:PIC|OCCURS).*?DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # Multi-filler patterns for alignment and padding
            'multi_filler': re.compile(r'^\s*\d+\s+FILLER\s+PIC\s+X\((\d+)\)', re.IGNORECASE | re.MULTILINE),
            'filler_alignment': re.compile(r'^\s*\d+\s+FILLER\s+PIC\s+([X9S]+(?:\(\d+\))?)', re.IGNORECASE | re.MULTILINE),
            'sync_alignment': re.compile(r'\bSYNCHRONIZED\b|\bSYNC\b', re.IGNORECASE),
            
            # Complex OCCURS patterns
            'occurs_complex': re.compile(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES(?:\s+DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*))?(?:\s+INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*(?:(?:ASCENDING|DESCENDING)\s+KEY\s+(?:IS\s+)?([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s*', re.IGNORECASE),
            'occurs_nested': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+OCCURS\s+(\d+)', re.IGNORECASE | re.MULTILINE),
            'occurs_multi_dimension': re.compile(r'OCCURS\s+\d+.*?OCCURS\s+\d+', re.IGNORECASE | re.DOTALL),
            
            # Complex REDEFINES patterns
            'redefines_complex': re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+REDEFINES\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'redefines_chain': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*(?:\s+REDEFINES\s+[A-Z][A-Z0-9-]*)*)', re.IGNORECASE),
            'redefines_with_occurs': re.compile(r'REDEFINES\s+([A-Z][A-Z0-9-]*)\s+OCCURS', re.IGNORECASE),
            
            # Enhanced REPLACING patterns for copybook customization
            'replacing_variable': re.compile(r'==([A-Z][A-Z0-9-]*)==', re.IGNORECASE),
            'replacing_literal': re.compile(r'==([\'"][^\']*[\'"])==', re.IGNORECASE),
            'replacing_numeric': re.compile(r'==(\d+)==', re.IGNORECASE),
            'replacing_pic': re.compile(r'==PIC\s+([X9S\(\)]+)==', re.IGNORECASE),
            'replacing_usage': re.compile(r'==USAGE\s+([A-Z-]+)==', re.IGNORECASE),
            
            # Copybook business domain patterns
            'customer_record': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:CUSTOMER|CLIENT|CUST)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'account_record': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:ACCOUNT|ACCT)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'transaction_record': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:TRANSACTION|TRANS|TXN)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'address_record': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:ADDRESS|ADDR)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'date_field': re.compile(r'^\s*\d+\s+([A-Z][A-Z0-9-]*(?:DATE|DT)[A-Z0-9-]*)\s+PIC', re.IGNORECASE | re.MULTILINE),
            'amount_field': re.compile(r'^\s*\d+\s+([A-Z][A-Z0-9-]*(?:AMOUNT|AMT|TOTAL|SUM)[A-Z0-9-]*)\s+PIC', re.IGNORECASE | re.MULTILINE),
        }

    def _init_mq_patterns(self):
        """Initialize comprehensive IBM MQ/WebSphere MQ patterns"""
        self.mq_patterns = {
            # MQ API Call patterns with comprehensive parameter detection
            'mq_mqconn': re.compile(r'CALL\s+[\'"]MQCONN[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqconnx': re.compile(r'CALL\s+[\'"]MQCONNX[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqdisc': re.compile(r'CALL\s+[\'"]MQDISC[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqopen': re.compile(r'CALL\s+[\'"]MQOPEN[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqclose': re.compile(r'CALL\s+[\'"]MQCLOSE[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqput': re.compile(r'CALL\s+[\'"]MQPUT[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqput1': re.compile(r'CALL\s+[\'"]MQPUT1[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqget': re.compile(r'CALL\s+[\'"]MQGET[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqbegin': re.compile(r'CALL\s+[\'"]MQBEGIN[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqcmit': re.compile(r'CALL\s+[\'"]MQCMIT[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqback': re.compile(r'CALL\s+[\'"]MQBACK[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqinq': re.compile(r'CALL\s+[\'"]MQINQ[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqset': re.compile(r'CALL\s+[\'"]MQSET[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqsub': re.compile(r'CALL\s+[\'"]MQSUB[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqsubrq': re.compile(r'CALL\s+[\'"]MQSUBRQ[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'mq_mqcb': re.compile(r'CALL\s+[\'"]MQCB[\'"](?:\s+USING\s+(.*?))?', re.IGNORECASE),
            
            # MQ Data Structure patterns with enhanced field detection
            'mq_message_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQMD|MSG-DESC)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_object_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQOD|OBJ-DESC)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_put_options': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQPMO|PUT-OPT)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_get_options': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQGMO|GET-OPT)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_subscription_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQSD|SUB-DESC)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_connection_options': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQCNO|CONN-OPT)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_message_handle': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQMH|MSG-HDL)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_property_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQPD|PROP-DESC)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            'mq_callback_descriptor': re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*(?:MQCBD|CB-DESC)[A-Z0-9-]*)', re.IGNORECASE | re.MULTILINE),
            
            # MQ Message Format and Type patterns
            'mq_message_format': re.compile(r'(?:FORMAT|MSG-FORMAT)\s+(?:VALUE\s+)?[\'"]([A-Z0-9\s]+)[\'"]', re.IGNORECASE),
            'mq_message_type': re.compile(r'(?:MSG-TYPE|MESSAGE-TYPE)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_persistence': re.compile(r'(?:PERSISTENCE|MSG-PERSISTENCE)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_priority': re.compile(r'(?:PRIORITY|MSG-PRIORITY)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_expiry': re.compile(r'(?:EXPIRY|MSG-EXPIRY)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            
            # MQ Queue and Topic patterns
            'mq_queue_name': re.compile(r'(?:QUEUE-NAME|Q-NAME)\s+(?:VALUE\s+)?[\'"]([A-Z0-9.\-_/]+)[\'"]', re.IGNORECASE),
            'mq_queue_manager': re.compile(r'(?:QUEUE-MANAGER|Q-MGR|QMGR)\s+(?:VALUE\s+)?[\'"]([A-Z0-9.\-_]+)[\'"]', re.IGNORECASE),
            'mq_topic_string': re.compile(r'(?:TOPIC-STRING|TOPIC)\s+(?:VALUE\s+)?[\'"]([A-Z0-9.\-_/]+)[\'"]', re.IGNORECASE),
            'mq_channel_name': re.compile(r'(?:CHANNEL-NAME|CHANNEL)\s+(?:VALUE\s+)?[\'"]([A-Z0-9.\-_]+)[\'"]', re.IGNORECASE),
            'mq_connection_name': re.compile(r'(?:CONNECTION-NAME|CONN-NAME)\s+(?:VALUE\s+)?[\'"]([A-Z0-9.\-_():]+)[\'"]', re.IGNORECASE),
            
            # MQ Error Handling patterns
            'mq_completion_code': re.compile(r'(?:COMPLETION-CODE|COMP-CODE)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_reason_code': re.compile(r'(?:REASON-CODE|REASON)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_error_check': re.compile(r'IF\s+([A-Z][A-Z0-9-]*(?:COMP|REASON)[A-Z0-9-]*)\s*(?:=|EQUAL)', re.IGNORECASE),
            'mq_rc_check': re.compile(r'IF\s+([A-Z][A-Z0-9-]*RC[A-Z0-9-]*)\s*(?:=|NOT\s*=|EQUAL|NOT\s+EQUAL)', re.IGNORECASE),
            
            # MQ Transaction patterns
            'mq_syncpoint': re.compile(r'(?:SYNCPOINT|SYNC-POINT)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_unit_of_work': re.compile(r'(?:UNIT-OF-WORK|UOW)', re.IGNORECASE),
            'mq_commit_check': re.compile(r'(?:MQCMIT|COMMIT)', re.IGNORECASE),
            'mq_backout_check': re.compile(r'(?:MQBACK|BACKOUT|ROLLBACK)', re.IGNORECASE),
            
            # MQ Performance and Monitoring patterns
            'mq_wait_interval': re.compile(r'(?:WAIT-INTERVAL|WAIT-TIME)\s+(?:VALUE\s+)?([0-9]+)', re.IGNORECASE),
            'mq_get_wait': re.compile(r'MQGET.*?WAIT-INTERVAL', re.IGNORECASE | re.DOTALL),
            'mq_browse_message': re.compile(r'(?:BROWSE-FIRST|BROWSE-NEXT)', re.IGNORECASE),
            'mq_message_id': re.compile(r'(?:MESSAGE-ID|MSG-ID)', re.IGNORECASE),
            'mq_correlation_id': re.compile(r'(?:CORRELATION-ID|CORREL-ID)', re.IGNORECASE),
        }
    def _init_db2_patterns(self):
        """Initialize comprehensive DB2 stored procedure patterns"""
        self.db2_patterns = {
            # DB2 Stored Procedure Creation patterns
            'db2_create_procedure': re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+([A-Z][A-Z0-9_]*\.?[A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_procedure_body': re.compile(r'BEGIN\s+(.*?)\s+END', re.IGNORECASE | re.DOTALL),
            'db2_language_sql': re.compile(r'LANGUAGE\s+SQL', re.IGNORECASE),
            'db2_parameter_style': re.compile(r'PARAMETER\s+STYLE\s+(SQL|GENERAL|JAVA|DB2DARI)', re.IGNORECASE),
            'db2_deterministic': re.compile(r'\b(DETERMINISTIC|NOT\s+DETERMINISTIC)\b', re.IGNORECASE),
            'db2_reads_sql': re.compile(r'\b(READS\s+SQL\s+DATA|MODIFIES\s+SQL\s+DATA|NO\s+SQL|CONTAINS\s+SQL)\b', re.IGNORECASE),
            'db2_external_action': re.compile(r'\b(EXTERNAL\s+ACTION|NO\s+EXTERNAL\s+ACTION)\b', re.IGNORECASE),
            'db2_fenced': re.compile(r'\b(FENCED|NOT\s+FENCED)\b', re.IGNORECASE),
            'db2_security': re.compile(r'SECURITY\s+(DEFINER|INVOKER)', re.IGNORECASE),
            
            # DB2 Parameter patterns with comprehensive type detection
            'db2_parameter': re.compile(r'(IN|OUT|INOUT)\s+([A-Z][A-Z0-9_]*)\s+(VARCHAR\(\d+\)|CHAR\(\d+\)|INTEGER|DECIMAL\(\d+,\d+\)|DATE|TIME|TIMESTAMP|BLOB|CLOB|SMALLINT|BIGINT|REAL|DOUBLE|NUMERIC\(\d+,?\d*\))(?:\s+DEFAULT\s+([^,)]+))?', re.IGNORECASE),
            'db2_parameter_list': re.compile(r'\((.*?)\)', re.IGNORECASE | re.DOTALL),
            'db2_return_parameter': re.compile(r'RETURNS\s+([A-Z][A-Z0-9_]*)\s+(VARCHAR\(\d+\)|INTEGER|DECIMAL\(\d+,\d+\)|DATE|TIMESTAMP)', re.IGNORECASE),
            
            # DB2 Variable Declaration patterns
            'db2_declare_variable': re.compile(r'DECLARE\s+([A-Z][A-Z0-9_]*)\s+(VARCHAR\(\d+\)|CHAR\(\d+\)|INTEGER|DECIMAL\(\d+,\d+\)|DATE|TIME|TIMESTAMP|BLOB|CLOB|SMALLINT|BIGINT|REAL|DOUBLE)(?:\s+DEFAULT\s+([^;]+))?', re.IGNORECASE),
            'db2_declare_condition': re.compile(r'DECLARE\s+([A-Z][A-Z0-9_]*)\s+CONDITION\s+FOR\s+(SQLSTATE\s+[\'"][0-9A-Z]{5}[\'"]|SQLCODE\s+[0-9-]+)', re.IGNORECASE),
            'db2_declare_cursor': re.compile(r'DECLARE\s+([A-Z][A-Z0-9_]*)\s+CURSOR(?:\s+WITH\s+(HOLD|RETURN))?\s+FOR\s+(.*?)(?=;|DECLARE|BEGIN)', re.IGNORECASE | re.DOTALL),
            'db2_declare_handler': re.compile(r'DECLARE\s+(CONTINUE|EXIT|UNDO)\s+HANDLER\s+FOR\s+(.*?)\s+(.*?)(?=;|DECLARE|BEGIN)', re.IGNORECASE | re.DOTALL),
            'db2_declare_temporary_table': re.compile(r'DECLARE\s+GLOBAL\s+TEMPORARY\s+TABLE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            
            # DB2 SQL Statement patterns within procedures
            'db2_select_into': re.compile(r'SELECT\s+(.*?)\s+INTO\s+(.*?)\s+FROM\s+(.*?)(?=;|$)', re.IGNORECASE | re.DOTALL),
            'db2_insert_statement': re.compile(r'INSERT\s+INTO\s+([A-Z][A-Z0-9_]*\.?[A-Z][A-Z0-9_]*)\s*\((.*?)\)\s*VALUES\s*\((.*?)\)', re.IGNORECASE | re.DOTALL),
            'db2_update_statement': re.compile(r'UPDATE\s+([A-Z][A-Z0-9_]*\.?[A-Z][A-Z0-9_]*)\s+SET\s+(.*?)(?:\s+WHERE\s+(.*?))?(?=;|$)', re.IGNORECASE | re.DOTALL),
            'db2_delete_statement': re.compile(r'DELETE\s+FROM\s+([A-Z][A-Z0-9_]*\.?[A-Z][A-Z0-9_]*)(?:\s+WHERE\s+(.*?))?(?=;|$)', re.IGNORECASE | re.DOTALL),
            'db2_merge_statement': re.compile(r'MERGE\s+INTO\s+([A-Z][A-Z0-9_]*\.?[A-Z][A-Z0-9_]*)', re.IGNORECASE),
            
            # DB2 Cursor Operations
            'db2_open_cursor': re.compile(r'OPEN\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_fetch_cursor': re.compile(r'FETCH\s+(?:FROM\s+)?([A-Z][A-Z0-9_]*)\s+INTO\s+(.*)', re.IGNORECASE),
            'db2_close_cursor': re.compile(r'CLOSE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            
            # DB2 Control Flow patterns
            'db2_if_statement': re.compile(r'IF\s+(.*?)\s+THEN\s+(.*?)(?:\s+ELSE\s+(.*?))?\s+END\s+IF', re.IGNORECASE | re.DOTALL),
            'db2_case_statement': re.compile(r'CASE\s+(.*?)\s+END\s+CASE', re.IGNORECASE | re.DOTALL),
            'db2_while_loop': re.compile(r'WHILE\s+(.*?)\s+DO\s+(.*?)\s+END\s+WHILE', re.IGNORECASE | re.DOTALL),
            'db2_for_loop': re.compile(r'FOR\s+(.*?)\s+AS\s+(.*?)\s+DO\s+(.*?)\s+END\s+FOR', re.IGNORECASE | re.DOTALL),
            'db2_repeat_loop': re.compile(r'REPEAT\s+(.*?)\s+UNTIL\s+(.*?)\s+END\s+REPEAT', re.IGNORECASE | re.DOTALL),
            'db2_loop_statement': re.compile(r'(?:LOOP|LOOP_LABEL):\s+LOOP\s+(.*?)\s+END\s+LOOP', re.IGNORECASE | re.DOTALL),
            
            # DB2 Exception Handling
            'db2_signal_statement': re.compile(r'SIGNAL\s+(SQLSTATE\s+[\'"][0-9A-Z]{5}[\'"]|[A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_resignal_statement': re.compile(r'RESIGNAL\s*(SQLSTATE\s+[\'"][0-9A-Z]{5}[\'"]|[A-Z][A-Z0-9_]*)?', re.IGNORECASE),
            'db2_get_diagnostics': re.compile(r'GET\s+DIAGNOSTICS\s+(.*)', re.IGNORECASE),
            
            # DB2 Transaction Control
            'db2_commit': re.compile(r'COMMIT(?:\s+WORK)?', re.IGNORECASE),
            'db2_rollback': re.compile(r'ROLLBACK(?:\s+WORK)?(?:\s+TO\s+SAVEPOINT\s+([A-Z][A-Z0-9_]*))?', re.IGNORECASE),
            'db2_savepoint': re.compile(r'SAVEPOINT\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_release_savepoint': re.compile(r'RELEASE\s+(?:TO\s+)?SAVEPOINT\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            
            # DB2 Dynamic SQL patterns
            'db2_prepare': re.compile(r'PREPARE\s+([A-Z][A-Z0-9_]*)\s+FROM\s+(.*)', re.IGNORECASE),
            'db2_execute': re.compile(r'EXECUTE\s+([A-Z][A-Z0-9_]*)(?:\s+USING\s+(.*?))?', re.IGNORECASE),
            'db2_execute_immediate': re.compile(r'EXECUTE\s+IMMEDIATE\s+(.*)', re.IGNORECASE),
            'db2_describe': re.compile(r'DESCRIBE\s+(?:INPUT|OUTPUT)?\s*([A-Z][A-Z0-9_]*)\s+INTO\s+(.*)', re.IGNORECASE),
            
            # DB2 Result Set patterns
            'db2_allocate_cursor': re.compile(r'ALLOCATE\s+([A-Z][A-Z0-9_]*)\s+CURSOR\s+FOR\s+RESULT\s+SET\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_associate_locators': re.compile(r'ASSOCIATE\s+LOCATORS\s*\((.*?)\)\s+WITH\s+PROCEDURE\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE),
            'db2_return_result_set': re.compile(r'(?:RETURN|RETURNS)\s+(?:RESULT\s+)?(?:SET|SETS)', re.IGNORECASE),
            
            # DB2 XML and LOB patterns
            'db2_xml_document': re.compile(r'XMLDOCUMENT\s*\((.*?)\)', re.IGNORECASE),
            'db2_xml_element': re.compile(r'XMLELEMENT\s*\((.*?)\)', re.IGNORECASE),
            'db2_lob_locator': re.compile(r'([A-Z][A-Z0-9_]*)\s+(BLOB|CLOB|DBCLOB)\s+LOCATOR', re.IGNORECASE),
            
            # DB2 Performance and Optimization patterns
            'db2_with_ur': re.compile(r'WITH\s+UR', re.IGNORECASE),
            'db2_with_cs': re.compile(r'WITH\s+CS', re.IGNORECASE),
            'db2_with_rs': re.compile(r'WITH\s+RS', re.IGNORECASE),
            'db2_with_rr': re.compile(r'WITH\s+RR', re.IGNORECASE),
            'db2_optimize_for': re.compile(r'OPTIMIZE\s+FOR\s+(\d+)\s+ROWS?', re.IGNORECASE),
            'db2_fetch_first': re.compile(r'FETCH\s+FIRST\s+(\d+)\s+ROWS?\s+ONLY', re.IGNORECASE),
            
            # DB2 Security and Authorization
            'db2_grant': re.compile(r'GRANT\s+(.*?)\s+ON\s+(.*?)\s+TO\s+(.*)', re.IGNORECASE),
            'db2_revoke': re.compile(r'REVOKE\s+(.*?)\s+ON\s+(.*?)\s+FROM\s+(.*)', re.IGNORECASE),
            'db2_current_user': re.compile(r'CURRENT\s+USER', re.IGNORECASE),
            'db2_session_user': re.compile(r'SESSION_USER', re.IGNORECASE),
            'db2_system_user': re.compile(r'SYSTEM_USER', re.IGNORECASE),
        }

    def _init_sql_patterns(self):
        """Initialize comprehensive SQL patterns for embedded SQL"""
        self.sql_patterns = {
            # Enhanced embedded SQL patterns
            'sql_include_sqlca': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+SQLCA', re.IGNORECASE),
            'sql_include_sqlda': re.compile(r'EXEC\s+SQL\s+INCLUDE\s+SQLDA', re.IGNORECASE),
            'sql_declare_table': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9_]*)\s+TABLE\s*\((.*?)\)', re.IGNORECASE | re.DOTALL),
            'sql_declare_statement': re.compile(r'EXEC\s+SQL\s+DECLARE\s+([A-Z][A-Z0-9_]*)\s+STATEMENT', re.IGNORECASE),
            
            # COBOL Stored Procedure SQL patterns
            'cobol_sql_procedure': re.compile(r'EXEC\s+SQL\s+CREATE\s+PROCEDURE\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cobol_procedure_call': re.compile(r'EXEC\s+SQL\s+CALL\s+([A-Z][A-Z0-9_]*)\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE),
            'cobol_result_set': re.compile(r'EXEC\s+SQL\s+ASSOCIATE\s+(?:RESULT\s+SET\s+)?LOCATOR\s*\((.*?)\)\s+END-EXEC', re.IGNORECASE),
            'cobol_allocate_cursor': re.compile(r'EXEC\s+SQL\s+ALLOCATE\s+([A-Z][A-Z0-9_]*)\s+CURSOR\s+FOR\s+RESULT\s+SET\s+([A-Z][A-Z0-9_]*)\s+END-EXEC', re.IGNORECASE),
            
            # Advanced SQL operations
            'sql_dynamic_prepare': re.compile(r'EXEC\s+SQL\s+PREPARE\s+([A-Z][A-Z0-9_]*)\s+FROM\s+([A-Z][A-Z0-9-]*)\s+END-EXEC', re.IGNORECASE),
            'sql_dynamic_execute': re.compile(r'EXEC\s+SQL\s+EXECUTE\s+([A-Z][A-Z0-9_]*)(?:\s+USING\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s+END-EXEC', re.IGNORECASE),
            'sql_dynamic_open': re.compile(r'EXEC\s+SQL\s+OPEN\s+([A-Z][A-Z0-9_]*)(?:\s+USING\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*))?\s+END-EXEC', re.IGNORECASE),
            
            # SQL Error handling patterns
            'sql_whenever_sqlerror': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+SQLERROR\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|STOP)', re.IGNORECASE),
            'sql_whenever_not_found': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+NOT\s+FOUND\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|STOP)', re.IGNORECASE),
            'sql_whenever_sqlwarning': re.compile(r'EXEC\s+SQL\s+WHENEVER\s+SQLWARNING\s+(CONTINUE|GOTO\s+[A-Z][A-Z0-9-]*|STOP)', re.IGNORECASE),
            
            # SQL Communication Area patterns
            'sqlca_check': re.compile(r'IF\s+SQLCODE\s*(?:=|NOT\s*=|<|>|<=|>=)\s*([0-9-]+)', re.IGNORECASE),
            'sqlstate_check': re.compile(r'IF\s+SQLSTATE\s*(?:=|NOT\s*=)\s*[\'"]([0-9A-Z]{5})[\'"]', re.IGNORECASE),
            'sql_return_code': re.compile(r'([A-Z][A-Z0-9-]*(?:RC|RETURN-CODE|RET-CODE))', re.IGNORECASE),
        }

    def _init_jcl_patterns(self):
        """Initialize comprehensive JCL patterns"""
        self.jcl_patterns = {
            # Enhanced JCL statement patterns
            'job_card': re.compile(r'^//(\S+)\s+JOB\s+(.*?)(?=\n//|\n\*|$)', re.MULTILINE | re.DOTALL),
            'job_step': re.compile(r'^//(\S+)\s+EXEC\s+(.*?)(?=\n//|\n\*|$)', re.MULTILINE | re.DOTALL),
            'dd_statement': re.compile(r'^//(\S+)\s+DD\s+(.*?)(?=\n//|\n\*|$)', re.MULTILINE | re.DOTALL),
            'proc_definition': re.compile(r'^//(\S+)\s+PROC(?:\s+(.*?))?(?=\n//|\n\*|$)', re.MULTILINE | re.DOTALL),
            'pend_statement': re.compile(r'^//\s+PEND', re.MULTILINE),
            'include_statement': re.compile(r'^//\s+INCLUDE\s+MEMBER=(\S+)', re.MULTILINE | re.IGNORECASE),
            'jcllib_statement': re.compile(r'^//\s+JCLLIB\s+ORDER=\((.*?)\)', re.MULTILINE | re.IGNORECASE),
            
            # JCL Parameter patterns
            'proc_call': re.compile(r'EXEC\s+(\S+)(?:,(.*))?', re.IGNORECASE),
            'program_call': re.compile(r'EXEC\s+PGM=(\S+)(?:,(.*))?', re.IGNORECASE),
            'dataset': re.compile(r'DSN=([^,\s]+)', re.IGNORECASE),
            'dataset_disposition': re.compile(r'DISP=\(([^)]+)\)', re.IGNORECASE),
            'dataset_space': re.compile(r'SPACE=\(([^)]+)\)', re.IGNORECASE),
            'dataset_unit': re.compile(r'UNIT=([^,\s]+)', re.IGNORECASE),
            'dataset_volume': re.compile(r'VOL=(?:SER=)?([^,\s]+)', re.IGNORECASE),
            'dataset_dcb': re.compile(r'DCB=\(([^)]+)\)', re.IGNORECASE),
            
            # JCL Control statements
            'set_statement': re.compile(r'^//\s+SET\s+([^=]+)=([^\s,]+)', re.MULTILINE),
            'if_statement': re.compile(r'^//\s+IF\s+(.*?)\s+THEN', re.MULTILINE),
            'else_statement': re.compile(r'^//\s+ELSE', re.MULTILINE),
            'endif_statement': re.compile(r'^//\s+ENDIF', re.MULTILINE),
            'condition_parameter': re.compile(r'COND=\(([^)]+)\)', re.IGNORECASE),
            'restart_parameter': re.compile(r'RESTART=([A-Z0-9]+)(?:\.([A-Z0-9]+))?', re.IGNORECASE),
            'return_code_check': re.compile(r'\bRC\s*(=|EQ|NE|GT|LT|GE|LE)\s*(\d+)', re.IGNORECASE),
            
            # JCL Symbols and substitution
            'symbolic_parameter': re.compile(r'&([A-Z][A-Z0-9]*)', re.IGNORECASE),
            'system_symbol': re.compile(r'&(SYSUID|SYSPLEX|SYSCLONE|SYSNAME|LDATE|LTIME)', re.IGNORECASE),
            'jcl_comment': re.compile(r'^\*.*', re.MULTILINE),
            'inline_comment': re.compile(r'\s+\*\s+(.*)', re.MULTILINE),
            
            # Output and SYSOUT patterns
            'sysout_class': re.compile(r'SYSOUT=([A-Z*])', re.IGNORECASE),
            'output_class': re.compile(r'CLASS=([A-Z0-9])', re.IGNORECASE),
            'output_dest': re.compile(r'DEST=([^,\s]+)', re.IGNORECASE),
            'output_forms': re.compile(r'FORMS=([^,\s]+)', re.IGNORECASE),
            'output_copies': re.compile(r'COPIES=(\d+)', re.IGNORECASE),
        }

    def _init_cics_patterns(self):
        """Initialize comprehensive CICS patterns"""
        self.cics_patterns = {
            # Enhanced Terminal operations with comprehensive parameter validation
            'cics_send_map': re.compile(r'EXEC\s+CICS\s+SEND\s+(?:MAP\s*\(([^)]+)\)\s*)?(?:MAPSET\s*\(([^)]+)\)\s*)?(?:FROM\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:DATAONLY|MAPONLY|ALARM|FREEKB|FRSET|CURSOR\s*\(([^)]+)\)|TERMINAL|ERASE|ERASEAUP|PRINT|NLEOM|L40|L64|L80|ACCUM|PAGING|NOAUTOPAGE|LAST|FMHPARM\s*\(([^)]+)\)|STRFIELD|REQID\s*\(([^)]+)\)|TRAILER\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive_map': re.compile(r'EXEC\s+CICS\s+RECEIVE\s+(?:MAP\s*\(([^)]+)\)\s*)?(?:MAPSET\s*\(([^)]+)\)\s*)?(?:INTO\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:ASIS|BUFFER|TERMINAL)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_send_text': re.compile(r'EXEC\s+CICS\s+SEND\s+TEXT\s+(?:FROM\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:ERASE|FREEKB|ALARM|TERMINAL|PRINT|ACCUM|CURSOR|HEADER\s*\(([^)]+)\)|TRAILER\s*\(([^)]+)\)|JUSTIFY\s*\(([^)]+)\)|NLEOM|FORMFEED)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_receive': re.compile(r'EXEC\s+CICS\s+RECEIVE\s+(?:INTO\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:MAXLENGTH\s*\(([^)]+)\)\s*)?(?:ASIS|BUFFER|TERMINAL|NOTRUNCATE)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Enhanced File operations with comprehensive VSAM and error handling
            'cics_read': re.compile(r'EXEC\s+CICS\s+READ\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:INTO\s*\(([^)]+)\)\s*)?(?:RIDFLD\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:KEYLENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:EQUAL|GTEQ|GENERIC|UPDATE|TOKEN\s*\(([^)]+)\)|CONSISTENT|UNCOMMITTED|REPEATABLE)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_write': re.compile(r'EXEC\s+CICS\s+WRITE\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:FROM\s*\(([^)]+)\)\s*)?(?:RIDFLD\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:KEYLENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:MASSINSERT|TOKEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_rewrite': re.compile(r'EXEC\s+CICS\s+REWRITE\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:FROM\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:TOKEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_delete': re.compile(r'EXEC\s+CICS\s+DELETE\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:RIDFLD\s*\(([^)]+)\)\s*)?(?:KEYLENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:TOKEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_startbr': re.compile(r'EXEC\s+CICS\s+STARTBR\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:RIDFLD\s*\(([^)]+)\)\s*)?(?:KEYLENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:REQID\s*\(([^)]+)\)\s*)?(?:EQUAL|GTEQ|GENERIC)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readnext': re.compile(r'EXEC\s+CICS\s+READNEXT\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:INTO\s*\(([^)]+)\)\s*)?(?:RIDFLD\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:KEYLENGTH\s*\(([^)]+)\)\s*)?(?:REQID\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:TOKEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readprev': re.compile(r'EXEC\s+CICS\s+READPREV\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:INTO\s*\(([^)]+)\)\s*)?(?:RIDFLD\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:KEYLENGTH\s*\(([^)]+)\)\s*)?(?:REQID\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:TOKEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_endbr': re.compile(r'EXEC\s+CICS\s+ENDBR\s+(?:FILE\s*\(([^)]+)\)\s*)?(?:REQID\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:TOKEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Enhanced Program control with comprehensive flow analysis
            'cics_link': re.compile(r'EXEC\s+CICS\s+LINK\s+(?:PROGRAM\s*\(([^)]+)\)\s*)?(?:COMMAREA\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:SYNCONRETURN|TRANSID\s*\(([^)]+)\)|INPUTMSG\s*\(([^)]+)\)|INPUTMSGLEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_xctl': re.compile(r'EXEC\s+CICS\s+XCTL\s+(?:PROGRAM\s*\(([^)]+)\)\s*)?(?:COMMAREA\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:INPUTMSG\s*\(([^)]+)\)|INPUTMSGLEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_return': re.compile(r'EXEC\s+CICS\s+RETURN\s*(?:TRANSID\s*\(([^)]+)\)\s*)?(?:COMMAREA\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:IMMEDIATE|INPUTMSG\s*\(([^)]+)\)|INPUTMSGLEN\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_load': re.compile(r'EXEC\s+CICS\s+LOAD\s+(?:PROGRAM\s*\(([^)]+)\)\s*)?(?:SET\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:ENTRY\s*\(([^)]+)\)\s*)?(?:HOLD)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_release': re.compile(r'EXEC\s+CICS\s+RELEASE\s+(?:PROGRAM\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # Enhanced Error handling with comprehensive context tracking
            'cics_handle_condition': re.compile(r'EXEC\s+CICS\s+HANDLE\s+CONDITION\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_handle_aid': re.compile(r'EXEC\s+CICS\s+HANDLE\s+AID\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_handle_abend': re.compile(r'EXEC\s+CICS\s+HANDLE\s+ABEND\s+(?:PROGRAM\s*\(([^)]+)\)\s*)?(?:LABEL\s*\(([^)]+)\)\s*)?(?:CANCEL|RESET)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_push_handle': re.compile(r'EXEC\s+CICS\s+PUSH\s+HANDLE\s*END-EXEC', re.IGNORECASE),
            'cics_pop_handle': re.compile(r'EXEC\s+CICS\s+POP\s+HANDLE\s*END-EXEC', re.IGNORECASE),
            'cics_ignore_condition': re.compile(r'EXEC\s+CICS\s+IGNORE\s+CONDITION\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_resp': re.compile(r'RESP\s*\(([A-Z][A-Z0-9-]*)\)', re.IGNORECASE),
            'cics_resp2': re.compile(r'RESP2\s*\(([A-Z][A-Z0-9-]*)\)', re.IGNORECASE),
            'cics_nohandle': re.compile(r'\bNOHANDLE\b', re.IGNORECASE),
            
            # CICS Transaction and Timing control
            'cics_asktime': re.compile(r'EXEC\s+CICS\s+ASKTIME\s*(?:ABSTIME\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_formattime': re.compile(r'EXEC\s+CICS\s+FORMATTIME\s+(?:ABSTIME\s*\(([^)]+)\)\s*)?(?:YYDDD\s*\(([^)]+)\)|YYMMDD\s*\(([^)]+)\)|MMDDYY\s*\(([^)]+)\)|DDMMYY\s*\(([^)]+)\)|TIME\s*\(([^)]+)\)|TIMESEP\s*\(([^)]+)\)|DATESEP\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_delay': re.compile(r'EXEC\s+CICS\s+DELAY\s+(?:INTERVAL\s*\(([^)]+)\)|FOR\s+(?:HOURS\s*\(([^)]+)\)|MINUTES\s*\(([^)]+)\)|SECONDS\s*\(([^)]+)\)))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_start': re.compile(r'EXEC\s+CICS\s+START\s+(?:TRANSID\s*\(([^)]+)\)\s*)?(?:INTERVAL\s*\(([^)]+)\)|AT\s*\(([^)]+)\))?(?:DATA\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:TERMID\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:REQID\s*\(([^)]+)\)\s*)?(?:PROTECT|NOCHECK)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_cancel': re.compile(r'EXEC\s+CICS\s+CANCEL\s+(?:REQID\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?(?:TRANSID\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # CICS Storage and Task management
            'cics_getmain': re.compile(r'EXEC\s+CICS\s+GETMAIN\s+(?:SET\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:FLENGTH\s*\(([^)]+)\)\s*)?(?:INITIMG\s*\(([^)]+)\)\s*)?(?:SHARED|NOSUSPEND|CICSDATAKEY|USERDATAKEY)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_freemain': re.compile(r'EXEC\s+CICS\s+FREEMAIN\s+(?:DATA\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:FLENGTH\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_address': re.compile(r'EXEC\s+CICS\s+ADDRESS\s+(?:EIB\s*\(([^)]+)\)|TWA\s*\(([^)]+)\)|CWA\s*\(([^)]+)\)|CSA\s*\(([^)]+)\)|COMMAREA\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_assign': re.compile(r'EXEC\s+CICS\s+ASSIGN\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # CICS Temporary Storage and Transient Data
            'cics_writeq_ts': re.compile(r'EXEC\s+CICS\s+WRITEQ\s+TS\s+(?:QUEUE\s*\(([^)]+)\)\s*)?(?:FROM\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:ITEM\s*\(([^)]+)\)\s*)?(?:REWRITE|MAIN|AUXILIARY|SYSID\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readq_ts': re.compile(r'EXEC\s+CICS\s+READQ\s+TS\s+(?:QUEUE\s*\(([^)]+)\)\s*)?(?:INTO\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:ITEM\s*\(([^)]+)\)\s*)?(?:NEXT|NUMITEMS\s*\(([^)]+)\)|SYSID\s*\(([^)]+)\))*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_deleteq_ts': re.compile(r'EXEC\s+CICS\s+DELETEQ\s+TS\s+(?:QUEUE\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_writeq_td': re.compile(r'EXEC\s+CICS\s+WRITEQ\s+TD\s+(?:QUEUE\s*\(([^)]+)\)\s*)?(?:FROM\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_readq_td': re.compile(r'EXEC\s+CICS\s+READQ\s+TD\s+(?:QUEUE\s*\(([^)]+)\)\s*)?(?:INTO\s*\(([^)]+)\)\s*)?(?:LENGTH\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_deleteq_td': re.compile(r'EXEC\s+CICS\s+DELETEQ\s+TD\s+(?:QUEUE\s*\(([^)]+)\)\s*)?(?:SYSID\s*\(([^)]+)\)\s*)?END-EXEC', re.IGNORECASE | re.DOTALL),
            
            # CICS Syncpoint and Recovery
            'cics_syncpoint': re.compile(r'EXEC\s+CICS\s+SYNCPOINT\s*(?:ROLLBACK)*\s*END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_syncpoint_rollback': re.compile(r'EXEC\s+CICS\s+SYNCPOINT\s+ROLLBACK\s*END-EXEC', re.IGNORECASE),
            
            # CICS Web Services and Internet support
            'cics_web_open': re.compile(r'EXEC\s+CICS\s+WEB\s+OPEN\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_web_close': re.compile(r'EXEC\s+CICS\s+WEB\s+CLOSE\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_web_send': re.compile(r'EXEC\s+CICS\s+WEB\s+SEND\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_web_receive': re.compile(r'EXEC\s+CICS\s+WEB\s+RECEIVE\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_document_create': re.compile(r'EXEC\s+CICS\s+DOCUMENT\s+CREATE\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
            'cics_document_insert': re.compile(r'EXEC\s+CICS\s+DOCUMENT\s+INSERT\s+(.*?)\s+END-EXEC', re.IGNORECASE | re.DOTALL),
        }

    def _init_bms_patterns(self):
        """Initialize comprehensive BMS patterns"""
        self.bms_patterns = {
            # Enhanced BMS Mapset patterns
            'bms_mapset': re.compile(r'(\w+)\s+DFHMSD\s+(.*?)(?=\w+\s+(?:DFHMSD|DFHMDI)|$)', re.IGNORECASE | re.DOTALL),
            'bms_mapset_params': re.compile(r'TYPE=(?:&?)(\w+)|MODE=(\w+)|LANG=(\w+)|STORAGE=(\w+)|DSATTS=\(([^)]+)\)|MAPATTS=\(([^)]+)\)|EXTATT=(\w+)|TERM=(\w+)|CTRL=\(([^)]+)\)|TIOAPFX=(\w+)', re.IGNORECASE),
            
            # Enhanced BMS Map patterns  
            'bms_map': re.compile(r'(\w+)\s+DFHMDI\s+(.*?)(?=\w+\s+(?:DFHMDI|DFHMDF|DFHMSD)|$)', re.IGNORECASE | re.DOTALL),
            'bms_map_params': re.compile(r'SIZE=\((\d+),(\d+)\)|LINE=(\d+)|COLUMN=(\d+)|JUSTIFY=\(([^)]+)\)|HEADER=(\w+)|TRAILER=(\w+)|CTRL=\(([^)]+)\)|DSATTS=\(([^)]+)\)|MAPATTS=\(([^)]+)\)', re.IGNORECASE),
            
            # Enhanced BMS Field patterns with comprehensive attributes
            'bms_field': re.compile(r'(\w+)\s+DFHMDF\s+(.*?)(?=\w+\s+(?:DFHMDF|DFHMDI|DFHMSD)|$)', re.IGNORECASE | re.DOTALL),
            'bms_field_params': re.compile(r'POS=\((\d+),(\d+)\)|LENGTH=(\d+)|ATTRB=\(([^)]+)\)|INITIAL=([\'"][^\']*[\'"]|\w+)|PICIN=([\'"][^\']*[\'"]|\w+)|PICOUT=([\'"][^\']*[\'"]|\w+)|XINIT=([0-9A-F]+)|JUSTIFY=\(([^)]+)\)|OCCURS=(\d+)|COLOR=(\w+)|HILIGHT=(\w+)|VALIDN=\(([^)]+)\)|OUTLINE=(\w+)|PS=(\d+)|SOSI=(\w+)|TRANSP|FSET|IC|CURSOR', re.IGNORECASE),
            
            # BMS Special patterns
            'bms_mapset_end': re.compile(r'\s+DFHMSD\s+TYPE=FINAL', re.IGNORECASE),
            'bms_copy_member': re.compile(r'COPY\s+(\w+)', re.IGNORECASE),
            'bms_symbolic_map': re.compile(r'(\w+)\s+DFHMSD\s+.*?TYPE=(?:&?)DSECT', re.IGNORECASE | re.DOTALL),
            
            # BMS Attribute patterns
            'bms_pos': re.compile(r'POS=\((\d+),(\d+)\)', re.IGNORECASE),
            'bms_length': re.compile(r'LENGTH=(\d+)', re.IGNORECASE),
            'bms_attrb': re.compile(r'ATTRB=\(([^)]+)\)', re.IGNORECASE),
            'bms_initial': re.compile(r'INITIAL=([\'"][^\']*[\'"]|\w+)', re.IGNORECASE),
            'bms_color': re.compile(r'COLOR=(\w+)', re.IGNORECASE),
            'bms_hilight': re.compile(r'HILIGHT=(\w+)', re.IGNORECASE),
            'bms_occurs': re.compile(r'OCCURS=(\d+)', re.IGNORECASE),
            
            # BMS Field attributes
            'bms_askip': re.compile(r'\bASKIP\b', re.IGNORECASE),
            'bms_prot': re.compile(r'\bPROT\b', re.IGNORECASE),
            'bms_unprot': re.compile(r'\bUNPROT\b', re.IGNORECASE),
            'bms_num': re.compile(r'\bNUM\b', re.IGNORECASE),
            'bms_brt': re.compile(r'\bBRT\b', re.IGNORECASE),
            'bms_dark': re.compile(r'\bDARK\b', re.IGNORECASE),
            'bms_norm': re.compile(r'\bNORM\b', re.IGNORECASE),
            'bms_ic': re.compile(r'\bIC\b', re.IGNORECASE),
            'bms_fset': re.compile(r'\bFSET\b', re.IGNORECASE),
        }
# ==================== MISSING ENHANCED PARSING METHODS ====================

    async def _parse_cobol_divisions_enhanced(self, content: str, program_name: str, 
                                            program_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse COBOL divisions with enhanced LLM analysis"""
        chunks = []
        
        # Find all divisions
        divisions = [
            ('IDENTIFICATION', self.cobol_patterns['identification_division']),
            ('ENVIRONMENT', self.cobol_patterns['environment_division']),
            ('DATA', self.cobol_patterns['data_division']),
            ('PROCEDURE', self.cobol_patterns['procedure_division'])
        ]
        
        for div_name, pattern in divisions:
            match = pattern.search(content)
            if match:
                # Extract division content
                start_pos = match.start()
                
                # Find next division or end of file
                next_div_pos = len(content)
                for next_div_name, next_pattern in divisions:
                    if next_div_name != div_name:
                        next_match = next_pattern.search(content, start_pos + 1)
                        if next_match and next_match.start() < next_div_pos:
                            next_div_pos = next_match.start()
                
                division_content = content[start_pos:next_div_pos].strip()
                
                # Analyze division with LLM
                division_analysis = await self._analyze_with_llm_cached(
                    division_content, f'cobol_{div_name.lower()}_division',
                    """
                    Analyze this COBOL {division_name} division:
                    
                    {content}
                    
                    Identify:
                    1. Key elements and their business purpose
                    2. Compliance with COBOL standards
                    3. Complexity indicators
                    4. Potential issues or improvements
                    
                    Return as JSON:
                    {{
                        "key_elements": ["element1", "element2"],
                        "business_purpose": "program identification and setup",
                        "compliance_level": "high",
                        "complexity_score": 3,
                        "issues": ["issue1"],
                        "suggestions": ["suggestion1"]
                    }}
                    """,
                    division_name=div_name
                )
                
                business_context = {
                    'division_type': div_name.lower(),
                    'business_purpose': division_analysis.get('analysis', {}).get('business_purpose', ''),
                    'complexity_score': division_analysis.get('analysis', {}).get('complexity_score', 0),
                    'compliance_level': division_analysis.get('analysis', {}).get('compliance_level', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_{div_name}_DIV",
                    chunk_type=f"cobol_{div_name.lower()}_division",
                    content=division_content,
                    metadata={
                        'division_name': div_name,
                        'llm_analysis': division_analysis.get('analysis', {}),
                        'division_size': len(division_content.split('\n'))
                    },
                    business_context=business_context,
                    confidence_score=division_analysis.get('confidence_score', 0.8),
                    line_start=content[:start_pos].count('\n'),
                    line_end=content[:next_div_pos].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    # Replace the _parse_cobol_sections_enhanced method with this safer version:

    async def _parse_cobol_sections_enhanced(self, content: str, program_name: str, 
                                        program_analysis: Dict[str, Any] = None) -> List[CodeChunk]:
        """Parse COBOL sections with enhanced business context - FULLY SAFE VERSION"""
        chunks = []
        
        # Provide default if program_analysis is None
        if program_analysis is None:
            program_analysis = {'analysis': {}, 'confidence_score': 0.5}
        
        try:
            # Find all sections - check if pattern exists first
            if 'section' not in self.cobol_patterns:
                self.logger.warning("Section pattern not found in cobol_patterns")
                return chunks
                
            section_pattern = self.cobol_patterns['section']
            if not section_pattern:
                self.logger.warning("Section pattern is None")
                return chunks
                
            section_matches = list(section_pattern.finditer(content))
            self.logger.info(f"Found {len(section_matches)} section matches")
            
            if not section_matches:
                self.logger.info("No sections found in content")
                return chunks
            
        except Exception as e:
            self.logger.error(f"Error finding sections: {e}")
            return chunks
        
        for i, match in enumerate(section_matches):
            try:
                # Safely extract section name
                section_name = 'UNKNOWN_SECTION'
                if match and match.groups():
                    if len(match.groups()) >= 1 and match.group(1):
                        section_name = match.group(1).strip()
                    else:
                        section_name = f'SECTION_{i+1}'
                else:
                    section_name = f'SECTION_{i+1}'
                
                start_pos = match.start()
                
                # Find end of section (next section or division)
                end_pos = len(content)  # Default to end of content
                
                # Try to find next section
                if i + 1 < len(section_matches):
                    end_pos = section_matches[i + 1].start()
                else:
                    # Look for next division
                    try:
                        if 'procedure_division' in self.cobol_patterns and self.cobol_patterns['procedure_division']:
                            next_div = self.cobol_patterns['procedure_division'].search(content, start_pos + 1)
                            if next_div and next_div.start() < end_pos:
                                end_pos = next_div.start()
                    except Exception as div_error:
                        self.logger.debug(f"Error finding next division: {div_error}")
                
                # Extract section content safely
                try:
                    section_content = content[start_pos:end_pos].strip()
                except Exception as content_error:
                    self.logger.warning(f"Error extracting section content: {content_error}")
                    section_content = content[start_pos:start_pos+500]  # Fallback to first 500 chars
                
                # Only analyze if content is substantial
                if len(section_content) > 50:
                    try:
                        section_analysis = await self._analyze_with_llm_cached(
                            section_content, 'cobol_section',
                            """
                            Analyze this COBOL section and return ONLY a JSON object:

                            {content}

                            Return exactly this JSON format:
                            {{
                                "purpose": "brief description of what this section does",
                                "functionality": "main function performed",
                                "complexity": "low",
                                "business_impact": "medium"
                            }}
                            """
                        )
                    except Exception as llm_error:
                        self.logger.warning(f"LLM analysis failed for section {section_name}: {llm_error}")
                        section_analysis = {
                            'analysis': {
                                'purpose': 'section_processing',
                                'functionality': 'cobol_logic',
                                'complexity': 'medium',
                                'business_impact': 'standard'
                            },
                            'confidence_score': 0.3
                        }
                else:
                    section_analysis = {
                        'analysis': {
                            'purpose': 'small_section',
                            'functionality': 'minimal_processing',
                            'complexity': 'low',
                            'business_impact': 'low'
                        },
                        'confidence_score': 0.7
                    }
                
                # Safely extract analysis results
                analysis_data = section_analysis.get('analysis', {})
                business_context = {
                    'section_name': section_name,
                    'purpose': analysis_data.get('purpose', 'unknown'),
                    'functionality': analysis_data.get('functionality', 'unknown'),
                    'business_impact': analysis_data.get('business_impact', 'unknown')
                }
                
                # Create chunk safely
                try:
                    chunk = CodeChunk(
                        program_name=program_name,
                        chunk_id=f"{program_name[:20]}_SEC_{section_name[:20]}_{i}",
                        chunk_type="cobol_section",
                        content=section_content,
                        metadata={
                            'section_name': section_name,
                            'llm_analysis': analysis_data,
                            'section_lines': len(section_content.split('\n')),
                            'section_index': i
                        },
                        business_context=business_context,
                        confidence_score=section_analysis.get('confidence_score', 0.8),
                        line_start=content[:start_pos].count('\n'),
                        line_end=content[:end_pos].count('\n')
                    )
                    chunks.append(chunk)
                    self.logger.debug(f"✅ Created section chunk for {section_name}")
                    
                except Exception as chunk_error:
                    self.logger.error(f"Failed to create chunk for section {section_name}: {chunk_error}")
                    continue
                    
            except Exception as section_error:
                self.logger.error(f"Error processing section {i}: {section_error}")
                continue
        
        self.logger.info(f"✅ Successfully created {len(chunks)} section chunks")
        return chunks

    # Also add this safer pattern checking method to avoid similar issues:

    def _safe_get_pattern(self, pattern_name: str):
        """Safely get a pattern from cobol_patterns"""
        try:
            if hasattr(self, 'cobol_patterns') and self.cobol_patterns:
                if pattern_name in self.cobol_patterns:
                    pattern = self.cobol_patterns[pattern_name]
                    if pattern is not None:
                        return pattern
                    else:
                        self.logger.warning(f"Pattern {pattern_name} is None")
                        return None
                else:
                    self.logger.warning(f"Pattern {pattern_name} not found in cobol_patterns")
                    return None
            else:
                self.logger.error("cobol_patterns not initialized")
                return None
        except Exception as e:
            self.logger.error(f"Error accessing pattern {pattern_name}: {e}")
            return None

    
    async def _parse_data_items_enhanced(self, content: str, program_name: str,
                                   program_analysis: Dict[str, Any] = None) -> List[CodeChunk]:
        """Parse COBOL data items with enhanced business analysis - FIXED SIGNATURE"""
        chunks = []
        
        # Provide default if program_analysis is None
        if program_analysis is None:
            program_analysis = {'analysis': {}, 'confidence_score': 0.5}
        
        # Find all data items
        data_matches = list(self.cobol_patterns['data_item'].finditer(content))
        
        if not data_matches:
            return chunks
        
        # Group data items by level hierarchy
        try:
            data_groups = self._group_data_items_by_hierarchy(data_matches)
        except Exception as e:
            self.logger.warning(f"Data grouping failed: {e}")
            return chunks
        
        for group in data_groups:
            if len(group['items']) > 1:  # Only process groups with multiple items
                group_content = '\n'.join([item['match'].group(0) for item in group['items']])
                
                # Only analyze substantial groups
                if len(group_content) > 100:
                    try:
                        data_analysis = await self._analyze_with_llm_cached(
                            group_content, 'cobol_data_group',
                            """
                            Analyze this COBOL data group:
                            
                            {content}
                            
                            Identify:
                            1. Business entity represented
                            2. Data usage patterns
                            3. Field relationships and dependencies
                            4. Data validation requirements
                            5. Performance considerations
                            
                            Return as JSON:
                            {{
                                "business_entity": "customer_record",
                                "usage_patterns": ["input", "processing", "output"],
                                "field_relationships": [
                                    {{"parent": "field1", "children": ["field1a", "field1b"]}}
                                ],
                                "validation_requirements": ["required_fields", "format_checks"],
                                "performance_impact": "medium"
                            }}
                            """
                        )
                    except Exception as e:
                        self.logger.warning(f"LLM analysis failed for data group: {e}")
                        data_analysis = {
                            'analysis': {'error': str(e), 'business_entity': 'unknown'},
                            'confidence_score': 0.3
                        }
                else:
                    data_analysis = {
                        'analysis': {'business_entity': 'small_data_group'},
                        'confidence_score': 0.6
                    }
                
                business_context = {
                    'data_group_type': 'hierarchical_structure',
                    'business_entity': data_analysis.get('analysis', {}).get('business_entity', ''),
                    'usage_patterns': data_analysis.get('analysis', {}).get('usage_patterns', []),
                    'field_count': len(group['items']),
                    'hierarchy_depth': group['max_level'] - group['min_level'] + 1
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_DATA_{group['group_id']}",
                    chunk_type="cobol_data_group",
                    content=group_content,
                    metadata={
                        'field_count': len(group['items']),
                        'level_range': f"{group['min_level']}-{group['max_level']}",
                        'llm_analysis': data_analysis.get('analysis', {}),
                        'group_fields': [item['field_name'] for item in group['items']]
                    },
                    business_context=business_context,
                    confidence_score=data_analysis.get('confidence_score', 0.7),
                    line_start=content[:group['start_pos']].count('\n'),
                    line_end=content[:group['end_pos']].count('\n')
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _parse_procedure_division_enhanced(self, content: str, program_name: str,
                                           program_analysis: Dict[str, Any] = None) -> List[CodeChunk]:
        """Parse COBOL procedure division with comprehensive flow analysis - FIXED SIGNATURE"""
        chunks = []
        
        # Provide default if program_analysis is None
        if program_analysis is None:
            program_analysis = {'analysis': {}, 'confidence_score': 0.5}
        
        # Find procedure division
        proc_div_match = self.cobol_patterns['procedure_division'].search(content)
        if not proc_div_match:
            return chunks
        
        proc_start = proc_div_match.start()
        proc_content = content[proc_start:]
        
        # Parse components with error handling
        parsing_methods = [
            (self._parse_paragraphs_enhanced, "paragraphs"),
            (self._parse_perform_statements_enhanced, "PERFORM statements"), 
            (self._parse_control_flow_enhanced, "control flow"),
            (self._parse_file_operations_enhanced, "file operations")
        ]
        
        for method, description in parsing_methods:
            try:
                component_chunks = await method(proc_content, program_name)
                chunks.extend(component_chunks)
                self.logger.info(f"✅ Parsed {len(component_chunks)} {description} chunks")
            except Exception as e:
                self.logger.warning(f"Failed to parse {description}: {e}")
        
        return chunks

    async def _parse_paragraphs_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COBOL paragraphs with business logic analysis"""
        chunks = []
        
        paragraph_matches = list(self.cobol_patterns['paragraph'].finditer(content))
        
        for i, match in enumerate(paragraph_matches):
            para_name = match.group(1)
            start_pos = match.start()
            
            # Find paragraph end
            if i + 1 < len(paragraph_matches):
                end_pos = paragraph_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            para_content = content[start_pos:end_pos].strip()
            
            # Analyze paragraph with LLM
            para_analysis = await self._analyze_with_llm_cached(
                para_content, 'cobol_paragraph',
                """
                Analyze this COBOL paragraph:
                
                {content}
                
                Identify:
                1. Primary business function
                2. Operations performed
                3. Error handling present
                4. Complexity level
                5. Performance characteristics
                
                Return as JSON:
                {{
                    "business_function": "customer_validation",
                    "operations": ["read_file", "validate_data", "update_record"],
                    "error_handling": true,
                    "complexity_level": "medium",
                    "performance_characteristics": "io_intensive"
                }}
                """
            )
            
            business_context = {
                'paragraph_name': para_name,
                'business_function': para_analysis.get('analysis', {}).get('business_function', ''),
                'operations': para_analysis.get('analysis', {}).get('operations', []),
                'has_error_handling': para_analysis.get('analysis', {}).get('error_handling', False),
                'complexity_level': para_analysis.get('analysis', {}).get('complexity_level', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_PARA_{para_name}",
                chunk_type="cobol_paragraph",
                content=para_content,
                metadata={
                    'paragraph_name': para_name,
                    'llm_analysis': para_analysis.get('analysis', {}),
                    'statement_count': len([line for line in para_content.split('\n') if line.strip() and not line.strip().startswith('*')])
                },
                business_context=business_context,
                confidence_score=para_analysis.get('confidence_score', 0.8),
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_perform_statements_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse PERFORM statements with flow analysis"""
        chunks = []
        
        # Different PERFORM patterns
        perform_patterns = {
            'simple': self.cobol_patterns['perform_simple'],
            'until': self.cobol_patterns['perform_until'],
            'varying': self.cobol_patterns['perform_varying'],
            'thru': self.cobol_patterns['perform_thru'],
            'times': self.cobol_patterns['perform_times'],
            'inline': self.cobol_patterns['perform_inline']
        }
        
        for perform_type, pattern in perform_patterns.items():
            matches = list(pattern.finditer(content))
            
            for match in matches:
                perform_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'cobol_perform_{perform_type}',
                    """
                    Analyze this COBOL PERFORM statement:
                    
                    {content}
                    
                    Identify:
                    1. Control flow type and purpose
                    2. Loop characteristics (if applicable)
                    3. Called paragraphs or inline logic
                    4. Performance implications
                    5. Business logic pattern
                    
                    Return as JSON:
                    {{
                        "control_flow_type": "conditional_loop",
                        "loop_characteristics": "until_condition_met",
                        "called_targets": ["para1", "para2"],
                        "performance_impact": "medium",
                        "business_pattern": "data_processing_loop"
                    }}
                    """
                )
                
                business_context = {
                    'perform_type': perform_type,
                    'control_flow_type': perform_analysis.get('analysis', {}).get('control_flow_type', ''),
                    'business_pattern': perform_analysis.get('analysis', {}).get('business_pattern', ''),
                    'performance_impact': perform_analysis.get('analysis', {}).get('performance_impact', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_PERFORM_{hash(match.group(0))%10000}",
                    chunk_type=f"cobol_perform_{perform_type}",
                    content=match.group(0),
                    metadata={
                        'perform_type': perform_type,
                        'llm_analysis': perform_analysis.get('analysis', {}),
                        'targets': perform_analysis.get('analysis', {}).get('called_targets', [])
                    },
                    business_context=business_context,
                    confidence_score=perform_analysis.get('confidence_score', 0.8),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_control_flow_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse control flow statements with decision analysis"""
        chunks = []
        
        # IF statements
        if_matches = list(self.cobol_patterns['if_then_else'].finditer(content))
        for match in if_matches:
            if_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'cobol_if_statement',
                """
                Analyze this COBOL IF statement:
                
                {content}
                
                Identify:
                1. Decision logic and conditions
                2. Business rules implemented
                3. Complexity of conditional logic
                4. Error handling within branches
                5. Data validation patterns
                
                Return as JSON:
                {{
                    "decision_logic": "customer_status_check",
                    "business_rules": ["active_customer_validation"],
                    "complexity": "medium",
                    "has_error_handling": true,
                    "validation_patterns": ["status_code_check"]
                }}
                """
            )
            
            business_context = {
                'statement_type': 'conditional',
                'decision_logic': if_analysis.get('analysis', {}).get('decision_logic', ''),
                'business_rules': if_analysis.get('analysis', {}).get('business_rules', []),
                'complexity': if_analysis.get('analysis', {}).get('complexity', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_IF_{hash(match.group(0))%10000}",
                chunk_type="cobol_if_statement",
                content=match.group(0),
                metadata={
                    'statement_type': 'if_then_else',
                    'llm_analysis': if_analysis.get('analysis', {}),
                    'has_else': 'ELSE' in match.group(0).upper()
                },
                business_context=business_context,
                confidence_score=if_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # EVALUATE statements
        evaluate_matches = list(self.cobol_patterns['evaluate_block'].finditer(content))
        for match in evaluate_matches:
            evaluate_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'cobol_evaluate_statement',
                """
                Analyze this COBOL EVALUATE statement:
                
                {content}
                
                Identify:
                1. Selection criteria and cases
                2. Business logic patterns
                3. Decision tree complexity
                4. Case coverage completeness
                5. Default handling (WHEN OTHER)
                
                Return as JSON:
                {{
                    "selection_criteria": "transaction_type",
                    "business_patterns": ["transaction_routing"],
                    "complexity": "high",
                    "case_count": 5,
                    "has_default": true
                }}
                """
            )
            
            business_context = {
                'statement_type': 'multi_way_selection',
                'selection_criteria': evaluate_analysis.get('analysis', {}).get('selection_criteria', ''),
                'business_patterns': evaluate_analysis.get('analysis', {}).get('business_patterns', []),
                'case_count': evaluate_analysis.get('analysis', {}).get('case_count', 0)
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_EVALUATE_{hash(match.group(0))%10000}",
                chunk_type="cobol_evaluate_statement",
                content=match.group(0),
                metadata={
                    'statement_type': 'evaluate',
                    'llm_analysis': evaluate_analysis.get('analysis', {}),
                    'when_count': content.upper().count('WHEN ')
                },
                business_context=business_context,
                confidence_score=evaluate_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_file_operations_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse file operations with I/O analysis"""
        chunks = []
        
        file_operations = {
            'open': self.cobol_patterns['open_statement'],
            'close': self.cobol_patterns['close_statement'],
            'read': self.cobol_patterns['read_statement'],
            'write': self.cobol_patterns['write_statement'],
            'rewrite': self.cobol_patterns['rewrite_statement'],
            'delete': self.cobol_patterns['delete_statement']
        }
        
        for op_type, pattern in file_operations.items():
            matches = list(pattern.finditer(content))
            
            for match in matches:
                file_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'cobol_file_{op_type}',
                    """
                    Analyze this COBOL file operation:
                    
                    {content}
                    
                    Identify:
                    1. File access pattern and purpose
                    2. Data processing workflow
                    3. Error handling strategy
                    4. Performance implications
                    5. Business transaction context
                    
                    Return as JSON:
                    {{
                        "access_pattern": "sequential_read",
                        "processing_purpose": "customer_data_retrieval",
                        "error_handling": "exception_based",
                        "performance_impact": "high",
                        "transaction_context": "batch_processing"
                    }}
                    """
                )
                
                business_context = {
                    'operation_type': op_type,
                    'access_pattern': file_analysis.get('analysis', {}).get('access_pattern', ''),
                    'processing_purpose': file_analysis.get('analysis', {}).get('processing_purpose', ''),
                    'transaction_context': file_analysis.get('analysis', {}).get('transaction_context', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_FILE_{op_type.upper()}_{hash(match.group(0))%10000}",
                    chunk_type=f"cobol_file_{op_type}",
                    content=match.group(0),
                    metadata={
                        'operation_type': op_type,
                        'llm_analysis': file_analysis.get('analysis', {}),
                        'file_name': match.group(1) if match.groups() else ''
                    },
                    business_context=business_context,
                    confidence_score=file_analysis.get('confidence_score', 0.8),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    def _safe_parse_sql_patterns(self, content: str, pattern_dict: dict) -> List[Dict[str, Any]]:
        """Safely parse SQL patterns with defensive group extraction"""
        results = []
        
        for pattern_name, pattern in pattern_dict.items():
            try:
                matches = list(pattern.finditer(content))
                for match in matches:
                    # Create a safe match result
                    match_result = {
                        'pattern_name': pattern_name,
                        'full_match': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'groups': []
                    }
                    
                    # Safely extract all groups
                    if match.groups():
                        for i in range(len(match.groups())):
                            try:
                                group_value = match.group(i + 1)
                                match_result['groups'].append(group_value if group_value else '')
                            except IndexError:
                                match_result['groups'].append('')
                    
                    results.append(match_result)
                    
            except Exception as e:
                self.logger.warning(f"Error parsing pattern {pattern_name}: {e}")
                continue
        
        return results

    async def _parse_sql_blocks_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse SQL blocks with host variable analysis - SIMPLIFIED SIGNATURE"""
        chunks = []
        
        sql_matches = list(self.cobol_patterns['sql_block'].finditer(content))
        
        for match in sql_matches:
            sql_content = match.group(1).strip() if match.groups() else match.group(0).strip()
            
            # Only analyze substantial SQL blocks
            if len(sql_content) > 20:
                try:
                    sql_analysis = await self._analyze_with_llm_cached(
                        sql_content, 'cobol_sql_block',
                        """
                        Analyze this embedded SQL block:
                        
                        {content}
                        
                        Identify:
                        1. SQL operation type and complexity
                        2. Host variables used
                        3. Database tables accessed
                        4. Performance characteristics
                        5. Business transaction purpose
                        
                        Return as JSON:
                        {{
                            "operation_type": "select_with_join",
                            "complexity": "medium",
                            "host_variables": [":customer-id", ":customer-name"],
                            "tables_accessed": ["CUSTOMER", "ACCOUNT"],
                            "performance_impact": "medium",
                            "business_purpose": "customer_account_lookup"
                        }}
                        """
                    )
                except Exception as e:
                    self.logger.warning(f"LLM analysis failed for SQL block: {e}")
                    sql_analysis = {
                        'analysis': {'error': str(e), 'operation_type': 'unknown'},
                        'confidence_score': 0.3
                    }
            else:
                sql_analysis = {
                    'analysis': {'operation_type': 'simple_sql'},
                    'confidence_score': 0.6
                }
            
            # Extract host variables safely
            try:
                host_vars = self.cobol_patterns['sql_host_var'].findall(sql_content)
            except:
                host_vars = []
            
            business_context = {
                'sql_type': 'embedded_sql',
                'operation_type': sql_analysis.get('analysis', {}).get('operation_type', ''),
                'business_purpose': sql_analysis.get('analysis', {}).get('business_purpose', ''),
                'host_variable_count': len(host_vars),
                'tables_accessed': sql_analysis.get('analysis', {}).get('tables_accessed', [])
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_SQL_{hash(sql_content)%10000}",
                chunk_type="cobol_sql_block",
                content=match.group(0),
                metadata={
                    'sql_content': sql_content,
                    'host_variables': host_vars,
                    'llm_analysis': sql_analysis.get('analysis', {}),
                    'sql_length': len(sql_content)
                },
                business_context=business_context,
                confidence_score=sql_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_cics_commands_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse CICS commands with transaction context - SIMPLIFIED SIGNATURE"""
        chunks = []
        
        # Major CICS command patterns - simplified approach
        cics_pattern = re.compile(r'EXEC\s+CICS\s+(\w+).*?END-EXEC', re.IGNORECASE | re.DOTALL)
        matches = list(cics_pattern.finditer(content))
        
        for match in matches:
            cmd_type = match.group(1).lower() if match.groups() else 'unknown'
            
            try:
                cics_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'cics_{cmd_type}',
                    """
                    Analyze this CICS command:
                    
                    {content}
                    
                    Identify:
                    1. Transaction flow purpose
                    2. Resource management
                    3. Error handling approach
                    4. User interaction pattern
                    5. Business function
                    
                    Return as JSON:
                    {{
                        "transaction_purpose": "customer_inquiry_response",
                        "resource_management": "file_access",
                        "error_handling": "condition_based",
                        "interaction_pattern": "conversational",
                        "business_function": "customer_service"
                    }}
                    """
                )
            except Exception as e:
                self.logger.warning(f"LLM analysis failed for CICS command: {e}")
                cics_analysis = {
                    'analysis': {'error': str(e), 'transaction_purpose': 'unknown'},
                    'confidence_score': 0.3
                }
            
            business_context = {
                'cics_command': cmd_type,
                'transaction_purpose': cics_analysis.get('analysis', {}).get('transaction_purpose', ''),
                'business_function': cics_analysis.get('analysis', {}).get('business_function', ''),
                'interaction_pattern': cics_analysis.get('analysis', {}).get('interaction_pattern', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_CICS_{cmd_type.upper()}_{hash(match.group(0))%10000}",
                chunk_type=f"cics_{cmd_type}",
                content=match.group(0),
                metadata={
                    'command_type': cmd_type,
                    'llm_analysis': cics_analysis.get('analysis', {}),
                    'command_length': len(match.group(0))
                },
                business_context=business_context,
                confidence_score=cics_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_copy_statements_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse COPY statements with replacement analysis - SIMPLIFIED SIGNATURE"""
        chunks = []
        
        copy_matches = list(self.cobol_patterns['copy_statement'].finditer(content))
        
        for match in copy_matches:
            copybook_name = match.group(1) if match.groups() else 'UNKNOWN'
            library = match.group(2) if match.groups() and len(match.groups()) > 1 else None
            replacing_clause = match.group(3) if match.groups() and len(match.groups()) > 2 else None
            
            try:
                copy_analysis = await self._analyze_with_llm_cached(
                    match.group(0), 'cobol_copy_statement',
                    """
                    Analyze this COPY statement:
                    
                    {content}
                    
                    Identify:
                    1. Copybook purpose and domain
                    2. Replacement strategy (if any)
                    3. Code reuse pattern
                    4. Integration complexity
                    5. Maintenance implications
                    
                    Return as JSON:
                    {{
                        "copybook_purpose": "customer_data_structure",
                        "domain": "customer_management",
                        "replacement_strategy": "parameter_substitution",
                        "reuse_pattern": "standard_record_layout",
                        "maintenance_impact": "low"
                    }}
                    """
                )
            except Exception as e:
                self.logger.warning(f"LLM analysis failed for COPY statement: {e}")
                copy_analysis = {
                    'analysis': {'error': str(e), 'copybook_purpose': 'unknown'},
                    'confidence_score': 0.3
                }
            
            business_context = {
                'copybook_name': copybook_name,
                'library': library,
                'has_replacing': replacing_clause is not None,
                'copybook_purpose': copy_analysis.get('analysis', {}).get('copybook_purpose', ''),
                'domain': copy_analysis.get('analysis', {}).get('domain', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_COPY_{copybook_name}",
                chunk_type="cobol_copy_statement",
                content=match.group(0),
                metadata={
                    'copybook_name': copybook_name,
                    'library': library,
                    'replacing_clause': replacing_clause,
                    'llm_analysis': copy_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=copy_analysis.get('confidence_score', 0.9),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks
    async def _parse_jcl_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced JCL parsing with job flow analysis"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # Get LLM analysis for overall JCL structure
        jcl_analysis = await self._llm_analyze_complex_pattern(content, 'business_logic')
        
        # Parse JOB card
        job_chunks = await self._parse_jcl_job_card(content, program_name, jcl_analysis)
        chunks.extend(job_chunks)
        
        # Parse job steps
        step_chunks = await self._parse_jcl_steps_enhanced(content, program_name, jcl_analysis)
        chunks.extend(step_chunks)
        
        # Parse DD statements
        dd_chunks = await self._parse_jcl_dd_statements_enhanced(content, program_name, jcl_analysis)
        chunks.extend(dd_chunks)
        
        # Parse PROC definitions
        proc_chunks = await self._parse_jcl_proc_definitions(content, program_name, jcl_analysis)
        chunks.extend(proc_chunks)
        
        # Parse control statements
        control_chunks = await self._parse_jcl_control_statements(content, program_name, jcl_analysis)
        chunks.extend(control_chunks)
        
        return chunks

    async def _parse_jcl_job_card(self, content: str, program_name: str, 
                                jcl_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse JCL JOB card with resource analysis"""
        chunks = []
        
        job_match = self.jcl_patterns['job_card'].search(content)
        if job_match:
            job_name = job_match.group(1)
            job_params = job_match.group(2)
            
            job_analysis = await self._analyze_with_llm_cached(
                job_match.group(0), 'jcl_job_card',
                """
                Analyze this JCL JOB card:
                
                {content}
                
                Identify:
                1. Job purpose and business function
                2. Resource requirements
                3. Scheduling characteristics
                4. Security and authorization
                5. Performance expectations
                
                Return as JSON:
                {{
                    "business_function": "daily_customer_processing",
                    "resource_requirements": "high_memory",
                    "scheduling": "batch_daily",
                    "security_level": "standard",
                    "performance_class": "production"
                }}
                """
            )
            
            business_context = {
                'job_name': job_name,
                'business_function': job_analysis.get('analysis', {}).get('business_function', ''),
                'scheduling': job_analysis.get('analysis', {}).get('scheduling', ''),
                'performance_class': job_analysis.get('analysis', {}).get('performance_class', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_JOB_CARD",
                chunk_type="jcl_job_card",
                content=job_match.group(0),
                metadata={
                    'job_name': job_name,
                    'job_parameters': job_params,
                    'llm_analysis': job_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=job_analysis.get('confidence_score', 0.8),
                line_start=content[:job_match.start()].count('\n'),
                line_end=content[:job_match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_steps_enhanced(self, content: str, program_name: str,
                                      jcl_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse JCL job steps with execution flow analysis"""
        chunks = []
        
        step_matches = list(self.jcl_patterns['job_step'].finditer(content))
        
        for i, match in enumerate(step_matches):
            step_name = match.group(1)
            step_params = match.group(2)
            
            step_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'jcl_job_step',
                """
                Analyze this JCL job step:
                
                {content}
                
                Identify:
                1. Program function and purpose
                2. Data processing workflow
                3. Error handling strategy
                4. Resource utilization
                5. Business process integration
                
                Return as JSON:
                {{
                    "program_function": "customer_data_validation",
                    "workflow_type": "data_transformation",
                    "error_handling": "conditional_execution",
                    "resource_usage": "cpu_intensive",
                    "business_integration": "customer_onboarding_process"
                }}
                """
            )
            
            business_context = {
                'step_name': step_name,
                'step_sequence': i + 1,
                'program_function': step_analysis.get('analysis', {}).get('program_function', ''),
                'workflow_type': step_analysis.get('analysis', {}).get('workflow_type', ''),
                'business_integration': step_analysis.get('analysis', {}).get('business_integration', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_STEP_{step_name}",
                chunk_type="jcl_job_step",
                content=match.group(0),
                metadata={
                    'step_name': step_name,
                    'step_parameters': step_params,
                    'step_sequence': i + 1,
                    'llm_analysis': step_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=step_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_dd_statements_enhanced(self, content: str, program_name: str,
                                              jcl_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse JCL DD statements with data flow analysis"""
        chunks = []
        
        dd_matches = list(self.jcl_patterns['dd_statement'].finditer(content))
        
        for match in dd_matches:
            dd_name = match.group(1)
            dd_params = match.group(2)
            
            # Extract dataset name if present
            dsn_match = self.jcl_patterns['dataset'].search(dd_params)
            dataset_name = dsn_match.group(1) if dsn_match else None
            
            dd_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'jcl_dd_statement',
                """
                Analyze this JCL DD statement:
                
                {content}
                
                Identify:
                1. Data flow direction and purpose
                2. Dataset characteristics
                3. Storage and access patterns
                4. Business data category
                5. Performance implications
                
                Return as JSON:
                {{
                    "data_flow": "input_processing",
                    "dataset_purpose": "customer_master_file",
                    "access_pattern": "sequential_read",
                    "data_category": "customer_data",
                    "performance_impact": "io_intensive"
                }}
                """
            )
            
            business_context = {
                'dd_name': dd_name,
                'dataset_name': dataset_name,
                'data_flow': dd_analysis.get('analysis', {}).get('data_flow', ''),
                'dataset_purpose': dd_analysis.get('analysis', {}).get('dataset_purpose', ''),
                'data_category': dd_analysis.get('analysis', {}).get('data_category', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_DD_{dd_name}",
                chunk_type="jcl_dd_statement",
                content=match.group(0),
                metadata={
                    'dd_name': dd_name,
                    'dataset_name': dataset_name,
                    'dd_parameters': dd_params,
                    'llm_analysis': dd_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=dd_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks
    
    # ==================== HELPER METHODS AND UTILITIES ====================

    def _group_data_items_by_hierarchy(self, data_matches: List) -> List[Dict[str, Any]]:
        """Group data items by their hierarchical relationships"""
        groups = []
        current_group = None
        group_id = 1
        
        for match in data_matches:
            try:
                level = int(match.group(1))
                field_name = match.group(2)
                
                # Start new group on level 01
                if level == 1:
                    if current_group:
                        groups.append(current_group)
                    
                    current_group = {
                        'group_id': group_id,
                        'items': [],
                        'min_level': level,
                        'max_level': level,
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    }
                    group_id += 1
                
                if current_group:
                    current_group['items'].append({
                        'level': level,
                        'field_name': field_name,
                        'match': match,
                        'position': match.start()
                    })
                    current_group['max_level'] = max(current_group['max_level'], level)
                    current_group['end_pos'] = match.end()
                    
            except (ValueError, IndexError):
                continue
        
        if current_group:
            groups.append(current_group)
        
        return groups

    async def _parse_jcl_proc_definitions(self, content: str, program_name: str,
                                        jcl_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse JCL PROC definitions"""
        chunks = []
        
        proc_matches = list(self.jcl_patterns['proc_definition'].finditer(content))
        
        for match in proc_matches:
            proc_name = match.group(1)
            proc_params = match.group(2) if match.groups() and len(match.groups()) > 1 else ""
            
            # Find PEND statement
            pend_match = self.jcl_patterns['pend_statement'].search(content, match.end())
            if pend_match:
                proc_content = content[match.start():pend_match.end()]
            else:
                proc_content = match.group(0)
            
            proc_analysis = await self._analyze_with_llm_cached(
                proc_content, 'jcl_proc_definition',
                """
                Analyze this JCL PROC definition:
                
                {content}
                
                Identify:
                1. Procedure purpose and reusability
                2. Parameter substitution strategy
                3. Step sequence and dependencies
                4. Business process automation
                5. Maintenance characteristics
                
                Return as JSON:
                {{
                    "procedure_purpose": "customer_data_backup",
                    "reusability": "high",
                    "parameter_strategy": "symbolic_substitution",
                    "process_automation": "scheduled_maintenance",
                    "maintenance_level": "low"
                }}
                """
            )
            
            business_context = {
                'proc_name': proc_name,
                'procedure_purpose': proc_analysis.get('analysis', {}).get('procedure_purpose', ''),
                'reusability': proc_analysis.get('analysis', {}).get('reusability', ''),
                'process_automation': proc_analysis.get('analysis', {}).get('process_automation', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_PROC_{proc_name}",
                chunk_type="jcl_proc_definition",
                content=proc_content,
                metadata={
                    'proc_name': proc_name,
                    'proc_parameters': proc_params,
                    'llm_analysis': proc_analysis.get('analysis', {}),
                    'proc_length': len(proc_content.split('\n'))
                },
                business_context=business_context,
                confidence_score=proc_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(proc_content)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_jcl_control_statements(self, content: str, program_name: str,
                                          jcl_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse JCL control statements (IF/THEN/ELSE/ENDIF)"""
        chunks = []
        
        # Find IF statements
        if_matches = list(self.jcl_patterns['if_statement'].finditer(content))
        
        for match in if_matches:
            if_condition = match.group(1)
            
            # Find corresponding ENDIF
            endif_match = self.jcl_patterns['endif_statement'].search(content, match.end())
            if endif_match:
                control_block = content[match.start():endif_match.end()]
            else:
                control_block = match.group(0)
            
            control_analysis = await self._analyze_with_llm_cached(
                control_block, 'jcl_control_flow',
                """
                Analyze this JCL control flow:
                
                {content}
                
                Identify:
                1. Conditional logic purpose
                2. Error handling strategy
                3. Business rule implementation
                4. Execution flow complexity
                5. Recovery mechanisms
                
                Return as JSON:
                {{
                    "logic_purpose": "step_failure_handling",
                    "error_strategy": "conditional_restart",
                    "business_rules": ["data_validation_required"],
                    "complexity": "medium",
                    "recovery_mechanism": "automatic_retry"
                }}
                """
            )
            
            business_context = {
                'control_type': 'conditional_execution',
                'logic_purpose': control_analysis.get('analysis', {}).get('logic_purpose', ''),
                'error_strategy': control_analysis.get('analysis', {}).get('error_strategy', ''),
                'complexity': control_analysis.get('analysis', {}).get('complexity', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_CONTROL_{hash(if_condition)%10000}",
                chunk_type="jcl_control_flow",
                content=control_block,
                metadata={
                    'condition': if_condition,
                    'llm_analysis': control_analysis.get('analysis', {}),
                    'control_length': len(control_block.split('\n'))
                },
                business_context=business_context,
                confidence_score=control_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.start() + len(control_block)].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_bms_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced BMS parsing with map layout analysis"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # Parse mapset definitions
        mapset_chunks = await self._parse_bms_mapsets_enhanced(content, program_name)
        chunks.extend(mapset_chunks)
        
        # Parse map definitions
        map_chunks = await self._parse_bms_maps_enhanced(content, program_name)
        chunks.extend(map_chunks)
        
        # Parse field definitions
        field_chunks = await self._parse_bms_fields_enhanced(content, program_name)
        chunks.extend(field_chunks)
        
        return chunks

    async def _parse_bms_mapsets_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse BMS mapset definitions with UI analysis"""
        chunks = []
        
        mapset_matches = list(self.bms_patterns['bms_mapset'].finditer(content))
        
        for match in mapset_matches:
            mapset_name = match.group(1)
            mapset_params = match.group(2)
            
            mapset_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'bms_mapset',
                """
                Analyze this BMS mapset definition:
                
                {content}
                
                Identify:
                1. User interface purpose and design
                2. Terminal characteristics and compatibility
                3. Screen layout complexity
                4. Business function served
                5. User interaction patterns
                
                Return as JSON:
                {{
                    "ui_purpose": "customer_inquiry_screen",
                    "terminal_type": "3270",
                    "layout_complexity": "medium",
                    "business_function": "customer_service",
                    "interaction_pattern": "form_based"
                }}
                """
            )
            
            business_context = {
                'mapset_name': mapset_name,
                'ui_purpose': mapset_analysis.get('analysis', {}).get('ui_purpose', ''),
                'business_function': mapset_analysis.get('analysis', {}).get('business_function', ''),
                'interaction_pattern': mapset_analysis.get('analysis', {}).get('interaction_pattern', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_MAPSET_{mapset_name}",
                chunk_type="bms_mapset",
                content=match.group(0),
                metadata={
                    'mapset_name': mapset_name,
                    'mapset_parameters': mapset_params,
                    'llm_analysis': mapset_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=mapset_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_bms_maps_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse BMS map definitions with screen layout analysis"""
        chunks = []
        
        map_matches = list(self.bms_patterns['bms_map'].finditer(content))
        
        for match in map_matches:
            map_name = match.group(1)
            map_params = match.group(2)
            
            # Extract size information
            size_match = self.bms_patterns['bms_map_params'].search(map_params)
            screen_size = (size_match.group(1), size_match.group(2)) if size_match and size_match.group(1) else None
            
            map_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'bms_map',
                """
                Analyze this BMS map definition:
                
                {content}
                
                Identify:
                1. Screen layout structure and design
                2. User workflow facilitated
                3. Data entry patterns
                4. Navigation and usability
                5. Business process integration
                
                Return as JSON:
                {{
                    "layout_structure": "tabular_data_entry",
                    "user_workflow": "customer_information_update",
                    "data_entry_pattern": "sequential_fields",
                    "navigation": "function_key_driven",
                    "process_integration": "customer_maintenance"
                }}
                """
            )
            
            business_context = {
                'map_name': map_name,
                'screen_size': screen_size,
                'layout_structure': map_analysis.get('analysis', {}).get('layout_structure', ''),
                'user_workflow': map_analysis.get('analysis', {}).get('user_workflow', ''),
                'process_integration': map_analysis.get('analysis', {}).get('process_integration', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_MAP_{map_name}",
                chunk_type="bms_map",
                content=match.group(0),
                metadata={
                    'map_name': map_name,
                    'screen_size': screen_size,
                    'map_parameters': map_params,
                    'llm_analysis': map_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=map_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_bms_fields_enhanced(self, content: str, program_name: str) -> List[CodeChunk]:
        """Parse BMS field definitions with UI element analysis"""
        chunks = []
        
        field_matches = list(self.bms_patterns['bms_field'].finditer(content))
        
        # Group fields by their maps
        field_groups = self._group_bms_fields_by_map(field_matches, content)
        
        for group in field_groups:
            group_content = '\n'.join([field['match'].group(0) for field in group['fields']])
            
            field_analysis = await self._analyze_with_llm_cached(
                group_content, 'bms_field_group',
                """
                Analyze this group of BMS fields:
                
                {content}
                
                Identify:
                1. Field grouping purpose and organization
                2. Data validation and input controls
                3. User interaction design
                4. Business data representation
                5. Accessibility and usability features
                
                Return as JSON:
                {{
                    "grouping_purpose": "customer_contact_information",
                    "validation_controls": ["required_fields", "format_validation"],
                    "interaction_design": "tabbed_interface",
                    "data_representation": "customer_profile",
                    "usability_features": ["field_highlighting", "error_indication"]
                }}
                """
            )
            
            business_context = {
                'field_group_type': 'ui_section',
                'grouping_purpose': field_analysis.get('analysis', {}).get('grouping_purpose', ''),
                'data_representation': field_analysis.get('analysis', {}).get('data_representation', ''),
                'field_count': len(group['fields'])
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_FIELDS_{group['group_id']}",
                chunk_type="bms_field_group",
                content=group_content,
                metadata={
                    'field_count': len(group['fields']),
                    'field_names': [field['field_name'] for field in group['fields']],
                    'llm_analysis': field_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=field_analysis.get('confidence_score', 0.8),
                line_start=content[:group['start_pos']].count('\n'),
                line_end=content[:group['end_pos']].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _group_bms_fields_by_map(self, field_matches: List, content: str) -> List[Dict[str, Any]]:
        """Group BMS fields by their containing maps"""
        groups = []
        current_group = None
        group_id = 1
        
        # Find map boundaries
        map_matches = list(self.bms_patterns['bms_map'].finditer(content))
        
        for field_match in field_matches:
            field_name = field_match.group(1)
            field_pos = field_match.start()
            
            # Find which map this field belongs to
            containing_map = None
            for map_match in map_matches:
                if map_match.start() < field_pos:
                    # Check if there's another map between this one and the field
                    next_map = None
                    for next_map_match in map_matches:
                        if next_map_match.start() > map_match.start() and next_map_match.start() < field_pos:
                            next_map = next_map_match
                            break
                    
                    if not next_map:
                        containing_map = map_match
            
            # Group fields by map
            if containing_map:
                map_name = containing_map.group(1)
                
                # Check if we need to start a new group
                if not current_group or current_group.get('map_name') != map_name:
                    if current_group:
                        groups.append(current_group)
                    
                    current_group = {
                        'group_id': group_id,
                        'map_name': map_name,
                        'fields': [],
                        'start_pos': field_pos,
                        'end_pos': field_pos
                    }
                    group_id += 1
                
                current_group['fields'].append({
                    'field_name': field_name,
                    'match': field_match,
                    'position': field_pos
                })
                current_group['end_pos'] = field_match.end()
        
        if current_group:
            groups.append(current_group)
        
        return groups

    async def _parse_cics_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced CICS parsing - wrapper around COBOL parsing with CICS focus"""
        # CICS programs are essentially COBOL programs with CICS commands
        # Parse as COBOL first, then add CICS-specific analysis
        return await self._parse_cobol_with_enhanced_analysis(content, filename)

    # ==================== MAIN PROCESSING METHOD ====================
    

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
    
        """Process a single code file with enhanced business rule validation and LLM analysis"""
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
            
            # Parse based on file type with enhanced business context and LLM analysis
            if file_type == 'cobol':
                chunks = await self._parse_cobol_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'jcl':
                chunks = await self._parse_jcl_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'copybook':
                chunks = await self._parse_copybook_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'bms':
                chunks = await self._parse_bms_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'cics':
                chunks = await self._parse_cics_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'db2_procedure':
                chunks = await self._parse_db2_procedure_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'cobol_stored_procedure':
                chunks = await self._parse_cobol_stored_procedure_with_enhanced_analysis(content, str(file_path.name))
            elif file_type == 'mq_program':
                chunks = await self._parse_mq_program_with_enhanced_analysis(content, str(file_path.name))
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

            # Generate control flow analysis for applicable file types
            if file_type in ['cobol', 'cics', 'cobol_stored_procedure']:
                control_flow = await self._analyze_control_flow(chunks)
                await self._store_control_flow_analysis(control_flow, self._extract_program_name(content, file_path))
            
            # Generate and store field lineage for data-intensive file types
            if file_type in ['cobol', 'cics', 'copybook', 'cobol_stored_procedure']:
                lineage_records = await self._generate_field_lineage(self._extract_program_name(content, file_path), chunks)
                await self._store_field_lineage(lineage_records)
            
            # Store specialized analysis for specific file types
            if file_type == 'copybook':
                await self._store_copybook_analysis(chunks, self._extract_program_name(content, file_path))
            elif file_type == 'mq_program':
                await self._store_mq_analysis(chunks, content, self._extract_program_name(content, file_path))
            elif file_type == 'db2_procedure':
                await self._store_db2_procedure_analysis(chunks, self._extract_program_name(content, file_path))
            
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
            
            if result.get("status") == "success":
                await self._extract_and_store_program_relationships(content, file_path.name, file_type)
                await self._extract_and_store_copybook_relationships(content, file_path.name)
                await self._extract_and_store_file_relationships(content, file_path.name)
                await self._extract_and_store_field_cross_reference(content, file_path.name, file_type)
                
                # Generate impact analysis
                await self._generate_impact_analysis(file_path.name, file_type)


            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {str(e)}")
            return self._add_processing_info({
                "status": "error",
                "file_name": file_path.name,
                "error": str(e)
            })
    
    async def _generate_impact_analysis(self, artifact_name: str, artifact_type: str):
        """Generate impact analysis for an artifact"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            impacts = []
            
            if artifact_type in ['cobol', 'cics']:
                # Programs that call this program
                cursor.execute("""
                    SELECT DISTINCT calling_program, call_type 
                    FROM program_relationships 
                    WHERE called_program = ?
                """, (artifact_name,))
                
                for row in cursor.fetchall():
                    impacts.append({
                        'source_artifact': row[0],
                        'source_type': 'program',
                        'dependent_artifact': artifact_name,
                        'dependent_type': artifact_type,
                        'relationship_type': f'calls_via_{row[1]}',
                        'impact_level': 'high',
                        'change_propagation': 'interface_change_required'
                    })
                
                # Programs that use same copybooks
                cursor.execute("""
                    SELECT DISTINCT cr2.program_name, cr1.copybook_name
                    FROM copybook_relationships cr1
                    JOIN copybook_relationships cr2 ON cr1.copybook_name = cr2.copybook_name
                    WHERE cr1.program_name = ? AND cr2.program_name != ?
                """, (artifact_name, artifact_name))
                
                for row in cursor.fetchall():
                    impacts.append({
                        'source_artifact': artifact_name,
                        'source_type': artifact_type,
                        'dependent_artifact': row[0],
                        'dependent_type': 'program',
                        'relationship_type': f'shares_copybook_{row[1]}',
                        'impact_level': 'medium',
                        'change_propagation': 'data_structure_dependency'
                    })
            
            # Store impact analysis
            for impact in impacts:
                cursor.execute("""
                    INSERT OR REPLACE INTO impact_analysis 
                    (source_artifact, source_type, dependent_artifact, dependent_type,
                    relationship_type, impact_level, change_propagation)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    impact['source_artifact'],
                    impact['source_type'],
                    impact['dependent_artifact'],
                    impact['dependent_type'],
                    impact['relationship_type'],
                    impact['impact_level'],
                    impact['change_propagation']
                ))
            
            conn.commit()
        except Exception as e:
            self.logger.error(f"Processing failed for {artifact_name}: {str(e)}")
            return self._add_processing_info({
                "status": "error",
                "file_name": artifact_name,
                "error": str(e)
            })

    # ==================== MISSING ENHANCED PARSING METHODS ====================

    async def _parse_cobol_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced COBOL parsing with comprehensive LLM analysis - FIXED VERSION"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        self.logger.info(f"🔍 Starting enhanced COBOL analysis for {program_name}")
        
        try:
            # Get LLM analysis for overall program structure
            program_analysis = await self._llm_analyze_complex_pattern(content, 'business_logic')
            self.logger.info(f"✅ Program analysis completed for {program_name}")
        except Exception as e:
            self.logger.warning(f"Program analysis failed for {program_name}: {e}")
            program_analysis = {'analysis': {}, 'confidence_score': 0.5}
        
        # Parse divisions with enhanced validation and LLM insights
        try:
            self.logger.info(f"📊 Parsing divisions for {program_name}")
            division_chunks = await self._parse_cobol_divisions_enhanced(content, program_name, program_analysis)
            chunks.extend(division_chunks)
            self.logger.info(f"✅ Parsed {len(division_chunks)} division chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"Division parsing failed for {program_name}: {e}")
        
        # Parse sections with comprehensive context - PASS program_analysis
        try:
            self.logger.info(f"📊 Parsing sections for {program_name}")
            section_chunks = await self._parse_cobol_sections_enhanced(content, program_name, program_analysis)
            chunks.extend(section_chunks)
            self.logger.info(f"✅ Parsed {len(section_chunks)} section chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"Section parsing failed for {program_name}: {e}")
        
        # Parse data items with advanced business rule validation - PASS program_analysis
        try:
            self.logger.info(f"📊 Parsing data items for {program_name}")
            data_chunks = await self._parse_data_items_enhanced(content, program_name, program_analysis)
            chunks.extend(data_chunks)
            self.logger.info(f"✅ Parsed {len(data_chunks)} data chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"Data parsing failed for {program_name}: {e}")
        
        # Parse procedure division with comprehensive flow analysis - PASS program_analysis
        try:
            self.logger.info(f"📊 Parsing procedure division for {program_name}")
            procedure_chunks = await self._parse_procedure_division_enhanced(content, program_name, program_analysis)
            chunks.extend(procedure_chunks)
            self.logger.info(f"✅ Parsed {len(procedure_chunks)} procedure chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"Procedure division parsing failed for {program_name}: {e}")
        
        # Parse SQL blocks with enhanced host variable validation - NO extra parameters
        try:
            self.logger.info(f"📊 Parsing SQL blocks for {program_name}")
            sql_chunks = await self._parse_sql_blocks_enhanced(content, program_name)
            chunks.extend(sql_chunks)
            self.logger.info(f"✅ Parsed {len(sql_chunks)} SQL chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"SQL parsing failed for {program_name}: {e}")
        
        # Parse CICS commands with transaction context - NO extra parameters
        try:
            self.logger.info(f"📊 Parsing CICS commands for {program_name}")
            cics_chunks = await self._parse_cics_commands_enhanced(content, program_name)
            chunks.extend(cics_chunks)
            self.logger.info(f"✅ Parsed {len(cics_chunks)} CICS chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"CICS parsing failed for {program_name}: {e}")
        
        # Parse COPY statements with replacement analysis - NO extra parameters
        try:
            self.logger.info(f"📊 Parsing COPY statements for {program_name}")
            copy_chunks = await self._parse_copy_statements_enhanced(content, program_name)
            chunks.extend(copy_chunks)
            self.logger.info(f"✅ Parsed {len(copy_chunks)} COPY chunks for {program_name}")
        except Exception as e:
            self.logger.warning(f"COPY parsing failed for {program_name}: {e}")
        
        self.logger.info(f"🎯 Total chunks created for {program_name}: {len(chunks)}")
        return chunks
    async def _parse_copybook_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced copybook parsing with comprehensive layout analysis"""
        chunks = []
        copybook_name = self._extract_program_name(content, Path(filename))
        
        # Analyze copybook structure with LLM
        structure_analysis = await self._llm_analyze_complex_pattern(content, 'data_relationships')
        
        # Detect layout type and complexity
        layout_info = await self._analyze_copybook_layout(content, copybook_name)
        
        # Parse multi-layout copybooks
        if layout_info['layout_type'] == CopybookLayoutType.MULTI_RECORD:
            layout_chunks = await self._parse_multi_record_layouts(content, copybook_name, structure_analysis)
            chunks.extend(layout_chunks)
        
        # Parse conditional layouts based on indicators
        elif layout_info['layout_type'] == CopybookLayoutType.CONDITIONAL_LAYOUT:
            conditional_chunks = await self._parse_conditional_layouts(content, copybook_name, structure_analysis)
            chunks.extend(conditional_chunks)
        
        # Parse REDEFINES structures
        elif layout_info['layout_type'] == CopybookLayoutType.REDEFINES_LAYOUT:
            redefines_chunks = await self._parse_redefines_structures(content, copybook_name, structure_analysis)
            chunks.extend(redefines_chunks)
        
        # Parse OCCURS structures
        elif layout_info['layout_type'] == CopybookLayoutType.OCCURS_LAYOUT:
            occurs_chunks = await self._parse_occurs_structures(content, copybook_name, structure_analysis)
            chunks.extend(occurs_chunks)
        
        # Default single record layout
        else:
            single_chunks = await self._parse_single_record_layout(content, copybook_name, structure_analysis)
            chunks.extend(single_chunks)
        
        # Parse REPLACING parameters
        replacing_chunks = await self._parse_replacing_parameters(content, copybook_name)
        chunks.extend(replacing_chunks)
        
        # Parse multi-filler patterns for alignment
        filler_chunks = await self._parse_multi_filler_patterns(content, copybook_name)
        chunks.extend(filler_chunks)
        
        return chunks

    async def _parse_single_record_layout(self, content: str, copybook_name: str,
                                        structure_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse single record layout copybook"""
        chunks = []
        
        # Find all data items
        data_matches = list(self.copybook_patterns['record_type_field'].finditer(content))
        if not data_matches:
            # Fallback to general data item pattern
            data_matches = list(self.cobol_patterns['data_item'].finditer(content))
        
        if data_matches:
            # Group related data items
            data_groups = self._group_data_items_by_hierarchy(data_matches)
            
            for group in data_groups:
                group_content = '\n'.join([item['match'].group(0) for item in group['items']])
                
                # Analyze with LLM
                layout_analysis = await self._analyze_with_llm_cached(
                    group_content, 'single_record_layout',
                    """
                    Analyze this single record layout:
                    
                    {content}
                    
                    Identify:
                    1. Business entity represented
                    2. Data organization pattern
                    3. Field relationships
                    4. Usage context
                    
                    Return as JSON:
                    {{
                        "business_entity": "customer_record",
                        "organization_pattern": "hierarchical",
                        "field_relationships": ["parent_child", "sequential"],
                        "usage_context": "transaction_processing"
                    }}
                    """
                )
                
                business_context = {
                    'layout_type': 'single_record',
                    'business_entity': layout_analysis.get('analysis', {}).get('business_entity', ''),
                    'organization_pattern': layout_analysis.get('analysis', {}).get('organization_pattern', ''),
                    'field_count': len(group['items'])
                }
                
                chunk = CodeChunk(
                    program_name=copybook_name,
                    chunk_id=f"{copybook_name}_SINGLE_LAYOUT_{group['group_id']}",
                    chunk_type="copybook_single_layout",
                    content=group_content,
                    metadata={
                        'layout_type': 'single_record',
                        'field_count': len(group['items']),
                        'llm_analysis': layout_analysis.get('analysis', {})
                    },
                    business_context=business_context,
                    confidence_score=layout_analysis.get('confidence_score', 0.8),
                    line_start=content[:group['start_pos']].count('\n'),
                    line_end=content[:group['end_pos']].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_mq_program_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced IBM MQ program parsing with comprehensive message flow analysis"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # First parse as COBOL with MQ enhancements
        base_cobol_chunks = await self._parse_cobol_with_enhanced_analysis(content, filename)
        chunks.extend(base_cobol_chunks)
        
        # Analyze MQ-specific patterns with LLM
        mq_analysis = await self._llm_analyze_mq_patterns(content)
        
        # Parse MQ API call sequences
        api_chunks = await self._parse_mq_api_sequences(content, program_name, mq_analysis)
        chunks.extend(api_chunks)
        
        # Parse MQ data structures with comprehensive field analysis
        structure_chunks = await self._parse_mq_data_structures_enhanced(content, program_name, mq_analysis)
        chunks.extend(structure_chunks)
        
        # Parse message flow patterns
        flow_chunks = await self._parse_mq_message_flows(content, program_name, mq_analysis)
        chunks.extend(flow_chunks)
        
        # Parse error handling patterns specific to MQ
        error_chunks = await self._parse_mq_error_handling(content, program_name, mq_analysis)
        chunks.extend(error_chunks)
        
        # Parse transaction scope and commit patterns
        transaction_chunks = await self._parse_mq_transaction_patterns(content, program_name, mq_analysis)
        chunks.extend(transaction_chunks)
        
        return chunks

    async def _parse_mq_data_structures_enhanced(self, content: str, program_name: str,
                                               mq_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse MQ data structures with enhanced analysis"""
        chunks = []
        
        # MQ data structure patterns
        mq_structures = {
            'message_descriptor': self.mq_patterns['mq_message_descriptor'],
            'object_descriptor': self.mq_patterns['mq_object_descriptor'],
            'put_options': self.mq_patterns['mq_put_options'],
            'get_options': self.mq_patterns['mq_get_options'],
            'connection_options': self.mq_patterns['mq_connection_options']
        }
        
        for struct_type, pattern in mq_structures.items():
            matches = list(pattern.finditer(content))
            
            for match in matches:
                struct_name = match.group(1) if match.groups() else 'unknown'
                
                # Find the complete structure definition
                start_pos = match.start()
                end_pos = self._find_structure_end(content, start_pos, struct_name)
                structure_content = content[start_pos:end_pos]
                
                struct_analysis = await self._analyze_with_llm_cached(
                    structure_content, f'mq_{struct_type}',
                    """
                    Analyze this MQ data structure:
                    
                    {content}
                    
                    Identify:
                    1. Structure purpose and usage
                    2. Key fields and their roles
                    3. Message processing context
                    4. Performance implications
                    
                    Return as JSON:
                    {{
                        "structure_purpose": "message_control_block",
                        "key_fields": ["format", "type", "persistence"],
                        "processing_context": "async_messaging",
                        "performance_impact": "low"
                    }}
                    """
                )
                
                business_context = {
                    'structure_type': struct_type,
                    'structure_name': struct_name,
                    'structure_purpose': struct_analysis.get('analysis', {}).get('structure_purpose', ''),
                    'processing_context': struct_analysis.get('analysis', {}).get('processing_context', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_MQ_STRUCT_{struct_type}_{struct_name}",
                    chunk_type=f"mq_{struct_type}",
                    content=structure_content,
                    metadata={
                        'structure_type': struct_type,
                        'structure_name': struct_name,
                        'llm_analysis': struct_analysis.get('analysis', {})
                    },
                    business_context=business_context,
                    confidence_score=struct_analysis.get('confidence_score', 0.8),
                    line_start=content[:start_pos].count('\n'),
                    line_end=content[:end_pos].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_mq_message_flows(self, content: str, program_name: str,
                                    mq_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse MQ message flow patterns"""
        chunks = []
        
        # Look for message flow patterns
        flow_patterns = [
            (r'MQPUT.*?MQGET', 'request_reply'),
            (r'MQPUT1', 'fire_and_forget'),
            (r'MQSUB.*?MQGET', 'publish_subscribe'),
            (r'MQOPEN.*?MQPUT.*?MQCLOSE', 'persistent_messaging'),
            (r'MQBEGIN.*?MQPUT.*?MQCMIT', 'transactional_messaging')
        ]
        
        for pattern_regex, flow_type in flow_patterns:
            matches = list(re.finditer(pattern_regex, content, re.IGNORECASE | re.DOTALL))
            
            for match in matches:
                flow_content = match.group(0)
                
                flow_analysis = await self._analyze_with_llm_cached(
                    flow_content, f'mq_message_flow_{flow_type}',
                    """
                    Analyze this MQ message flow pattern:
                    
                    {content}
                    
                    Identify:
                    1. Flow characteristics and reliability
                    2. Error handling strategy
                    3. Performance considerations
                    4. Business process alignment
                    
                    Return as JSON:
                    {{
                        "flow_characteristics": "async_reliable",
                        "error_handling": "exception_based",
                        "performance": "high_throughput",
                        "business_alignment": "order_processing"
                    }}
                    """
                )
                
                business_context = {
                    'flow_type': flow_type,
                    'flow_characteristics': flow_analysis.get('analysis', {}).get('flow_characteristics', ''),
                    'business_alignment': flow_analysis.get('analysis', {}).get('business_alignment', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_MQ_FLOW_{flow_type}_{hash(flow_content)%10000}",
                    chunk_type="mq_message_flow",
                    content=flow_content,
                    metadata={
                        'flow_type': flow_type,
                        'llm_analysis': flow_analysis.get('analysis', {})
                    },
                    business_context=business_context,
                    confidence_score=flow_analysis.get('confidence_score', 0.7),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_mq_error_handling(self, content: str, program_name: str,
                                     mq_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse MQ error handling patterns"""
        chunks = []
        
        # Find error handling patterns
        error_patterns = list(self.mq_patterns['mq_error_check'].finditer(content))
        error_patterns.extend(list(self.mq_patterns['mq_rc_check'].finditer(content)))
        
        for match in error_patterns:
            error_field = match.group(1) if match.groups() else 'unknown'
            
            # Get surrounding context for error handling
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(content), match.end() + 200)
            error_context = content[start_pos:end_pos]
            
            error_analysis = await self._analyze_with_llm_cached(
                error_context, 'mq_error_handling',
                """
                Analyze this MQ error handling:
                
                {content}
                
                Identify:
                1. Error detection strategy
                2. Recovery mechanisms
                3. Business impact handling
                4. Logging and monitoring
                
                Return as JSON:
                {{
                    "detection_strategy": "return_code_checking",
                    "recovery_mechanisms": ["retry", "deadletter"],
                    "business_impact": "transaction_rollback",
                    "monitoring": "comprehensive"
                }}
                """
            )
            
            business_context = {
                'error_type': 'mq_error_handling',
                'error_field': error_field,
                'detection_strategy': error_analysis.get('analysis', {}).get('detection_strategy', ''),
                'business_impact': error_analysis.get('analysis', {}).get('business_impact', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_MQ_ERROR_{error_field}",
                chunk_type="mq_error_handling",
                content=error_context,
                metadata={
                    'error_field': error_field,
                    'llm_analysis': error_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=error_analysis.get('confidence_score', 0.8),
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_mq_transaction_patterns(self, content: str, program_name: str,
                                           mq_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse MQ transaction patterns"""
        chunks = []
        
        # Find transaction patterns
        transaction_patterns = [
            (self.mq_patterns['mq_mqbegin'], 'transaction_begin'),
            (self.mq_patterns['mq_mqcmit'], 'transaction_commit'),
            (self.mq_patterns['mq_mqback'], 'transaction_rollback')
        ]
        
        for pattern, trans_type in transaction_patterns:
            matches = list(pattern.finditer(content))
            
            for match in matches:
                trans_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'mq_transaction_{trans_type}',
                    """
                    Analyze this MQ transaction operation:
                    
                    {content}
                    
                    Identify:
                    1. Transaction scope and boundaries
                    2. Consistency guarantees
                    3. Recovery implications
                    4. Performance impact
                    
                    Return as JSON:
                    {{
                        "transaction_scope": "local_unit_of_work",
                        "consistency": "strict_acid",
                        "recovery": "automatic_rollback",
                        "performance_impact": "medium"
                    }}
                    """
                )
                
                business_context = {
                    'transaction_type': trans_type,
                    'transaction_scope': trans_analysis.get('analysis', {}).get('transaction_scope', ''),
                    'consistency': trans_analysis.get('analysis', {}).get('consistency', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_MQ_TRANS_{trans_type}_{hash(match.group(0))%10000}",
                    chunk_type="mq_transaction_pattern",
                    content=match.group(0),
                    metadata={
                        'transaction_type': trans_type,
                        'llm_analysis': trans_analysis.get('analysis', {})
                    },
                    business_context=business_context,
                    confidence_score=trans_analysis.get('confidence_score', 0.8),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_db2_procedure_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced DB2 stored procedure parsing with comprehensive SQL analysis"""
        chunks = []
        procedure_name = self._extract_program_name(content, Path(filename))
        
        # Analyze DB2 procedure structure with LLM
        db2_analysis = await self._llm_analyze_db2_patterns(content)
        
        # Parse procedure signature and parameters
        signature_chunk = await self._parse_db2_procedure_signature(content, procedure_name, db2_analysis)
        if signature_chunk:
            chunks.append(signature_chunk)
        
        # Parse variable declarations with type analysis
        declaration_chunks = await self._parse_db2_declarations_enhanced(content, procedure_name, db2_analysis)
        chunks.extend(declaration_chunks)
        
        # Parse cursor definitions with SQL analysis
        cursor_chunks = await self._parse_db2_cursors_enhanced(content, procedure_name, db2_analysis)
        chunks.extend(cursor_chunks)
        
        # Parse exception handlers with comprehensive error analysis
        handler_chunks = await self._parse_db2_exception_handlers_enhanced(content, procedure_name, db2_analysis)
        chunks.extend(handler_chunks)
        
        # Parse SQL statements with performance analysis
        sql_chunks = await self._parse_db2_sql_statements_enhanced(content, procedure_name, db2_analysis)
        chunks.extend(sql_chunks)
        
        # Parse control flow structures
        control_chunks = await self._parse_db2_control_flow_enhanced(content, procedure_name, db2_analysis)
        chunks.extend(control_chunks)
        
        # Parse dynamic SQL patterns
        dynamic_chunks = await self._parse_db2_dynamic_sql(content, procedure_name, db2_analysis)
        chunks.extend(dynamic_chunks)
        
        return chunks

    async def _parse_db2_procedure_signature(self, content: str, procedure_name: str,
                                           db2_analysis: Dict[str, Any]) -> Optional[CodeChunk]:
        """Parse DB2 procedure signature"""
        
        # Find CREATE PROCEDURE statement
        proc_match = self.db2_patterns['db2_create_procedure'].search(content)
        if not proc_match:
            return None
        
        # Extract full procedure signature
        signature_start = proc_match.start()
        signature_end = content.find('BEGIN', signature_start)
        if signature_end == -1:
            signature_end = signature_start + 500  # Fallback
        
        signature_content = content[signature_start:signature_end].strip()
        
        # Analyze signature with LLM
        signature_analysis = await self._analyze_with_llm_cached(
            signature_content, 'db2_procedure_signature',
            """
            Analyze this DB2 procedure signature:
            
            {content}
            
            Identify:
            1. Parameter types and purposes
            2. Return value characteristics
            3. Security and access patterns
            4. Performance characteristics
            
            Return as JSON:
            {{
                "parameter_types": ["input", "output", "inout"],
                "parameter_purposes": ["customer_id", "result_set"],
                "return_characteristics": "single_value",
                "security_level": "standard",
                "performance_class": "query_intensive"
            }}
            """
        )
        
        business_context = {
            'signature_type': 'procedure_definition',
            'parameter_types': signature_analysis.get('analysis', {}).get('parameter_types', []),
            'performance_class': signature_analysis.get('analysis', {}).get('performance_class', 'unknown')
        }
        
        return CodeChunk(
            program_name=procedure_name,
            chunk_id=f"{procedure_name}_SIGNATURE",
            chunk_type="db2_procedure_signature",
            content=signature_content,
            metadata={
                'procedure_name': procedure_name,
                'llm_analysis': signature_analysis.get('analysis', {})
            },
            business_context=business_context,
            confidence_score=signature_analysis.get('confidence_score', 0.9),
            line_start=content[:signature_start].count('\n'),
            line_end=content[:signature_end].count('\n')
        )

    def _find_structure_end(self, content: str, start_pos: int, struct_name: str) -> int:
        """Find the end of a data structure definition"""
        lines = content[start_pos:].split('\n')
        end_pos = start_pos
        level_stack = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('*'):
                continue
            
            # Look for level numbers
            level_match = re.match(r'^\s*(\d+)', stripped_line)
            if level_match:
                level = int(level_match.group(1))
                
                if i == 0:  # First line, establish base level
                    level_stack = [level]
                else:
                    # Check if we're back to same or higher level (end of structure)
                    if level <= level_stack[0] and i > 0:
                        break
            
            end_pos += len(line) + 1  # +1 for newline
        
        return min(start_pos + end_pos, len(content))
    
    # ==================== MISSING DB2 PARSING METHODS ====================

    async def _parse_db2_declarations_enhanced(self, content: str, procedure_name: str,
                                             db2_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse DB2 variable declarations with type analysis"""
        chunks = []
        
        # Find variable declarations
        var_matches = list(self.db2_patterns['db2_declare_variable'].finditer(content))
        condition_matches = list(self.db2_patterns['db2_declare_condition'].finditer(content))
        
        for match in var_matches:
            var_name = match.group(1)
            var_type = match.group(2)
            default_value = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
            
            var_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'db2_variable_declaration',
                """
                Analyze this DB2 variable declaration:
                
                {content}
                
                Identify:
                1. Variable purpose and usage pattern
                2. Data type appropriateness
                3. Default value significance
                4. Performance implications
                
                Return as JSON:
                {{
                    "variable_purpose": "temporary_storage",
                    "usage_pattern": "loop_counter",
                    "type_appropriateness": "optimal",
                    "performance_impact": "minimal"
                }}
                """
            )
            
            business_context = {
                'declaration_type': 'variable',
                'variable_name': var_name,
                'variable_type': var_type,
                'variable_purpose': var_analysis.get('analysis', {}).get('variable_purpose', ''),
                'has_default': default_value is not None
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_VAR_{var_name}",
                chunk_type="db2_variable_declaration",
                content=match.group(0),
                metadata={
                    'variable_name': var_name,
                    'variable_type': var_type,
                    'default_value': default_value,
                    'llm_analysis': var_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=var_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        # Process condition declarations
        for match in condition_matches:
            condition_name = match.group(1)
            condition_for = match.group(2)
            
            condition_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'db2_condition_declaration',
                """
                Analyze this DB2 condition declaration:
                
                {content}
                
                Identify:
                1. Error handling strategy
                2. Condition scope and usage
                3. Recovery implications
                4. Business impact
                
                Return as JSON:
                {{
                    "error_strategy": "specific_handling",
                    "condition_scope": "procedure_level",
                    "recovery_approach": "graceful_degradation",
                    "business_impact": "low"
                }}
                """
            )
            
            business_context = {
                'declaration_type': 'condition',
                'condition_name': condition_name,
                'condition_for': condition_for,
                'error_strategy': condition_analysis.get('analysis', {}).get('error_strategy', '')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_COND_{condition_name}",
                chunk_type="db2_condition_declaration",
                content=match.group(0),
                metadata={
                    'condition_name': condition_name,
                    'condition_for': condition_for,
                    'llm_analysis': condition_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=condition_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_db2_cursors_enhanced(self, content: str, procedure_name: str,
                                        db2_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse DB2 cursor definitions with SQL analysis"""
        chunks = []
        
        # Find cursor declarations
        cursor_matches = list(self.db2_patterns['db2_declare_cursor'].finditer(content))
        
        for match in cursor_matches:
            cursor_name = match.group(1)
            cursor_options = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
            cursor_sql = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
            
            cursor_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'db2_cursor_declaration',
                """
                Analyze this DB2 cursor declaration:
                
                {content}
                
                Identify:
                1. Cursor purpose and data access pattern
                2. Performance characteristics
                3. Scrollability and updatability
                4. Business data being processed
                
                Return as JSON:
                {{
                    "cursor_purpose": "result_set_processing",
                    "access_pattern": "sequential_forward",
                    "performance": "optimized_for_throughput",
                    "scrollable": false,
                    "updatable": true,
                    "business_data": "customer_transactions"
                }}
                """
            )
            
            business_context = {
                'cursor_name': cursor_name,
                'cursor_purpose': cursor_analysis.get('analysis', {}).get('cursor_purpose', ''),
                'access_pattern': cursor_analysis.get('analysis', {}).get('access_pattern', ''),
                'business_data': cursor_analysis.get('analysis', {}).get('business_data', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_CURSOR_{cursor_name}",
                chunk_type="db2_cursor",
                content=match.group(0),
                metadata={
                    'cursor_name': cursor_name,
                    'cursor_options': cursor_options,
                    'cursor_sql': cursor_sql,
                    'llm_analysis': cursor_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=cursor_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_db2_exception_handlers_enhanced(self, content: str, procedure_name: str,
                                                   db2_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse DB2 exception handlers with comprehensive error analysis"""
        chunks = []
        
        # Find exception handlers
        handler_matches = list(self.db2_patterns['db2_declare_handler'].finditer(content))
        
        for match in handler_matches:
            handler_type = match.group(1)  # CONTINUE, EXIT, UNDO
            handler_condition = match.group(2)
            handler_action = match.group(3)
            
            handler_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'db2_exception_handler',
                """
                Analyze this DB2 exception handler:
                
                {content}
                
                Identify:
                1. Error handling strategy and approach
                2. Recovery mechanisms available
                3. Business continuity impact
                4. Logging and monitoring needs
                
                Return as JSON:
                {{
                    "error_strategy": "graceful_recovery",
                    "recovery_mechanisms": ["rollback", "retry", "alternative_path"],
                    "business_continuity": "high",
                    "monitoring_needs": "comprehensive",
                    "handler_effectiveness": "robust"
                }}
                """
            )
            
            business_context = {
                'handler_type': handler_type,
                'handler_condition': handler_condition,
                'error_strategy': handler_analysis.get('analysis', {}).get('error_strategy', ''),
                'business_continuity': handler_analysis.get('analysis', {}).get('business_continuity', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_HANDLER_{handler_type}_{hash(handler_condition)%10000}",
                chunk_type="db2_exception_handler",
                content=match.group(0),
                metadata={
                    'handler_type': handler_type,
                    'handler_condition': handler_condition,
                    'handler_action': handler_action,
                    'llm_analysis': handler_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=handler_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_db2_sql_statements_enhanced(self, content: str, procedure_name: str,
                                               db2_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse SQL statements with performance analysis"""
        chunks = []
        
        # SQL statement patterns
        sql_patterns = {
            'select_into': self.db2_patterns['db2_select_into'],
            'insert': self.db2_patterns['db2_insert_statement'],
            'update': self.db2_patterns['db2_update_statement'],
            'delete': self.db2_patterns['db2_delete_statement'],
            'merge': self.db2_patterns['db2_merge_statement']
        }
        
        for sql_type, pattern in sql_patterns.items():
            matches = list(pattern.finditer(content))
            
            for match in matches:
                sql_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'db2_sql_{sql_type}',
                    """
                    Analyze this DB2 SQL statement:
                    
                    {content}
                    
                    Identify:
                    1. SQL complexity and performance characteristics
                    2. Tables and data accessed
                    3. Business operation performed
                    4. Optimization opportunities
                    
                    Return as JSON:
                    {{
                        "sql_complexity": "medium",
                        "performance_characteristics": "index_scan",
                        "tables_accessed": ["CUSTOMER", "ORDERS"],
                        "business_operation": "customer_order_lookup",
                        "optimization_opportunities": ["add_index", "rewrite_join"],
                        "estimated_cost": "low"
                    }}
                    """
                )
                
                business_context = {
                    'sql_type': sql_type,
                    'business_operation': sql_analysis.get('analysis', {}).get('business_operation', ''),
                    'performance_characteristics': sql_analysis.get('analysis', {}).get('performance_characteristics', ''),
                    'complexity': sql_analysis.get('analysis', {}).get('sql_complexity', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=procedure_name,
                    chunk_id=f"{procedure_name}_SQL_{sql_type.upper()}_{hash(match.group(0))%10000}",
                    chunk_type=f"db2_sql_{sql_type}",
                    content=match.group(0),
                    metadata={
                        'sql_type': sql_type,
                        'tables_accessed': sql_analysis.get('analysis', {}).get('tables_accessed', []),
                        'llm_analysis': sql_analysis.get('analysis', {}),
                        'complexity_score': self._calculate_sql_complexity(match.group(0))
                    },
                    business_context=business_context,
                    confidence_score=sql_analysis.get('confidence_score', 0.8),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_db2_control_flow_enhanced(self, content: str, procedure_name: str,
                                             db2_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse DB2 control flow structures"""
        chunks = []
        
        # Control flow patterns
        control_patterns = {
            'if_statement': self.db2_patterns['db2_if_statement'],
            'case_statement': self.db2_patterns['db2_case_statement'],
            'while_loop': self.db2_patterns['db2_while_loop'],
            'for_loop': self.db2_patterns['db2_for_loop'],
            'repeat_loop': self.db2_patterns['db2_repeat_loop']
        }
        
        for control_type, pattern in control_patterns.items():
            matches = list(pattern.finditer(content))
            
            for match in matches:
                control_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'db2_control_{control_type}',
                    """
                    Analyze this DB2 control flow structure:
                    
                    {content}
                    
                    Identify:
                    1. Control logic purpose and complexity
                    2. Loop characteristics and termination
                    3. Business logic implementation
                    4. Performance implications
                    
                    Return as JSON:
                    {{
                        "control_purpose": "data_processing_loop",
                        "logic_complexity": "medium",
                        "termination_strategy": "condition_based",
                        "business_logic": "batch_record_processing",
                        "performance_impact": "moderate",
                        "optimization_potential": "high"
                    }}
                    """
                )
                
                business_context = {
                    'control_type': control_type,
                    'control_purpose': control_analysis.get('analysis', {}).get('control_purpose', ''),
                    'business_logic': control_analysis.get('analysis', {}).get('business_logic', ''),
                    'complexity': control_analysis.get('analysis', {}).get('logic_complexity', 'unknown')
                }
                
                chunk = CodeChunk(
                    program_name=procedure_name,
                    chunk_id=f"{procedure_name}_CONTROL_{control_type.upper()}_{hash(match.group(0))%10000}",
                    chunk_type=f"db2_control_{control_type}",
                    content=match.group(0),
                    metadata={
                        'control_type': control_type,
                        'llm_analysis': control_analysis.get('analysis', {})
                    },
                    business_context=business_context,
                    confidence_score=control_analysis.get('confidence_score', 0.8),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_db2_dynamic_sql(self, content: str, procedure_name: str,
                                   db2_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse dynamic SQL patterns"""
        chunks = []
        
        # Dynamic SQL patterns
        dynamic_patterns = {
            'prepare': self.db2_patterns['db2_prepare'],
            'execute': self.db2_patterns['db2_execute'],
            'execute_immediate': self.db2_patterns['db2_execute_immediate']
        }
        
        for dynamic_type, pattern in dynamic_patterns.items():
            matches = list(pattern.finditer(content))
            
            for match in matches:
                dynamic_analysis = await self._analyze_with_llm_cached(
                    match.group(0), f'db2_dynamic_{dynamic_type}',
                    """
                    Analyze this DB2 dynamic SQL:
                    
                    {content}
                    
                    Identify:
                    1. Dynamic SQL purpose and flexibility
                    2. Security implications and SQL injection risks
                    3. Performance characteristics
                    4. Business use case justification
                    
                    Return as JSON:
                    {{
                        "dynamic_purpose": "flexible_query_construction",
                        "flexibility_level": "high",
                        "security_risk": "medium",
                        "performance_impact": "variable",
                        "business_justification": "user_defined_queries",
                        "mitigation_strategies": ["parameter_binding", "validation"]
                    }}
                    """
                )
                
                business_context = {
                    'dynamic_type': dynamic_type,
                    'dynamic_purpose': dynamic_analysis.get('analysis', {}).get('dynamic_purpose', ''),
                    'security_risk': dynamic_analysis.get('analysis', {}).get('security_risk', 'unknown'),
                    'business_justification': dynamic_analysis.get('analysis', {}).get('business_justification', '')
                }
                
                chunk = CodeChunk(
                    program_name=procedure_name,
                    chunk_id=f"{procedure_name}_DYNAMIC_{dynamic_type.upper()}_{hash(match.group(0))%10000}",
                    chunk_type=f"db2_dynamic_{dynamic_type}",
                    content=match.group(0),
                    metadata={
                        'dynamic_type': dynamic_type,
                        'llm_analysis': dynamic_analysis.get('analysis', {}),
                        'security_considerations': dynamic_analysis.get('analysis', {}).get('mitigation_strategies', [])
                    },
                    business_context=business_context,
                    confidence_score=dynamic_analysis.get('confidence_score', 0.8),
                    line_start=content[:match.start()].count('\n'),
                    line_end=content[:match.end()].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    def _calculate_sql_complexity(self, sql_statement: str) -> int:
        """Calculate SQL complexity score"""
        complexity = 0
        sql_upper = sql_statement.upper()
        
        # Basic operations
        if 'SELECT' in sql_upper:
            complexity += 1
        if 'INSERT' in sql_upper:
            complexity += 1
        if 'UPDATE' in sql_upper:
            complexity += 2
        if 'DELETE' in sql_upper:
            complexity += 2
        
        # Joins
        complexity += sql_upper.count('JOIN') * 2
        complexity += sql_upper.count('INNER JOIN') * 1
        complexity += sql_upper.count('LEFT JOIN') * 2
        complexity += sql_upper.count('RIGHT JOIN') * 2
        complexity += sql_upper.count('FULL JOIN') * 3
        
        # Subqueries
        complexity += sql_upper.count('(SELECT') * 3
        
        # Aggregations
        complexity += sql_upper.count('GROUP BY') * 2
        complexity += sql_upper.count('HAVING') * 2
        complexity += sql_upper.count('ORDER BY') * 1
        
        # Window functions
        complexity += sql_upper.count('OVER(') * 3
        
        # CTEs
        complexity += sql_upper.count('WITH') * 2
        
        return min(complexity, 20)  # Cap at 20
    
    # ==================== MISSING HELPER METHODS ====================

    async def _analyze_copybook_layout(self, content: str, copybook_name: str) -> Dict[str, Any]:
        """Analyze copybook layout type and complexity"""
        
        # Count different layout indicators
        layout_indicators = len(self.copybook_patterns['layout_indicator'].findall(content))
        record_types = len(self.copybook_patterns['record_type_field'].findall(content))
        redefines_count = len(self.copybook_patterns['redefines_complex'].findall(content))
        occurs_count = len(self.copybook_patterns['occurs_complex'].findall(content))
        conditional_fields = len(self.copybook_patterns['conditional_field'].findall(content))
        
        # Determine layout type
        if layout_indicators > 1 or record_types > 1:
            layout_type = CopybookLayoutType.MULTI_RECORD
        elif conditional_fields > 0:
            layout_type = CopybookLayoutType.CONDITIONAL_LAYOUT
        elif redefines_count > 2:
            layout_type = CopybookLayoutType.REDEFINES_LAYOUT
        elif occurs_count > 1:
            layout_type = CopybookLayoutType.OCCURS_LAYOUT
        else:
            layout_type = CopybookLayoutType.SINGLE_RECORD
        
        # Calculate complexity score
        complexity_score = (layout_indicators * 3 + record_types * 2 + 
                          redefines_count * 2 + occurs_count * 1 + conditional_fields * 2)
        
        return {
            'layout_type': layout_type,
            'complexity_score': min(complexity_score, 20),
            'layout_indicators': layout_indicators,
            'record_types': record_types,
            'redefines_count': redefines_count,
            'occurs_count': occurs_count,
            'conditional_fields': conditional_fields
        }

    async def _validate_with_size_limit(self, content: str, validator_method, context: str) -> List:
        """Validate content with size limits for business rules"""
        
        # Truncate large content for validation
        safe_content = self._safe_truncate_content(content, max_chars=2000)
        
        try:
            return await validator_method(safe_content)
        except Exception as e:
            self.logger.warning(f"Business validation failed for {context}: {e}")
            return [] 
        
    async def _parse_multi_record_layouts(self, content: str, copybook_name: str, 
                                        structure_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse copybooks with multiple record layouts"""
        chunks = []
        
        # Find record type indicators
        layout_matches = list(self.copybook_patterns['layout_indicator'].finditer(content))
        record_type_matches = list(self.copybook_patterns['record_type_field'].finditer(content))
        
        # Combine and sort by position
        all_indicators = []
        for match in layout_matches:
            all_indicators.append({
                'type': 'layout_indicator',
                'name': match.group(1),
                'position': match.start(),
                'match': match
            })
        
        for match in record_type_matches:
            all_indicators.append({
                'type': 'record_type',
                'name': match.group(1),
                'position': match.start(),
                'match': match
            })
        
        all_indicators.sort(key=lambda x: x['position'])
        
        # Parse each record layout
        for i, indicator in enumerate(all_indicators):
            start_pos = indicator['position']
            
            # Find end position
            if i + 1 < len(all_indicators):
                end_pos = all_indicators[i + 1]['position']
            else:
                end_pos = len(content)
            
            # Extract layout content
            layout_content = content[start_pos:end_pos].strip()
            
            # Analyze with LLM for business context
            layout_analysis = await self._analyze_with_llm_cached(
                layout_content, 'copybook_layout',
                """
                Analyze this copybook record layout:
                
                {content}
                
                Identify:
                1. Business purpose and domain
                2. Key data elements and their roles
                3. Data validation requirements
                4. Usage patterns and frequency
                
                Return as JSON:
                {{
                    "business_purpose": "customer data record",
                    "domain": "customer_management",
                    "key_elements": [
                        {{"field": "field1", "role": "identifier", "validation": "required"}}
                    ],
                    "usage_patterns": ["batch_processing", "online_inquiry"],
                    "data_sensitivity": "medium"
                }}
                """
            )
            
            business_context = {
                'layout_type': 'multi_record',
                'record_indicator': indicator['name'],
                'business_analysis': layout_analysis.get('analysis', {}),
                'confidence_score': layout_analysis.get('confidence_score', 0.7)
            }
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_LAYOUT_{indicator['name']}",
                chunk_type="copybook_record_layout",
                content=layout_content,
                metadata={
                    'layout_indicator': indicator['name'],
                    'layout_position': i + 1,
                    'total_layouts': len(all_indicators),
                    'llm_analysis': layout_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=layout_analysis.get('confidence_score', 0.7),
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_conditional_layouts(self, content: str, copybook_name: str,
                                       structure_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse copybooks with conditional field layouts"""
        chunks = []
        
        # Find conditional fields (DEPENDING ON clauses)
        conditional_matches = list(self.copybook_patterns['conditional_field'].finditer(content))
        
        for match in conditional_matches:
            field_name = match.group(1)
            depending_field = match.group(2)
            
            # Get context around the conditional field
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(content), match.end() + 200)
            context_content = content[start_pos:end_pos]
            
            conditional_analysis = await self._analyze_with_llm_cached(
                context_content, 'conditional_field',
                """
                Analyze this conditional field definition:
                
                {content}
                
                Focus on field: {field_name} depending on {depending_field}
                
                Identify:
                1. Business logic for the condition
                2. Possible values and their meanings
                3. Data integrity implications
                4. Processing complexity
                
                Return as JSON:
                {{
                    "business_logic": "variable length based on transaction type",
                    "condition_values": [
                        {{"value": "01", "meaning": "customer record", "field_count": 15}}
                    ],
                    "integrity_rules": ["depending_field must be set before access"],
                    "complexity_impact": "medium"
                }}
                """,
                field_name=field_name,
                depending_field=depending_field
            )
            
            business_context = {
                'field_type': 'conditional',
                'depending_on': depending_field,
                'business_logic': conditional_analysis.get('analysis', {}).get('business_logic', ''),
                'complexity_impact': conditional_analysis.get('analysis', {}).get('complexity_impact', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_CONDITIONAL_{field_name}",
                chunk_type="copybook_conditional_field",
                content=match.group(0),
                metadata={
                    'field_name': field_name,
                    'depending_field': depending_field,
                    'conditional_analysis': conditional_analysis.get('analysis', {}),
                    'confidence_score': conditional_analysis.get('confidence_score', 0.7)
                },
                business_context=business_context,
                confidence_score=conditional_analysis.get('confidence_score', 0.7),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_redefines_structures(self, content: str, copybook_name: str,
                                        structure_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse copybooks with complex REDEFINES structures"""
        chunks = []
        
        # Find all REDEFINES patterns
        redefines_matches = list(self.copybook_patterns['redefines_complex'].finditer(content))
        
        # Group related redefines
        redefines_groups = self._group_redefines_structures(redefines_matches, content)
        
        for group in redefines_groups:
            # Extract the complete redefines structure
            start_pos = group['start_pos']
            end_pos = group['end_pos']
            structure_content = content[start_pos:end_pos]
            
            # Analyze redefines business purpose with LLM
            redefines_analysis = await self._analyze_with_llm_cached(
                structure_content, 'redefines_structure',
                """
                Analyze this REDEFINES structure:
                
                {content}
                
                Identify:
                1. Business purpose of the overlay
                2. Different data interpretations
                3. Usage scenarios for each view
                4. Data integrity considerations
                
                Return as JSON:
                {{
                    "business_purpose": "dual interpretation of amount field",
                    "data_views": [
                        {{"view": "numeric", "purpose": "calculations", "format": "packed decimal"}},
                        {{"view": "display", "purpose": "reporting", "format": "formatted text"}}
                    ],
                    "usage_scenarios": ["batch calculations", "report generation"],
                    "integrity_considerations": ["both views must be synchronized"]
                }}
                """
            )
            
            business_context = {
                'structure_type': 'redefines',
                'original_field': group['original_field'],
                'redefining_fields': group['redefining_fields'],
                'business_purpose': redefines_analysis.get('analysis', {}).get('business_purpose', ''),
                'data_views': redefines_analysis.get('analysis', {}).get('data_views', [])
            }
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_REDEFINES_{group['original_field']}",
                chunk_type="copybook_redefines_structure",
                content=structure_content,
                metadata={
                    'original_field': group['original_field'],
                    'redefining_fields': group['redefining_fields'],
                    'redefines_analysis': redefines_analysis.get('analysis', {}),
                    'structure_complexity': len(group['redefining_fields'])
                },
                business_context=business_context,
                confidence_score=redefines_analysis.get('confidence_score', 0.7),
                line_start=content[:start_pos].count('\n'),
                line_end=content[:end_pos].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_occurs_structures(self, content: str, copybook_name: str,
                                     structure_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse copybooks with OCCURS structures (arrays and tables)"""
        chunks = []
        
        # Find all OCCURS patterns
        occurs_matches = list(self.copybook_patterns['occurs_complex'].finditer(content))
        
        for match in occurs_matches:
            min_occurs = int(match.group(1)) if match.group(1) else 0
            max_occurs = int(match.group(2)) if match.group(2) else min_occurs
            depending_field = match.group(3) if match.group(3) else None
            indexed_fields = match.group(4) if match.group(4) else None
            key_fields = match.group(5) if match.group(5) else None
            
            # Extract the complete OCCURS structure context
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(content), match.end() + 300)
            occurs_content = content[start_pos:end_pos]
            
            # Analyze OCCURS business purpose with LLM
            occurs_analysis = await self._analyze_with_llm_cached(
                occurs_content, 'occurs_structure',
                """
                Analyze this OCCURS structure:
                
                {content}
                
                Min occurs: {min_occurs}, Max occurs: {max_occurs}
                Depending on: {depending_field}
                Indexed by: {indexed_fields}
                Key fields: {key_fields}
                
                Identify:
                1. Business purpose of the array/table
                2. Data organization and access patterns
                3. Performance implications
                4. Business rules for sizing
                
                Return as JSON:
                {{
                    "business_purpose": "customer address history table",
                    "data_organization": "chronological_array",
                    "access_patterns": ["sequential_scan", "indexed_lookup"],
                    "performance_implications": "memory_intensive_for_large_arrays",
                    "sizing_rules": "max 12 addresses per customer"
                }}
                """,
                min_occurs=min_occurs,
                max_occurs=max_occurs,
                depending_field=depending_field or "N/A",
                indexed_fields=indexed_fields or "N/A",
                key_fields=key_fields or "N/A"
            )
            
            business_context = {
                'structure_type': 'occurs',
                'min_occurs': min_occurs,
                'max_occurs': max_occurs,
                'is_variable': max_occurs != min_occurs or depending_field is not None,
                'depending_field': depending_field,
                'indexed_fields': indexed_fields.split(',') if indexed_fields else [],
                'key_fields': key_fields.split(',') if key_fields else [],
                'business_purpose': occurs_analysis.get('analysis', {}).get('business_purpose', ''),
                'access_patterns': occurs_analysis.get('analysis', {}).get('access_patterns', [])
            }
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_OCCURS_{hash(match.group(0))%10000}",
                chunk_type="copybook_occurs_structure",
                content=match.group(0),
                metadata={
                    'occurs_details': {
                        'min_occurs': min_occurs,
                        'max_occurs': max_occurs,
                        'depending_field': depending_field,
                        'indexed_fields': indexed_fields,
                        'key_fields': key_fields
                    },
                    'occurs_analysis': occurs_analysis.get('analysis', {}),
                    'performance_impact': occurs_analysis.get('analysis', {}).get('performance_implications', 'unknown')
                },
                business_context=business_context,
                confidence_score=occurs_analysis.get('confidence_score', 0.7),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _group_redefines_structures(self, redefines_matches: List, content: str) -> List[Dict[str, Any]]:
        """Group related REDEFINES structures together"""
        groups = []
        
        for match in redefines_matches:
            level = int(match.group(1))
            redefining_field = match.group(2)
            original_field = match.group(3)
            
            # Find if this belongs to an existing group
            group_found = False
            for group in groups:
                if original_field == group['original_field']:
                    group['redefining_fields'].append(redefining_field)
                    group['end_pos'] = match.end()
                    group_found = True
                    break
            
            if not group_found:
                groups.append({
                    'original_field': original_field,
                    'redefining_fields': [redefining_field],
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        return groups

    async def _parse_replacing_parameters(self, content: str, copybook_name: str) -> List[CodeChunk]:
        """Parse REPLACING parameters for copybook customization"""
        chunks = []
        
        # Find all types of replacing patterns
        replacing_patterns = {
            'variable': self.copybook_patterns['replacing_variable'],
            'literal': self.copybook_patterns['replacing_literal'],
            'numeric': self.copybook_patterns['replacing_numeric'],
            'pic': self.copybook_patterns['replacing_pic'],
            'usage': self.copybook_patterns['replacing_usage']
        }
        
        all_replacements = []
        for pattern_type, pattern in replacing_patterns.items():
            matches = list(pattern.finditer(content))
            for match in matches:
                all_replacements.append({
                    'type': pattern_type,
                    'value': match.group(1),
                    'position': match.start(),
                    'match': match
                })
        
        if all_replacements:
            # Group replacements by proximity
            replacement_groups = self._group_replacing_parameters(all_replacements)
            
            for group in replacement_groups:
                replacements_content = '\n'.join([r['match'].group(0) for r in group['replacements']])
                
                business_context = {
                    'customization_type': 'replacing_parameters',
                    'parameter_types': list(set([r['type'] for r in group['replacements']])),
                    'customization_scope': 'copybook_instantiation',
                    'flexibility_level': 'high' if len(group['replacements']) > 3 else 'medium'
                }
                
                chunk = CodeChunk(
                    program_name=copybook_name,
                    chunk_id=f"{copybook_name}_REPLACING_{group['group_id']}",
                    chunk_type="copybook_replacing_parameters",
                    content=replacements_content,
                    metadata={
                        'replacement_count': len(group['replacements']),
                        'parameter_types': [r['type'] for r in group['replacements']],
                        'parameter_values': [r['value'] for r in group['replacements']]
                    },
                    business_context=business_context,
                    confidence_score=0.9,  # High confidence for pattern matching
                    line_start=content[:group['start_pos']].count('\n'),
                    line_end=content[:group['end_pos']].count('\n')
                )
                chunks.append(chunk)
        
        return chunks

    async def _parse_multi_filler_patterns(self, content: str, copybook_name: str) -> List[CodeChunk]:
        """Parse multi-filler patterns for data alignment and padding analysis"""
        chunks = []
        
        # Find all filler patterns
        filler_matches = list(self.copybook_patterns['multi_filler'].finditer(content))
        alignment_matches = list(self.copybook_patterns['filler_alignment'].finditer(content))
        sync_matches = list(self.copybook_patterns['sync_alignment'].finditer(content))
        
        # Analyze filler patterns for alignment strategy
        if filler_matches or alignment_matches or sync_matches:
            filler_analysis = await self._analyze_filler_patterns(content, filler_matches, alignment_matches, sync_matches)
            
            business_context = {
                'alignment_strategy': filler_analysis['strategy'],
                'padding_purpose': filler_analysis['purpose'],
                'memory_impact': filler_analysis['memory_impact'],
                'performance_consideration': filler_analysis['performance']
            }
            
            # Create a summary chunk for the overall filler strategy
            filler_content = '\n'.join([match.group(0) for match in filler_matches[:5]])  # Sample of fillers
            
            chunk = CodeChunk(
                program_name=copybook_name,
                chunk_id=f"{copybook_name}_FILLER_STRATEGY",
                chunk_type="copybook_filler_analysis",
                content=filler_content,
                metadata={
                    'total_fillers': len(filler_matches),
                    'alignment_fillers': len(alignment_matches),
                    'sync_markers': len(sync_matches),
                    'filler_analysis': filler_analysis
                },
                business_context=business_context,
                confidence_score=0.8,
                line_start=0,
                line_end=0
            )
            chunks.append(chunk)
        
        return chunks

    def _group_replacing_parameters(self, replacements: List[Dict]) -> List[Dict[str, Any]]:
        """Group REPLACING parameters by proximity"""
        if not replacements:
            return []
        
        replacements.sort(key=lambda x: x['position'])
        groups = []
        current_group = {'replacements': [replacements[0]], 'group_id': 1}
        current_group['start_pos'] = replacements[0]['position']
        current_group['end_pos'] = replacements[0]['position']
        
        for i in range(1, len(replacements)):
            # If the next replacement is within 200 characters, add to current group
            if replacements[i]['position'] - current_group['end_pos'] < 200:
                current_group['replacements'].append(replacements[i])
                current_group['end_pos'] = replacements[i]['position']
            else:
                # Start new group
                groups.append(current_group)
                current_group = {
                    'replacements': [replacements[i]], 
                    'group_id': len(groups) + 1,
                    'start_pos': replacements[i]['position'],
                    'end_pos': replacements[i]['position']
                }
        
        groups.append(current_group)
        return groups

    async def _analyze_filler_patterns(self, content: str, filler_matches: List, 
                                     alignment_matches: List, sync_matches: List) -> Dict[str, Any]:
        """Analyze filler patterns to determine alignment strategy"""
        
        # Calculate filler sizes
        filler_sizes = []
        for match in filler_matches:
            size = int(match.group(1))
            filler_sizes.append(size)
        
        # Determine alignment strategy
        if sync_matches:
            strategy = "hardware_alignment"
            purpose = "optimize_memory_access"
            performance = "high_performance"
        elif any(size % 4 == 0 for size in filler_sizes):
            strategy = "word_boundary_alignment"
            purpose = "memory_efficiency"
            performance = "good_performance"
        elif filler_sizes and max(filler_sizes) > 100:
            strategy = "padding_for_compatibility"
            purpose = "legacy_system_compatibility"
            performance = "memory_overhead"
        else:
            strategy = "minimal_padding"
            purpose = "structure_completion"
            performance = "standard"
        
        # Calculate memory impact
        total_filler_bytes = sum(filler_sizes)
        memory_impact = "high" if total_filler_bytes > 1000 else "medium" if total_filler_bytes > 100 else "low"
        
        return {
            'strategy': strategy,
            'purpose': purpose,
            'performance': performance,
            'memory_impact': memory_impact,
            'total_filler_bytes': total_filler_bytes,
            'filler_count': len(filler_matches),
            'alignment_count': len(alignment_matches),
            'sync_count': len(sync_matches)
        }

    async def _llm_analyze_mq_patterns(self, content: str) -> Dict[str, Any]:
        """Use LLM to analyze MQ patterns - FIXED with size limits"""
        
        # SAFETY: Truncate content to prevent oversized prompts
        safe_content = self._safe_truncate_content(content, max_chars=2500)
        
        return await self._analyze_with_llm_cached(
            safe_content, 'mq_analysis',
            """
            Analyze this IBM MQ program sample for message queuing patterns:
            
            {content}
            
            Identify:
            1. Message flow patterns (point-to-point, publish-subscribe, request-reply)
            2. Queue management strategy (persistent vs transient connections)
            3. Transaction boundaries and commit strategies
            4. Error handling and recovery patterns
            5. Performance characteristics and bottlenecks
            
            Return as JSON:
            {{
                "message_patterns": [
                    {{"pattern": "request_reply", "queues": ["REQUEST.Q", "REPLY.Q"], "volume": "high"}}
                ],
                "connection_strategy": "persistent",
                "transaction_scope": "message_level",
                "error_handling": [
                    {{"error_type": "connection_failure", "strategy": "retry_with_backoff"}}
                ],
                "performance_characteristics": {{"throughput": "medium", "latency": "low"}},
                "reliability_level": "high"
            }}
            """
        )

    async def _parse_mq_api_sequences(self, content: str, program_name: str, 
                                    mq_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse MQ API call sequences to understand message flow"""
        chunks = []
        
        # Find all MQ API calls in sequence
        api_calls = []
        for api_name, pattern in self.mq_patterns.items():
            if api_name.startswith('mq_mq'):  # Only actual API calls
                matches = list(pattern.finditer(content))
                for match in matches:
                    api_calls.append({
                        'api': api_name,
                        'position': match.start(),
                        'match': match,
                        'parameters': match.group(1) if match.groups() else ""
                    })
        
        # Sort by position to understand call sequence
        api_calls.sort(key=lambda x: x['position'])
        
        # Group API calls into logical sequences
        sequences = self._group_mq_api_sequences(api_calls)
        
        for sequence in sequences:
            sequence_content = '\n'.join([call['match'].group(0) for call in sequence['calls']])
            
            # Analyze sequence with LLM
            sequence_analysis = await self._analyze_with_llm_cached(
                sequence_content, 'mq_sequence',
                """
                Analyze this MQ API call sequence:
                
                {content}
                
                Identify:
                1. Message flow purpose and business function
                2. Transaction boundaries
                3. Error handling completeness
                4. Performance implications
                
                Return as JSON:
                {{
                    "business_purpose": "customer order processing",
                    "flow_type": "request_response",
                    "transaction_safe": true,
                    "error_handling_complete": false,
                    "performance_risk": "medium"
                }}
                """
            )
            
            business_context = {
                'sequence_type': sequence['type'],
                'api_count': len(sequence['calls']),
                'business_purpose': sequence_analysis.get('analysis', {}).get('business_purpose', ''),
                'transaction_safe': sequence_analysis.get('analysis', {}).get('transaction_safe', False),
                'performance_risk': sequence_analysis.get('analysis', {}).get('performance_risk', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name[:20]}_MQ_SEQUENCE_{sequence['sequence_id']}",
                chunk_type="mq_api_sequence",
                content=sequence_content,
                metadata={
                    'api_calls': [call['api'] for call in sequence['calls']],
                    'sequence_analysis': sequence_analysis.get('analysis', {}),
                    'call_count': len(sequence['calls'])
                },
                business_context=business_context,
                confidence_score=sequence_analysis.get('confidence_score', 0.7),
                line_start=content[:sequence['start_pos']].count('\n'),
                line_end=content[:sequence['end_pos']].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _group_mq_api_sequences(self, api_calls: List[Dict]) -> List[Dict[str, Any]]:
        """Group MQ API calls into logical sequences"""
        sequences = []
        current_sequence = None
        sequence_id = 1
        
        for call in api_calls:
            api_name = call['api']
            
            # Connection sequence
            if api_name in ['mq_mqconn', 'mq_mqconnx']:
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = {
                    'type': 'connection_sequence',
                    'sequence_id': sequence_id,
                    'calls': [call],
                    'start_pos': call['position'],
                    'end_pos': call['position']
                }
                sequence_id += 1
            
            # Message operations
            elif api_name in ['mq_mqput', 'mq_mqput1', 'mq_mqget']:
                if current_sequence and current_sequence['type'] in ['connection_sequence', 'message_sequence']:
                    current_sequence['calls'].append(call)
                    current_sequence['end_pos'] = call['position']
                    current_sequence['type'] = 'message_sequence'
                else:
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = {
                        'type': 'message_sequence',
                        'sequence_id': sequence_id,
                        'calls': [call],
                        'start_pos': call['position'],
                        'end_pos': call['position']
                    }
                    sequence_id += 1
            
            # Transaction operations
            elif api_name in ['mq_mqbegin', 'mq_mqcmit', 'mq_mqback']:
                if current_sequence and current_sequence['type'] in ['connection_sequence', 'message_sequence', 'transaction_sequence']:
                    current_sequence['calls'].append(call)
                    current_sequence['end_pos'] = call['position']
                    current_sequence['type'] = 'transaction_sequence'
                else:
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = {
                        'type': 'transaction_sequence',
                        'sequence_id': sequence_id,
                        'calls': [call],
                        'start_pos': call['position'],
                        'end_pos': call['position']
                    }
                    sequence_id += 1
            
            # Other operations - add to current sequence or create new one
            else:
                if current_sequence:
                    current_sequence['calls'].append(call)
                    current_sequence['end_pos'] = call['position']
                else:
                    current_sequence = {
                        'type': 'general_sequence',
                        'sequence_id': sequence_id,
                        'calls': [call],
                        'start_pos': call['position'],
                        'end_pos': call['position']
                    }
                    sequence_id += 1
        
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences
    
    """
    Missing Enhanced Code Parser Agent Functions
    Complete implementation of missing methods for COBOL stored procedures and related functionality
    """

    # ==================== MISSING COBOL STORED PROCEDURE PARSING ====================

    async def parse_cobol_stored_procedure_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Enhanced COBOL stored procedure parsing with comprehensive error handling"""
        chunks = []
        procedure_name = self._extract_program_name(content, Path(filename))
        
        try:
            # Analyze COBOL SP patterns with LLM
            sp_analysis = await self._llm_analyze_cobol_sp_patterns(content)
            
            # First parse as regular COBOL with SP enhancements
            base_cobol_chunks = await self._parse_cobol_with_enhanced_analysis(content, filename)
            chunks.extend(base_cobol_chunks)
            
            # Parse SQL communication areas (SQLCA, SQLDA) with safety
            try:
                comm_chunks = await self._parse_sql_communication_areas_enhanced_safe(content, procedure_name, sp_analysis)
                chunks.extend(comm_chunks)
            except Exception as e:
                self.logger.warning(f"Failed to parse SQL communication areas: {e}")
            
            # Parse embedded SQL with enhanced host variable validation
            try:
                embedded_sql_chunks = await self._parse_embedded_sql_enhanced(content, procedure_name, sp_analysis)
                chunks.extend(embedded_sql_chunks)
            except Exception as e:
                self.logger.warning(f"Failed to parse embedded SQL: {e}")
            
            # Parse other components with individual error handling
            for parse_method, description in [
                (self._parse_result_set_handling, "result set handling"),
                (self._parse_procedure_calls_enhanced, "procedure calls"),
                (self._parse_host_variables_enhanced, "host variables"),
                (self._parse_cobol_transaction_patterns, "transaction patterns")
            ]:
                try:
                    component_chunks = await parse_method(content, procedure_name, sp_analysis)
                    chunks.extend(component_chunks)
                except Exception as e:
                    self.logger.warning(f"Failed to parse {description}: {e}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to parse COBOL stored procedure {procedure_name}: {e}")
            # Fallback to basic COBOL parsing
            return await self._parse_cobol_with_enhanced_analysis(content, filename)
        
    async def _parse_host_variables_enhanced(self, content: str, procedure_name: str,
                                    sp_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse host variables with enhanced SQL integration analysis"""
        chunks = []
        
        # Find host variable declarations in WORKING-STORAGE
        ws_section_match = self.cobol_patterns['working_storage'].search(content)
        if not ws_section_match:
            return chunks
        
        ws_start = ws_section_match.end()
        # Find end of working storage (next section or procedure division)
        next_section = None
        for pattern_name, pattern in self.cobol_patterns.items():
            if 'section' in pattern_name.lower() or 'division' in pattern_name.lower():
                match = pattern.search(content, ws_start)
                if match and (not next_section or match.start() < next_section):
                    next_section = match.start()
        
        if not next_section:
            next_section = len(content)
        
        ws_content = content[ws_start:next_section]
        
        # Find all data items that could be host variables
        data_items = list(self.cobol_patterns['data_item'].finditer(ws_content))
        
        # Group related host variables
        host_var_groups = self._group_host_variables(data_items, ws_content)
        
        for group in host_var_groups:
            group_content = '\n'.join([item['match'].group(0) for item in group['variables']])
            
            # Analyze with LLM for SQL integration patterns
            host_var_analysis = await self._analyze_with_llm_cached(
                group_content, 'cobol_host_variables',
                """
                Analyze these COBOL host variables for SQL integration:
                
                {content}
                
                Identify:
                1. SQL data type compatibility
                2. Parameter passing patterns
                3. Null indicator usage
                4. Data conversion requirements
                5. Performance implications
                
                Return as JSON:
                {{
                    "sql_compatibility": "fully_compatible",
                    "parameter_patterns": ["input", "output", "inout"],
                    "null_indicators": true,
                    "conversion_needed": ["date_format", "decimal_precision"],
                    "performance_impact": "minimal",
                    "usage_context": "batch_processing"
                }}
                """
            )
            
            business_context = {
                'variable_group_type': 'sql_host_variables',
                'sql_compatibility': host_var_analysis.get('analysis', {}).get('sql_compatibility', ''),
                'parameter_patterns': host_var_analysis.get('analysis', {}).get('parameter_patterns', []),
                'usage_context': host_var_analysis.get('analysis', {}).get('usage_context', 'unknown'),
                'variable_count': len(group['variables'])
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_HOST_VARS_{group['group_id']}",
                chunk_type="cobol_host_variables",
                content=group_content,
                metadata={
                    'variable_count': len(group['variables']),
                    'variable_names': [var['field_name'] for var in group['variables']],
                    'llm_analysis': host_var_analysis.get('analysis', {}),
                    'group_purpose': group['purpose']
                },
                business_context=business_context,
                confidence_score=host_var_analysis.get('confidence_score', 0.7),
                line_start=ws_content[:group['start_pos']].count('\n'),
                line_end=ws_content[:group['end_pos']].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_cobol_transaction_patterns(self, content: str, procedure_name: str,
                                        sp_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse COBOL transaction coordination patterns"""
        chunks = []
        
        # Find SQL transaction control statements
        commit_matches = list(self.sql_patterns['sql_commit'].finditer(content))
        rollback_matches = list(self.sql_patterns['sql_rollback'].finditer(content))
        
        for match in commit_matches + rollback_matches:
            trans_type = 'commit' if 'COMMIT' in match.group(0).upper() else 'rollback'
            
            # Get surrounding context
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(content), match.end() + 200)
            context_content = content[start_pos:end_pos]
            
            trans_analysis = await self._analyze_with_llm_cached(
                context_content, f'cobol_transaction_{trans_type}',
                """
                Analyze this COBOL transaction control:
                
                {content}
                
                Identify:
                1. Transaction boundary management
                2. Error handling integration
                3. Data consistency strategy
                4. Recovery implications
                5. Business process alignment
                
                Return as JSON:
                {{
                    "boundary_management": "explicit_control",
                    "error_integration": "comprehensive",
                    "consistency_strategy": "acid_compliant",
                    "recovery_approach": "automatic_rollback",
                    "business_alignment": "batch_processing"
                }}
                """
            )
            
            business_context = {
                'transaction_type': trans_type,
                'boundary_management': trans_analysis.get('analysis', {}).get('boundary_management', ''),
                'consistency_strategy': trans_analysis.get('analysis', {}).get('consistency_strategy', ''),
                'business_alignment': trans_analysis.get('analysis', {}).get('business_alignment', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_TRANS_{trans_type.upper()}_{hash(match.group(0))%10000}",
                chunk_type=f"cobol_transaction_{trans_type}",
                content=match.group(0),
                metadata={
                    'transaction_type': trans_type,
                    'context_content': context_content,
                    'llm_analysis': trans_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=trans_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks
    
    def _group_host_variables(self, data_items: List, ws_content: str) -> List[Dict[str, Any]]:
        """Group host variables by their SQL usage patterns"""
        groups = []
        current_group = None
        group_id = 1
        
        for i, match in enumerate(data_items):
            try:
                # Safely extract match groups with fallbacks
                if not match.groups() or len(match.groups()) < 2:
                    continue
                    
                level_str = match.group(1)
                field_name = match.group(2)
                field_content = match.group(0)
                
                # Validate level is numeric
                if not level_str or not level_str.isdigit():
                    continue
                    
                level = int(level_str)
                
                # Determine if this looks like a host variable
                is_host_var = False
                host_var_indicators = [
                    ':' in field_name,  # Explicit host variable indicator
                    'PIC' in field_content.upper(),  # Has picture clause
                    any(sql_type in field_content.upper() for sql_type in ['COMP-3', 'COMP', 'DISPLAY']),
                    field_name.upper().endswith(('-IND', '-NULL', '-LEN'))  # Common host var suffixes
                ]
                
                if any(host_var_indicators) or level in [1, 5, 77]:  # Common host variable levels
                    is_host_var = True
                
                if is_host_var:
                    # Determine grouping strategy
                    if level == 1:  # Start new group for 01 level
                        if current_group and current_group['variables']:
                            groups.append(current_group)
                        
                        current_group = {
                            'group_id': group_id,
                            'variables': [],
                            'purpose': self._determine_host_var_purpose(field_name, field_content),
                            'start_pos': match.start(),
                            'end_pos': match.end()
                        }
                        group_id += 1
                    
                    if current_group:
                        current_group['variables'].append({
                            'level': level,
                            'field_name': field_name,
                            'match': match,
                            'position': match.start()
                        })
                        current_group['end_pos'] = match.end()
                        
            except (ValueError, IndexError, AttributeError) as e:
                # Log the error and continue processing
                self.logger.debug(f"Error processing data item {i}: {e}")
                continue
        
        if current_group and current_group['variables']:
            groups.append(current_group)
        
        return groups

    def _determine_host_var_purpose(self, field_name: str, field_content: str) -> str:
        """Determine the purpose of host variable group"""
        if not field_name or not field_content:
            return 'general_purpose'
            
        field_upper = field_name.upper()
        content_upper = field_content.upper()
        
        if any(indicator in field_upper for indicator in ['CUSTOMER', 'CUST']):
            return 'customer_data'
        elif any(indicator in field_upper for indicator in ['ORDER', 'ORD']):
            return 'order_data'
        elif any(indicator in field_upper for indicator in ['ACCOUNT', 'ACCT']):
            return 'account_data'
        elif any(indicator in field_upper for indicator in ['TRANS', 'TXN']):
            return 'transaction_data'
        elif any(indicator in field_upper for indicator in ['IND', 'NULL']):
            return 'indicator_variables'
        elif any(indicator in field_upper for indicator in ['WORK', 'TEMP']):
            return 'working_variables'
        elif 'PIC S9' in content_upper:
            return 'numeric_data'
        elif 'PIC X' in content_upper:
            return 'character_data'
        else:
            return 'general_purpose'

# ==================== MISSING STORAGE METHODS ====================

    async def _store_chunks_enhanced(self, chunks: List[CodeChunk], file_hash: str):
        """Store chunks with enhanced metadata and validation"""
        if not chunks:
            return
        
        try:
            async with self._engine_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for chunk in chunks:
                    cursor.execute("""
                        INSERT OR REPLACE INTO program_chunks 
                        (program_name, chunk_id, chunk_type, content, metadata, 
                        business_context, file_hash, confidence_score, llm_analysis,
                        line_start, line_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.program_name,
                        chunk.chunk_id,
                        chunk.chunk_type,
                        chunk.content,
                        json.dumps(chunk.metadata),
                        json.dumps(chunk.business_context),
                        file_hash,
                        chunk.confidence_score,
                        json.dumps(chunk.metadata.get('llm_analysis', {})),
                        chunk.line_start,
                        chunk.line_end
                    ))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Stored {len(chunks)} enhanced chunks")
                
        except Exception as e:
            self.logger.error(f"Failed to store enhanced chunks: {str(e)}")
            raise

    async def _verify_chunks_stored(self, program_name: str) -> int:
        """Verify chunks were stored correctly"""
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

    async def _generate_metadata_enhanced(self, chunks: List[CodeChunk], file_type: str, 
                                        business_violations: List[BusinessRuleViolation]) -> Dict[str, Any]:
        """Generate enhanced metadata with business insights"""
        
        # Calculate chunk statistics
        chunk_types = {}
        total_confidence = 0.0
        llm_analyzed_chunks = 0
        business_contexts = {}
        
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_confidence += chunk.confidence_score
            
            if chunk.metadata.get('llm_analysis'):
                llm_analyzed_chunks += 1
            
            # Aggregate business contexts
            for key, value in chunk.business_context.items():
                if key not in business_contexts:
                    business_contexts[key] = []
                if value and value not in business_contexts[key]:
                    business_contexts[key].append(value)
        
        # Calculate complexity metrics
        complexity_indicators = {
            'high_complexity_chunks': sum(1 for chunk in chunks if chunk.confidence_score < 0.7),
            'llm_analysis_coverage': llm_analyzed_chunks / len(chunks) if chunks else 0,
            'average_confidence': total_confidence / len(chunks) if chunks else 0,
            'business_violation_rate': len(business_violations) / len(chunks) if chunks else 0
        }
        
        # Categorize business violations
        violation_summary = {}
        for violation in business_violations:
            severity = violation.severity
            violation_summary[severity] = violation_summary.get(severity, 0) + 1
        
        return {
            'file_type': file_type,
            'total_chunks': len(chunks),
            'chunk_type_distribution': chunk_types,
            'complexity_metrics': complexity_indicators,
            'business_contexts': business_contexts,
            'business_violations': {
                'total': len(business_violations),
                'by_severity': violation_summary
            },
            'quality_metrics': {
                'average_confidence': complexity_indicators['average_confidence'],
                'llm_coverage': complexity_indicators['llm_analysis_coverage'],
                'high_quality_chunks': sum(1 for chunk in chunks if chunk.confidence_score >= 0.8)
            },
            'analysis_timestamp': dt.now().isoformat()
        }

    async def _store_copybook_analysis(self, chunks: List[CodeChunk], copybook_name: str):
        """Store copybook-specific analysis"""
        try:
            # Aggregate copybook analysis from chunks
            layout_types = []
            field_hierarchies = {}
            occurs_structures = []
            redefines_structures = []
            replacing_parameters = []
            business_domain = 'unknown'
            complexity_score = 0
            
            for chunk in chunks:
                if chunk.chunk_type.startswith('copybook_'):
                    # Extract layout information
                    if 'layout_type' in chunk.metadata:
                        layout_types.append(chunk.metadata['layout_type'])
                    
                    # Extract business domain
                    if chunk.business_context.get('business_entity'):
                        business_domain = chunk.business_context['business_entity']
                    
                    # Aggregate complexity
                    if 'complexity_score' in chunk.metadata:
                        complexity_score += chunk.metadata['complexity_score']
                    
                    # Extract structure information
                    if chunk.chunk_type == 'copybook_occurs_structure':
                        occurs_structures.append(chunk.metadata.get('occurs_details', {}))
                    elif chunk.chunk_type == 'copybook_redefines_structure':
                        redefines_structures.append({
                            'original_field': chunk.metadata.get('original_field'),
                            'redefining_fields': chunk.metadata.get('redefining_fields', [])
                        })
                    elif chunk.chunk_type == 'copybook_replacing_parameters':
                        replacing_parameters.extend(chunk.metadata.get('parameter_values', []))
            
            # Store in copybook_structures table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO copybook_structures 
                (copybook_name, layout_type, record_layouts, field_hierarchy,
                occurs_structures, redefines_structures, replacing_parameters,
                business_domain, complexity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                copybook_name,
                ','.join(set(layout_types)) if layout_types else 'single_record',
                json.dumps([]),  # Would need more detailed analysis
                json.dumps(field_hierarchies),
                json.dumps(occurs_structures),
                json.dumps(redefines_structures),
                json.dumps(replacing_parameters),
                business_domain,
                min(complexity_score, 20)  # Cap at 20
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored copybook analysis for {copybook_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store copybook analysis: {str(e)}")

    async def _store_mq_analysis(self, chunks: List[CodeChunk], content: str, program_name: str):
        """Store MQ-specific analysis"""
        try:
            # Aggregate MQ analysis from chunks
            connection_types = []
            message_paradigms = []
            queue_operations = []
            flow_patterns = []
            transaction_scopes = []
            error_strategies = []
            performance_chars = {}
            business_purpose = 'unknown'
            
            for chunk in chunks:
                if chunk.chunk_type.startswith('mq_'):
                    # Extract MQ-specific metadata
                    if chunk.business_context.get('connection_strategy'):
                        connection_types.append(chunk.business_context['connection_strategy'])
                    
                    if chunk.business_context.get('flow_type'):
                        message_paradigms.append(chunk.business_context['flow_type'])
                    
                    if chunk.chunk_type == 'mq_api_sequence':
                        api_calls = chunk.metadata.get('api_calls', [])
                        queue_operations.extend(api_calls)
                    
                    if chunk.chunk_type == 'mq_message_flow':
                        flow_patterns.append(chunk.business_context.get('flow_type', ''))
                    
                    if chunk.chunk_type == 'mq_transaction_pattern':
                        transaction_scopes.append(chunk.business_context.get('transaction_scope', ''))
                    
                    if chunk.chunk_type == 'mq_error_handling':
                        error_strategies.append(chunk.business_context.get('detection_strategy', ''))
                    
                    # Extract business purpose
                    if chunk.business_context.get('business_alignment'):
                        business_purpose = chunk.business_context['business_alignment']
            
            # Store in mq_analysis table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO mq_analysis 
                (program_name, connection_type, message_paradigm, queue_operations,
                message_flow_patterns, transaction_scope, error_handling_strategy,
                performance_characteristics, business_purpose)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_name,
                ','.join(set(connection_types)) if connection_types else 'unknown',
                ','.join(set(message_paradigms)) if message_paradigms else 'unknown',
                json.dumps(list(set(queue_operations))),
                json.dumps(list(set(flow_patterns))),
                ','.join(set(transaction_scopes)) if transaction_scopes else 'unknown',
                ','.join(set(error_strategies)) if error_strategies else 'unknown',
                json.dumps(performance_chars),
                business_purpose
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored MQ analysis for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store MQ analysis: {str(e)}")

    async def _store_db2_procedure_analysis(self, chunks: List[CodeChunk], procedure_name: str):
        """Store DB2 procedure-specific analysis"""
        try:
            # Aggregate DB2 analysis from chunks
            parameters = []
            sql_operations = []
            cursors = []
            exception_handlers = []
            transaction_control = {}
            performance_hints = []
            complexity_score = 0
            security_classification = 'standard'
            
            for chunk in chunks:
                if chunk.chunk_type.startswith('db2_'):
                    # Extract DB2-specific metadata
                    if chunk.chunk_type == 'db2_procedure_signature':
                        parameters = chunk.metadata.get('llm_analysis', {}).get('parameter_types', [])
                    
                    elif chunk.chunk_type.startswith('db2_sql_'):
                        sql_operations.append({
                            'type': chunk.business_context.get('sql_type', ''),
                            'complexity': chunk.business_context.get('complexity', ''),
                            'tables': chunk.metadata.get('tables_accessed', [])
                        })
                        complexity_score += chunk.metadata.get('complexity_score', 0)
                    
                    elif chunk.chunk_type == 'db2_cursor':
                        cursors.append({
                            'name': chunk.metadata.get('cursor_name', ''),
                            'purpose': chunk.business_context.get('cursor_purpose', ''),
                            'performance': chunk.business_context.get('access_pattern', '')
                        })
                    
                    elif chunk.chunk_type == 'db2_exception_handler':
                        exception_handlers.append({
                            'type': chunk.metadata.get('handler_type', ''),
                            'strategy': chunk.business_context.get('error_strategy', '')
                        })
                    
                    elif chunk.chunk_type.startswith('db2_control_'):
                        if 'transaction' in chunk.metadata.get('llm_analysis', {}).get('control_purpose', ''):
                            transaction_control = chunk.metadata.get('llm_analysis', {})
                    
                    # Extract performance hints
                    optimization_opportunities = chunk.metadata.get('llm_analysis', {}).get('optimization_opportunities', [])
                    performance_hints.extend(optimization_opportunities)
            
            # Store in db2_procedure_analysis table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO db2_procedure_analysis 
                (procedure_name, parameter_list, sql_operations, cursor_definitions,
                exception_handlers, transaction_control, performance_hints,
                business_logic_complexity, security_classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                procedure_name,
                json.dumps(parameters),
                json.dumps(sql_operations),
                json.dumps(cursors),
                json.dumps(exception_handlers),
                json.dumps(transaction_control),
                json.dumps(list(set(performance_hints))),
                min(complexity_score, 20),  # Cap at 20
                security_classification
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored DB2 procedure analysis for {procedure_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store DB2 procedure analysis: {str(e)}")
            

    async def _parse_generic(self, content: str, filename: str) -> List[CodeChunk]:
        """Generic parsing for unknown file types"""
        chunks = []
        program_name = self._extract_program_name(content, Path(filename))
        
        # Simple line-based chunking with basic analysis
        lines = content.split('\n')
        chunk_size = 50  # lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name[:20]}_GENERIC_{i//chunk_size + 1}",
                    chunk_type="generic_code_block",
                    content=chunk_content,
                    metadata={
                        'chunk_number': i//chunk_size + 1,
                        'line_count': len(chunk_lines),
                        'start_line': i + 1,
                        'end_line': min(i + chunk_size, len(lines))
                    },
                    business_context={'file_type': 'unknown'},
                    confidence_score=0.5,
                    line_start=i,
                    line_end=min(i + chunk_size, len(lines))
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _extract_and_store_program_relationships(self, content: str, program_name: str, file_type: str):
        """Extract and store all program relationships"""
        try:
            relationships = []
            
            if file_type in ['cobol', 'cics']:
                # Extract COBOL CALL statements
                relationships.extend(self._extract_cobol_calls(content, program_name))
                
                # Extract CICS LINK/XCTL calls
                relationships.extend(self._extract_cics_calls(content, program_name))
                
            elif file_type == 'jcl':
                # Extract JCL EXEC PGM calls
                relationships.extend(self._extract_jcl_calls(content, program_name))
            
            # Store relationships
            if relationships:
                await self._store_program_relationships(relationships)
                
        except Exception as e:
            self.logger.error(f"Failed to extract program relationships: {str(e)}")

    def _extract_cobol_calls(self, content: str, calling_program: str) -> List[Dict]:
        """Extract COBOL CALL statements"""
        relationships = []
        
        # Pattern for CALL statements
        call_pattern = r'CALL\s+[\'"]([^\'\"]+)[\'"](?:\s+USING\s+([^\.]+))?'
        call_matches = re.finditer(call_pattern, content, re.IGNORECASE)
        
        for match in call_matches:
            called_program = match.group(1).strip()
            parameters = match.group(2).strip() if match.group(2) else ""
            
            # Find the paragraph/section containing this call
            call_location = self._find_containing_paragraph(content, match.start())
            
            # Check if call is conditional
            conditional = self._is_conditional_statement(content, match.start())
            
            relationships.append({
                'calling_program': calling_program,
                'called_program': called_program,
                'call_type': 'CALL',
                'call_location': call_location,
                'parameters': parameters.split() if parameters else [],
                'call_statement': match.group(0),
                'conditional_call': conditional,
                'line_number': content[:match.start()].count('\n') + 1
            })
        
        return relationships

    def _extract_cics_calls(self, content: str, calling_program: str) -> List[Dict]:
        """Extract CICS LINK and XCTL calls"""
        relationships = []
        
        # CICS LINK pattern
        link_pattern = r'EXEC\s+CICS\s+LINK\s+[^.]*?PROGRAM\s*\(\s*[\'"]?([^\'\")\s]+)[\'"]?\s*\)[^.]*?END-EXEC'
        link_matches = re.finditer(link_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in link_matches:
            called_program = match.group(1).strip()
            call_location = self._find_containing_paragraph(content, match.start())
            
            # Extract COMMAREA info
            commarea_match = re.search(r'COMMAREA\s*\(\s*([^)]+)\s*\)', match.group(0), re.IGNORECASE)
            parameters = [commarea_match.group(1)] if commarea_match else []
            
            relationships.append({
                'calling_program': calling_program,
                'called_program': called_program,
                'call_type': 'CICS_LINK',
                'call_location': call_location,
                'parameters': parameters,
                'call_statement': match.group(0),
                'conditional_call': self._is_conditional_statement(content, match.start()),
                'line_number': content[:match.start()].count('\n') + 1
            })
        
        # CICS XCTL pattern
        xctl_pattern = r'EXEC\s+CICS\s+XCTL\s+[^.]*?PROGRAM\s*\(\s*[\'"]?([^\'\")\s]+)[\'"]?\s*\)[^.]*?END-EXEC'
        xctl_matches = re.finditer(xctl_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in xctl_matches:
            called_program = match.group(1).strip()
            call_location = self._find_containing_paragraph(content, match.start())
            
            commarea_match = re.search(r'COMMAREA\s*\(\s*([^)]+)\s*\)', match.group(0), re.IGNORECASE)
            parameters = [commarea_match.group(1)] if commarea_match else []
            
            relationships.append({
                'calling_program': calling_program,
                'called_program': called_program,
                'call_type': 'CICS_XCTL',
                'call_location': call_location,
                'parameters': parameters,
                'call_statement': match.group(0),
                'conditional_call': self._is_conditional_statement(content, match.start()),
                'line_number': content[:match.start()].count('\n') + 1
            })
        
        return relationships

    def _extract_jcl_calls(self, content: str, calling_job: str) -> List[Dict]:
        """Extract JCL EXEC PGM calls"""
        relationships = []
        
        # JCL EXEC PGM pattern
        exec_pattern = r'^//(\w+)\s+EXEC\s+(?:PGM=([^,\s]+)|PROC=([^,\s]+)|([^,\s]+))'
        exec_matches = re.finditer(exec_pattern, content, re.IGNORECASE | re.MULTILINE)
        
        for match in exec_matches:
            step_name = match.group(1)
            program_name = match.group(2) or match.group(3) or match.group(4)
            
            if program_name:
                call_type = 'JCL_EXEC_PGM' if match.group(2) else 'JCL_EXEC_PROC'
                
                relationships.append({
                    'calling_program': calling_job,
                    'called_program': program_name.strip(),
                    'call_type': call_type,
                    'call_location': step_name,
                    'parameters': [],
                    'call_statement': match.group(0),
                    'conditional_call': False,
                    'line_number': content[:match.start()].count('\n') + 1
                })
        
        return relationships

    async def _extract_and_store_copybook_relationships(self, content: str, program_name: str):
        """Extract and store copybook relationships"""
        try:
            relationships = []
            
            # Enhanced COPY statement pattern
            copy_pattern = r'COPY\s+([A-Z][A-Z0-9-]*)(?:\s+IN\s+([A-Z][A-Z0-9-]*))?(?:\s+REPLACING\s+(.*?))?(?=\s*\.|\s*$)'
            copy_matches = re.finditer(copy_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for match in copy_matches:
                copybook_name = match.group(1).strip()
                library_name = match.group(2).strip() if match.group(2) else ""
                replacing_clause = match.group(3).strip() if match.group(3) else ""
                
                # Determine copy location (which section)
                copy_location = self._determine_copy_location(content, match.start())
                usage_context = self._determine_usage_context(copy_location)
                
                relationships.append({
                    'program_name': program_name,
                    'copybook_name': copybook_name,
                    'copy_location': copy_location,
                    'replacing_clause': replacing_clause,
                    'copy_statement': match.group(0),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'usage_context': usage_context
                })
            
            if relationships:
                await self._store_copybook_relationships(relationships)
                
        except Exception as e:
            self.logger.error(f"Failed to extract copybook relationships: {str(e)}")

    async def _extract_and_store_file_relationships(self, content: str, program_name: str):
        """Extract and store file access relationships"""
        try:
            relationships = []
            
            # SELECT statements (file assignments)
            select_pattern = r'SELECT\s+([A-Z][A-Z0-9-]*)\s+ASSIGN\s+TO\s+([^\s\.]+)'
            select_matches = re.finditer(select_pattern, content, re.IGNORECASE)
            
            for match in select_matches:
                logical_file = match.group(1).strip()
                physical_file = match.group(2).strip()
                
                relationships.append({
                    'program_name': program_name,
                    'logical_file_name': logical_file,
                    'physical_file_name': physical_file,
                    'access_type': 'SELECT',
                    'access_mode': '',
                    'record_format': '',
                    'access_location': 'FILE-CONTROL',
                    'line_number': content[:match.start()].count('\n') + 1
                })
            
            # FD statements
            fd_pattern = r'FD\s+([A-Z][A-Z0-9-]*)'
            fd_matches = re.finditer(fd_pattern, content, re.IGNORECASE)
            
            for match in fd_matches:
                file_name = match.group(1).strip()
                
                relationships.append({
                    'program_name': program_name,
                    'logical_file_name': file_name,
                    'physical_file_name': '',
                    'access_type': 'FD',
                    'access_mode': '',
                    'record_format': '',
                    'access_location': 'FILE-SECTION',
                    'line_number': content[:match.start()].count('\n') + 1
                })
            
            # File operations (READ, WRITE, etc.)
            file_ops = ['READ', 'write', 'rewrite', 'delete', 'open', 'close']
            for op in file_ops:
                op_pattern = rf'\b{op}\s+([A-Z][A-Z0-9-]*)'
                op_matches = re.finditer(op_pattern, content, re.IGNORECASE)
                
                for match in op_matches:
                    file_name = match.group(1).strip()
                    access_location = self._find_containing_paragraph(content, match.start())
                    
                    relationships.append({
                        'program_name': program_name,
                        'logical_file_name': file_name,
                        'physical_file_name': '',
                        'access_type': op.upper(),
                        'access_mode': '',
                        'record_format': '',
                        'access_location': access_location,
                        'line_number': content[:match.start()].count('\n') + 1
                    })
            
            if relationships:
                await self._store_file_relationships(relationships)
                
        except Exception as e:
            self.logger.error(f"Failed to extract file relationships: {str(e)}")

    async def _extract_and_store_field_cross_reference(self, content: str, source_name: str, source_type: str):
        """Extract and store comprehensive field cross-reference"""
        try:
            field_refs = []
            
            # Data item pattern for definitions
            data_pattern = r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+(.*?)\.?\s*$'
            data_matches = re.finditer(data_pattern, content, re.MULTILINE | re.IGNORECASE)
            
            for match in data_matches:
                level = int(match.group(1))
                field_name = match.group(2).strip()
                definition = match.group(3).strip()
                
                # Skip comment lines
                if match.group(0).strip().startswith('*'):
                    continue
                
                # Determine definition location
                def_location = self._determine_field_definition_location(content, match.start())
                
                # Extract field attributes
                pic_clause = self._extract_pic_clause(definition)
                usage_clause = self._extract_usage_clause(definition)
                data_type = self._determine_data_type_enhanced(definition)
                
                # Determine parent field for nested structures
                parent_field = self._determine_parent_field(content, match.start(), level)
                
                # Extract business domain
                business_domain = self._infer_business_domain(field_name)
                
                # Build qualified name for nested fields
                qualified_name = self._build_qualified_name(field_name, parent_field)
                
                field_refs.append({
                    'field_name': field_name,
                    'qualified_name': qualified_name,
                    'source_type': source_type,
                    'source_name': source_name,
                    'definition_location': def_location,
                    'data_type': data_type,
                    'picture_clause': pic_clause,
                    'usage_clause': usage_clause,
                    'level_number': level,
                    'parent_field': parent_field,
                    'occurs_info': json.dumps(self._extract_occurs_info(definition)),
                    'business_domain': business_domain
                })
            
            if field_refs:
                await self._store_field_cross_reference(field_refs)
                
        except Exception as e:
            self.logger.error(f"Failed to extract field cross-reference: {str(e)}")

    # STORAGE METHODS

    async def _store_program_relationships(self, relationships: List[Dict]):
        """Store program relationships in database"""
        if not relationships:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rel in relationships:
                cursor.execute("""
                    INSERT OR REPLACE INTO program_relationships 
                    (calling_program, called_program, call_type, call_location, 
                    parameters, call_statement, conditional_call, line_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel['calling_program'],
                    rel['called_program'],
                    rel['call_type'],
                    rel['call_location'],
                    json.dumps(rel['parameters']),
                    rel['call_statement'],
                    rel['conditional_call'],
                    rel['line_number']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(relationships)} program relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to store program relationships: {str(e)}")

    async def _store_copybook_relationships(self, relationships: List[Dict]):
        """Store copybook relationships in database"""
        if not relationships:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rel in relationships:
                cursor.execute("""
                    INSERT OR REPLACE INTO copybook_relationships 
                    (program_name, copybook_name, copy_location, replacing_clause,
                    copy_statement, line_number, usage_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel['program_name'],
                    rel['copybook_name'],
                    rel['copy_location'],
                    rel['replacing_clause'],
                    rel['copy_statement'],
                    rel['line_number'],
                    rel['usage_context']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(relationships)} copybook relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to store copybook relationships: {str(e)}")

    async def _store_file_relationships(self, relationships: List[Dict]):
        """Store file access relationships in database"""
        if not relationships:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rel in relationships:
                cursor.execute("""
                    INSERT OR IGNORE INTO file_access_relationships 
                    (program_name, logical_file_name, physical_file_name, access_type,
                    access_mode, record_format, access_location, line_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel['program_name'],
                    rel['logical_file_name'],
                    rel['physical_file_name'],
                    rel['access_type'],
                    rel['access_mode'],
                    rel['record_format'],
                    rel['access_location'],
                    rel['line_number']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(relationships)} file relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to store file relationships: {str(e)}")

    async def _store_field_cross_reference(self, field_refs: List[Dict]):
        """Store field cross-reference in database"""
        if not field_refs:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for ref in field_refs:
                cursor.execute("""
                    INSERT OR REPLACE INTO field_cross_reference 
                    (field_name, qualified_name, source_type, source_name,
                    definition_location, data_type, picture_clause, usage_clause,
                    level_number, parent_field, occurs_info, business_domain)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ref['field_name'],
                    ref['qualified_name'],
                    ref['source_type'],
                    ref['source_name'],
                    ref['definition_location'],
                    ref['data_type'],
                    ref['picture_clause'],
                    ref['usage_clause'],
                    ref['level_number'],
                    ref['parent_field'],
                    ref['occurs_info'],
                    ref['business_domain']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(field_refs)} field cross-references")
            
        except Exception as e:
            self.logger.error(f"Failed to store field cross-reference: {str(e)}")

    # HELPER METHODS

    def _find_containing_paragraph(self, content: str, position: int) -> str:
        """Find the paragraph containing the given position"""
        # Look backwards for paragraph definition
        before_content = content[:position]
        
        # Find last paragraph before this position
        paragraph_pattern = r'([A-Z0-9][A-Z0-9-]*)\s*\.\s*$'
        matches = list(re.finditer(paragraph_pattern, before_content, re.MULTILINE | re.IGNORECASE))
        
        if matches:
            return matches[-1].group(1)
        
        return 'UNKNOWN-PARAGRAPH'

    def _is_conditional_statement(self, content: str, position: int) -> bool:
        """Check if statement at position is within conditional logic"""
        # Look backward for IF/WHEN/EVALUATE without corresponding END
        before_content = content[:position]
        
        # Count IF and END-IF
        if_count = len(re.findall(r'\bIF\b', before_content, re.IGNORECASE))
        end_if_count = len(re.findall(r'\bEND-IF\b', before_content, re.IGNORECASE))
        
        # Count WHEN without END-EVALUATE
        when_count = len(re.findall(r'\bWHEN\b', before_content, re.IGNORECASE))
        eval_count = len(re.findall(r'\bEVALUATE\b', before_content, re.IGNORECASE))
        end_eval_count = len(re.findall(r'\bEND-EVALUATE\b', before_content, re.IGNORECASE))
        
        return (if_count > end_if_count) or (eval_count > end_eval_count and when_count > 0)

    def _determine_copy_location(self, content: str, position: int) -> str:
        """Determine which section contains the COPY statement"""
        before_content = content[:position]
        
        sections = [
            ('WORKING-STORAGE SECTION', 'WORKING-STORAGE'),
            ('FILE SECTION', 'FILE-SECTION'),
            ('LINKAGE SECTION', 'LINKAGE-SECTION'),
            ('LOCAL-STORAGE SECTION', 'LOCAL-STORAGE'),
            ('PROCEDURE DIVISION', 'PROCEDURE-DIVISION')
        ]
        
        current_section = 'UNKNOWN'
        for section_pattern, section_name in sections:
            if section_pattern in before_content.upper():
                current_section = section_name
        
        return current_section

    def _determine_usage_context(self, copy_location: str) -> str:
        """Determine usage context based on copy location"""
        context_map = {
            'WORKING-STORAGE': 'data_definition',
            'FILE-SECTION': 'file_record_definition',
            'LINKAGE-SECTION': 'parameter_definition',
            'LOCAL-STORAGE': 'local_data_definition',
            'PROCEDURE-DIVISION': 'procedure_logic'
        }
        
        return context_map.get(copy_location, 'unknown')

    def _determine_field_definition_location(self, content: str, position: int) -> str:
        """Determine where a field is defined"""
        return self._determine_copy_location(content, position)

    def _determine_parent_field(self, content: str, position: int, current_level: int) -> str:
        """Determine parent field for nested structures"""
        before_content = content[:position]
        lines = before_content.split('\n')
        
        # Look backwards for a field with lower level number
        for line in reversed(lines):
            level_match = re.match(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)', line, re.IGNORECASE)
            if level_match:
                level = int(level_match.group(1))
                if level < current_level:
                    return level_match.group(2)
        
        return ''

    def _build_qualified_name(self, field_name: str, parent_field: str) -> str:
        """Build qualified field name"""
        if parent_field:
            return f"{parent_field}.{field_name}"
        return field_name
# ==================== DATABASE AND STORAGE METHODS ====================

    def _init_enhanced_database(self):
        """Initialize enhanced database schema with comprehensive tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced main chunks table
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
                    confidence_score REAL DEFAULT 1.0,
                    llm_analysis TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    line_start INTEGER,
                    line_end INTEGER,
                    UNIQUE(program_name, chunk_id)
                )
            """)
            
            # Enhanced business rule violations table
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
                    confidence_score REAL DEFAULT 1.0,
                    auto_fixable BOOLEAN DEFAULT 0,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Enhanced control flow analysis table
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
                    complexity_score INTEGER DEFAULT 0,
                    business_function TEXT,
                    performance_impact TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Enhanced field lineage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    program_name TEXT,
                    paragraph TEXT,
                    operation TEXT,
                    source_file TEXT,
                    target_file TEXT,
                    transformation_logic TEXT,
                    data_type TEXT,
                    business_domain TEXT,
                    sensitivity_level TEXT,
                    last_used TIMESTAMP,
                    read_in TEXT,
                    updated_in TEXT,
                    purged_in TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Copybook structure analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS copybook_structures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    copybook_name TEXT NOT NULL,
                    layout_type TEXT NOT NULL,
                    record_layouts TEXT,
                    field_hierarchy TEXT,
                    occurs_structures TEXT,
                    redefines_structures TEXT,
                    replacing_parameters TEXT,
                    business_domain TEXT,
                    complexity_score INTEGER,
                    usage_frequency INTEGER DEFAULT 0,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # MQ program analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mq_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    connection_type TEXT,
                    message_paradigm TEXT,
                    queue_operations TEXT,
                    message_flow_patterns TEXT,
                    transaction_scope TEXT,
                    error_handling_strategy TEXT,
                    performance_characteristics TEXT,
                    business_purpose TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # DB2 stored procedure analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS db2_procedure_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    procedure_name TEXT NOT NULL,
                    parameter_list TEXT,
                    sql_operations TEXT,
                    cursor_definitions TEXT,
                    exception_handlers TEXT,
                    transaction_control TEXT,
                    performance_hints TEXT,
                    business_logic_complexity INTEGER,
                    security_classification TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # LLM analysis cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    analysis_type TEXT NOT NULL,
                    analysis_result TEXT,
                    confidence_score REAL,
                    model_version TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS program_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calling_program TEXT NOT NULL,
                called_program TEXT NOT NULL,
                call_type TEXT NOT NULL, -- LINK, XCTL, CALL, EXEC
                call_location TEXT, -- paragraph/section where call occurs
                parameters TEXT, -- JSON array of parameters passed
                call_statement TEXT, -- actual call statement
                conditional_call BOOLEAN DEFAULT 0, -- if call is within IF/WHEN
                line_number INTEGER,
                created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(calling_program, called_program, call_type, call_location)
                )
            """)
            
            # Copybook relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS copybook_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    copybook_name TEXT NOT NULL,
                    copy_location TEXT NOT NULL, -- section where copied (WS, LS, FD, etc)
                    replacing_clause TEXT, -- REPLACING clause if any
                    copy_statement TEXT, -- full COPY statement
                    line_number INTEGER,
                    usage_context TEXT, -- data_definition, procedure_logic, etc
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(program_name, copybook_name, copy_location)
                )
            """)
            
            # File access relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_access_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_name TEXT NOT NULL,
                    logical_file_name TEXT NOT NULL, -- logical file name
                    physical_file_name  TEXT, -- DDname or dataset name
                    access_type TEXT NOT NULL, -- SELECT, FD, READ, WRITE, REWRITE, DELETE
                    access_mode TEXT, -- INPUT, OUTPUT, I-O, EXTEND
                    record_format TEXT, -- from FD
                    access_location TEXT, -- paragraph where accessed
                    line_number INTEGER,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Cross-reference table for fields across programs/copybooks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_cross_reference (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    qualified_name TEXT, -- full qualified name if nested
                    source_type TEXT NOT NULL, -- program, copybook
                    source_name TEXT NOT NULL, -- program or copybook name
                    definition_location TEXT, -- WS, LS, FD, LINKAGE
                    data_type TEXT,
                    picture_clause TEXT,
                    usage_clause TEXT,
                    level_number INTEGER,
                    parent_field TEXT, -- for nested structures
                    occurs_info TEXT, -- JSON for array info
                    business_domain TEXT, -- financial, temporal, identifier, etc
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(field_name, source_name, definition_location)
                )
            """)
            
            # Program impact analysis - what affects what
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS impact_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_artifact TEXT NOT NULL, -- program/copybook/file name
                    source_type TEXT NOT NULL, -- program, copybook, file
                    dependent_artifact TEXT NOT NULL,
                    dependent_type TEXT NOT NULL,
                    relationship_type TEXT NOT NULL, -- calls, includes, accesses, defines
                    impact_level TEXT DEFAULT 'medium', -- low, medium, high, critical
                    change_propagation TEXT, -- how changes propagate
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_artifact, dependent_artifact, relationship_type)
                )
            """)
            
            # Performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_name ON program_chunks(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON program_chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON program_chunks(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_business_rules ON business_rule_violations(program_name, rule_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_lineage_field ON field_lineage(field_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_lineage_program ON field_lineage(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_copybook_name ON copybook_structures(copybook_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mq_program ON mq_analysis(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_db2_procedure ON db2_procedure_analysis(procedure_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_hash ON llm_analysis_cache(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_type ON llm_analysis_cache(analysis_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prog_rel_calling ON program_relationships(calling_program)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prog_rel_called ON program_relationships(called_program)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_copy_rel_program ON copybook_relationships(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_copy_rel_copybook ON copybook_relationships(copybook_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_rel_program ON file_access_relationships(program_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_rel_file ON file_access_relationships(logical_file_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_xref_field ON field_cross_reference(field_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_xref_source ON field_cross_reference(source_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_impact_source ON impact_analysis(source_artifact)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_impact_dependent ON impact_analysis(dependent_artifact)")
            


            conn.commit()
            conn.close()
            
            self.logger.info("Enhanced database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Enhanced database initialization failed: {str(e)}")
            raise

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
                    (program_name, rule_type, rule_name, severity, description, 
                     line_number, context, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    'business_rule',
                    violation.rule,
                    violation.severity,
                    violation.context,
                    violation.line_number,
                    violation.context,
                    1.0  # Default confidence for rule-based violations
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(violations)} business violations for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store business violations: {str(e)}")

    async def _analyze_control_flow(self, chunks: List[CodeChunk]) -> List[ControlFlowPath]:
        """Analyze control flow paths from parsed chunks"""
        paths = []
        
        # Extract paragraphs and their relationships
        paragraphs = {}
        perform_calls = {}
        
        for chunk in chunks:
            if chunk.chunk_type == "cobol_paragraph":
                para_name = chunk.metadata.get('paragraph_name', '')
                paragraphs[para_name] = {
                    'chunk': chunk,
                    'called_by': [],
                    'calls': []
                }
            elif 'perform' in chunk.chunk_type:
                targets = chunk.metadata.get('targets', [])
                perform_calls[chunk.chunk_id] = targets
        
        # Build call relationships
        for chunk_id, targets in perform_calls.items():
            for target in targets:
                if target in paragraphs:
                    paragraphs[target]['called_by'].append(chunk_id)
        
        # Create control flow paths
        path_id = 1
        for para_name, para_info in paragraphs.items():
            if para_info['called_by']:  # Only include called paragraphs
                path = ControlFlowPath(
                    path_id=f"PATH_{path_id}",
                    entry_point=para_name,
                    exit_points=[],  # Would need more analysis
                    conditions=[],   # Would need condition analysis
                    called_paragraphs=para_info['calls'],
                    data_accessed=[]  # Would need data flow analysis
                )
                paths.append(path)
                path_id += 1
        
        return paths

    async def _store_control_flow_analysis(self, paths: List[ControlFlowPath], program_name: str):
        """Store control flow analysis in database"""
        if not paths:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for path in paths:
                cursor.execute("""
                    INSERT OR REPLACE INTO control_flow_paths 
                    (program_name, path_id, entry_point, exit_points, conditions,
                     called_paragraphs, data_accessed, complexity_score, business_function)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    path.path_id,
                    path.entry_point,
                    json.dumps(path.exit_points),
                    json.dumps(path.conditions),
                    json.dumps(path.called_paragraphs),
                    json.dumps(path.data_accessed),
                    len(path.called_paragraphs),  # Simple complexity metric
                    'unknown'  # Would need business analysis
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(paths)} control flow paths for {program_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store control flow analysis: {str(e)}")

    async def _generate_field_lineage(self, program_name: str, chunks: List[CodeChunk]) -> List[Dict[str, Any]]:
        """Generate field lineage records from parsed chunks"""
        lineage_records = []
        
        for chunk in chunks:
            if chunk.chunk_type in ["cobol_data_group", "copybook_record_layout"]:
                fields = chunk.metadata.get('group_fields', [])
                for field in fields:
                    lineage_record = {
                        'field_name': field,
                        'program_name': program_name,
                        'paragraph': chunk.chunk_id,
                        'operation': 'definition',
                        'data_type': 'unknown',  # Would need PIC analysis
                        'business_domain': chunk.business_context.get('business_entity', 'unknown'),
                        'sensitivity_level': 'standard'
                    }
                    lineage_records.append(lineage_record)
        
        return lineage_records

    async def _store_field_lineage(self, lineage_records: List[Dict[str, Any]]):
        """Store field lineage records in database"""
        if not lineage_records:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for record in lineage_records:
                cursor.execute("""
                    INSERT OR REPLACE INTO field_lineage 
                    (field_name, program_name, paragraph, operation, data_type,
                     business_domain, sensitivity_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record['field_name'],
                    record['program_name'],
                    record['paragraph'],
                    record['operation'],
                    record['data_type'],
                    record['business_domain'],
                    record['sensitivity_level']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(lineage_records)} field lineage records")
            
        except Exception as e:
            self.logger.error(f"Failed to store field lineage: {str(e)}")

    
    # Add a method to force LLM analysis for debugging
    async def _force_llm_analysis_for_cobol(self, content: str, program_name: str) -> Dict[str, Any]:
        """Force LLM analysis for debugging purposes"""
        self.logger.info(f"🔍 FORCING LLM analysis for {program_name}")
        
        # Force basic business logic analysis
        business_analysis = await self._analyze_with_llm_cached(
            content, 'forced_business_analysis',
            """
            Analyze this COBOL program for business logic:
            
            {content}
            
            Identify:
            1. Main business purpose
            2. Key operations performed
            3. Data processing patterns
            4. Integration points
            5. Complexity assessment
            
            Return as JSON:
            {{
                "business_purpose": "description of main purpose",
                "key_operations": ["operation1", "operation2"],
                "data_patterns": ["pattern1", "pattern2"],
                "integration_points": ["database", "files", "other_programs"],
                "complexity": "low|medium|high",
                "maintainability": "good|fair|poor"
            }}
            """
        )
        
        return business_analysis



    async def _llm_analyze_db2_patterns(self, content: str) -> Dict[str, Any]:
        """Use LLM to analyze DB2 patterns - FIXED with size limits"""
        
        # SAFETY: Truncate content to prevent oversized prompts
        safe_content = self._safe_truncate_content(content, max_chars=2500)
        
        return await self._analyze_with_llm_cached(
            safe_content, 'db2_analysis',
            """
            Analyze this DB2 stored procedure sample for SQL patterns and complexity:
            
            {content}
            
            Identify:
            1. SQL operation complexity and performance characteristics
            2. Data access patterns and table relationships
            3. Transaction management and consistency
            4. Error handling and exception management
            5. Business logic implementation
            
            Return as JSON:
            {{
                "sql_complexity": "high",
                "data_access_patterns": ["sequential_scan", "indexed_lookup"],
                "transaction_management": "explicit_commit",
                "error_handling": "comprehensive",
                "business_logic": ["data_validation", "business_rules"],
                "performance_characteristics": {{"throughput": "medium", "latency": "low"}}
            }}
            """
        )

    async def _llm_analyze_cobol_sp_patterns(self, content: str) -> Dict[str, Any]:
        """Use LLM to analyze COBOL stored procedure patterns - FIXED with size limits"""
        
        # SAFETY: Truncate content to prevent oversized prompts
        safe_content = self._safe_truncate_content(content, max_chars=2500)
        
        return await self._analyze_with_llm_cached(
            safe_content, 'cobol_sp_analysis',
            """
            Analyze this COBOL stored procedure sample for SQL integration patterns:
            
            {content}
            
            Identify:
            1. Host variable usage and data binding
            2. SQL error handling strategies
            3. Result set processing patterns
            4. Transaction coordination
            5. Business process integration
            
            Return as JSON:
            {{
                "host_variable_usage": "comprehensive",
                "error_handling_strategy": "sqlcode_based",
                "result_set_processing": "cursor_based",
                "transaction_coordination": "cobol_managed",
                "business_integration": "batch_processing"
            }}
            """
        )

    async def _llm_analyze_cobol_sp_patterns(self, content: str) -> Dict[str, Any]:
        """Use LLM to analyze COBOL stored procedure patterns"""
        return await self._analyze_with_llm_cached(
            content, 'cobol_sp_analysis',
            """
            Analyze this COBOL stored procedure for SQL integration patterns:
            
            {content}
            
            Identify:
            1. Host variable usage and data binding
            2. SQL error handling strategies
            3. Result set processing patterns
            4. Transaction coordination
            5. Business process integration
            
            Return as JSON:
            {{
                "host_variable_usage": "comprehensive",
                "error_handling_strategy": "sqlcode_based",
                "result_set_processing": "cursor_based",
                "transaction_coordination": "cobol_managed",
                "business_integration": "batch_processing"
            }}
            """
        )

    async def _parse_sql_communication_areas_enhanced_safe(self, content: str, procedure_name: str,
                                                sp_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse SQL communication areas with enhanced safety"""
        chunks = []
        
        # Use safe pattern parsing
        sql_comm_patterns = {
            'sqlca': self.sql_patterns['sql_include_sqlca'],
            'sqlda': self.sql_patterns['sql_include_sqlda']
        }
        
        matches = self._safe_parse_sql_patterns(content, sql_comm_patterns)
        
        for match_info in matches:
            comm_type = 'SQLCA' if 'sqlca' in match_info['pattern_name'] else 'SQLDA'
            
            comm_analysis = await self._analyze_with_llm_cached(
                match_info['full_match'], f'sql_communication_{comm_type.lower()}',
                """
                Analyze this SQL communication area:
                
                {content}
                
                Identify:
                1. Error handling integration
                2. Status checking patterns
                3. Performance monitoring capability
                4. Transaction state management
                
                Return as JSON:
                {{
                    "error_integration": "comprehensive",
                    "status_checking": "after_each_sql",
                    "performance_monitoring": "enabled",
                    "transaction_management": "automatic"
                }}
                """
            )
            
            business_context = {
                'communication_type': comm_type,
                'error_integration': comm_analysis.get('analysis', {}).get('error_integration', ''),
                'status_checking': comm_analysis.get('analysis', {}).get('status_checking', ''),
                'purpose': 'sql_error_handling'
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_{comm_type}_INCLUDE",
                chunk_type=f"sql_communication_{comm_type.lower()}",
                content=match_info['full_match'],
                metadata={
                    'communication_type': comm_type,
                    'llm_analysis': comm_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=comm_analysis.get('confidence_score', 0.9),
                line_start=content[:match_info['start']].count('\n'),
                line_end=content[:match_info['end']].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _safe_extract_groups(self, match, expected_groups=1, default_values=None):
        """Safely extract regex groups with defaults"""
        if not match or not match.groups():
            return default_values or ['unknown'] * expected_groups
        
        result = []
        for i in range(expected_groups):
            if i < len(match.groups()) and match.group(i + 1):
                result.append(match.group(i + 1))
            else:
                default_val = default_values[i] if default_values and i < len(default_values) else 'unknown'
                result.append(default_val)
        
        return result
    
    async def _parse_embedded_sql_enhanced(self, content: str, procedure_name: str,
                                 sp_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse embedded SQL with enhanced host variable analysis"""
        chunks = []
        
        # Find WHENEVER statements
        whenever_matches = list(self.sql_patterns['sql_whenever_sqlerror'].finditer(content))
        whenever_matches.extend(list(self.sql_patterns['sql_whenever_not_found'].finditer(content)))
        whenever_matches.extend(list(self.sql_patterns['sql_whenever_sqlwarning'].finditer(content)))
        
        for match in whenever_matches:
            # Safely extract the whenever type
            whenever_type = self._safe_extract_groups(match, 1, ['unknown'])[0]
            
            whenever_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'sql_whenever_statement',
                """
                Analyze this SQL WHENEVER statement:
                
                {content}
                
                Identify:
                1. Error handling strategy
                2. Program flow impact
                3. Recovery mechanisms
                4. Business continuity approach
                
                Return as JSON:
                {{
                    "error_strategy": "goto_error_handler",
                    "flow_impact": "structured_exception_handling",
                    "recovery_mechanism": "rollback_and_retry",
                    "business_continuity": "transaction_safe"
                }}
                """
            )
            
            business_context = {
                'statement_type': 'error_handling',
                'error_strategy': whenever_analysis.get('analysis', {}).get('error_strategy', ''),
                'flow_impact': whenever_analysis.get('analysis', {}).get('flow_impact', ''),
                'business_continuity': whenever_analysis.get('analysis', {}).get('business_continuity', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_WHENEVER_{hash(match.group(0))%10000}",
                chunk_type="sql_whenever_statement",
                content=match.group(0),
                metadata={
                    'whenever_type': whenever_type,
                    'llm_analysis': whenever_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=whenever_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks


    async def _parse_result_set_handling(self, content: str, procedure_name: str,
                                   sp_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse result set handling patterns"""
        chunks = []
        
        # Find result set related SQL
        result_set_matches = list(self.sql_patterns['cobol_result_set'].finditer(content))
        allocate_matches = list(self.sql_patterns['cobol_allocate_cursor'].finditer(content))
        
        for match in result_set_matches + allocate_matches:
            rs_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'sql_result_set',
                """
                Analyze this result set handling:
                
                {content}
                
                Identify:
                1. Result set processing pattern
                2. Memory management approach
                3. Performance characteristics
                4. Business data flow
                
                Return as JSON:
                {{
                    "processing_pattern": "cursor_based_iteration",
                    "memory_management": "automatic_cleanup",
                    "performance": "streaming_optimized",
                    "data_flow": "batch_to_online"
                }}
                """
            )
            
            business_context = {
                'processing_type': 'result_set_handling',
                'processing_pattern': rs_analysis.get('analysis', {}).get('processing_pattern', ''),
                'performance': rs_analysis.get('analysis', {}).get('performance', ''),
                'data_flow': rs_analysis.get('analysis', {}).get('data_flow', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_RESULT_SET_{hash(match.group(0))%10000}",
                chunk_type="sql_result_set_handling",
                content=match.group(0),
                metadata={
                    'llm_analysis': rs_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=rs_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    async def _parse_procedure_calls_enhanced(self, content: str, procedure_name: str,
                                    sp_analysis: Dict[str, Any]) -> List[CodeChunk]:
        """Parse procedure calls with parameter analysis"""
        chunks = []
        
        # Find CALL statements
        call_matches = list(self.sql_patterns['cobol_procedure_call'].finditer(content))
        
        for match in call_matches:
            # Safely extract procedure name and parameters
            extracted = self._safe_extract_groups(match, 2, ['unknown', ''])
            called_proc = extracted[0]
            parameters = extracted[1]
            
            call_analysis = await self._analyze_with_llm_cached(
                match.group(0), 'sql_procedure_call',
                """
                Analyze this procedure call:
                
                {content}
                
                Identify:
                1. Parameter passing strategy
                2. Transaction coordination
                3. Error propagation
                4. Business process integration
                
                Return as JSON:
                {{
                    "parameter_strategy": "reference_passing",
                    "transaction_coordination": "nested_transaction",
                    "error_propagation": "exception_bubbling",
                    "business_integration": "service_orchestration"
                }}
                """
            )
            
            business_context = {
                'call_type': 'procedure_invocation',
                'called_procedure': called_proc,
                'parameter_strategy': call_analysis.get('analysis', {}).get('parameter_strategy', ''),
                'business_integration': call_analysis.get('analysis', {}).get('business_integration', 'unknown')
            }
            
            chunk = CodeChunk(
                program_name=procedure_name,
                chunk_id=f"{procedure_name}_CALL_{called_proc}",
                chunk_type="sql_procedure_call",
                content=match.group(0),
                metadata={
                    'called_procedure': called_proc,
                    'parameters': parameters,
                    'llm_analysis': call_analysis.get('analysis', {})
                },
                business_context=business_context,
                confidence_score=call_analysis.get('confidence_score', 0.8),
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks

    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing metadata to result"""
        result["processor"] = "CodeParserAgent"
        result["processor_version"] = "3.0.0-LLM-Enhanced"
        result["api_integration"] = True
        result["llm_enhanced"] = True
        return result


    # ==================== CORE PROCESSING METHODS ====================

    def _generate_file_hash(self, content: str, file_path: Path) -> str:
        """Generate unique hash for file content and metadata"""
        try:
            stat_info = file_path.stat()
            hash_input = f"{file_path.name}:{stat_info.st_mtime}:{len(content)}:{content[:100]}"
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception as e:
            # Fallback if file stat fails
            hash_input = f"{file_path.name}:{len(content)}:{content[:100]}"
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
            # Try COBOL PROGRAM-ID first
            program_match = self.cobol_patterns['program_id'].search(content)
            if program_match:
                return program_match.group(1).strip()
            
            # Try JCL job name
            job_match = self.jcl_patterns['job_card'].search(content)
            if job_match:
                return job_match.group(1).strip()
            
            # Try DB2 procedure name
            if hasattr(self, 'db2_patterns'):
                db2_match = self.db2_patterns['db2_create_procedure'].search(content)
                if db2_match:
                    proc_name = db2_match.group(1).strip()
                    # Remove schema if present
                    if '.' in proc_name:
                        proc_name = proc_name.split('.')[-1]
                    return proc_name
            
            # Extract from filename
            if isinstance(file_path, str):
                file_path = Path(file_path)
            filename = file_path.name
            
            # Remove common extensions
            for ext in ['.cbl', '.cob', '.jcl', '.copy', '.cpy', '.bms', '.sql', '.db2', '.mqt']:
                if filename.lower().endswith(ext):
                    return filename[:-len(ext)]
            
            return file_path.stem
            
        except Exception as e:
            self.logger.error(f"Error extracting program name: {str(e)}")
            if isinstance(file_path, (str, Path)):
                return Path(file_path).stem or "UNKNOWN_PROGRAM"
            return "UNKNOWN_PROGRAM"

    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Enhanced file type detection with comprehensive business rule ordering"""
        content_upper = content.upper()
        
        # Order matters - check most specific patterns first
        
        # DB2 Stored Procedure detection (most specific SQL patterns)
        if self._is_db2_procedure(content_upper):
            return 'db2_procedure'
        
        # MQ Program detection (check BEFORE COBOL stored procedure)
        if self._is_mq_program(content_upper):
            return 'mq_program'
        
        # COBOL Stored Procedure detection (after MQ check)
        if self._is_cobol_stored_procedure(content_upper):
            return 'cobol_stored_procedure'
        
        # BMS detection (most specific mapset format)
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
        elif suffix_lower == '.mqt':
            return 'mq_program'
        
        return 'unknown'

    

    def _is_db2_procedure(self, content_upper: str) -> bool:
        """Check if file is DB2 stored procedure"""
        return (any(marker in content_upper for marker in ['CREATE PROCEDURE', 'CREATE OR REPLACE PROCEDURE']) and
                any(marker in content_upper for marker in ['LANGUAGE SQL', 'PARAMETER STYLE', 'BEGIN ATOMIC', 'BEGIN']))

    def _is_cobol_stored_procedure(self, content_upper: str) -> bool:
        """Enhanced COBOL stored procedure detection - more restrictive"""
        # Must have PROCEDURE DIVISION (basic COBOL requirement)
        if 'PROCEDURE DIVISION' not in content_upper:
            return False
        
        # Check for explicit SQL stored procedure creation
        if 'EXEC SQL CREATE PROCEDURE' in content_upper:
            return True
        
        # Check for strong SQL integration indicators
        sql_indicators = [
            'SQLCA' in content_upper,
            'EXEC SQL' in content_upper,
            'SQLCODE' in content_upper,
            'SQLSTATE' in content_upper
        ]
        
        # Must have at least 2 SQL indicators AND specific stored procedure patterns
        sql_count = sum(sql_indicators)
        
        # Stored procedure specific patterns
        sp_patterns = [
            'EXEC SQL CALL' in content_upper,
            'RESULT SET' in content_upper,
            'ASSOCIATE LOCATOR' in content_upper,
            'ALLOCATE CURSOR' in content_upper
        ]
        
        sp_count = sum(sp_patterns)
        
        # Require strong evidence of stored procedure usage
        return sql_count >= 2 and sp_count >= 1
    
    def _is_mq_program(self, content_upper: str) -> bool:
        """Enhanced MQ program detection"""
        # MQ API calls (more comprehensive)
        mq_api_calls = [
            'MQOPEN', 'MQPUT', 'MQGET', 'MQCLOSE', 'MQCONN', 'MQCONNX', 'MQDISC', 
            'MQBEGIN', 'MQCMIT', 'MQBACK', 'MQPUT1', 'MQINQ', 'MQSET', 'MQSUB'
        ]
        
        # Count different patterns of MQ calls
        call_count = 0
        for api in mq_api_calls:
            # Check for CALL statements
            call_count += content_upper.count(f'CALL "{api}"')
            call_count += content_upper.count(f"CALL '{api}'")
            call_count += content_upper.count(f'CALL {api}')
        
        # MQ data structures
        mq_structures = ['MQMD', 'MQOD', 'MQPMO', 'MQGMO', 'MQSD', 'MQCNO', 'MQHCONN', 'MQHOBJ']
        structure_count = sum(content_upper.count(structure) for structure in mq_structures)
        
        # MQ specific variables and constants
        mq_constants = ['MQCC_', 'MQRC_', 'MQOT_', 'MQOO_', 'MQGMO_', 'MQPMO_']
        constant_count = sum(content_upper.count(constant) for constant in mq_constants)
        
        # Enhanced detection logic
        is_mq = (
            call_count >= 1 or  # At least 1 MQ API call
            structure_count >= 1 or  # At least 1 MQ structure
            constant_count >= 2 or  # At least 2 MQ constants
            ('QUEUE' in content_upper and 'MANAGER' in content_upper and call_count > 0)
        )
        
        return is_mq
    
    def _is_bms_file(self, content_upper: str) -> bool:
        """Check if file is BMS mapset"""
        return any(marker in content_upper for marker in ['DFHMSD', 'DFHMDI', 'DFHMDF'])

    def _is_heavy_cics_program(self, content_upper: str) -> bool:
        """Check if file is CICS-heavy program"""
        cics_count = content_upper.count('EXEC CICS')
        total_lines = content_upper.count('\n') + 1
        # More than 10% of lines contain CICS commands or at least 5 CICS commands
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
        has_copy_indicators = any(ind in content_upper for ind in ['COPY', 'REPLACING', 'OCCURS', 'REDEFINES'])
        
        return (has_data_items and no_divisions and len(content_upper.split('\n')) < 500) or has_copy_indicators

    async def _read_file_with_encoding(self, file_path: Path) -> Optional[str]:
        """Enhanced file reading with multiple encoding attempts"""
        encodings = ['utf-8', 'cp1252', 'latin1', 'ascii', 'utf-16', 'cp037', 'cp500']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                self.logger.debug(f"Successfully read file with {encoding} encoding")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                self.logger.warning(f"Error reading file with {encoding}: {e}")
                continue
        
        self.logger.error(f"Failed to read file {file_path} with any encoding")
        return None

    # ==================== LLM INTEGRATION METHODS ====================

    async def _analyze_with_llm_cached(self, content: str, analysis_type: str, 
                                  prompt_template: str, **kwargs) -> Dict[str, Any]:
        """Enhanced LLM analysis with PROPER chunking and comprehensive analysis"""
        
        # SAFETY CHECK: Validate content
        if not content or not content.strip():
            return {
                'analysis': {'error': 'Empty content provided'},
                'confidence_score': 0.0,
                'cached': False
            }
        
        # Check estimated token count
        estimated_tokens = self.context_manager.estimate_tokens(content)
        self.logger.info(f"🤖 LLM Analysis: {analysis_type} (content: {len(content)} chars, ~{estimated_tokens} tokens)")
        
        # Generate cache key for the FULL content (not truncated)
        cache_key_data = f"{content[:500]}:{analysis_type}:{prompt_template[:100]}"
        content_hash = hashlib.sha256(cache_key_data.encode()).hexdigest()
        
        # Check cache first
        cached_result = await self._check_llm_cache(content_hash, analysis_type)
        if cached_result:
            return cached_result
        
        # Determine processing strategy based on content size
        if estimated_tokens <= 1500:  # Small content - direct analysis
            return await self._analyze_direct(content, analysis_type, prompt_template, content_hash, **kwargs)
        else:  # Large content - chunking analysis
            return await self._analyze_with_chunking(content, analysis_type, prompt_template, content_hash, **kwargs)

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
            
            cached_result = cursor.fetchone()
            if cached_result:
                self.logger.info(f"📋 Cache HIT for {analysis_type}")
                cursor.execute("""
                    UPDATE llm_analysis_cache 
                    SET last_accessed = CURRENT_TIMESTAMP 
                    WHERE content_hash = ?
                """, (content_hash,))
                conn.commit()
                conn.close()
                
                return {
                    'analysis': json.loads(cached_result[0]),
                    'confidence_score': cached_result[1],
                    'cached': True
                }
            
            conn.close()
            return None
        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {e}")
            return None

    async def _analyze_direct(self, content: str, analysis_type: str, prompt_template: str, 
                        content_hash: str, **kwargs) -> Dict[str, Any]:
        """Direct LLM analysis for small content"""
        try:
            # Build prompt
            final_prompt = prompt_template.format(content=content, **kwargs)
            
            # Final safety check on prompt size
            prompt_tokens = self.context_manager.estimate_tokens(final_prompt)
            if prompt_tokens > 1800:
                self.logger.warning(f"⚠️ Prompt too large for direct analysis: {prompt_tokens} tokens, falling back to chunking")
                return await self._analyze_with_chunking(content, analysis_type, prompt_template, content_hash, **kwargs)
            
            self.logger.info(f"🚀 Direct analysis: ~{prompt_tokens} tokens")
            
            # Call LLM API
            response = await self.call_api(final_prompt, {
                "temperature": 0.1,
                "max_tokens": 1000
            })
            
            # Parse response
            analysis_result = self._parse_llm_response_enhanced(response, analysis_type)
            analysis_result['processing_method'] = 'direct'
            analysis_result['content_size'] = len(content)
            
            # Cache result
            await self._cache_llm_result(content_hash, analysis_type, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Direct LLM analysis failed: {e}")
            return {
                'analysis': self._create_fallback_analysis("", analysis_type),
                'confidence_score': 0.3,
                'error': str(e),
                'processing_method': 'fallback'
            }

    async def _analyze_with_chunking(self, content: str, analysis_type: str, prompt_template: str, 
                                content_hash: str, **kwargs) -> Dict[str, Any]:
        """Comprehensive chunked analysis for large content"""
        
        self.logger.info(f"📊 Starting chunked analysis for {analysis_type}")
        
        try:
            # Determine chunk type based on analysis type
            chunk_type = self._determine_chunk_type(analysis_type, content)
            
            # Create intelligent chunks using the context manager
            chunks = self.context_manager.chunk_content_intelligently(
                content, 
                chunk_type=chunk_type, 
                overlap=200
            )
            
            if not chunks:
                self.logger.warning("No chunks created, falling back to truncated analysis")
                truncated_content = self._safe_truncate_content(content, max_chars=3000)
                return await self._analyze_direct(truncated_content, analysis_type, prompt_template, content_hash, **kwargs)
            
            self.logger.info(f"🔍 Analyzing {len(chunks)} chunks for {analysis_type}")
            
            # Analyze each chunk
            chunk_analyses = []
            successful_analyses = 0
            
            for i, chunk_info in enumerate(chunks):
                try:
                    chunk_analysis = await self._analyze_single_chunk(
                        chunk_info, analysis_type, prompt_template, i, len(chunks), **kwargs
                    )
                    
                    if chunk_analysis and 'analysis' in chunk_analysis:
                        chunk_analyses.append(chunk_analysis)
                        successful_analyses += 1
                        self.logger.debug(f"✅ Chunk {i+1}/{len(chunks)} analyzed successfully")
                    else:
                        self.logger.warning(f"⚠️ Chunk {i+1}/{len(chunks)} analysis failed")
                    
                    # Add small delay to prevent API rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.warning(f"Chunk {i+1} analysis failed: {e}")
                    continue
            
            # Check if we have enough successful analyses
            if successful_analyses == 0:
                self.logger.error("❌ All chunk analyses failed")
                return {
                    'analysis': self._create_fallback_analysis("", analysis_type),
                    'confidence_score': 0.2,
                    'error': 'All chunk analyses failed',
                    'processing_method': 'failed_chunking'
                }
            
            # Aggregate results from all chunks
            aggregated_analysis = self._aggregate_chunk_analyses(chunk_analyses, analysis_type)
            
            # Calculate overall confidence based on successful chunk ratio
            chunk_success_ratio = successful_analyses / len(chunks)
            base_confidence = sum(ca.get('confidence_score', 0.5) for ca in chunk_analyses) / len(chunk_analyses)
            overall_confidence = base_confidence * chunk_success_ratio
            
            final_result = {
                'analysis': aggregated_analysis,
                'confidence_score': overall_confidence,
                'processing_method': 'intelligent_chunking',
                'chunks_processed': len(chunks),
                'chunks_successful': successful_analyses,
                'chunk_success_ratio': chunk_success_ratio,
                'original_content_size': len(content),
                'cached': False
            }
            
            # Cache the aggregated result
            await self._cache_llm_result(content_hash, analysis_type, final_result)
            
            self.logger.info(f"✅ Chunked analysis complete: {successful_analyses}/{len(chunks)} chunks successful")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Chunked analysis failed: {e}")
            return {
                'analysis': self._create_fallback_analysis("", analysis_type),
                'confidence_score': 0.3,
                'error': str(e),
                'processing_method': 'failed_chunking'
            }

    def _determine_chunk_type(self, analysis_type: str, content: str) -> str:
        """Determine the best chunk type based on analysis type and content"""
    
        content_upper = content.upper()
        
        # Map analysis types to chunk types
        if 'cobol' in analysis_type:
            if 'IDENTIFICATION DIVISION' in content_upper:
                return 'cobol'
            else:
                return 'code'
        elif 'sql' in analysis_type or 'db2' in analysis_type:
            return 'sql'
        elif 'jcl' in analysis_type:
            return 'jcl'
        elif 'cics' in analysis_type:
            return 'cobol'  # CICS is COBOL-based
        elif 'mq' in analysis_type:
            return 'cobol'  # MQ programs are typically COBOL
        else:
            return 'code'

    async def _analyze_single_chunk(self, chunk_info: Dict[str, Any], analysis_type: str, 
                                prompt_template: str, chunk_index: int, total_chunks: int, 
                                **kwargs) -> Dict[str, Any]:
        """Analyze a single chunk with enhanced context"""
        
        chunk_content = chunk_info['content']
        
        # Build enhanced prompt with chunk context
        enhanced_prompt = self._build_enhanced_prompt(
            prompt_template, chunk_content, chunk_info, analysis_type, **kwargs
        )
        
        # Check prompt size
        prompt_tokens = self.context_manager.estimate_tokens(enhanced_prompt)
        if prompt_tokens > 1800:
            # If still too large, truncate the chunk content but keep context
            self.logger.warning(f"Chunk {chunk_index+1} still too large, truncating")
            truncated_content = self._safe_truncate_content(chunk_content, max_chars=2000)
            enhanced_prompt = self._build_enhanced_prompt(
                prompt_template, truncated_content, chunk_info, analysis_type, **kwargs
            )
        
        try:
            # Call LLM API for this chunk
            response = await self.call_api(enhanced_prompt, {
                "temperature": 0.1,
                "max_tokens": 800  # Smaller response for chunks
            })
            
            # Parse response
            chunk_analysis = self._parse_llm_response_enhanced(response, analysis_type)
            
            # Add chunk metadata
            chunk_analysis['chunk_info'] = {
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'chunk_type': chunk_info.get('chunk_type', 'unknown'),
                'estimated_tokens': chunk_info.get('estimated_tokens', 0)
            }
            
            return chunk_analysis
            
        except Exception as e:
            self.logger.warning(f"Single chunk analysis failed: {e}")
            return {
                'analysis': self._create_fallback_analysis(chunk_content[:100], analysis_type),
                'confidence_score': 0.3,
                'error': str(e),
                'chunk_info': {'chunk_index': chunk_index, 'failed': True}
        }


    def _build_enhanced_prompt(self, prompt_template: str, chunk_content: str, 
                         chunk_info: Dict[str, Any], analysis_type: str, **kwargs) -> str:
        """Build enhanced prompt with chunk context information"""
        
        # Add chunk context information
        chunk_context = ""
        if chunk_info.get('total_chunks', 1) > 1:
            chunk_context = f"""
            
    CHUNK CONTEXT:
    - This is chunk {chunk_info['chunk_index'] + 1} of {chunk_info['total_chunks']} total chunks
    - Chunk type: {chunk_info.get('chunk_type', 'unknown')}
    - Estimated tokens in this chunk: {chunk_info.get('estimated_tokens', 'unknown')}
    - Overlap with adjacent chunks: {chunk_info.get('overlap_start', False)} / {chunk_info.get('overlap_end', False)}

    Please analyze this chunk as part of a larger codebase. Focus on complete elements within this chunk and note any incomplete elements at boundaries.
            """
        
        # Build the enhanced prompt
        enhanced_prompt = prompt_template.format(
            content=chunk_content,
            **kwargs
        ) + chunk_context
        
        return enhanced_prompt


    def _parse_llm_response_enhanced(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Simple response parsing for direct (non-chunked) analysis"""
        
        result = {
            'analysis': {},
            'confidence_score': 0.5
        }
        
        # Try JSON parsing first
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_text = response[json_start:json_end]
                parsed_analysis = json.loads(json_text)
                result['analysis'] = parsed_analysis
                result['confidence_score'] = 0.8
                return result
        except json.JSONDecodeError:
            pass
        
        # Fallback to structured text parsing
        try:
            result['analysis'] = self._parse_structured_text_response(response, analysis_type)
            result['confidence_score'] = 0.6
        except Exception:
            result['analysis'] = self._create_fallback_analysis(response, analysis_type)
            result['confidence_score'] = 0.4
        
        return result

    def _aggregate_chunk_analyses(self, chunk_analyses: List[Dict[str, Any]], 
                                analysis_type: str) -> Dict[str, Any]:
        """Aggregate multiple chunk analyses into a comprehensive result"""
        
        if len(chunk_analyses) == 1:
            return chunk_analyses[0]['analysis']
        
        aggregated = {
            'analysis_type': analysis_type,
            'total_chunks_analyzed': len(chunk_analyses),
            'aggregation_method': 'intelligent_merge'
        }
        
        # Type-specific aggregation logic
        if analysis_type in ['cobol_section', 'cobol_paragraph']:
            aggregated.update(self._aggregate_cobol_analysis(chunk_analyses))
        elif 'sql' in analysis_type or 'db2' in analysis_type:
            aggregated.update(self._aggregate_sql_analysis(chunk_analyses))
        elif 'mq' in analysis_type:
            aggregated.update(self._aggregate_mq_analysis(chunk_analyses))
        elif 'cics' in analysis_type:
            aggregated.update(self._aggregate_cics_analysis(chunk_analyses))
        else:
            aggregated.update(self._aggregate_generic_analysis(chunk_analyses))
        
        return aggregated
    
    async def _cache_llm_result(self, content_hash: str, analysis_type: str, 
                               analysis_result: Dict[str, Any]):
        """Cache LLM analysis result for future use"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract analysis and confidence
            analysis_data = analysis_result.get('analysis', {})
            confidence_score = analysis_result.get('confidence_score', 0.5)
            
            cursor.execute("""
                INSERT OR REPLACE INTO llm_analysis_cache 
                (content_hash, analysis_type, analysis_result, confidence_score, 
                 model_version, created_timestamp, last_accessed)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (
                content_hash,
                analysis_type,
                json.dumps(analysis_data),
                confidence_score,
                "enhanced_3.0"  # Model version
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Cached LLM result for {analysis_type} (hash: {content_hash[:8]}...)")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache LLM result: {e}")

    # ==================== FIELD ANALYSIS HELPER METHODS ====================
    
    def _extract_pic_clause(self, definition: str) -> str:
        """Extract PIC clause from field definition"""
        pic_match = re.search(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', 
                             definition, re.IGNORECASE)
        return pic_match.group(1) if pic_match else ""
    
    def _extract_usage_clause(self, definition: str) -> str:
        """Extract USAGE clause from field definition"""
        usage_match = re.search(
            r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX|POINTER)', 
            definition, re.IGNORECASE
        )
        return usage_match.group(1) if usage_match else ""
    
    def _extract_occurs_info(self, definition: str) -> Dict[str, Any]:
        """Extract OCCURS information from field definition"""
        occurs_info = {}
        
        # Basic OCCURS pattern
        occurs_match = re.search(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES', 
                                definition, re.IGNORECASE)
        if occurs_match:
            occurs_info['min_occurs'] = int(occurs_match.group(1))
            occurs_info['max_occurs'] = int(occurs_match.group(2)) if occurs_match.group(2) else int(occurs_match.group(1))
            
            # DEPENDING ON clause
            depending_match = re.search(r'DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', 
                                       definition, re.IGNORECASE)
            if depending_match:
                occurs_info['depending_on'] = depending_match.group(1)
            
            # INDEXED BY clause
            indexed_match = re.search(r'INDEXED\s+BY\s+([A-Z][A-Z0-9-]*(?:\s*,\s*[A-Z][A-Z0-9-]*)*)', 
                                     definition, re.IGNORECASE)
            if indexed_match:
                occurs_info['indexed_by'] = [idx.strip() for idx in indexed_match.group(1).split(',')]
        
        return occurs_info
    
    def _determine_data_type_enhanced(self, definition: str) -> str:
        """Determine enhanced data type from field definition"""
        definition_upper = definition.upper()
        
        # Check for specific PIC patterns
        if 'PIC' in definition_upper:
            pic_match = re.search(r'PIC(?:TURE)?\s+(?:IS\s+)?([X9AVSN\(\)\+\-\.,/ZB*$]+)', 
                                 definition_upper)
            if pic_match:
                pic_clause = pic_match.group(1)
                
                if 'X' in pic_clause:
                    return 'alphanumeric'
                elif '9' in pic_clause:
                    if 'V' in pic_clause or '.' in pic_clause:
                        return 'decimal'
                    else:
                        return 'numeric'
                elif 'A' in pic_clause:
                    return 'alphabetic'
                elif 'S' in pic_clause:
                    return 'signed_numeric'
        
        # Check for USAGE clause
        if 'COMP' in definition_upper:
            if 'COMP-3' in definition_upper:
                return 'packed_decimal'
            elif 'COMP-1' in definition_upper:
                return 'single_float'
            elif 'COMP-2' in definition_upper:
                return 'double_float'
            else:
                return 'binary'
        
        # Check for group items (no PIC clause)
        if 'PIC' not in definition_upper and re.search(r'^\s*\d+\s+[A-Z]', definition):
            return 'group_item'
        
        return 'unknown'
    
    def _infer_business_domain(self, field_name: str) -> str:
        """Infer business domain from field name"""
        field_upper = field_name.upper()
        
        # Financial domain
        if any(term in field_upper for term in ['AMOUNT', 'AMT', 'BALANCE', 'BAL', 'RATE', 'FEE', 'COST', 'PRICE']):
            return 'financial'
        
        # Temporal domain
        if any(term in field_upper for term in ['DATE', 'TIME', 'TIMESTAMP', 'DT', 'TM', 'YEAR', 'MONTH', 'DAY']):
            return 'temporal'
        
        # Customer domain
        if any(term in field_upper for term in ['CUSTOMER', 'CUST', 'CLIENT', 'NAME', 'ADDR', 'ADDRESS', 'PHONE', 'EMAIL']):
            return 'customer'
        
        # Account domain
        if any(term in field_upper for term in ['ACCOUNT', 'ACCT', 'ACC', 'NUMBER', 'NUM', 'ID', 'IDENTIFIER']):
            return 'account'
        
        # Transaction domain
        if any(term in field_upper for term in ['TRANS', 'TXN', 'TRANSACTION', 'PAYMENT', 'TRANSFER']):
            return 'transaction'
        
        # System/Control domain
        if any(term in field_upper for term in ['STATUS', 'STAT', 'FLAG', 'IND', 'INDICATOR', 'CODE', 'TYPE']):
            return 'control'
        
        # Product domain
        if any(term in field_upper for term in ['PRODUCT', 'PROD', 'ITEM', 'SERVICE', 'SVC']):
            return 'product'
        
        # Location domain
        if any(term in field_upper for term in ['BRANCH', 'LOCATION', 'LOC', 'REGION', 'AREA', 'ZONE']):
            return 'location'
        
        return 'general'

    # ==================== COBOL STORED PROCEDURE PARSING ====================
    
    async def _parse_cobol_stored_procedure_with_enhanced_analysis(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse COBOL stored procedure with enhanced analysis - corrected method name"""
        return await self.parse_cobol_stored_procedure_with_enhanced_analysis(content, filename)

    # ==================== COPYBOOK LAYOUT ANALYSIS METHODS ====================
    
    async def _analyze_copybook_layout(self, content: str, copybook_name: str) -> Dict[str, Any]:
        """Analyze copybook layout type and complexity"""
        
        # Count different layout indicators using safe pattern access
        layout_indicators = 0
        record_types = 0
        redefines_count = 0
        occurs_count = 0
        conditional_fields = 0
        
        try:
            # Safely check for patterns
            if hasattr(self, 'copybook_patterns'):
                layout_indicators = len(self.copybook_patterns.get('layout_indicator', re.compile(r'(?!)')).findall(content))
                record_types = len(self.copybook_patterns.get('record_type_field', re.compile(r'(?!)')).findall(content))
                redefines_count = len(self.copybook_patterns.get('redefines_complex', re.compile(r'(?!)')).findall(content))
                occurs_count = len(self.copybook_patterns.get('occurs_complex', re.compile(r'(?!)')).findall(content))
                conditional_fields = len(self.copybook_patterns.get('conditional_field', re.compile(r'(?!)')).findall(content))
        except Exception as e:
            self.logger.warning(f"Pattern matching failed in copybook analysis: {e}")
        
        # Determine layout type
        from enum import Enum
        
        # Define enum if not imported
        class CopybookLayoutType(Enum):
            SINGLE_RECORD = "single_record"
            MULTI_RECORD = "multi_record" 
            CONDITIONAL_LAYOUT = "conditional_layout"
            REDEFINES_LAYOUT = "redefines_layout"
            OCCURS_LAYOUT = "occurs_layout"
        
        if layout_indicators > 1 or record_types > 1:
            layout_type = CopybookLayoutType.MULTI_RECORD
        elif conditional_fields > 0:
            layout_type = CopybookLayoutType.CONDITIONAL_LAYOUT
        elif redefines_count > 2:
            layout_type = CopybookLayoutType.REDEFINES_LAYOUT
        elif occurs_count > 1:
            layout_type = CopybookLayoutType.OCCURS_LAYOUT
        else:
            layout_type = CopybookLayoutType.SINGLE_RECORD
        
        # Calculate complexity score
        complexity_score = (layout_indicators * 3 + record_types * 2 + 
                          redefines_count * 2 + occurs_count * 1 + conditional_fields * 2)
        
        return {
            'layout_type': layout_type,
            'complexity_score': min(complexity_score, 20),
            'layout_indicators': layout_indicators,
            'record_types': record_types,
            'redefines_count': redefines_count,
            'occurs_count': occurs_count,
            'conditional_fields': conditional_fields
        }

    # ==================== IMPACT ANALYSIS METHODS ====================
    
    async def _generate_impact_analysis(self, artifact_name: str, artifact_type: str):
        """Generate impact analysis for an artifact"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            impacts = []
            
            if artifact_type in ['cobol', 'cics']:
                # Programs that call this program
                cursor.execute("""
                    SELECT DISTINCT calling_program, call_type 
                    FROM program_relationships 
                    WHERE called_program = ?
                """, (artifact_name,))
                
                for row in cursor.fetchall():
                    impacts.append({
                        'source_artifact': row[0],
                        'source_type': 'program',
                        'dependent_artifact': artifact_name,
                        'dependent_type': artifact_type,
                        'relationship_type': f'calls_via_{row[1]}',
                        'impact_level': 'high',
                        'change_propagation': 'interface_change_required'
                    })
                
                # Programs that use same copybooks
                cursor.execute("""
                    SELECT DISTINCT cr2.program_name, cr1.copybook_name
                    FROM copybook_relationships cr1
                    JOIN copybook_relationships cr2 ON cr1.copybook_name = cr2.copybook_name
                    WHERE cr1.program_name = ? AND cr2.program_name != ?
                """, (artifact_name, artifact_name))
                
                for row in cursor.fetchall():
                    impacts.append({
                        'source_artifact': artifact_name,
                        'source_type': artifact_type,
                        'dependent_artifact': row[0],
                        'dependent_type': 'program',
                        'relationship_type': f'shares_copybook_{row[1]}',
                        'impact_level': 'medium',
                        'change_propagation': 'data_structure_dependency'
                    })
            
            # Store impact analysis
            for impact in impacts:
                cursor.execute("""
                    INSERT OR REPLACE INTO impact_analysis 
                    (source_artifact, source_type, dependent_artifact, dependent_type,
                    relationship_type, impact_level, change_propagation)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    impact['source_artifact'],
                    impact['source_type'],
                    impact['dependent_artifact'],
                    impact['dependent_type'],
                    impact['relationship_type'],
                    impact['impact_level'],
                    impact['change_propagation']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Generated {len(impacts)} impact relationships for {artifact_name}")
            
        except Exception as e:
            self.logger.error(f"Impact analysis failed for {artifact_name}: {str(e)}")

    # ==================== ADDITIONAL HELPER METHODS ====================
    
    def _validate_chunk_content(self, chunk_content: str, min_length: int = 10) -> bool:
        """Validate chunk content before processing"""
        if not chunk_content or len(chunk_content.strip()) < min_length:
            return False
        
        # Check for meaningful content (not just whitespace or comments)
        lines = chunk_content.strip().split('\n')
        meaningful_lines = [line for line in lines 
                          if line.strip() and not line.strip().startswith('*')]
        
        return len(meaningful_lines) > 0
    
    def _estimate_processing_complexity(self, content: str, file_type: str) -> str:
        """Estimate processing complexity for the content"""
        content_length = len(content)
        line_count = content.count('\n') + 1
        
        # Base complexity on file type and size
        if file_type in ['db2_procedure', 'cobol_stored_procedure']:
            complexity_threshold_low = 500
            complexity_threshold_high = 2000
        elif file_type in ['mq_program', 'cics']:
            complexity_threshold_low = 1000  
            complexity_threshold_high = 3000
        else:
            complexity_threshold_low = 800
            complexity_threshold_high = 2500
        
        # Count complexity indicators
        complexity_indicators = 0
        if 'PERFORM' in content.upper():
            complexity_indicators += content.upper().count('PERFORM')
        if 'IF' in content.upper():
            complexity_indicators += content.upper().count('IF')
        if 'EVALUATE' in content.upper():
            complexity_indicators += content.upper().count('EVALUATE') * 2
        if 'EXEC SQL' in content.upper():
            complexity_indicators += content.upper().count('EXEC SQL')
        if 'EXEC CICS' in content.upper():
            complexity_indicators += content.upper().count('EXEC CICS')
        
        # Determine complexity
        if content_length < complexity_threshold_low and complexity_indicators < 10:
            return 'low'
        elif content_length > complexity_threshold_high or complexity_indicators > 25:
            return 'high'
        else:
            return 'medium'
    
    def _create_processing_summary(self, chunks: List, violations: List, 
                                 file_type: str, processing_time: float) -> Dict[str, Any]:
        """Create a comprehensive processing summary"""
        
        chunk_type_counts = {}
        for chunk in chunks:
            chunk_type = getattr(chunk, 'chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
        
        violation_severity_counts = {}
        for violation in violations:
            severity = getattr(violation, 'severity', 'unknown')
            violation_severity_counts[severity] = violation_severity_counts.get(severity, 0) + 1
        
        # Calculate quality metrics
        total_confidence = sum(getattr(chunk, 'confidence_score', 0.5) for chunk in chunks)
        avg_confidence = total_confidence / len(chunks) if chunks else 0.0
        
        high_confidence_chunks = sum(1 for chunk in chunks 
                                   if getattr(chunk, 'confidence_score', 0.5) >= 0.8)
        
        return {
            'processing_summary': {
                'total_chunks': len(chunks),
                'chunk_types': chunk_type_counts,
                'total_violations': len(violations),
                'violation_severities': violation_severity_counts,
                'quality_metrics': {
                    'average_confidence': round(avg_confidence, 3),
                    'high_confidence_chunks': high_confidence_chunks,
                    'confidence_percentage': round((high_confidence_chunks / len(chunks)) * 100, 1) if chunks else 0
                },
                'processing_metrics': {
                    'file_type': file_type,
                    'processing_time_seconds': round(processing_time, 2),
                    'chunks_per_second': round(len(chunks) / max(processing_time, 0.1), 2)
                }
            }
        }

    # ==================== ERROR RECOVERY METHODS ====================
    
    def _recover_from_parsing_error(self, content: str, filename: str, error: Exception) -> List:
        """Attempt to recover from parsing errors with fallback parsing"""
        self.logger.warning(f"Attempting error recovery for {filename}: {error}")
        
        try:
            # Try simple line-based chunking as fallback
            chunks = []
            program_name = self._extract_program_name(content, filename)
            
            lines = content.split('\n')
            chunk_size = 30  # Smaller chunks for error recovery
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunk_content = '\n'.join(chunk_lines)
                
                if chunk_content.strip():
                    # Import here to avoid circular imports
                    from dataclasses import dataclass
                    
                    @dataclass 
                    class SimpleChunk:
                        program_name: str
                        chunk_id: str
                        chunk_type: str
                        content: str
                        metadata: dict
                        business_context: dict
                        confidence_score: float
                        line_start: int
                        line_end: int
                    
                    chunk = SimpleChunk(
                        program_name=program_name,
                        chunk_id=f"{program_name}_RECOVERY_{i//chunk_size + 1}",
                        chunk_type="recovery_chunk",
                        content=chunk_content,
                        metadata={
                            'recovery_mode': True,
                            'original_error': str(error),
                            'chunk_number': i//chunk_size + 1,
                            'line_count': len(chunk_lines)
                        },
                        business_context={'parsing_mode': 'error_recovery'},
                        confidence_score=0.3,  # Low confidence for recovery chunks
                        line_start=i,
                        line_end=min(i + chunk_size, len(lines))
                    )
                    chunks.append(chunk)
            
            self.logger.info(f"Error recovery created {len(chunks)} fallback chunks")
            return chunks
            
        except Exception as recovery_error:
            self.logger.error(f"Error recovery also failed: {recovery_error}")
            return []
    
    def _log_parsing_metrics(self, filename: str, chunks_created: int, 
                           processing_time: float, file_type: str):
        """Log comprehensive parsing metrics"""
        self.logger.info(f"""
        📊 PARSING METRICS for {filename}:
        ├── File Type: {file_type}
        ├── Chunks Created: {chunks_created}
        ├── Processing Time: {processing_time:.2f}s
        ├── Chunks/Second: {chunks_created / max(processing_time, 0.1):.1f}
        └── Status: {'✅ SUCCESS' if chunks_created > 0 else '❌ NO CHUNKS'}
        """)

# ==================== ADDITIONAL UTILITY FUNCTIONS ====================

    def create_safe_chunk_id(program_name: str, chunk_type: str, 
                            identifier: str = "", max_length: int = 100) -> str:
        """Create a safe chunk ID with length limits"""
        # Clean inputs
        safe_program = re.sub(r'[^A-Z0-9_-]', '', program_name.upper())[:20]
        safe_type = re.sub(r'[^A-Z0-9_-]', '', chunk_type.upper())[:30]
        safe_identifier = re.sub(r'[^A-Z0-9_-]', '', identifier.upper())[:20]
        
        # Build ID
        parts = [safe_program, safe_type]
        if safe_identifier:
            parts.append(safe_identifier)
        
        chunk_id = "_".join(parts)
        
        # Ensure it doesn't exceed max length
        if len(chunk_id) > max_length:
            chunk_id = chunk_id[:max_length-8] + "_" + hashlib.md5(chunk_id.encode()).hexdigest()[:7]
        
        return chunk_id

    def sanitize_business_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize business context to ensure serializability"""
        sanitized = {}
        
        for key, value in context.items():
            try:
                # Convert key to string
                safe_key = str(key)[:50]  # Limit key length
                
                # Handle different value types
                if isinstance(value, (str, int, float, bool)):
                    sanitized[safe_key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert to list and sanitize elements
                    sanitized[safe_key] = [str(item)[:100] for item in value[:10]]  # Limit items
                elif isinstance(value, dict):
                    # Recursively sanitize nested dict
                    sanitized[safe_key] = sanitize_business_context(value)
                else:
                    # Convert unknown types to string
                    sanitized[safe_key] = str(value)[:200]
                    
            except Exception as e:
                # Skip problematic entries
                continue
        
        return sanitized

    # ==================== PATTERN VALIDATION HELPERS ====================

    def validate_cobol_patterns(patterns: Dict[str, Any]) -> Dict[str, bool]:
        """Validate COBOL pattern definitions"""
        validation_results = {}
        
        required_patterns = [
            'program_id', 'procedure_division', 'data_division',
            'working_storage', 'perform_simple', 'if_statement',
            'sql_block', 'copy_statement'
        ]
        
        for pattern_name in required_patterns:
            try:
                pattern = patterns.get(pattern_name)
                if pattern and hasattr(pattern, 'pattern'):
                    # Test the pattern with a simple string
                    test_result = pattern.search("TEST PROGRAM-ID. TESTPROG.")
                    validation_results[pattern_name] = True
                else:
                    validation_results[pattern_name] = False
            except Exception:
                validation_results[pattern_name] = False
        
        return validation_results

    def create_fallback_patterns() -> Dict[str, Any]:
        """Create fallback patterns if main patterns fail to load"""
        import re
        
        fallback_patterns = {
            'program_id': re.compile(r'PROGRAM-ID\s*\.\s*([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'procedure_division': re.compile(r'PROCEDURE\s+DIVISION', re.IGNORECASE),
            'data_division': re.compile(r'DATA\s+DIVISION', re.IGNORECASE),
            'working_storage': re.compile(r'WORKING-STORAGE\s+SECTION', re.IGNORECASE),
            'perform_simple': re.compile(r'\bPERFORM\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'if_statement': re.compile(r'\bIF\s+', re.IGNORECASE),
            'sql_block': re.compile(r'EXEC\s+SQL.*?END-EXEC', re.DOTALL | re.IGNORECASE),
            'copy_statement': re.compile(r'\bCOPY\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        }
        
        return fallback_patterns
        def _aggregate_cobol_analysis(self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Aggregate COBOL-specific analyses"""
            all_purposes = []
            all_operations = []
            complexity_scores = []
            business_functions = []
            
            for chunk_analysis in chunk_analyses:
                analysis = chunk_analysis.get('analysis', {})
                
                if 'purpose' in analysis:
                    all_purposes.append(analysis['purpose'])
                if 'operations' in analysis:
                    if isinstance(analysis['operations'], list):
                        all_operations.extend(analysis['operations'])
                    else:
                        all_operations.append(analysis['operations'])
                if 'complexity' in analysis:
                    complexity_mapping = {'low': 1, 'medium': 2, 'high': 3}
                    complexity_scores.append(complexity_mapping.get(analysis['complexity'], 2))
                if 'business_function' in analysis:
                    business_functions.append(analysis['business_function'])
            
            # Determine overall complexity
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 2
            overall_complexity = 'low' if avg_complexity < 1.5 else 'high' if avg_complexity > 2.5 else 'medium'
            
            return {
                'consolidated_purpose': ' | '.join(set(all_purposes)) if all_purposes else 'multiple_functions',
                'all_operations': list(set(all_operations)),
                'overall_complexity': overall_complexity,
                'business_functions': list(set(business_functions)),
                'analysis_coverage': 'comprehensive'
            }

    def _aggregate_sql_analysis(self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate SQL-specific analyses"""
        all_operations = []
        all_tables = []
        complexity_indicators = []
        
        for chunk_analysis in chunk_analyses:
            analysis = chunk_analysis.get('analysis', {})
            
            if 'operation_type' in analysis:
                all_operations.append(analysis['operation_type'])
            if 'tables_accessed' in analysis:
                if isinstance(analysis['tables_accessed'], list):
                    all_tables.extend(analysis['tables_accessed'])
            if 'complexity' in analysis:
                complexity_indicators.append(analysis['complexity'])
        
        return {
            'sql_operations': list(set(all_operations)),
            'all_tables_accessed': list(set(all_tables)),
            'operation_complexity': max(set(complexity_indicators), key=complexity_indicators.count) if complexity_indicators else 'medium',
            'database_integration': 'comprehensive' if len(set(all_tables)) > 3 else 'standard'
        }

    def _aggregate_generic_analysis(self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generic aggregation for other analysis types"""
        all_keys = set()
        aggregated_data = {}
        
        # Collect all unique keys from all analyses
        for chunk_analysis in chunk_analyses:
            analysis = chunk_analysis.get('analysis', {})
            all_keys.update(analysis.keys())
        
        # Aggregate values for each key
        for key in all_keys:
            values = []
            for chunk_analysis in chunk_analyses:
                analysis = chunk_analysis.get('analysis', {})
                if key in analysis:
                    value = analysis[key]
                    if isinstance(value, list):
                        values.extend(value)
                    else:
                        values.append(value)
            
            # Determine aggregated value
            if values:
                if all(isinstance(v, str) for v in values):
                    aggregated_data[key] = list(set(values))
                elif all(isinstance(v, (int, float)) for v in values):
                    aggregated_data[key] = sum(values) / len(values)
                else:
                    aggregated_data[key] = values[0]  # Take first value as default
        
        return aggregated_data
    
    def _aggregate_mq_analysis(self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate MQ-specific analyses"""
        flow_patterns = []
        queue_operations = []
        business_purposes = []
        
        for chunk_analysis in chunk_analyses:
            analysis = chunk_analysis.get('analysis', {})
            
            if 'flow_type' in analysis:
                flow_patterns.append(analysis['flow_type'])
            if 'business_purpose' in analysis:
                business_purposes.append(analysis['business_purpose'])
            if 'queue_operations' in analysis:
                if isinstance(analysis['queue_operations'], list):
                    queue_operations.extend(analysis['queue_operations'])
        
        return {
            'message_flow_patterns': list(set(flow_patterns)),
            'queue_operations': list(set(queue_operations)),
            'consolidated_business_purpose': max(set(business_purposes), key=business_purposes.count) if business_purposes else 'message_processing',
            'mq_integration_level': 'comprehensive' if len(set(queue_operations)) > 3 else 'standard'
        }

    def _aggregate_cics_analysis(self, chunk_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate CICS-specific analyses"""
        transaction_purposes = []
        resource_types = []
        business_functions = []
        
        for chunk_analysis in chunk_analyses:
            analysis = chunk_analysis.get('analysis', {})
            
            if 'transaction_purpose' in analysis:
                transaction_purposes.append(analysis['transaction_purpose'])
            if 'resource_management' in analysis:
                resource_types.append(analysis['resource_management'])
            if 'business_function' in analysis:
                business_functions.append(analysis['business_function'])
        
        return {
            'transaction_patterns': list(set(transaction_purposes)),
            'resource_management_types': list(set(resource_types)),
            'business_functions': list(set(business_functions)),
            'cics_complexity': 'high' if len(set(transaction_purposes)) > 3 else 'medium'
        }

    def _parse_structured_text_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse structured text response when JSON parsing fails"""
        result = {}
        
        # Common patterns based on analysis type
        if analysis_type in ['cobol_section', 'cobol_paragraph']:
            # Look for purpose/functionality patterns
            if 'purpose' in response.lower():
                purpose_match = re.search(r'purpose[:\s]+([^\n.]+)', response, re.IGNORECASE)
                if purpose_match:
                    result['purpose'] = purpose_match.group(1).strip()
            
            if 'function' in response.lower():
                func_match = re.search(r'function[:\s]+([^\n.]+)', response, re.IGNORECASE)
                if func_match:
                    result['functionality'] = func_match.group(1).strip()
            
            # Look for complexity indicators
            complexity_keywords = ['simple', 'basic', 'low', 'medium', 'high', 'complex']
            for keyword in complexity_keywords:
                if keyword in response.lower():
                    result['complexity'] = keyword
                    break
        
        elif analysis_type in ['cobol_sql_block', 'sql_whenever_statement']:
            # SQL-specific patterns
            if 'select' in response.lower():
                result['operation_type'] = 'select'
            elif 'insert' in response.lower():
                result['operation_type'] = 'insert'
            elif 'update' in response.lower():
                result['operation_type'] = 'update'
            elif 'delete' in response.lower():
                result['operation_type'] = 'delete'
            else:
                result['operation_type'] = 'unknown'
            
            # Look for table names (simple pattern)
            table_pattern = re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', response)
            if table_pattern:
                result['tables_accessed'] = table_pattern[:3]  # Max 3 tables
        
        elif analysis_type.startswith('cics_'):
            # CICS-specific patterns
            if 'send' in response.lower():
                result['transaction_purpose'] = 'screen_output'
            elif 'receive' in response.lower():
                result['transaction_purpose'] = 'screen_input'
            elif 'read' in response.lower() or 'write' in response.lower():
                result['transaction_purpose'] = 'file_operation'
            else:
                result['transaction_purpose'] = 'general_transaction'
        
        elif analysis_type in ['mq_sequence', 'mq_message_flow']:
            # MQ-specific patterns
            if 'put' in response.lower():
                result['business_purpose'] = 'message_sending'
            elif 'get' in response.lower():
                result['business_purpose'] = 'message_receiving'
            else:
                result['business_purpose'] = 'message_processing'
        
        # If no specific patterns found, create generic result
        if not result:
            result = {
                'analysis_type': analysis_type,
                'parsed_from': 'structured_text',
                'confidence': 'medium'
            }
        
        return result

    def _create_fallback_analysis(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Create fallback analysis when all parsing fails"""
    
        # Basic fallback based on analysis type
        fallbacks = {
            'cobol_section': {
                'purpose': 'cobol_section_processing',
                'functionality': 'program_logic',
                'complexity': 'medium',
                'business_impact': 'standard'
            },
            'cobol_paragraph': {
                'business_function': 'data_processing',
                'operations': ['processing'],
                'error_handling': False,
                'complexity_level': 'medium'
            },
            'cobol_sql_block': {
                'operation_type': 'database_operation',
                'complexity': 'medium',
                'host_variables': [],
                'tables_accessed': [],
                'business_purpose': 'data_access'
            },
            'cics_send_map': {
                'transaction_purpose': 'user_interface',
                'resource_management': 'screen_handling',
                'business_function': 'user_interaction'
            },
            'cobol_copy_statement': {
                'copybook_purpose': 'data_structure',
                'domain': 'business_data',
                'reuse_pattern': 'standard_include'
            },
            'mq_message_flow': {
                'business_purpose': 'message_processing',
                'flow_characteristics': 'standard_messaging',
                'business_alignment': 'integration'
            }
        }
        
        # Get specific fallback or create generic one
        if analysis_type in fallbacks:
            result = fallbacks[analysis_type].copy()
        else:
            result = {
                'analysis_type': analysis_type,
                'status': 'fallback_analysis',
                'confidence': 'low',
                'note': 'Generated from fallback due to parsing issues'
            }
        
        # Add response snippet if available
        if response and len(response) > 10:
            result['raw_response_snippet'] = response[:100]
        
        return result

    def _safe_truncate_content(self, content: str, max_chars: int = 4000) -> str:
        """Safely truncate content to prevent oversized prompts"""
        if not content:
            return ""
        
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at logical boundaries
        lines = content.split('\n')
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) + 1 > max_chars:
                break
            truncated_lines.append(line)
            current_length += len(line) + 1
        
        truncated_content = '\n'.join(truncated_lines)
        
        # Add truncation indicator
        if len(content) > max_chars:
            truncated_content += "\n\n[... CONTENT TRUNCATED FOR ANALYSIS ...]"
        
        return truncated_content


    async def _llm_analyze_complex_pattern(self, content: str, pattern_type: str) -> Dict[str, Any]:
        """Use LLM to analyze complex patterns - FIXED with size limits"""
        
        # SAFETY: Truncate content to prevent oversized prompts
        safe_content = self._safe_truncate_content(content, max_chars=3000)
        
        prompts = {
            'control_flow': """
            Analyze the control flow in this code sample:
            
            {content}
            
            Identify:
            1. Entry points and exit points
            2. Decision points and conditions
            3. Loop structures and iterations
            4. Error handling paths
            5. Business logic complexity
            
            Return as JSON:
            {{
                "entry_points": ["point1", "point2"],
                "exit_points": ["exit1", "exit2"],
                "decision_points": [
                    {{"condition": "condition1", "complexity": "high"}}
                ],
                "loops": [
                    {{"type": "while", "condition": "loop_condition"}}
                ],
                "error_handling": [
                    {{"type": "exception", "handler": "error_handler"}}
                ],
                "complexity_score": 7,
                "maintainability": "medium"
            }}
            """,
            
            'data_relationships': """
            Analyze the data relationships in this code sample:
            
            {content}
            
            Identify:
            1. Data dependencies and transformations
            2. Input/output data flows
            3. Data validation patterns
            4. Business rules affecting data
            5. Performance implications
            
            Return as JSON:
            {{
                "data_dependencies": [
                    {{"source": "field1", "target": "field2", "transformation": "copy"}}
                ],
                "io_flows": [
                    {{"direction": "input", "source": "file1", "fields": ["field1", "field2"]}}
                ],
                "validations": [
                    {{"field": "field1", "rule": "required", "business_impact": "high"}}
                ],
                "business_rules": [
                    {{"rule": "rule1", "fields": ["field1"], "logic": "business logic"}}
                ],
                "performance_impact": "medium"
            }}
            """,
            
            'business_logic': """
            Analyze the business logic in this code sample:
            
            {content}
            
            Identify:
            1. Business functions and purposes
            2. Business rules implementation
            3. Domain entities and relationships
            4. Processing patterns
            5. Integration points
            
            Return as JSON:
            {{
                "business_functions": [
                    {{"name": "function1", "purpose": "customer processing", "criticality": "high"}}
                ],
                "business_rules": [
                    {{"rule": "validation_rule", "implementation": "code_pattern", "domain": "finance"}}
                ],
                "domain_entities": [
                    {{"entity": "customer", "attributes": ["name", "id"], "operations": ["read", "update"]}}
                ],
                "processing_patterns": [
                    {{"pattern": "batch_processing", "frequency": "daily", "volume": "high"}}
                ],
                "integration_points": [
                    {{"type": "database", "system": "db2", "operations": ["read", "write"]}}
                ]
            }}
            """
        }
        
        if pattern_type not in prompts:
            return {'error': f'Unknown pattern type: {pattern_type}'}
        
        try:
            # Use the regular cached analysis with safe content
            result = await self._analyze_with_llm_cached(
                safe_content, pattern_type, prompts[pattern_type]
            )
            
            # Add truncation metadata
            if len(content) > 3000:
                if 'analysis' in result and isinstance(result['analysis'], dict):
                    result['analysis']['content_truncated'] = True
                    result['analysis']['original_size'] = len(content)
                    result['analysis']['analyzed_size'] = len(safe_content)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Complex pattern analysis failed: {e}")
            return {
                'analysis': {'error': str(e), 'pattern_type': pattern_type},
                'confidence_score': 0.3
            }

    
    # ==================== PUBLIC API METHODS ====================

    def cleanup(self):
        """Enhanced cleanup method for coordinator integration"""
        self.logger.info("🧹 Cleaning up Enhanced Code Parser Agent resources...")
        
        # Clear any cached data
        self._processed_files.clear()
        
        # Call parent cleanup
        super().cleanup()
        
        self.logger.info("✅ Enhanced Code Parser Agent cleanup completed")

    def get_version_info(self) -> Dict[str, str]:
        """Get enhanced version and capability information for air-gapped deployment"""
        return {
            "agent_name": "CodeParserAgent",
            "version": "3.0.0-LLM-Enhanced-AirGapped",
            "base_agent": "BaseOpulenceAgent",
            "deployment_mode": "AIR_GAPPED",
            "api_compatible": True,
            "coordinator_integration": "API-based with LLM analysis",
            "external_dependencies": "NONE - Fully air-gapped",
            "tokenizer": "Built-in character estimation (no tiktoken)",
            "enhanced_capabilities": [
                "LLM-powered complex pattern analysis",
                "Air-gapped intelligent token estimation",
                "Comprehensive copybook layout parsing",
                "Multi-record and conditional layout support",
                "Advanced REDEFINES and OCCURS handling",
                "REPLACING parameter analysis",
                "Multi-filler pattern recognition",
                "IBM MQ message flow analysis",
                "DB2 stored procedure comprehensive parsing",
                "COBOL stored procedure integration",
                "Enhanced business rule validation",
                "Confidence scoring and quality metrics",
                "Cached LLM analysis for performance",
                "Zero external network dependencies"
            ],
            "supported_file_types": [".cbl", ".cob", ".jcl", ".cpy", ".copy", ".bms", ".sql", ".db2", ".mqt"],
            "supported_technologies": [
                "IBM COBOL (all dialects)",
                "IBM JCL", 
                "IBM CICS",
                "IBM BMS",
                "IBM DB2",
                "IBM MQ/WebSphere MQ",
                "SQL Stored Procedures",
                "COBOL Stored Procedures",
                "Complex Copybook Structures",
                "Enterprise Middleware Integration"
            ],
            "llm_integration": {
                "analysis_types": ["control_flow", "data_relationships", "business_logic"],
                "caching_enabled": True,
                "confidence_scoring": True,
                "fallback_patterns": True,
                "tokenizer": "built_in_estimation"
            },
            "database_schema_version": "3.0",
            "business_rules_enabled": True,
            "enterprise_ready": True,
            "performance_optimized": True,
            "security_level": "AIR_GAPPED_SECURE"
        }

# Export the enhanced class
__all__ = ['CodeParserAgent']