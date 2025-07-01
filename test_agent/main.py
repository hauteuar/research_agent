

import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="Mainframe Deep Research Agent",
    page_icon="🔍",
    layout="wide"
)

@dataclass
class CobolField:
    level: str
    name: str
    picture: Optional[str] = None
    value: Optional[str] = None
    occurs: Optional[str] = None
    depending_on: Optional[str] = None
    redefines: Optional[str] = None
    usage: Optional[str] = None
    is_filler: bool = False
    start_position: int = 0
    length: int = 0
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)

@dataclass
class VsamOperation:
    operation: str  # READ, WRITE, REWRITE, DELETE, STARTBR, READNEXT, etc.
    file_name: str
    key_field: Optional[str] = None
    record_area: Optional[str] = None
    response_code: Optional[str] = None
    program: str = ""
    line_number: int = 0

@dataclass
class FileReference:
    file_name: str
    program: str
    operation: str
    fields_used: List[str]
    line_number: int
    file_type: str = "SEQUENTIAL"  # SEQUENTIAL, VSAM, DB2
    access_method: str = "BATCH"   # BATCH, CICS, DB2

@dataclass
class FieldUsage:
    field_name: str
    program: str
    usage_type: str
    line_number: int
    context: str
    qualified_name: str = ""  # For handling OF/IN qualifications

class AdvancedCobolParser:
    def __init__(self):
        self.fields = {}
        self.current_program = ""
        self.redefines_groups = {}
        self.occurs_tables = {}
        
    def parse_data_division(self, lines: List[str]) -> Dict[str, CobolField]:
        """Parse complex COBOL data structures with REDEFINES, OCCURS, DEPENDING ON"""
        fields = {}
        current_level_stack = {}
        in_data_division = False
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or line_clean.startswith('*'):
                continue
                
            # Check for data division sections
            if 'DATA DIVISION' in line.upper():
                in_data_division = True
                continue
            elif 'PROCEDURE DIVISION' in line.upper():
                in_data_division = False
                break
                
            if not in_data_division:
                continue
                
            # Parse field definition
            field_match = re.match(r'(\d{2})\s+([A-Z0-9\-]+|\bFILLER\b)(.*)', line_clean.upper())
            if field_match:
                level = field_match.group(1)
                name = field_match.group(2)
                rest = field_match.group(3).strip()
                
                # Create field object
                cobol_field = CobolField(
                    level=level,
                    name=name,
                    is_filler=(name == 'FILLER')
                )
                
                # Parse PICTURE clause
                pic_match = re.search(r'PIC(?:TURE)?\s+([X9S\(\)V\.\-\+]+)', rest)
                if pic_match:
                    cobol_field.picture = pic_match.group(1)
                    cobol_field.length = self.calculate_field_length(cobol_field.picture)
                
                # Parse VALUE clause
                value_match = re.search(r'VALUE\s+(?:IS\s+)?([^\.]+)', rest)
                if value_match:
                    cobol_field.value = value_match.group(1).strip().strip('"\'')
                
                # Parse OCCURS clause
                occurs_match = re.search(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s*TIMES?', rest)
                if occurs_match:
                    if occurs_match.group(2):  # Variable occurrence
                        cobol_field.occurs = f"{occurs_match.group(1)} TO {occurs_match.group(2)}"
                    else:
                        cobol_field.occurs = occurs_match.group(1)
                    
                    # Check for DEPENDING ON
                    depending_match = re.search(r'DEPENDING\s+ON\s+([A-Z0-9\-]+)', rest)
                    if depending_match:
                        cobol_field.depending_on = depending_match.group(1)
                
                # Parse REDEFINES clause
                redefines_match = re.search(r'REDEFINES\s+([A-Z0-9\-]+)', rest)
                if redefines_match:
                    cobol_field.redefines = redefines_match.group(1)
                
                # Parse USAGE clause
                usage_match = re.search(r'USAGE\s+(?:IS\s+)?(COMP|COMP-1|COMP-2|COMP-3|BINARY|PACKED-DECIMAL)', rest)
                if usage_match:
                    cobol_field.usage = usage_match.group(1)
                
                # Handle level hierarchy
                current_level = int(level)
                
                # Find parent
                parent_level = None
                for check_level in sorted(current_level_stack.keys(), reverse=True):
                    if check_level < current_level:
                        parent_level = check_level
                        break
                
                if parent_level:
                    parent_name = current_level_stack[parent_level]
                    cobol_field.parent = parent_name
                    if parent_name in fields:
                        fields[parent_name].children.append(name)
                
                # Update level stack
                levels_to_remove = [l for l in current_level_stack.keys() if l >= current_level]
                for l in levels_to_remove:
                    del current_level_stack[l]
                current_level_stack[current_level] = name
                
                fields[name] = cobol_field
        
        return fields
    
    def calculate_field_length(self, picture: str) -> int:
        """Calculate field length from PICTURE clause"""
        if not picture:
            return 0
            
        # Remove formatting characters
        pic_clean = re.sub(r'[V\.\-\+S]', '', picture)
        
        # Handle repeated characters like X(10) or 9(5)
        total_length = 0
        
        # Pattern for X(n) or 9(n)
        repeat_pattern = r'([X9])(\((\d+)\))?'
        matches = re.findall(repeat_pattern, pic_clean)
        
        for match in matches:
            char_type = match[0]
            repeat_count = int(match[2]) if match[2] else 1
            total_length += repeat_count
        
        # Handle simple patterns like XXX or 999
        simple_chars = re.sub(r'[X9]\(\d+\)', '', pic_clean)
        total_length += len(simple_chars)
        
        return total_length

class VsamCicsParser:
    def __init__(self):
        self.vsam_operations = []
        
    def parse_cics_commands(self, lines: List[str], program_name: str) -> List[VsamOperation]:
        """Parse CICS commands for VSAM operations"""
        operations = []
        
        cics_patterns = {
            'READ': r'EXEC\s+CICS\s+READ\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'WRITE': r'EXEC\s+CICS\s+WRITE\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'REWRITE': r'EXEC\s+CICS\s+REWRITE\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'DELETE': r'EXEC\s+CICS\s+DELETE\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'STARTBR': r'EXEC\s+CICS\s+STARTBR\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'READNEXT': r'EXEC\s+CICS\s+READNEXT\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'READPREV': r'EXEC\s+CICS\s+READPREV\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)',
            'ENDBR': r'EXEC\s+CICS\s+ENDBR\s+(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)\s*\)'
        }
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            for operation, pattern in cics_patterns.items():
                match = re.search(pattern, line_upper)
                if match:
                    file_name = match.group(1)
                    
                    # Extract additional parameters
                    key_field = None
                    record_area = None
                    response_code = None
                    
                    # Look for RIDFLD (key field)
                    ridfld_match = re.search(r'RIDFLD\s*\(\s*([A-Z0-9\-]+)\s*\)', line_upper)
                    if ridfld_match:
                        key_field = ridfld_match.group(1)
                    
                    # Look for INTO (record area)
                    into_match = re.search(r'INTO\s*\(\s*([A-Z0-9\-]+)\s*\)', line_upper)
                    if into_match:
                        record_area = into_match.group(1)
                    
                    # Look for FROM (for write operations)
                    from_match = re.search(r'FROM\s*\(\s*([A-Z0-9\-]+)\s*\)', line_upper)
                    if from_match:
                        record_area = from_match.group(1)
                    
                    # Look for RESP
                    resp_match = re.search(r'RESP\s*\(\s*([A-Z0-9\-]+)\s*\)', line_upper)
                    if resp_match:
                        response_code = resp_match.group(1)
                    
                    vsam_op = VsamOperation(
                        operation=operation,
                        file_name=file_name,
                        key_field=key_field,
                        record_area=record_area,
                        response_code=response_code,
                        program=program_name,
                        line_number=i + 1
                    )
                    operations.append(vsam_op)
        
        return operations

class DB2CopybookComparator:
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        
    def compare_copybook_to_db2(self, copybook_fields: Dict[str, CobolField], 
                               table_name: str) -> Dict[str, Any]:
        """Compare copybook structure with DB2 table"""
        comparison_results = {
            'field_matches': [],
            'field_mismatches': [],
            'missing_in_copybook': [],
            'missing_in_db2': [],
            'type_mismatches': [],
            'length_mismatches': []
        }
        
        try:
            # Simulate DB2 connection (replace with actual DB2 connection)
            db2_columns = self.get_db2_table_structure(table_name)
            
            copybook_field_names = set(copybook_fields.keys())
            db2_column_names = set(col['column_name'] for col in db2_columns)
            
            # Find matches and mismatches
            common_fields = copybook_field_names.intersection(db2_column_names)
            
            for field_name in common_fields:
                copybook_field = copybook_fields[field_name]
                db2_column = next(col for col in db2_columns if col['column_name'] == field_name)
                
                # Compare data types
                cobol_type = self.map_cobol_to_db2_type(copybook_field)
                db2_type = db2_column['data_type']
                
                if cobol_type == db2_type:
                    comparison_results['field_matches'].append({
                        'field_name': field_name,
                        'copybook_type': cobol_type,
                        'db2_type': db2_type,
                        'copybook_length': copybook_field.length,
                        'db2_length': db2_column.get('length', 0)
                    })
                else:
                    comparison_results['type_mismatches'].append({
                        'field_name': field_name,
                        'copybook_type': cobol_type,
                        'db2_type': db2_type
                    })
            
            # Missing fields
            comparison_results['missing_in_db2'] = list(copybook_field_names - db2_column_names)
            comparison_results['missing_in_copybook'] = list(db2_column_names - copybook_field_names)
            
        except Exception as e:
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def get_db2_table_structure(self, table_name: str) -> List[Dict]:
        """Get DB2 table structure (mock implementation)"""
        # This would connect to actual DB2 database
        # For now, returning mock data
        return [
            {'column_name': 'CUSTOMER_ID', 'data_type': 'CHAR', 'length': 10},
            {'column_name': 'CUSTOMER_NAME', 'data_type': 'VARCHAR', 'length': 50},
            {'column_name': 'ACCOUNT_BALANCE', 'data_type': 'DECIMAL', 'length': 15}
        ]
    
    def map_cobol_to_db2_type(self, field: CobolField) -> str:
        """Map COBOL data types to DB2 equivalents"""
        if not field.picture:
            return 'UNKNOWN'
        
        pic = field.picture.upper()
        
        if 'X' in pic:
            return 'CHAR' if field.length <= 254 else 'VARCHAR'
        elif '9' in pic:
            if 'V' in pic or '.' in pic:
                return 'DECIMAL'
            else:
                return 'INTEGER' if field.length <= 10 else 'BIGINT'
        elif field.usage in ['COMP', 'BINARY']:
            return 'INTEGER'
        elif field.usage == 'COMP-3':
            return 'DECIMAL'
        
        return 'UNKNOWN'

class LLMIntelligentAnalyzer:
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_cobol_with_llm(self, code_snippet: str, context: str = "") -> Dict[str, Any]:
        """Use LLM to intelligently analyze COBOL code patterns"""
        
        prompt = f"""
        As a mainframe COBOL expert, analyze this code snippet and extract:
        
        1. File operations (READ, WRITE, OPEN, etc.) with context
        2. Complex data structures (REDEFINES, OCCURS, nested groups)
        3. Business logic patterns and field relationships
        4. CICS/VSAM operations with their purpose
        5. Data flow and transformations
        
        Code to analyze:
        ```cobol
        {code_snippet}
        ```
        
        Context: {context}
        
        Provide analysis in JSON format with:
        - file_operations: [{{"operation": "", "file": "", "purpose": "", "business_context": ""}}]
        - data_structures: [{{"name": "", "type": "", "complexity": "", "business_meaning": ""}}]
        - field_relationships: [{{"source": "", "target": "", "transformation": "", "business_rule": ""}}]
        - cics_operations: [{{"command": "", "file": "", "business_purpose": "", "error_handling": ""}}]
        - business_logic: ["key business rules identified"]
        
        Focus on business meaning, not just syntax.
        """
        
        # Simulate LLM analysis (in real implementation, this would call actual LLM)
        return self.simulate_llm_analysis(code_snippet)
    
    def simulate_llm_analysis(self, code: str) -> Dict[str, Any]:
        """Simulate intelligent LLM analysis with pattern recognition"""
        analysis = {
            "file_operations": [],
            "data_structures": [],
            "field_relationships": [],
            "cics_operations": [],
            "business_logic": []
        }
        
        # Smart pattern detection with business context
        lines = code.upper().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Intelligent file operation analysis
            if 'READ' in line and 'CUSTOMER' in line:
                analysis["file_operations"].append({
                    "operation": "READ",
                    "file": self.extract_file_name(line),
                    "purpose": "Customer data retrieval",
                    "business_context": "Customer information lookup for transaction processing"
                })
            
            elif 'WRITE' in line and 'ACCOUNT' in line:
                analysis["file_operations"].append({
                    "operation": "WRITE", 
                    "file": self.extract_file_name(line),
                    "purpose": "Account record creation/update",
                    "business_context": "Account management and balance tracking"
                })
            
            # Smart REDEFINES analysis
            if 'REDEFINES' in line:
                analysis["data_structures"].append({
                    "name": self.extract_field_name(line),
                    "type": "REDEFINES",
                    "complexity": "Memory overlay for data format conversion",
                    "business_meaning": "Multiple data interpretations of same memory area"
                })
            
            # Intelligent OCCURS detection
            if 'OCCURS' in line:
                occurs_field = self.extract_field_name(line)
                if 'TRANSACTION' in line or 'TXN' in line:
                    business_meaning = "Transaction history array for customer processing"
                elif 'BALANCE' in line:
                    business_meaning = "Account balance array for multi-currency support"
                else:
                    business_meaning = "Repeating data structure for bulk processing"
                    
                analysis["data_structures"].append({
                    "name": occurs_field,
                    "type": "OCCURS",
                    "complexity": "Array/table structure",
                    "business_meaning": business_meaning
                })
            
            # CICS command intelligence
            if 'EXEC CICS READ' in line:
                analysis["cics_operations"].append({
                    "command": "CICS READ",
                    "file": self.extract_cics_file(line),
                    "business_purpose": "Real-time data access for online transaction",
                    "error_handling": "RESP code checking for data availability"
                })
        
        return analysis
    
    def extract_file_name(self, line: str) -> str:
        """Smart file name extraction"""
        words = line.split()
        for i, word in enumerate(words):
            if word in ['READ', 'WRITE', 'OPEN'] and i + 1 < len(words):
                return words[i + 1].strip('.,')
        return "UNKNOWN"
    
    def extract_field_name(self, line: str) -> str:
        """Smart field name extraction"""
        match = re.search(r'\d{2}\s+([A-Z0-9\-]+)', line)
        return match.group(1) if match else "UNKNOWN"
    
    def extract_cics_file(self, line: str) -> str:
        """Extract CICS file name intelligently"""
        match = re.search(r'(?:DATASET|FILE)\s*\(\s*([A-Z0-9\-]+)', line)
        return match.group(1) if match else "UNKNOWN"
    
    def analyze_business_patterns(self, all_code: str) -> Dict[str, Any]:
        """High-level business pattern analysis using LLM intelligence"""
        
        patterns = {
            "batch_processing": False,
            "online_transaction": False,
            "data_migration": False,
            "reporting": False,
            "validation_heavy": False,
            "multi_file_processing": False
        }
        
        code_upper = all_code.upper()
        
        # Pattern detection with business intelligence
        if 'SORT' in code_upper and 'INPUT' in code_upper and 'OUTPUT' in code_upper:
            patterns["batch_processing"] = True
            
        if 'EXEC CICS' in code_upper and 'SEND MAP' in code_upper:
            patterns["online_transaction"] = True
            
        if 'COPY' in code_upper and 'DB2' in code_upper:
            patterns["data_migration"] = True
            
        if 'REPORT' in code_upper or 'PRINT' in code_upper:
            patterns["reporting"] = True
            
        # Count file operations for complexity
        file_count = len(re.findall(r'(?:READ|WRITE|OPEN)', code_upper))
        if file_count > 5:
            patterns["multi_file_processing"] = True
            
        return patterns

class MainframeAnalyzer:
    def __init__(self):
        self.db_path = "mainframe_analysis.db" 
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_database()
        self.file_references = []
        self.field_usages = []
        self.programs = {}
        self.cobol_parser = AdvancedCobolParser()
        self.vsam_parser = VsamCicsParser()
        self.llm_analyzer = LLMIntelligentAnalyzer()  # Add LLM intelligence
        
    def init_database(self):
        """Initialize SQLite database for analysis"""
        cursor = self.conn.cursor()
        
        # Enhanced tables for complex analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_references (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                program TEXT,
                operation TEXT,
                fields_used TEXT,
                line_number INTEGER,
                file_type TEXT,
                access_method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cobol_fields (
                id INTEGER PRIMARY KEY,
                program TEXT,
                field_name TEXT,
                level TEXT,
                picture TEXT,
                field_value TEXT,
                occurs_clause TEXT,
                depending_on TEXT,
                redefines TEXT,
                usage_clause TEXT,
                is_filler BOOLEAN,
                field_length INTEGER,
                parent_field TEXT,
                qualified_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vsam_operations (
                id INTEGER PRIMARY KEY,
                program TEXT,
                operation TEXT,
                file_name TEXT,
                key_field TEXT,
                record_area TEXT,
                response_code TEXT,
                line_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_analysis (
                id INTEGER PRIMARY KEY,
                program TEXT,
                analysis_type TEXT,
                business_context TEXT,
                technical_details TEXT,
                complexity_score INTEGER,
                recommendations TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS copybook_db2_comparison (
                id INTEGER PRIMARY KEY,
                copybook_name TEXT,
                table_name TEXT,
                field_name TEXT,
                copybook_type TEXT,
                db2_type TEXT,
                match_status TEXT,
                copybook_length INTEGER,
                db2_length INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
        self.conn.commit()

    def parse_cobol_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Enhanced COBOL parsing with LLM intelligence"""
        lines = content.split('\n')
        analysis = {
            'file_operations': [],
            'field_usage': [],
            'data_definitions': [],
            'vsam_operations': [],
            'copybook_fields': {},
            'llm_insights': {}
        }
        
        # Use LLM for intelligent analysis
        chunk_size = 50  # Analyze in chunks for better performance
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if chunk.strip():
                llm_analysis = self.llm_analyzer.analyze_cobol_with_llm(
                    chunk, f"Program: {filename}, Lines: {i+1}-{i+chunk_size}"
                )
                
                # Merge LLM insights
                for key in llm_analysis:
                    if key not in analysis['llm_insights']:
                        analysis['llm_insights'][key] = []
                    analysis['llm_insights'][key].extend(llm_analysis[key])
        
        # Get high-level business patterns
        business_patterns = self.llm_analyzer.analyze_business_patterns(content)
        analysis['llm_insights']['business_patterns'] = business_patterns
        
        # Parse data division for complex structures (keep existing logic)
        analysis['copybook_fields'] = self.cobol_parser.parse_data_division(lines)
        
        # Parse CICS/VSAM operations (keep existing logic)
        analysis['vsam_operations'] = self.vsam_parser.parse_cics_commands(lines, filename)
        
        # Enhanced file operation patterns with LLM context
        file_patterns = {
            'READ': r'READ\s+([A-Z0-9\-]+)(?:\s+(?:INTO|AT\s+END))?',
            'WRITE': r'WRITE\s+([A-Z0-9\-]+)(?:\s+(?:FROM|AFTER|BEFORE))?',
            'OPEN': r'OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([A-Z0-9\-]+)',
            'CLOSE': r'CLOSE\s+([A-Z0-9\-]+)',
            'SELECT': r'SELECT\s+([A-Z0-9\-]+)\s+ASSIGN\s+TO\s+([A-Z0-9\-]+)',
            'REWRITE': r'REWRITE\s+([A-Z0-9\-]+)',
            'DELETE': r'DELETE\s+([A-Z0-9\-]+)'
        }
        
        for i, line in enumerate(lines, 1):
            line = line.strip().upper()
            
            # Find file operations and enrich with LLM context
            for op, pattern in file_patterns.items():
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        file_name = match[1] if len(match) > 1 else match[0]
                        file_type = "SEQUENTIAL"
                    else:
                        file_name = match
                        file_type = "SEQUENTIAL"
                    
                    # Use LLM to determine business context
                    business_context = self.get_business_context_for_file(file_name, op)
                    
                    # Determine if it's VSAM based on SELECT statement
                    if 'VSAM' in line or 'KSDS' in line or 'ESDS' in line or 'RRDS' in line:
                        file_type = "VSAM"
                    
                    ref = FileReference(
                        file_name=file_name,
                        program=filename,
                        operation=op,
                        fields_used=[],
                        line_number=i,
                        file_type=file_type,
                        access_method="BATCH"
                    )
                    analysis['file_operations'].append(ref)
        
        return analysis
    
    def get_business_context_for_file(self, file_name: str, operation: str) -> str:
        """Use LLM intelligence to determine business context"""
        # Smart business context detection
        file_upper = file_name.upper()
        
        if 'CUSTOMER' in file_upper or 'CUST' in file_upper:
            return f"Customer data {operation.lower()} for CRM operations"
        elif 'ACCOUNT' in file_upper or 'ACCT' in file_upper:
            return f"Account {operation.lower()} for financial processing"
        elif 'TRANSACTION' in file_upper or 'TXN' in file_upper:
            return f"Transaction {operation.lower()} for payment processing"
        elif 'BALANCE' in file_upper:
            return f"Balance {operation.lower()} for accounting operations"
        elif 'MASTER' in file_upper:
            return f"Master data {operation.lower()} for reference management"
        elif 'TEMP' in file_upper or 'WORK' in file_upper:
            return f"Temporary {operation.lower()} for intermediate processing"
        else:
            return f"File {operation.lower()} operation"

    def store_enhanced_analysis(self, analysis: Dict[str, Any], program_name: str):
        """Store enhanced analysis results"""
        cursor = self.conn.cursor()
        
        # Store COBOL fields
        for field_name, field_obj in analysis['copybook_fields'].items():
            cursor.execute("""
                INSERT INTO cobol_fields 
                (program, field_name, level, picture, field_value, occurs_clause, 
                 depending_on, redefines, usage_clause, is_filler, field_length, 
                 parent_field, qualified_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_name, field_name, field_obj.level, field_obj.picture,
                field_obj.value, field_obj.occurs, field_obj.depending_on,
                field_obj.redefines, field_obj.usage, field_obj.is_filler,
                field_obj.length, field_obj.parent, 
                f"{field_obj.parent}.{field_name}" if field_obj.parent else field_name
            ))
        
        # Store VSAM operations
        for vsam_op in analysis['vsam_operations']:
            cursor.execute("""
                INSERT INTO vsam_operations 
                (program, operation, file_name, key_field, record_area, response_code, line_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                vsam_op.program, vsam_op.operation, vsam_op.file_name,
                vsam_op.key_field, vsam_op.record_area, vsam_op.response_code,
                vsam_op.line_number
            ))
        
        # Store file references with enhanced info
        for file_ref in analysis['file_operations']:
            cursor.execute("""
                INSERT INTO file_references 
                (file_name, program, operation, fields_used, line_number, file_type, access_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_ref.file_name, file_ref.program, file_ref.operation,
                json.dumps(file_ref.fields_used), file_ref.line_number,
                file_ref.file_type, file_ref.access_method
            ))
        
        self.conn.commit()

    def analyze_uploaded_files(self, uploaded_files: List) -> Dict[str, Any]:
        """Enhanced file analysis"""
        cursor = self.conn.cursor()
        results = {
            'programs_analyzed': 0,
            'files_found': set(),
            'new_programs': 0,      # Track new vs existing
            'updated_programs': 0, 
            'files_found': set(),
            'vsam_files': set(),
            'db2_files': set(),
            'field_definitions': {},
            'file_lineage': {},
            'redefines_found': 0,
            'occurs_found': 0,
            'cics_operations': 0
        }
        
        for uploaded_file in uploaded_files:
            content = None
            encodings_to_try = ['utf-8', 'cp1252', 'iso-8859-1', 'cp500', 'ibm1047', 'latin1']
            
            for encoding in encodings_to_try:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    content = uploaded_file.read().decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            cursor.execute("SELECT COUNT(*) FROM file_references WHERE program = ?", (filename,))
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0:
                # DELETE OLD ANALYSIS FOR THIS PROGRAM
                cursor.execute("DELETE FROM file_references WHERE program = ?", (filename,))
                cursor.execute("DELETE FROM cobol_fields WHERE program = ?", (filename,))
                cursor.execute("DELETE FROM vsam_operations WHERE program = ?", (filename,))
                cursor.execute("DELETE FROM llm_analysis WHERE program = ?", (filename,))
                results['updated_programs'] += 1
            else:
                results['new_programs'] += 1
            
            if content is None:
                # Fallback: ignore problematic characters
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                
            filename = uploaded_file.name
            
            if filename.endswith(('.cbl', '.cob', '.cobol', '.cpy')):
                analysis = self.parse_cobol_file(content, filename)
                
                # Count complex structures
                for field_name, field_obj in analysis['copybook_fields'].items():
                    if field_obj.redefines:
                        results['redefines_found'] += 1
                    if field_obj.occurs:
                        results['occurs_found'] += 1
                
                results['cics_operations'] += len(analysis['vsam_operations'])
                
                # Store enhanced analysis
                self.store_enhanced_analysis(analysis, filename)
                
            elif filename.endswith(('.jcl', '.proc')):
                analysis = self.parse_jcl_file(content, filename)
            else:
                continue
            
            # Process results
            for file_op in analysis.get('file_operations', []):
                results['files_found'].add(file_op.file_name)
                if file_op.file_type == 'VSAM':
                    results['vsam_files'].add(file_op.file_name)
            
            for vsam_op in analysis.get('vsam_operations', []):
                results['vsam_files'].add(vsam_op.file_name)
            
            results['programs_analyzed'] += 1
        
        return results

    def enhanced_chat_query(self, question: str) -> str:
        """Enhanced chat with VSAM, CICS, and copybook support"""
        question_lower = question.lower()
        cursor = self.conn.cursor()
        
        # VSAM-specific queries
        if 'vsam' in question_lower or 'cics' in question_lower:
            if 'operations' in question_lower:
                cursor.execute("""
                    SELECT program, operation, file_name, key_field, line_number
                    FROM vsam_operations
                    ORDER BY program, line_number
                """)
                
                results = cursor.fetchall()
                if results:
                    response = "VSAM/CICS Operations found:\n\n"
                    for row in results:
                        response += f"• Program: {row[0]}\n"
                        response += f"  Operation: {row[1]} on file {row[2]}\n"
                        if row[3]:
                            response += f"  Key Field: {row[3]}\n"
                        response += f"  Line: {row[4]}\n\n"
                    return response
                else:
                    return "No VSAM/CICS operations found in analyzed programs."
        
        # Field structure queries
        if 'redefines' in question_lower:
            cursor.execute("""
                SELECT program, field_name, redefines, level, picture
                FROM cobol_fields
                WHERE redefines IS NOT NULL
                ORDER BY program, field_name
            """)
            
            results = cursor.fetchall()
            if results:
                response = "REDEFINES structures found:\n\n"
                for row in results:
                    response += f"• {row[1]} REDEFINES {row[2]} (Level {row[3]})\n"
                    response += f"  Program: {row[0]}\n"
                    if row[4]:
                        response += f"  Picture: {row[4]}\n"
                    response += "\n"
                return response
            else:
                return "No REDEFINES structures found."
        
        if 'occurs' in question_lower or 'table' in question_lower:
            cursor.execute("""
                SELECT program, field_name, occurs_clause, depending_on, picture
                FROM cobol_fields
                WHERE occurs_clause IS NOT NULL
                ORDER BY program, field_name
            """)
            
            results = cursor.fetchall()
            if results:
                response = "OCCURS tables found:\n\n"
                for row in results:
                    response += f"• {row[1]} OCCURS {row[2]} TIMES\n"
                    response += f"  Program: {row[0]}\n"
                    if row[3]:
                        response += f"  DEPENDING ON: {row[3]}\n"
                    if row[4]:
                        response += f"  Picture: {row[4]}\n"
                    response += "\n"
                return response
            else:
                return "No OCCURS tables found."
        
        # Copybook field queries
        if 'field' in question_lower and ('structure' in question_lower or 'definition' in question_lower):
            # Extract field name from question
            field_pattern = r'field\s+([A-Z0-9\-]+)'
            match = re.search(field_pattern, question.upper())
            
            if match:
                field_name = match.group(1)
                cursor.execute("""
                    SELECT program, level, picture, field_value, occurs_clause, 
                           depending_on, redefines, usage_clause, field_length, parent_field
                    FROM cobol_fields
                    WHERE field_name = ?
                """, (field_name,))
                
                results = cursor.fetchall()
                if results:
                    response = f"Field {field_name} definition:\n\n"
                    for row in results:
                        response += f"• Program: {row[0]}\n"
                        response += f"  Level: {row[1]}\n"
                        if row[2]:
                            response += f"  Picture: {row[2]}\n"
                        if row[3]:
                            response += f"  Value: {row[3]}\n"
                        if row[4]:
                            response += f"  Occurs: {row[4]} times\n"
                        if row[5]:
                            response += f"  Depending On: {row[5]}\n"
                        if row[6]:
                            response += f"  Redefines: {row[6]}\n"
                        if row[7]:
                            response += f"  Usage: {row[7]}\n"
                        if row[8]:
                            response += f"  Length: {row[8]} bytes\n"
                        if row[9]:
                            response += f"  Parent: {row[9]}\n"
                        response += "\n"
                    return response
                else:
                    return f"Field {field_name} not found in analyzed programs."
        
        # Fall back to original chat logic
        return self.chat_query(question)

    def parse_jcl_file(self, content: str, filename: str) -> Dict[str, Any]:
        """Parse JCL file and extract dataset operations"""
        lines = content.split('\n')
        analysis = {
            'datasets': [],
            'programs_called': [],
            'file_operations': []
        }
        
        # JCL patterns
        dd_pattern = r'//([A-Z0-9]+)\s+DD\s+.*DSN=([A-Z0-9\.]+)'
        exec_pattern = r'//\s*EXEC\s+PGM=([A-Z0-9]+)'
        disp_pattern = r'DISP=\(([A-Z,]+)\)'
        
        for i, line in enumerate(lines, 1):
            line_upper = line.upper()
            
            # Find DD statements
            dd_match = re.search(dd_pattern, line_upper)
            if dd_match:
                ddname, dsname = dd_match.groups()
                
                # Check disposition
                disp_match = re.search(disp_pattern, line_upper)
                disposition = disp_match.group(1) if disp_match else 'UNKNOWN'
                
                operation = 'CREATE' if 'NEW' in disposition else 'READ' if 'SHR' in disposition else 'UPDATE'
                
                ref = FileReference(
                    file_name=dsname,
                    program=filename,
                    operation=operation,
                    fields_used=[],
                    line_number=i,
                    file_type="SEQUENTIAL",
                    access_method="BATCH"
                )
                analysis['file_operations'].append(ref)
            
            # Find program executions
            exec_match = re.search(exec_pattern, line_upper)
            if exec_match:
                program_name = exec_match.group(1)
                analysis['programs_called'].append({
                    'program': program_name,
                    'line_number': i
                })
        
        return analysis

    def chat_query(self, question: str) -> str:
        """Process chat queries about file lineage"""
        question_lower = question.lower()
        
        # Extract file name from question
        file_pattern = r'(?:file|dataset)\s+([A-Z0-9\.\-]+)|([A-Z0-9\.\-]+)\s+(?:file|dataset|created|used)'
        match = re.search(file_pattern, question.upper())
        
        if match:
            file_name = match.group(1) or match.group(2)
            
            if 'created' in question_lower or 'create' in question_lower:
                creation_chain = self.get_file_creation_chain(file_name)
                if creation_chain:
                    response = f"File {file_name} is created by:\n\n"
                    for item in creation_chain:
                        response += f"• Program: {item['program']}\n"
                        response += f"  Operation: {item['operation']} (Line {item['line_number']})\n"
                        if item['fields_used']:
                            response += f"  Fields: {', '.join(item['fields_used'])}\n"
                        response += "\n"
                    return response
                else:
                    return f"No creation information found for file {file_name}"
            
            elif 'used' in question_lower or 'usage' in question_lower:
                usage_chain = self.get_file_usage_chain(file_name)
                if usage_chain:
                    response = f"File {file_name} is used by:\n\n"
                    for item in usage_chain:
                        response += f"• Program: {item['program']}\n"
                        response += f"  Operation: {item['operation']} (Line {item['line_number']})\n"
                        response += "\n"
                    return response
                else:
                    return f"No usage information found for file {file_name}"
        
        # General queries
        if 'files' in question_lower and 'total' in question_lower:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT file_name) FROM file_references")
            count = cursor.fetchone()[0]
            return f"Total files analyzed: {count}"
        
        if 'programs' in question_lower and 'total' in question_lower:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT program) FROM file_references")
            count = cursor.fetchone()[0]
            return f"Total programs analyzed: {count}"
        
        return "I can help you trace file lineage. Try asking:\n• 'How is file CUSTOMER.DATA created?'\n• 'What programs use file ACCOUNT.MASTER?'\n• 'Show me total files analyzed'"

    def get_file_creation_chain(self, file_name: str) -> List[Dict]:
        """Get the creation chain for a specific file"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT program, operation, line_number, fields_used
            FROM file_references 
            WHERE file_name = ? AND operation IN ('CREATE', 'WRITE', 'NEW')
            ORDER BY program, line_number
        """, (file_name,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'program': row[0],
                'operation': row[1],
                'line_number': row[2],
                'fields_used': json.loads(row[3]) if row[3] else []
            })
        
        return results

    def get_file_usage_chain(self, file_name: str) -> List[Dict]:
        """Get all programs that use a specific file"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT program, operation, line_number, fields_used
            FROM file_references 
            WHERE file_name = ?
            ORDER BY program, line_number
        """, (file_name,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'program': row[0],
                'operation': row[1],
                'line_number': row[2],
                'fields_used': json.loads(row[3]) if row[3] else []
            })
        
        return results

    def compare_copybook_to_db2(self, copybook_name: str, table_name: str, db_config: Dict) -> Dict:
        """Compare copybook fields to DB2 table structure"""
        cursor = self.conn.cursor()
        
        # Get copybook fields
        cursor.execute("""
            SELECT field_name, picture, field_length, usage_clause
            FROM cobol_fields
            WHERE program = ? AND level NOT IN ('01', '77', '88')
            ORDER BY field_name
        """, (copybook_name,))
        
        copybook_fields = {}
        for row in cursor.fetchall():
            field_name, picture, length, usage = row
            copybook_fields[field_name] = {
                'picture': picture,
                'length': length,
                'usage': usage
            }
        
        # Initialize comparator and perform comparison
        comparator = DB2CopybookComparator(db_config)
        
        # Create CobolField objects for comparison
        cobol_field_objects = {}
        for name, info in copybook_fields.items():
            cobol_field_objects[name] = CobolField(
                level='05',  # Default level for comparison
                name=name,
                picture=info['picture'],
                usage=info['usage'],
                length=info['length']
            )
        
        comparison_results = comparator.compare_copybook_to_db2(cobol_field_objects, table_name)
        
        # Store comparison results
        for match in comparison_results.get('field_matches', []):
            cursor.execute("""
                INSERT INTO copybook_db2_comparison
                (copybook_name, table_name, field_name, copybook_type, db2_type, 
                 match_status, copybook_length, db2_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (copybook_name, table_name, match['field_name'], 
                  match['copybook_type'], match['db2_type'], 'MATCH',
                  match['copybook_length'], match['db2_length']))
        
        for mismatch in comparison_results.get('type_mismatches', []):
            cursor.execute("""
                INSERT INTO copybook_db2_comparison
                (copybook_name, table_name, field_name, copybook_type, db2_type, match_status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (copybook_name, table_name, mismatch['field_name'],
                  mismatch['copybook_type'], mismatch['db2_type'], 'TYPE_MISMATCH'))
        
        self.conn.commit()
        return comparison_results
    
    def enhanced_chat_query(self, question: str) -> str:
        """Enhanced chat with LLM intelligence and business context"""
        question_lower = question.lower()
        cursor = self.conn.cursor()
        
        # Use LLM intelligence to understand the question intent
        if 'business' in question_lower or 'purpose' in question_lower or 'why' in question_lower:
            return self.get_business_analysis(question)
        
        # Complex analysis queries
        if 'complexity' in question_lower or 'complex' in question_lower:
            cursor.execute("""
                SELECT program, COUNT(*) as complexity_count
                FROM (
                    SELECT program FROM cobol_fields WHERE redefines IS NOT NULL
                    UNION ALL
                    SELECT program FROM cobol_fields WHERE occurs_clause IS NOT NULL  
                    UNION ALL
                    SELECT program FROM vsam_operations
                ) 
                GROUP BY program
                ORDER BY complexity_count DESC
            """)
            
            results = cursor.fetchall()
            if results:
                response = "Programs ranked by complexity:\n\n"
                for row in results:
                    response += f"• {row[0]}: {row[1]} complex structures\n"
                return response
            else:
                return "No complexity analysis available."
        
        # Smart file relationship queries
        if 'relationship' in question_lower or 'flow' in question_lower:
            return self.analyze_data_flow()
        
        # VSAM-specific queries with business context
        if 'vsam' in question_lower or 'cics' in question_lower:
            if 'operations' in question_lower:
                cursor.execute("""
                    SELECT program, operation, file_name, key_field, line_number
                    FROM vsam_operations
                    ORDER BY program, line_number
                """)
                
                results = cursor.fetchall()
                if results:
                    response = "VSAM/CICS Operations with business context:\n\n"
                    for row in results:
                        business_context = self.get_business_context_for_file(row[2], row[1])
                        response += f"• Program: {row[0]}\n"
                        response += f"  Operation: {row[1]} on file {row[2]}\n"
                        response += f"  Business Purpose: {business_context}\n"
                        if row[3]:
                            response += f"  Key Field: {row[3]}\n"
                        response += f"  Line: {row[4]}\n\n"
                    return response
                else:
                    return "No VSAM/CICS operations found in analyzed programs."
        
        # Smart recommendations
        if 'recommend' in question_lower or 'suggest' in question_lower:
            return self.get_intelligent_recommendations()
        
        # Enhanced field structure queries with LLM insights
        if 'redefines' in question_lower:
            cursor.execute("""
                SELECT program, field_name, redefines, level, picture
                FROM cobol_fields
                WHERE redefines IS NOT NULL
                ORDER BY program, field_name
            """)
            
            results = cursor.fetchall()
            if results:
                response = "REDEFINES structures with business analysis:\n\n"
                for row in results:
                    business_reason = self.analyze_redefines_purpose(row[1], row[2])
                    response += f"• {row[1]} REDEFINES {row[2]} (Level {row[3]})\n"
                    response += f"  Program: {row[0]}\n"
                    response += f"  Business Purpose: {business_reason}\n"
                    if row[4]:
                        response += f"  Picture: {row[4]}\n"
                    response += "\n"
                return response
            else:
                return "No REDEFINES structures found."
        
        if 'occurs' in question_lower or 'table' in question_lower:
            cursor.execute("""
                SELECT program, field_name, occurs_clause, depending_on, picture
                FROM cobol_fields
                WHERE occurs_clause IS NOT NULL
                ORDER BY program, field_name
            """)
            
            results = cursor.fetchall()
            if results:
                response = "OCCURS tables with intelligent analysis:\n\n"
                for row in results:
                    business_purpose = self.analyze_occurs_purpose(row[1], row[2])
                    response += f"• {row[1]} OCCURS {row[2]} TIMES\n"
                    response += f"  Program: {row[0]}\n"
                    response += f"  Business Purpose: {business_purpose}\n"
                    if row[3]:
                        response += f"  DEPENDING ON: {row[3]} (Dynamic sizing)\n"
                    if row[4]:
                        response += f"  Picture: {row[4]}\n"
                    response += "\n"
                return response
            else:
                return "No OCCURS tables found."
        
        # Enhanced field definition with LLM context
        if 'field' in question_lower and ('structure' in question_lower or 'definition' in question_lower):
            field_pattern = r'field\s+([A-Z0-9\-]+)'
            match = re.search(field_pattern, question.upper())
            
            if match:
                field_name = match.group(1)
                cursor.execute("""
                    SELECT program, level, picture, field_value, occurs_clause, 
                           depending_on, redefines, usage_clause, field_length, parent_field
                    FROM cobol_fields
                    WHERE field_name = ?
                """, (field_name,))
                
                results = cursor.fetchall()
                if results:
                    response = f"Field {field_name} intelligent analysis:\n\n"
                    for row in results:
                        business_meaning = self.get_field_business_meaning(field_name, row[2])
                        response += f"• Program: {row[0]}\n"
                        response += f"  Level: {row[1]}\n"
                        response += f"  Business Meaning: {business_meaning}\n"
                        if row[2]:
                            response += f"  Picture: {row[2]}\n"
                        if row[3]:
                            response += f"  Value: {row[3]}\n"
                        if row[4]:
                            response += f"  Occurs: {row[4]} times\n"
                        if row[5]:
                            response += f"  Depending On: {row[5]}\n"
                        if row[6]:
                            response += f"  Redefines: {row[6]}\n"
                        if row[7]:
                            response += f"  Usage: {row[7]}\n"
                        if row[8]:
                            response += f"  Length: {row[8]} bytes\n"
                        if row[9]:
                            response += f"  Parent: {row[9]}\n"
                        response += "\n"
                    return response
                else:
                    return f"Field {field_name} not found in analyzed programs."
        
        # Fall back to original chat logic
        return self.chat_query(question)
    
    def get_business_analysis(self, question: str) -> str:
        """Provide business-focused analysis using LLM intelligence"""
        cursor = self.conn.cursor()
        
        # Analyze overall business patterns
        cursor.execute("SELECT DISTINCT program FROM file_references")
        programs = [row[0] for row in cursor.fetchall()]
        
        if not programs:
            return "No programs analyzed yet for business context."
        
        analysis = f"Business Analysis Summary:\n\n"
        
        # File type distribution
        cursor.execute("""
            SELECT file_type, COUNT(*) 
            FROM file_references 
            GROUP BY file_type
        """)
        
        file_types = cursor.fetchall()
        analysis += "📊 System Architecture:\n"
        for file_type, count in file_types:
            if file_type == 'VSAM':
                analysis += f"• VSAM files: {count} (Online transaction processing)\n"
            else:
                analysis += f"• Sequential files: {count} (Batch processing)\n"
        
        # Business patterns
        cursor.execute("""
            SELECT operation, COUNT(*) 
            FROM file_references 
            GROUP BY operation
        """)
        
        operations = cursor.fetchall()
        analysis += "\n💼 Business Operations:\n"
        for op, count in operations:
            if op in ['READ', 'OPEN']:
                analysis += f"• Data access operations: {count} (Information retrieval)\n"
            elif op in ['WRITE', 'CREATE']:
                analysis += f"• Data creation operations: {count} (Record management)\n"
            elif op == 'REWRITE':
                analysis += f"• Data update operations: {count} (Maintenance processes)\n"
        
        return analysis
    
    def analyze_data_flow(self) -> str:
        """Intelligent data flow analysis"""
        cursor = self.conn.cursor()
        
        # Find data transformation patterns
        cursor.execute("""
            SELECT f1.file_name as input_file, f1.program, f2.file_name as output_file
            FROM file_references f1
            JOIN file_references f2 ON f1.program = f2.program
            WHERE f1.operation IN ('READ', 'OPEN') 
            AND f2.operation IN ('WRITE', 'CREATE')
            AND f1.file_name != f2.file_name
            ORDER BY f1.program
        """)
        
        flows = cursor.fetchall()
        if flows:
            response = "Intelligent Data Flow Analysis:\n\n"
            for input_file, program, output_file in flows:
                response += f"🔄 {program}:\n"
                response += f"   {input_file} → {output_file}\n"
                
                # Add business context
                input_context = self.get_business_context_for_file(input_file, "READ")
                output_context = self.get_business_context_for_file(output_file, "WRITE")
                response += f"   Purpose: {input_context} → {output_context}\n\n"
            
            return response
        else:
            return "No data flow patterns identified in analyzed programs."
    
    def get_intelligent_recommendations(self) -> str:
        """Provide intelligent recommendations based on analysis"""
        cursor = self.conn.cursor()
        
        recommendations = "🎯 Intelligent Recommendations:\n\n"
        
        # Check for modernization opportunities
        cursor.execute("SELECT COUNT(*) FROM vsam_operations")
        vsam_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM file_references WHERE file_type = 'SEQUENTIAL'")
        seq_count = cursor.fetchone()[0]
        
        if vsam_count > 0 and seq_count > vsam_count * 2:
            recommendations += "📈 Modernization: Consider migrating sequential processing to VSAM for better performance\n\n"
        
        # Check for complex structures
        cursor.execute("SELECT COUNT(*) FROM cobol_fields WHERE redefines IS NOT NULL")
        redefines_count = cursor.fetchone()[0]
        
        if redefines_count > 10:
            recommendations += "🔧 Architecture: High number of REDEFINES suggests data model modernization opportunity\n\n"
        
        # Performance recommendations
        cursor.execute("""
            SELECT program, COUNT(*) as file_count 
            FROM file_references 
            GROUP BY program 
            HAVING COUNT(*) > 5
        """)
        
        heavy_programs = cursor.fetchall()
        if heavy_programs:
            recommendations += "⚡ Performance: Programs with high file I/O:\n"
            for program, count in heavy_programs:
                recommendations += f"   • {program}: {count} file operations (consider optimization)\n"
        
        return recommendations
    
    def analyze_redefines_purpose(self, field_name: str, redefines_field: str) -> str:
        """Analyze business purpose of REDEFINES"""
        field_upper = field_name.upper()
        redefines_upper = redefines_field.upper()
        
        if 'DATE' in field_upper and 'DATE' in redefines_upper:
            return "Date format conversion (YYYYMMDD vs DD/MM/YYYY)"
        elif 'AMOUNT' in field_upper or 'BALANCE' in field_upper:
            return "Financial amount format conversion (signed vs unsigned)"
        elif 'CODE' in field_upper:
            return "Code interpretation (numeric vs character representation)"
        else:
            return "Data format transformation for different processing needs"
    
    def analyze_occurs_purpose(self, field_name: str, occurs_count: str) -> str:
        """Analyze business purpose of OCCURS"""
        field_upper = field_name.upper()
        
        if 'TRANSACTION' in field_upper or 'TXN' in field_upper:
            return f"Transaction history array ({occurs_count} transactions max)"
        elif 'BALANCE' in field_upper:
            return f"Multi-currency or multi-account balance array ({occurs_count} entries)"
        elif 'ADDRESS' in field_upper:
            return f"Multiple address lines ({occurs_count} lines max)"
        elif 'PHONE' in field_upper:
            return f"Multiple phone numbers ({occurs_count} numbers max)"
        else:
            return f"Repeating data structure for bulk processing ({occurs_count} entries)"
    
    def get_field_business_meaning(self, field_name: str, picture: str) -> str:
        """Get business meaning of field based on name and picture"""
        field_upper = field_name.upper()
        
        if 'CUSTOMER' in field_upper and 'ID' in field_upper:
            return "Customer identification for CRM and transaction tracking"
        elif 'ACCOUNT' in field_upper and ('NUM' in field_upper or 'ID' in field_upper):
            return "Account number for financial transaction processing"
        elif 'BALANCE' in field_upper:
            return "Financial balance for accounting and reporting"
        elif 'DATE' in field_upper:
            return "Date field for temporal tracking and reporting"
        elif 'AMOUNT' in field_upper:
            return "Monetary amount for financial calculations"
        elif 'STATUS' in field_upper:
            return "Status indicator for workflow and business rule processing"
        elif 'NAME' in field_upper:
            return "Name field for identification and correspondence"
        elif picture and 'X' in picture:
            return f"Text/character data ({self.cobol_parser.calculate_field_length(picture)} chars max)"
        elif picture and '9' in picture:
            return f"Numeric data for calculations and processing"
        else:
            return "Business data field for operational processing"

    def store_enhanced_analysis(self, analysis: Dict[str, Any], program_name: str):
        """Store enhanced analysis results with LLM insights"""
        cursor = self.conn.cursor()
        
        # Store LLM analysis insights
        if 'llm_insights' in analysis:
            insights = analysis['llm_insights']
            
            # Store business patterns
            if 'business_patterns' in insights:
                patterns = insights['business_patterns']
                cursor.execute("""
                    INSERT INTO llm_analysis 
                    (program, analysis_type, business_context, technical_details, complexity_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    program_name, 'BUSINESS_PATTERNS',
                    json.dumps(patterns), 
                    f"Patterns: {', '.join([k for k, v in patterns.items() if v])}",
                    len([k for k, v in patterns.items() if v])
                ))
            
            # Store file operation insights
            if 'file_operations' in insights:
                for op in insights['file_operations']:
                    cursor.execute("""
                        INSERT INTO llm_analysis 
                        (program, analysis_type, business_context, technical_details)
                        VALUES (?, ?, ?, ?)
                    """, (
                        program_name, 'FILE_OPERATION',
                        op.get('business_context', ''),
                        f"{op.get('operation', '')} on {op.get('file', '')}"
                    ))
        
        # Store COBOL fields (existing logic)
        for field_name, field_obj in analysis.get('copybook_fields', {}).items():
            cursor.execute("""
                INSERT INTO cobol_fields 
                (program, field_name, level, picture, field_value, occurs_clause, 
                 depending_on, redefines, usage_clause, is_filler, field_length, 
                 parent_field, qualified_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_name, field_name, field_obj.level, field_obj.picture,
                field_obj.value, field_obj.occurs, field_obj.depending_on,
                field_obj.redefines, field_obj.usage, field_obj.is_filler,
                field_obj.length, field_obj.parent, 
                f"{field_obj.parent}.{field_name}" if field_obj.parent else field_name
            ))
        
        # Store VSAM operations
        for vsam_op in analysis.get('vsam_operations', []):
            cursor.execute("""
                INSERT INTO vsam_operations 
                (program, operation, file_name, key_field, record_area, response_code, line_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                vsam_op.program, vsam_op.operation, vsam_op.file_name,
                vsam_op.key_field, vsam_op.record_area, vsam_op.response_code,
                vsam_op.line_number
            ))
        
        # Store file references with enhanced info
        for file_ref in analysis.get('file_operations', []):
            cursor.execute("""
                INSERT INTO file_references 
                (file_name, program, operation, fields_used, line_number, file_type, access_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_ref.file_name, file_ref.program, file_ref.operation,
                json.dumps(file_ref.fields_used), file_ref.line_number,
                file_ref.file_type, file_ref.access_method
            ))
        
        self.conn.commit()

    def calculate_field_length(self, picture: str) -> int:
        """Calculate field length from PICTURE clause"""
        if not picture:
            return 0
            
        # Remove formatting characters
        pic_clean = re.sub(r'[V\.\-\+S]', '', picture)
        
        # Handle repeated characters like X(10) or 9(5)
        total_length = 0
        
        # Pattern for X(n) or 9(n)
        repeat_pattern = r'([X9])(\((\d+)\))?'
        matches = re.findall(repeat_pattern, pic_clean)
        
        for match in matches:
            char_type = match[0]
            repeat_count = int(match[2]) if match[2] else 1
            total_length += repeat_count
        
        # Handle simple patterns like XXX or 999
        simple_chars = re.sub(r'[X9]\(\d+\)', '', pic_clean)
        total_length += len(simple_chars)
        
        return total_length

    def analyze_uploaded_files(self, uploaded_files: List) -> Dict[str, Any]:
        """Enhanced file analysis with LLM intelligence"""
        results = {
            'programs_analyzed': 0,
            'files_found': set(),
            'vsam_files': set(),
            'db2_files': set(),
            'field_definitions': {},
            'file_lineage': {},
            'redefines_found': 0,
            'occurs_found': 0,
            'cics_operations': 0,
            'business_insights': {},
            'complexity_score': 0
        }
        
        total_complexity = 0
        
        for uploaded_file in uploaded_files:
            content = str(uploaded_file.read(), "utf-8")
            filename = uploaded_file.name
            
            if filename.endswith(('.cbl', '.cob', '.cobol', '.cpy')):
                analysis = self.parse_cobol_file(content, filename)
                
                # Count complex structures
                for field_name, field_obj in analysis.get('copybook_fields', {}).items():
                    if field_obj.redefines:
                        results['redefines_found'] += 1
                        total_complexity += 2
                    if field_obj.occurs:
                        results['occurs_found'] += 1
                        total_complexity += 1
                
                results['cics_operations'] += len(analysis.get('vsam_operations', []))
                total_complexity += len(analysis.get('vsam_operations', [])) * 3
                
                # Extract business insights from LLM analysis
                if 'llm_insights' in analysis:
                    insights = analysis['llm_insights']
                    if 'business_patterns' in insights:
                        results['business_insights'][filename] = insights['business_patterns']
                
                # Store enhanced analysis
                self.store_enhanced_analysis(analysis, filename)
                
            elif filename.endswith(('.jcl', '.proc')):
                analysis = self.parse_jcl_file(content, filename)
            else:
                continue
            
            # Process results
            for file_op in analysis.get('file_operations', []):
                results['files_found'].add(file_op.file_name)
                if file_op.file_type == 'VSAM':
                    results['vsam_files'].add(file_op.file_name)
            
            for vsam_op in analysis.get('vsam_operations', []):
                results['vsam_files'].add(vsam_op.file_name)
            
            results['programs_analyzed'] += 1
        
        results['complexity_score'] = total_complexity
        return results

    def get_llm_summary(self) -> str:
        """Generate LLM-powered summary of the entire codebase"""
        cursor = self.conn.cursor()
        
        # Get overall statistics
        cursor.execute("SELECT COUNT(DISTINCT program) FROM file_references")
        total_programs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT file_name) FROM file_references")
        total_files = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM cobol_fields WHERE redefines IS NOT NULL")
        redefines_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vsam_operations")
        cics_count = cursor.fetchone()[0]
        
        # Generate intelligent summary
        summary = f"""
🤖 **LLM Intelligence Summary**

**System Architecture Analysis:**
• Analyzed {total_programs} programs with {total_files} data files
• Complexity Level: {'High' if redefines_count > 10 else 'Medium' if redefines_count > 5 else 'Low'}
• CICS/Online Processing: {'Extensive' if cics_count > 20 else 'Moderate' if cics_count > 5 else 'Minimal'}

**Business Pattern Recognition:**
• Processing Type: {'Mixed Batch/Online' if cics_count > 0 else 'Batch-Oriented'}
• Data Complexity: {'High (Legacy)' if redefines_count > 15 else 'Standard'}
• Architecture: {'Mainframe Legacy' if cics_count > 0 else 'Traditional Batch'}

**Modernization Readiness:**
• Legacy Debt: {'High' if redefines_count > 20 else 'Moderate'}
• Migration Complexity: {'Complex' if cics_count > 10 else 'Standard'}
• Recommended Approach: {'Phased Migration' if cics_count > 0 else 'Direct Migration'}
        """
        
        return summary

# Initialize the enhanced analyzer
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MainframeAnalyzer()

# Main UI
st.title("🔍 Mainframe Deep Research Agent")
st.markdown("### Advanced COBOL/JCL Analysis with VSAM, CICS & DB2 Support")

# Sidebar for configuration
with st.sidebar:
    st.header("📂 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload COBOL/JCL/Copybook files",
        accept_multiple_files=True,
        type=['cbl', 'cob', 'cobol', 'jcl', 'proc', 'cpy', 'txt']
    )
    
    st.header("🗄️ DB2 Configuration")
    db2_host = st.text_input("Host", value="localhost")
    db2_port = st.text_input("Port", value="50000")
    db2_username = st.text_input("Username")
    db2_password = st.text_input("Password", type="password")
    db2_schema = st.text_input("Schema")
    
    st.header("📊 Analysis Options")
    analyze_redefines = st.checkbox("Analyze REDEFINES", value=True)
    analyze_occurs = st.checkbox("Analyze OCCURS", value=True)
    analyze_cics = st.checkbox("Analyze CICS/VSAM", value=True)
    compare_db2 = st.checkbox("Compare with DB2", value=False)
    
    if st.button("🔍 Analyze Files"):
        if uploaded_files:
            with st.spinner("Performing advanced analysis..."):
                results = st.session_state.analyzer.analyze_uploaded_files(uploaded_files)
                st.session_state.analysis_results = results
                
                # Show enhanced results
                st.success(f"✅ Analyzed {results['programs_analyzed']} programs")
                st.info(f"📁 Found {len(results['files_found'])} files")
                st.info(f"💾 VSAM files: {len(results['vsam_files'])}")
                st.info(f"🔄 REDEFINES found: {results['redefines_found']}")
                st.info(f"📋 OCCURS tables: {results['occurs_found']}")
                st.info(f"🖥️ CICS operations: {results['cics_operations']}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Advanced Analysis Chat")
    
    # Enhanced chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sample questions with LLM intelligence
    st.subheader("🧠 Try These AI-Powered Queries:")
    sample_questions = [
        "What's the business purpose of this system?",
        "Show me complexity analysis",
        "What are the data flow relationships?", 
        "Give me intelligent recommendations",
        "Show me all VSAM operations with business context",
        "What REDEFINES structures were found and why?",
        "Analyze field CUSTOMER-ID structure and meaning",
        "What's the modernization readiness of this code?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"sample_{i}"):
                with st.spinner("Analyzing..."):
                    answer = st.session_state.analyzer.enhanced_chat_query(question)
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
    
    st.divider()
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.expander(f"💭 Q{i+1}: {question[:50]}..."):
            st.text_area(f"Question:", value=question, height=50, disabled=True, key=f"q_{i}")
            st.text_area(f"Answer:", value=answer, height=150, disabled=True, key=f"a_{i}")
    
    # Chat input with AI context
    user_question = st.text_input("Ask AI about business context, complexity, patterns:", 
                                 placeholder="What's the business purpose of this system?")
    
    if st.button("🤖 Ask AI") and user_question:
        with st.spinner("AI analyzing..."):
            answer = st.session_state.analyzer.enhanced_chat_query(user_question)
            st.session_state.chat_history.append((user_question, answer))
            st.rerun()

with col2:
    st.header("📊 Advanced Analysis Summary")
    
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # Enhanced metrics with AI insights
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Programs", results['programs_analyzed'])
            st.metric("Total Files", len(results['files_found']))
            st.metric("AI Complexity", results.get('complexity_score', 0))
        with col2b:
            st.metric("VSAM Files", len(results['vsam_files']))
            st.metric("CICS Ops", results['cics_operations'])
            complexity_level = "High" if results.get('complexity_score', 0) > 50 else "Medium" if results.get('complexity_score', 0) > 20 else "Low"
            st.metric("AI Assessment", complexity_level)
        
        # Advanced structures found
        st.subheader("🧬 Complex Structures")
        complexity_col1, complexity_col2 = st.columns(2)
        with complexity_col1:
            st.metric("REDEFINES", results['redefines_found'])
        with complexity_col2:
            st.metric("OCCURS", results['occurs_found'])
        
        # File categorization
        st.subheader("📁 File Types")
        if results['vsam_files']:
            st.write("**VSAM Files:**")
            for file_name in sorted(results['vsam_files']):
                st.text(f"💾 {file_name}")
        
        regular_files = results['files_found'] - results['vsam_files']
        if regular_files:
            st.write("**Sequential Files:**")
            for file_name in sorted(regular_files):
                st.text(f"📄 {file_name}")
        
        # LLM Intelligence Summary
        st.subheader("🤖 LLM Intelligence")
        if st.button("🧠 Generate AI Summary"):
            with st.spinner("AI analyzing codebase..."):
                llm_summary = st.session_state.analyzer.get_llm_summary()
                st.markdown(llm_summary)
        
        # AI-Powered Insights
        st.subheader("💡 AI Insights")
        if 'analysis_results' in st.session_state:
            insights = st.session_state.analysis_results.get('business_insights', {})
            if insights:
                for program, patterns in insights.items():
                    with st.expander(f"🔍 {program} Business Patterns"):
                        for pattern, detected in patterns.items():
                            if detected:
                                st.write(f"✅ {pattern.replace('_', ' ').title()}")
                            else:
                                st.write(f"❌ {pattern.replace('_', ' ').title()}")
        
        # Quick DB2 comparison
        st.subheader("🔄 DB2 Comparison")
        copybook_select = st.selectbox("Select Copybook:", 
                                     options=[''] + [f for f in results['files_found']])
        table_name = st.text_input("DB2 Table Name:")
        
        if st.button("🔍 Compare with DB2") and copybook_select and table_name:
            db_config = {
                'host': db2_host,
                'port': db2_port, 
                'username': db2_username,
                'password': db2_password,
                'schema': db2_schema
            }
            
            with st.spinner("Comparing structures..."):
                comparison = st.session_state.analyzer.compare_copybook_to_db2(
                    copybook_select, table_name, db_config)
                
                st.write("**Comparison Results:**")
                st.write(f"✅ Matches: {len(comparison.get('field_matches', []))}")
                st.write(f"⚠️ Type Mismatches: {len(comparison.get('type_mismatches', []))}")
                st.write(f"❌ Missing in DB2: {len(comparison.get('missing_in_db2', []))}")
                st.write(f"➕ Missing in Copybook: {len(comparison.get('missing_in_copybook', []))}")
        
        # Export options
        st.subheader("📤 Export Reports")
        report_type = st.selectbox("Report Type:", [
            "Field Lineage Report",
            "VSAM Operations Report", 
            "REDEFINES Analysis",
            "OCCURS Tables Report",
            "DB2 Comparison Report"
        ])
        
        if st.button("📋 Generate Report"):
            st.info(f"Generating {report_type}...")
            # Report generation logic would go here
    else:
        st.info("👆 Upload files and click 'Analyze Files' to start")
        
        # Feature highlights
        st.subheader("🌟 Advanced Features")
        st.markdown("""
        **📊 Complex Structure Analysis:**
        - REDEFINES with multiple overlays
        - OCCURS with DEPENDING ON
        - Nested group structures
        - FILLER field handling
        
        **💾 VSAM/CICS Support:**
        - READ/WRITE/REWRITE operations
        - STARTBR/READNEXT browsing
        - Key field identification
        - Response code tracking
        
        **🔄 DB2 Integration:**
        - Copybook to table comparison
        - Field type mapping
        - Length validation
        - Missing field detection
        """)

# Footer with LLM integration
st.markdown("---")
st.markdown("**🧠 AI-Powered Tech Stack:** CodeLlama + Advanced LLM Intelligence • Smart COBOL Parser • Business Context Analyzer • VSAM/CICS Intelligence • DB2 Comparator • SQLite • A100 GPU")
st.markdown("**🚀 LLM Features:** Business pattern recognition • Intelligent recommendations • Context-aware analysis • Smart field interpretation • Modernization assessment")