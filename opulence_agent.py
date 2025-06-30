"""
Mainframe Deep Research Agent - Pure Analysis Focus
Separated batch processing and interactive analysis
"""

import os
import sqlite3
import pandas as pd
import streamlit as st
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import re
import json
from datetime import datetime
import hashlib

# Core Data Models
@dataclass
class ComponentAnalysis:
    """Complete analysis of a mainframe component"""
    name: str
    component_type: str  # PROGRAM, FILE, JOB, COPYBOOK, TABLE
    subtype: str  # COBOL, JCL, VSAM, DB2, CICS, etc.
    
    # Operations Analysis
    operations: List[Dict[str, Any]] = field(default_factory=list)
    file_operations: List[Dict[str, Any]] = field(default_factory=list)
    data_operations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Field Analysis
    fields_defined: List[Dict[str, Any]] = field(default_factory=list)
    fields_used: List[Dict[str, Any]] = field(default_factory=list)
    fields_modified: List[Dict[str, Any]] = field(default_factory=list)
    fields_unused: List[Dict[str, Any]] = field(default_factory=list)
    
    # Lifecycle & Dependencies
    creates_files: Set[str] = field(default_factory=set)
    reads_files: Set[str] = field(default_factory=set)
    updates_files: Set[str] = field(default_factory=set)
    deletes_files: Set[str] = field(default_factory=set)
    
    calls_programs: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    
    # Business Logic
    business_functions: List[str] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    calculations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    complexity_score: int = 0
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    
    # Raw content for chat analysis
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComponentDetector:
    """Auto-detect component type and analyze accordingly"""
    
    def __init__(self):
        self.detection_patterns = {
            'COBOL_PROGRAM': [
                r'IDENTIFICATION\s+DIVISION',
                r'PROGRAM-ID\s*\.',
                r'DATA\s+DIVISION',
                r'PROCEDURE\s+DIVISION'
            ],
            'JCL_JOB': [
                r'^//\w+\s+JOB\s+',
                r'^//\w+\s+EXEC\s+PGM=',
                r'^//\w+\s+DD\s+'
            ],
            'COPYBOOK': [
                r'^\s*\d+\s+\w+\s+PIC',
                r'^\s*\d+\s+FILLER',
                r'COPY\s+REPLACING'
            ],
            'DB2_SQL': [
                r'CREATE\s+TABLE',
                r'SELECT.*FROM',
                r'INSERT\s+INTO',
                r'EXEC\s+SQL'
            ],
            'CICS_MAP': [
                r'DFHMSD',
                r'DFHMDI',
                r'DFHMDF'
            ],
            'VSAM_CLUSTER': [
                r'DEFINE\s+CLUSTER',
                r'DEFINE\s+AIX',
                r'LISTCAT'
            ]
        }
    
    def detect_component_type(self, content: str, filename: str = "") -> Tuple[str, str]:
        """Auto-detect component type and subtype"""
        content_upper = content.upper()
        
        # Check file extension first
        if filename:
            ext = filename.split('.')[-1].upper()
            if ext in ['CBL', 'COB', 'COBOL']:
                return 'PROGRAM', 'COBOL'
            elif ext in ['JCL', 'JOB']:
                return 'JOB', 'JCL'
            elif ext in ['CPY', 'COPY']:
                return 'COPYBOOK', 'COBOL'
            elif ext in ['SQL', 'DDL']:
                return 'TABLE', 'DB2'
        
        # Pattern-based detection
        for component_type, patterns in self.detection_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, content_upper))
            if matches >= 2:  # Require multiple pattern matches
                if component_type == 'COBOL_PROGRAM':
                    return 'PROGRAM', 'COBOL'
                elif component_type == 'JCL_JOB':
                    return 'JOB', 'JCL'
                elif component_type == 'COPYBOOK':
                    return 'COPYBOOK', 'COBOL'
                elif component_type == 'DB2_SQL':
                    return 'TABLE', 'DB2'
                elif component_type == 'CICS_MAP':
                    return 'MAP', 'CICS'
                elif component_type == 'VSAM_CLUSTER':
                    return 'FILE', 'VSAM'
        
        return 'UNKNOWN', 'UNKNOWN'

class BatchProcessor:
    """Handles batch loading, indexing, and chunking"""
    
    def __init__(self, db_path: str = "mainframe_analysis.db"):
        self.db_path = db_path
        self.detector = ComponentDetector()
        self.setup_database()
    
    def setup_database(self):
        """Setup analysis database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Components table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS components (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                component_type TEXT,
                subtype TEXT,
                content TEXT,
                content_hash TEXT,
                analysis_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Operations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY,
                component_name TEXT,
                operation_type TEXT,
                target_name TEXT,
                operation_details TEXT,
                line_number INTEGER,
                context TEXT,
                FOREIGN KEY (component_name) REFERENCES components (name)
            )
        ''')
        
        # Field lineage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS field_lineage (
                id INTEGER PRIMARY KEY,
                component_name TEXT,
                field_name TEXT,
                action_type TEXT,
                line_number INTEGER,
                context TEXT,
                field_definition TEXT,
                data_type TEXT,
                usage_frequency INTEGER DEFAULT 1,
                FOREIGN KEY (component_name) REFERENCES components (name)
            )
        ''')
        
        # Dependencies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependencies (
                id INTEGER PRIMARY KEY,
                source_component TEXT,
                target_component TEXT,
                dependency_type TEXT,
                relationship_details TEXT,
                FOREIGN KEY (source_component) REFERENCES components (name),
                FOREIGN KEY (target_component) REFERENCES components (name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def batch_load_components(self, file_paths: List[str]) -> Dict[str, Any]:
        """Load and process multiple components in batch"""
        results = {
            'processed': 0,
            'errors': 0,
            'components': []
        }
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                filename = os.path.basename(file_path)
                component_type, subtype = self.detector.detect_component_type(content, filename)
                
                # Create component analysis
                analysis = ComponentAnalysis(
                    name=filename,
                    component_type=component_type,
                    subtype=subtype,
                    content=content
                )
                
                # Perform analysis based on type
                if subtype == 'COBOL':
                    self._analyze_cobol_component(analysis)
                elif subtype == 'JCL':
                    self._analyze_jcl_component(analysis)
                elif subtype == 'DB2':
                    self._analyze_db2_component(analysis)
                
                # Store in database
                self._store_component_analysis(analysis)
                
                results['components'].append(analysis.name)
                results['processed'] += 1
                
            except Exception as e:
                results['errors'] += 1
                print(f"Error processing {file_path}: {e}")
        
        return results
    
    def _analyze_cobol_component(self, analysis: ComponentAnalysis):
        """Comprehensive COBOL program analysis"""
        lines = analysis.content.split('\n')
        analysis.lines_of_code = len([l for l in lines if l.strip() and not l.strip().startswith('*')])
        
        current_section = None
        field_definitions = {}
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            # Track sections
            if 'IDENTIFICATION DIVISION' in line.upper():
                current_section = 'IDENTIFICATION'
            elif 'DATA DIVISION' in line.upper():
                current_section = 'DATA'
            elif 'PROCEDURE DIVISION' in line.upper():
                current_section = 'PROCEDURE'
            
            # Field definitions in DATA DIVISION
            if current_section == 'DATA':
                field_match = re.search(r'(\d+)\s+(\w+)\s+PIC\s+([X9V\(\)S\+\-\.]+)', line.upper())
                if field_match:
                    level, field_name, pic_clause = field_match.groups()
                    field_definitions[field_name] = {
                        'level': level,
                        'pic': pic_clause,
                        'line': line_num,
                        'definition': line
                    }
                    analysis.fields_defined.append({
                        'name': field_name,
                        'level': level,
                        'picture': pic_clause,
                        'line_number': line_num,
                        'definition': line
                    })
            
            # File operations
            file_ops = [
                (r'OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+(\w+)', 'OPEN'),
                (r'READ\s+(\w+)', 'READ'),
                (r'WRITE\s+(\w+)', 'WRITE'),
                (r'REWRITE\s+(\w+)', 'REWRITE'),
                (r'DELETE\s+(\w+)', 'DELETE'),
                (r'CLOSE\s+(\w+)', 'CLOSE')
            ]
            
            for pattern, operation in file_ops:
                match = re.search(pattern, line.upper())
                if match:
                    file_name = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    analysis.file_operations.append({
                        'operation': operation,
                        'file_name': file_name,
                        'line_number': line_num,
                        'context': line
                    })
                    
                    # Update lifecycle sets
                    if operation == 'OPEN':
                        mode = match.group(1) if len(match.groups()) > 1 else 'UNKNOWN'
                        if mode in ['OUTPUT', 'EXTEND']:
                            analysis.creates_files.add(file_name)
                        else:
                            analysis.reads_files.add(file_name)
                    elif operation in ['READ']:
                        analysis.reads_files.add(file_name)
                    elif operation in ['WRITE', 'REWRITE']:
                        analysis.updates_files.add(file_name)
                    elif operation == 'DELETE':
                        analysis.deletes_files.add(file_name)
            
            # Field usage analysis
            if current_section == 'PROCEDURE':
                # MOVE statements
                move_match = re.search(r'MOVE\s+(\w+)\s+TO\s+(\w+)', line.upper())
                if move_match:
                    source_field, target_field = move_match.groups()
                    analysis.fields_used.append({
                        'name': source_field,
                        'action': 'READ',
                        'line_number': line_num,
                        'context': line
                    })
                    analysis.fields_modified.append({
                        'name': target_field,
                        'action': 'write',
                        'line_number': line_num,
                        'context': line
                    })
                
                # COMPUTE statements
                compute_match = re.search(r'COMPUTE\s+(\w+)\s*=', line.upper())
                if compute_match:
                    field_name = compute_match.group(1)
                    analysis.fields_modified.append({
                        'name': field_name,
                        'action': 'compute',
                        'line_number': line_num,
                        'context': line
                    })
                    analysis.calculations.append({
                        'target_field': field_name,
                        'line_number': line_num,
                        'expression': line
                    })
                
                # Decision points
                if_match = re.search(r'IF\s+(.+?)(\s+THEN|\s*$)', line.upper())
                if if_match:
                    condition = if_match.group(1)
                    analysis.decision_points.append({
                        'type': 'IF',
                        'condition': condition,
                        'line_number': line_num,
                        'context': line
                    })
                    analysis.cyclomatic_complexity += 1
                
                # EVALUATE statements
                eval_match = re.search(r'EVALUATE\s+(\w+)', line.upper())
                if eval_match:
                    analysis.decision_points.append({
                        'type': 'EVALUATE',
                        'variable': eval_match.group(1),
                        'line_number': line_num,
                        'context': line
                    })
                    analysis.cyclomatic_complexity += 1
                
                # PERFORM statements (program calls)
                perform_match = re.search(r'PERFORM\s+(\w+)', line.upper())
                if perform_match:
                    section_name = perform_match.group(1)
                    analysis.calls_programs.add(section_name)
                    analysis.operations.append({
                        'type': 'PERFORM',
                        'target': section_name,
                        'line_number': line_num,
                        'context': line
                    })
                
                # Business logic patterns
                business_patterns = [
                    (r'CALCULATE|COMPUTE|ADD|SUBTRACT|MULTIPLY|DIVIDE', 'CALCULATION'),
                    (r'VALIDATE|CHECK|VERIFY', 'VALIDATION'),
                    (r'FORMAT|CONVERT|TRANSFORM', 'TRANSFORMATION'),
                    (r'TOTAL|SUM|COUNT|AVERAGE', 'AGGREGATION'),
                    (r'REPORT|PRINT|DISPLAY', 'REPORTING')
                ]
                
                for pattern, function_type in business_patterns:
                    if re.search(pattern, line.upper()):
                        analysis.business_functions.append(function_type)
        
        # Identify unused fields
        defined_fields = {f['name'] for f in analysis.fields_defined}
        used_fields = {f['name'] for f in analysis.fields_used}
        modified_fields = {f['name'] for f in analysis.fields_modified}
        
        unused_fields = defined_fields - used_fields - modified_fields
        analysis.fields_unused = [{'name': field, 'reason': 'never_referenced'} 
                                for field in unused_fields]
        
        # Calculate complexity score
        analysis.complexity_score = (
            analysis.lines_of_code * 0.1 +
            analysis.cyclomatic_complexity * 5 +
            len(analysis.file_operations) * 2 +
            len(analysis.fields_defined) * 0.5
        )
    
    def _analyze_jcl_component(self, analysis: ComponentAnalysis):
        """Comprehensive JCL job analysis"""
        lines = analysis.content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//*'):
                continue
            
            # Job statement
            job_match = re.search(r'^//(\w+)\s+JOB\s+(.+)', line.upper())
            if job_match:
                job_name = job_match.group(1)
                analysis.operations.append({
                    'type': 'JOB_DEFINITION',
                    'target': job_name,
                    'line_number': line_num,
                    'context': line
                })
            
            # EXEC statements
            exec_match = re.search(r'^//(\w+)\s+EXEC\s+(?:PGM=)?(\w+)', line.upper())
            if exec_match:
                step_name, program_name = exec_match.groups()
                analysis.calls_programs.add(program_name)
                analysis.operations.append({
                    'type': 'EXEC_PROGRAM',
                    'target': program_name,
                    'step': step_name,
                    'line_number': line_num,
                    'context': line
                })
            
            # DD statements
            dd_match = re.search(r'^//(\w+)\s+DD\s+(.+)', line.upper())
            if dd_match:
                dd_name, dd_params = dd_match.groups()
                
                # Extract DSN
                dsn_match = re.search(r'DSN=([^,\s]+)', dd_params)
                if dsn_match:
                    dataset_name = dsn_match.group(1)
                    
                    # Extract DISP
                    disp_match = re.search(r'DISP=\(([^)]+)\)', dd_params)
                    disposition = disp_match.group(1) if disp_match else 'OLD'
                    
                    # Determine operation type
                    if 'NEW' in disposition:
                        analysis.creates_files.add(dataset_name)
                        operation_type = 'CREATE'
                    elif any(x in disposition for x in ['OLD', 'SHR']):
                        analysis.reads_files.add(dataset_name)
                        operation_type = 'READ'
                    elif 'MOD' in disposition:
                        analysis.updates_files.add(dataset_name)
                        operation_type = 'UPDATE'
                    else:
                        operation_type = 'ACCESS'
                    
                    analysis.file_operations.append({
                        'operation': operation_type,
                        'file_name': dataset_name,
                        'dd_name': dd_name,
                        'disposition': disposition,
                        'line_number': line_num,
                        'context': line
                    })
    
    def _analyze_db2_component(self, analysis: ComponentAnalysis):
        """Comprehensive DB2 SQL analysis"""
        content_upper = analysis.content.upper()
        lines = analysis.content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            
            # Table operations
            sql_ops = [
                (r'CREATE\s+TABLE\s+(\w+)', 'CREATE_TABLE'),
                (r'SELECT.*?FROM\s+(\w+)', 'SELECT'),
                (r'INSERT\s+INTO\s+(\w+)', 'INSERT'),
                (r'UPDATE\s+(\w+)', 'UPDATE'),
                (r'DELETE\s+FROM\s+(\w+)', 'DELETE'),
                (r'DROP\s+TABLE\s+(\w+)', 'DROP_TABLE')
            ]
            
            for pattern, operation in sql_ops:
                matches = re.finditer(pattern, line.upper())
                for match in matches:
                    table_name = match.group(1)
                    analysis.data_operations.append({
                        'operation': operation,
                        'table_name': table_name,
                        'line_number': line_num,
                        'context': line
                    })
                    
                    # Update lifecycle
                    if operation in ['CREATE_TABLE']:
                        analysis.creates_files.add(table_name)
                    elif operation in ['SELECT']:
                        analysis.reads_files.add(table_name)
                    elif operation in ['INSERT', 'UPDATE']:
                        analysis.updates_files.add(table_name)
                    elif operation in ['DELETE', 'DROP_TABLE']:
                        analysis.deletes_files.add(table_name)
            
            # Column definitions
            col_match = re.search(r'(\w+)\s+(VARCHAR|CHAR|INTEGER|DECIMAL|DATE|TIMESTAMP)', line.upper())
            if col_match:
                col_name, data_type = col_match.groups()
                analysis.fields_defined.append({
                    'name': col_name,
                    'data_type': data_type,
                    'line_number': line_num,
                    'definition': line
                })
    
    def _store_component_analysis(self, analysis: ComponentAnalysis):
        """Store component analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store component
        content_hash = hashlib.md5(analysis.content.encode()).hexdigest()
        analysis_json = json.dumps({
            'operations': analysis.operations,
            'file_operations': analysis.file_operations,
            'data_operations': analysis.data_operations,
            'fields_defined': analysis.fields_defined,
            'fields_used': analysis.fields_used,
            'fields_modified': analysis.fields_modified,
            'fields_unused': analysis.fields_unused,
            'creates_files': list(analysis.creates_files),
            'reads_files': list(analysis.reads_files),
            'updates_files': list(analysis.updates_files),
            'deletes_files': list(analysis.deletes_files),
            'calls_programs': list(analysis.calls_programs),
            'business_functions': analysis.business_functions,
            'decision_points': analysis.decision_points,
            'calculations': analysis.calculations,
            'complexity_score': analysis.complexity_score,
            'lines_of_code': analysis.lines_of_code,
            'cyclomatic_complexity': analysis.cyclomatic_complexity
        })
        
        cursor.execute('''
            INSERT OR REPLACE INTO components 
            (name, component_type, subtype, content, content_hash, analysis_data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (analysis.name, analysis.component_type, analysis.subtype, 
              analysis.content, content_hash, analysis_json, datetime.now()))
        
        # Store operations
        cursor.execute('DELETE FROM operations WHERE component_name = ?', (analysis.name,))
        for op in analysis.file_operations + analysis.data_operations + analysis.operations:
            cursor.execute('''
                INSERT INTO operations 
                (component_name, operation_type, target_name, operation_details, line_number, context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (analysis.name, op.get('operation', op.get('type')), 
                  op.get('file_name', op.get('table_name', op.get('target'))),
                  json.dumps(op), op.get('line_number'), op.get('context')))
        
        # Store field lineage
        cursor.execute('DELETE FROM field_lineage WHERE component_name = ?', (analysis.name,))
        all_fields = analysis.fields_defined + analysis.fields_used + analysis.fields_modified
        for field in all_fields:
            cursor.execute('''
                INSERT INTO field_lineage 
                (component_name, field_name, action_type, line_number, context, field_definition, data_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (analysis.name, field['name'], field.get('action', 'defined'),
                  field.get('line_number'), field.get('context'), 
                  field.get('definition'), field.get('data_type')))
        
        conn.commit()
        conn.close()

class InteractiveAnalyzer:
    """Interactive chat-based component analysis"""
    
    def __init__(self, db_path: str = "mainframe_analysis.db"):
        self.db_path = db_path
    
    def get_component_analysis(self, component_name: str) -> Dict[str, Any]:
        """Get complete analysis for a specific component"""
        conn = sqlite3.connect(self.db_path)
        
        # Get component data
        component_query = '''
            SELECT name, component_type, subtype, analysis_data, content
            FROM components 
            WHERE name = ?
        '''
        component_df = pd.read_sql_query(component_query, conn, params=(component_name,))
        
        if component_df.empty:
            return {'error': f'Component {component_name} not found'}
        
        component_data = component_df.iloc[0]
        analysis_data = json.loads(component_data['analysis_data'])
        
        # Get operations
        operations_query = '''
            SELECT operation_type, target_name, line_number, context
            FROM operations 
            WHERE component_name = ?
            ORDER BY line_number
        '''
        operations_df = pd.read_sql_query(operations_query, conn, params=(component_name,))
        
        # Get field lineage
        fields_query = '''
            SELECT field_name, action_type, line_number, context, data_type
            FROM field_lineage 
            WHERE component_name = ?
            ORDER BY field_name, line_number
        '''
        fields_df = pd.read_sql_query(fields_query, conn, params=(component_name,))
        
        # Get dependencies
        deps_query = '''
            SELECT target_component, dependency_type, relationship_details
            FROM dependencies 
            WHERE source_component = ?
            UNION
            SELECT source_component, dependency_type, relationship_details
            FROM dependencies 
            WHERE target_component = ?
        '''
        deps_df = pd.read_sql_query(deps_query, conn, params=(component_name, component_name))
        
        conn.close()
        
        return {
            'component': {
                'name': component_data['name'],
                'type': component_data['component_type'],
                'subtype': component_data['subtype'],
                'content': component_data['content']
            },
            'analysis': analysis_data,
            'operations': operations_df.to_dict('records'),
            'fields': fields_df.to_dict('records'),
            'dependencies': deps_df.to_dict('records')
        }
    
    def get_lifecycle_analysis(self, file_name: str) -> Dict[str, Any]:
        """Get complete lifecycle analysis for a file"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                component_name,
                operation_type,
                line_number,
                context
            FROM operations 
            WHERE target_name = ?
            ORDER BY component_name, line_number
        '''
        
        df = pd.read_sql_query(query, conn, params=(file_name,))
        conn.close()
        
        if df.empty:
            return {'error': f'No operations found for file {file_name}'}
        
        lifecycle = {
            'file_name': file_name,
            'total_operations': len(df),
            'components_using': df['component_name'].nunique(),
            'operations_by_component': df.groupby('component_name')['operation_type'].apply(list).to_dict(),
            'operations_timeline': df.to_dict('records')
        }
        
        return lifecycle
    
    def get_field_lineage(self, field_name: str) -> Dict[str, Any]:
        """Get complete field lineage across components"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                component_name,
                action_type,
                line_number,
                context,
                data_type
            FROM field_lineage 
            WHERE field_name = ?
            ORDER BY component_name, line_number
        '''
        
        df = pd.read_sql_query(query, conn, params=(field_name,))
        conn.close()
        
        if df.empty:
            return {'error': f'No usage found for field {field_name}'}
        
        lineage = {
            'field_name': field_name,
            'total_usage': len(df),
            'components_using': df['component_name'].nunique(),
            'usage_by_component': df.groupby('component_name')['action_type'].apply(list).to_dict(),
            'usage_timeline': df.to_dict('records'),
            'data_types': df['data_type'].dropna().unique().tolist()
        }
        
        return lineage
    
    def chat_with_component(self, component_name: str, question: str) -> str:
        """Chat interface for component analysis"""
        analysis = self.get_component_analysis(component_name)
        
        if 'error' in analysis:
            return analysis['error']
        
        # Simple rule-based responses - can be enhanced with LLM
        question_lower = question.lower()
        
        if 'lifecycle' in question_lower or 'files' in question_lower:
            files_created = analysis['analysis'].get('creates_files', [])
            files_read = analysis['analysis'].get('reads_files', [])
            files_updated = analysis['analysis'].get('updates_files', [])
            
            response = f"**{component_name} File Lifecycle:**\n"
            if files_created:
                response += f"Creates: {', '.join(files_created)}\n"
            if files_read:
                response += f"Reads: {', '.join(files_read)}\n"
            if files_updated:
                response += f"Updates: {', '.join(files_updated)}\n"
            
            return response
        
        elif 'field' in question_lower or 'data' in question_lower:
            fields_defined = analysis['analysis'].get('fields_defined', [])
            fields_unused = analysis['analysis'].get('fields_unused', [])
            
            response = f"**{component_name} Field Analysis:**\n"
            response += f"Total fields defined: {len(fields_defined)}\n"
            response += f"Unused fields: {len(fields_unused)}\n"
            
            if fields_unused:
                response += f"Unused fields: {', '.join([f['name'] for f in fields_unused])}\n"
            
            return response
        
        elif 'complexity' in question_lower or 'metrics' in question_lower:
            complexity = analysis['analysis'].get('complexity_score', 0)
            loc = analysis['analysis'].get('lines_of_code', 0)
            cyclomatic = analysis['analysis'].get('cyclomatic_complexity', 0)
            
            response = f"**{component_name} Complexity Metrics:**\n"
            response += f"Lines of Code: {loc}\n"
            response += f"Cyclomatic Complexity: {cyclomatic}\n"
            response += f"Overall Complexity Score: {complexity:.2f}\n"
            
            return response
        
        elif 'business' in question_lower or 'logic' in question_lower:
            business_functions = analysis['analysis'].get('business_functions', [])
            calculations = analysis['analysis'].get('calculations', [])
            decisions = analysis['analysis'].get('decision_points', [])
            
            response = f"**{component_name} Business Logic:**\n"
            if business_functions:
                response += f"Functions: {', '.join(set(business_functions))}\n"
            response += f"Calculations: {len(calculations)}\n"
            response += f"Decision Points: {len(decisions)}\n"
            
            return response
        
        elif 'operation' in question_lower:
            file_ops = analysis['analysis'].get('file_operations', [])
            data_ops = analysis['analysis'].get('data_operations', [])
            
            response = f"**{component_name} Operations:**\n"
            response += f"File Operations: {len(file_ops)}\n"
            response += f"Data Operations: {len(data_ops)}\n"
            
            # Group operations by type
            op_types = {}
            for op in file_ops + data_ops:
                op_type = op.get('operation', 'unknown')
                op_types[op_type] = op_types.get(op_type, 0) + 1
            
            for op_type, count in op_types.items():
                response += f"{op_type}: {count}\n"
            
            return response
        
        else:
            # General overview
            component = analysis['component']
            analysis_data = analysis['analysis']
            
            response = f"**{component_name} Overview:**\n"
            response += f"Type: {component['type']} ({component['subtype']})\n"
            response += f"Lines of Code: {analysis_data.get('lines_of_code', 0)}\n"
            response += f"File Operations: {len(analysis_data.get('file_operations', []))}\n"
            response += f"Fields Defined: {len(analysis_data.get('fields_defined', []))}\n"
            response += f"Complexity Score: {analysis_data.get('complexity_score', 0):.2f}\n"
            
            return response

class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, db_path: str = "mainframe_analysis.db"):
        self.db_path = db_path
    
    def generate_master_files_report(self) -> pd.DataFrame:
        """Generate master files usage report"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                target_name as file_name,
                COUNT(DISTINCT component_name) as components_using,
                COUNT(*) as total_operations,
                GROUP_CONCAT(DISTINCT operation_type) as operation_types,
                GROUP_CONCAT(DISTINCT component_name) as components
            FROM operations 
            WHERE target_name IS NOT NULL
            GROUP BY target_name
            HAVING components_using > 1
            ORDER BY components_using DESC, total_operations DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def generate_component_summary_report(self) -> pd.DataFrame:
        """Generate component summary report"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                c.name,
                c.component_type,
                c.subtype,
                COUNT(DISTINCT o.target_name) as files_accessed,
                COUNT(o.id) as total_operations,
                COUNT(DISTINCT f.field_name) as fields_defined
            FROM components c
            LEFT JOIN operations o ON c.name = o.component_name
            LEFT JOIN field_lineage f ON c.name = f.component_name AND f.action_type = 'defined'
            GROUP BY c.name, c.component_type, c.subtype
            ORDER BY total_operations DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def generate_field_usage_report(self) -> pd.DataFrame:
        """Generate cross-component field usage report"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                field_name,
                COUNT(DISTINCT component_name) as components_using,
                COUNT(*) as total_usage,
                GROUP_CONCAT(DISTINCT action_type) as action_types,
                GROUP_CONCAT(DISTINCT component_name) as components,
                GROUP_CONCAT(DISTINCT data_type) as data_types
            FROM field_lineage
            GROUP BY field_name
            HAVING components_using > 1
            ORDER BY components_using DESC, total_usage DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def generate_unused_fields_report(self) -> pd.DataFrame:
        """Generate unused fields report"""
        conn = sqlite3.connect(self.db_path)
        
        # Find fields that are only defined but never used
        query = '''
            SELECT 
                component_name,
                field_name,
                data_type,
                line_number,
                context
            FROM field_lineage
            WHERE field_name NOT IN (
                SELECT DISTINCT field_name 
                FROM field_lineage 
                WHERE action_type IN ('read', 'write', 'compute')
            )
            AND action_type = 'defined'
            ORDER BY component_name, field_name
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def generate_complexity_report(self) -> pd.DataFrame:
        """Generate complexity analysis report"""
        conn = sqlite3.connect(self.db_path)
        
        # Extract complexity metrics from analysis_data JSON
        query = '''
            SELECT 
                name,
                component_type,
                subtype,
                json_extract(analysis_data, '$.complexity_score') as complexity_score,
                json_extract(analysis_data, '$.lines_of_code') as lines_of_code,
                json_extract(analysis_data, '$.cyclomatic_complexity') as cyclomatic_complexity,
                json_array_length(json_extract(analysis_data, '$.file_operations')) as file_operations,
                json_array_length(json_extract(analysis_data, '$.fields_defined')) as fields_defined
            FROM components
            ORDER BY complexity_score DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

# Streamlit UI for Interactive Analysis
def create_streamlit_ui():
    """Create Streamlit interface for batch processing and interactive analysis"""
    st.set_page_config(page_title="Mainframe Analysis Engine", layout="wide")
    
    st.title("🖥️ Mainframe Deep Research Agent")
    st.markdown("### Batch Processing & Interactive Component Analysis")
    
    # Initialize processors
    if 'batch_processor' not in st.session_state:
        st.session_state.batch_processor = BatchProcessor()
        st.session_state.interactive_analyzer = InteractiveAnalyzer()
        st.session_state.report_generator = ReportGenerator()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", [
        "Batch Processing", 
        "Component Analysis", 
        "Reports", 
        "Interactive Chat"
    ])
    
    if mode == "Batch Processing":
        st.header("📥 Batch Component Processing")
        
        uploaded_files = st.file_uploader(
            "Upload mainframe components (up to 50 files)",
            accept_multiple_files=True,
            type=['cbl', 'cob', 'cobol', 'jcl', 'job', 'sql', 'cpy', 'copy', 'txt']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            auto_detect = st.checkbox("Auto-detect component types", value=True)
        with col2:
            chunk_size = st.slider("Processing chunk size", 5, 50, 20)
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            if uploaded_files:
                with st.spinner("Processing components..."):
                    # Save uploaded files temporarily
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_files.append(temp_path)
                    
                    # Process in batch
                    results = st.session_state.batch_processor.batch_load_components(temp_files)
                    
                    # Clean up temp files
                    for temp_path in temp_files:
                        os.remove(temp_path)
                    
                    # Display results
                    st.success(f"✅ Processed {results['processed']} components successfully")
                    if results['errors'] > 0:
                        st.warning(f"⚠️ {results['errors']} components had errors")
                    
                    st.subheader("Processed Components")
                    for component in results['components']:
                        st.write(f"• {component}")
            else:
                st.warning("Please upload at least one component file")
    
    elif mode == "Component Analysis":
        st.header("🔍 Individual Component Analysis")
        
        # Get list of available components
        conn = sqlite3.connect(st.session_state.batch_processor.db_path)
        components_df = pd.read_sql_query(
            "SELECT name, component_type, subtype FROM components ORDER BY name", 
            conn
        )
        conn.close()
        
        if not components_df.empty:
            selected_component = st.selectbox(
                "Select Component to Analyze",
                components_df['name'].tolist()
            )
            
            if selected_component:
                analysis = st.session_state.interactive_analyzer.get_component_analysis(selected_component)
                
                if 'error' not in analysis:
                    component_data = analysis['component']
                    analysis_data = analysis['analysis']
                    
                    # Component Overview
                    st.subheader(f"📋 {selected_component} Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Type", f"{component_data['type']}")
                        st.metric("Subtype", f"{component_data['subtype']}")
                    with col2:
                        st.metric("Lines of Code", analysis_data.get('lines_of_code', 0))
                        st.metric("Complexity Score", f"{analysis_data.get('complexity_score', 0):.2f}")
                    with col3:
                        st.metric("File Operations", len(analysis_data.get('file_operations', [])))
                        st.metric("Fields Defined", len(analysis_data.get('fields_defined', [])))
                    with col4:
                        st.metric("Cyclomatic Complexity", analysis_data.get('cyclomatic_complexity', 0))
                        st.metric("Unused Fields", len(analysis_data.get('fields_unused', [])))
                    
                    # Detailed Analysis Tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "File Operations", "Field Analysis", "Business Logic", "Dependencies", "Source Code"
                    ])
                    
                    with tab1:
                        st.subheader("File Operations")
                        if analysis['operations']:
                            ops_df = pd.DataFrame(analysis['operations'])
                            st.dataframe(ops_df, use_container_width=True)
                        else:
                            st.info("No file operations found")
                    
                    with tab2:
                        st.subheader("Field Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Defined Fields**")
                            if analysis_data.get('fields_defined'):
                                fields_df = pd.DataFrame(analysis_data['fields_defined'])
                                st.dataframe(fields_df, use_container_width=True)
                        
                        with col2:
                            st.write("**Unused Fields**")
                            if analysis_data.get('fields_unused'):
                                unused_df = pd.DataFrame(analysis_data['fields_unused'])
                                st.dataframe(unused_df, use_container_width=True)
                            else:
                                st.success("No unused fields found")
                    
                    with tab3:
                        st.subheader("Business Logic")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Business Functions**")
                            functions = analysis_data.get('business_functions', [])
                            if functions:
                                function_counts = pd.Series(functions).value_counts()
                                st.bar_chart(function_counts)
                            else:
                                st.info("No business functions identified")
                        
                        with col2:
                            st.write("**Decision Points**")
                            decisions = analysis_data.get('decision_points', [])
                            if decisions:
                                st.write(f"Total decision points: {len(decisions)}")
                                decisions_df = pd.DataFrame(decisions)
                                st.dataframe(decisions_df, use_container_width=True)
                            else:
                                st.info("No decision points found")
                    
                    with tab4:
                        st.subheader("Dependencies")
                        if analysis['dependencies']:
                            deps_df = pd.DataFrame(analysis['dependencies'])
                            st.dataframe(deps_df, use_container_width=True)
                        else:
                            st.info("No dependencies found")
                    
                    with tab5:
                        st.subheader("Source Code")
                        st.code(component_data['content'], language='cobol')
                
                else:
                    st.error(analysis['error'])
        else:
            st.info("No components available. Please process some components first in Batch Processing mode.")
    
    elif mode == "Reports":
        st.header("📊 Analysis Reports")
        
        report_type = st.selectbox("Select Report Type", [
            "Master Files Usage",
            "Component Summary", 
            "Field Usage Across Components",
            "Unused Fields",
            "Complexity Analysis"
        ])
        
        if st.button("Generate Report", type="primary"):
            if report_type == "Master Files Usage":
                df = st.session_state.report_generator.generate_master_files_report()
                st.subheader("Master Files Usage Report")
                st.dataframe(df, use_container_width=True)
                
            elif report_type == "Component Summary":
                df = st.session_state.report_generator.generate_component_summary_report()
                st.subheader("Component Summary Report")
                st.dataframe(df, use_container_width=True)
                
            elif report_type == "Field Usage Across Components":
                df = st.session_state.report_generator.generate_field_usage_report()
                st.subheader("Cross-Component Field Usage Report")
                st.dataframe(df, use_container_width=True)
                
            elif report_type == "Unused Fields":
                df = st.session_state.report_generator.generate_unused_fields_report()
                st.subheader("Unused Fields Report")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    st.warning(f"Found {len(df)} potentially unused fields")
                else:
                    st.success("No unused fields found!")
                
            elif report_type == "Complexity Analysis":
                df = st.session_state.report_generator.generate_complexity_report()
                st.subheader("Complexity Analysis Report")
                st.dataframe(df, use_container_width=True)
                
                # Complexity visualization
                if not df.empty and 'complexity_score' in df.columns:
                    st.subheader("Complexity Distribution")
                    st.bar_chart(df.set_index('name')['complexity_score'])
    
    elif mode == "Interactive Chat":
        st.header("💬 Chat with Components")
        
        # Get list of available components
        conn = sqlite3.connect(st.session_state.batch_processor.db_path)
        components_df = pd.read_sql_query(
            "SELECT name, component_type, subtype FROM components ORDER BY name", 
            conn
        )
        conn.close()
        
        if not components_df.empty:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_component = st.selectbox(
                    "Select Component",
                    components_df['name'].tolist()
                )
                
                st.subheader("Quick Questions")
                if st.button("Show lifecycle"):
                    st.session_state.chat_question = "Show me the lifecycle and file operations"
                if st.button("Show fields"):
                    st.session_state.chat_question = "What fields are defined and which are unused?"
                if st.button("Show complexity"):
                    st.session_state.chat_question = "What is the complexity and metrics?"
                if st.button("Show business logic"):
                    st.session_state.chat_question = "What business logic and functions are implemented?"
            
            with col2:
                st.subheader(f"Chat about {selected_component}")
                
                # Chat input
                user_question = st.text_input(
                    "Ask about the component:",
                    value=st.session_state.get('chat_question', ''),
                    placeholder="e.g., What files does this program create? What fields are unused?"
                )
                
                if user_question:
                    response = st.session_state.interactive_analyzer.chat_with_component(
                        selected_component, user_question
                    )
                    st.markdown(response)
                    
                    # Clear the session state question
                    if 'chat_question' in st.session_state:
                        del st.session_state.chat_question
                
                # Additional analysis options
                st.subheader("Deep Dive Analysis")
                
                analysis_option = st.selectbox("Select Analysis Type", [
                    "File Lifecycle Analysis",
                    "Field Lineage Analysis"
                ])
                
                target_name = st.text_input("Enter file/field name:")
                
                if st.button("Analyze") and target_name:
                    if analysis_option == "File Lifecycle Analysis":
                        lifecycle = st.session_state.interactive_analyzer.get_lifecycle_analysis(target_name)
                        if 'error' not in lifecycle:
                            st.json(lifecycle)
                        else:
                            st.error(lifecycle['error'])
                    
                    elif analysis_option == "Field Lineage Analysis":
                        lineage = st.session_state.interactive_analyzer.get_field_lineage(target_name)
                        if 'error' not in lineage:
                            st.json(lineage)
                        else:
                            st.error(lineage['error'])
        else:
            st.info("No components available. Please process some components first.")

if __name__ == "__main__":
    create_streamlit_ui()