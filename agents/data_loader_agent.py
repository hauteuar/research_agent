# agents/data_loader_agent.py
"""
Agent 3: Batch Data Loader & Table Mapper
Handles DB2 DDL, DCLGEN files, CSV layouts, and data ingestion
"""

import asyncio
import sqlite3
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
import logging
from datetime import datetime as dt
import zipfile
import uuid
import hashlib
from agents.base_agent import BaseOpulenceAgent

from vllm import AsyncLLMEngine, SamplingParams

class DataLoaderAgent(BaseOpulenceAgent):
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
                 gpu_id: int = None, coordinator=None):
        super().__init__(coordinator, "data_loader", db_path, gpu_id)        # REMOVE: self.llm_engine = llm_engine  
        self._engine = None  # Cached engine reference (starts as None)
        
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        # NEW: Lazy loading tracking
        self._engine_loaded = False
        self._using_shared_engine = False
        
        # Initialize SQLite tables for data mapping
        self._init_data_tables()
    
    # NEW: Lazy loading engine getter
    async def get_engine(self):
        """Get LLM engine with lazy loading and sharing"""
        if self._engine is None and self.coordinator:
            try:
                # Get assigned GPU for data_loader agent type
                assigned_gpu = self.coordinator.agent_gpu_assignments.get("data_loader")
                if assigned_gpu is not None:
                    # Get shared engine from coordinator
                    self._engine = await self.coordinator.get_shared_llm_engine(assigned_gpu)
                    self.gpu_id = assigned_gpu
                    self._using_shared_engine = True
                    self._engine_loaded = True
                    self.logger.info(f"✅ DataLoader using shared engine on GPU {assigned_gpu}")
                else:
                    raise ValueError("No GPU assigned for data_loader agent type")
            except Exception as e:
                self.logger.error(f"❌ Failed to get shared engine: {e}")
                raise
        
        return self._engine

    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM - lazy loading version"""
        try:
            # LAZY LOAD: Get engine only when needed
            engine = await self.get_engine()
            if engine is None:
                raise RuntimeError("No LLM engine available")

            await asyncio.sleep(0.1)

            # Try new API first (with request_id)
            request_id = str(uuid.uuid4())
            try:
                async for output in engine.generate(prompt, sampling_params, request_id=request_id):
                    result = output
                    break
                return result.outputs[0].text.strip()
            except TypeError as e:
                if "request_id" in str(e):
                    # Fallback to old API (without request_id)
                    async for output in engine.generate(prompt, sampling_params):
                        result = output
                        break
                    return result.outputs[0].text.strip()
                else:
                    raise e
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""
    
    # ADD this helper method
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'data_loader'
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_loaded_lazily'] = self._engine_loaded
        return result
    
    def _init_data_tables(self):
        """Initialize SQLite tables for data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS table_schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT UNIQUE,
                schema_type TEXT,  -- 'db2_ddl', 'dclgen', 'csv_layout', 'user_defined'
                field_definitions TEXT,  -- JSON array of field definitions
                source_file TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS data_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                sample_data TEXT,  -- JSON array of sample records
                record_count INTEGER,
                data_quality_score REAL,
                source_file TEXT,
                loaded_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (table_name) REFERENCES table_schemas (table_name)
            );
            
            CREATE TABLE IF NOT EXISTS field_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                field_name TEXT,
                field_type TEXT,
                field_length INTEGER,
                field_precision INTEGER,
                is_nullable BOOLEAN,
                default_value TEXT,
                field_description TEXT,
                business_meaning TEXT,
                data_classification TEXT,  -- 'PII', 'Financial', 'Operational', etc.
                FOREIGN KEY (table_name) REFERENCES table_schemas (table_name)
            );
            
            CREATE TABLE IF NOT EXISTS record_layouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layout_name TEXT UNIQUE,
                record_types TEXT,  -- JSON array of record type definitions
                layout_description TEXT,
                source_file TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        conn.close()
    
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a data file (CSV, DDL, DCLGEN, layout)"""
        try:
            #await self._ensure_llm_engine()
            file_extension = file_path.suffix.lower()
            file_content = self._read_file_safely(file_path)
            
            if file_extension == '.csv':
                result = await self._process_csv_file(file_path, file_content)
            elif file_extension in ['.ddl', '.sql']:
                result = await self._process_ddl_file(file_path, file_content)
            elif 'dclgen' in file_path.name.lower() or file_extension == '.dcl':
                result = await self._process_dclgen_file(file_path, file_content)
            elif file_extension in ['.cpy', '.copy']:  # ADD THIS
                result = await self._process_copybook_file(file_path, file_content)  # ADD THIS
            elif file_extension == '.json':
                result = await self._process_layout_json(file_path, file_content)
            elif file_extension == '.zip':
                result = await self._process_zip_file(file_path)
            else:
                # Try to auto-detect file type
                result = await self._auto_detect_and_process(file_path, file_content)
            
            # CHANGE: Use helper method instead of direct return
            return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {str(e)}")
            return self._add_processing_info({
                "status": "error",
                "file_name": file_path.name,
                "error": str(e)
            })
    
    def _read_file_safely(self, file_path: Path) -> str:
        """Safely read file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Fallback to binary mode and ignore errors
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    async def _process_csv_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Process CSV file and create corresponding table"""
        try:
            #await self._ensure_llm_engine()
            # Detect delimiter and structure
            delimiter = self._detect_csv_delimiter(content)
            
            # Read CSV with pandas
            df = pd.read_csv(file_path, delimiter=delimiter, nrows=1000)  # Sample first 1000 rows
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Infer data types and generate schema
            schema = await self._infer_csv_schema(df, file_path.name)
            
            # Create SQLite table
            table_name = self._generate_table_name(file_path.name)
            await self._create_sqlite_table(table_name, schema)
            
            # Load sample data
            sample_size = min(1000, len(df))
            sample_df = df.head(sample_size)
            
            # Store in SQLite
            await self._store_table_schema(table_name, 'csv_layout', schema, file_path.name)
            await self._store_sample_data(table_name, sample_df, file_path.name)
            await self._store_csv_as_chunks(table_name, schema, sample_df, file_path.name)
        
            
            # Analyze data quality
            quality_score = await self._analyze_data_quality(sample_df)
            
            # Generate field descriptions using LLM
            field_descriptions = await self._generate_field_descriptions(df.columns.tolist(), sample_df)
            
            return {
                "status": "success",
                "file_name": file_path.name,
                "table_name": table_name,
                "schema": schema,
                "record_count": len(df),
                "sample_count": sample_size,
                "data_quality_score": quality_score,
                "field_descriptions": field_descriptions,
                "chunks_created": len(schema) + 1 
            }
            
        except Exception as e:
            self.logger.error(f"CSV processing failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _detect_csv_delimiter(self, content: str) -> str:
        """Detect CSV delimiter"""
        sample_lines = content.split('\n')[:5]
        sample_text = '\n'.join(sample_lines)
        
        delimiters = [',', '|', '\t', ';', ':']
        delimiter_counts = {}
        
        for delimiter in delimiters:
            count = sample_text.count(delimiter)
            if count > 0:
                # Check consistency across lines
                line_counts = [line.count(delimiter) for line in sample_lines if line.strip()]
                if line_counts and len(set(line_counts)) <= 2:  # Allow some variation
                    delimiter_counts[delimiter] = count
        
        if delimiter_counts:
            return max(delimiter_counts, key=delimiter_counts.get)
        
        return ','  # Default to comma
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean column name for SQLite compatibility"""
        # Remove special characters and spaces, replace hyphens with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col_name))
        clean_name = re.sub(r'_{2,}', '_', clean_name)  # Replace multiple underscores
        clean_name = clean_name.strip('_')
        
        # Ensure it starts with letter or underscore
        if clean_name and clean_name[0].isdigit():
            clean_name = 'col_' + clean_name
        
        return clean_name or 'unnamed_column'

    
    async def _infer_csv_schema(self, df: pd.DataFrame, filename: str) -> List[Dict[str, Any]]:
        """Infer schema from CSV data using LLM assistance"""
        schema = []
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            # Basic type inference
            if pd.api.types.is_numeric_dtype(col_data):
                if pd.api.types.is_integer_dtype(col_data):
                    sql_type = "INTEGER"
                else:
                    sql_type = "REAL"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                sql_type = "TIMESTAMP"
            else:
                # String type - determine length
                max_length = col_data.astype(str).str.len().max()
                if pd.isna(max_length):
                    max_length = 255
                sql_type = f"VARCHAR({min(int(max_length * 1.2), 1000)})"
            
            # Get sample values for LLM analysis
            sample_values = col_data.head(10).tolist()
            
            field_info = {
                "name": column,
                "type": sql_type,
                "nullable": df[column].isnull().any(),
                "sample_values": sample_values,
                "unique_count": df[column].nunique(),
                "null_count": df[column].isnull().sum()
            }
            
            schema.append(field_info)
        
        # Use LLM to enhance schema with business context
        enhanced_schema = await self._enhance_schema_with_llm(schema, filename)
        
        return enhanced_schema
    
    async def _store_csv_as_chunks(self, table_name: str, schema: List[Dict], 
                                sample_df: pd.DataFrame, source_file: str):
        """Store CSV information as chunks for vector indexing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table structure chunk
            table_chunk = {
                "program_name": table_name,
                "chunk_id": f"{table_name}_TABLE_STRUCTURE", 
                "chunk_type": "csv_structure",
                "content": f"Table: {table_name}\nColumns: {', '.join([f['name'] for f in schema])}",
                "metadata": json.dumps({
                    "table_name": table_name,
                    "column_count": len(schema),
                    "record_count": len(sample_df),
                    "source_file": source_file,
                    "file_type": "csv"
                })
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO program_chunks 
                (program_name, chunk_id, chunk_type, content, metadata, line_start, line_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                table_chunk["program_name"],
                table_chunk["chunk_id"],
                table_chunk["chunk_type"], 
                table_chunk["content"],
                table_chunk["metadata"],
                0, 1
            ))
            
            # Create field chunks
            for i, field in enumerate(schema):
                field_content = f"Field: {field['name']}, Type: {field['type']}"
                if field.get('sample_values'):
                    field_content += f", Sample: {field['sample_values'][:3]}"
                    
                field_chunk = {
                    "program_name": table_name,
                    "chunk_id": f"{table_name}_FIELD_{field['name']}",
                    "chunk_type": "csv_field",
                    "content": field_content,
                    "metadata": json.dumps({
                        "field_name": field['name'],
                        "field_type": field['type'], 
                        "nullable": field.get('nullable', True),
                        "sample_values": field.get('sample_values', [])[:5],
                        "table_name": table_name,
                        "source_file": source_file,
                        "file_type": "csv"
                    })
                }
                
                cursor.execute("""
                    INSERT OR REPLACE INTO program_chunks 
                    (program_name, chunk_id, chunk_type, content, metadata, line_start, line_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    field_chunk["program_name"],
                    field_chunk["chunk_id"],
                    field_chunk["chunk_type"],
                    field_chunk["content"], 
                    field_chunk["metadata"],
                    i + 1, i + 2
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store CSV chunks: {str(e)}")
    
    async def _enhance_schema_with_llm(self, schema: List[Dict], filename: str) -> List[Dict[str, Any]]:
        """Use LLM to enhance schema with business context using chunking"""
        #await self._ensure_llm_engine()
        
        # Process schema in chunks if it's large
        if len(schema) > 10:
            schema_chunks = self._chunk_field_content(schema, max_fields_per_chunk=8)
            enhanced_schemas = []
            
            for chunk in schema_chunks:
                chunk_result = await self._enhance_schema_chunk(chunk, filename)
                enhanced_schemas.extend(chunk_result)
            
            return enhanced_schemas
        else:
            return await self._enhance_schema_chunk(schema, filename)
    
    async def _enhance_schema_chunk(self, schema_chunk: List[Dict], filename: str) -> List[Dict[str, Any]]:
        """Enhance a single chunk of schema fields"""
        prompt = f"""
        Analyze this database schema for file "{filename}" and enhance it with business context:
        
        Schema:
        {json.dumps(schema_chunk, indent=2)}
        
        For each field, provide:
        1. Business meaning/purpose
        2. Data classification (PII, Financial, Operational, Reference, etc.)
        3. Potential data quality issues
        4. Suggested improvements to field definition
        
        Return enhanced schema as JSON with additional fields:
        - business_meaning
        - data_classification  
        - quality_concerns
        - recommended_type (if different from current)
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1500)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('[') if '[' in response_text else response_text.find('{')
                json_end = response_text.rfind(']') + 1 if ']' in response_text else response_text.rfind('}') + 1
                enhanced_data = json.loads(response_text[json_start:json_end])
                
                # Merge enhanced data with original schema
                if isinstance(enhanced_data, list) and len(enhanced_data) == len(schema_chunk):
                    for i, enhancement in enumerate(enhanced_data):
                        if isinstance(enhancement, dict):
                            schema_chunk[i].update(enhancement)
                
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM schema enhancement: {str(e)}")
        
        return schema_chunk

    async def _parse_ddl_statements(self, ddl_content: str) -> List[Dict[str, Any]]:
        """Parse DDL statements to extract table definitions - FIXED"""
        #await self._ensure_llm_engine()
        tables = []
        
        # Use LLM to parse complex DDL
        prompt = f"""
        Parse this DB2 DDL and extract table definitions:
        
        {ddl_content}
        
        For each CREATE TABLE statement, extract:
        1. Table name
        2. Column definitions with types and constraints
        3. Primary keys
        4. Foreign keys
        
        Return as JSON array:
        [{{
            "table_name": "TABLE_NAME",
            "ddl_type": "CREATE TABLE",
            "columns": [
                {{"name": "COLUMN1", "type": "VARCHAR(50)", "nullable": true, "primary_key": false}},
                {{"name": "COLUMN2", "type": "INTEGER", "nullable": false, "primary_key": true}}
            ],
            "primary_keys": ["COLUMN2"],
            "foreign_keys": []
        }}]
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2000)
        
        try:
            # FIX: Use the helper method instead of direct LLM call
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '[' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                tables = json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"LLM DDL parsing failed, using regex fallback: {str(e)}")
            tables = self._parse_ddl_with_regex(ddl_content)
        
        return tables

    
    
    def _generate_table_name(self, filename: str) -> str:
        """Generate SQLite table name from filename"""
        base_name = Path(filename).stem
        # Replace hyphens with underscores for SQL compatibility
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
        table_name = re.sub(r'_{2,}', '_', table_name)
        table_name = table_name.lower().strip('_')
        
        # Ensure it starts with letter or underscore
        if table_name and table_name[0].isdigit():
            table_name = 'tbl_' + table_name
            
        return table_name or 'unknown_table'
    
    async def _create_sqlite_table(self, table_name: str, schema: List[Dict[str, Any]]):
        """Create SQLite table with enhanced COBOL field handling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            clean_table_name = self._generate_table_name(table_name)
            
            # Drop table if exists
            cursor.execute(f"DROP TABLE IF EXISTS `{clean_table_name}`")
            
            # Filter out REDEFINES fields and prepare columns
            columns = []
            for field in schema:
                # Skip REDEFINES fields (they overlay existing fields)
                if field.get('redefines'):
                    continue
                    
                field_name = field['name']
                field_type = field['type']
                
                # Clean field name if needed
                if '-' in field_name or ' ' in field_name:
                    field_name = self._clean_column_name(field_name)
                
                column_def = f"`{field_name}` {field_type}"
                
                # Add constraints
                if not field.get('nullable', True):
                    column_def += " NOT NULL"
                
                # Add default values
                if field.get('default_value'):
                    default_val = field['default_value']
                    if field_type.startswith('VARCHAR') or field_type.startswith('CHAR'):
                        column_def += f" DEFAULT '{default_val}'"
                    else:
                        column_def += f" DEFAULT {default_val}"
                
                columns.append(column_def)
            
            if not columns:
                raise ValueError(f"No valid columns found for table {clean_table_name}")
            
            create_sql = f"CREATE TABLE `{clean_table_name}` ({', '.join(columns)})"
            self.logger.info(f"Creating table with SQL: {create_sql}")
            cursor.execute(create_sql)
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to create table {table_name}: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def _store_table_schema(self, table_name: str, schema_type: str, 
                                schema: List[Dict], source_file: str):
        """Store table schema metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO table_schemas 
            (table_name, schema_type, field_definitions, source_file, last_modified)
            VALUES (?, ?, ?, ?, ?)
        """, (table_name, schema_type, json.dumps(schema), source_file, dt.now()))
        
        # Store individual field information
        cursor.execute("DELETE FROM field_catalog WHERE table_name = ?", (table_name,))
        
        for field in schema:
            cursor.execute("""
                INSERT INTO field_catalog 
                (table_name, field_name, field_type, is_nullable, field_description, 
                 business_meaning, data_classification)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name, field['name'], field['type'], field.get('nullable', True),
                field.get('description', ''), field.get('business_meaning', ''),
                field.get('data_classification', '')
            ))
        
        conn.commit()
        conn.close()
    
    async def _store_sample_data(self, table_name: str, df: pd.DataFrame, source_file: str):
        """Store sample data in both metadata and actual table"""
        conn = sqlite3.connect(self.db_path)
        
        # Store sample records as JSON metadata
        sample_records = df.head(100).to_dict('records')  # First 100 records
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO data_samples 
            (table_name, sample_data, record_count, source_file)
            VALUES (?, ?, ?, ?)
        """, (table_name, json.dumps(sample_records, default=str), len(df), source_file))
        
        # Insert actual data into the table
        try:
            df.to_sql(table_name, conn, if_exists='append', index=False)
        except Exception as e:
            self.logger.warning(f"Failed to insert data into {table_name}: {str(e)}")
        
        conn.commit()
        conn.close()
    
    async def _analyze_data_quality(self, df: pd.DataFrame) -> float:
        """Analyze data quality and return score 0-1"""
        try:
            total_cells = df.size
            if total_cells == 0:
                return 0.0
            
            # Calculate quality metrics
            null_cells = df.isnull().sum().sum()
            null_ratio = null_cells / total_cells
            
            # Check for duplicate rows
            duplicate_ratio = df.duplicated().sum() / len(df)
            
            # Check for data consistency
            consistency_scores = []
            for column in df.select_dtypes(include=['object']).columns:
                unique_values = df[column].nunique()
                total_values = df[column].count()
                if total_values > 0:
                    consistency_scores.append(min(unique_values / total_values, 1.0))
            
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
            
            # Combined quality score
            quality_score = (
                (1 - null_ratio) * 0.4 +           # 40% weight for completeness
                (1 - duplicate_ratio) * 0.3 +       # 30% weight for uniqueness
                avg_consistency * 0.3                # 30% weight for consistency
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Data quality analysis failed: {str(e)}")
            return 0.5  # Default score
    
    async def _generate_field_descriptions(self, columns: List[str], sample_df: pd.DataFrame) -> Dict[str, str]:
        """Generate field descriptions using LLM - FIXED"""
        #await self._ensure_llm_engine()
        descriptions = {}
        
        # Process columns in batches
        batch_size = 10
        for i in range(0, len(columns), batch_size):
            batch_columns = columns[i:i + batch_size]
            
            # Prepare sample data for this batch
            batch_samples = {}
            for col in batch_columns:
                if col in sample_df.columns:
                    sample_values = sample_df[col].dropna().head(5).tolist()
                    batch_samples[col] = sample_values
            
            prompt = f"""
            Analyze these database fields and provide business descriptions:
            
            Fields with sample data:
            {json.dumps(batch_samples, indent=2, default=str)}
            
            For each field, provide a concise business description (1-2 sentences) that explains:
            1. What the field represents
            2. Its likely business purpose
            3. Data format if apparent
            
            Return as JSON: {{"field_name": "description", ...}}
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=800)
            
            try:
                # FIX: Use the helper method instead of direct LLM call
                response_text = await self._generate_with_llm(prompt, sampling_params)
                if '{' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    batch_descriptions = json.loads(response_text[json_start:json_end])
                    descriptions.update(batch_descriptions)
            except Exception as e:
                self.logger.warning(f"Failed to parse field descriptions: {str(e)}")
                # Fallback descriptions
                for col in batch_columns:
                    descriptions[col] = f"Data field: {col}"
        
        return descriptions

    async def _process_ddl_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Process DB2 DDL file"""
        try:
            # Parse DDL statements
            tables = await self._parse_ddl_statements(content)
            
            results = []
            for table_info in tables:
                table_name = table_info['table_name']
                schema = table_info['columns']
                
                # Create SQLite table
                await self._create_sqlite_table(table_name, schema)
                
                # Store schema metadata
                await self._store_table_schema(table_name, 'db2_ddl', schema, file_path.name)
                
                results.append({
                    "table_name": table_name,
                    "column_count": len(schema),
                    "ddl_type": table_info.get('ddl_type', 'CREATE TABLE')
                })
            
            return {
                "status": "success",
                "file_name": file_path.name,
                "tables_created": len(tables),
                "table_details": results
            }
            
        except Exception as e:
            self.logger.error(f"DDL processing failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _parse_ddl_statements(self, ddl_content: str) -> List[Dict[str, Any]]:
        """Parse DDL statements to extract table definitions"""
        #await self._ensure_llm_engine()
        tables = []
        
        # Use LLM to parse complex DDL
        prompt = f"""
        Parse this DB2 DDL and extract table definitions:
        
        {ddl_content}
        
        For each CREATE TABLE statement, extract:
        1. Table name
        2. Column definitions with types and constraints
        3. Primary keys
        4. Foreign keys
        
        Return as JSON array:
        [{{
            "table_name": "TABLE_NAME",
            "ddl_type": "CREATE TABLE",
            "columns": [
                {{"name": "COLUMN1", "type": "VARCHAR(50)", "nullable": true, "primary_key": false}},
                {{"name": "COLUMN2", "type": "INTEGER", "nullable": false, "primary_key": true}}
            ],
            "primary_keys": ["COLUMN2"],
            "foreign_keys": []
        }}]
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2000)
        #result = await self.llm_engine.generate(prompt, sampling_params)
        async for output in self.llm_engine.generate(prompt, sampling_params):
            result = output
            break

        try:
            response_text = result.outputs[0].text.strip()
            if '[' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                tables = json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"LLM DDL parsing failed, using regex fallback: {str(e)}")
            tables = self._parse_ddl_with_regex(ddl_content)
        
        return tables
    
    def _parse_ddl_with_regex(self, ddl_content: str) -> List[Dict[str, Any]]:
        """Fallback DDL parsing using regex"""
        tables = []
        
        # Simple regex patterns for basic DDL parsing
        table_pattern = re.compile(r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)', re.DOTALL | re.IGNORECASE)
        
        for match in table_pattern.finditer(ddl_content):
            table_name = match.group(1)
            columns_text = match.group(2)
            
            columns = []
            # Parse column definitions
            column_lines = [line.strip() for line in columns_text.split(',') if line.strip()]
            
            for line in column_lines:
                # Basic column parsing
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0]
                    col_type = parts[1]
                    nullable = 'NOT NULL' not in line.upper()
                    
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "nullable": nullable,
                        "primary_key": False
                    })
            
            if columns:
                tables.append({
                    "table_name": table_name,
                    "ddl_type": "CREATE TABLE",
                    "columns": columns,
                    "primary_keys": [],
                    "foreign_keys": []
                })
        
        return tables
    
    async def _process_dclgen_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Process DCLGEN file"""
        try:
            # Parse DCLGEN structure
            dclgen_info = await self._parse_dclgen_structure(content, file_path.name)
            
            if dclgen_info:
                table_name = dclgen_info['table_name']
                schema = dclgen_info['columns']
                
                # Create SQLite table
                await self._create_sqlite_table(table_name, schema)
                
                # Store schema metadata
                await self._store_table_schema(table_name, 'dclgen', schema, file_path.name)
                
                return {
                    "status": "success",
                    "file_name": file_path.name,
                    "table_name": table_name,
                    "column_count": len(schema),
                    "host_variables": dclgen_info.get('host_variables', [])
                }
            else:
                return {"status": "error", "error": "Could not parse DCLGEN structure"}
                
        except Exception as e:
            self.logger.error(f"DCLGEN processing failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _parse_dclgen_structure(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """Parse DCLGEN structure using LLM - FIXED"""
        #await self._ensure_llm_engine()
        prompt = f"""
        Parse this DB2 DCLGEN file and extract the table structure:
        
        {content}
        
        Extract:
        1. Table name (from DECLARE statement)
        2. Host variable definitions
        3. Corresponding column information
        
        Return as JSON:
        {{
            "table_name": "TABLE_NAME",
            "columns": [
                {{"name": "COLUMN1", "type": "VARCHAR(50)", "nullable": true, "cobol_definition": "01 HOST-VAR1 PIC X(50)."}},
                {{"name": "COLUMN2", "type": "INTEGER", "nullable": false, "cobol_definition": "01 HOST-VAR2 PIC S9(9) COMP."}}
            ],
            "host_variables": ["HOST-VAR1", "HOST-VAR2"]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=1500)
        
        try:
            # FIX: Use the helper method instead of direct LLM call
            response_text = await self._generate_with_llm(prompt, sampling_params)
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"DCLGEN parsing failed: {str(e)}")
        
        return None

    
    async def _process_layout_json(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Process JSON layout file"""
        try:
            layout_data = json.loads(content)
            
            if 'record_layouts' in layout_data:
                return await self._process_record_layouts(layout_data, file_path.name)
            elif 'table_schema' in layout_data:
                return await self._process_table_schema_json(layout_data, file_path.name)
            else:
                return {"status": "error", "error": "Unknown JSON layout format"}
                
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {str(e)}"}
    
    async def _process_record_layouts(self, layout_data: Dict, filename: str) -> Dict[str, Any]:
        """Process record layout definitions"""
        try:
            layouts_processed = []
            
            for layout in layout_data['record_layouts']:
                layout_name = layout['name']
                record_types = layout.get('record_types', [])
                
                # Store record layout
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO record_layouts 
                    (layout_name, record_types, layout_description, source_file)
                    VALUES (?, ?, ?, ?)
                """, (
                    layout_name,
                    json.dumps(record_types),
                    layout.get('description', ''),
                    filename
                ))
                
                conn.commit()
                conn.close()
                
                layouts_processed.append({
                    "layout_name": layout_name,
                    "record_type_count": len(record_types)
                })
            
            return {
                "status": "success",
                "file_name": filename,
                "layouts_processed": len(layouts_processed),
                "layout_details": layouts_processed
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _process_zip_file(self, file_path: Path) -> Dict[str, Any]:
        """Process ZIP file containing multiple data files"""
        try:
            results = []
            
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if not file_info.is_dir():
                        # Extract and process each file
                        extracted_content = zip_file.read(file_info.filename)
                        
                        # Create temporary file path
                        temp_path = Path(file_info.filename)
                        
                        # Write content to process
                        try:
                            content = extracted_content.decode('utf-8')
                        except UnicodeDecodeError:
                            content = extracted_content.decode('latin-1')
                        
                        # Process the extracted file
                        result = await self._auto_detect_and_process(temp_path, content)
                        result['source_zip'] = file_path.name
                        results.append(result)
            
            return {
                "status": "success",
                "zip_file": file_path.name,
                "files_processed": len(results),
                "results": results
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _auto_detect_and_process(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Auto-detect file type and process accordingly"""
        content_upper = content.upper()
        
        if 'CREATE TABLE' in content_upper:
            return await self._process_ddl_file(file_path, content)
        elif 'DECLARE' in content_upper and 'TABLE' in content_upper:
            return await self._process_dclgen_file(file_path, content)
        elif re.search(r'^\s*\d+\s+[A-Z][A-Z0-9-]*\s+PIC', content_upper, re.MULTILINE):  # ADD THIS
        # Looks like a COBOL copybook
            return await self._process_copybook_file(file_path, content)
        elif ',' in content and '\n' in content:
            # Likely CSV
            return await self._process_csv_file(file_path, content)
        else:
            return {
                "status": "unknown_format",
                "file_name": file_path.name,
                "message": "Could not determine file format"
            }
    
    async def _process_copybook_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Process COBOL copybook file with enhanced debugging"""
        try:
            self.logger.info(f"🔍 Processing copybook: {file_path.name}")
            self.logger.debug(f"📄 Content preview (first 500 chars): {content[:500]}")
            
            # Enhanced debugging: Check content format
            lines = content.split('\n')
            self.logger.info(f"📊 Total lines: {len(lines)}")
            
            # Check for COBOL patterns
            cobol_patterns = {
                'level_numbers': len([line for line in lines if re.match(r'^\s*\d+\s+', line)]),
                'pic_clauses': len([line for line in lines if 'PIC' in line.upper()]),
                'field_names': len([line for line in lines if re.match(r'^\s*\d+\s+[A-Z]', line)]),
                'filler_fields': len([line for line in lines if 'FILLER' in line.upper()])
            }
            
            self.logger.info(f"📋 COBOL patterns found: {cobol_patterns}")
            
            # If no COBOL patterns found, it might not be a copybook
            if cobol_patterns['level_numbers'] == 0:
                return {
                    "status": "error", 
                    "error": "No COBOL level numbers found - file may not be a copybook",
                    "file_name": file_path.name,
                    "debug_info": {
                        "content_preview": content[:200],
                        "patterns_found": cobol_patterns
                    }
                }
            
            # Parse copybook structure using LLM with fallback
            copybook_info = await self._parse_copybook_structure_with_fallback(content, file_path.name)
            
            if copybook_info:
                table_name = copybook_info['record_name']
                schema = copybook_info['fields']
                
                self.logger.info(f"✅ Parsed copybook: {table_name} with {len(schema)} fields")
                
                # Create SQLite table from copybook fields
                await self._create_sqlite_table(table_name, schema)
                
                # Store schema metadata
                await self._store_table_schema(table_name, 'copybook', schema, file_path.name)
                
                # Store as chunks for vector indexing
                await self._store_copybook_as_chunks(table_name, copybook_info, content, file_path.name)
                
                return {
                    "status": "success",
                    "file_name": file_path.name,
                    "table_name": table_name,
                    "field_count": len(schema),
                    "record_structure": copybook_info.get('structure_type', 'flat'),
                    "chunks_created": len(schema) + 1,
                    "debug_info": {
                        "parsing_method": copybook_info.get('parsing_method', 'llm'),
                        "patterns_found": cobol_patterns
                    }
                }
            else:
                return {
                    "status": "error", 
                    "error": "Could not parse copybook structure with any method",
                    "file_name": file_path.name,
                    "debug_info": {
                        "content_preview": content[:200],
                        "patterns_found": cobol_patterns,
                        "suggested_fixes": [
                            "Check if file is a valid COBOL copybook",
                            "Verify level numbers are present (01, 05, 10, etc.)",
                            "Ensure PIC clauses are formatted correctly",
                            "Check for proper COBOL field syntax"
                        ]
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Copybook processing failed: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "file_name": file_path.name,
                "debug_info": {
                    "exception_type": type(e).__name__,
                    "content_length": len(content)
                }
            }

    async def _parse_copybook_structure_with_fallback(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """Parse copybook with multiple fallback methods"""
        
        # Method 1: Try LLM parsing first
        try:
            self.logger.info("🤖 Attempting LLM-based copybook parsing...")
            llm_result = await self._parse_copybook_structure_llm(content, filename)
            if llm_result and llm_result.get('fields'):
                llm_result['parsing_method'] = 'llm'
                return llm_result
        except Exception as e:
            self.logger.warning(f"LLM parsing failed: {str(e)}")
        
        # Method 2: Enhanced regex parsing
        try:
            self.logger.info("🔍 Attempting enhanced regex parsing...")
            regex_result = self._parse_copybook_with_enhanced_regex(content, filename)
            if regex_result and regex_result.get('fields'):
                regex_result['parsing_method'] = 'enhanced_regex'
                return regex_result
        except Exception as e:
            self.logger.warning(f"Enhanced regex parsing failed: {str(e)}")
        
        # Method 3: Simple regex fallback
        try:
            self.logger.info("📝 Attempting simple regex parsing...")
            simple_result = self._parse_copybook_with_simple_regex(content, filename)
            if simple_result and simple_result.get('fields'):
                simple_result['parsing_method'] = 'simple_regex'
                return simple_result
        except Exception as e:
            self.logger.warning(f"Simple regex parsing failed: {str(e)}")
        
        return None

    # ADD THIS NEW METHOD
    async def _store_copybook_as_chunks(self, table_name: str, copybook_info: Dict, 
                                  original_content: str, source_file: str):
        """Store copybook information as chunks for vector indexing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean table name for consistency
            clean_table_name = self._generate_table_name(table_name)
            
            # Create overall record structure chunk
            record_chunk = {
                "program_name": clean_table_name,
                "chunk_id": f"{clean_table_name}_RECORD_STRUCTURE",
                "chunk_type": "copybook_structure",
                "content": original_content,
                "metadata": json.dumps({
                    "record_name": copybook_info['record_name'],
                    "structure_type": copybook_info.get('structure_type', 'flat'),
                    "total_fields": len(copybook_info['fields']),
                    "source_file": source_file,
                    "file_type": "copybook"
                }),
                "line_start": 0,
                "line_end": len(original_content.split('\n'))
            }
            
            # Insert record structure chunk
            cursor.execute("""
                INSERT OR REPLACE INTO program_chunks 
                (program_name, chunk_id, chunk_type, content, metadata, line_start, line_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record_chunk["program_name"],
                record_chunk["chunk_id"], 
                record_chunk["chunk_type"],
                record_chunk["content"],
                record_chunk["metadata"],
                record_chunk["line_start"],
                record_chunk["line_end"]
            ))
            
            # Create individual field chunks
            for i, field in enumerate(copybook_info['fields']):
                # Use cleaned field name
                field_name = self._clean_column_name(field['name'])
                
                field_chunk = {
                    "program_name": clean_table_name,
                    "chunk_id": f"{clean_table_name}_FIELD_{field_name}",
                    "chunk_type": "copybook_field",
                    "content": f"{field['level']} {field_name} {field.get('pic_clause', '')}",
                    "metadata": json.dumps({
                        "field_name": field_name,
                        "original_field_name": field.get('original_name', field_name),
                        "field_type": field['type'],
                        "level": field.get('level', '05'),
                        "pic_clause": field.get('pic_clause', ''),
                        "nullable": field.get('nullable', True),
                        "record_name": copybook_info['record_name'],
                        "source_file": source_file,
                        "file_type": "copybook"
                    }),
                    "line_start": i,
                    "line_end": i + 1
                }
                
                cursor.execute("""
                    INSERT OR REPLACE INTO program_chunks 
                    (program_name, chunk_id, chunk_type, content, metadata, line_start, line_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    field_chunk["program_name"],
                    field_chunk["chunk_id"],
                    field_chunk["chunk_type"], 
                    field_chunk["content"],
                    field_chunk["metadata"],
                    field_chunk["line_start"],
                    field_chunk["line_end"]
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored {len(copybook_info['fields']) + 1} chunks for copybook {clean_table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store copybook chunks: {str(e)}")
            raise
    
        
    
    async def _parse_copybook_chunked(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """Parse large copybook files by processing them in chunks"""
        try:
            self.logger.info(f"📦 Processing large copybook {filename} in chunks...")
            
            # Split content into logical chunks (preserve record structure)
            chunks = self._chunk_copybook_by_structure(content)
            
            if not chunks:
                self.logger.warning("No valid chunks created from copybook content")
                return None
            
            self.logger.info(f"Created {len(chunks)} chunks for processing")
            
            # Process each chunk
            all_fields = []
            record_name = None
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Create chunk-specific prompt
                prompt = f"""
                You are a COBOL expert. Parse this copybook chunk and extract field definitions.
                
                COPYBOOK CHUNK {i+1} of {len(chunks)}:
                {chunk}
                
                Extract field definitions with these rules:
                1. Only include elementary items (fields with PIC clauses)
                2. Convert COBOL names to SQL-safe names (replace hyphens with underscores)
                3. Handle FILLER fields by giving them unique names
                4. Skip REDEFINES fields (they're overlays)
                5. Convert PIC clauses to appropriate SQL types
                6. If this is chunk 1, also extract the record name from 01 level
                
                Return ONLY valid JSON in this exact format:
                {{
                    "record_name": "RECORD_NAME" (only if found in this chunk),
                    "fields": [
                        {{
                            "name": "FIELD_NAME",
                            "type": "VARCHAR(50)",
                            "level": "05",
                            "pic_clause": "PIC X(50)",
                            "nullable": true
                        }}
                    ],
                    "chunk_number": {i+1}
                }}
                """
                
                sampling_params = SamplingParams(temperature=0.1, max_tokens=1500)
                
                try:
                    response_text = await self._generate_with_llm(prompt, sampling_params)
                    chunk_result = self._extract_json_from_response(response_text)
                    
                    if chunk_result:
                        # Extract record name from first chunk if available
                        if i == 0 and 'record_name' in chunk_result and chunk_result['record_name']:
                            record_name = chunk_result['record_name']
                        
                        # Collect fields from this chunk
                        if 'fields' in chunk_result and isinstance(chunk_result['fields'], list):
                            chunk_fields = chunk_result['fields']
                            self.logger.info(f"Chunk {i+1} yielded {len(chunk_fields)} fields")
                            all_fields.extend(chunk_fields)
                            
                        chunk_results.append({
                            'chunk_number': i+1,
                            'field_count': len(chunk_result.get('fields', [])),
                            'success': True
                        })
                    else:
                        self.logger.warning(f"Failed to parse chunk {i+1}")
                        chunk_results.append({
                            'chunk_number': i+1,
                            'field_count': 0,
                            'success': False,
                            'error': 'JSON parsing failed'
                        })
                    
                    # Small delay between chunks to avoid overwhelming the LLM
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.3)
                        
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    chunk_results.append({
                        'chunk_number': i+1,
                        'field_count': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Combine results
            if not all_fields:
                self.logger.error("No fields extracted from any chunks")
                return None
            
            # Use filename-based record name if not found in chunks
            if not record_name:
                record_name = self._generate_table_name(filename)
            
            # Clean up field names and ensure uniqueness
            processed_fields = self._deduplicate_and_clean_fields(all_fields)
            
            self.logger.info(f"✅ Chunked parsing complete: {len(processed_fields)} total fields")
            
            result = {
                "record_name": record_name,
                "structure_type": "flat",
                "fields": processed_fields,
                "parsing_method": "chunked_llm",
                "chunk_count": len(chunks),
                "chunk_results": chunk_results,
                "total_fields_found": len(all_fields),
                "final_field_count": len(processed_fields)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chunked copybook parsing failed: {str(e)}")
            return None

    def _chunk_copybook_by_structure(self, content: str) -> List[str]:
        """Split copybook content into logical chunks preserving structure"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = 3000  # Characters per chunk (safe for LLM context)
        
        # Always include 01 level record in first chunk
        record_line = None
        for i, line in enumerate(lines):
            if re.match(r'^\s*01\s+[A-Z]', line, re.IGNORECASE):
                record_line = line
                break
        
        # Start first chunk with record line if found
        if record_line:
            current_chunk.append(record_line)
            current_size = len(record_line)
        
        for line in lines:
            line_length = len(line)
            
            # Skip the record line if we already added it
            if record_line and line.strip() == record_line.strip():
                continue
            
            # Check if adding this line would exceed chunk size
            if current_size + line_length > max_chunk_size and current_chunk:
                # Try to break at a logical boundary (level number)
                if re.match(r'^\s*\d+\s+[A-Z]', line, re.IGNORECASE):
                    # This line starts a new field, good place to break
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                    
                    # Include record line reference in subsequent chunks
                    if record_line and len(chunks) > 1:
                        current_chunk.append(f"* Record: {record_line.strip()}")
                        current_size += len(current_chunk[0])
            
            current_chunk.append(line)
            current_size += line_length
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Filter out very small chunks (likely just comments)
        meaningful_chunks = []
        for chunk in chunks:
            field_lines = [line for line in chunk.split('\n') 
                        if re.match(r'^\s*\d+\s+[A-Z]', line, re.IGNORECASE)]
            if len(field_lines) >= 2:  # At least 2 field definitions
                meaningful_chunks.append(chunk)
        
        return meaningful_chunks

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in LLM response")
                return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.debug(f"Raw response: {response_text[:300]}...")
            return None

    def _deduplicate_and_clean_fields(self, fields: List[Dict]) -> List[Dict]:
        """Remove duplicates and clean up field names"""
        seen_names = set()
        clean_fields = []
        filler_counter = 1
        
        for field in fields:
            if not isinstance(field, dict) or 'name' not in field:
                continue
            
            original_name = field['name']
            clean_name = self._clean_column_name(original_name)
            
            # Handle FILLER fields
            if clean_name.upper().startswith('FILLER'):
                clean_name = f"FILLER_{filler_counter:03d}"
                filler_counter += 1
            
            # Ensure unique names
            base_name = clean_name
            counter = 1
            while clean_name in seen_names:
                clean_name = f"{base_name}_{counter}"
                counter += 1
            
            seen_names.add(clean_name)
            
            # Update field with clean name
            field['name'] = clean_name
            if 'original_name' not in field:
                field['original_name'] = original_name
            
            # Set defaults for missing properties
            field.setdefault('nullable', True)
            field.setdefault('level', '05')
            field.setdefault('pic_clause', '')
            field.setdefault('type', 'VARCHAR(50)')
            
            clean_fields.append(field)
        
        return clean_fields       
    async def _parse_copybook_structure_llm(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """LLM-based parsing with better error handling"""
        
        # Check if content is too large for LLM
        if len(content) > 4000:  # Rough character limit
            self.logger.info("📏 Content too large, using chunked processing...")
            return await self._parse_copybook_chunked(content, filename)
        
        prompt = f"""
        You are a COBOL expert. Parse this copybook and extract field definitions.

        COPYBOOK CONTENT:
        {content}

        Extract all field definitions with these rules:
        1. Only include elementary items (fields with PIC clauses)
        2. Convert COBOL names to SQL-safe names (replace hyphens with underscores)
        3. Handle FILLER fields by giving them unique names like FILLER_001
        4. Skip REDEFINES fields (they're overlays)
        5. Convert PIC clauses to appropriate SQL types

        Return ONLY valid JSON in this exact format:
        {{
            "record_name": "RECORD_NAME",
            "fields": [
                {{
                    "name": "FIELD_NAME",
                    "type": "VARCHAR(50)",
                    "level": "05",
                    "pic_clause": "PIC X(50)",
                    "nullable": true
                }}
            ]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2000)
        
        try:
            response_text = await self._generate_with_llm(prompt, sampling_params)
            
            # Enhanced JSON extraction
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_result = json.loads(json_str)
                
                # Validate structure
                if self._validate_copybook_result(parsed_result):
                    return self._post_process_copybook_result(parsed_result)
                else:
                    self.logger.warning("LLM result failed validation")
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.debug(f"Raw LLM response: {response_text[:500]}...")
        except Exception as e:
            self.logger.error(f"LLM parsing error: {str(e)}")
        
        return None
        
    def _validate_copybook_result(self, result: Dict) -> bool:
        """Validate parsed copybook result"""
        if not isinstance(result, dict):
            return False
        
        if 'fields' not in result or not isinstance(result['fields'], list):
            return False
        
        if len(result['fields']) == 0:
            return False
        
        # Check each field has required properties
        for field in result['fields']:
            if not isinstance(field, dict):
                return False
            if 'name' not in field or 'type' not in field:
                return False
            if not field['name'] or not field['type']:
                return False
        
        return True
    
    def _post_process_copybook_result(self, result: Dict) -> Dict:
        """Post-process and clean up copybook result"""
        # Ensure record name is SQL-safe
        if 'record_name' not in result or not result['record_name']:
            result['record_name'] = 'UNKNOWN_RECORD'
        
        result['record_name'] = self._clean_column_name(result['record_name'])
        
        # Clean up fields
        filler_counter = 1
        seen_names = set()
        
        for field in result['fields']:
            # Clean field name
            if field['name'].upper().startswith('FILLER'):
                field['name'] = f"FILLER_{filler_counter:03d}"
                filler_counter += 1
            else:
                field['name'] = self._clean_column_name(field['name'])
            
            # Ensure unique names
            original_name = field['name']
            counter = 1
            while field['name'] in seen_names:
                field['name'] = f"{original_name}_{counter}"
                counter += 1
            seen_names.add(field['name'])
            
            # Set defaults
            field.setdefault('nullable', True)
            field.setdefault('level', '05')
            field.setdefault('pic_clause', '')
        
        result['structure_type'] = 'flat'
        return result

    def _parse_copybook_with_enhanced_regex(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """Enhanced regex-based copybook parsing"""
        try:
            fields = []
            record_name = None
            filler_counter = 1
            
            # Find 01 level record name (more flexible pattern)
            record_patterns = [
                r'^\s*01\s+([A-Z][A-Z0-9-]*)',  # Standard format
                r'^\s*1\s+([A-Z][A-Z0-9-]*)',   # Without leading zero
                r'01\s+([A-Z][A-Z0-9-]*)',      # No line start anchor
            ]
            
            for pattern in record_patterns:
                record_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
                if record_match:
                    record_name = self._clean_column_name(record_match.group(1))
                    break
            
            if not record_name:
                record_name = self._generate_table_name(filename)
            
            # Enhanced field patterns (try multiple formats)
            field_patterns = [
                # Standard: level field-name PIC clause
                r'^\s*(\d+)\s+([A-Z][A-Z0-9-]*)\s+PIC\s+([X9AVSZ]+(?:\([0-9,]+\))?)',
                # With FILLER: level FILLER PIC clause  
                r'^\s*(\d+)\s+(FILLER)\s+PIC\s+([X9AVSZ]+(?:\([0-9,]+\))?)',
                # Flexible spacing
                r'(\d+)\s+([A-Z][A-Z0-9-]*)\s+PIC\s+([X9AVSZ]+(?:\([0-9,]+\))?)',
            ]
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('*'):  # Skip comments
                    continue
                    
                for pattern in field_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        level = match.group(1)
                        field_name = match.group(2)
                        pic_clause = match.group(3)
                        
                        # Skip group items (level < 49 without PIC usually means group)
                        if int(level) <= 49 and not pic_clause:
                            continue
                        
                        # Handle FILLER
                        if field_name.upper() == 'FILLER':
                            clean_name = f"FILLER_{filler_counter:03d}"
                            filler_counter += 1
                        else:
                            clean_name = self._clean_column_name(field_name)
                        
                        # Convert PIC to SQL type
                        sql_type = self._convert_pic_to_sql(pic_clause)
                        
                        fields.append({
                            "name": clean_name,
                            "type": sql_type,
                            "level": level,
                            "pic_clause": f"PIC {pic_clause}",
                            "nullable": True
                        })
                        break  # Found match, move to next line
            
            if fields:
                return {
                    "record_name": record_name,
                    "structure_type": "flat",
                    "fields": fields
                }
                
        except Exception as e:
            self.logger.error(f"Enhanced regex parsing failed: {str(e)}")
        
        return None

    def _parse_copybook_with_simple_regex(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """Simple regex parsing as last resort"""
        try:
            fields = []
            record_name = self._generate_table_name(filename)
            
            # Very simple pattern - just find anything that looks like a field
            lines = content.split('\n')
            field_counter = 1
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                
                # Look for level number followed by name
                simple_match = re.match(r'^\s*(\d+)\s+([A-Z][A-Z0-9-]+)', line, re.IGNORECASE)
                if simple_match:
                    level = simple_match.group(1)
                    field_name = simple_match.group(2)
                    
                    # Skip 01 level (record name)
                    if level == '01' or level == '1':
                        continue
                    
                    clean_name = self._clean_column_name(field_name)
                    
                    # Default to VARCHAR for unknown types
                    fields.append({
                        "name": clean_name,
                        "type": "VARCHAR(50)",
                        "level": level,
                        "pic_clause": "",
                        "nullable": True
                    })
                    
                    field_counter += 1
                    
                    # Limit to reasonable number of fields
                    if field_counter > 100:
                        break
            
            if fields:
                return {
                    "record_name": record_name,
                    "structure_type": "flat", 
                    "fields": fields
                }
                
        except Exception as e:
            self.logger.error(f"Simple regex parsing failed: {str(e)}")
        
        return None

    def _parse_copybook_with_regex(self, content: str, filename: str) -> Optional[Dict[str, Any]]:
        """Enhanced copybook parsing with FILLER, REDEFINES, and other COBOL constructs"""
        try:
            fields = []
            record_name = None
            filler_counter = 1  # Counter for FILLER fields
            
            # Find 01 level record name
            record_pattern = re.compile(r'^\s*01\s+([A-Z][A-Z0-9-]*)', re.MULTILINE)
            record_match = record_pattern.search(content)
            if record_match:
                record_name = self._clean_column_name(record_match.group(1))
            else:
                record_name = self._generate_table_name(filename)
            
            # Enhanced field pattern to capture more COBOL constructs
            field_pattern = re.compile(
                r'^\s*(\d+)\s+'  # Level number
                r'([A-Z][A-Z0-9-]*|FILLER)\s+'  # Field name or FILLER
                r'(?:PIC\s+([X9AVSx]+(?:\([0-9,]+\))?)\s*)?'  # PIC clause (optional)
                r'(.*?)$',  # Rest of line (for REDEFINES, VALUE, etc.)
                re.MULTILINE | re.IGNORECASE
            )
            
            for match in field_pattern.finditer(content):
                level = match.group(1)
                field_name = match.group(2)
                pic_clause = match.group(3) if match.group(3) else None
                rest_of_line = match.group(4).strip() if match.group(4) else ""
                
                # Handle FILLER fields
                if field_name.upper() == 'FILLER':
                    clean_field_name = f"FILLER_{filler_counter:03d}"
                    filler_counter += 1
                    is_filler = True
                else:
                    clean_field_name = self._clean_column_name(field_name)
                    is_filler = False
                
                # Parse additional clauses
                field_info = self._parse_field_clauses(rest_of_line, pic_clause)
                
                # Skip REDEFINES fields from table creation (they're overlays)
                if field_info.get('redefines'):
                    self.logger.info(f"Skipping REDEFINES field: {field_name}")
                    continue
                
                # Skip group items without PIC clause unless they have special meaning
                if not pic_clause and int(level) < 49:
                    # This might be a group item - check if it should be included
                    if not self._should_include_group_item(rest_of_line):
                        continue
                
                # Convert PIC clause to SQL type
                if pic_clause:
                    sql_type = self._convert_pic_to_sql(pic_clause)
                else:
                    # Group item or special field
                    sql_type = "VARCHAR(1)"  # Placeholder for group items
                
                field_data = {
                    "name": clean_field_name,
                    "original_name": field_name,
                    "type": sql_type,
                    "level": level,
                    "pic_clause": f"PIC {pic_clause}" if pic_clause else "",
                    "nullable": True,
                    "is_filler": is_filler,
                    **field_info  # Add parsed clauses
                }
                
                fields.append(field_data)
            
            if fields:
                return {
                    "record_name": record_name,
                    "structure_type": self._determine_structure_type(content),
                    "fields": fields,
                    "filler_count": filler_counter - 1
                }
        
        except Exception as e:
            self.logger.error(f"Enhanced copybook parsing failed: {str(e)}")
        
        return None

    def _parse_field_clauses(self, rest_of_line: str, pic_clause: str) -> Dict[str, Any]:
        """Parse additional COBOL field clauses"""
        clauses = {}
        line_upper = rest_of_line.upper()
        
        # REDEFINES clause
        redefines_match = re.search(r'REDEFINES\s+([A-Z][A-Z0-9-]*)', line_upper)
        if redefines_match:
            clauses['redefines'] = redefines_match.group(1)
        
        # VALUE clause
        value_match = re.search(r'VALUE\s+(?:IS\s+)?([\'"][^\'\"]*[\'"]|\S+)', line_upper)
        if value_match:
            clauses['default_value'] = value_match.group(1).strip('\'"')
        
        # OCCURS clause
        occurs_match = re.search(r'OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES', line_upper)
        if occurs_match:
            min_occurs = int(occurs_match.group(1))
            max_occurs = int(occurs_match.group(2)) if occurs_match.group(2) else min_occurs
            clauses['occurs'] = {
                'min': min_occurs,
                'max': max_occurs,
                'is_variable': max_occurs != min_occurs
            }
        
        # DEPENDING ON clause
        depending_match = re.search(r'DEPENDING\s+ON\s+([A-Z][A-Z0-9-]*)', line_upper)
        if depending_match:
            clauses['depending_on'] = depending_match.group(1)
        
        # USAGE clause
        usage_match = re.search(r'USAGE\s+(?:IS\s+)?(COMP(?:-[0-9])?|BINARY|DISPLAY|PACKED-DECIMAL|INDEX)', line_upper)
        if usage_match:
            clauses['usage'] = usage_match.group(1)
            # Adjust SQL type based on usage
            if pic_clause and usage_match.group(1) in ['COMP-3', 'PACKED-DECIMAL']:
                clauses['adjusted_type'] = 'DECIMAL'
        
        # SYNC/SYNCHRONIZED clause
        if 'SYNC' in line_upper or 'SYNCHRONIZED' in line_upper:
            clauses['synchronized'] = True
        
        # JUSTIFIED clause
        if 'JUSTIFIED' in line_upper or 'JUST' in line_upper:
            clauses['justified'] = True
        
        return clauses

    def _should_include_group_item(self, rest_of_line: str) -> bool:
        """Determine if a group item should be included in the table"""
        # Don't include group items that are just containers
        # Include them if they have VALUE or special clauses
        line_upper = rest_of_line.upper()
        
        # Include if it has a VALUE clause
        if 'VALUE' in line_upper:
            return True
        
        # Include if it's a special type (like dates, counters)
        special_indicators = ['DATE', 'TIME', 'COUNT', 'TOTAL', 'FLAG']
        if any(indicator in line_upper for indicator in special_indicators):
            return True
        
        return False
    
    def _determine_structure_type(self, content: str) -> str:
        """Determine if the copybook has hierarchical or flat structure"""
        lines = content.split('\n')
        level_counts = {}
        
        for line in lines:
            level_match = re.match(r'^\s*(\d+)\s+', line)
            if level_match:
                level = int(level_match.group(1))
                level_counts[level] = level_counts.get(level, 0) + 1
        
        # If we have multiple levels beyond 01, it's hierarchical
        significant_levels = [level for level in level_counts.keys() if level > 1]
        
        if len(significant_levels) > 2:  # More than just 01 and one sub-level
            return "hierarchical"
        else:
            return "flat"

    def _convert_pic_to_sql(self, pic_clause: str) -> str:
        """Enhanced PIC clause to SQL type conversion"""
        pic_upper = pic_clause.upper()
        
        # Extract numbers from parentheses
        numbers = re.findall(r'\((\d+(?:,\d+)?)\)', pic_clause)
        
        if 'X' in pic_upper:
            # Alphanumeric
            if numbers:
                # Handle X(n) format
                length = int(numbers[0].replace(',', ''))
            else:
                # Count X's manually
                length = pic_upper.count('X')
            return f"VARCHAR({min(length, 4000)})"  # Cap at reasonable size
        
        elif 'V' in pic_upper:
            # Decimal with implied decimal point
            parts = pic_upper.split('V')
            integer_part = parts[0].count('9') if parts[0] else 0
            decimal_part = parts[1].count('9') if len(parts) > 1 else 0
            
            # Handle parentheses format like 9(5)V9(2)
            if numbers:
                if len(numbers) >= 2:
                    integer_part = int(numbers[0])
                    decimal_part = int(numbers[1])
                elif 'V' in pic_upper and numbers:
                    # Might be 9(7)V99 format
                    integer_part = int(numbers[0])
                    decimal_part = parts[1].count('9')
            
            total_digits = integer_part + decimal_part
            return f"DECIMAL({total_digits}, {decimal_part})"
        
        elif '9' in pic_upper:
            # Numeric
            if numbers:
                digit_count = int(numbers[0].replace(',', ''))
            else:
                digit_count = pic_upper.count('9')
            
            # Check for signed
            is_signed = 'S' in pic_upper
            
            if digit_count <= 4:
                return "SMALLINT" if is_signed else "INTEGER"
            elif digit_count <= 9:
                return "INTEGER"
            elif digit_count <= 18:
                return "BIGINT"
            else:
                return f"DECIMAL({digit_count}, 0)"
        
        elif 'A' in pic_upper:
            # Alphabetic
            if numbers:
                length = int(numbers[0])
            else:
                length = pic_upper.count('A')
            return f"VARCHAR({length})"
        
        elif 'N' in pic_upper:
            # National (Unicode)
            if numbers:
                length = int(numbers[0])
            else:
                length = pic_upper.count('N')
            return f"NVARCHAR({length})"
        
        else:
            # Default for unknown formats
            return "VARCHAR(255)"


    async def get_component_info(self, component_name: str) -> Dict[str, Any]:
        """Get detailed information about a data component"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if it's a table
            cursor.execute("""
                SELECT ts.*, ds.record_count, ds.data_quality_score
                FROM table_schemas ts
                LEFT JOIN data_samples ds ON ts.table_name = ds.table_name
                WHERE ts.table_name = ? OR ts.table_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            table_info = cursor.fetchone()
            
            if table_info:
                # Get field information
                cursor.execute("""
                    SELECT field_name, field_type, is_nullable, field_description, 
                           business_meaning, data_classification
                    FROM field_catalog
                    WHERE table_name = ?
                """, (table_info[1],))  # table_name is at index 1
                
                fields = cursor.fetchall()
                
                # Get sample data
                cursor.execute("""
                    SELECT sample_data FROM data_samples WHERE table_name = ? LIMIT 1
                """, (table_info[1],))
                
                sample_result = cursor.fetchone()
                sample_data = json.loads(sample_result[0]) if sample_result and sample_result[0] else []
                
                conn.close()
                
                return {
                    "component_type": "table",
                    "table_name": table_info[1],
                    "schema_type": table_info[2],
                    "field_definitions": json.loads(table_info[3]) if table_info[3] else [],
                    "source_file": table_info[4],
                    "record_count": table_info[6] if len(table_info) > 6 else 0,
                    "data_quality_score": table_info[7] if len(table_info) > 7 else None,
                    "fields": [
                        {
                            "name": f[0], "type": f[1], "nullable": f[2],
                            "description": f[3], "business_meaning": f[4],
                            "classification": f[5]
                        } for f in fields
                    ],
                    "sample_data": sample_data[:10]  # First 10 records
                }
            
            # Check if it's a record layout
            cursor.execute("""
                SELECT * FROM record_layouts WHERE layout_name = ? OR layout_name LIKE ?
            """, (component_name, f"%{component_name}%"))
            
            layout_info = cursor.fetchone()
            conn.close()
            
            if layout_info:
                return {
                    "component_type": "record_layout",
                    "layout_name": layout_info[1],
                    "record_types": json.loads(layout_info[2]) if layout_info[2] else [],
                    "description": layout_info[3],
                    "source_file": layout_info[4]
                }
            
            return {
                "component_type": "not_found",
                "message": f"Component '{component_name}' not found in data catalog"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get component info: {str(e)}")
            return {"component_type": "error", "error": str(e)}
    
    async def get_data_lineage_info(self, component_name: str) -> Dict[str, Any]:
        """Get data lineage information for a component - FIXED"""
        try:
            #await self._ensure_llm_engine()
            component_info = await self.get_component_info(component_name)
            
            if component_info["component_type"] == "table":
                # Analyze data lineage using LLM
                prompt = f"""
                Analyze the data lineage for this table:
                
                Table: {component_name}
                Fields: {[f["name"] for f in component_info.get("fields", [])]}
                Source File: {component_info.get("source_file", "Unknown")}
                
                Based on the field names and table structure, suggest:
                1. Likely source systems
                2. Downstream consumers
                3. Data transformation points
                4. Business processes that use this data
                
                Return as structured analysis.
                """
                
                sampling_params = SamplingParams(temperature=0.3, max_tokens=800)
                
                try:
                    # FIX: Use the helper method instead of direct LLM call
                    result_text = await self._generate_with_llm(prompt, sampling_params)
                except Exception as e:
                    result_text = f"Analysis failed: {str(e)}"
                
                return {
                    "component_name": component_name,
                    "lineage_analysis": result_text,
                    "component_details": component_info
                }
            
            return {
                "component_name": component_name,
                "message": "Lineage analysis only available for tables",
                "component_details": component_info
            }
            
        except Exception as e:
            return {"error": str(e)}

    async def compare_table_structures(self, table1: str, table2: str) -> Dict[str, Any]:
        """Compare two table structures"""
        try:
            info1 = await self.get_component_info(table1)
            info2 = await self.get_component_info(table2)
            
            if info1["component_type"] != "table" or info2["component_type"] != "table":
                return {"error": "Both components must be tables"}
            
            fields1 = {f["name"]: f for f in info1.get("fields", [])}
            fields2 = {f["name"]: f for f in info2.get("fields", [])}
            
            common_fields = set(fields1.keys()) & set(fields2.keys())
            only_in_table1 = set(fields1.keys()) - set(fields2.keys())
            only_in_table2 = set(fields2.keys()) - set(fields1.keys())
            
            # Check for type differences in common fields
            type_differences = []
            for field in common_fields:
                if fields1[field]["type"] != fields2[field]["type"]:
                    type_differences.append({
                        "field": field,
                        "table1_type": fields1[field]["type"],
                        "table2_type": fields2[field]["type"]
                    })
            
            return {
                "table1": table1,
                "table2": table2,
                "common_fields": list(common_fields),
                "only_in_table1": list(only_in_table1),
                "only_in_table2": list(only_in_table2),
                "type_differences": type_differences,
                "similarity_score": len(common_fields) / max(len(fields1), len(fields2)) if fields1 or fields2 else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough token estimation: 1 token ≈ 4 characters"""
        return len(text) // 4

    def _chunk_copybook_content(self, content: str, max_chunk_tokens: int = 600) -> List[str]:
        """Split copybook content into chunks that fit within token limits"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        base_instructions_tokens = 400  # Estimated tokens for instructions
        
        for line in lines:
            line_tokens = self._estimate_token_count(line)
            
            # If adding this line would exceed limit, start new chunk
            if current_tokens + line_tokens + base_instructions_tokens > max_chunk_tokens:
                if current_chunk:  # Don't create empty chunks
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _chunk_field_content(self, fields_data: List[Dict], max_fields_per_chunk: int = 8) -> List[List[Dict]]:
        """Split field data into smaller chunks for processing"""
        chunks = []
        for i in range(0, len(fields_data), max_fields_per_chunk):
            chunk = fields_data[i:i + max_fields_per_chunk]
            chunks.append(chunk)
        return chunks

    async def _generate_with_llm_chunked(self, prompt_template: str, content: str, 
                                    sampling_params, content_placeholder: str = "{CONTENT}") -> str:
        """Generate text with LLM using chunking if content is too large"""
        
        # First, try with full content
        full_prompt = prompt_template.replace(content_placeholder, content)
        estimated_tokens = self._estimate_token_count(full_prompt)
        
        if estimated_tokens <= 900:  # Safe margin below 1024
            return await self._generate_with_llm(full_prompt, sampling_params)
        
        # Need to chunk the content
        self.logger.info(f"Prompt too large ({estimated_tokens} tokens), chunking content...")
        
        if "copybook" in prompt_template.lower():
            chunks = self._chunk_copybook_content(content)
        else:
            # For other content types, split by lines
            lines = content.split('\n')
            chunk_size = len(lines) // ((estimated_tokens // 900) + 1)
            chunks = ['\n'.join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
        
        # Process each chunk
        all_results = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = prompt_template.replace(content_placeholder, chunk)
            chunk_prompt += f"\n\nNote: This is chunk {i+1} of {len(chunks)}. Maintain consistent formatting."
            
            try:
                result = await self._generate_with_llm(chunk_prompt, sampling_params)
                all_results.append(result)
                if i < len(chunks) - 1:  # Don't wait after last chunk
                    await asyncio.sleep(0.5)  # 500ms delay

            except Exception as e:
                self.logger.error(f"Failed to process chunk {i+1}: {str(e)}")
                all_results.append("")
        
        # Combine results
        return self._combine_chunked_results(all_results)

    def _combine_chunked_results(self, results: List[str]) -> str:
        """Combine multiple LLM results into a single coherent response"""
        combined_fields = []
        
        for result in results:
            if not result.strip():
                continue
                
            try:
                # Extract JSON from each result
                if '{' in result:
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1
                    chunk_json = json.loads(result[json_start:json_end])
                    
                    if 'fields' in chunk_json and isinstance(chunk_json['fields'], list):
                        combined_fields.extend(chunk_json['fields'])
                    elif isinstance(chunk_json, list):
                        combined_fields.extend(chunk_json)
                        
            except Exception as e:
                self.logger.warning(f"Failed to parse chunked result: {str(e)}")
        
        # Return combined result in expected format
        if combined_fields:
            return json.dumps({
                "record_name": "COMBINED_RECORD",
                "structure_type": "flat",
                "fields": combined_fields,
                "chunked_processing": True
            })
        else:
            return ""