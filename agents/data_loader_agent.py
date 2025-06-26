# agents/data_loader_agent.py
"""
Agent 3: Batch Data Loader & Table Mapper
Handles DB2 DDL, DCLGEN files, CSV layouts, and data ingestion
Updated with dynamic GPU allocation support
"""

import asyncio
import sqlite3
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
import logging
from datetime import datetime
import zipfile
import hashlib
import os

from vllm import AsyncLLMEngine, SamplingParams

class DataLoaderAgent:
    """Agent for loading and mapping data files and schemas"""
    
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, gpu_id: int = None):
        self.llm_engine = llm_engine
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(__name__)
        
        # For dynamic creation without LLM engine
        self._engine_created = False
        
        # Initialize SQLite tables for data mapping
        self._init_data_tables()
    
    async def _ensure_llm_engine(self):
        """Ensure LLM engine is available, create if needed"""
        if self.llm_engine is None and not self._engine_created:
            await self._create_llm_engine()
    
    async def _create_llm_engine(self):
        """Create LLM engine with GPU forcing"""
        try:
            # Import GPU forcer
            from gpu_force_fix import GPUForcer
            
            # Find best available GPU
            best_gpu = GPUForcer.find_best_gpu_with_memory(2.0)
            
            if best_gpu is None:
                raise RuntimeError("No suitable GPU found for LLM engine")
            
            self.logger.info(f"Creating LLM engine for DataLoader on GPU {best_gpu}")
            
            # Force GPU environment
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                GPUForcer.force_gpu_environment(best_gpu)
                
                # Create engine with forced GPU
                from vllm import AsyncLLMEngine
                engine_args = GPUForcer.create_vllm_engine_args(
                    "codellama/CodeLlama-7b-Instruct-hf",  # Default model
                    4096
                )
                
                self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.gpu_id = best_gpu
                self._engine_created = True
                
                self.logger.info(f"✅ DataLoader LLM engine created on GPU {best_gpu}")
                
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                    
        except Exception as e:
            self.logger.error(f"Failed to create LLM engine for DataLoader: {str(e)}")
            raise
    
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
        """Process a data file (CSV, DDL, DCLGEN, layout) with GPU forcing"""
        try:
            # Ensure we have an LLM engine
            await self._ensure_llm_engine()
            
            self.logger.info(f"Processing file {file_path} with DataLoader on GPU {self.gpu_id}")
            
            file_extension = file_path.suffix.lower()
            file_content = self._read_file_safely(file_path)
            
            result = {}
            
            if file_extension == '.csv':
                result = await self._process_csv_file(file_path, file_content)
            elif file_extension in ['.ddl', '.sql']:
                result = await self._process_ddl_file(file_path, file_content)
            elif 'dclgen' in file_path.name.lower() or file_extension == '.dcl':
                result = await self._process_dclgen_file(file_path, file_content)
            elif file_extension == '.json':
                result = await self._process_layout_json(file_path, file_content)
            elif file_extension == '.zip':
                result = await self._process_zip_file(file_path)
            else:
                # Try to auto-detect file type
                result = await self._auto_detect_and_process(file_path, file_content)
            
            # Add GPU info to result
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'data_loader'
            
            return result
                
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {str(e)}")
            return {
                "status": "error",
                "file_name": file_path.name,
                "error": str(e),
                "gpu_used": self.gpu_id,
                "agent_type": "data_loader"
            }
    
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
            # Ensure LLM engine is ready
            await self._ensure_llm_engine()
            
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
                "field_descriptions": field_descriptions
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
        # Remove special characters and spaces
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
    
    async def _enhance_schema_with_llm(self, schema: List[Dict], filename: str) -> List[Dict[str, Any]]:
        """Use LLM to enhance schema with business context"""
        try:
            await self._ensure_llm_engine()
            
            prompt = f"""
            Analyze this database schema for file "{filename}" and enhance it with business context:
            
            Schema:
            {json.dumps(schema, indent=2)}
            
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
            result = await self.llm_engine.generate(prompt, sampling_params)
            
            try:
                response_text = result.outputs[0].text.strip()
                if '{' in response_text:
                    json_start = response_text.find('[') if '[' in response_text else response_text.find('{')
                    json_end = response_text.rfind(']') + 1 if ']' in response_text else response_text.rfind('}') + 1
                    enhanced_data = json.loads(response_text[json_start:json_end])
                    
                    # Merge enhanced data with original schema
                    if isinstance(enhanced_data, list) and len(enhanced_data) == len(schema):
                        for i, enhancement in enumerate(enhanced_data):
                            if isinstance(enhancement, dict):
                                schema[i].update(enhancement)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM schema enhancement: {str(e)}")
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Schema enhancement failed: {str(e)}")
            return schema  # Return original schema on failure
    
    # ... (rest of the methods remain the same, but ensure they call await self._ensure_llm_engine() before using self.llm_engine)
    
    async def _generate_field_descriptions(self, columns: List[str], sample_df: pd.DataFrame) -> Dict[str, str]:
        """Generate field descriptions using LLM"""
        descriptions = {}
        
        try:
            await self._ensure_llm_engine()
            
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
                result = await self.llm_engine.generate(prompt, sampling_params)
                
                try:
                    response_text = result.outputs[0].text.strip()
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
        
        except Exception as e:
            self.logger.error(f"Field description generation failed: {str(e)}")
            # Fallback descriptions
            for col in columns:
                descriptions[col] = f"Data field: {col}"
        
        return descriptions
    
    # Add all the remaining methods from the original class, ensuring they call await self._ensure_llm_engine()
    # before using self.llm_engine...
    
    def _generate_table_name(self, filename: str) -> str:
        """Generate SQLite table name from filename"""
        base_name = Path(filename).stem
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
        table_name = re.sub(r'_{2,}', '_', table_name)
        return table_name.lower().strip('_')
    
    async def _create_sqlite_table(self, table_name: str, schema: List[Dict[str, Any]]):
        """Create SQLite table from schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table
        columns = []
        for field in schema:
            column_def = f"{field['name']} {field['type']}"
            if not field.get('nullable', True):
                column_def += " NOT NULL"
            columns.append(column_def)
        
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        cursor.execute(create_sql)
        
        conn.commit()
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
        """, (table_name, schema_type, json.dumps(schema), source_file, datetime.now()))
        
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
            
            conn.close()
            
            return {
                "component_type": "not_found",
                "message": f"Component '{component_name}' not found in data catalog"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get component info: {str(e)}")
            return {"component_type": "error", "error": str(e)}
    
    # Continue with remaining methods, ensuring each one that uses self.llm_engine 
    # calls await self._ensure_llm_engine() first...
    
    async def get_data_lineage_info(self, component_name: str) -> Dict[str, Any]:
        """Get data lineage information for a component"""
        try:
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
                result = await self.llm_engine.generate(prompt, sampling_params)
                
                return {
                    "component_name": component_name,
                    "lineage_analysis": result.outputs[0].text,
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