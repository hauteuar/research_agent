# agents/db2_comparator_agent.py
"""
Agent 7: DB2 Comparator Agent
Compares data between DB2 tables and SQLite tables (limited to 10K rows)
"""

import asyncio
import sqlite3
import pandas as pd
import logging
import os
import uuid
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime as dt
import json
import hashlib
from contextlib import asynccontextmanager

# DB2 connection imports
try:
    import ibm_db
    import ibm_db_dbi
    DB2_AVAILABLE = True
except ImportError:
    DB2_AVAILABLE = False
    logging.warning("IBM DB2 drivers not available. DB2 comparison will be limited.")

from vllm import AsyncLLMEngine, SamplingParams

class DB2ComparatorAgent:
    """Agent for comparing data between DB2 and SQLite"""
    
    def __init__(self, llm_engine: AsyncLLMEngine, db_path: str, gpu_id: int, max_rows: int = 10000):
        self.llm_engine = llm_engine
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.max_rows = max_rows
        self.logger = logging.getLogger(__name__)
        
        # DB2 connection parameters (loaded from environment or config)
        self.db2_config = {
            "database": os.getenv("DB2_DATABASE", "TESTDB"),
            "hostname": os.getenv("DB2_HOSTNAME", "localhost"), 
            "port": os.getenv("DB2_PORT", "50000"),
            "username": os.getenv("DB2_USERNAME", "db2user"),
            "password": os.getenv("DB2_PASSWORD", "password")
        }
        
        self._init_comparison_tables()
    
    def _init_comparison_tables(self):
        """Initialize tables for tracking comparisons"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS comparison_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sqlite_component TEXT NOT NULL,
                        db2_component TEXT NOT NULL,
                        comparison_type TEXT NOT NULL,
                        match_percentage REAL DEFAULT 0,
                        differences_count INTEGER DEFAULT 0,
                        comparison_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        comparison_details TEXT,
                        status TEXT DEFAULT 'completed'
                    );
                    
                    CREATE TABLE IF NOT EXISTS data_validation_rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        table_name TEXT NOT NULL,
                        field_name TEXT,
                        validation_rule TEXT NOT NULL,
                        rule_description TEXT,
                        threshold_value REAL,
                        is_active BOOLEAN DEFAULT 1,
                        created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS db2_connection_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        connection_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        operation_type TEXT NOT NULL,
                        response_time_ms INTEGER
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_comparison_timestamp 
                    ON comparison_history(comparison_timestamp);
                    
                    CREATE INDEX IF NOT EXISTS idx_sqlite_component 
                    ON comparison_history(sqlite_component);
                """)
                
                conn.commit()
                self.logger.info("Comparison tables initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize comparison tables: {str(e)}")
            raise
    
    async def compare_data(self, sqlite_component: str = None, db2_component: str = None, 
                          fields: List[str] = None, filter_condition: str = None) -> Dict[str, Any]:
        """Compare data between SQLite and DB2 components"""
        try:
            # Validate inputs
            if not sqlite_component and not db2_component:
                return {"error": "At least one component name is required"}
            
            # If called with single component name, try to match components
            if sqlite_component and not db2_component:
                db2_component = sqlite_component
            elif db2_component and not sqlite_component:
                sqlite_component = db2_component
            
            self.logger.info(f"Starting comparison: SQLite '{sqlite_component}' vs DB2 '{db2_component}'")
            
            # Get SQLite data
            sqlite_data = await self._get_sqlite_data(sqlite_component, fields, filter_condition)
            if "error" in sqlite_data:
                return sqlite_data
            
            # Get DB2 data
            db2_data = await self._get_db2_data(db2_component, fields, filter_condition)
            if "error" in db2_data:
                return db2_data
            
            # Perform comparison
            comparison_result = await self._perform_data_comparison(
                sqlite_data, db2_data, sqlite_component, db2_component
            )
            
            # Store comparison history
            await self._store_comparison_result(
                sqlite_component, db2_component, comparison_result
            )
            
            self.logger.info(f"Comparison completed with {comparison_result.get('match_percentage', 0):.2f}% match")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Data comparison failed: {str(e)}")
            return {"error": f"Data comparison failed: {str(e)}"}
    
    async def _get_sqlite_data(self, component_name: str, fields: List[str] = None, 
                              filter_condition: str = None) -> Dict[str, Any]:
        """Get data from SQLite component with proper error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (component_name,))
                
                if not cursor.fetchone():
                    return {"error": f"SQLite table '{component_name}' not found"}
                
                # Get table schema first
                cursor.execute(f"PRAGMA table_info({component_name})")
                table_columns = [column[1] for column in cursor.fetchall()]
                
                # Validate requested fields
                if fields:
                    invalid_fields = [f for f in fields if f not in table_columns]
                    if invalid_fields:
                        return {"error": f"Invalid fields: {invalid_fields}. Available fields: {table_columns}"}
                    field_list = ", ".join([f'"{field}"' for field in fields])
                else:
                    field_list = "*"
                
                # Build base query
                base_query = f'SELECT {field_list} FROM "{component_name}"'
                
                # Add filter condition if provided (basic validation)
                if filter_condition:
                    # Simple validation - check for dangerous keywords
                    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'TRUNCATE']
                    if any(keyword in filter_condition.upper() for keyword in dangerous_keywords):
                        return {"error": "Invalid filter condition - potentially dangerous SQL detected"}
                    base_query += f" WHERE {filter_condition}"
                
                base_query += f" LIMIT {self.max_rows}"
                
                # Execute query
                df = pd.read_sql_query(base_query, conn)
                
                return {
                    "data": df,
                    "record_count": len(df),
                    "columns": df.columns.tolist(),
                    "source": "sqlite",
                    "table_name": component_name
                }
                
        except Exception as e:
            self.logger.error(f"SQLite data retrieval failed for '{component_name}': {str(e)}")
            return {"error": f"SQLite data retrieval failed: {str(e)}"}
    
    async def _get_db2_data(self, component_name: str, fields: List[str] = None, 
                           filter_condition: str = None) -> Dict[str, Any]:
        """Get data from DB2 component"""
        if not DB2_AVAILABLE:
            # Simulate DB2 data for demonstration
            return await self._simulate_db2_data(component_name, fields, filter_condition)
        
        try:
            # Connect to DB2
            async with self._db2_connection() as db2_conn:
                if not db2_conn:
                    return {"error": "Failed to connect to DB2"}
                
                # Get table schema first
                schema_query = """
                    SELECT COLNAME 
                    FROM SYSCAT.COLUMNS 
                    WHERE TABNAME = ? 
                    ORDER BY COLNO
                """
                
                df_schema = pd.read_sql(schema_query, db2_conn, params=[component_name.upper()])
                
                if df_schema.empty:
                    return {"error": f"DB2 table '{component_name}' not found"}
                
                table_columns = df_schema['COLNAME'].tolist()
                
                # Validate requested fields
                if fields:
                    invalid_fields = [f for f in fields if f.upper() not in [col.upper() for col in table_columns]]
                    if invalid_fields:
                        return {"error": f"Invalid fields: {invalid_fields}. Available fields: {table_columns}"}
                    field_list = ", ".join([f'"{field}"' for field in fields])
                else:
                    field_list = "*"
                
                # Build query
                query = f'SELECT {field_list} FROM "{component_name.upper()}"'
                
                if filter_condition:
                    # Simple validation for DB2
                    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'TRUNCATE']
                    if any(keyword in filter_condition.upper() for keyword in dangerous_keywords):
                        return {"error": "Invalid filter condition - potentially dangerous SQL detected"}
                    query += f" WHERE {filter_condition}"
                
                query += f" FETCH FIRST {self.max_rows} ROWS ONLY"
                
                # Execute query
                df = pd.read_sql(query, db2_conn)
                
                return {
                    "data": df,
                    "record_count": len(df),
                    "columns": df.columns.tolist(),
                    "source": "db2",
                    "table_name": component_name
                }
                
        except Exception as e:
            self.logger.error(f"DB2 data retrieval failed for '{component_name}': {str(e)}")
            return {"error": f"DB2 data retrieval failed: {str(e)}"}
    
    async def _simulate_db2_data(self, component_name: str, fields: List[str] = None, 
                                filter_condition: str = None) -> Dict[str, Any]:
        """Simulate DB2 data for demonstration purposes"""
        try:
            self.logger.info(f"Simulating DB2 data for '{component_name}'")
            
            # Get SQLite data to base simulation on
            sqlite_data = await self._get_sqlite_data(component_name, fields, filter_condition)
            
            if "error" in sqlite_data:
                return {"error": f"Cannot simulate DB2 data: {sqlite_data['error']}"}
            
            df_sqlite = sqlite_data["data"]
            
            # Create simulated DB2 data with some variations
            df_db2 = df_sqlite.copy()
            
            # Introduce some intentional differences for demonstration
            if len(df_db2) > 0:
                # Modify 5% of records to show differences
                num_changes = max(1, min(len(df_db2) // 20, 10))  # Cap at 10 changes
                
                for i in range(num_changes):
                    row_idx = i % len(df_db2)
                    
                    # Modify a random numeric column if available
                    numeric_cols = df_db2.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        col = numeric_cols[0]
                        current_value = df_db2.iloc[row_idx, df_db2.columns.get_loc(col)]
                        if pd.notna(current_value):
                            df_db2.iloc[row_idx, df_db2.columns.get_loc(col)] = current_value + 1
            
            return {
                "data": df_db2,
                "record_count": len(df_db2),
                "columns": df_db2.columns.tolist(),
                "source": "db2_simulated",
                "table_name": component_name
            }
            
        except Exception as e:
            self.logger.error(f"DB2 simulation failed for '{component_name}': {str(e)}")
            return {"error": f"DB2 simulation failed: {str(e)}"}
    
    @asynccontextmanager
    async def _db2_connection(self):
        """Async context manager for DB2 connections"""
        connection = None
        start_time = dt.now()
        
        try:
            if not DB2_AVAILABLE:
                yield None
                return
            
            dsn = (
                f"DATABASE={self.db2_config['database']};"
                f"HOSTNAME={self.db2_config['hostname']};"
                f"PORT={self.db2_config['port']};"
                f"PROTOCOL=TCPIP;"
                f"UID={self.db2_config['username']};"
                f"PWD={self.db2_config['password']};"
            )
            
            conn = ibm_db.connect(dsn, "", "")
            connection = ibm_db_dbi.Connection(conn)
            
            response_time = (dt.now() - start_time).total_seconds() * 1000
            self._log_connection_attempt(True, response_time_ms=int(response_time))
            
            yield connection
            
        except Exception as e:
            response_time = (dt.now() - start_time).total_seconds() * 1000
            self.logger.error(f"DB2 connection failed: {str(e)}")
            self._log_connection_attempt(False, str(e), int(response_time))
            yield None
            
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass
    
    def _log_connection_attempt(self, success: bool, error_message: str = None, 
                               response_time_ms: int = None, operation_type: str = "data_comparison"):
        """Log DB2 connection attempt"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO db2_connection_log (success, error_message, operation_type, response_time_ms)
                    VALUES (?, ?, ?, ?)
                """, (success, error_message, operation_type, response_time_ms))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log connection attempt: {str(e)}")
    
    async def _perform_data_comparison(self, sqlite_data: Dict, db2_data: Dict, 
                                     sqlite_component: str, db2_component: str) -> Dict[str, Any]:
        """Perform detailed data comparison"""
        try:
            df_sqlite = sqlite_data["data"]
            df_db2 = db2_data["data"]
            
            # Basic statistics
            sqlite_count = len(df_sqlite)
            db2_count = len(df_db2)
            
            # Column comparison
            sqlite_cols = set(df_sqlite.columns)
            db2_cols = set(df_db2.columns)
            
            common_columns = sqlite_cols & db2_cols
            sqlite_only_cols = sqlite_cols - db2_cols
            db2_only_cols = db2_cols - sqlite_cols
            
            if not common_columns:
                return {
                    "error": "No common columns found between datasets",
                    "sqlite_columns": list(sqlite_cols),
                    "db2_columns": list(db2_cols),
                    "match_percentage": 0,
                    "differences_count": max(sqlite_count, db2_count)
                }
            
            # Perform record-by-record comparison on common columns
            comparison_details = await self._compare_records(
                df_sqlite[list(common_columns)], 
                df_db2[list(common_columns)],
                list(common_columns)
            )
            
            # Generate comparison summary using LLM
            comparison_summary = await self._generate_comparison_summary(
                sqlite_component, db2_component, comparison_details
            )
            
            return {
                "sqlite_component": sqlite_component,
                "db2_component": db2_component,
                "sqlite_count": sqlite_count,
                "db2_count": db2_count,
                "common_columns": list(common_columns),
                "sqlite_only_columns": list(sqlite_only_cols),
                "db2_only_columns": list(db2_only_cols),
                "comparison_details": comparison_details,
                "comparison_summary": comparison_summary,
                "match_percentage": comparison_details.get("match_percentage", 0),
                "differences_count": comparison_details.get("differences_count", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Data comparison failed: {str(e)}")
            return {"error": f"Data comparison failed: {str(e)}"}
    
    async def _compare_records(self, df_sqlite: pd.DataFrame, df_db2: pd.DataFrame, 
                              columns: List[str]) -> Dict[str, Any]:
        """Compare records between two dataframes"""
        try:
            # Ensure both dataframes have the same columns in the same order
            df_sqlite = df_sqlite[columns].copy()
            df_db2 = df_db2[columns].copy()
            
            # Handle NaN values consistently
            df_sqlite = df_sqlite.fillna('__NULL__')
            df_db2 = df_db2.fillna('__NULL__')
            
            # Create composite keys for comparison
            def create_key(row):
                # Convert to string and handle floating point precision
                row_str = '|'.join([str(val) for val in row.values])
                return hashlib.md5(row_str.encode('utf-8')).hexdigest()
            
            sqlite_keys = df_sqlite.apply(create_key, axis=1)
            db2_keys = df_db2.apply(create_key, axis=1)
            
            # Find matching and differing records
            sqlite_key_counts = sqlite_keys.value_counts()
            db2_key_counts = db2_keys.value_counts()
            
            # Calculate matches considering duplicates
            matching_count = 0
            for key in sqlite_key_counts.index:
                if key in db2_key_counts.index:
                    matching_count += min(sqlite_key_counts[key], db2_key_counts[key])
            
            sqlite_only_count = len(df_sqlite) - matching_count
            db2_only_count = len(df_db2) - matching_count
            
            # Calculate match percentage
            total_records = max(len(df_sqlite), len(df_db2))
            match_percentage = (matching_count / total_records * 100) if total_records > 0 else 0
            
            # Get sample records for analysis
            sample_data = self._get_sample_records(df_sqlite, df_db2, sqlite_keys, db2_keys)
            
            return {
                "matching_count": matching_count,
                "sqlite_only_count": sqlite_only_count,
                "db2_only_count": db2_only_count,
                "match_percentage": match_percentage,
                "differences_count": sqlite_only_count + db2_only_count,
                "sample_matching_records": sample_data["matching"],
                "sample_differences": sample_data["differences"]
            }
            
        except Exception as e:
            self.logger.error(f"Record comparison failed: {str(e)}")
            return {"error": f"Record comparison failed: {str(e)}"}
    
    def _get_sample_records(self, df_sqlite: pd.DataFrame, df_db2: pd.DataFrame, 
                           sqlite_keys: pd.Series, db2_keys: pd.Series) -> Dict[str, List]:
        """Get sample records for analysis"""
        try:
            matching_records = []
            differences = []
            
            # Find matching records
            common_keys = set(sqlite_keys) & set(db2_keys)
            if common_keys:
                sample_keys = list(common_keys)[:3]  # Sample of 3
                for key in sample_keys:
                    sqlite_idx = sqlite_keys[sqlite_keys == key].index[0]
                    record = df_sqlite.iloc[sqlite_idx].to_dict()
                    # Convert __NULL__ back to None for display
                    record = {k: (None if v == '__NULL__' else v) for k, v in record.items()}
                    matching_records.append(record)
            
            # Find SQLite-only records
            sqlite_only_keys = set(sqlite_keys) - set(db2_keys)
            if sqlite_only_keys:
                sample_keys = list(sqlite_only_keys)[:3]  # Sample of 3
                for key in sample_keys:
                    sqlite_idx = sqlite_keys[sqlite_keys == key].index[0]
                    record = df_sqlite.iloc[sqlite_idx].to_dict()
                    record = {k: (None if v == '__NULL__' else v) for k, v in record.items()}
                    differences.append({
                        "type": "sqlite_only",
                        "record": record
                    })
            
            # Find DB2-only records
            db2_only_keys = set(db2_keys) - set(sqlite_keys)
            if db2_only_keys:
                sample_keys = list(db2_only_keys)[:3]  # Sample of 3
                for key in sample_keys:
                    db2_idx = db2_keys[db2_keys == key].index[0]
                    record = df_db2.iloc[db2_idx].to_dict()
                    record = {k: (None if v == '__NULL__' else v) for k, v in record.items()}
                    differences.append({
                        "type": "db2_only", 
                        "record": record
                    })
            
            return {
                "matching": matching_records,
                "differences": differences
            }
            
        except Exception as e:
            self.logger.error(f"Sample record extraction failed: {str(e)}")
            return {"matching": [], "differences": []}
    
    async def _generate_comparison_summary(self, sqlite_comp: str, db2_comp: str, 
                                         details: Dict) -> str:
        """Generate comparison summary using LLM"""
        try:
            prompt = f"""
            Generate a professional data comparison summary for SQLite component '{sqlite_comp}' and DB2 component '{db2_comp}':
            
            Comparison Results:
            - Matching Records: {details.get('matching_count', 0)}
            - SQLite Only Records: {details.get('sqlite_only_count', 0)}
            - DB2 Only Records: {details.get('db2_only_count', 0)}
            - Match Percentage: {details.get('match_percentage', 0):.2f}%
            - Total Differences: {details.get('differences_count', 0)}
            
            Provide:
            1. Overall data consistency assessment
            2. Potential causes for differences
            3. Data quality implications
            4. Recommended actions
            5. Risk assessment for data discrepancies
            
            Format as a concise professional summary (max 500 words).
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=800)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
            
            return result.outputs[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return f"Summary generation failed: {str(e)}"
    
    async def _store_comparison_result(self, sqlite_comp: str, db2_comp: str, result: Dict):
        """Store comparison result in history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO comparison_history 
                    (sqlite_component, db2_component, comparison_type, match_percentage, 
                     differences_count, comparison_details, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sqlite_comp,
                    db2_comp,
                    "data_comparison",
                    result.get("match_percentage", 0),
                    result.get("differences_count", 0),
                    json.dumps(result, default=str),
                    "completed" if "error" not in result else "failed"
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store comparison result: {str(e)}")
    
    async def get_comparison_history(self, component_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get comparison history with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if component_name:
                    cursor.execute("""
                        SELECT * FROM comparison_history 
                        WHERE sqlite_component = ? OR db2_component = ?
                        ORDER BY comparison_timestamp DESC
                        LIMIT ?
                    """, (component_name, component_name, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM comparison_history 
                        ORDER BY comparison_timestamp DESC
                        LIMIT ?
                    """, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                history = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    if record["comparison_details"]:
                        try:
                            record["comparison_details"] = json.loads(record["comparison_details"])
                        except json.JSONDecodeError:
                            pass
                    history.append(record)
                
                return history
                
        except Exception as e:
            self.logger.error(f"Failed to get comparison history: {str(e)}")
            return []
    
    async def validate_data_quality(self, component_name: str, validation_rules: List[Dict] = None) -> Dict[str, Any]:
        """Validate data quality for a component"""
        try:
            # Get component data
            data_result = await self._get_sqlite_data(component_name)
            if "error" in data_result:
                return data_result
            
            df = data_result["data"]
            
            # Apply validation rules
            validation_results = []
            
            # Default validation rules
            default_rules = [
                {"rule": "null_check", "description": "Check for null values", "threshold": 5.0},
                {"rule": "duplicate_check", "description": "Check for duplicate records", "threshold": 1.0},
                {"rule": "data_type_check", "description": "Check data type consistency"},
                {"rule": "range_check", "description": "Check for reasonable value ranges", "threshold": 5.0}
            ]
            
            rules_to_apply = validation_rules or default_rules
            
            for rule in rules_to_apply:
                rule_result = await self._apply_validation_rule(df, rule)
                validation_results.append(rule_result)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(validation_results)
            
            # Generate quality report
            quality_report = await self._generate_quality_report(
                component_name, validation_results, quality_score
            )
            
            return {
                "component_name": component_name,
                "record_count": len(df),
                "validation_results": validation_results,
                "quality_score": quality_score,
                "quality_report": quality_report,
                "timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {str(e)}")
            return {"error": f"Data quality validation failed: {str(e)}"}
    
    async def _apply_validation_rule(self, df: pd.DataFrame, rule: Dict) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        rule_type = rule["rule"]
        threshold = rule.get("threshold", 5.0)
        
        try:
            if rule_type == "null_check":
                null_counts = df.isnull().sum()
                total_cells = len(df) * len(df.columns)
                null_percentage = (null_counts.sum() / total_cells) * 100 if total_cells > 0 else 0
                
                return {
                    "rule": rule_type,
                    "description": rule["description"],
                    "status": "PASS" if null_percentage < threshold else "FAIL",
                    "score": max(0, 100 - null_percentage),
                    "details": {
                        "null_percentage": round(null_percentage, 2),
                        "threshold": threshold,
                        "null_counts_by_column": null_counts.to_dict()
                    }
                }
            
            elif rule_type == "duplicate_check":
                duplicate_count = df.duplicated().sum()
                duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
                
                return {
                    "rule": rule_type,
                    "description": rule["description"],
                    "status": "PASS" if duplicate_percentage < threshold else "FAIL",
                    "score": max(0, 100 - duplicate_percentage * 10),
                    "details": {
                        "duplicate_count": duplicate_count,
                        "duplicate_percentage": round(duplicate_percentage, 2),
                        "threshold": threshold
                    }
                }
            
            elif rule_type == "data_type_check":
                type_issues = []
                for column in df.columns:
                    if df[column].dtype == 'object':
                        # Check for mixed types in object columns
                        sample_types = df[column].dropna().apply(type).value_counts()
                        if len(sample_types) > 1:
                            type_issues.append({
                                "column": column,
                                "issue": "mixed_types",
                                "types_found": [str(t) for t in sample_types.index]
                            })
                
                return {
                    "rule": rule_type,
                    "description": rule["description"],
                    "status": "PASS" if not type_issues else "FAIL",
                    "score": 100 if not type_issues else max(0, 100 - len(type_issues) * 20),
                    "details": {
                        "type_issues": type_issues,
                        "issues_count": len(type_issues)
                    }
                }
            
            elif rule_type == "range_check":
                range_issues = []
                numeric_columns = df.select_dtypes(include=['number']).columns
                
                for column in numeric_columns:
                    col_data = df[column].dropna()
                    if len(col_data) > 0:
                        q1 = col_data.quantile(0.25)
                        q3 = col_data.quantile(0.75)
                        iqr = q3 - q1
                        
                        if iqr > 0:  # Avoid division by zero
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                            outlier_percentage = (len(outliers) / len(col_data)) * 100
                            
                            if outlier_percentage > threshold:
                                range_issues.append({
                                    "column": column,
                                    "outlier_count": len(outliers),
                                    "outlier_percentage": round(outlier_percentage, 2),
                                    "lower_bound": lower_bound,
                                    "upper_bound": upper_bound
                                })
                
                return {
                    "rule": rule_type,
                    "description": rule["description"],
                    "status": "PASS" if not range_issues else "WARNING",
                    "score": 100 if not range_issues else max(0, 100 - len(range_issues) * 15),
                    "details": {
                        "range_issues": range_issues,
                        "issues_count": len(range_issues),
                        "threshold": threshold
                    }
                }
            
            else:
                return {
                    "rule": rule_type,
                    "description": rule["description"],
                    "status": "SKIPPED",
                    "score": 0,
                    "details": {"error": "Unknown rule type"}
                }
                
        except Exception as e:
            self.logger.error(f"Validation rule '{rule_type}' failed: {str(e)}")
            return {
                "rule": rule_type,
                "description": rule["description"],
                "status": "ERROR",
                "score": 0,
                "details": {"error": str(e)}
            }
    
    def _calculate_quality_score(self, validation_results: List[Dict]) -> float:
        """Calculate overall data quality score"""
        if not validation_results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # Weight different rule types
        rule_weights = {
            "null_check": 1.0,
            "duplicate_check": 0.8,
            "data_type_check": 0.9,
            "range_check": 0.6
        }
        
        for result in validation_results:
            rule_type = result["rule"]
            score = result.get("score", 0)
            weight = rule_weights.get(rule_type, 1.0)
            
            total_score += score * weight
            total_weight += weight
        
        return round(total_score / total_weight, 2) if total_weight > 0 else 0.0
    
    async def _generate_quality_report(self, component_name: str, 
                                     validation_results: List[Dict], quality_score: float) -> str:
        """Generate data quality report using LLM"""
        try:
            prompt = f"""
            Generate a data quality assessment report for component '{component_name}':
            
            Overall Quality Score: {quality_score:.2f}/100
            
            Validation Results Summary:
            {json.dumps([{
                'rule': r['rule'], 
                'status': r['status'], 
                'score': r.get('score', 0),
                'details': r.get('details', {})
            } for r in validation_results], indent=2)}
            
            Provide:
            1. Executive summary of data quality
            2. Key issues identified and their impact
            3. Recommended remediation actions (prioritized)
            4. Data governance considerations
            5. Quality improvement recommendations
            
            Format as a professional assessment (max 600 words).
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=1000)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
            
            return result.outputs[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Quality report generation failed: {str(e)}")
            return f"Quality report generation failed: {str(e)}"
    
    async def reconcile_data_differences(self, sqlite_component: str, db2_component: str) -> Dict[str, Any]:
        """Analyze and suggest reconciliation for data differences"""
        try:
            # Get latest comparison
            comparison_history = await self.get_comparison_history(sqlite_component, limit=1)
            
            if not comparison_history:
                return {"error": "No comparison history found. Please run comparison first."}
            
            latest_comparison = comparison_history[0]
            comparison_details = latest_comparison.get("comparison_details", {})
            
            # Check if this is the right comparison
            if (latest_comparison.get("sqlite_component") != sqlite_component or 
                latest_comparison.get("db2_component") != db2_component):
                return {"error": f"No recent comparison found for {sqlite_component} vs {db2_component}"}
            
            # Analyze differences
            differences_count = comparison_details.get("differences_count", 0)
            match_percentage = comparison_details.get("match_percentage", 0)
            
            if differences_count == 0:
                return {
                    "message": "No differences found between datasets",
                    "reconciliation_needed": False,
                    "match_percentage": match_percentage
                }
            
            # Generate reconciliation plan
            reconciliation_plan = await self._generate_reconciliation_plan(
                sqlite_component, db2_component, comparison_details
            )
            
            return {
                "sqlite_component": sqlite_component,
                "db2_component": db2_component,
                "differences_analyzed": differences_count,
                "match_percentage": match_percentage,
                "reconciliation_plan": reconciliation_plan,
                "reconciliation_needed": True,
                "comparison_timestamp": latest_comparison.get("comparison_timestamp")
            }
            
        except Exception as e:
            self.logger.error(f"Reconciliation analysis failed: {str(e)}")
            return {"error": f"Reconciliation analysis failed: {str(e)}"}
    
    async def _generate_reconciliation_plan(self, sqlite_comp: str, db2_comp: str, 
                                          comparison_details: Dict) -> str:
        """Generate reconciliation plan using LLM"""
        try:
            prompt = f"""
            Generate a data reconciliation plan for differences between '{sqlite_comp}' and '{db2_comp}':
            
            Comparison Summary:
            - Match Percentage: {comparison_details.get('match_percentage', 0):.2f}%
            - Total Differences: {comparison_details.get('differences_count', 0)}
            - SQLite Only Records: {comparison_details.get('sqlite_only_count', 0)}
            - DB2 Only Records: {comparison_details.get('db2_only_count', 0)}
            - Matching Records: {comparison_details.get('matching_count', 0)}
            
            Sample Differences (if available):
            {json.dumps(comparison_details.get('sample_differences', [])[:3], indent=2)}
            
            Provide:
            1. Root cause analysis for the differences
            2. Step-by-step reconciliation procedure
            3. Data validation checkpoints
            4. Risk mitigation strategies
            5. Testing and verification steps
            6. Timeline and resource recommendations
            
            Format as a detailed technical reconciliation plan (max 800 words).
            """
            
            sampling_params = SamplingParams(temperature=0.3, max_tokens=1200)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
            
            return result.outputs[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Reconciliation plan generation failed: {str(e)}")
            return f"Reconciliation plan generation failed: {str(e)}"
    
    async def get_db2_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get DB2 table schema information"""
        if not DB2_AVAILABLE:
            return {"error": "DB2 drivers not available"}
        
        try:
            async with self._db2_connection() as db2_conn:
                if not db2_conn:
                    return {"error": "Failed to connect to DB2"}
                
                # Query DB2 system catalog for table schema
                schema_query = """
                    SELECT 
                        COLNAME as column_name,
                        TYPENAME as data_type,
                        LENGTH as column_length,
                        SCALE as decimal_scale,
                        NULLS as nullable,
                        DEFAULT as default_value,
                        REMARKS as column_comment,
                        COLNO as column_order
                    FROM SYSCAT.COLUMNS 
                    WHERE TABNAME = ?
                    ORDER BY COLNO
                """
                
                df_schema = pd.read_sql(schema_query, db2_conn, params=[table_name.upper()])
                
                if df_schema.empty:
                    return {"error": f"Table '{table_name}' not found in DB2"}
                
                # Get table statistics
                stats_query = """
                    SELECT 
                        CARD as row_count,
                        NPAGES as page_count,
                        STATS_TIME as last_stats_update
                    FROM SYSCAT.TABLES 
                    WHERE TABNAME = ?
                """
                
                df_stats = pd.read_sql(stats_query, db2_conn, params=[table_name.upper()])
                
                schema_info = {
                    "table_name": table_name,
                    "column_count": len(df_schema),
                    "columns": df_schema.to_dict('records'),
                    "statistics": df_stats.to_dict('records')[0] if not df_stats.empty else {}
                }
                
                return schema_info
                
        except Exception as e:
            self.logger.error(f"Schema retrieval failed for '{table_name}': {str(e)}")
            return {"error": f"Schema retrieval failed: {str(e)}"}
    
    async def generate_comparison_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for comparison dashboard"""
        try:
            # Get comparison history summary
            history = await self.get_comparison_history(limit=100)
            
            if not history:
                return {"message": "No comparison data available"}
            
            # Analyze trends and metrics
            dashboard_data = {
                "summary": {
                    "total_comparisons": len(history),
                    "successful_comparisons": len([h for h in history if h.get("status") == "completed"]),
                    "average_match_rate": 0,
                    "last_comparison": None
                },
                "recent_comparisons": history[:10],
                "match_rate_trends": [],
                "component_activity": {},
                "quality_metrics": {
                    "high_match_rate": 0,  # >95%
                    "medium_match_rate": 0,  # 80-95%
                    "low_match_rate": 0   # <80%
                }
            }
            
            # Calculate metrics
            if history:
                dashboard_data["summary"]["last_comparison"] = history[0]["comparison_timestamp"]
                
                match_rates = [h["match_percentage"] for h in history if h.get("match_percentage") is not None]
                if match_rates:
                    dashboard_data["summary"]["average_match_rate"] = round(sum(match_rates) / len(match_rates), 2)
                
                # Categorize by match rate
                for rate in match_rates:
                    if rate >= 95:
                        dashboard_data["quality_metrics"]["high_match_rate"] += 1
                    elif rate >= 80:
                        dashboard_data["quality_metrics"]["medium_match_rate"] += 1
                    else:
                        dashboard_data["quality_metrics"]["low_match_rate"] += 1
            
            # Match rate trends (last 30 comparisons)
            recent_history = history[:30]
            for comparison in reversed(recent_history):
                if comparison.get("match_percentage") is not None:
                    dashboard_data["match_rate_trends"].append({
                        "timestamp": comparison["comparison_timestamp"],
                        "match_percentage": comparison["match_percentage"],
                        "comparison_id": comparison["id"]
                    })
            
            # Component activity analysis
            component_counts = {}
            for comparison in history:
                sqlite_comp = comparison["sqlite_component"]
                db2_comp = comparison["db2_component"]
                
                component_counts[sqlite_comp] = component_counts.get(sqlite_comp, 0) + 1
                if sqlite_comp != db2_comp:  # Avoid double counting if same name
                    component_counts[db2_comp] = component_counts.get(db2_comp, 0) + 1
            
            dashboard_data["component_activity"] = dict(
                sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            )
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {str(e)}")
            return {"error": f"Dashboard data generation failed: {str(e)}"}
    
    def configure_db2_connection(self, config: Dict[str, str]):
        """Configure DB2 connection parameters"""
        # Validate configuration
        required_keys = ["database", "hostname", "port", "username", "password"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        self.db2_config.update(config)
        self.logger.info("DB2 configuration updated successfully")
    
    async def test_db2_connection(self) -> Dict[str, Any]:
        """Test DB2 connection"""
        try:
            start_time = dt.now()
            
            async with self._db2_connection() as conn:
                if conn:
                    # Test with a simple query
                    test_query = "SELECT 1 FROM SYSIBM.SYSDUMMY1"
                    result = pd.read_sql(test_query, conn)
                    
                    response_time = (dt.now() - start_time).total_seconds() * 1000
                    
                    return {
                        "status": "success", 
                        "message": "DB2 connection successful",
                        "response_time_ms": int(response_time),
                        "test_result": result.iloc[0, 0] if not result.empty else None
                    }
                else:
                    return {
                        "status": "failed", 
                        "message": "Failed to establish DB2 connection",
                        "response_time_ms": (dt.now() - start_time).total_seconds() * 1000
                    }
                    
        except Exception as e:
            response_time = (dt.now() - start_time).total_seconds() * 1000
            self.logger.error(f"DB2 connection test failed: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "response_time_ms": int(response_time)
            }
    
    async def get_connection_health(self) -> Dict[str, Any]:
        """Get connection health statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent connection attempts
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts,
                        AVG(response_time_ms) as avg_response_time,
                        MAX(connection_attempt) as last_attempt
                    FROM db2_connection_log 
                    WHERE connection_attempt >= datetime('now', '-24 hours')
                """)
                
                stats = cursor.fetchone()
                
                if stats[0] > 0:  # If there are attempts
                    success_rate = (stats[1] / stats[0]) * 100
                    return {
                        "total_attempts_24h": stats[0],
                        "successful_attempts_24h": stats[1],
                        "success_rate_percentage": round(success_rate, 2),
                        "average_response_time_ms": round(stats[2], 2) if stats[2] else 0,
                        "last_attempt": stats[3],
                        "status": "healthy" if success_rate > 80 else "degraded" if success_rate > 50 else "unhealthy"
                    }
                else:
                    return {
                        "message": "No connection attempts in the last 24 hours",
                        "status": "unknown"
                    }
                    
        except Exception as e:
            self.logger.error(f"Connection health check failed: {str(e)}")
            return {"error": f"Connection health check failed: {str(e)}"}
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up old comparison data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records to be deleted
                cursor.execute("""
                    SELECT COUNT(*) FROM comparison_history 
                    WHERE comparison_timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                records_to_delete = cursor.fetchone()[0]
                
                # Delete old comparison history
                cursor.execute("""
                    DELETE FROM comparison_history 
                    WHERE comparison_timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                # Delete old connection logs
                cursor.execute("""
                    DELETE FROM db2_connection_log 
                    WHERE connection_attempt < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                connection_logs_deleted = cursor.rowcount
                
                conn.commit()
                
                return {
                    "comparison_records_deleted": records_to_delete,
                    "connection_logs_deleted": connection_logs_deleted,
                    "cleanup_date": dt.now().isoformat(),
                    "days_kept": days_to_keep
                }
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {str(e)}")
            return {"error": f"Data cleanup failed: {str(e)}"}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration"""
        return {
            "agent_name": "DB2 Comparator Agent",
            "version": "1.0.0",
            "db2_available": DB2_AVAILABLE,
            "max_rows_limit": self.max_rows,
            "db_path": self.db_path,
            "gpu_id": self.gpu_id,
            "db2_config": {
                "database": self.db2_config["database"],
                "hostname": self.db2_config["hostname"],
                "port": self.db2_config["port"],
                "username": self.db2_config["username"],
                "password": "***masked***"
            },
            "status": "ready"
        }