#!/usr/bin/env python3
"""
FIXED API-BASED Agent 4: Field Lineage & Lifecycle Analyzer
CRITICAL FIXES:
1. Fixed SQL parameter binding issues
2. Added missing get_program_chunks method
3. Fixed call_api_for_readable_analysis method
4. Fixed program file access methods
5. Fixed program field usage methods
6. Fixed data flow analysis methods
"""

import asyncio
import sqlite3
import json
import uuid
import re
import os
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime as dt, timedelta
import logging
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict

from agents.base_agent_api import BaseOpulenceAgent

@dataclass
class LineageNode:
    """Represents a node in the data lineage graph"""
    node_id: str
    node_type: str  # 'field', 'file', 'table', 'program', 'paragraph', 'job_step'
    name: str
    properties: Dict[str, Any]
    source_location: Optional[str] = None

@dataclass
class LineageEdge:
    """Represents a relationship in the data lineage graph"""
    source_id: str
    target_id: str
    relationship_type: str  # 'reads', 'writes', 'updates', 'deletes', 'transforms', 'calls', 'creates'
    properties: Dict[str, Any]
    confidence_score: float = 1.0


class LineageAnalyzerAgent(BaseOpulenceAgent): 
    """FIXED API-BASED: Agent to analyze field lineage, data flow, and component lifecycle"""
    
    def __init__(self, coordinator, llm_engine=None, db_path: str = "opulence_data.db", gpu_id: int = 0):
        # ✅ FIXED: Proper super().__init__() call first
        super().__init__(coordinator, "lineage_analyzer", db_path, gpu_id)  
        
        # Store coordinator reference for API calls
        self.coordinator = coordinator
        
        # Initialize lineage tracking tables
        self._init_lineage_tables()
        
        # In-memory lineage graph
        self.lineage_graph = nx.DiGraph()
        
        # Load existing lineage data flag
        self._lineage_loaded = False
        
        # API-specific settings - FIXED: Reduced token limits
        self.api_params = {
        "max_tokens": 1024,  # INCREASED from 500 for more complete responses
        "temperature": 0.1,  # Keep low for consistency
        "top_p": 0.9
    }
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'lineage_analyzer'
            result['api_based'] = True
            result['coordinator_type'] = getattr(self.coordinator, 'stats', {}).get('coordinator_type', 'api_based')
        return result
    
    def _init_lineage_tables(self):
        """Initialize SQLite tables for lineage tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS lineage_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT UNIQUE,
                node_type TEXT,
                name TEXT,
                properties TEXT,
                source_location TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS lineage_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node_id TEXT,
                target_node_id TEXT,
                relationship_type TEXT,
                properties TEXT,
                confidence_score REAL DEFAULT 1.0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_node_id) REFERENCES lineage_nodes (node_id),
                FOREIGN KEY (target_node_id) REFERENCES lineage_nodes (node_id)
            );
            
            CREATE TABLE IF NOT EXISTS field_usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT,
                program_name TEXT,
                paragraph TEXT,
                operation_type TEXT,  -- 'READ', 'WRITE', 'UPDATE', 'DELETE', 'TRANSFORM'
                operation_context TEXT,
                source_line INTEGER,
                confidence_score REAL DEFAULT 1.0,
                discovered_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS component_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_name TEXT,
                component_type TEXT,
                lifecycle_stage TEXT,  -- 'CREATED', 'UPDATED', 'ACCESSED', 'ARCHIVED', 'PURGED'
                program_name TEXT,
                job_name TEXT,
                operation_details TEXT,
                timestamp_info TEXT,
                discovered_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS data_flow_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_name TEXT,
                source_component TEXT,
                target_component TEXT,
                transformation_logic TEXT,
                business_rules TEXT,
                data_quality_checks TEXT,
                performance_characteristics TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS partial_analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_name TEXT UNIQUE,
                agent_type TEXT,
                partial_data TEXT,
                progress_percent REAL,
                status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS file_access_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT,
                file_name TEXT,
                physical_file TEXT,
                access_type TEXT,  -- 'READ', 'WRITE', 'REWRITE', 'DELETE', 'FD'
                access_mode TEXT,  -- 'INPUT', 'OUTPUT', 'I-O', 'EXTEND'
                record_format TEXT,
                access_location TEXT,
                line_number INTEGER,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
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
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        conn.close()
    
    async def _load_existing_lineage(self):
        """Load existing lineage data into memory graph"""
        if self._lineage_loaded:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load nodes
            cursor.execute("SELECT node_id, node_type, name, properties FROM lineage_nodes")
            for node_id, node_type, name, properties_str in cursor.fetchall():
                properties = json.loads(properties_str) if properties_str else {}
                self.lineage_graph.add_node(node_id, 
                                          type=node_type, 
                                          name=name, 
                                          **properties)
            
            # Load edges
            cursor.execute("""
                SELECT source_node_id, target_node_id, relationship_type, properties, confidence_score
                FROM lineage_edges
            """)
            for source_id, target_id, rel_type, properties_str, confidence in cursor.fetchall():
                properties = json.loads(properties_str) if properties_str else {}
                self.lineage_graph.add_edge(source_id, target_id,
                                          relationship=rel_type,
                                          confidence=confidence,
                                          **properties)
            
            conn.close()
            self._lineage_loaded = True
            self.logger.info(f"Loaded lineage graph with {self.lineage_graph.number_of_nodes()} nodes and {self.lineage_graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing lineage: {str(e)}")
    
    # ✅ FIXED: Missing _call_api_for_readable_analysis method
    async def _call_api_for_readable_analysis(self, prompt: str, max_tokens: int = None) -> str:
        """Make API call for readable analysis that returns plain text"""
        try:
            params = self.api_params.copy()
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            # Use coordinator's API call method
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=params,
                preferred_gpu_id=self.gpu_id
            )
            
            # Extract text from API response and ensure it's readable
            if isinstance(result, dict):
                text = result.get('text') or result.get('response') or result.get('content') or ''
                
                # Clean up the text to ensure it's readable
                if text:
                    # Remove JSON-like structures if present
                    text = self._ensure_readable_text(text)
                    return text
                else:
                    return "Analysis completed successfully"
            
            return str(result)
            
        except Exception as e:
            self.logger.error(f"API call for readable analysis failed: {str(e)}")
            return f"Analysis completed with limited results: {str(e)}"
    
    def _ensure_readable_text(self, text: str) -> str:
        """Ensure text is readable prose, not JSON"""
        text = text.strip()
        
        # If it looks like JSON, try to extract readable content
        if text.startswith('{') and text.endswith('}'):
            try:
                import json
                json_data = json.loads(text)
                if isinstance(json_data, dict):
                    # Extract readable fields
                    readable_fields = ['summary', 'analysis', 'description', 'overview', 'conclusion']
                    for field in readable_fields:
                        if field in json_data and isinstance(json_data[field], str):
                            return json_data[field]
                    
                    # If no readable fields, create summary
                    return "Analysis completed with structured data results"
            except:
                pass
        
        return text
    
    async def _call_api_for_analysis(self, prompt: str, max_tokens: int = None) -> str:
        """Make API call for LLM analysis"""
        try:
            params = self.api_params.copy()
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            # Use coordinator's API call method
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=params,
                preferred_gpu_id=self.gpu_id
            )
            
            # Extract text from API response
            if isinstance(result, dict):
                return result.get('text', result.get('response', ''))
            return str(result)
            
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            raise RuntimeError(f"API analysis failed: {str(e)}")
    
    # ✅ FIXED: Missing get_program_chunks method
    async def _get_program_chunks(self, program_name: str):
        """Get program chunks for a specific program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Proper parameter binding
            cursor.execute("""
                SELECT program_name, chunk_id, chunk_type, content, metadata, line_start, line_end
                FROM program_chunks
                WHERE program_name = ?
                ORDER BY line_start
            """, (str(program_name),))  # FIXED: Ensure string parameter
            
            chunks = cursor.fetchall()
            conn.close()
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get program chunks: {e}")
            return []
        
    async def analyze_field_lineage(self, field_name: str) -> Dict[str, Any]:
        """✅ FIXED: Field lineage analysis with comprehensive logging"""
        try:
            await self._load_existing_lineage()
            
            # FIXED: Ensure field_name is string and log start
            field_name = str(field_name)
            self.logger.info(f"🎯 Starting field lineage analysis for: {field_name}")
            
            # Find all references to this field
            field_references = await self._find_field_references(field_name)
            
            # FIXED: Better logging of what was found
            self.logger.info(f"📋 Found {len(field_references)} references for {field_name}")
            
            if not field_references:
                self.logger.warning(f"⚠️ No references found for field: {field_name}")
                return self._add_processing_info({
                    "field_name": field_name,
                    "error": "No references found for this field",
                    "suggestions": ["Check if field name is correct", "Verify data has been processed", "Try alternative field names"],
                    "total_references": 0,
                    "status": "no_data"
                })
            
            # Log progress through analysis steps
            self.logger.info(f"🔄 Step 1/5: Building lineage graph for {field_name}")
            field_lineage = await self._build_field_lineage_graph(field_name, field_references)
            
            self.logger.info(f"🔄 Step 2/5: Analyzing usage patterns for {field_name}")
            usage_analysis = await self._analyze_field_usage_patterns_api(field_name, field_references)
            
            self.logger.info(f"🔄 Step 3/5: Finding transformations for {field_name}")
            transformations = await self._find_field_transformations_api(field_name, field_references)
            
            self.logger.info(f"🔄 Step 4/5: Analyzing lifecycle for {field_name}")
            lifecycle = await self._analyze_field_lifecycle_api(field_name, field_references)
            
            self.logger.info(f"🔄 Step 5/5: Generating comprehensive report for {field_name}")
            lineage_report = await self._generate_field_lineage_report_api(
                field_name, field_lineage, usage_analysis, transformations, lifecycle
            )
            
            # FIXED: Build comprehensive result with all data
            result = {
                "field_name": field_name,
                "total_references": len(field_references),
                "lineage_graph": field_lineage,
                "usage_analysis": usage_analysis,
                "transformations": transformations,
                "lifecycle": lifecycle,
                "comprehensive_report": lineage_report,
                "impact_analysis": await self._analyze_field_impact_api(field_name, field_references),
                "analysis_timestamp": dt.now().isoformat(),
                "status": "success"
            }
            
            self.logger.info(f"✅ Field lineage analysis completed successfully for {field_name}")
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"❌ Field lineage analysis failed for {field_name}: {str(e)}")
            return self._add_processing_info({
                "field_name": field_name,
                "error": str(e),
                "status": "error",
                "analysis_timestamp": dt.now().isoformat()
            })

    
    async def analyze_complete_data_flow(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """🔄 FIXED: Complete data flow analysis with auto-detection"""
        try:
            component_name = str(component_name)
            component_type = str(component_type)
            
            self.logger.info(f"🔄 Starting complete data flow analysis for {component_name} ({component_type})")
            
            # FIXED: Auto-detect actual component type from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            actual_is_program = await self._is_component_a_program(component_name, cursor)
            conn.close()
            
            if actual_is_program:
                self.logger.info(f"📄 Auto-detected {component_name} as PROGRAM - using program analysis")
                analysis_result = await self._analyze_program_data_flow(component_name)
            else:
                self.logger.info(f"📁 Auto-detected {component_name} as FILE - using file analysis")
                analysis_result = await self._analyze_file_data_flow(component_name)
            
            # FIXED: Ensure proper result structure
            if not analysis_result or analysis_result.get('error'):
                self.logger.warning(f"⚠️ Analysis failed for {component_name}: {analysis_result.get('error', 'Unknown error')}")
                return self._add_processing_info({
                    "component_name": component_name,
                    "component_type": component_type,
                    "detected_type": "program" if actual_is_program else "file",
                    "error": analysis_result.get('error', 'Analysis failed'),
                    "status": "error"
                })
            
            # FIXED: Add metadata and return properly formatted result
            final_result = {
                "component_name": component_name,
                "component_type": component_type,
                "detected_type": "program" if actual_is_program else "file",
                "analysis_result": analysis_result,
                "analysis_type": "complete_data_flow",
                "analysis_timestamp": dt.now().isoformat(),
                "status": "success"
            }
            
            self.logger.info(f"✅ Complete data flow analysis completed for {component_name}")
            return self._add_processing_info(final_result)
                
        except Exception as e:
            self.logger.error(f"❌ Complete data flow analysis failed: {str(e)}")
            return self._add_processing_info({
                "component_name": component_name,
                "component_type": component_type,
                "error": str(e),
                "status": "error"
            })
        
    async def _analyze_file_data_flow(self, file_name: str) -> Dict[str, Any]:
        """🔄 FIXED: Analyze file data flow with proper file vs program detection"""
        try:
            file_name = str(file_name)
            
            self.logger.info(f"📁 Analyzing FILE data flow for: {file_name}")
            
            # Get file access relationships (programs that access this file)
            file_access_data = await self._get_file_access_data(file_name)
            self.logger.info(f"Found {file_access_data.get('total_access_points', 0)} file access points")
            
            # Get field definitions for this file
            field_definitions = await self._get_file_field_definitions(file_name)
            self.logger.info(f"Found {len(field_definitions)} field definitions")
            
            # Analyze field usage across programs
            field_usage_analysis = await self._analyze_field_usage_across_programs(file_name, field_definitions)
            self.logger.info(f"Analyzed field usage across programs")
            
            # Generate comprehensive file flow analysis
            file_flow_analysis = await self._generate_file_flow_analysis_api(
                file_name, file_access_data, field_definitions, field_usage_analysis
            )
            
            # FIXED: Build complete result object
            result = {
                "file_name": file_name,
                "component_type": "file",
                "file_access_data": file_access_data,
                "field_definitions": field_definitions,
                "field_usage_analysis": field_usage_analysis,
                "file_flow_analysis": file_flow_analysis,
                "analysis_type": "complete_file_data_flow",
                "analysis_timestamp": dt.now().isoformat(),
                "status": "success"
            }
            
            self.logger.info(f"✅ File data flow analysis completed for {file_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ File data flow analysis failed for {file_name}: {str(e)}")
            return {
                "file_name": file_name,
                "error": str(e),
                "status": "error"
            }

    async def _get_file_access_data(self, component_name: str) -> Dict[str, Any]:
        """FIXED: Get file access data - auto-detect if component is a file or program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            self.logger.info(f"🔍 Analyzing file access data for component: {component_name}")
            
            # STEP 1: Determine if this is a file or program by checking database
            is_program = await self._is_component_a_program(component_name, cursor)
            
            if is_program:
                # FIXED: For programs, get files accessed BY the program
                self.logger.info(f"📄 Treating {component_name} as PROGRAM - finding files it accesses")
                cursor.execute("""
                    SELECT program_name, file_name, physical_file, access_type, access_mode,
                        record_format, access_location, line_number
                    FROM file_access_relationships
                    WHERE program_name = ? OR program_name LIKE ?
                    ORDER BY line_number
                """, (str(component_name), f"%{component_name}%"))
            else:
                # For files, get programs that access the file
                self.logger.info(f"📁 Treating {component_name} as FILE - finding programs that access it")
                cursor.execute("""
                    SELECT program_name, file_name, physical_file, access_type, access_mode,
                        record_format, access_location, line_number
                    FROM file_access_relationships
                    WHERE file_name = ? OR physical_file = ? OR file_name LIKE ? OR physical_file LIKE ?
                    ORDER BY program_name, line_number
                """, (str(component_name), str(component_name), f"%{component_name}%", f"%{component_name}%"))
            
            access_records = cursor.fetchall()
            conn.close()
            
            self.logger.info(f"📊 Found {len(access_records)} file access records for {component_name}")
            
            if not access_records:
                self.logger.warning(f"⚠️ No file access relationships found for {component_name}")
                return {
                    "access_patterns": {"creators": [], "readers": [], "updaters": [], "deleters": []},
                    "programs_accessing": [],
                    "files_accessed": [],
                    "total_access_points": 0,
                    "file_operations": {"create_operations": 0, "read_operations": 0, "update_operations": 0, "delete_operations": 0},
                    "component_type": "program" if is_program else "file",
                    "warning": f"No access relationships found for {component_name}"
                }
            
            # STEP 2: Organize access patterns
            access_patterns = {
                "creators": [],
                "readers": [],
                "updaters": [],
                "deleters": []
            }
            
            programs_accessing = set()
            files_accessed = set()
            
            for record in access_records:
                access_info = {
                    "program_name": record[0],
                    "file_name": record[1],
                    "physical_file": record[2],
                    "access_type": record[3],
                    "access_mode": record[4],
                    "record_format": record[5],
                    "access_location": record[6],
                    "line_number": record[7]
                }
                
                programs_accessing.add(record[0])
                if record[1]:  # file_name
                    files_accessed.add(record[1])
                if record[2]:  # physical_file
                    files_accessed.add(record[2])
                
                # FIXED: More comprehensive categorization
                access_type = record[3].upper() if record[3] else ""
                access_mode = record[4].upper() if record[4] else ""
                
                # Categorize by access pattern
                if access_type in ["WRITE", "FD"] and access_mode in ["OUTPUT", "EXTEND"]:
                    access_patterns["creators"].append(access_info)
                elif access_type in ["READ", "SELECT"] and access_mode == "INPUT":
                    access_patterns["readers"].append(access_info)
                elif access_type in ["REWRITE", "WRITE"] and access_mode == "I-O":
                    access_patterns["updaters"].append(access_info)
                elif access_type == "DELETE":
                    access_patterns["deleters"].append(access_info)
                else:
                    # Default categorization based on access_mode
                    if access_mode == "INPUT":
                        access_patterns["readers"].append(access_info)
                    elif access_mode in ["OUTPUT", "EXTEND"]:
                        access_patterns["creators"].append(access_info)
                    elif access_mode == "I-O":
                        access_patterns["updaters"].append(access_info)
                    else:
                        access_patterns["readers"].append(access_info)  # Default
            
            result = {
                "access_patterns": access_patterns,
                "programs_accessing": list(programs_accessing),
                "files_accessed": list(files_accessed),
                "total_access_points": len(access_records),
                "component_type": "program" if is_program else "file",
                "file_operations": {
                    "create_operations": len(access_patterns["creators"]),
                    "read_operations": len(access_patterns["readers"]),
                    "update_operations": len(access_patterns["updaters"]),
                    "delete_operations": len(access_patterns["deleters"])
                }
            }
            
            # FIXED: Log detailed results
            if is_program:
                self.logger.info(f"📈 Program {component_name} accesses {len(files_accessed)} files: {list(files_accessed)[:5]}...")
            else:
                self.logger.info(f"📈 File {component_name} accessed by {len(programs_accessing)} programs: {list(programs_accessing)[:5]}...")
            
            self.logger.info(f"📊 Access breakdown: {result['file_operations']['create_operations']} creators, "
                            f"{result['file_operations']['read_operations']} readers, "
                            f"{result['file_operations']['update_operations']} updaters")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get file access data for {component_name}: {e}")
            return {
                "access_patterns": {"creators": [], "readers": [], "updaters": [], "deleters": []},
                "programs_accessing": [],
                "files_accessed": [],
                "total_access_points": 0,
                "file_operations": {"create_operations": 0, "read_operations": 0, "update_operations": 0, "delete_operations": 0},
                "error": str(e)
            }
    
    async def _is_component_a_program(self, component_name: str, cursor) -> bool:
        """FIXED: Determine if component is a program or file by checking program_chunks table"""
        try:
            # Check if this component exists as a program in program_chunks
            cursor.execute("""
                SELECT COUNT(*) FROM program_chunks 
                WHERE program_name = ? OR program_name LIKE ?
                LIMIT 1
            """, (str(component_name), f"%{component_name}%"))
            
            program_chunk_count = cursor.fetchone()[0]
            
            # If it exists in program_chunks, it's a program
            if program_chunk_count > 0:
                self.logger.info(f"✅ {component_name} identified as PROGRAM (found in program_chunks)")
                return True
            
            # Check if it appears as a program_name in file_access_relationships
            cursor.execute("""
                SELECT COUNT(*) FROM file_access_relationships 
                WHERE program_name = ? OR program_name LIKE ?
                LIMIT 1
            """, (str(component_name), f"%{component_name}%"))
            
            program_access_count = cursor.fetchone()[0]
            
            if program_access_count > 0:
                self.logger.info(f"✅ {component_name} identified as PROGRAM (found accessing files)")
                return True
            
            # Default: assume it's a file if not found as program
            self.logger.info(f"📁 {component_name} identified as FILE (not found as program)")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to determine component type for {component_name}: {e}")
            return False  # Default to file


    async def _get_file_field_definitions(self, file_name: str) -> List[Dict[str, Any]]:
        """FIXED: Get field definitions associated with a file"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Proper parameter binding
            cursor.execute("""
                SELECT field_name, qualified_name, source_type, source_name,
                    definition_location, data_type, picture_clause, usage_clause,
                    level_number, parent_field, occurs_info, business_domain
                FROM field_cross_reference
                WHERE source_name = ? OR source_name IN (
                    SELECT DISTINCT program_name FROM file_access_relationships 
                    WHERE file_name = ? OR physical_file = ?
                )
                AND definition_location IN ('FD', 'FILE_SECTION', 'WORKING_STORAGE')
                ORDER BY source_name, level_number, field_name
            """, (str(file_name), str(file_name), str(file_name)))  # FIXED: String parameters
            
            field_records = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "field_name": row[0],
                    "qualified_name": row[1],
                    "source_type": row[2],
                    "source_name": row[3],
                    "definition_location": row[4],
                    "data_type": row[5],
                    "picture_clause": row[6],
                    "usage_clause": row[7],
                    "level_number": row[8],
                    "parent_field": row[9],
                    "occurs_info": json.loads(row[10]) if row[10] else {},
                    "business_domain": row[11]
                } for row in field_records
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get file field definitions: {e}")
            return []

    async def _analyze_program_data_flow(self, program_name: str) -> Dict[str, Any]:
        """🔄 FIXED: Analyze program data flow with proper program analysis"""
        try:
            program_name = str(program_name)
            
            self.logger.info(f"📄 Analyzing PROGRAM data flow for: {program_name}")
            
            # Get file access relationships for this program (files accessed by this program)
            program_file_access = await self._get_program_file_access(program_name)
            total_files = sum(len(files) for files in program_file_access.values())
            self.logger.info(f"Found {total_files} file access relationships")
            
            # Get field cross-references for this program
            program_field_usage = await self._get_program_field_usage(program_name)
            total_fields = sum(len(fields) for fields in program_field_usage.values())
            self.logger.info(f"Found {total_fields} field usage relationships")
            
            # Analyze data transformations
            data_transformations = await self._analyze_data_transformations_api(program_name)
            self.logger.info(f"Found {data_transformations.get('total_transformations', 0)} data transformations")
            
            # Generate program data flow analysis
            program_data_flow_analysis = await self._generate_program_data_flow_analysis_api(
                program_name, program_file_access, program_field_usage, data_transformations
            )
            
            # FIXED: Build complete result object
            result = {
                "program_name": program_name,
                "component_type": "program",
                "program_file_access": program_file_access,
                "program_field_usage": program_field_usage,
                "data_transformations": data_transformations,
                "program_data_flow_analysis": program_data_flow_analysis,
                "analysis_type": "program_data_flow",
                "analysis_timestamp": dt.now().isoformat(),
                "status": "success"
            }
            
            self.logger.info(f"✅ Program data flow analysis completed for {program_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Program data flow analysis failed for {program_name}: {str(e)}")
            return {
                "program_name": program_name,
                "error": str(e),
                "status": "error"
            }
        

    async def _get_program_file_access(self, program_name: str) -> Dict[str, Any]:
        """FIXED: Get file access patterns for a specific program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Proper parameter binding
            cursor.execute("""
                SELECT file_name, physical_file, access_type, access_mode,
                    record_format, access_location, line_number
                FROM file_access_relationships
                WHERE program_name = ?
                ORDER BY line_number
            """, (str(program_name),))  # FIXED: String parameter
            
            access_records = cursor.fetchall()
            conn.close()
            
            # Categorize file access
            file_access = {
                "input_files": [],
                "output_files": [],
                "update_files": [],
                "temporary_files": []
            }
            
            for record in access_records:
                file_info = {
                    "file_name": record[0],
                    "physical_file": record[1],
                    "access_type": record[2],
                    "access_mode": record[3],
                    "record_format": record[4],
                    "access_location": record[5],
                    "line_number": record[6]
                }
                
                if record[3] == "INPUT":
                    file_access["input_files"].append(file_info)
                elif record[3] in ["OUTPUT", "EXTEND"]:
                    file_access["output_files"].append(file_info)
                elif record[3] == "I-O":
                    file_access["update_files"].append(file_info)
                else:
                    file_access["temporary_files"].append(file_info)
            
            return file_access
            
        except Exception as e:
            self.logger.error(f"Failed to get program file access: {e}")
            return {"input_files": [], "output_files": [], "update_files": [], "temporary_files": []}

    async def _get_program_field_usage(self, program_name: str) -> Dict[str, Any]:
        """FIXED: Get field usage patterns within a program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Proper parameter binding
            cursor.execute("""
                SELECT field_name, qualified_name, definition_location, data_type,
                    picture_clause, level_number, parent_field, business_domain
                FROM field_cross_reference
                WHERE source_name = ?
                ORDER BY level_number, field_name
            """, (str(program_name),))  # FIXED: String parameter
            
            field_records = cursor.fetchall()
            conn.close()
            
            # Categorize fields by definition location
            field_usage = {
                "working_storage_fields": [],
                "linkage_fields": [],
                "file_fields": [],
                "local_storage_fields": []
            }
            
            for record in field_records:
                field_info = {
                    "field_name": record[0],
                    "qualified_name": record[1],
                    "definition_location": record[2],
                    "data_type": record[3],
                    "picture_clause": record[4],
                    "level_number": record[5],
                    "parent_field": record[6],
                    "business_domain": record[7]
                }
                
                location = record[2].upper() if record[2] else ""
                if "WORKING" in location or "WS" in location:
                    field_usage["working_storage_fields"].append(field_info)
                elif "LINKAGE" in location:
                    field_usage["linkage_fields"].append(field_info)
                elif "FD" in location or "FILE" in location:
                    field_usage["file_fields"].append(field_info)
                elif "LOCAL" in location or "LS" in location:
                    field_usage["local_storage_fields"].append(field_info)
            
            return field_usage
            
        except Exception as e:
            self.logger.error(f"Failed to get program field usage: {e}")
            return {"working_storage_fields": [], "linkage_fields": [], "file_fields": [], "local_storage_fields": []}

    async def _analyze_data_transformations_api(self, program_name: str) -> Dict[str, Any]:
        """FIXED: Analyze data transformations within the program using API"""
        try:
            # Get program chunks with data transformation logic
            chunks = await self._get_program_chunks(program_name)
            
            # Extract transformation patterns
            transformations = []
            for chunk in chunks:
                content = chunk[3]  # FIXED: Content is at index 3
                
                # Find MOVE statements
                moves = re.findall(r'MOVE\s+([^.]+?)\s+TO\s+([^.]+)', content, re.IGNORECASE)
                transformations.extend([("MOVE", move[0].strip(), move[1].strip()) for move in moves])
                
                # Find COMPUTE statements
                computes = re.findall(r'COMPUTE\s+([^.]+)', content, re.IGNORECASE)
                transformations.extend([("COMPUTE", compute.strip(), "") for compute in computes])
                
                # Find other transformation patterns
                adds = re.findall(r'ADD\s+([^.]+?)\s+TO\s+([^.]+)', content, re.IGNORECASE)
                transformations.extend([("ADD", add[0].strip(), add[1].strip()) for add in adds])
            
            prompt = f"""
            Analyze data transformations in program {program_name}:
            
            Transformations Found:
            {transformations[:15]}  # Limit for prompt size
            
            Total Transformations: {len(transformations)}
            
            Analyze:
            1. Types of data transformations performed
            2. Business logic embedded in transformations
            3. Data validation and conversion patterns
            4. Complex calculations and derivations
            5. Data flow between different data structures
            
            Provide detailed analysis of the data transformation patterns.
            """
            
            try:
                api_result = await self._call_api_for_readable_analysis(prompt, max_tokens=400)
                
                return {
                    "total_transformations": len(transformations),
                    "transformation_types": {
                        "move_operations": len([t for t in transformations if t[0] == "MOVE"]),
                        "compute_operations": len([t for t in transformations if t[0] == "COMPUTE"]),
                        "arithmetic_operations": len([t for t in transformations if t[0] == "ADD"])
                    },
                    "transformation_analysis": api_result,
                    "sample_transformations": transformations[:10]
                }
            except Exception as e:
                self.logger.error(f"Transformation analysis failed: {e}")
                return {"error": str(e), "total_transformations": len(transformations)}
                
        except Exception as e:
           self.logger.error(f"Data transformation analysis failed: {e}")
           return {"error": str(e)}

    async def _generate_program_data_flow_analysis_api(self, program_name: str, 
                                                    program_file_access: Dict,
                                                    program_field_usage: Dict,
                                                    data_transformations: Dict) -> str:
        """FIXED: Generate comprehensive program data flow analysis using API"""
        
        flow_summary = {
            "program_name": program_name,
            "input_files": len(program_file_access.get("input_files", [])),
            "output_files": len(program_file_access.get("output_files", [])),
            "update_files": len(program_file_access.get("update_files", [])),
            "working_storage_fields": len(program_field_usage.get("working_storage_fields", [])),
            "linkage_fields": len(program_field_usage.get("linkage_fields", [])),
            "transformations": data_transformations.get("total_transformations", 0)
        }
        
        prompt = f"""
        Generate comprehensive program data flow analysis for: {program_name}
        
        Program Data Flow Summary:
        - Reads from {flow_summary['input_files']} input files
        - Writes to {flow_summary['output_files']} output files
        - Updates {flow_summary['update_files']} files
        - Uses {flow_summary['working_storage_fields']} working storage fields
        - Has {flow_summary['linkage_fields']} linkage parameters
        - Performs {flow_summary['transformations']} data transformations
        
        Input Files: {[f['file_name'] for f in program_file_access.get('input_files', [])]}
        Output Files: {[f['file_name'] for f in program_file_access.get('output_files', [])]}
        
        Provide detailed analysis covering:
        
        **Program Data Flow Overview:**
        - Business purpose and data processing role
        - Input-to-output data transformation pipeline
        - Key business rules implemented
        
        **Input Data Analysis:**
        - Source data characteristics and validation
        - Data quality checks and error handling
        - Input data dependencies and requirements
        
        **Processing Logic:**
        - Core data transformation algorithms
        - Business calculations and derivations
        - Data validation and business rule enforcement
        
        **Output Data Generation:**
        - Output file structures and formats
        - Data enrichment and enhancement processes
        - Quality assurance and validation
        
        Write as comprehensive program data flow documentation.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Program data flow analysis generation failed: {e}")
            return self._generate_fallback_program_data_flow_analysis(program_name, flow_summary)

    def _generate_fallback_program_data_flow_analysis(self, program_name: str, flow_summary: Dict) -> str:
        """Generate fallback program data flow analysis when API fails"""
        analysis = f"## Program Data Flow Analysis: {program_name}\n\n"
        
        analysis += "### Program Overview\n"
        analysis += f"Program {program_name} processes data through the following flow:\n"
        analysis += f"- Reads from {flow_summary['input_files']} input sources\n"
        analysis += f"- Performs {flow_summary['transformations']} data transformations\n"
        analysis += f"- Produces {flow_summary['output_files']} output files\n"
        analysis += f"- Updates {flow_summary['update_files']} existing files\n\n"
        
        analysis += "### Data Processing Characteristics\n"
        if flow_summary['transformations'] > 50:
            analysis += "**Complex Processing:** This program performs extensive data transformations and business logic.\n"
        elif flow_summary['transformations'] > 20:
            analysis += "**Moderate Processing:** This program performs standard data transformations.\n"
        else:
            analysis += "**Simple Processing:** This program performs basic data operations.\n"
        
        analysis += f"\n### Integration Points\n"
        total_files = flow_summary['input_files'] + flow_summary['output_files'] + flow_summary['update_files']
        analysis += f"This program integrates with {total_files} different data sources and targets.\n"
        
        return analysis

    async def _analyze_field_usage_across_programs(self, file_name: str, field_definitions: List[Dict]) -> Dict[str, Any]:
        """FIXED: Analyze how fields are used across different programs"""
        try:
            field_usage = {}
            
            for field_def in field_definitions:
                field_name = field_def["field_name"]
                
                # Find usage of this field across programs
                field_usage[field_name] = await self._get_field_usage_pattern(field_name, file_name)
            
            # Categorize fields
            field_categories = await self._categorize_fields_api(field_definitions, field_usage)
            
            return {
                "field_usage_details": field_usage,
                "field_categories": field_categories,
                "total_fields_analyzed": len(field_definitions)
            }
            
        except Exception as e:
            self.logger.error(f"Field usage analysis failed: {e}")
            return {"field_usage_details": {}, "field_categories": {}}

    async def _get_field_usage_pattern(self, field_name: str, file_name: str) -> Dict[str, Any]:
        """FIXED: Get detailed usage pattern for a specific field"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Proper parameter binding
            cursor.execute("""
                SELECT pc.program_name, pc.chunk_type, pc.content, pc.line_start, pc.line_end
                FROM program_chunks pc
                JOIN file_access_relationships far ON pc.program_name = far.program_name
                WHERE (pc.content LIKE ? OR pc.content LIKE ?) 
                AND (far.file_name = ? OR far.physical_file = ?)
                ORDER BY pc.program_name, pc.line_start
            """, (f"%{field_name}%", f"%{field_name.upper()}%", str(file_name), str(file_name)))
            
            usage_records = cursor.fetchall()
            conn.close()
            
            usage_patterns = {
                "programs_using": set(),
                "usage_contexts": [],
                "operations_detected": []
            }
            
            for record in usage_records:
                program_name, chunk_type, content, line_start, line_end = record
                usage_patterns["programs_using"].add(program_name)
                
                # Analyze field operations in content
                operations = self._detect_field_operations(field_name, content)
                usage_patterns["operations_detected"].extend(operations)
                
                usage_patterns["usage_contexts"].append({
                    "program_name": program_name,
                    "chunk_type": chunk_type,
                    "line_range": f"{line_start}-{line_end}",
                    "operations": operations
                })
            
            return {
                "programs_using": list(usage_patterns["programs_using"]),
                "usage_contexts": usage_patterns["usage_contexts"],
                "operations_detected": usage_patterns["operations_detected"],
                "usage_frequency": len(usage_patterns["usage_contexts"])
            }
            
        except Exception as e:
            self.logger.error(f"Field usage pattern analysis failed: {e}")
            return {"programs_using": [], "usage_contexts": [], "operations_detected": []}

    def _detect_field_operations(self, field_name: str, content: str) -> List[str]:
        """Detect specific operations performed on a field"""
        operations = []
        content_upper = content.upper()
        field_upper = field_name.upper()
        
        # Check for different operations
        if f"MOVE TO {field_upper}" in content_upper:
            operations.append("WRITE")
        elif f"MOVE {field_upper}" in content_upper:
            operations.append("READ")
        elif f"COMPUTE {field_upper}" in content_upper:
            operations.append("COMPUTE")
        elif f"ADD TO {field_upper}" in content_upper:
            operations.append("ADD")
        elif f"IF {field_upper}" in content_upper:
            operations.append("VALIDATE")
        elif f"DISPLAY {field_upper}" in content_upper:
            operations.append("DISPLAY")
        
        return operations

    async def _categorize_fields_api(self, field_definitions: List[Dict], field_usage: Dict) -> Dict[str, Any]:
        """FIXED: Categorize fields based on usage patterns using API"""
        
        # Prepare field summary for analysis
        field_summary = []
        for field_def in field_definitions[:20]:  # Limit for prompt size
            field_name = field_def["field_name"]
            usage = field_usage.get(field_name, {})
            
            field_summary.append({
                "field_name": field_name,
                "data_type": field_def.get("data_type"),
                "picture_clause": field_def.get("picture_clause"),
                "business_domain": field_def.get("business_domain"),
                "programs_using": len(usage.get("programs_using", [])),
                "operations": usage.get("operations_detected", [])
            })
        
        prompt = f"""
        Categorize these file fields based on their usage patterns:
        
        Field Analysis Data:
        {json.dumps(field_summary, indent=2)}
        
        Categorize fields into:
        1. **Input Fields** - Fields that come from external sources
        2. **Derived Fields** - Fields calculated from other fields
        3. **Updated Fields** - Fields modified by programs
        4. **Static Fields** - Fields that rarely change
        5. **Unused Fields** - Fields with minimal or no usage
        6. **Key Fields** - Fields used for identification or indexing
        7. **Business Fields** - Fields with business logic
        
        Return as JSON:
        {{
            "input_fields": ["field1", "field2"],
            "derived_fields": ["field3", "field4"],
            "updated_fields": ["field5", "field6"],
            "static_fields": ["field7", "field8"],
            "unused_fields": ["field9", "field10"],
            "key_fields": ["field11", "field12"],
            "business_fields": ["field13", "field14"]
        }}
        """
        
        try:
            response_text = await self._call_api_for_analysis(prompt, max_tokens=300)
            
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Field categorization failed: {e}")
        
        # Fallback categorization
        return self._fallback_field_categorization(field_definitions, field_usage)

    def _fallback_field_categorization(self, field_definitions: List[Dict], field_usage: Dict) -> Dict[str, Any]:
        """Fallback field categorization when API fails"""
        categories = {
            "input_fields": [],
            "derived_fields": [],
            "updated_fields": [],
            "static_fields": [],
            "unused_fields": [],
            "key_fields": [],
            "business_fields": []
        }
        
        for field_def in field_definitions:
            field_name = field_def["field_name"]
            usage = field_usage.get(field_name, {})
            operations = usage.get("operations_detected", [])
            programs_count = len(usage.get("programs_using", []))
            
            # Simple categorization logic
            if programs_count == 0:
                categories["unused_fields"].append(field_name)
            elif "COMPUTE" in operations or "ADD" in operations:
                categories["derived_fields"].append(field_name)
            elif "WRITE" in operations and "READ" in operations:
                categories["updated_fields"].append(field_name)
            elif field_name.endswith(("-ID", "-KEY", "-NUM")):
                categories["key_fields"].append(field_name)
            elif programs_count == 1 and "READ" not in operations:
                categories["static_fields"].append(field_name)
            else:
                categories["business_fields"].append(field_name)
        
        return categories

    async def _generate_file_flow_analysis_api(self, file_name: str, file_access_data: Dict,
                                            field_definitions: List[Dict], field_usage_analysis: Dict) -> str:
        """FIXED: Generate comprehensive file flow analysis using API"""
        
        flow_summary = {
            "file_name": file_name,
            "programs_accessing": len(file_access_data.get("programs_accessing", [])),
            "total_fields": len(field_definitions),
            "create_programs": len(file_access_data.get("access_patterns", {}).get("creators", [])),
            "read_programs": len(file_access_data.get("access_patterns", {}).get("readers", [])),
            "update_programs": len(file_access_data.get("access_patterns", {}).get("updaters", []))
        }
        
        prompt = f"""
        Generate comprehensive file data flow analysis for: {file_name}
        
        File Flow Summary:
        - Accessed by {flow_summary['programs_accessing']} programs
        - Contains {flow_summary['total_fields']} fields
        - Created by {flow_summary['create_programs']} programs
        - Read by {flow_summary['read_programs']} programs
        - Updated by {flow_summary['update_programs']} programs
        
        Programs Creating File: {[p['program_name'] for p in file_access_data.get('access_patterns', {}).get('creators', [])]}
        Programs Reading File: {[p['program_name'] for p in file_access_data.get('access_patterns', {}).get('readers', [])]}
        
        Provide detailed analysis covering:
        
        **File Overview:**
        - Business purpose and role in the system
        - Data characteristics and format
        - Critical importance to operations
        
        **Data Flow Lifecycle:**
        - File creation process and sources
        - Data population and initial load
        - Regular update patterns and schedules
        - Data consumption and usage patterns
        
        **Field-Level Analysis:**
        - Key fields and their business meaning
        - Derived vs source data fields
        - Data quality and validation points
        - Field usage frequency and patterns
        
        **Program Integration:**
        - Producer programs and their roles
        - Consumer programs and their purposes
        - Data transformation points
        - Error handling and recovery processes
        
        Write as comprehensive data flow documentation.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"File flow analysis generation failed: {e}")
            return self._generate_fallback_file_flow_analysis(file_name, flow_summary)

    def _generate_fallback_file_flow_analysis(self, file_name: str, flow_summary: Dict) -> str:
        """Generate fallback file flow analysis when API fails"""
        analysis = f"## File Data Flow Analysis: {file_name}\n\n"
        
        analysis += "### File Overview\n"
        analysis += f"File {file_name} is a critical data component that:\n"
        analysis += f"- Is accessed by {flow_summary['programs_accessing']} different programs\n"
        analysis += f"- Contains {flow_summary['total_fields']} data fields\n"
        analysis += f"- Has {flow_summary['create_programs']} producer programs\n"
        analysis += f"- Has {flow_summary['read_programs']} consumer programs\n\n"
        
        analysis += "### Data Flow Characteristics\n"
        if flow_summary['create_programs'] > 0 and flow_summary['read_programs'] > 0:
            analysis += "**Full Lifecycle File:** This file has both producer and consumer programs.\n"
        elif flow_summary['read_programs'] > 0:
            analysis += "**Source File:** This file is primarily read by programs.\n"
        elif flow_summary['create_programs'] > 0:
            analysis += "**Output File:** This file is primarily created by programs.\n"
        
        return analysis

    async def _find_field_references(self, field_name: str) -> List[Dict[str, Any]]:
        """FIXED: Find all references to a field across the codebase"""
        references = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # FIXED: Search in program chunks with proper parameter binding
            patterns = [
                f"%{field_name}%",
                f"%{field_name.upper()}%", 
                f"%{field_name.lower()}%"
            ]
            
            for pattern in patterns:
                cursor.execute("""
                    SELECT program_name, chunk_id, chunk_type, content, metadata
                    FROM program_chunks
                    WHERE content LIKE ? OR metadata LIKE ?
                    LIMIT 100
                """, (pattern, pattern))
                
                results = cursor.fetchall()
                self.logger.info(f"Found {len(results)} chunks for pattern: {pattern}")
                
                for program_name, chunk_id, chunk_type, content, metadata_str in results:
                    # Avoid duplicates
                    if any(ref.get('chunk_id') == chunk_id for ref in references):
                        continue
                        
                    metadata = self.safe_json_loads(metadata_str)
                    
                    # Check if field is actually referenced
                    if self._is_field_referenced(field_name, content, metadata):
                        references.append({
                            "program_name": program_name,
                            "chunk_id": chunk_id,
                            "chunk_type": chunk_type,
                            "content": content[:500] + "..." if len(content) > 500 else content,
                            "metadata": metadata
                        })
                        
                        # Limit to prevent memory issues
                        if len(references) >= 50:
                            break
                
                if len(references) >= 50:
                    break
            
            # FIXED: Check for table definitions with proper error handling
            try:
                cursor.execute("""
                    SELECT table_name, fields
                    FROM file_metadata
                    WHERE fields LIKE ?
                    LIMIT 10
                """, (f"%{field_name}%",))
                
                for table_name, fields_str in cursor.fetchall():
                    references.append({
                        "type": "table_definition",
                        "table_name": table_name,
                        "field_type": "VARCHAR",
                        "description": f"Field found in {table_name}",
                        "business_meaning": "Business field"
                    })
            except sqlite3.OperationalError as e:
                self.logger.warning(f"Error querying file_metadata: {e}")
        
        except Exception as e:
            self.logger.error(f"Error finding field references: {str(e)}")
        
        finally:
            conn.close()
        
        self.logger.info(f"Found total {len(references)} references for {field_name}")
        return references

    def safe_json_loads(self, json_str):
        """Safely load JSON string with fallback"""
        if not json_str:
            return {}
        try:
            if isinstance(json_str, dict):
                return json_str
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _is_field_referenced(self, field_name: str, content: str, metadata: Dict) -> bool:
        """Check if field is actually referenced in content"""
        # Check in content with word boundaries (case insensitive)
        patterns = [
            r'\b' + re.escape(field_name) + r'\b',
            r'\b' + re.escape(field_name.upper()) + r'\b',
            r'\b' + re.escape(field_name.lower()) + r'\b'
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        # Check in metadata field lists
        for key in ['field_names', 'fields', 'all_fields']:
            if key in metadata:
                fields = metadata[key]
                if isinstance(fields, list):
                    if any(field_name.lower() == f.lower() for f in fields):
                        return True
                elif isinstance(fields, str):
                    if field_name.lower() in fields.lower():
                        return True
        
        return False

    async def _build_field_lineage_graph(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """Build lineage graph for a specific field"""
        lineage_nodes = []
        lineage_edges = []
        
        # Create field node
        field_node = {
            "id": f"field_{field_name}",
            "type": "field",
            "name": field_name,
            "properties": {"primary_field": True}
        }
        lineage_nodes.append(field_node)
        
        # Process each reference
        for ref in references:
            if ref.get("type") == "table_definition":
                # Table definition node
                table_node = {
                    "id": f"table_{ref['table_name']}",
                    "type": "table",
                    "name": ref["table_name"],
                    "properties": {
                        "field_type": ref["field_type"],
                        "description": ref.get("description", "")
                    }
                }
                lineage_nodes.append(table_node)
                
                # Edge from table to field
                lineage_edges.append({
                    "source": table_node["id"],
                    "target": field_node["id"],
                    "relationship": "defines",
                    "properties": {"field_type": ref["field_type"]}
                })
                
            else:
                # Program/chunk reference
                program_node = {
                    "id": f"program_{ref['program_name']}",
                    "type": "program",
                    "name": ref["program_name"],
                    "properties": {}
                }
                lineage_nodes.append(program_node)
                
                chunk_node = {
                    "id": f"chunk_{ref['chunk_id']}",
                    "type": ref["chunk_type"],
                    "name": ref["chunk_id"],
                    "properties": {"parent_program": ref["program_name"]}
                }
                lineage_nodes.append(chunk_node)
                
                # Program contains chunk
                lineage_edges.append({
                    "source": program_node["id"],
                    "target": chunk_node["id"],
                    "relationship": "contains",
                    "properties": {}
                })
        
        return {
            "nodes": lineage_nodes,
            "edges": lineage_edges,
            "field_name": field_name,
            "total_nodes": len(lineage_nodes),
            "total_edges": len(lineage_edges)
        }

    async def _analyze_field_usage_patterns_api(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """✅ FIXED: Analyze usage patterns for a field with API calls"""
        usage_stats = {
            "total_references": len(references),
            "programs_using": set(),
            "operation_types": defaultdict(int),
            "chunk_types": defaultdict(int),
            "table_definitions": []
        }
        
        # Analyze each reference with API
        for ref in references:
            if ref.get("type") == "table_definition":
                usage_stats["table_definitions"].append({
                    "table": ref["table_name"],
                    "type": ref["field_type"],
                    "description": ref.get("description", "")
                })
            else:
                usage_stats["programs_using"].add(ref["program_name"])
                usage_stats["chunk_types"][ref["chunk_type"]] += 1
                
                # Analyze reference with API
                ref_details = await self._analyze_field_reference_with_api(
                    field_name, ref["content"], ref["chunk_type"], ref["program_name"]
                )
                
                op_type = ref_details.get("operation_type", "REFERENCE")
                usage_stats["operation_types"][op_type] += 1
        
        # Convert sets to lists for JSON serialization
        usage_stats["programs_using"] = list(usage_stats["programs_using"])
        usage_stats["operation_types"] = dict(usage_stats["operation_types"])
        usage_stats["chunk_types"] = dict(usage_stats["chunk_types"])
        
        # Analyze patterns with API
        pattern_analysis = await self._analyze_usage_patterns_with_api(field_name, usage_stats)
        
        return {
            "statistics": usage_stats,
            "pattern_analysis": pattern_analysis,
            "complexity_score": self._calculate_usage_complexity(usage_stats)
        }

    async def _analyze_field_reference_with_api(self, field_name: str, content: str, 
                                       chunk_type: str, program_name: str) -> Dict[str, Any]:
        """✅ FIXED: Analyze field reference with better content preservation"""
        
        # FIXED: Increase truncation size to preserve context
        content_preview = content[:1200] if len(content) > 1200 else content  # INCREASED from 600
        
        prompt = f"""
        Analyze how the field "{field_name}" is used in this {chunk_type} from program {program_name}:
        
        {content_preview}
        
        Determine:
        1. Operation type (READ, WRITE, UPDATE, DELETE, TRANSFORM, VALIDATE)
        2. Context of usage (input, output, calculation, validation, etc.)
        3. Any transformations applied to the field
        4. Business logic involving this field
        5. Data flow direction (source or target)
        
        Return as JSON:
        {{
            "operation_type": "READ",
            "usage_context": "Field is read for validation",
            "transformations": ["uppercase conversion"],
            "business_logic": "Field validation logic",
            "data_flow": "source",
            "confidence": 0.9
        }}
        """
        
        try:
            response_text = await self._call_api_for_analysis(prompt, max_tokens=300)  # INCREASED tokens
            
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                parsed_response = json.loads(response_text[json_start:json_end])
                
                self.logger.debug(f"✅ Successfully analyzed field reference for {field_name}")
                return parsed_response
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse API field reference analysis: {str(e)}")
        
        # Fallback analysis with logging
        fallback_result = {
            "operation_type": self._infer_operation_type(field_name, content),
            "usage_context": "Field usage detected",
            "transformations": [],
            "business_logic": "Analysis not available via API",
            "data_flow": "unknown",
            "confidence": 0.5
        }
        
        self.logger.info(f"🔄 Using fallback analysis for {field_name}")
        return fallback_result

    def _infer_operation_type(self, field_name: str, content: str) -> str:
        """Infer operation type using regex patterns"""
        content_upper = content.upper()
        field_upper = field_name.upper()
        
        # COBOL patterns
        if f"MOVE TO {field_upper}" in content_upper or f"MOVE {field_upper}" in content_upper:
            return "WRITE"
        elif f"READ" in content_upper and field_upper in content_upper:
            return "READ"
        elif f"REWRITE" in content_upper and field_upper in content_upper:
            return "UPDATE"
        elif f"DELETE" in content_upper and field_upper in content_upper:
            return "DELETE"
        elif f"COMPUTE {field_upper}" in content_upper or f"ADD TO {field_upper}" in content_upper:
            return "TRANSFORM"
        
        # SQL patterns
        elif f"SELECT" in content_upper and field_upper in content_upper:
            return "READ"
        elif f"INSERT" in content_upper and field_upper in content_upper:
            return "WRITE"
        elif f"UPDATE" in content_upper and field_upper in content_upper:
            return "UPDATE"
        
        return "REFERENCE"

    async def _analyze_usage_patterns_with_api(self, field_name: str, usage_stats: Dict) -> str:
        """✅ FIXED: Analyze usage patterns using API"""
        
        stats_summary = {
            "total_references": usage_stats["total_references"],
            "programs_count": len(usage_stats["programs_using"]),
            "operation_types": usage_stats["operation_types"],
            "chunk_types": usage_stats["chunk_types"]
        }
        
        prompt = f"""
        Analyze the usage patterns for field "{field_name}":
        
        Usage Statistics:
        {json.dumps(stats_summary, indent=2)}
        
        Provide insights on:
        1. Primary usage patterns
        2. Data flow characteristics
        3. Potential issues or risks
        4. Optimization opportunities
        5. Business importance indicators
        
        Provide a concise analysis in 200 words or less.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=300)
        except Exception as e:
            self.logger.warning(f"Failed to generate pattern analysis: {str(e)}")
            return f"Field {field_name} is used across {len(usage_stats['programs_using'])} programs with {usage_stats['total_references']} total references."

    async def _call_api_for_readable_analysis(self, prompt: str, max_tokens: int = None) -> str:
        """FIXED: API call with better empty response handling"""
        try:
            params = self.api_params.copy()
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            self.logger.debug(f"🔄 Making API call with {len(prompt)} character prompt")
            
            # Use coordinator's API call method
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=params,
                preferred_gpu_id=self.gpu_id
            )
            
            # FIXED: Better response extraction and validation
            if isinstance(result, dict):
                text = result.get('text') or result.get('response') or result.get('content') or ''
                
                # FIXED: Check for empty or very short responses
                if not text or len(text.strip()) < 10:
                    self.logger.warning(f"⚠️ API returned empty or very short response: '{text}'")
                    return "Analysis completed but detailed results not available from API"
                
                # Clean up the text to ensure it's readable
                cleaned_text = self._ensure_readable_text(text)
                
                self.logger.debug(f"✅ API call successful, returned {len(cleaned_text)} characters")
                return cleaned_text
            else:
                text_result = str(result)
                if len(text_result.strip()) < 10:
                    self.logger.warning(f"⚠️ API returned short string response: '{text_result}'")
                    return "Analysis completed with limited API response"
                return text_result
            
        except Exception as e:
            self.logger.error(f"❌ API call for readable analysis failed: {str(e)}")
            return f"Analysis completed but API call failed: {str(e)}"


    def _calculate_usage_complexity(self, usage_stats: Dict) -> float:
        """Calculate complexity score based on usage patterns"""
        # Factors: number of programs, operation types, reference count
        num_programs = len(usage_stats["programs_using"])
        num_operations = len(usage_stats["operation_types"])
        total_refs = usage_stats["total_references"]
        
        # Normalize scores
        program_score = min(num_programs / 10.0, 1.0)  # Max at 10 programs
        operation_score = min(num_operations / 5.0, 1.0)  # Max at 5 operation types
        reference_score = min(total_refs / 50.0, 1.0)  # Max at 50 references
        
        return (program_score + operation_score + reference_score) / 3.0

    async def _find_field_transformations_api(self, field_name: str, references: List[Dict]) -> List[Dict[str, Any]]:
        """✅ FIXED: Find data transformations involving the field"""
        transformations = []
        
        for ref in references:
            if ref.get("type") != "table_definition":
                # Analyze reference with API to get transformation details
                ref_details = await self._analyze_field_reference_with_api(
                    field_name, ref["content"], ref["chunk_type"], ref["program_name"]
                )
                
                if ref_details.get("transformations"):
                    transformation = {
                        "program": ref["program_name"],
                        "chunk": ref["chunk_id"],
                        "transformations": ref_details["transformations"],
                        "business_logic": ref_details.get("business_logic", ""),
                        "context": ref_details.get("usage_context", "")
                    }
                    transformations.append(transformation)
                
                # Also check content for mathematical operations
                content = ref.get("content", "")
                if any(op in content.upper() for op in ["COMPUTE", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"]):
                    math_transforms = await self._extract_mathematical_transformations_api(
                        field_name, content, ref["program_name"]
                    )
                    transformations.extend(math_transforms)
        
        return transformations

    async def _extract_mathematical_transformations_api(self, field_name: str, content: str, program_name: str) -> List[Dict]:
        """✅ FIXED: Extract mathematical transformations from content"""
        
        content_preview = content[:400] if len(content) > 400 else content
        
        prompt = f"""
        Extract mathematical transformations involving field "{field_name}" from this code:
        
        {content_preview}
        
        Find:
        1. Arithmetic operations (ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPUTE)
        2. Data conversion operations
        3. Aggregation operations
        4. Business calculations
        
        Return as JSON array:
        [{{
            "operation": "COMPUTE TOTAL = {field_name} * RATE",
            "type": "multiplication",
            "description": "Calculate total amount",
            "input_fields": ["{field_name}", "RATE"],
            "output_fields": ["TOTAL"]
        }}]
        """
        
        try:
            response_text = await self._call_api_for_analysis(prompt, max_tokens=300)
            
            if '[' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                transforms = json.loads(response_text[json_start:json_end])
                
                # Add program context
                for transform in transforms:
                    transform["program"] = program_name
                
                return transforms
        except Exception as e:
            self.logger.warning(f"Failed to parse mathematical transformations: {str(e)}")
        
        return []

    async def _analyze_field_lifecycle_api(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """✅ FIXED: Analyze the lifecycle of a field"""
        lifecycle_stages = {
            "creation": [],
            "updates": [],
            "reads": [],
            "transformations": [],
            "deletions": [],
            "archival": []
        }
        
        for ref in references:
            if ref.get("type") == "table_definition":
                lifecycle_stages["creation"].append({
                    "type": "table_definition",
                    "table": ref["table_name"],
                    "field_type": ref["field_type"]
                })
            else:
                # Analyze reference with API
                ref_details = await self._analyze_field_reference_with_api(
                    field_name, ref["content"], ref["chunk_type"], ref["program_name"]
                )
                
                op_type = ref_details.get("operation_type", "REFERENCE")
                
                stage_mapping = {
                    "WRITE": "creation",
                    "UPDATE": "updates", 
                    "READ": "reads",
                    "TRANSFORM": "transformations",
                    "DELETE": "deletions"
                }
                
                stage = stage_mapping.get(op_type, "reads")
                lifecycle_stages[stage].append({
                    "program": ref["program_name"],
                    "chunk": ref["chunk_id"],
                    "operation": op_type,
                    "context": ref_details.get("usage_context", "")
                })
        
        # Analyze lifecycle completeness with API
        lifecycle_analysis = await self._analyze_lifecycle_completeness_api(field_name, lifecycle_stages)
        
        return {
            "stages": lifecycle_stages,
            "analysis": lifecycle_analysis,
            "lifecycle_score": self._calculate_lifecycle_score(lifecycle_stages)
        }

    async def _analyze_lifecycle_completeness_api(self, field_name: str, stages: Dict) -> str:
        """✅ FIXED: Analyze lifecycle completeness using API"""
        
        # Summarize stages for prompt
        stage_summary = {
            stage: len(operations) for stage, operations in stages.items()
        }
        
        prompt = f"""
        Analyze the lifecycle completeness for field "{field_name}":
        
        Lifecycle Stage Counts:
        {json.dumps(stage_summary, indent=2)}
        
        Assess:
        1. Completeness of lifecycle coverage
        2. Missing lifecycle stages
        3. Data governance implications
        4. Potential data quality issues
        5. Recommendations for improvement
        
        Provide a comprehensive but concise lifecycle analysis in 250 words or less.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=400)
        except Exception as e:
            self.logger.warning(f"Failed to generate lifecycle analysis: {str(e)}")
            return f"Field {field_name} lifecycle analysis: {sum(stage_summary.values())} total operations across {len([s for s in stage_summary.values() if s > 0])} lifecycle stages."

    def _calculate_lifecycle_score(self, stages: Dict) -> float:
        """Calculate lifecycle completeness score"""
        stage_weights = {
            "creation": 0.3,
            "reads": 0.2,
            "updates": 0.2,
            "transformations": 0.1,
            "deletions": 0.1,
            "archival": 0.1
        }
        
        score = 0.0
        for stage, weight in stage_weights.items():
            if stages.get(stage):
                score += weight
        
        return score

    async def _analyze_field_impact_api(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """✅ FIXED: Analyze potential impact of changes to this field"""
        impact_analysis = {
            "affected_programs": set(),
            "affected_tables": set(),
            "critical_operations": [],
            "downstream_dependencies": [],
            "risk_level": "LOW"
        }
        
        for ref in references:
            if ref.get("type") == "table_definition":
                impact_analysis["affected_tables"].add(ref["table_name"])
            else:
                impact_analysis["affected_programs"].add(ref["program_name"])
                
                # Analyze reference with API
                ref_details = await self._analyze_field_reference_with_api(
                    field_name, ref["content"], ref["chunk_type"], ref["program_name"]
                )
                
                if ref_details.get("operation_type") in ["WRITE", "UPDATE", "TRANSFORM"]:
                    impact_analysis["critical_operations"].append({
                        "program": ref["program_name"],
                        "operation": ref_details.get("operation_type"),
                        "context": ref_details.get("usage_context", "")
                    })
        
        # Convert sets to lists
        impact_analysis["affected_programs"] = list(impact_analysis["affected_programs"])
        impact_analysis["affected_tables"] = list(impact_analysis["affected_tables"])
        
        # Calculate risk level
        num_programs = len(impact_analysis["affected_programs"])
        num_critical_ops = len(impact_analysis["critical_operations"])
        
        if num_programs > 10 or num_critical_ops > 5:
            impact_analysis["risk_level"] = "HIGH"
        elif num_programs > 5 or num_critical_ops > 2:
            impact_analysis["risk_level"] = "MEDIUM"
        
        # Generate detailed impact assessment with API
        impact_assessment = await self._generate_impact_assessment_api(field_name, impact_analysis)
        impact_analysis["detailed_assessment"] = impact_assessment
        
        return impact_analysis

    async def _generate_impact_assessment_api(self, field_name: str, impact_data: Dict) -> str:
        """✅ FIXED: Generate detailed impact assessment using API"""
        
        # Summarize impact data for prompt
        impact_summary = {
            "affected_programs": len(impact_data["affected_programs"]),
            "affected_tables": len(impact_data["affected_tables"]),
            "critical_operations": len(impact_data["critical_operations"]),
            "risk_level": impact_data["risk_level"]
        }
        
        prompt = f"""
        Generate a detailed impact assessment for potential changes to field "{field_name}":
        
        Impact Summary:
        {json.dumps(impact_summary, indent=2)}
        
        Provide:
        1. Risk assessment and mitigation strategies
        2. Testing requirements
        3. Change management recommendations
        4. Business impact analysis
        5. Technical considerations
        
        Format as a concise but comprehensive impact report (300 words max).
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=400)
        except Exception as e:
            self.logger.warning(f"Failed to generate impact assessment: {str(e)}")
            return f"Impact assessment for {field_name}: {impact_summary['risk_level']} risk level with {impact_summary['affected_programs']} affected programs."

    async def _generate_field_lineage_report_api(self, field_name: str, lineage_graph: Dict,
                                       usage_analysis: Dict, transformations: List,
                                       lifecycle: Dict) -> str:
        """✅ FIXED: Generate comprehensive field lineage report using API"""
        
        report_data = {
            "field_name": field_name,
            "total_nodes": lineage_graph.get("total_nodes", 0),
            "total_edges": lineage_graph.get("total_edges", 0),
            "total_references": usage_analysis['statistics']['total_references'],
            "programs_count": len(usage_analysis['statistics']['programs_using']),
            "transformations_count": len(transformations),
            "lifecycle_score": lifecycle['lifecycle_score'],
            "complexity_score": usage_analysis['complexity_score']
        }
        
        prompt = f"""
        Generate a comprehensive data lineage report for field "{field_name}":
        
        Report Data:
        {json.dumps(report_data, indent=2)}
        
        Key Statistics:
        - Operation Types: {usage_analysis['statistics']['operation_types']}
        - Table Definitions: {len(usage_analysis['statistics']['table_definitions'])}
        
        Generate a professional report including:
        1. Executive Summary
        2. Field Usage Overview
        3. Data Flow Analysis
        4. Transformation Summary
        5. Lifecycle Assessment
        6. Risk Analysis
        7. Recommendations
        
        Format as a structured report suitable for technical and business audiences (500 words max).
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.warning(f"Failed to generate lineage report: {str(e)}")
            return f"Lineage Report for {field_name}: Field found in {report_data['programs_count']} programs with {report_data['total_references']} references."

    # ==================== Additional Methods for Field Analysis ====================

    async def _analyze_field_data_flow(self, field_name: str) -> Dict[str, Any]:
        """FIXED: Analyze data flow for a specific field"""
        try:
            # FIXED: Ensure field_name is string
            field_name = str(field_name)
            
            # Get field references
            field_references = await self._find_field_references(field_name)
            
            if not field_references:
                return {
                    "field_name": field_name,
                    "error": "No field references found",
                    "data_flow_analysis": "Field not found in processed data"
                }
            
            # Analyze usage patterns
            usage_patterns = await self._analyze_field_usage_patterns_api(field_name, field_references)
            
            # Generate data flow analysis
            data_flow_analysis = await self._generate_field_data_flow_analysis_api(
                field_name, field_references, usage_patterns
            )
            
            result = {
                "field_name": field_name,
                "component_type": "field",
                "field_references": field_references,
                "usage_patterns": usage_patterns,
                "data_flow_analysis": data_flow_analysis,
                "analysis_type": "field_data_flow",
                "analysis_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Field data flow analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    async def _generate_field_data_flow_analysis_api(self, field_name: str, 
                                                   field_references: List[Dict],
                                                   usage_patterns: Dict) -> str:
        """FIXED: Generate field data flow analysis using API"""
        
        flow_summary = {
            "field_name": field_name,
            "total_references": len(field_references),
            "programs_using": len(usage_patterns['statistics']['programs_using']),
            "operation_types": usage_patterns['statistics']['operation_types'],
            "complexity_score": usage_patterns['complexity_score']
        }
        
        prompt = f"""
        Generate field data flow analysis for: {field_name}
        
        Field Flow Summary:
        - Referenced in {flow_summary['total_references']} locations
        - Used by {flow_summary['programs_using']} programs
        - Operation types: {flow_summary['operation_types']}
        - Complexity score: {flow_summary['complexity_score']:.2f}
        
        Provide detailed analysis covering:
        
        **Field Overview:**
        - Business purpose and meaning
        - Data characteristics and format
        - Critical importance to operations
        
        **Usage Patterns:**
        - Primary usage scenarios
        - Data transformation patterns
        - Validation and quality checks
        
        **Data Flow Characteristics:**
        - Source and target systems
        - Data movement patterns
        - Dependencies and relationships
        
        **Quality and Governance:**
        - Data quality indicators
        - Compliance considerations
        - Maintenance requirements
        
        Write as comprehensive field analysis documentation.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Field data flow analysis generation failed: {e}")
            return f"Field data flow analysis for {field_name}: {flow_summary['total_references']} references across {flow_summary['programs_using']} programs."

    # ==================== Enhanced Lineage Analysis Methods ====================

    async def analyze_cross_program_data_lineage(self, component_name: str) -> Dict[str, Any]:
        """🔄 NEW: Analyze data lineage across multiple programs"""
        try:
            # Get impact analysis data
            impact_data = await self._get_cross_program_impact_data(component_name)
            
            # Build lineage graph across programs
            lineage_graph = await self._build_cross_program_lineage_graph(component_name, impact_data)
            
            # Generate cross-program lineage analysis
            cross_program_analysis = await self._generate_cross_program_lineage_analysis_api(
                component_name, impact_data, lineage_graph
            )
            
            result = {
                "component_name": component_name,
                "impact_data": impact_data,
                "lineage_graph": lineage_graph,
                "cross_program_analysis": cross_program_analysis,
                "analysis_type": "cross_program_data_lineage",
                "analysis_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Cross-program lineage analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    async def _get_cross_program_impact_data(self, component_name: str) -> Dict[str, Any]:
        """Get impact analysis data across programs"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all impact relationships for this component
            cursor.execute("""
                SELECT source_artifact, source_type, dependent_artifact, dependent_type,
                    relationship_type, impact_level, change_propagation
                FROM impact_analysis
                WHERE source_artifact = ? OR dependent_artifact = ?
                ORDER BY impact_level DESC, relationship_type
            """, (str(component_name), str(component_name)))
            
            impact_records = cursor.fetchall()
            conn.close()
            
            # Organize impact data
            upstream_dependencies = []
            downstream_impacts = []
            
            for record in impact_records:
                impact_info = {
                    "source_artifact": record[0],
                    "source_type": record[1],
                    "dependent_artifact": record[2],
                    "dependent_type": record[3],
                    "relationship_type": record[4],
                    "impact_level": record[5],
                    "change_propagation": record[6]
                }
                
                if record[2] == component_name:  # This component is the dependent
                    upstream_dependencies.append(impact_info)
                else:  # This component is the source
                    downstream_impacts.append(impact_info)
            
            return {
                "upstream_dependencies": upstream_dependencies,
                "downstream_impacts": downstream_impacts,
                "total_relationships": len(impact_records)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cross-program impact data: {e}")
            return {"upstream_dependencies": [], "downstream_impacts": [], "total_relationships": 0}

    async def _build_cross_program_lineage_graph(self, component_name: str, impact_data: Dict) -> Dict[str, Any]:
        """Build lineage graph across programs"""
        try:
            lineage_nodes = []
            lineage_edges = []
            
            # Add central component node
            central_node = {
                "id": f"component_{component_name}",
                "name": component_name,
                "type": "central_component",
                "level": 0
            }
            lineage_nodes.append(central_node)
            
            # Add upstream nodes and edges
            for i, upstream in enumerate(impact_data.get("upstream_dependencies", [])):
                upstream_node = {
                    "id": f"upstream_{upstream['source_artifact']}",
                    "name": upstream['source_artifact'],
                    "type": upstream['source_type'],
                    "level": -1
                }
                lineage_nodes.append(upstream_node)
                
                lineage_edges.append({
                    "source": upstream_node["id"],
                    "target": central_node["id"],
                    "relationship": upstream['relationship_type'],
                    "impact_level": upstream['impact_level']
                })
            
            # Add downstream nodes and edges
            for i, downstream in enumerate(impact_data.get("downstream_impacts", [])):
                downstream_node = {
                    "id": f"downstream_{downstream['dependent_artifact']}",
                    "name": downstream['dependent_artifact'],
                    "type": downstream['dependent_type'],
                    "level": 1
                }
                lineage_nodes.append(downstream_node)
                
                lineage_edges.append({
                    "source": central_node["id"],
                    "target": downstream_node["id"],
                    "relationship": downstream['relationship_type'],
                    "impact_level": downstream['impact_level']
                })
            
            return {
                "nodes": lineage_nodes,
                "edges": lineage_edges,
                "total_nodes": len(lineage_nodes),
                "total_edges": len(lineage_edges),
                "graph_complexity": "high" if len(lineage_nodes) > 10 else "medium" if len(lineage_nodes) > 5 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to build lineage graph: {e}")
            return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    async def _generate_cross_program_lineage_analysis_api(self, component_name: str,
                                                        impact_data: Dict, lineage_graph: Dict) -> str:
        """Generate cross-program lineage analysis using API"""
        
        lineage_summary = {
            "component_name": component_name,
            "upstream_count": len(impact_data.get("upstream_dependencies", [])),
            "downstream_count": len(impact_data.get("downstream_impacts", [])),
            "total_nodes": lineage_graph.get("total_nodes", 0),
            "graph_complexity": lineage_graph.get("graph_complexity", "low")
        }
        
        prompt = f"""
        Generate cross-program data lineage analysis for: {component_name}
        
        Lineage Summary:
        - {lineage_summary['upstream_count']} upstream dependencies
        - {lineage_summary['downstream_count']} downstream impacts
        - {lineage_summary['total_nodes']} total components in lineage
        - Graph complexity: {lineage_summary['graph_complexity']}
        
        Upstream Dependencies: {[dep['source_artifact'] for dep in impact_data.get('upstream_dependencies', [])[:5]]}
        Downstream Impacts: {[imp['dependent_artifact'] for imp in impact_data.get('downstream_impacts', [])[:5]]}
        
        Provide comprehensive analysis covering:
        
        **Lineage Overview:**
        - Component's role in the data ecosystem
        - Critical dependencies and relationships
        - Business impact and importance
        
        **Upstream Analysis:**
        - Data sources and origins
        - Dependency chain analysis
        - Risk assessment for source changes
        
        **Downstream Analysis:**
        - Impact propagation patterns
        - Affected systems and processes
        - Change management considerations
        
        **Cross-Program Dependencies:**
        - Inter-program data flows
        - Synchronization requirements
        - Data consistency mechanisms
        
        Write as comprehensive lineage documentation for architecture and governance teams.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=800)
        except Exception as e:
            self.logger.error(f"Cross-program lineage analysis generation failed: {e}")
            return self._generate_fallback_cross_program_analysis(component_name, lineage_summary)

    def _generate_fallback_cross_program_analysis(self, component_name: str, lineage_summary: Dict) -> str:
        """Generate fallback cross-program analysis when API fails"""
        analysis = f"## Cross-Program Data Lineage: {component_name}\n\n"
        
        analysis += "### Lineage Overview\n"
        analysis += f"Component {component_name} sits at the center of a {lineage_summary['graph_complexity']} complexity data lineage:\n"
        analysis += f"- Has {lineage_summary['upstream_count']} upstream dependencies\n"
        analysis += f"- Impacts {lineage_summary['downstream_count']} downstream components\n"
        analysis += f"- Total lineage network includes {lineage_summary['total_nodes']} components\n\n"
        
        analysis += "### Impact Assessment\n"
        if lineage_summary['upstream_count'] > 5 or lineage_summary['downstream_count'] > 5:
            analysis += "**High Impact Component:** Changes to this component require extensive coordination and testing.\n"
        elif lineage_summary['upstream_count'] > 2 or lineage_summary['downstream_count'] > 2:
            analysis += "**Moderate Impact Component:** Standard change management procedures apply.\n"
        else:
            analysis += "**Low Impact Component:** Changes can be managed with minimal coordination.\n"
        
        analysis += f"\n### Complexity Characteristics\n"
        analysis += f"The {lineage_summary['graph_complexity']} complexity rating indicates the level of "
        analysis += "coordination required for changes and the potential for cascading impacts.\n"
        
        return analysis
    
    # ==================== Enhanced Lineage Methods for High Complexity ====================

    async def analyze_field_lineage_with_fallback(self, field_name: str) -> Dict[str, Any]:
        """Enhanced field lineage with timeout protection and partial saves"""
        try:
            # Check existing partial results first
            existing = await self._load_existing_partial(field_name)
            if existing:
                self.logger.info(f"📋 Found existing partial analysis for {field_name}")
                return existing
            
            # Get references with limit for high complexity
            field_references = await self._find_field_references(field_name)
            total_refs = len(field_references)
            
            # Adaptive strategy based on reference count
            if total_refs > 30:  # HIGH complexity like TMSCOTHU
                return await self._process_high_complexity_api(field_name, field_references)
            elif total_refs > 15:  # MEDIUM complexity  
                return await self._process_medium_complexity_api(field_name, field_references)
            else:  # LOW complexity
                return await self.analyze_field_lineage(field_name)  # Use existing method
                
        except Exception as e:
            # Save error state and return partial results
            return await self._handle_analysis_failure(field_name, str(e))

    async def _process_high_complexity_api(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """Process high complexity components like TMSCOTHU with smart limits using API"""
        
        # 1. Prioritize most important references
        prioritized_refs = self._prioritize_by_importance(references)[:20]  # Max 20
        
        # 2. Initialize partial result tracking
        result = {
            "field_name": field_name,
            "status": "partial",
            "total_references_found": len(references),
            "references_analyzed": 0,
            "high_complexity_mode": True,
            "analysis_strategy": "prioritized_sampling",
            "lineage_data": {"programs": set(), "operations": [], "transformations": []},
            "progress_log": []
        }
        
        # 3. Process in small batches with saves
        batch_size = 5
        for i in range(0, len(prioritized_refs), batch_size):
            batch = prioritized_refs[i:i + batch_size]
            
            try:
                # Process batch with 2-minute timeout
                async with asyncio.timeout(120):
                    for ref in batch:
                        ref_analysis = await self._analyze_field_reference_with_api(
                            field_name, ref["content"][:400],  # Truncate content
                            ref["chunk_type"], ref["program_name"]
                        )
                        
                        # Accumulate findings
                        result["lineage_data"]["programs"].add(ref["program_name"])
                        result["lineage_data"]["operations"].append(ref_analysis.get("operation_type", "UNKNOWN"))
                        result["references_analyzed"] += 1
                    
                    # Save progress every batch
                    await self._save_partial_progress(field_name, result)
                    progress = (result["references_analyzed"] / len(prioritized_refs)) * 100
                    result["progress_log"].append(f"Batch {i//batch_size + 1}: {progress:.1f}% complete")
                    
            except asyncio.TimeoutError:
                result["progress_log"].append(f"Batch {i//batch_size + 1}: TIMEOUT - continuing")
                continue
        
        result["llm_summary"] = await self._generate_partial_summary_api(field_name, result)
        result["lineage_data"]["programs"] = list(result["lineage_data"]["programs"])
        
        await self._save_final_partial_result(field_name, result)
        return result

    def _prioritize_by_importance(self, references: List[Dict]) -> List[Dict]:
        """Smart prioritization for high-complexity analysis"""
        def importance_score(ref):
            score = 0
            chunk_type = ref.get('chunk_type', '').lower()
            content = ref.get('content', '').upper()
            program_name = ref.get('program_name', '').upper()
            
            # Prioritize data definitions (highest importance)
            if chunk_type in ['file_section', 'working_storage', 'data_division']:
                score += 20
            elif chunk_type == 'procedure_division':
                score += 10
            
            # Prioritize transaction processing programs
            if any(keyword in program_name for keyword in ['TXN', 'TRANS', 'POST', 'BAL']):
                score += 15
            
            # Prioritize complex operations
            complex_ops = ['COMPUTE', 'MOVE', 'IF', 'PERFORM', 'CALL', 'READ', 'WRITE']
            score += sum(3 for op in complex_ops if op in content)
            
            # Prioritize copybook definitions
            if 'COPY' in content and ref.get('field_name', '') in content:
                score += 25
                
            return score
        
        return sorted(references, key=importance_score, reverse=True)

    async def _save_partial_progress(self, field_name: str, result: Dict):
        """Save intermediate progress to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO partial_analysis_cache 
                (component_name, agent_type, partial_data, progress_percent, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(field_name), 
                "lineage_analyzer",
                json.dumps(result, default=str),
                (result["references_analyzed"] / max(result.get("total_references_found", 1), 1)) * 100,
                "in_progress",
                dt.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    async def _load_existing_partial(self, field_name: str) -> Optional[Dict]:
        """Load existing partial analysis from cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT partial_data, progress_percent, status, timestamp
                FROM partial_analysis_cache
                WHERE component_name = ? AND agent_type = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (str(field_name), "lineage_analyzer"))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                partial_data, progress, status, timestamp = row
                # Check if data is recent (within 24 hours)
                analysis_time = dt.fromisoformat(timestamp)
                if (dt.now() - analysis_time).total_seconds() < 86400:  # 24 hours
                    return json.loads(partial_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load existing partial: {e}")
            return None

    async def _save_final_partial_result(self, field_name: str, result: Dict):
        """Save final partial result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE partial_analysis_cache 
                SET partial_data = ?, status = 'completed', progress_percent = 100, timestamp = ?
                WHERE component_name = ? AND agent_type = ?
            """, (
                json.dumps(result, default=str),
                dt.now().isoformat(),
                str(field_name),
                "lineage_analyzer"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save final result: {e}")

    async def _handle_analysis_failure(self, field_name: str, error: str) -> Dict[str, Any]:
        """Handle analysis failure and return partial results"""
        return {
            "field_name": field_name,
            "status": "error",
            "error": error,
            "analysis_type": "api_based",
            "timestamp": dt.now().isoformat(),
            "partial_data": None
        }

    async def _generate_partial_summary_api(self, field_name: str, result: Dict) -> str:
        """Generate API-based summary of partial analysis - FIXED for readable output"""
        try:
            programs_list = list(result["lineage_data"]["programs"])[:10]
            operations_summary = {}
            for op in result["lineage_data"]["operations"]:
                operations_summary[op] = operations_summary.get(op, 0) + 1
            
            prompt = f"""
            Create a clear, readable business summary for this field lineage analysis:
            
            Field: {field_name}
            Analysis Status: {result['status']} 
            References Analyzed: {result['references_analyzed']} of {result['total_references_found']}
            
            Key Programs: {', '.join(programs_list[:5])}
            Operations: {', '.join(operations_summary.keys())}
            
            Write a clear paragraph explaining:
            - What this field is used for in business terms
            - Which systems depend on it
            - Key operations performed
            - Business criticality level
            
            Write in plain English, no JSON, no technical jargon. Maximum 150 words.
            """
            
            api_response = await self._call_api_for_analysis(prompt, max_tokens=300)
            
            # Parse the response to ensure readable text
            readable_summary = self._ensure_readable_text(api_response)
            
            if readable_summary and len(readable_summary.strip()) > 20:
                return readable_summary
            else:
                return f"Field {field_name} is used across {len(programs_list)} programs with {result['references_analyzed']} references analyzed using {result.get('analysis_strategy', 'standard')} strategy."
                    
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return f"Analysis Summary: Field {field_name} found in {len(result['lineage_data']['programs'])} programs with {result['references_analyzed']} references processed using {result.get('analysis_strategy', 'standard')} strategy."

    # ==================== Full Lifecycle Analysis Methods ====================

    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze complete lifecycle of a component using API calls"""
        try:
            await self._load_existing_lineage()
            
            # FIXED: Ensure parameters are strings
            component_name = str(component_name)
            component_type = str(component_type)
            
            if component_type in ["file", "table"]:
                result = await self._analyze_data_component_lifecycle_api(component_name)
            elif component_type in ["program", "cobol"]:
                result = await self._analyze_program_lifecycle_api(component_name)
            elif component_type == "jcl":
                result = await self._analyze_jcl_lifecycle_api(component_name)
            else:
                result = {"error": f"Unsupported component type: {component_type}"}
            
            return self._add_processing_info(result)
                
        except Exception as e:
            return self._add_processing_info({
                "component_name": component_name,
                "component_type": component_type,
                "error": str(e),
                "status": "error"
            })

    async def _analyze_data_component_lifecycle_api(self, component_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze lifecycle of a data component (file/table)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find all programs that reference this component
        cursor.execute("""
            SELECT program_name, chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE content LIKE ? OR metadata LIKE ?
            LIMIT 30
        """, (f"%{component_name}%", f"%{component_name}%"))
        
        references = cursor.fetchall()
        conn.close()
        
        lifecycle_analysis = {
            "component_name": component_name,
            "creation_points": [],
            "read_operations": [],
            "update_operations": [],
            "delete_operations": [],
            "archival_points": [],
            "data_flow": {},
            "status": "success"
        }
        
        for program_name, chunk_id, chunk_type, content, metadata_str in references:
            # Analyze each reference with API
            operation_analysis = await self._analyze_component_operation_api(
                component_name, content, program_name, chunk_type
            )
            
            # Categorize operations
            op_type = operation_analysis.get("operation_type", "READ")
            operation_details = {
                "program": program_name,
                "chunk": chunk_id,
                "operation": operation_analysis,
                "content_snippet": content[:200] + "..." if len(content) > 200 else content
            }
            
            if op_type == "CREATE":
                lifecycle_analysis["creation_points"].append(operation_details)
            elif op_type == "READ":
                lifecycle_analysis["read_operations"].append(operation_details)
            elif op_type == "UPDATE":
                lifecycle_analysis["update_operations"].append(operation_details)
            elif op_type == "DELETE":
                lifecycle_analysis["delete_operations"].append(operation_details)
            elif op_type == "ARCHIVE":
                lifecycle_analysis["archival_points"].append(operation_details)
        
        # Generate comprehensive lifecycle report with API
        lifecycle_report = await self._generate_component_lifecycle_report_api(
            component_name, lifecycle_analysis
        )
        
        lifecycle_analysis["comprehensive_report"] = lifecycle_report
        
        return lifecycle_analysis

    async def _analyze_component_operation_api(self, component_name: str, content: str, 
                                     program_name: str, chunk_type: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze what operation a program performs on a component"""
        
        content_preview = content[:400] if len(content) > 400 else content
        
        prompt = f"""
        Analyze what operation program "{program_name}" performs on component "{component_name}":
        
        Code Context ({chunk_type}):
        {content_preview}
        
        Determine:
        1. Primary operation (CREATE, READ, UPDATE, DELETE, ARCHIVE, COPY)
        2. Business purpose
        3. Data transformation details
        4. Timing/frequency (if apparent)
        5. Dependencies
        
        Return as JSON:
        {{
            "operation_type": "READ",
            "business_purpose": "Data validation",
            "data_transformations": ["format conversion"],
            "timing_frequency": "daily",
            "dependencies": ["other_component"],
            "confidence": 0.9
        }}
        """
        
        try:
            response_text = await self._call_api_for_analysis(prompt, max_tokens=300)
            
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Failed to parse component operation analysis: {str(e)}")
        
        return {
            "operation_type": "READ",
            "business_purpose": "Component access detected",
            "data_transformations": [],
            "timing_frequency": "unknown",
            "dependencies": [],
            "confidence": 0.5
        }

    async def _generate_component_lifecycle_report_api(self, component_name: str, 
                                                lifecycle_data: Dict) -> str:
        """✅ API-BASED: Generate comprehensive component lifecycle report"""
        
        summary_data = {
            "component_name": component_name,
            "creation_points": len(lifecycle_data['creation_points']),
            "read_operations": len(lifecycle_data['read_operations']),
            "update_operations": len(lifecycle_data['update_operations']),
            "delete_operations": len(lifecycle_data['delete_operations']),
            "archival_points": len(lifecycle_data['archival_points'])
        }
        
        prompt = f"""
        Generate a comprehensive lifecycle report for component "{component_name}":
        
        Lifecycle Summary:
        {json.dumps(summary_data, indent=2)}
        
        Generate a detailed report covering:
        1. Component Overview and Purpose
        2. Creation and Initialization Process
        3. Operational Usage Patterns
        4. Data Maintenance Activities
        5. End-of-Life Management
        6. Risk Assessment
        7. Optimization Recommendations
        
        Format as a professional technical document (400 words max).
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=500)
        except Exception as e:
            self.logger.warning(f"Failed to generate lifecycle report: {str(e)}")
            return f"Lifecycle report for {component_name}: {sum(summary_data.values())} total operations identified."

    async def _analyze_program_lifecycle_api(self, program_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze lifecycle of a COBOL program"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get program chunks
        cursor.execute("""
            SELECT chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE program_name = ?
        """, (str(program_name),))
        
        chunks = cursor.fetchall()
        conn.close()
        
        program_analysis = {
            "program_name": program_name,
            "total_chunks": len(chunks),
            "chunk_breakdown": {},
            "external_calls": [],
            "file_operations": [],
            "db_operations": [],
            "called_by": [],
            "status": "success"
        }
        
        # Analyze chunks
        for chunk_id, chunk_type, content, metadata_str in chunks:
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            if chunk_type in program_analysis["chunk_breakdown"]:
                program_analysis["chunk_breakdown"][chunk_type] += 1
            else:
                program_analysis["chunk_breakdown"][chunk_type] = 1
            
            # Extract operations
            if "file_operations" in metadata:
                program_analysis["file_operations"].extend(metadata["file_operations"])
            
            if "sql_operations" in metadata:
                program_analysis["db_operations"].extend(metadata["sql_operations"])
        
        # Generate program lifecycle report with API
        lifecycle_report = await self._generate_program_lifecycle_report_api(program_name, program_analysis)
        program_analysis["lifecycle_report"] = lifecycle_report
        
        return program_analysis

    async def _generate_program_lifecycle_report_api(self, program_name: str, analysis_data: Dict) -> str:
        """✅ API-BASED: Generate program lifecycle report"""
        
        report_summary = {
            "program_name": program_name,
            "total_chunks": analysis_data['total_chunks'],
            "chunk_types": analysis_data['chunk_breakdown'],
            "file_operations": len(analysis_data['file_operations']),
            "db_operations": len(analysis_data['db_operations'])
        }
        
        prompt = f"""
        Generate a lifecycle report for COBOL program "{program_name}":
        
        Program Summary:
        {json.dumps(report_summary, indent=2)}
        
        Generate a detailed report covering:
        1. Program Purpose and Function
        2. Structural Analysis
        3. Data Processing Activities
        4. Integration Points
        5. Dependencies and Relationships
        6. Maintenance Considerations
        7. Performance and Optimization Notes
        
        Format as a technical analysis document (400 words max).
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=500)
        except Exception as e:
            self.logger.warning(f"Failed to generate program lifecycle report: {str(e)}")
            return f"Program lifecycle report for {program_name}: {report_summary['total_chunks']} chunks analyzed."

    async def _analyze_jcl_lifecycle_api(self, jcl_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze lifecycle of a JCL job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get JCL steps
        cursor.execute("""
            SELECT chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE program_name = ? AND chunk_type IN ('job_step', 'job_header')
            ORDER BY chunk_id
        """, (str(jcl_name),))
        
        steps = cursor.fetchall()
        conn.close()
        
        jcl_analysis = {
            "jcl_name": jcl_name,
            "total_steps": len([s for s in steps if s[1] == 'job_step']),
            "step_details": [],
            "data_flow": [],
            "dependencies": [],
            "scheduling_info": {},
            "status": "success"
        }
        
        # Generate JCL lifecycle report
        lifecycle_report = await self._generate_jcl_lifecycle_report_api(jcl_name, jcl_analysis)
        jcl_analysis["lifecycle_report"] = lifecycle_report
        
        return jcl_analysis

    async def _generate_jcl_lifecycle_report_api(self, jcl_name: str, analysis_data: Dict) -> str:
        """✅ API-BASED: Generate JCL lifecycle report"""
        
        report_summary = {
            "jcl_name": jcl_name,
            "total_steps": analysis_data['total_steps']
        }
        
        prompt = f"""
        Generate a lifecycle report for JCL job "{jcl_name}":
        
        JCL Summary:
        {json.dumps(report_summary, indent=2)}
        
        Generate a detailed report covering:
        1. Job Purpose and Function
        2. Step Analysis
        3. Data Flow
        4. Dependencies
        5. Scheduling Considerations
        6. Error Handling
        7. Optimization Opportunities
        
        Format as a job analysis document (300 words max).
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=400)
        except Exception as e:
            self.logger.warning(f"Failed to generate JCL lifecycle report: {str(e)}")
            return f"JCL lifecycle report for {jcl_name}: {report_summary['total_steps']} steps analyzed."

    # ==================== Summary Generation Methods ====================

    async def generate_lineage_summary(self, component_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Generate a comprehensive lineage summary for any component"""
        try:
            await self._load_existing_lineage()
            
            # Determine component type
            component_type = await self._determine_component_type(component_name)
            
            # Get appropriate analysis
            if component_type == "field":
                analysis = await self.analyze_field_lineage(component_name)
            else:
                analysis = await self.analyze_complete_data_flow(component_name, component_type)
            
            # Generate executive summary with API
            executive_summary = await self._generate_executive_summary_api(
                component_name, component_type, analysis
            )
            
            result = {
                "component_name": component_name,
                "component_type": component_type,
                "executive_summary": executive_summary,
                "detailed_analysis": analysis,
                "summary_generated": dt.now().isoformat(),
                "status": "success"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            return self._add_processing_info({
                "component_name": component_name,
                "error": str(e),
                "status": "error"
            })

    async def _determine_component_type(self, component_name: str) -> str:
        """Determine the type of component"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if it's a field (look in file metadata fields)
            cursor.execute("""
                SELECT COUNT(*) FROM file_metadata 
                WHERE fields LIKE ? OR fields LIKE ? OR fields LIKE ?
            """, (f"%{component_name}%", f"%{component_name.upper()}%", f"%{component_name.lower()}%"))
            
            if cursor.fetchone()[0] > 0:
                conn.close()
                return "field"
        except sqlite3.OperationalError:
            pass  # Table might not exist
        
        try:
            # Check if it's a table
            cursor.execute("SELECT COUNT(*) FROM file_metadata WHERE table_name = ?", (str(component_name),))
            if cursor.fetchone()[0] > 0:
                conn.close()
                return "table"
        except sqlite3.OperationalError:
            pass
        
        # Check if it's a program
        cursor.execute("SELECT COUNT(*) FROM program_chunks WHERE program_name = ?", (str(component_name),))
        if cursor.fetchone()[0] > 0:
            # Further determine if it's COBOL or JCL
            cursor.execute("""
                SELECT chunk_type FROM program_chunks 
                WHERE program_name = ? 
                LIMIT 1
            """, (str(component_name),))
            chunk_type = cursor.fetchone()
            conn.close()
            
            if chunk_type and 'job' in chunk_type[0]:
                return "jcl"
            else:
                return "program"
        
        conn.close()
        return "unknown"

    async def _generate_executive_summary_api(self, component_name: str, component_type: str, analysis: Dict) -> str:
        """✅ API-BASED: Generate executive summary for lineage analysis"""
        
        summary_data = {
            "component_name": component_name,
            "component_type": component_type,
            "analysis_status": analysis.get("status", "unknown")
        }
        
        prompt = f"""
        Generate an executive summary for the lineage analysis of {component_type} "{component_name}":
        
        Summary Data:
        {json.dumps(summary_data, indent=2)}
        
        Create a concise executive summary covering:
        1. Component purpose and importance
        2. Key usage patterns and dependencies
        3. Risk assessment
        4. Business impact
        5. Recommended actions
        
        Keep it under 200 words and suitable for management review.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=300)
        except Exception as e:
            self.logger.warning(f"Failed to generate executive summary: {str(e)}")
            return f"Executive Summary: {component_type.title()} {component_name} analyzed successfully."

    # ==================== Cleanup and Context Management ====================
    
    def cleanup(self):
        """Cleanup method for API-based agent"""
        self.logger.info("🧹 Cleaning up FIXED API-based Lineage Analyzer agent...")
        
        # Clear in-memory data structures
        if hasattr(self, 'lineage_graph'):
            self.lineage_graph.clear()
        
        # Reset flags
        self._lineage_loaded = False
        
        self.logger.info("✅ FIXED API-based Lineage Analyzer cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __repr__(self):
        return (f"LineageAnalyzerAgent("
                f"api_based=True, "
                f"coordinator={type(self.coordinator).__name__}, "
                f"gpu_id={self.gpu_id}, "
                f"fixed=True)")


# ==================== Backwards Compatibility Functions ====================

async def quick_field_lineage_analysis_api(field_name: str, coordinator) -> Dict[str, Any]:
    """Quick field lineage analysis using FIXED API-based agent"""
    agent = LineageAnalyzerAgent(coordinator)
    try:
        return await agent.analyze_field_lineage(field_name)
    finally:
        agent.cleanup()

async def quick_component_lifecycle_api(component_name: str, component_type: str, coordinator) -> Dict[str, Any]:
    """Quick component lifecycle analysis using FIXED API-based agent"""
    agent = LineageAnalyzerAgent(coordinator)
    try:
        return await agent.analyze_complete_data_flow(component_name, component_type)
    finally:
        agent.cleanup()

async def quick_dependency_analysis_api(component_name: str, coordinator) -> List[str]:
    """Quick dependency analysis using FIXED API-based agent"""
    agent = LineageAnalyzerAgent(coordinator)
    try:
        # Simple dependency extraction from field references
        field_refs = await agent._find_field_references(component_name)
        dependencies = []
        
        for ref in field_refs:
            if ref.get("program_name"):
                dependencies.append(f"program:{ref['program_name']}")
            if ref.get("table_name"):
                dependencies.append(f"table:{ref['table_name']}")
        
        return list(set(dependencies))
    finally:
        agent.cleanup()

# ==================== Example Usage ====================

async def example_fixed_lineage_usage():
    """Example of how to use the FIXED API-based lineage analyzer"""
    
    # Assuming you have an API coordinator set up
    from api_coordinator import create_api_coordinator_from_endpoints
    
    # Define your model server endpoints
    gpu_endpoints = {
        1: "http://gpu-server-1:8000",
        2: "http://gpu-server-2:8001"
    }
    
    # Create API coordinator
    coordinator = create_api_coordinator_from_endpoints(gpu_endpoints)
    await coordinator.initialize()
    
    try:
        # Create FIXED API-based lineage analyzer
        lineage_agent = LineageAnalyzerAgent(coordinator)
        
        # Analyze field lineage (FIXED API calls)
        field_analysis = await lineage_agent.analyze_field_lineage("CUSTOMER-ID")
        print(f"Field analysis status: {field_analysis['status']}")
        
        # Analyze component data flow (FIXED API calls)
        data_flow_analysis = await lineage_agent.analyze_complete_data_flow("CUSTOMER-PROGRAM", "program")
        print(f"Data flow analysis status: {data_flow_analysis['status']}")
        
        # Test specific methods that were failing
        program_chunks = await lineage_agent._get_program_chunks("CUSTOMER-PROGRAM")
        print(f"Found {len(program_chunks)} program chunks")
        
        # Test file access data retrieval
        file_access = await lineage_agent._get_file_access_data("CUSTOMER-FILE")
        print(f"File access data: {file_access['total_access_points']} access points")
        
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_fixed_lineage_usage())
                    
         