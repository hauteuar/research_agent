# agents/lineage_analyzer_agent.py
"""
Agent 4: Field Lineage & Lifecycle Analyzer
Tracks field usage, data flow, and component lifecycle across the mainframe system
"""

import asyncio
import sqlite3
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict

from vllm import AsyncLLMEngine, SamplingParams

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

class LineageAnalyzerAgent:
    """Agent for analyzing field lineage and component lifecycle"""
    
    def __init__(self, llm_engine: AsyncLLMEngine, db_path: str, gpu_id: int):
        self.llm_engine = llm_engine
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize lineage tracking tables
        self._init_lineage_tables()
        
        # In-memory lineage graph
        self.lineage_graph = nx.DiGraph()
        
        # Load existing lineage data
        asyncio.create_task(self._load_existing_lineage())
    
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
        """)
        
        conn.commit()
        conn.close()
    
    async def _load_existing_lineage(self):
        """Load existing lineage data into memory graph"""
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
            self.logger.info(f"Loaded lineage graph with {self.lineage_graph.number_of_nodes()} nodes and {self.lineage_graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing lineage: {str(e)}")
    
    async def analyze_field_lineage(self, field_name: str) -> Dict[str, Any]:
        """Analyze complete lineage for a specific field"""
        try:
            # Find all references to this field
            field_references = await self._find_field_references(field_name)
            
            # Build lineage graph for this field
            field_lineage = await self._build_field_lineage_graph(field_name, field_references)
            
            # Analyze usage patterns
            usage_analysis = await self._analyze_field_usage_patterns(field_name, field_references)
            
            # Find data transformations
            transformations = await self._find_field_transformations(field_name, field_references)
            
            # Identify lifecycle stages
            lifecycle = await self._analyze_field_lifecycle(field_name, field_references)
            
            # Generate comprehensive report
            lineage_report = await self._generate_field_lineage_report(
                field_name, field_lineage, usage_analysis, transformations, lifecycle
            )
            
            return {
                "field_name": field_name,
                "lineage_graph": field_lineage,
                "usage_analysis": usage_analysis,
                "transformations": transformations,
                "lifecycle": lifecycle,
                "comprehensive_report": lineage_report,
                "impact_analysis": await self._analyze_field_impact(field_name, field_references)
            }
            
        except Exception as e:
            self.logger.error(f"Field lineage analysis failed for {field_name}: {str(e)}")
            return {"error": str(e)}
    
    async def _find_field_references(self, field_name: str) -> List[Dict[str, Any]]:
        """Find all references to a field across the codebase"""
        references = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search in program chunks
        cursor.execute("""
            SELECT program_name, chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE content LIKE ? OR metadata LIKE ?
        """, (f"%{field_name}%", f"%{field_name}%"))
        
        for program_name, chunk_id, chunk_type, content, metadata_str in cursor.fetchall():
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            # Check if field is actually referenced
            if self._is_field_referenced(field_name, content, metadata):
                ref_details = await self._analyze_field_reference_with_llm(
                    field_name, content, chunk_type, program_name
                )
                
                references.append({
                    "program_name": program_name,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "content": content,
                    "metadata": metadata,
                    "reference_details": ref_details
                })
        
        # Search in field catalog
        cursor.execute("""
            SELECT table_name, field_type, field_description, business_meaning
            FROM field_catalog
            WHERE field_name = ? OR field_name LIKE ?
        """, (field_name, f"%{field_name}%"))
        
        for table_name, field_type, description, business_meaning in cursor.fetchall():
            references.append({
                "type": "table_definition",
                "table_name": table_name,
                "field_type": field_type,
                "description": description,
                "business_meaning": business_meaning
            })
        
        conn.close()
        return references
    
    def _is_field_referenced(self, field_name: str, content: str, metadata: Dict) -> bool:
        """Check if field is actually referenced in content"""
        # Check in content with word boundaries
        pattern = r'\b' + re.escape(field_name) + r'\b'
        if re.search(pattern, content, re.IGNORECASE):
            return True
        
        # Check in metadata field lists
        for key in ['field_names', 'fields']:
            if key in metadata and field_name in metadata[key]:
                return True
        
        return False
    
    async def _analyze_field_reference_with_llm(self, field_name: str, content: str, 
                                               chunk_type: str, program_name: str) -> Dict[str, Any]:
        """Analyze how a field is referenced using LLM"""
        prompt = f"""
        Analyze how the field "{field_name}" is used in this {chunk_type} from program {program_name}:
        
        {content[:800]}...
        
        Determine:
        1. Operation type (READ, WRITE, UPDATE, DELETE, TRANSFORM, VALIDATE)
        2. Context of usage (input, output, calculation, validation, etc.)
        3. Any transformations applied to the field
        4. Business logic involving this field
        5. Data flow direction (source or target)
        
        Return as JSON:
        {{
            "operation_type": "READ/WRITE/UPDATE/etc",
            "usage_context": "description",
            "transformations": ["transformation1", "transformation2"],
            "business_logic": "description",
            "data_flow": "source/target/intermediate",
            "confidence": 0.9
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=500)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM field reference analysis: {str(e)}")
        
        # Fallback analysis
        return {
            "operation_type": self._infer_operation_type(field_name, content),
            "usage_context": "Field usage detected",
            "transformations": [],
            "business_logic": "Analysis not available",
            "data_flow": "unknown",
            "confidence": 0.5
        }
    
    def _infer_operation_type(self, field_name: str, content: str) -> str:
        """Infer operation type using regex patterns"""
        content_upper = content.upper()
        field_upper = field_name.upper()
        
        # COBOL patterns
        if f"MOVE {field_upper}" in content_upper or f"MOVE TO {field_upper}" in content_upper:
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
                
                # Chunk references field
                ref_details = ref.get("reference_details", {})
                operation_type = ref_details.get("operation_type", "REFERENCE")
                
                if operation_type in ["READ", "REFERENCE"]:
                    lineage_edges.append({
                        "source": field_node["id"],
                        "target": chunk_node["id"],
                        "relationship": "read_by",
                        "properties": ref_details
                    })
                else:
                    lineage_edges.append({
                        "source": chunk_node["id"],
                        "target": field_node["id"],
                        "relationship": operation_type.lower(),
                        "properties": ref_details
                    })
        
        return {
            "nodes": lineage_nodes,
            "edges": lineage_edges,
            "field_name": field_name
        }
    
    async def _analyze_field_usage_patterns(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """Analyze usage patterns for a field"""
        usage_stats = {
            "total_references": len(references),
            "programs_using": set(),
            "operation_types": defaultdict(int),
            "chunk_types": defaultdict(int),
            "table_definitions": []
        }
        
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
                
                ref_details = ref.get("reference_details", {})
                op_type = ref_details.get("operation_type", "REFERENCE")
                usage_stats["operation_types"][op_type] += 1
        
        # Convert sets to lists for JSON serialization
        usage_stats["programs_using"] = list(usage_stats["programs_using"])
        usage_stats["operation_types"] = dict(usage_stats["operation_types"])
        usage_stats["chunk_types"] = dict(usage_stats["chunk_types"])
        
        # Analyze patterns with LLM
        pattern_analysis = await self._analyze_usage_patterns_with_llm(field_name, usage_stats)
        
        return {
            "statistics": usage_stats,
            "pattern_analysis": pattern_analysis,
            "complexity_score": self._calculate_usage_complexity(usage_stats)
        }
    
    async def _analyze_usage_patterns_with_llm(self, field_name: str, usage_stats: Dict) -> str:
        """Analyze usage patterns using LLM"""
        prompt = f"""
        Analyze the usage patterns for field "{field_name}":
        
        Usage Statistics:
        {json.dumps(usage_stats, indent=2)}
        
        Provide insights on:
        1. Primary usage patterns
        2. Data flow characteristics
        3. Potential issues or risks
        4. Optimization opportunities
        5. Business importance indicators
        
        Provide a comprehensive analysis in narrative form.
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=800)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
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
    
    async def _find_field_transformations(self, field_name: str, references: List[Dict]) -> List[Dict[str, Any]]:
        """Find data transformations involving the field"""
        transformations = []
        
        for ref in references:
            if ref.get("type") != "table_definition":
                ref_details = ref.get("reference_details", {})
                
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
                    math_transforms = await self._extract_mathematical_transformations(
                        field_name, content, ref["program_name"]
                    )
                    transformations.extend(math_transforms)
        
        return transformations
    
    async def _extract_mathematical_transformations(self, field_name: str, content: str, program_name: str) -> List[Dict]:
        """Extract mathematical transformations from content"""
        prompt = f"""
        Extract mathematical transformations involving field "{field_name}" from this code:
        
        {content[:600]}...
        
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
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
    
    async def _analyze_field_lifecycle(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """Analyze the lifecycle of a field"""
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
                ref_details = ref.get("reference_details", {})
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
        
        # Analyze lifecycle completeness
        lifecycle_analysis = await self._analyze_lifecycle_completeness(field_name, lifecycle_stages)
        
        return {
            "stages": lifecycle_stages,
            "analysis": lifecycle_analysis,
            "lifecycle_score": self._calculate_lifecycle_score(lifecycle_stages)
        }
    
    async def _analyze_lifecycle_completeness(self, field_name: str, stages: Dict) -> str:
        """Analyze lifecycle completeness using LLM"""
        prompt = f"""
        Analyze the lifecycle completeness for field "{field_name}":
        
        Lifecycle Stages:
        {json.dumps(stages, indent=2)}
        
        Assess:
        1. Completeness of lifecycle coverage
        2. Missing lifecycle stages
        3. Data governance implications
        4. Potential data quality issues
        5. Recommendations for improvement
        
        Provide comprehensive lifecycle analysis.
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=700)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
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
    
    async def _analyze_field_impact(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """Analyze potential impact of changes to this field"""
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
                
                ref_details = ref.get("reference_details", {})
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
        
        # Generate detailed impact assessment
        impact_assessment = await self._generate_impact_assessment(field_name, impact_analysis)
        impact_analysis["detailed_assessment"] = impact_assessment
        
        return impact_analysis
    
    async def _generate_impact_assessment(self, field_name: str, impact_data: Dict) -> str:
        """Generate detailed impact assessment using LLM"""
        prompt = f"""
        Generate a detailed impact assessment for potential changes to field "{field_name}":
        
        Impact Data:
        {json.dumps(impact_data, indent=2, default=str)}
        
        Provide:
        1. Risk assessment and mitigation strategies
        2. Testing requirements
        3. Change management recommendations
        4. Business impact analysis
        5. Technical considerations
        
        Format as a comprehensive impact report.
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=1000)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
    async def _generate_field_lineage_report(self, field_name: str, lineage_graph: Dict,
                                           usage_analysis: Dict, transformations: List,
                                           lifecycle: Dict) -> str:
        """Generate comprehensive field lineage report using LLM"""
        prompt = f"""
        Generate a comprehensive data lineage report for field "{field_name}":
        
        Lineage Graph: {len(lineage_graph['nodes'])} nodes, {len(lineage_graph['edges'])} relationships
        Usage Analysis: {usage_analysis['statistics']['total_references']} total references across {len(usage_analysis['statistics']['programs_using'])} programs
        Transformations: {len(transformations)} transformation operations identified
        Lifecycle Score: {lifecycle['lifecycle_score']:.2f}
        
        Key Statistics:
        - Operation Types: {usage_analysis['statistics']['operation_types']}
        - Table Definitions: {len(usage_analysis['statistics']['table_definitions'])}
        - Complexity Score: {usage_analysis['complexity_score']:.2f}
        
        Generate a professional report including:
        1. Executive Summary
        2. Field Usage Overview
        3. Data Flow Analysis
        4. Transformation Summary
        5. Lifecycle Assessment
        6. Risk Analysis
        7. Recommendations
        
        Format as a structured report suitable for technical and business audiences.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1500)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """Analyze complete lifecycle of a component (file, table, program)"""
        try:
            if component_type in ["file", "table"]:
                return await self._analyze_data_component_lifecycle(component_name)
            elif component_type in ["program", "cobol"]:
                return await self._analyze_program_lifecycle(component_name)
            elif component_type == "jcl":
                return await self._analyze_jcl_lifecycle(component_name)
            else:
                return {"error": f"Unsupported component type: {component_type}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def find_dependencies(self, component_name: str) -> List[str]:
        """Find all dependencies for a component"""
        try:
            dependencies = set()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find direct references in code
            cursor.execute("""
                SELECT program_name, content, metadata
                FROM program_chunks
                WHERE content LIKE ? OR metadata LIKE ?
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            for program_name, content, metadata_str in cursor.fetchall():
                # Extract dependencies using LLM
                chunk_dependencies = await self._extract_dependencies_from_chunk(
                    component_name, content, program_name
                )
                dependencies.update(chunk_dependencies)
            
            # Find table dependencies
            cursor.execute("""
                SELECT field_name, table_name
                FROM field_catalog
                WHERE field_name = ? OR table_name = ?
            """, (component_name, component_name))
            
            for field_name, table_name in cursor.fetchall():
                if field_name != component_name:
                    dependencies.add(f"field:{field_name}")
                if table_name != component_name:
                    dependencies.add(f"table:{table_name}")
            
            conn.close()
            
            return list(dependencies)
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            return []
    
    async def _extract_dependencies_from_chunk(self, component_name: str, content: str, program_name: str) -> Set[str]:
        """Extract dependencies from a code chunk"""
        prompt = f"""
        Extract all dependencies for component "{component_name}" from this code:
        
        Program: {program_name}
        Content:
        {content[:500]}...
        
        Find dependencies including:
        1. Called programs/modules
        2. Referenced files/datasets
        3. Database tables
        4. Copybooks
        5. Parameters/variables
        
        Return as JSON array: ["dependency1", "dependency2", ...]
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=300)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '[' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                dependencies = json.loads(response_text[json_start:json_end])
                return set(dependencies)
        except Exception as e:
            self.logger.warning(f"Failed to parse dependencies: {str(e)}")
        
        return set()
    
    async def _analyze_data_component_lifecycle(self, component_name: str) -> Dict[str, Any]:
        """Analyze lifecycle of a data component (file/table)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find all programs that reference this component
        cursor.execute("""
            SELECT program_name, chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE content LIKE ? OR metadata LIKE ?
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
            "data_flow": {}
        }
        
        for program_name, chunk_id, chunk_type, content, metadata_str in references:
            # Analyze each reference with LLM
            operation_analysis = await self._analyze_component_operation(
                component_name, content, program_name, chunk_type
            )
            
            # Categorize operations
            op_type = operation_analysis.get("operation_type", "READ")
            operation_details = {
                "program": program_name,
                "chunk": chunk_id,
                "operation": operation_analysis,
                "content_snippet": content[:200]
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
        
        # Generate comprehensive lifecycle report
        lifecycle_report = await self._generate_component_lifecycle_report(
            component_name, lifecycle_analysis
        )
        
        lifecycle_analysis["comprehensive_report"] = lifecycle_report
        
        return lifecycle_analysis
    
    async def _analyze_component_operation(self, component_name: str, content: str, 
                                         program_name: str, chunk_type: str) -> Dict[str, Any]:
        """Analyze what operation a program performs on a component"""
        prompt = f"""
        Analyze what operation program "{program_name}" performs on component "{component_name}":
        
        Code Context ({chunk_type}):
        {content[:600]}...
        
        Determine:
        1. Primary operation (CREATE, READ, UPDATE, DELETE, ARCHIVE, COPY)
        2. Business purpose
        3. Data transformation details
        4. Timing/frequency (if apparent)
        5. Dependencies
        
        Return as JSON:
        {{
            "operation_type": "CREATE/READ/UPDATE/etc",
            "business_purpose": "description",
            "data_transformations": ["transformation1"],
            "timing_frequency": "daily/monthly/on-demand/etc",
            "dependencies": ["dependency1"],
            "confidence": 0.9
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=400)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
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
    
    async def _generate_component_lifecycle_report(self, component_name: str, 
                                                 lifecycle_data: Dict) -> str:
        """Generate comprehensive component lifecycle report"""
        prompt = f"""
        Generate a comprehensive lifecycle report for component "{component_name}":
        
        Lifecycle Analysis:
        - Creation Points: {len(lifecycle_data['creation_points'])}
        - Read Operations: {len(lifecycle_data['read_operations'])}
        - Update Operations: {len(lifecycle_data['update_operations'])}
        - Delete Operations: {len(lifecycle_data['delete_operations'])}
        - Archival Points: {len(lifecycle_data['archival_points'])}
        
        Generate a detailed report covering:
        1. Component Overview and Purpose
        2. Creation and Initialization Process
        3. Operational Usage Patterns
        4. Data Maintenance Activities
        5. End-of-Life Management
        6. Risk Assessment
        7. Optimization Recommendations
        
        Format as a professional technical document.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1200)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
    async def _analyze_program_lifecycle(self, program_name: str) -> Dict[str, Any]:
        """Analyze lifecycle of a COBOL program"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get program chunks
        cursor.execute("""
            SELECT chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE program_name = ?
        """, (program_name,))
        
        chunks = cursor.fetchall()
        
        # Find programs that call this program
        cursor.execute("""
            SELECT program_name, chunk_id, content
            FROM program_chunks
            WHERE content LIKE ? OR content LIKE ?
        """, (f"%CALL {program_name}%", f"%PERFORM {program_name}%"))
        
        callers = cursor.fetchall()
        conn.close()
        
        program_analysis = {
            "program_name": program_name,
            "total_chunks": len(chunks),
            "chunk_breakdown": {},
            "external_calls": [],
            "file_operations": [],
            "db_operations": [],
            "called_by": []
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
        
        # Analyze callers
        for caller_program, chunk_id, content in callers:
            call_analysis = await self._analyze_program_call(program_name, content, caller_program)
            program_analysis["called_by"].append(call_analysis)
        
        # Generate program lifecycle report
        lifecycle_report = await self._generate_program_lifecycle_report(program_name, program_analysis)
        program_analysis["lifecycle_report"] = lifecycle_report
        
        return program_analysis
    
    async def _analyze_program_call(self, called_program: str, content: str, caller_program: str) -> Dict[str, Any]:
        """Analyze how a program is called"""
        prompt = f"""
        Analyze how program "{called_program}" is called by "{caller_program}":
        
        {content[:400]}...
        
        Determine:
        1. Call method (CALL, PERFORM, etc.)
        2. Parameters passed
        3. Return values expected
        4. Call frequency/conditions
        5. Business purpose of the call
        
        Return as JSON:
        {{
            "call_method": "CALL/PERFORM",
            "parameters": ["param1", "param2"],
            "return_values": ["return1"],
            "call_conditions": "description",
            "business_purpose": "description"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=300)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                call_data = json.loads(response_text[json_start:json_end])
                call_data["caller_program"] = caller_program
                return call_data
        except Exception as e:
            self.logger.warning(f"Failed to parse program call analysis: {str(e)}")
        
        return {
            "caller_program": caller_program,
            "call_method": "UNKNOWN",
            "parameters": [],
            "return_values": [],
            "call_conditions": "Call detected",
            "business_purpose": "Analysis not available"
        }
    
    async def _generate_program_lifecycle_report(self, program_name: str, analysis_data: Dict) -> str:
        """Generate program lifecycle report"""
        prompt = f"""
        Generate a lifecycle report for COBOL program "{program_name}":
        
        Program Analysis:
        - Total Chunks: {analysis_data['total_chunks']}
        - Chunk Types: {analysis_data['chunk_breakdown']}
        - File Operations: {len(analysis_data['file_operations'])}
        - DB Operations: {len(analysis_data['db_operations'])}
        - Called By: {len(analysis_data['called_by'])} programs
        
        Generate a detailed report covering:
        1. Program Purpose and Function
        2. Structural Analysis
        3. Data Processing Activities
        4. Integration Points
        5. Dependencies and Relationships
        6. Maintenance Considerations
        7. Performance and Optimization Notes
        
        Format as a technical analysis document.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
    async def _analyze_jcl_lifecycle(self, jcl_name: str) -> Dict[str, Any]:
        """Analyze lifecycle of a JCL job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get JCL steps
        cursor.execute("""
            SELECT chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE program_name = ? AND chunk_type IN ('job_step', 'job_header')
            ORDER BY chunk_id
        """, (jcl_name,))
        
        steps = cursor.fetchall()
        conn.close()
        
        jcl_analysis = {
            "jcl_name": jcl_name,
            "total_steps": len([s for s in steps if s[1] == 'job_step']),
            "step_details": [],
            "data_flow": [],
            "dependencies": [],
            "scheduling_info": {}
        }
        
        # Analyze each step
        for chunk_id, chunk_type, content, metadata_str in steps:
            if chunk_type == 'job_step':
                step_analysis = await self._analyze_jcl_step_lifecycle(content, chunk_id)
                jcl_analysis["step_details"].append(step_analysis)
        
        # Analyze overall job flow
        job_flow_analysis = await self._analyze_jcl_flow(jcl_name, jcl_analysis["step_details"])
        jcl_analysis["flow_analysis"] = job_flow_analysis
        
        return jcl_analysis
    
    async def _analyze_jcl_step_lifecycle(self, step_content: str, step_id: str) -> Dict[str, Any]:
        """Analyze individual JCL step lifecycle"""
        prompt = f"""
        Analyze this JCL job step lifecycle:
        
        Step ID: {step_id}
        Content:
        {step_content}
        
        Determine:
        1. Step purpose and function
        2. Input datasets and sources
        3. Output datasets and targets
        4. Programs executed
        5. Data transformations
        6. Error handling
        7. Resource requirements
        
        Return as JSON:
        {{
            "step_id": "{step_id}",
            "purpose": "description",
            "inputs": ["input1", "input2"],
            "outputs": ["output1", "output2"],
            "programs": ["prog1"],
            "transformations": ["transform1"],
            "error_handling": "description",
            "resources": {{"memory": "1GB", "time": "30min"}}
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=500)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Failed to parse JCL step analysis: {str(e)}")
        
        return {
            "step_id": step_id,
            "purpose": "Job step execution",
            "inputs": [],
            "outputs": [],
            "programs": [],
            "transformations": [],
            "error_handling": "Standard",
            "resources": {}
        }
    
    async def _analyze_jcl_flow(self, jcl_name: str, step_details: List[Dict]) -> str:
        """Analyze overall JCL job flow"""
        prompt = f"""
        Analyze the complete job flow for JCL "{jcl_name}":
        
        Job Steps:
        {json.dumps(step_details, indent=2)}
        
        Provide analysis of:
        1. Overall job purpose and business function
        2. Data flow between steps
        3. Critical dependencies
        4. Potential failure points
        5. Performance characteristics
        6. Scheduling considerations
        7. Optimization opportunities
        
        Format as comprehensive job flow analysis.
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=1000)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()
    
    async def generate_lineage_summary(self, component_name: str) -> Dict[str, Any]:
        """Generate a comprehensive lineage summary for any component"""
        try:
            # Determine component type
            component_type = await self._determine_component_type(component_name)
            
            # Get appropriate analysis
            if component_type == "field":
                analysis = await self.analyze_field_lineage(component_name)
            else:
                analysis = await self.analyze_full_lifecycle(component_name, component_type)
            
            # Find dependencies
            dependencies = await self.find_dependencies(component_name)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                component_name, component_type, analysis, dependencies
            )
            
            return {
                "component_name": component_name,
                "component_type": component_type,
                "executive_summary": executive_summary,
                "detailed_analysis": analysis,
                "dependencies": dependencies,
                "summary_generated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _determine_component_type(self, component_name: str) -> str:
        """Determine the type of component"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if it's a field
        cursor.execute("SELECT COUNT(*) FROM field_catalog WHERE field_name = ?", (component_name,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            return "field"
        
        # Check if it's a table
        cursor.execute("SELECT COUNT(*) FROM table_schemas WHERE table_name = ?", (component_name,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            return "table"
        
        # Check if it's a program
        cursor.execute("SELECT COUNT(*) FROM program_chunks WHERE program_name = ?", (component_name,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            # Further determine if it's COBOL or JCL
            cursor = sqlite3.connect(self.db_path).cursor()
            cursor.execute("""
                SELECT chunk_type FROM program_chunks 
                WHERE program_name = ? 
                LIMIT 1
            """, (component_name,))
            chunk_type = cursor.fetchone()
            conn.close()
            
            if chunk_type and 'job' in chunk_type[0]:
                return "jcl"
            else:
                return "program"
        
        conn.close()
        return "unknown"
    
    async def _generate_executive_summary(self, component_name: str, component_type: str, 
                                        analysis: Dict, dependencies: List[str]) -> str:
        """Generate executive summary for lineage analysis"""
        prompt = f"""
        Generate an executive summary for the lineage analysis of {component_type} "{component_name}":
        
        Component Type: {component_type}
        Dependencies Found: {len(dependencies)}
        Analysis Results: {str(analysis)[:500]}...
        
        Create a concise executive summary covering:
        1. Component purpose and importance
        2. Key usage patterns and dependencies
        3. Risk assessment
        4. Business impact
        5. Recommended actions
        
        Keep it under 300 words and suitable for management review.
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=400)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return result.outputs[0].text.strip()