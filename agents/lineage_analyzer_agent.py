# agents/lineage_analyzer_agent.py
"""
Agent 4: Field Lineage & Lifecycle Analyzer
Tracks field usage, data flow, and component lifecycle across the mainframe system
"""

import asyncio
import sqlite3
import json
import uuid
import re
import os
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
    
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
                 gpu_id: int = None, coordinator=None):
        self.llm_engine = llm_engine
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator  # ADD coordinator reference
        self.logger = logging.getLogger(__name__)
        
        # ADD these lines for coordinator integration
        self._engine_lock = asyncio.Lock()
        self._engine_created = False
        self._using_coordinator_llm = False
        
        # Initialize lineage tracking tables
        self._init_lineage_tables()
        
        # In-memory lineage graph
        self.lineage_graph = nx.DiGraph()
        
        # Load existing lineage data (but don't use create_task in __init__)
        self._lineage_loaded = False
    
    async def _ensure_llm_engine(self):
        """Ensure LLM engine is available - use coordinator first, fallback to own"""
        async with self._engine_lock:
            if self.llm_engine is not None:
                return  # Already have engine
            
            # Try to get SHARED engine from coordinator first
            if self.coordinator is not None:
                try:
                    # Check if coordinator already has engines we can share
                    existing_engines = list(self.coordinator.llm_engine_pool.keys())
                    
                    for engine_key in existing_engines:
                        gpu_id = int(engine_key.split('_')[1])
                        
                        # Check if this GPU has enough memory for sharing
                        try:
                            from gpu_force_fix import GPUForcer
                            memory_info = GPUForcer.check_gpu_memory(gpu_id)
                            free_gb = memory_info.get('free_gb', 0)
                            
                            if free_gb >= 1.0:  # Can share this GPU
                                self.llm_engine = self.coordinator.llm_engine_pool[engine_key]
                                self.gpu_id = gpu_id
                                self._using_coordinator_llm = True
                                self.logger.info(f"LineageAnalyzer SHARING coordinator's LLM on GPU {gpu_id}")
                                return
                        except Exception as e:
                            self.logger.warning(f"Error checking GPU {gpu_id} for sharing: {e}")
                            continue
                    
                    # If no engine can be shared, get a new GPU
                    best_gpu = await self.coordinator.get_available_gpu_for_agent("lineage_analyzer")
                    if best_gpu is not None:
                        engine = await self.coordinator.get_or_create_llm_engine(best_gpu)
                        self.llm_engine = engine
                        self.gpu_id = best_gpu
                        self._using_coordinator_llm = True
                        self.logger.info(f"LineageAnalyzer using coordinator's NEW LLM on GPU {best_gpu}")
                        return
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get LLM from coordinator: {e}")
            
            # Try global coordinator as fallback
            if not self._engine_created:
                try:
                    from opulence_coordinator import get_dynamic_coordinator
                    global_coordinator = get_dynamic_coordinator()
                    
                    # Try to share existing engines first
                    existing_engines = list(global_coordinator.llm_engine_pool.keys())
                    for engine_key in existing_engines:
                        gpu_id = int(engine_key.split('_')[1])
                        try:
                            from gpu_force_fix import GPUForcer
                            memory_info = GPUForcer.check_gpu_memory(gpu_id)
                            free_gb = memory_info.get('free_gb', 0)
                            
                            if free_gb >= 1.0:
                                self.llm_engine = global_coordinator.llm_engine_pool[engine_key]
                                self.gpu_id = gpu_id
                                self._using_coordinator_llm = True
                                self.logger.info(f"LineageAnalyzer SHARING global coordinator's LLM on GPU {gpu_id}")
                                return
                        except Exception as e:
                            continue
                    
                    # If no sharing possible, get new GPU
                    best_gpu = await global_coordinator.get_available_gpu_for_agent("lineage_analyzer")
                    if best_gpu is not None:
                        engine = await global_coordinator.get_or_create_llm_engine(best_gpu)
                        self.llm_engine = engine
                        self.gpu_id = best_gpu
                        self._using_coordinator_llm = True
                        self.logger.info(f"LineageAnalyzer using global coordinator's NEW LLM on GPU {best_gpu}")
                        return
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get LLM from global coordinator: {e}")
            
            # Last resort: create own engine
            if not self._engine_created:
                await self._create_fallback_llm_engine()

    async def _create_fallback_llm_engine(self):
        """Create own LLM engine as last resort"""
        try:
            from gpu_force_fix import GPUForcer
            
            # Find GPU with most memory
            best_gpu = None
            best_memory = 0
            
            for gpu_id in range(4):  # Check all 4 GPUs
                try:
                    memory_info = GPUForcer.check_gpu_memory(gpu_id)
                    free_gb = memory_info.get('free_gb', 0)
                    if free_gb > best_memory:
                        best_memory = free_gb
                        best_gpu = gpu_id
                except:
                    continue
            
            if best_gpu is None or best_memory < 0.5:
                raise RuntimeError(f"No GPU found with sufficient memory. Best: {best_memory:.1f}GB")
            
            self.logger.warning(f"LineageAnalyzer creating FALLBACK LLM on GPU {best_gpu} with {best_memory:.1f}GB")
            
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                GPUForcer.force_gpu_environment(best_gpu)
                
                # Create MINIMAL engine to reduce memory usage
                engine_args = GPUForcer.create_vllm_engine_args(
                    "microsoft/DialoGPT-small",  # Use smaller model as fallback
                    1024  # Smaller context
                )
                engine_args.gpu_memory_utilization = 0.2  # Use even less memory
                
                from vllm import AsyncLLMEngine
                self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.gpu_id = best_gpu
                self._engine_created = True
                self._using_coordinator_llm = False
                
                self.logger.info(f"✅ LineageAnalyzer fallback LLM created on GPU {best_gpu}")
                
            finally:
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                    
        except Exception as e:
            self.logger.error(f"Failed to create fallback LLM engine: {str(e)}")
            raise

    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'lineage_analyzer'
            result['using_coordinator_llm'] = self._using_coordinator_llm
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
    
    async def analyze_field_lineage(self, field_name: str) -> Dict[str, Any]:
        """Analyze complete lineage for a specific field"""
        try:
            await self._ensure_llm_engine()  # Ensure engine is available
            await self._load_existing_lineage()  # Load lineage data
            
            # Find all references to this field
            field_references = await self._find_field_references(field_name)
            
            if not field_references:
                return self._add_processing_info({
                    "field_name": field_name,
                    "error": "No references found for this field",
                    "suggestions": "Check if field name is correct or if data has been processed"
                })
            
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
            
            result = {
                "field_name": field_name,
                "lineage_graph": field_lineage,
                "usage_analysis": usage_analysis,
                "transformations": transformations,
                "lifecycle": lifecycle,
                "comprehensive_report": lineage_report,
                "impact_analysis": await self._analyze_field_impact(field_name, field_references),
                "status": "success"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Field lineage analysis failed for {field_name}: {str(e)}")
            return self._add_processing_info({
                "field_name": field_name,
                "error": str(e),
                "status": "error"
            })
    
    async def _find_field_references(self, field_name: str) -> List[Dict[str, Any]]:
        """Find all references to a field across the codebase"""
        references = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Search in program chunks with more flexible patterns
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
                
                for program_name, chunk_id, chunk_type, content, metadata_str in cursor.fetchall():
                    # Avoid duplicates
                    if any(ref.get('chunk_id') == chunk_id for ref in references):
                        continue
                        
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
                            "content": content[:500] + "..." if len(content) > 500 else content,  # Truncate for memory
                            "metadata": metadata,
                            "reference_details": ref_details
                        })
                        
                        # Limit to prevent memory issues
                        if len(references) >= 50:
                            break
                
                if len(references) >= 50:
                    break
            
            # Check for table definitions in a separate query
            try:
                cursor.execute("""
                    SELECT DISTINCT table_name, 'VARCHAR' as field_type, 
                           'Field definition' as field_description, 
                           'Business field' as business_meaning
                    FROM file_metadata
                    WHERE fields LIKE ?
                    LIMIT 10
                """, (f"%{field_name}%",))
                
                for table_name, field_type, description, business_meaning in cursor.fetchall():
                    references.append({
                        "type": "table_definition",
                        "table_name": table_name,
                        "field_type": field_type,
                        "description": description,
                        "business_meaning": business_meaning
                    })
            except sqlite3.OperationalError:
                # Table might not exist, that's ok
                pass
        
        except Exception as e:
            self.logger.error(f"Error finding field references: {str(e)}")
        
        finally:
            conn.close()
        
        return references
    
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
    
    async def _analyze_field_reference_with_llm(self, field_name: str, content: str, 
                                               chunk_type: str, program_name: str) -> Dict[str, Any]:
        """Analyze how a field is referenced using LLM"""
        await self._ensure_llm_engine()
        
        # Truncate content to avoid token limits
        content_preview = content[:600] if len(content) > 600 else content
        
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
            sampling_params = SamplingParams(temperature=0.2, max_tokens=400)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            
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
            "field_name": field_name,
            "total_nodes": len(lineage_nodes),
            "total_edges": len(lineage_edges)
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
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.3, max_tokens=400)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate pattern analysis: {str(e)}")
            return f"Field {field_name} is used across {len(usage_stats['programs_using'])} programs with {usage_stats['total_references']} total references."
    
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
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.2, max_tokens=500)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            
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
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.3, max_tokens=500)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
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
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.3, max_tokens=600)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate impact assessment: {str(e)}")
            return f"Impact assessment for {field_name}: {impact_summary['risk_level']} risk level with {impact_summary['affected_programs']} affected programs."
    
    async def _generate_field_lineage_report(self, field_name: str, lineage_graph: Dict,
                                           usage_analysis: Dict, transformations: List,
                                           lifecycle: Dict) -> str:
        """Generate comprehensive field lineage report using LLM"""
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate lineage report: {str(e)}")
            return f"Lineage Report for {field_name}: Field found in {report_data['programs_count']} programs with {report_data['total_references']} references."
    
    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """Analyze complete lifecycle of a component (file, table, program)"""
        try:
            await self._ensure_llm_engine()
            await self._load_existing_lineage()
            
            if component_type in ["file", "table"]:
                result = await self._analyze_data_component_lifecycle(component_name)
            elif component_type in ["program", "cobol"]:
                result = await self._analyze_program_lifecycle(component_name)
            elif component_type == "jcl":
                result = await self._analyze_jcl_lifecycle(component_name)
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
    
    async def find_dependencies(self, component_name: str) -> List[str]:
        """Find all dependencies for a component"""
        try:
            await self._ensure_llm_engine()
            
            dependencies = set()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find direct references in code
            cursor.execute("""
                SELECT program_name, content, metadata
                FROM program_chunks
                WHERE content LIKE ? OR metadata LIKE ?
                LIMIT 50
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            for program_name, content, metadata_str in cursor.fetchall():
                # Extract dependencies using LLM
                chunk_dependencies = await self._extract_dependencies_from_chunk(
                    component_name, content, program_name
                )
                dependencies.update(chunk_dependencies)
            
            # Find table dependencies
            try:
                cursor.execute("""
                    SELECT DISTINCT table_name
                    FROM file_metadata
                    WHERE fields LIKE ? OR table_name LIKE ?
                    LIMIT 20
                """, (f"%{component_name}%", f"%{component_name}%"))
                
                for (table_name,) in cursor.fetchall():
                    if table_name != component_name:
                        dependencies.add(f"table:{table_name}")
            except sqlite3.OperationalError:
                pass  # Table might not exist
            
            conn.close()
            
            return list(dependencies)
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            return []
    
    async def _extract_dependencies_from_chunk(self, component_name: str, content: str, program_name: str) -> Set[str]:
        """Extract dependencies from a code chunk"""
        await self._ensure_llm_engine()
        
        content_preview = content[:400] if len(content) > 400 else content
        
        prompt = f"""
        Extract all dependencies for component "{component_name}" from this code:
        
        Program: {program_name}
        Content:
        {content_preview}
        
        Find dependencies including:
        1. Called programs/modules
        2. Referenced files/datasets
        3. Database tables
        4. Copybooks
        5. Parameters/variables
        
        Return as JSON array: ["dependency1", "dependency2", ...]
        """
        
        try:
            sampling_params = SamplingParams(temperature=0.2, max_tokens=200)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            
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
        
        # Generate comprehensive lifecycle report
        lifecycle_report = await self._generate_component_lifecycle_report(
            component_name, lifecycle_analysis
        )
        
        lifecycle_analysis["comprehensive_report"] = lifecycle_report
        
        return lifecycle_analysis
    
    async def _analyze_component_operation(self, component_name: str, content: str, 
                                         program_name: str, chunk_type: str) -> Dict[str, Any]:
        """Analyze what operation a program performs on a component"""
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.2, max_tokens=300)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            
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
        await self._ensure_llm_engine()
        
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
            sampling_params = SamplingParams(temperature=0.2, max_tokens=800)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate lifecycle report: {str(e)}")
            return f"Lifecycle report for {component_name}: {sum(summary_data.values())} total operations identified."
    
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
            LIMIT 20
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
        await self._ensure_llm_engine()
        
        content_preview = content[:300] if len(content) > 300 else content
        
        prompt = f"""
        Analyze how program "{called_program}" is called by "{caller_program}":
        
        {content_preview}
        
        Determine:
        1. Call method (CALL, PERFORM, etc.)
        2. Parameters passed
        3. Return values expected
        4. Call frequency/conditions
        5. Business purpose of the call
        
        Return as JSON:
        {{
            "call_method": "CALL",
            "parameters": ["param1"],
            "return_values": ["return1"],
            "call_conditions": "description",
            "business_purpose": "description"
        }}
        """
        
        try:
            sampling_params = SamplingParams(temperature=0.2, max_tokens=250)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            
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
        await self._ensure_llm_engine()
        
        report_summary = {
            "program_name": program_name,
            "total_chunks": analysis_data['total_chunks'],
            "chunk_types": analysis_data['chunk_breakdown'],
            "file_operations": len(analysis_data['file_operations']),
            "db_operations": len(analysis_data['db_operations']),
            "called_by_count": len(analysis_data['called_by'])
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
            sampling_params = SamplingParams(temperature=0.2, max_tokens=800)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params,  request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate program lifecycle report: {str(e)}")
            return f"Program lifecycle report for {program_name}: {report_summary['total_chunks']} chunks analyzed."
    
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
            "scheduling_info": {},
            "status": "success"
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
        await self._ensure_llm_engine()
        
        content_preview = step_content[:400] if len(step_content) > 400 else step_content
        
        prompt = f"""
        Analyze this JCL job step lifecycle:
        
        Step ID: {step_id}
        Content:
        {content_preview}
        
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
            "inputs": ["input1"],
            "outputs": ["output1"],
            "programs": ["prog1"],
            "transformations": ["transform1"],
            "error_handling": "description",
            "resources": {{"memory": "1GB", "time": "30min"}}
        }}
        """
        
        try:
            sampling_params = SamplingParams(temperature=0.2, max_tokens=400)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            
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
        await self._ensure_llm_engine()
        
        flow_summary = {
            "jcl_name": jcl_name,
            "total_steps": len(step_details),
            "step_types": [step.get("purpose", "Unknown") for step in step_details]
        }
        
        prompt = f"""
        Analyze the complete job flow for JCL "{jcl_name}":
        
        Job Flow Summary:
        {json.dumps(flow_summary, indent=2)}
        
        Provide analysis of:
        1. Overall job purpose and business function
        2. Data flow between steps
        3. Critical dependencies
        4. Potential failure points
        5. Performance characteristics
        6. Scheduling considerations
        7. Optimization opportunities
        
        Format as comprehensive job flow analysis (300 words max).
        """
        
        try:
            sampling_params = SamplingParams(temperature=0.3, max_tokens=600)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate JCL flow analysis: {str(e)}")
            return f"JCL flow analysis for {jcl_name}: {len(step_details)} steps analyzed."
    
    async def generate_lineage_summary(self, component_name: str) -> Dict[str, Any]:
        """Generate a comprehensive lineage summary for any component"""
        try:
            await self._ensure_llm_engine()
            await self._load_existing_lineage()
            
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
            
            result = {
                "component_name": component_name,
                "component_type": component_type,
                "executive_summary": executive_summary,
                "detailed_analysis": analysis,
                "dependencies": dependencies,
                "summary_generated": datetime.now().isoformat(),
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
            cursor.execute("SELECT COUNT(*) FROM file_metadata WHERE table_name = ?", (component_name,))
            if cursor.fetchone()[0] > 0:
                conn.close()
                return "table"
        except sqlite3.OperationalError:
            pass
        
        # Check if it's a program
        cursor.execute("SELECT COUNT(*) FROM program_chunks WHERE program_name = ?", (component_name,))
        if cursor.fetchone()[0] > 0:
            # Further determine if it's COBOL or JCL
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
        await self._ensure_llm_engine()
        
        summary_data = {
            "component_name": component_name,
            "component_type": component_type,
            "dependencies_count": len(dependencies),
            "analysis_status": analysis.get("status", "unknown")
        }
        
        # Extract key metrics based on component type
        if component_type == "field" and "usage_analysis" in analysis:
            usage_stats = analysis["usage_analysis"].get("statistics", {})
            summary_data.update({
                "total_references": usage_stats.get("total_references", 0),
                "programs_using": len(usage_stats.get("programs_using", [])),
                "risk_level": analysis.get("impact_analysis", {}).get("risk_level", "UNKNOWN")
            })
        elif "total_chunks" in analysis:
            summary_data.update({
                "total_chunks": analysis["total_chunks"],
                "called_by_count": len(analysis.get("called_by", []))
            })
        
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
            sampling_params = SamplingParams(temperature=0.3, max_tokens=400)
            request_id = str(uuid.uuid4())
            result = await self.llm_engine.generate(prompt, sampling_params, request_id = request_id)
            return result.outputs[0].text.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate executive summary: {str(e)}")
            return f"Executive Summary: {component_type.title()} {component_name} analyzed with {len(dependencies)} dependencies found."
        