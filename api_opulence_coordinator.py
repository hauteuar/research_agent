#!/usr/bin/env python3
"""
FIXED Lineage Analyzer Agent - Multiple Small API Calls Version
CRITICAL FIXES:
1. Split large prompts into multiple focused API calls
2. Increased max_tokens and temperature for better analysis
3. Each analysis step uses specific, smaller prompts
4. Progressive analysis building approach
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
    """FIXED: Lineage Analyzer with Multiple Small API Calls"""
    
    def __init__(self, coordinator, llm_engine=None, db_path: str = "opulence_data.db", gpu_id: int = 0):
        super().__init__(coordinator, "lineage_analyzer", db_path, gpu_id)  
        
        self.coordinator = coordinator
        self._init_lineage_tables()
        self.lineage_graph = nx.DiGraph()
        self._lineage_loaded = False
        
        # FIXED: Better API parameters for Llama-3B with 4096 context
        self.api_params = {
            "max_tokens": 800,      # INCREASED: Better for analytical tasks
            "temperature": 0.2,     # INCREASED: More creative analysis
            "top_p": 0.9,
            "stop": ["\n\n\n", "###", "---"]  # Better stop tokens
        }
        
        # FIXED: Prompt size limits for 4096 context window
        self.max_prompt_chars = 1500  # Keep prompts small
        self.max_content_chars = 800  # Truncate content aggressively
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'lineage_analyzer'
            result['api_based'] = True
            result['multi_call_approach'] = True
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
            
            CREATE TABLE IF NOT EXISTS file_access_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_name TEXT NOT NULL,
                logical_file_name TEXT NOT NULL,
                physical_file_name TEXT,
                access_type TEXT NOT NULL,
                access_mode TEXT,
                line_number INTEGER,
                access_statement TEXT,
                record_format TEXT,
                created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
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
    
    async def _call_api_focused(self, prompt: str, analysis_type: str, step_name: str) -> str:
        """FIXED: Make focused API call with proper parameters"""
        try:
            # Ensure prompt is within limits
            if len(prompt) > self.max_prompt_chars:
                self.logger.warning(f"Truncating prompt from {len(prompt)} to {self.max_prompt_chars} chars")
                prompt = prompt[:self.max_prompt_chars] + "...\n\nProvide analysis based on the above content."
            
            self.logger.info(f"🔄 API Call: {step_name} ({analysis_type}) - {len(prompt)} chars")
            
            # Use coordinator's API call method with better parameters
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=self.api_params,
                preferred_gpu_id=self.gpu_id
            )
            
            # Extract and validate text
            if isinstance(result, dict):
                text = result.get('text') or result.get('response') or result.get('content') or ''
                
                if text and len(text.strip()) > 10:
                    self.logger.info(f"✅ {step_name}: Got {len(text)} chars response")
                    return text.strip()
                else:
                    self.logger.warning(f"⚠️ {step_name}: Empty or short response: '{text}'")
                    return f"{step_name} analysis completed with limited results"
            
            return str(result) if result else f"{step_name} analysis completed"
            
        except Exception as e:
            self.logger.error(f"❌ API call failed for {step_name}: {str(e)}")
            return f"{step_name} analysis failed: {str(e)}"
    
    def _normalize_component_name(self, component_name) -> str:
        """Normalize component name to handle tuples and other formats"""
        try:
            if isinstance(component_name, str):
                return component_name.strip()
            
            if isinstance(component_name, (tuple, list)) and len(component_name) > 0:
                first_element = component_name[0]
                if isinstance(first_element, str):
                    self.logger.info(f"🔧 Normalized tuple/list component name: {component_name} -> {first_element}")
                    return first_element.strip()
            
            normalized = str(component_name).strip()
            if normalized != str(component_name):
                self.logger.info(f"🔧 Converted component name to string: {component_name} -> {normalized}")
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"❌ Failed to normalize component name {component_name}: {e}")
            return str(component_name) if component_name else "UNKNOWN"

    async def analyze_field_lineage(self, field_name: str) -> Dict[str, Any]:
        """FIXED: Field lineage analysis with multiple focused API calls"""
        try:
            await self._load_existing_lineage()
            
            field_name = str(field_name)
            self.logger.info(f"🎯 Starting multi-step field lineage analysis for: {field_name}")
            
            # STEP 1: Find all references (no API call needed)
            field_references = await self._find_field_references(field_name)
            self.logger.info(f"📋 Found {len(field_references)} references for {field_name}")
            
            if not field_references:
                return self._add_processing_info({
                    "field_name": field_name,
                    "error": "No references found for this field",
                    "suggestions": ["Check if field name is correct", "Verify data has been processed"],
                    "total_references": 0,
                    "status": "no_data"
                })
            
            # Initialize results
            analysis_result = {
                "field_name": field_name,
                "total_references": len(field_references),
                "analysis_steps": {},
                "status": "in_progress"
            }
            
            # STEP 2: Build basic lineage graph (no API call)
            self.logger.info(f"🔄 Step 2: Building lineage graph for {field_name}")
            field_lineage = await self._build_field_lineage_graph(field_name, field_references)
            analysis_result["analysis_steps"]["lineage_graph"] = {
                "status": "success",
                "data": field_lineage
            }
            
            # STEP 3: Analyze usage patterns with focused API call
            self.logger.info(f"🔄 Step 3: Analyzing usage patterns for {field_name}")
            usage_analysis = await self._analyze_field_usage_patterns_focused(field_name, field_references)
            analysis_result["analysis_steps"]["usage_patterns"] = {
                "status": "success" if usage_analysis.get("pattern_summary") else "partial",
                "data": usage_analysis
            }
            
            # STEP 4: Find transformations with focused API call
            self.logger.info(f"🔄 Step 4: Finding transformations for {field_name}")
            transformations = await self._find_field_transformations_focused(field_name, field_references)
            analysis_result["analysis_steps"]["transformations"] = {
                "status": "success" if transformations.get("transformation_summary") else "partial",
                "data": transformations
            }
            
            # STEP 5: Analyze lifecycle with focused API call
            self.logger.info(f"🔄 Step 5: Analyzing lifecycle for {field_name}")
            lifecycle = await self._analyze_field_lifecycle_focused(field_name, field_references)
            analysis_result["analysis_steps"]["lifecycle"] = {
                "status": "success" if lifecycle.get("lifecycle_summary") else "partial",
                "data": lifecycle
            }
            
            # STEP 6: Generate impact analysis with focused API call
            self.logger.info(f"🔄 Step 6: Generating impact analysis for {field_name}")
            impact_analysis = await self._analyze_field_impact_focused(field_name, field_references)
            analysis_result["analysis_steps"]["impact_analysis"] = {
                "status": "success" if impact_analysis.get("impact_summary") else "partial",
                "data": impact_analysis
            }
            
            # STEP 7: Create final summary with focused API call
            self.logger.info(f"🔄 Step 7: Creating comprehensive summary for {field_name}")
            final_summary = await self._generate_field_summary_focused(field_name, analysis_result)
            analysis_result["comprehensive_summary"] = final_summary
            
            # Final status
            successful_steps = len([step for step in analysis_result["analysis_steps"].values() if step["status"] == "success"])
            total_steps = len(analysis_result["analysis_steps"])
            
            analysis_result.update({
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "completion_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
                "status": "success" if successful_steps >= 3 else "partial",
                "analysis_timestamp": dt.now().isoformat()
            })
            
            self.logger.info(f"✅ Field lineage analysis completed: {successful_steps}/{total_steps} steps successful")
            return self._add_processing_info(analysis_result)
            
        except Exception as e:
            self.logger.error(f"❌ Field lineage analysis failed for {field_name}: {str(e)}")
            return self._add_processing_info({
                "field_name": field_name,
                "error": str(e),
                "status": "error",
                "analysis_timestamp": dt.now().isoformat()
            })

    async def _analyze_field_usage_patterns_focused(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """FIXED: Focused API call for usage pattern analysis"""
        
        # Prepare condensed data for analysis
        usage_stats = {
            "total_references": len(references),
            "programs_using": set(),
            "operation_types": defaultdict(int),
            "chunk_types": defaultdict(int)
        }
        
        # Extract key information
        for ref in references:
            if ref.get("type") != "table_definition":
                usage_stats["programs_using"].add(ref.get("program_name", "unknown"))
                usage_stats["chunk_types"][ref.get("chunk_type", "unknown")] += 1
        
        # Convert to serializable format
        programs_list = list(usage_stats["programs_using"])[:5]  # Limit to top 5
        chunk_types_dict = dict(list(usage_stats["chunk_types"].items())[:3])  # Limit to top 3
        
        # FOCUSED PROMPT: Only usage patterns
        prompt = f"""Analyze usage patterns for field: {field_name}

Field Usage Data:
- Total references: {usage_stats['total_references']}
- Programs using field: {len(programs_list)}
- Key programs: {', '.join(programs_list)}
- Chunk types: {chunk_types_dict}

Provide analysis covering:
1. Primary usage patterns
2. Field access frequency 
3. Common operations on this field
4. Business purpose indicators

Keep response under 300 words."""

        try:
            pattern_analysis = await self._call_api_focused(prompt, "usage_patterns", "Usage Pattern Analysis")
            
            return {
                "statistics": {
                    "total_references": usage_stats["total_references"],
                    "programs_using": programs_list,
                    "chunk_types": chunk_types_dict
                },
                "pattern_summary": pattern_analysis,
                "analysis_type": "focused_usage_patterns"
            }
            
        except Exception as e:
            self.logger.error(f"Usage pattern analysis failed: {e}")
            return {
                "statistics": {
                    "total_references": usage_stats["total_references"],
                    "programs_using": programs_list,
                    "chunk_types": chunk_types_dict
                },
                "error": str(e)
            }

    async def _find_field_transformations_focused(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """FIXED: Focused API call for transformation analysis"""
        
        # Extract transformation indicators
        transformation_indicators = []
        program_contexts = []
        
        for ref in references[:5]:  # Limit to first 5 references
            if ref.get("type") != "table_definition":
                content = ref.get("content", "")[:300]  # Limit content size
                program_name = ref.get("program_name", "unknown")
                
                # Check for transformation keywords
                transform_keywords = ["MOVE", "COMPUTE", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"]
                found_transforms = [kw for kw in transform_keywords if kw in content.upper()]
                
                if found_transforms:
                    transformation_indicators.extend(found_transforms)
                    program_contexts.append({
                        "program": program_name,
                        "operations": found_transforms,
                        "context": content[:100] + "..." if len(content) > 100 else content
                    })
        
        if not transformation_indicators:
            return {
                "transformation_count": 0,
                "transformation_summary": f"No clear transformations found for field {field_name}",
                "analysis_type": "focused_transformations"
            }
        
        # FOCUSED PROMPT: Only transformations
        prompt = f"""Analyze data transformations for field: {field_name}

Transformation Context:
- Operations found: {list(set(transformation_indicators))}
- Programs with transformations: {len(program_contexts)}

Sample transformation contexts:
{json.dumps(program_contexts[:3], indent=2)}

Analyze:
1. Types of transformations applied
2. Business logic in transformations
3. Data conversion patterns
4. Calculation methods

Keep response under 300 words."""

        try:
            transform_analysis = await self._call_api_focused(prompt, "transformations", "Transformation Analysis")
            
            return {
                "transformation_count": len(transformation_indicators),
                "transformation_types": list(set(transformation_indicators)),
                "programs_with_transforms": [ctx["program"] for ctx in program_contexts],
                "transformation_summary": transform_analysis,
                "analysis_type": "focused_transformations"
            }
            
        except Exception as e:
            self.logger.error(f"Transformation analysis failed: {e}")
            return {
                "transformation_count": len(transformation_indicators),
                "error": str(e)
            }

    async def _analyze_field_lifecycle_focused(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """FIXED: Focused API call for lifecycle analysis"""
        
        # Extract lifecycle stages
        lifecycle_stages = {
            "creation": [],
            "updates": [],
            "reads": [],
            "validations": []
        }
        
        for ref in references[:5]:  # Limit references
            if ref.get("type") == "table_definition":
                lifecycle_stages["creation"].append({
                    "type": "table_definition",
                    "source": ref.get("table_name", "unknown")
                })
            else:
                content = ref.get("content", "").upper()
                program = ref.get("program_name", "unknown")
                
                # Simple lifecycle detection
                if any(op in content for op in ["MOVE TO", "WRITE", "UPDATE"]):
                    lifecycle_stages["updates"].append({"program": program, "type": "update"})
                elif any(op in content for op in ["MOVE FROM", "READ", "SELECT"]):
                    lifecycle_stages["reads"].append({"program": program, "type": "read"})
                elif any(op in content for op in ["IF", "WHEN", "CHECK"]):
                    lifecycle_stages["validations"].append({"program": program, "type": "validation"})
        
        # Count stages
        stage_counts = {stage: len(operations) for stage, operations in lifecycle_stages.items()}
        total_operations = sum(stage_counts.values())
        
        if total_operations == 0:
            return {
                "lifecycle_score": 0.0,
                "lifecycle_summary": f"Limited lifecycle information found for field {field_name}",
                "analysis_type": "focused_lifecycle"
            }
        
        # FOCUSED PROMPT: Only lifecycle
        prompt = f"""Analyze lifecycle for field: {field_name}

Lifecycle Operations Found:
- Creation operations: {stage_counts['creation']}
- Update operations: {stage_counts['updates']}
- Read operations: {stage_counts['reads']}
- Validation operations: {stage_counts['validations']}
- Total operations: {total_operations}

Programs involved: {len(set([op.get('program', 'unknown') for stage_ops in lifecycle_stages.values() for op in stage_ops]))}

Analyze:
1. Completeness of lifecycle coverage
2. Missing lifecycle stages
3. Data governance implications
4. Field management patterns

Keep response under 300 words."""

        try:
            lifecycle_analysis = await self._call_api_focused(prompt, "lifecycle", "Lifecycle Analysis")
            
            # Calculate lifecycle completeness score
            lifecycle_score = min(1.0, total_operations / 10.0)  # Max score at 10 operations
            
            return {
                "lifecycle_stages": stage_counts,
                "total_operations": total_operations,
                "lifecycle_score": lifecycle_score,
                "lifecycle_summary": lifecycle_analysis,
                "analysis_type": "focused_lifecycle"
            }
            
        except Exception as e:
            self.logger.error(f"Lifecycle analysis failed: {e}")
            return {
                "lifecycle_stages": stage_counts,
                "total_operations": total_operations,
                "lifecycle_score": 0.0,
                "error": str(e)
            }

    async def _analyze_field_impact_focused(self, field_name: str, references: List[Dict]) -> Dict[str, Any]:
        """FIXED: Focused API call for impact analysis"""
        
        # Analyze impact scope
        impact_scope = {
            "affected_programs": set(),
            "affected_files": set(),
            "critical_operations": [],
            "complexity_indicators": []
        }
        
        for ref in references[:8]:  # Limit to 8 references for impact analysis
            if ref.get("type") != "table_definition":
                program = ref.get("program_name", "unknown")
                content = ref.get("content", "")
                
                impact_scope["affected_programs"].add(program)
                
                # Check for critical operations
                critical_ops = ["REWRITE", "DELETE", "UPDATE", "CALL", "PERFORM"]
                found_critical = [op for op in critical_ops if op in content.upper()]
                if found_critical:
                    impact_scope["critical_operations"].extend(found_critical)
                    impact_scope["complexity_indicators"].append(f"{program}: {', '.join(found_critical)}")
        
        # Calculate risk metrics
        num_programs = len(impact_scope["affected_programs"])
        num_critical_ops = len(impact_scope["critical_operations"])
        
        # Determine risk level
        if num_programs > 5 or num_critical_ops > 3:
            risk_level = "HIGH"
        elif num_programs > 2 or num_critical_ops > 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # FOCUSED PROMPT: Only impact analysis
        prompt = f"""Analyze change impact for field: {field_name}

Impact Scope:
- Affected programs: {num_programs}
- Critical operations: {num_critical_ops}
- Risk level: {risk_level}
- Programs: {', '.join(list(impact_scope['affected_programs'])[:5])}
- Critical ops found: {list(set(impact_scope['critical_operations']))}

Analyze:
1. Risk assessment for field changes
2. Testing requirements
3. Change management recommendations
4. Business continuity considerations

Keep response under 300 words."""

        try:
            impact_analysis = await self._call_api_focused(prompt, "impact", "Impact Analysis")
            
            return {
                "affected_programs": list(impact_scope["affected_programs"]),
                "affected_program_count": num_programs,
                "critical_operations": list(set(impact_scope["critical_operations"])),
                "risk_level": risk_level,
                "complexity_indicators": impact_scope["complexity_indicators"][:3],
                "impact_summary": impact_analysis,
                "analysis_type": "focused_impact"
            }
            
        except Exception as e:
            self.logger.error(f"Impact analysis failed: {e}")
            return {
                "affected_programs": list(impact_scope["affected_programs"]),
                "risk_level": risk_level,
                "error": str(e)
            }

    async def _generate_field_summary_focused(self, field_name: str, analysis_result: Dict[str, Any]) -> str:
        """FIXED: Focused API call for final summary"""
        
        # Extract key findings from all steps
        summary_data = {
            "field_name": field_name,
            "total_references": analysis_result.get("total_references", 0),
            "successful_steps": analysis_result.get("successful_steps", 0),
            "completion_rate": analysis_result.get("completion_rate", 0)
        }
        
        # Extract key findings
        key_findings = []
        
        # Usage patterns
        if "usage_patterns" in analysis_result.get("analysis_steps", {}):
            usage_data = analysis_result["analysis_steps"]["usage_patterns"].get("data", {})
            stats = usage_data.get("statistics", {})
            if stats:
                key_findings.append(f"Used in {len(stats.get('programs_using', []))} programs")
        
        # Transformations
        if "transformations" in analysis_result.get("analysis_steps", {}):
            transform_data = analysis_result["analysis_steps"]["transformations"].get("data", {})
            if transform_data.get("transformation_count", 0) > 0:
                key_findings.append(f"Has {transform_data['transformation_count']} transformations")
        
        # Impact
        if "impact_analysis" in analysis_result.get("analysis_steps", {}):
            impact_data = analysis_result["analysis_steps"]["impact_analysis"].get("data", {})
            risk_level = impact_data.get("risk_level", "UNKNOWN")
            key_findings.append(f"Change risk: {risk_level}")
        
        # FOCUSED PROMPT: Only final summary
        prompt = f"""Create executive summary for field: {field_name}

Analysis Results:
- Total references found: {summary_data['total_references']}
- Analysis completion: {summary_data['completion_rate']:.1f}%
- Key findings: {'; '.join(key_findings)}

Create concise summary covering:
1. Field business purpose
2. Key usage characteristics  
3. Important dependencies
4. Risk considerations
5. Recommendations

Keep response under 400 words and write in clear business language."""

        try:
            final_summary = await self._call_api_focused(prompt, "summary", "Executive Summary")
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return f"Field {field_name} analysis completed with {summary_data['completion_rate']:.1f}% completion rate. Found {summary_data['total_references']} references across multiple programs. Key findings: {'; '.join(key_findings)}."

    async def analyze_complete_data_flow(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """FIXED: Complete data flow analysis with multiple focused API calls"""
        try:
            # Normalize inputs
            original_component_name = component_name
            component_name = self._normalize_component_name(component_name)
            component_type = str(component_type)
            
            self.logger.info(f"🔄 Starting multi-step data flow analysis for {component_name} ({component_type})")
            
            # Determine if it's a program or file
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            is_program = await self._is_component_a_program(component_name, cursor)
            conn.close()
            
            # Initialize result
            analysis_result = {
                "component_name": component_name,
                "original_component_name": original_component_name,
                "component_type": component_type,
                "detected_type": "program" if is_program else "file",
                "analysis_steps": {},
                "status": "in_progress"
            }
            
            if is_program:
                self.logger.info(f"📄 Analyzing as PROGRAM: {component_name}")
                analysis_result.update(await self._analyze_program_data_flow_focused(component_name))
            else:
                self.logger.info(f"📁 Analyzing as FILE: {component_name}")
                analysis_result.update(await self._analyze_file_data_flow_focused(component_name))
            
            # Final status determination
            successful_steps = len([step for step in analysis_result.get("analysis_steps", {}).values() 
                                  if step.get("status") == "success"])
            total_steps = len(analysis_result.get("analysis_steps", {}))
            
            analysis_result.update({
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "completion_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
                "status": "success" if successful_steps >= 2 else "partial",
                "analysis_timestamp": dt.now().isoformat()
            })
            
            self.logger.info(f"✅ Data flow analysis completed: {successful_steps}/{total_steps} steps successful")
            return self._add_processing_info(analysis_result)
                
        except Exception as e:
            self.logger.error(f"❌ Complete data flow analysis failed: {str(e)}")
            return self._add_processing_info({
                "component_name": self._normalize_component_name(component_name),
                "original_component_name": component_name,
                "component_type": component_type,
                "error": str(e),
                "status": "error"
            })

    async def _analyze_program_data_flow_focused(self, program_name: str) -> Dict[str, Any]:
        """FIXED: Program data flow analysis with multiple focused API calls"""
        try:
            self.logger.info(f"📄 Multi-step program data flow analysis for: {program_name}")
            
            analysis_steps = {}
            
            # STEP 1: Get file access data (no API call needed)
            self.logger.info(f"🔄 Step 1: Getting file access patterns for {program_name}")
            file_access_data = await self._get_program_file_access(program_name)
            analysis_steps["file_access"] = {
                "status": "success",
                "data": file_access_data
            }
            
            # STEP 2: Get field usage data (no API call needed)
            self.logger.info(f"🔄 Step 2: Getting field usage patterns for {program_name}")
            field_usage_data = await self._get_program_field_usage(program_name)
            analysis_steps["field_usage"] = {
                "status": "success",
                "data": field_usage_data
            }
            
            # STEP 3: Analyze data transformations with focused API call
            self.logger.info(f"🔄 Step 3: Analyzing data transformations for {program_name}")
            transformation_analysis = await self._analyze_program_transformations_focused(program_name, file_access_data, field_usage_data)
            analysis_steps["transformations"] = {
                "status": "success" if transformation_analysis.get("transformation_summary") else "partial",
                "data": transformation_analysis
            }
            
            # STEP 4: Generate program flow analysis with focused API call
            self.logger.info(f"🔄 Step 4: Generating program flow analysis for {program_name}")
            flow_analysis = await self._generate_program_flow_analysis_focused(program_name, file_access_data, field_usage_data, transformation_analysis)
            analysis_steps["flow_analysis"] = {
                "status": "success" if flow_analysis else "partial",
                "data": {"flow_summary": flow_analysis}
            }
            
            return {
                "analysis_steps": analysis_steps,
                "analysis_type": "program_data_flow"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Program data flow analysis failed: {e}")
            return {
                "analysis_steps": {"error": {"status": "error", "data": {"error": str(e)}}},
                "analysis_type": "program_data_flow"
            }

    async def _analyze_file_data_flow_focused(self, file_name: str) -> Dict[str, Any]:
        """FIXED: File data flow analysis with multiple focused API calls"""
        try:
            self.logger.info(f"📁 Multi-step file data flow analysis for: {file_name}")
            
            analysis_steps = {}
            
            # STEP 1: Get file access relationships (no API call needed)
            self.logger.info(f"🔄 Step 1: Getting file access relationships for {file_name}")
            file_access_data = await self._get_file_access_data(file_name)
            analysis_steps["access_relationships"] = {
                "status": "success",
                "data": file_access_data
            }
            
            # STEP 2: Get field definitions (no API call needed)
            self.logger.info(f"🔄 Step 2: Getting field definitions for {file_name}")
            field_definitions = await self._get_file_field_definitions(file_name)
            analysis_steps["field_definitions"] = {
                "status": "success",
                "data": {"field_count": len(field_definitions), "fields": field_definitions[:10]}  # Limit fields
            }
            
            # STEP 3: Analyze field usage patterns with focused API call
            self.logger.info(f"🔄 Step 3: Analyzing field usage patterns for {file_name}")
            field_usage_analysis = await self._analyze_file_field_usage_focused(file_name, field_definitions, file_access_data)
            analysis_steps["field_usage"] = {
                "status": "success" if field_usage_analysis.get("usage_summary") else "partial",
                "data": field_usage_analysis
            }
            
            # STEP 4: Generate file flow analysis with focused API call
            self.logger.info(f"🔄 Step 4: Generating file flow analysis for {file_name}")
            flow_analysis = await self._generate_file_flow_analysis_focused(file_name, file_access_data, field_definitions)
            analysis_steps["flow_analysis"] = {
                "status": "success" if flow_analysis else "partial",
                "data": {"flow_summary": flow_analysis}
            }
            
            return {
                "analysis_steps": analysis_steps,
                "analysis_type": "file_data_flow"
            }
            
        except Exception as e:
            self.logger.error(f"❌ File data flow analysis failed: {e}")
            return {
                "analysis_steps": {"error": {"status": "error", "data": {"error": str(e)}}},
                "analysis_type": "file_data_flow"
            }

    async def _analyze_program_transformations_focused(self, program_name: str, file_access_data: Dict, field_usage_data: Dict) -> Dict[str, Any]:
        """FIXED: Focused API call for program transformation analysis"""
        
        # Prepare transformation summary data
        input_files = file_access_data.get("input_files", [])
        output_files = file_access_data.get("output_files", [])
        working_storage_fields = field_usage_data.get("working_storage_fields", [])
        
        transformation_indicators = {
            "input_count": len(input_files),
            "output_count": len(output_files),
            "working_storage_count": len(working_storage_fields),
            "field_operations": []
        }
        
        # Sample field operations for analysis
        for field in working_storage_fields[:5]:  # Limit to 5 fields
            field_name = field.get("field_name", "unknown")
            data_type = field.get("data_type", "unknown")
            transformation_indicators["field_operations"].append({
                "field": field_name,
                "type": data_type,
                "location": field.get("definition_location", "unknown")
            })
        
        # FOCUSED PROMPT: Only transformation analysis
        prompt = f"""Analyze data transformations in program: {program_name}

Program Characteristics:
- Input files: {transformation_indicators['input_count']}
- Output files: {transformation_indicators['output_count']}
- Working storage fields: {transformation_indicators['working_storage_count']}

Sample field operations:
{json.dumps(transformation_indicators['field_operations'], indent=2)}

Input files: {[f.get('logical_file_name', 'unknown') for f in input_files[:3]]}
Output files: {[f.get('logical_file_name', 'unknown') for f in output_files[:3]]}

Analyze:
1. Data transformation patterns
2. Business logic implementation
3. Input-to-output data flow
4. Calculation and derivation methods

Keep response under 300 words."""

        try:
            transformation_analysis = await self._call_api_focused(prompt, "transformations", "Program Transformation Analysis")
            
            return {
                "transformation_indicators": transformation_indicators,
                "transformation_summary": transformation_analysis,
                "analysis_type": "focused_program_transformations"
            }
            
        except Exception as e:
            self.logger.error(f"Program transformation analysis failed: {e}")
            return {
                "transformation_indicators": transformation_indicators,
                "error": str(e)
            }

    async def _generate_program_flow_analysis_focused(self, program_name: str, file_access_data: Dict, 
                                                    field_usage_data: Dict, transformation_analysis: Dict) -> str:
        """FIXED: Focused API call for program flow analysis"""
        
        # Prepare flow summary
        flow_summary = {
            "program_name": program_name,
            "input_files": len(file_access_data.get("input_files", [])),
            "output_files": len(file_access_data.get("output_files", [])),
            "working_storage_fields": len(field_usage_data.get("working_storage_fields", [])),
            "transformations": transformation_analysis.get("transformation_indicators", {}).get("input_count", 0)
        }
        
        # FOCUSED PROMPT: Only program flow
        prompt = f"""Generate program data flow analysis for: {program_name}

Program Flow Summary:
- Reads from {flow_summary['input_files']} input files
- Writes to {flow_summary['output_files']} output files  
- Uses {flow_summary['working_storage_fields']} working storage fields
- Performs data transformations

Key Input Files: {[f.get('logical_file_name', 'unknown') for f in file_access_data.get('input_files', [])[:3]]}
Key Output Files: {[f.get('logical_file_name', 'unknown') for f in file_access_data.get('output_files', [])[:3]]}

Provide analysis covering:
1. Business purpose and data processing role
2. Input-to-output data transformation pipeline  
3. Key business rules and logic
4. Data quality and validation points

Write as clear business documentation, keep under 400 words."""

        try:
            return await self._call_api_focused(prompt, "flow_analysis", "Program Flow Analysis")
        except Exception as e:
            self.logger.error(f"Program flow analysis failed: {e}")
            return f"Program {program_name} processes data from {flow_summary['input_files']} input files to {flow_summary['output_files']} output files with {flow_summary['working_storage_fields']} working storage fields."

    async def _analyze_file_field_usage_focused(self, file_name: str, field_definitions: List[Dict], file_access_data: Dict) -> Dict[str, Any]:
        """FIXED: Focused API call for file field usage analysis"""
        
        # Prepare field usage summary
        usage_summary = {
            "total_fields": len(field_definitions),
            "programs_accessing": len(file_access_data.get("programs_accessing", [])),
            "access_operations": file_access_data.get("file_operations", {}),
            "key_fields": []
        }
        
        # Extract key field information
        for field in field_definitions[:5]:  # Limit to 5 key fields
            field_info = {
                "name": field.get("field_name", "unknown"),
                "type": field.get("data_type", "unknown"),
                "picture": field.get("picture_clause", ""),
                "level": field.get("level_number", 0)
            }
            usage_summary["key_fields"].append(field_info)
        
        # FOCUSED PROMPT: Only field usage
        prompt = f"""Analyze field usage patterns for file: {file_name}

Field Usage Summary:
- Total fields: {usage_summary['total_fields']}
- Programs accessing file: {usage_summary['programs_accessing']}
- Access operations: {usage_summary['access_operations']}

Key fields:
{json.dumps(usage_summary['key_fields'], indent=2)}

Programs accessing: {file_access_data.get('programs_accessing', [])[:5]}

Analyze:
1. Field usage frequency and patterns
2. Key business fields identification
3. Data access characteristics  
4. Field categorization (input, derived, static)

Keep response under 300 words."""

        try:
            usage_analysis = await self._call_api_focused(prompt, "field_usage", "File Field Usage Analysis")
            
            return {
                "usage_statistics": usage_summary,
                "usage_summary": usage_analysis,
                "analysis_type": "focused_file_field_usage"
            }
            
        except Exception as e:
            self.logger.error(f"File field usage analysis failed: {e}")
            return {
                "usage_statistics": usage_summary,
                "error": str(e)
            }

    async def _generate_file_flow_analysis_focused(self, file_name: str, file_access_data: Dict, field_definitions: List[Dict]) -> str:
        """FIXED: Focused API call for file flow analysis"""
        
        # Prepare file flow summary
        flow_summary = {
            "file_name": file_name,
            "programs_accessing": len(file_access_data.get("programs_accessing", [])),
            "total_fields": len(field_definitions),
            "create_programs": len(file_access_data.get("access_patterns", {}).get("creators", [])),
            "read_programs": len(file_access_data.get("access_patterns", {}).get("readers", [])),
            "update_programs": len(file_access_data.get("access_patterns", {}).get("updaters", []))
        }
        
        # FOCUSED PROMPT: Only file flow
        prompt = f"""Generate file data flow analysis for: {file_name}

File Flow Summary:
- Accessed by {flow_summary['programs_accessing']} programs
- Contains {flow_summary['total_fields']} fields
- Created by {flow_summary['create_programs']} programs
- Read by {flow_summary['read_programs']} programs  
- Updated by {flow_summary['update_programs']} programs

Key programs:
- Creators: {[p.get('program_name', 'unknown') for p in file_access_data.get('access_patterns', {}).get('creators', [])[:3]]}
- Readers: {[p.get('program_name', 'unknown') for p in file_access_data.get('access_patterns', {}).get('readers', [])[:3]]}

Provide analysis covering:
1. File business purpose and role
2. Data lifecycle and flow patterns
3. Producer-consumer relationships
4. Critical dependencies and usage

Write as clear business documentation, keep under 400 words."""

        try:
            return await self._call_api_focused(prompt, "file_flow", "File Flow Analysis")
        except Exception as e:
            self.logger.error(f"File flow analysis failed: {e}")
            return f"File {file_name} is accessed by {flow_summary['programs_accessing']} programs with {flow_summary['total_fields']} fields, involving {flow_summary['create_programs']} creators and {flow_summary['read_programs']} readers."

    # ==================== Keep existing helper methods unchanged ====================
    
    async def _find_field_references(self, field_name: str) -> List[Dict[str, Any]]:
        """Find all references to a field across the codebase"""
        references = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
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
                    LIMIT 50
                """, (pattern, pattern))
                
                results = cursor.fetchall()
                
                for program_name, chunk_id, chunk_type, content, metadata_str in results:
                    if any(ref.get('chunk_id') == chunk_id for ref in references):
                        continue
                        
                    metadata = self.safe_json_loads(metadata_str)
                    
                    if self._is_field_referenced(field_name, content, metadata):
                        references.append({
                            "program_name": program_name,
                            "chunk_id": chunk_id,
                            "chunk_type": chunk_type,
                            "content": content[:self.max_content_chars],  # Truncate content
                            "metadata": metadata
                        })
                        
                        if len(references) >= 20:  # Limit total references
                            break
                
                if len(references) >= 20:
                    break
            
            # Check for table definitions
            try:
                cursor.execute("""
                    SELECT table_name, fields
                    FROM file_metadata
                    WHERE fields LIKE ?
                    LIMIT 5
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
        for ref in references[:10]:  # Limit to 10 references for graph
            if ref.get("type") == "table_definition":
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
                
                lineage_edges.append({
                    "source": table_node["id"],
                    "target": field_node["id"],
                    "relationship": "defines",
                    "properties": {"field_type": ref["field_type"]}
                })
                
            else:
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

    async def _get_file_access_data(self, component_name: str) -> Dict[str, Any]:
        """Get file access data with component name normalization"""
        try:
            component_name = self._normalize_component_name(component_name)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            is_program = await self._is_component_a_program(component_name, cursor)
            
            if is_program:
                cursor.execute("""
                    SELECT program_name, logical_file_name, physical_file_name, access_type, access_mode,
                        record_format, line_number
                    FROM file_access_relationships
                    WHERE program_name = ? OR program_name LIKE ?
                    ORDER BY line_number
                """, (str(component_name), f"%{component_name}%"))
            else:
                cursor.execute("""
                    SELECT program_name, logical_file_name, physical_file_name, access_type, access_mode,
                        record_format, line_number
                    FROM file_access_relationships
                    WHERE logical_file_name = ? OR physical_file_name = ? 
                    OR logical_file_name LIKE ? OR physical_file_name LIKE ?
                    ORDER BY program_name, line_number
                """, (str(component_name), str(component_name), f"%{component_name}%", f"%{component_name}%"))
            
            access_records = cursor.fetchall()
            conn.close()
            
            if not access_records:
                return {
                    "access_patterns": {"creators": [], "readers": [], "updaters": [], "deleters": []},
                    "programs_accessing": [],
                    "files_accessed": [],
                    "total_access_points": 0,
                    "file_operations": {"create_operations": 0, "read_operations": 0, "update_operations": 0, "delete_operations": 0},
                    "component_type": "program" if is_program else "file"
                }
            
            # Organize access patterns
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
                    "logical_file_name": record[1],
                    "physical_file_name": record[2],
                    "access_type": record[3],
                    "access_mode": record[4],
                    "record_format": record[5],
                    "line_number": record[6] if len(record) > 6 else None
                }
                
                programs_accessing.add(record[0])
                if record[1]:
                    files_accessed.add(record[1])
                if record[2]:
                    files_accessed.add(record[2])
                
                access_type = record[3].upper() if record[3] else ""
                access_mode = record[4].upper() if record[4] else ""
                
                if access_type in ["WRITE", "FD"] and access_mode in ["OUTPUT", "EXTEND"]:
                    access_patterns["creators"].append(access_info)
                elif access_type in ["READ", "SELECT", "FILE_SELECT"] and access_mode == "INPUT":
                    access_patterns["readers"].append(access_info)
                elif access_type in ["REWRITE", "WRITE"] and access_mode == "I-O":
                    access_patterns["updaters"].append(access_info)
                elif access_type == "DELETE":
                    access_patterns["deleters"].append(access_info)
                else:
                    if access_mode == "INPUT":
                        access_patterns["readers"].append(access_info)
                    elif access_mode in ["OUTPUT", "EXTEND"]:
                        access_patterns["creators"].append(access_info)
                    elif access_mode == "I-O":
                        access_patterns["updaters"].append(access_info)
                    else:
                        access_patterns["readers"].append(access_info)
            
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
        """Determine if component is a program or file"""
        try:
            component_name = self._normalize_component_name(component_name)
            
            search_patterns = [
                (component_name, "exact match"),
                (f"{component_name}%", "starts with"),
                (f"%{component_name}", "ends with"),
                (f"%{component_name}%", "contains")
            ]
            
            for pattern, description in search_patterns:
                if pattern == component_name:
                    cursor.execute("""
                        SELECT COUNT(*), program_name FROM program_chunks 
                        WHERE program_name = ?
                        GROUP BY program_name
                        LIMIT 1
                    """, (pattern,))
                else:
                    cursor.execute("""
                        SELECT COUNT(*), program_name FROM program_chunks 
                        WHERE program_name LIKE ?
                        GROUP BY program_name
                        LIMIT 1
                    """, (pattern,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    return True
            
            # Check file_access_relationships as program_name
            for pattern, description in search_patterns:
                if pattern == component_name:
                    cursor.execute("""
                        SELECT COUNT(*), program_name FROM file_access_relationships 
                        WHERE program_name = ?
                        GROUP BY program_name
                        LIMIT 1
                    """, (pattern,))
                else:
                    cursor.execute("""
                        SELECT COUNT(*), program_name FROM file_access_relationships 
                        WHERE program_name LIKE ?
                        GROUP BY program_name
                        LIMIT 1
                    """, (pattern,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to determine component type for {component_name}: {e}")
            return False

    async def _get_file_field_definitions(self, file_name: str) -> List[Dict[str, Any]]:
        """Get field definitions associated with a file"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='field_cross_reference'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return []
            
            cursor.execute("""
                SELECT field_name, qualified_name, source_type, source_name,
                    definition_location, data_type, picture_clause, usage_clause,
                    level_number, parent_field, occurs_info, business_domain
                FROM field_cross_reference
                WHERE source_name = ? OR source_name IN (
                    SELECT DISTINCT program_name FROM file_access_relationships 
                    WHERE logical_file_name = ? OR physical_file_name = ?
                )
                AND definition_location IN ('FD', 'FILE_SECTION', 'WORKING_STORAGE')
                ORDER BY source_name, level_number, field_name
                LIMIT 50
            """, (str(file_name), str(file_name), str(file_name)))
            
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

    async def _get_program_file_access(self, program_name: str) -> Dict[str, Any]:
        """Get file access patterns for a specific program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT logical_file_name, physical_file_name, access_type, access_mode,
                    record_format, line_number
                FROM file_access_relationships
                WHERE program_name = ? OR program_name LIKE ?
                ORDER BY line_number
                LIMIT 50
            """, (str(program_name), f"%{program_name}%"))
            
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
                    "logical_file_name": record[0],
                    "physical_file_name": record[1],
                    "access_type": record[2],
                    "access_mode": record[3],
                    "record_format": record[4],
                    "line_number": record[5] if len(record) > 5 else None
                }
                
                access_mode = record[3] if record[3] else ""
                if access_mode == "INPUT":
                    file_access["input_files"].append(file_info)
                elif access_mode in ["OUTPUT", "EXTEND"]:
                    file_access["output_files"].append(file_info)
                elif access_mode == "I-O":
                    file_access["update_files"].append(file_info)
                else:
                    file_access["temporary_files"].append(file_info)
            
            return file_access
            
        except Exception as e:
            self.logger.error(f"Failed to get program file access: {e}")
            return {"input_files": [], "output_files": [], "update_files": [], "temporary_files": []}
        
    async def _get_program_field_usage(self, program_name: str) -> Dict[str, Any]:
        """Get field usage patterns within a program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT field_name, qualified_name, definition_location, data_type,
                    picture_clause, level_number, parent_field, business_domain
                FROM field_cross_reference
                WHERE source_name = ?
                ORDER BY level_number, field_name
                LIMIT 100
            """, (str(program_name),))
            
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

    # ==================== Additional API methods for cross-program analysis ====================

    async def analyze_cross_program_data_lineage(self, component_name: str) -> Dict[str, Any]:
        """FIXED: Cross-program lineage analysis with focused API calls"""
        try:
            component_name = self._normalize_component_name(component_name)
            
            self.logger.info(f"🔄 Starting cross-program lineage analysis for: {component_name}")
            
            analysis_steps = {}
            
            # STEP 1: Get impact analysis data (no API call needed)
            self.logger.info(f"🔄 Step 1: Getting cross-program impact data for {component_name}")
            impact_data = await self._get_cross_program_impact_data(component_name)
            analysis_steps["impact_data"] = {
                "status": "success",
                "data": impact_data
            }
            
            # STEP 2: Build lineage graph (no API call needed)
            self.logger.info(f"🔄 Step 2: Building cross-program lineage graph for {component_name}")
            lineage_graph = await self._build_cross_program_lineage_graph(component_name, impact_data)
            analysis_steps["lineage_graph"] = {
                "status": "success",
                "data": lineage_graph
            }
            
            # STEP 3: Generate cross-program analysis with focused API call
            self.logger.info(f"🔄 Step 3: Generating cross-program analysis for {component_name}")
            cross_program_analysis = await self._generate_cross_program_analysis_focused(component_name, impact_data, lineage_graph)
            analysis_steps["cross_program_analysis"] = {
                "status": "success" if cross_program_analysis else "partial",
                "data": {"analysis_summary": cross_program_analysis}
            }
            
            # Calculate success metrics
            successful_steps = len([step for step in analysis_steps.values() if step["status"] == "success"])
            total_steps = len(analysis_steps)
            
            result = {
                "component_name": component_name,
                "analysis_steps": analysis_steps,
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "completion_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
                "analysis_type": "cross_program_data_lineage",
                "analysis_timestamp": dt.now().isoformat(),
                "status": "success" if successful_steps >= 2 else "partial"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"❌ Cross-program lineage analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    async def _get_cross_program_impact_data(self, component_name: str) -> Dict[str, Any]:
        """Get impact analysis data across programs"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if impact_analysis table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='impact_analysis'
            """)
            
            if not cursor.fetchone():
                # Fallback: use program_chunks and file_access_relationships
                upstream_dependencies = []
                downstream_impacts = []
                
                # Find programs that this component depends on
                cursor.execute("""
                    SELECT DISTINCT program_name FROM file_access_relationships
                    WHERE logical_file_name = ? OR physical_file_name = ?
                    LIMIT 10
                """, (str(component_name), str(component_name)))
                
                for (program_name,) in cursor.fetchall():
                    upstream_dependencies.append({
                        "source_artifact": program_name,
                        "source_type": "program",
                        "dependent_artifact": component_name,
                        "dependent_type": "file",
                        "relationship_type": "accesses",
                        "impact_level": "medium",
                        "change_propagation": "dependent_updates_needed"
                    })
                
                conn.close()
                return {
                    "upstream_dependencies": upstream_dependencies,
                    "downstream_impacts": downstream_impacts,
                    "total_relationships": len(upstream_dependencies) + len(downstream_impacts)
                }
            
            # Use impact_analysis table if it exists
            cursor.execute("""
                SELECT source_artifact, source_type, dependent_artifact, dependent_type,
                    relationship_type, impact_level, change_propagation
                FROM impact_analysis
                WHERE source_artifact = ? OR dependent_artifact = ?
                ORDER BY impact_level DESC, relationship_type
                LIMIT 20
            """, (str(component_name), str(component_name)))
            
            impact_records = cursor.fetchall()
            conn.close()
            
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
            
            # Add upstream nodes and edges (limit to 5)
            for i, upstream in enumerate(impact_data.get("upstream_dependencies", [])[:5]):
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
            
            # Add downstream nodes and edges (limit to 5)
            for i, downstream in enumerate(impact_data.get("downstream_impacts", [])[:5]):
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

    async def _generate_cross_program_analysis_focused(self, component_name: str, impact_data: Dict, lineage_graph: Dict) -> str:
        """FIXED: Focused API call for cross-program lineage analysis"""
        
        # Prepare analysis summary
        lineage_summary = {
            "component_name": component_name,
            "upstream_count": len(impact_data.get("upstream_dependencies", [])),
            "downstream_count": len(impact_data.get("downstream_impacts", [])),
            "total_nodes": lineage_graph.get("total_nodes", 0),
            "graph_complexity": lineage_graph.get("graph_complexity", "low")
        }
        
        # Extract key dependencies
        upstream_components = [dep["source_artifact"] for dep in impact_data.get("upstream_dependencies", [])[:5]]
        downstream_components = [imp["dependent_artifact"] for imp in impact_data.get("downstream_impacts", [])[:5]]
        
        # FOCUSED PROMPT: Only cross-program analysis
        prompt = f"""Analyze cross-program lineage for: {component_name}

Lineage Summary:
- {lineage_summary['upstream_count']} upstream dependencies
- {lineage_summary['downstream_count']} downstream impacts  
- {lineage_summary['total_nodes']} total components in lineage
- Graph complexity: {lineage_summary['graph_complexity']}

Key Dependencies:
- Upstream: {upstream_components}
- Downstream: {downstream_components}

Provide analysis covering:
1. Component's role in data ecosystem
2. Critical dependencies and relationships
3. Cross-program impact assessment
4. Change management considerations

Write clear business analysis, keep under 400 words."""

        try:
            return await self._call_api_focused(prompt, "cross_program", "Cross-Program Lineage Analysis")
        except Exception as e:
            self.logger.error(f"Cross-program analysis failed: {e}")
            return f"Component {component_name} has {lineage_summary['upstream_count']} upstream dependencies and {lineage_summary['downstream_count']} downstream impacts across {lineage_summary['total_nodes']} total components with {lineage_summary['graph_complexity']} complexity."

    # ==================== Lifecycle Analysis Methods ====================

    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """FIXED: Complete lifecycle analysis with focused API calls"""
        try:
            await self._load_existing_lineage()
            
            component_name = str(component_name)
            component_type = str(component_type)
            
            self.logger.info(f"🔄 Starting lifecycle analysis for {component_name} ({component_type})")
            
            analysis_steps = {}
            
            if component_type in ["file", "table"]:
                # STEP 1: Analyze data component lifecycle
                self.logger.info(f"🔄 Step 1: Analyzing data component lifecycle for {component_name}")
                lifecycle_result = await self._analyze_data_component_lifecycle_focused(component_name)
                analysis_steps["lifecycle_analysis"] = {
                    "status": "success" if lifecycle_result.get("lifecycle_summary") else "partial",
                    "data": lifecycle_result
                }
                
            elif component_type in ["program", "cobol"]:
                # STEP 1: Analyze program lifecycle
                self.logger.info(f"🔄 Step 1: Analyzing program lifecycle for {component_name}")
                lifecycle_result = await self._analyze_program_lifecycle_focused(component_name)
                analysis_steps["lifecycle_analysis"] = {
                    "status": "success" if lifecycle_result.get("lifecycle_summary") else "partial",
                    "data": lifecycle_result
                }
                
            elif component_type == "jcl":
                # STEP 1: Analyze JCL lifecycle
                self.logger.info(f"🔄 Step 1: Analyzing JCL lifecycle for {component_name}")
                lifecycle_result = await self._analyze_jcl_lifecycle_focused(component_name)
                analysis_steps["lifecycle_analysis"] = {
                    "status": "success" if lifecycle_result.get("lifecycle_summary") else "partial",
                    "data": lifecycle_result
                }
            else:
                return self._add_processing_info({
                    "component_name": component_name,
                    "component_type": component_type,
                    "error": f"Unsupported component type: {component_type}",
                    "status": "error"
                })
            
            # Calculate success metrics
            successful_steps = len([step for step in analysis_steps.values() if step["status"] == "success"])
            total_steps = len(analysis_steps)
            
            result = {
                "component_name": component_name,
                "component_type": component_type,
                "analysis_steps": analysis_steps,
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "completion_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
                "analysis_type": "full_lifecycle",
                "analysis_timestamp": dt.now().isoformat(),
                "status": "success" if successful_steps > 0 else "error"
            }
            
            return self._add_processing_info(result)
                
        except Exception as e:
            return self._add_processing_info({
                "component_name": component_name,
                "component_type": component_type,
                "error": str(e),
                "status": "error"
            })

    async def _analyze_data_component_lifecycle_focused(self, component_name: str) -> Dict[str, Any]:
        """FIXED: Focused API call for data component lifecycle analysis"""
        
        # Get component references
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT program_name, chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE content LIKE ? OR metadata LIKE ?
            LIMIT 10
        """, (f"%{component_name}%", f"%{component_name}%"))
        
        references = cursor.fetchall()
        conn.close()
        
        # Analyze lifecycle stages
        lifecycle_stages = {
            "creation_points": [],
            "read_operations": [],
            "update_operations": [],
            "delete_operations": []
        }
        
        programs_involved = set()
        
        for program_name, chunk_id, chunk_type, content, metadata_str in references:
            programs_involved.add(program_name)
            
            # Simple operation detection
            content_upper = content.upper()
            if any(op in content_upper for op in ["CREATE", "WRITE", "OUTPUT"]):
                lifecycle_stages["creation_points"].append({"program": program_name, "chunk": chunk_id})
            elif any(op in content_upper for op in ["READ", "SELECT", "INPUT"]):
                lifecycle_stages["read_operations"].append({"program": program_name, "chunk": chunk_id})
            elif any(op in content_upper for op in ["UPDATE", "REWRITE", "MODIFY"]):
                lifecycle_stages["update_operations"].append({"program": program_name, "chunk": chunk_id})
            elif any(op in content_upper for op in ["DELETE", "REMOVE", "PURGE"]):
                lifecycle_stages["delete_operations"].append({"program": program_name, "chunk": chunk_id})
        
        # Prepare summary data
        lifecycle_summary_data = {
            "component_name": component_name,
            "programs_involved": len(programs_involved),
            "creation_points": len(lifecycle_stages["creation_points"]),
            "read_operations": len(lifecycle_stages["read_operations"]),
            "update_operations": len(lifecycle_stages["update_operations"]),
            "delete_operations": len(lifecycle_stages["delete_operations"])
        }
        
        # FOCUSED PROMPT: Only lifecycle analysis
        prompt = f"""Analyze lifecycle for data component: {component_name}

Lifecycle Summary:
- Programs involved: {lifecycle_summary_data['programs_involved']}
- Creation points: {lifecycle_summary_data['creation_points']}
- Read operations: {lifecycle_summary_data['read_operations']}
- Update operations: {lifecycle_summary_data['update_operations']}
- Delete operations: {lifecycle_summary_data['delete_operations']}

Programs: {list(programs_involved)[:5]}

Analyze:
1. Component lifecycle completeness
2. Data management patterns
3. Operational usage characteristics
4. Lifecycle governance recommendations

Keep response under 300 words."""

        try:
            lifecycle_analysis = await self._call_api_focused(prompt, "lifecycle", "Data Component Lifecycle Analysis")
            
            return {
                "lifecycle_stages": lifecycle_stages,
                "lifecycle_summary_data": lifecycle_summary_data,
                "lifecycle_summary": lifecycle_analysis,
                "analysis_type": "focused_data_component_lifecycle"
            }
            
        except Exception as e:
            self.logger.error(f"Data component lifecycle analysis failed: {e}")
            return {
                "lifecycle_stages": lifecycle_stages,
                "lifecycle_summary_data": lifecycle_summary_data,
                "error": str(e)
            }

    async def _analyze_program_lifecycle_focused(self, program_name: str) -> Dict[str, Any]:
        """FIXED: Focused API call for program lifecycle analysis"""
        
        # Get program chunks
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE program_name = ?
            LIMIT 20
        """, (str(program_name),))
        
        chunks = cursor.fetchall()
        conn.close()
        
        # Analyze program structure
        program_analysis = {
            "total_chunks": len(chunks),
            "chunk_breakdown": {},
            "complexity_indicators": []
        }
        
        for chunk_id, chunk_type, content, metadata_str in chunks:
            if chunk_type in program_analysis["chunk_breakdown"]:
                program_analysis["chunk_breakdown"][chunk_type] += 1
            else:
                program_analysis["chunk_breakdown"][chunk_type] = 1
            
            # Check for complexity indicators
            content_upper = content.upper()
            if any(indicator in content_upper for indicator in ["CALL", "PERFORM", "GO TO", "IF", "EVALUATE"]):
                program_analysis["complexity_indicators"].append(chunk_type)
        
        # FOCUSED PROMPT: Only program lifecycle
        prompt = f"""Analyze lifecycle for program: {program_name}

Program Structure:
- Total chunks: {program_analysis['total_chunks']}
- Chunk types: {program_analysis['chunk_breakdown']}
- Complexity indicators: {len(program_analysis['complexity_indicators'])}

Analyze:
1. Program structure and organization
2. Complexity and maintainability factors
3. Integration and dependency patterns
4. Lifecycle management recommendations

Keep response under 300 words."""

        try:
            program_lifecycle_analysis = await self._call_api_focused(prompt, "program_lifecycle", "Program Lifecycle Analysis")
            
            return {
                "program_structure": program_analysis,
                "lifecycle_summary": program_lifecycle_analysis,
                "analysis_type": "focused_program_lifecycle"
            }
            
        except Exception as e:
            self.logger.error(f"Program lifecycle analysis failed: {e}")
            return {
                "program_structure": program_analysis,
                "error": str(e)
            }

    async def _analyze_jcl_lifecycle_focused(self, jcl_name: str) -> Dict[str, Any]:
        """FIXED: Focused API call for JCL lifecycle analysis"""
        
        # Get JCL steps
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_id, chunk_type, content, metadata
            FROM program_chunks
            WHERE program_name = ? AND chunk_type IN ('job_step', 'job_header')
            ORDER BY chunk_id
            LIMIT 10
        """, (str(jcl_name),))
        
        steps = cursor.fetchall()
        conn.close()
        
        # Analyze JCL structure
        jcl_analysis = {
            "total_steps": len([s for s in steps if s[1] == 'job_step']),
            "step_types": {},
            "job_characteristics": []
        }
        
        for chunk_id, chunk_type, content, metadata_str in steps:
            if chunk_type in jcl_analysis["step_types"]:
                jcl_analysis["step_types"][chunk_type] += 1
            else:
                jcl_analysis["step_types"][chunk_type] = 1
            
            # Check for job characteristics
            content_upper = content.upper()
            if "EXEC" in content_upper:
                jcl_analysis["job_characteristics"].append("executable_step")
            if "DD" in content_upper:
                jcl_analysis["job_characteristics"].append("data_definition")
        
        # FOCUSED PROMPT: Only JCL lifecycle
        prompt = f"""Analyze lifecycle for JCL job: {jcl_name}

JCL Structure:
- Total steps: {jcl_analysis['total_steps']}
- Step types: {jcl_analysis['step_types']}
- Job characteristics: {len(set(jcl_analysis['job_characteristics']))}

Analyze:
1. Job structure and execution flow
2. Resource requirements and dependencies
3. Scheduling and operational characteristics
4. Job lifecycle management recommendations

Keep response under 300 words."""

        try:
            jcl_lifecycle_analysis = await self._call_api_focused(prompt, "jcl_lifecycle", "JCL Lifecycle Analysis")
            
            return {
                "jcl_structure": jcl_analysis,
                "lifecycle_summary": jcl_lifecycle_analysis,
                "analysis_type": "focused_jcl_lifecycle"
            }
            
        except Exception as e:
            self.logger.error(f"JCL lifecycle analysis failed: {e}")
            return {
                "jcl_structure": jcl_analysis,
                "error": str(e)
            }

    # ==================== Summary and Reporting Methods ====================

    async def generate_lineage_summary(self, component_name: str) -> Dict[str, Any]:
        """FIXED: Generate comprehensive lineage summary with focused API calls"""
        try:
            await self._load_existing_lineage()
            component_name = self._normalize_component_name(component_name)
            
            # Determine component type
            component_type = await self._determine_component_type(component_name)
            
            # Get appropriate analysis based on type
            if component_type == "field":
                analysis = await self.analyze_field_lineage(component_name)
            else:
                analysis = await self.analyze_complete_data_flow(component_name, component_type)
            
            # Generate executive summary with focused API call
            executive_summary = await self._generate_executive_summary_focused(component_name, component_type, analysis)
            
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
        """Determine component type with database checks"""
        
        # Check file extension first
        component_lower = component_name.lower()
        
        if any(component_lower.endswith(ext) for ext in ['.cbl', '.cob', '.cobol']):
            return "cobol"
        elif any(component_lower.endswith(ext) for ext in ['.copy', '.cpy', '.copybook']):
            return "copybook"
        elif any(component_lower.endswith(ext) for ext in ['.jcl', '.job', '.proc']):
            return "jcl"
        
        # Database-based detection
        try:
            def check_database():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                try:
                    # Check program_chunks
                    cursor.execute("""
                        SELECT chunk_type, COUNT(*) as count 
                        FROM program_chunks 
                        WHERE program_name = ? OR program_name LIKE ?
                        GROUP BY chunk_type
                        ORDER BY count DESC
                        LIMIT 5
                    """, (component_name, f"%{component_name}%"))
                    
                    chunk_types = cursor.fetchall()
                    
                    if chunk_types:
                        chunk_type_names = [ct.lower() for ct, _ in chunk_types]
                        
                        if any('job' in ct for ct in chunk_type_names):
                            return "jcl"
                        elif any(ct in ['working_storage', 'procedure_division', 'data_division'] for ct in chunk_type_names):
                            return "cobol"
                        elif any(ct in ['copybook', 'copy'] for ct in chunk_type_names):
                            return "copybook"
                    
                    # Check if it might be a field
                    cursor.execute("""
                        SELECT COUNT(*) FROM program_chunks
                        WHERE (content LIKE ? OR metadata LIKE ?) 
                        LIMIT 1
                    """, (f"%{component_name}%", f"%{component_name}%"))
                    
                    field_count = cursor.fetchone()[0]
                    
                    # Field detection heuristics
                    if (field_count > 0 and component_name.isupper() and 
                        ('_' in component_name or '-' in component_name or len(component_name) <= 30)):
                        return "field"
                    
                    return "cobol"  # Default
                    
                finally:
                    conn.close()
            
            # Run database check
            result = await asyncio.get_event_loop().run_in_executor(None, check_database)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Component type determination failed: {e}")
            return "cobol"

    async def _generate_executive_summary_focused(self, component_name: str, component_type: str, analysis: Dict) -> str:
        """FIXED: Focused API call for executive summary"""
        
        # Extract key metrics from analysis
        summary_data = {
            "component_name": component_name,
            "component_type": component_type,
            "analysis_status": analysis.get("status", "unknown"),
            "completion_rate": analysis.get("completion_rate", 0)
        }
        
        # Extract key findings
        key_findings = []
        
        if "total_references" in analysis:
            key_findings.append(f"Found {analysis['total_references']} references")
        
        if "analysis_steps" in analysis:
            successful_steps = analysis.get("successful_steps", 0)
            total_steps = analysis.get("total_steps", 0)
            key_findings.append(f"Completed {successful_steps}/{total_steps} analysis steps")
        
        # FOCUSED PROMPT: Only executive summary
        prompt = f"""Generate executive summary for: {component_name}

Component Analysis:
- Type: {component_type}
- Status: {summary_data['analysis_status']}
- Completion: {summary_data['completion_rate']:.1f}%
- Key findings: {'; '.join(key_findings)}

Create executive summary covering:
1. Component purpose and business importance
2. Key usage patterns and dependencies
3. Risk assessment and impact
4. Strategic recommendations

Write clear business language, keep under 300 words."""

        try:
            return await self._call_api_focused(prompt, "executive_summary", "Executive Summary Generation")
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            return f"Executive Summary for {component_name}: {component_type} component analyzed with {summary_data['completion_rate']:.1f}% completion rate. {'; '.join(key_findings)}."

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
                f"multiple_api_calls=True, "
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
        
        # Analyze field lineage with multiple focused API calls
        field_analysis = await lineage_agent.analyze_field_lineage("CUSTOMER-ID")
        print(f"Field analysis status: {field_analysis['status']}")
        print(f"Completion rate: {field_analysis.get('completion_rate', 0):.1f}%")
        
        # Analyze component data flow with multiple focused API calls
        data_flow_analysis = await lineage_agent.analyze_complete_data_flow("CUSTOMER-PROGRAM", "program")
        print(f"Data flow analysis status: {data_flow_analysis['status']}")
        print(f"Completion rate: {data_flow_analysis.get('completion_rate', 0):.1f}%")
        
        # Generate comprehensive summary
        summary = await lineage_agent.generate_lineage_summary("CUSTOMER-RECORD")
        print(f"Summary generated: {summary['status']}")
        
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_fixed_lineage_usage())