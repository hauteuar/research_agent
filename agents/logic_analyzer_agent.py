# agents/logic_analyzer_agent.py
"""
Agent 5: Logic Analyzer
Analyzes program logic with streaming support for complex COBOL/JCL operations
"""

import asyncio
import sqlite3
import json
import re
import uuid
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib

import torch
from vllm import AsyncLLMEngine, SamplingParams
from agents.base_agent import BaseOpulenceAgent

@dataclass
class LogicPattern:
    """Represents a detected logic pattern"""
    pattern_type: str
    pattern_name: str
    complexity_score: float
    description: str
    code_snippet: str
    recommendations: List[str]

@dataclass
class BusinessRule:
    """Represents a business rule extracted from code"""
    rule_id: str
    rule_type: str
    condition: str
    action: str
    fields_involved: List[str]
    confidence_score: float

class LogicAnalyzerAgent(BaseOpulenceAgent): 
    """Agent for analyzing program logic and business rules with lazy loading"""
    
    def __init__(self, coordinator, llm_engine: AsyncLLMEngine = None, 
                 db_path: str = "opulence_data.db", gpu_id: int = 0):
        
        # ✅ FIXED: Proper super().__init__() call first
        super().__init__(coordinator, "lineage_analyzer", db_path, gpu_id)
        # REMOVE: self.llm_engine = llm_engine
        self._engine = None  # Cached engine reference (starts as None)
        
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        # NEW: Lazy loading tracking
        self._engine_loaded = False
        self._using_shared_engine = False
                
        # Logic patterns to detect
        self.logic_patterns = {
            'conditional_logic': re.compile(r'\bIF\b.*?\bEND-IF\b', re.DOTALL | re.IGNORECASE),
            'loops': re.compile(r'\bPERFORM\b.*?\bUNTIL\b', re.IGNORECASE),
            'calculations': re.compile(r'\b(COMPUTE|ADD|SUBTRACT|MULTIPLY|DIVIDE)\b', re.IGNORECASE),
            'data_validation': re.compile(r'\b(NUMERIC|ALPHABETIC|ALPHANUMERIC)\b', re.IGNORECASE),
            'error_handling': re.compile(r'\b(ON\s+ERROR|INVALID\s+KEY|AT\s+END)\b', re.IGNORECASE),
            'file_operations': re.compile(r'\b(READ|WRITE|REWRITE|DELETE|OPEN|CLOSE)\b', re.IGNORECASE)
        }
        
        # Business rule patterns
        self.business_rule_patterns = {
            'validation_rules': re.compile(r'IF.*?(INVALID|ERROR|REJECT)', re.IGNORECASE),
            'calculation_rules': re.compile(r'COMPUTE.*?=.*', re.IGNORECASE),
            'transformation_rules': re.compile(r'MOVE.*?TO.*', re.IGNORECASE),
            'approval_rules': re.compile(r'IF.*?(APPROVE|REJECT|PENDING)', re.IGNORECASE)
        }

    
    async def get_engine(self):
        """Get LLM engine with lazy loading and sharing"""
        if self._engine is None and self.coordinator:
            try:
                # Get assigned GPU for logic_analyzer agent type
                assigned_gpu = self.coordinator.agent_gpu_assignments.get("logic_analyzer")
                if assigned_gpu is not None:
                    # Get shared engine from coordinator
                    self._engine = await self.coordinator.get_shared_llm_engine(assigned_gpu)
                    self.gpu_id = assigned_gpu
                    self._using_shared_engine = True
                    self._engine_loaded = True
                    self.logger.info(f"✅ LogicAnalyzer using shared engine on GPU {assigned_gpu}")
                else:
                    raise ValueError("No GPU assigned for logic_analyzer agent type")
            except Exception as e:
                self.logger.error(f"❌ Failed to get shared engine: {e}")
                raise
        
        return self._engine
    # ADD this new method
    
    # ADD this new method
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'logic_analyzer'
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_loaded_lazily'] = self._engine_loaded
        return result
        
    async def analyze_program(self, program_name: str) -> Dict[str, Any]:
        """Comprehensive analysis of a program's logic"""
        try:
            #await self._ensure_llm_engine()  # ADD this line
            
            # Get program chunks from database
            chunks = await self._get_program_chunks(program_name)
            
            if not chunks:
                return self._add_processing_info({"error": f"Program {program_name} not found"})
            
            # Analyze each chunk
            chunk_analyses = []
            total_complexity = 0
            
            for chunk in chunks:
                chunk_analysis = await self._analyze_chunk_logic(chunk)
                chunk_analyses.append(chunk_analysis)
                total_complexity += chunk_analysis.get('complexity_score', 0)
            
            # Extract business rules
            business_rules = await self._extract_business_rules(chunks)
            
            # Identify logic patterns
            logic_patterns = await self._identify_logic_patterns(chunks)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(chunk_analyses, logic_patterns)
            
            # Calculate overall metrics
            metrics = self._calculate_program_metrics(chunk_analyses, logic_patterns)
            
            result = {
                "program_name": program_name,
                "total_chunks": len(chunks),
                "complexity_score": total_complexity / len(chunks) if chunks else 0,
                "chunk_analyses": chunk_analyses,
                "business_rules": [rule.__dict__ for rule in business_rules],
                "logic_patterns": [pattern.__dict__ for pattern in logic_patterns],
                "recommendations": recommendations,
                "metrics": metrics,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Program analysis failed for {program_name}: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def stream_logic_analysis(self, program_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream logic analysis results as they're processed"""
        try:
            #await self._ensure_llm_engine()  # ADD this line
            
            chunks = await self._get_program_chunks(program_name)
            
            if not chunks:
                yield self._add_processing_info({"error": f"Program {program_name} not found"})
                return
            
            # Yield initial status
            yield self._add_processing_info({
                "status": "started",
                "program_name": program_name,
                "total_chunks": len(chunks),
                "progress": 0
            })
            
            chunk_analyses = []
            
            # Process chunks and stream results
            for i, chunk in enumerate(chunks):
                # Analyze chunk
                chunk_analysis = await self._analyze_chunk_logic(chunk)
                chunk_analyses.append(chunk_analysis)
                
                # Yield progress update
                progress = ((i + 1) / len(chunks)) * 100
                yield self._add_processing_info({
                    "status": "processing",
                    "progress": progress,
                    "current_chunk": chunk[2],  # chunk_id
                    "chunk_analysis": chunk_analysis
                })
            
            # Final analysis
            business_rules = await self._extract_business_rules(chunks)
            logic_patterns = await self._identify_logic_patterns(chunks)
            recommendations = await self._generate_recommendations(chunk_analyses, logic_patterns)
            metrics = self._calculate_program_metrics(chunk_analyses, logic_patterns)
            
            # Yield final results
            yield self._add_processing_info({
                "status": "completed",
                "progress": 100,
                "business_rules": [rule.__dict__ for rule in business_rules],
                "logic_patterns": [pattern.__dict__ for pattern in logic_patterns],
                "recommendations": recommendations,
                "metrics": metrics
            })
            
        except Exception as e:
            yield self._add_processing_info({"status": "error", "error": str(e)})
    
    async def _get_program_chunks(self, program_name: str) -> List[tuple]:
        """Get all chunks for a program from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, program_name, chunk_id, chunk_type, content, metadata
            FROM program_chunks 
            WHERE program_name = ?
            ORDER BY chunk_id
        """, (program_name,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        return chunks
    
    async def _analyze_chunk_logic(self, chunk: tuple) -> Dict[str, Any]:
        """Analyze logic in a single chunk"""
        chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = chunk
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        # Use LLM for detailed logic analysis
        logic_analysis = await self._llm_analyze_logic(content, chunk_type)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(content)
        
        # Detect patterns using regex
        detected_patterns = self._detect_patterns_regex(content)
        
        # Extract control flow
        control_flow = self._extract_control_flow(content)
        
        return {
            "chunk_id": chunk_id_str,
            "chunk_type": chunk_type,
            "complexity_score": complexity_score,
            "logic_analysis": logic_analysis,
            "detected_patterns": detected_patterns,
            "control_flow": control_flow,
            "line_count": len(content.split('\n')),
            "metadata": metadata
        }
    
    async def _llm_analyze_logic(self, content: str, chunk_type: str) -> Dict[str, Any]:
        """Use LLM to analyze logic in code chunk"""
        #await self._ensure_llm_engine()  # ADD this line
        
        prompt = f"""
        Analyze the logic in this {chunk_type} code chunk:
        
        {content[:1500]}...
        
        Provide analysis on:
        1. Main logical operations performed
        2. Decision points and conditions
        3. Data transformations
        4. Error handling mechanisms
        5. Business logic patterns
        6. Potential optimization opportunities
        
        Return as JSON:
        {{
            "main_operations": ["operation1", "operation2"],
            "decision_points": ["condition1", "condition2"],
            "data_transformations": ["transform1", "transform2"],
            "error_handling": ["mechanism1", "mechanism2"],
            "business_logic": "description",
            "optimizations": ["opportunity1", "opportunity2"]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=800)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        # Fallback analysis
        return {
            "main_operations": self._extract_operations(content),
            "decision_points": self._extract_decision_points(content),
            "data_transformations": self._extract_transformations(content),
            "error_handling": self._extract_error_handling(content),
            "business_logic": "Standard business processing",
            "optimizations": []
        }
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate cyclomatic complexity score"""
        # Count decision points
        decision_keywords = ['IF', 'WHEN', 'PERFORM', 'UNTIL', 'WHILE', 'CASE']
        decision_count = 0
        
        for keyword in decision_keywords:
            decision_count += len(re.findall(rf'\b{keyword}\b', content, re.IGNORECASE))
        
        # Count nested levels
        nesting_level = 0
        max_nesting = 0
        
        for line in content.split('\n'):
            if re.search(r'\bIF\b|\bPERFORM\b', line, re.IGNORECASE):
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif re.search(r'\bEND-IF\b|\bEND-PERFORM\b', line, re.IGNORECASE):
                nesting_level = max(0, nesting_level - 1)
        
        # Calculate complexity: base 1 + decisions + nesting penalty
        complexity = 1 + decision_count + (max_nesting * 0.5)
        
        return min(complexity, 10.0)  # Cap at 10
    
    def _detect_patterns_regex(self, content: str) -> List[str]:
        """Detect logic patterns using regex"""
        detected = []
        
        for pattern_name, pattern_regex in self.logic_patterns.items():
            if pattern_regex.search(content):
                detected.append(pattern_name)
        
        return detected
    
    def _extract_control_flow(self, content: str) -> Dict[str, Any]:
        """Extract control flow information"""
        control_flow = {
            "sequential_statements": 0,
            "conditional_statements": 0,
            "loop_statements": 0,
            "goto_statements": 0,
            "perform_statements": 0
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip().upper()
            
            if re.search(r'\bIF\b', line):
                control_flow["conditional_statements"] += 1
            elif re.search(r'\bPERFORM.*UNTIL\b', line):
                control_flow["loop_statements"] += 1
            elif re.search(r'\bGO\s+TO\b', line):
                control_flow["goto_statements"] += 1
            elif re.search(r'\bPERFORM\b', line):
                control_flow["perform_statements"] += 1
            elif line and not line.startswith('*'):
                control_flow["sequential_statements"] += 1
        
        return control_flow
    
    async def _extract_business_rules(self, chunks: List[tuple]) -> List[BusinessRule]:
        """Extract business rules from program chunks"""
        business_rules = []
        rule_id_counter = 1
        
        for chunk in chunks:
            content = chunk[4]  # content field
            
            # Look for validation rules
            for match in self.business_rule_patterns['validation_rules'].finditer(content):
                rule = await self._analyze_business_rule(
                    f"RULE_{rule_id_counter:03d}",
                    "validation",
                    match.group(0),
                    content
                )
                if rule:
                    business_rules.append(rule)
                    rule_id_counter += 1
            
            # Look for calculation rules
            for match in self.business_rule_patterns['calculation_rules'].finditer(content):
                rule = await self._analyze_business_rule(
                    f"RULE_{rule_id_counter:03d}",
                    "calculation",
                    match.group(0),
                    content
                )
                if rule:
                    business_rules.append(rule)
                    rule_id_counter += 1
        
        return business_rules
    
    async def _analyze_business_rule(self, rule_id: str, rule_type: str, 
                                   rule_code: str, context: str) -> Optional[BusinessRule]:
        """Analyze a specific business rule using LLM"""
        #await self._ensure_llm_engine()  # ADD this line
        
        prompt = f"""
        Analyze this business rule from COBOL code:
        
        Rule Type: {rule_type}
        Rule Code: {rule_code}
        
        Context (surrounding code):
        {context[:500]}...
        
        Extract:
        1. The condition that triggers this rule
        2. The action taken when condition is met
        3. Fields/variables involved
        4. Confidence in rule extraction (0.0-1.0)
        
        Return as JSON:
        {{
            "condition": "description of condition",
            "action": "description of action",
            "fields_involved": ["field1", "field2"],
            "confidence_score": 0.8
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                rule_data = json.loads(response_text[json_start:json_end])
                
                return BusinessRule(
                    rule_id=rule_id,
                    rule_type=rule_type,
                    condition=rule_data.get('condition', ''),
                    action=rule_data.get('action', ''),
                    fields_involved=rule_data.get('fields_involved', []),
                    confidence_score=rule_data.get('confidence_score', 0.5)
                )
        except:
            pass
        
        return None
    
    async def _identify_logic_patterns(self, chunks: List[tuple]) -> List[LogicPattern]:
        """Identify common logic patterns across chunks"""
        patterns = []
        pattern_id = 1
        
        for chunk in chunks:
            content = chunk[4]
            chunk_type = chunk[3]
            
            # Identify specific patterns using LLM
            identified_patterns = await self._llm_identify_patterns(content, chunk_type)
            
            for pattern_data in identified_patterns:
                pattern = LogicPattern(
                    pattern_type=pattern_data.get('type', 'unknown'),
                    pattern_name=f"PATTERN_{pattern_id:03d}",
                    complexity_score=pattern_data.get('complexity', 1.0),
                    description=pattern_data.get('description', ''),
                    code_snippet=pattern_data.get('snippet', content[:200]),
                    recommendations=pattern_data.get('recommendations', [])
                )
                patterns.append(pattern)
                pattern_id += 1
        
        return patterns
    
    async def _llm_identify_patterns(self, content: str, chunk_type: str) -> List[Dict[str, Any]]:
        """Use LLM to identify logic patterns"""
        #await self._ensure_llm_engine()  # ADD this line
        
        prompt = f"""
        Identify common logic patterns in this {chunk_type} code:
        
        {content[:1000]}...
        
        Look for patterns like:
        - Data validation patterns
        - Error handling patterns
        - Loop constructs
        - Conditional logic patterns
        - File processing patterns
        
        Return as JSON array:
        [
            {{
                "type": "validation",
                "description": "pattern description",
                "complexity": 2.5,
                "snippet": "code snippet",
                "recommendations": ["recommendation1", "recommendation2"]
            }}
        ]
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '[' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        return []
    
    async def _generate_recommendations(self, chunk_analyses: List[Dict], 
                                      logic_patterns: List[LogicPattern]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze complexity
        high_complexity_chunks = [
            chunk for chunk in chunk_analyses 
            if chunk.get('complexity_score', 0) > 7
        ]
        
        if high_complexity_chunks:
            recommendations.append(
                f"Consider refactoring {len(high_complexity_chunks)} high-complexity chunks "
                "to improve maintainability"
            )
        
        # Check for error handling
        chunks_without_error_handling = [
            chunk for chunk in chunk_analyses
            if 'error_handling' not in chunk.get('detected_patterns', [])
        ]
        
        if len(chunks_without_error_handling) > len(chunk_analyses) * 0.5:
            recommendations.append(
                "Add comprehensive error handling to improve robustness"
            )
        
        # Pattern-based recommendations
        pattern_types = [pattern.pattern_type for pattern in logic_patterns]
        
        if 'validation' not in pattern_types:
            recommendations.append(
                "Consider adding data validation patterns for better data quality"
            )
        
        if 'loops' in pattern_types:
            recommendations.append(
                "Review loop constructs for potential performance optimizations"
            )
        
        return recommendations
    
    def _calculate_program_metrics(self, chunk_analyses: List[Dict], 
                                 logic_patterns: List[LogicPattern]) -> Dict[str, Any]:
        """Calculate overall program metrics"""
        if not chunk_analyses:
            return {}
        
        complexity_scores = [chunk.get('complexity_score', 0) for chunk in chunk_analyses]
        
        return {
            "average_complexity": sum(complexity_scores) / len(complexity_scores),
            "max_complexity": max(complexity_scores),
            "min_complexity": min(complexity_scores),
            "total_patterns": len(logic_patterns),
            "pattern_types": list(set(pattern.pattern_type for pattern in logic_patterns)),
            "maintainability_score": self._calculate_maintainability_score(chunk_analyses),
            "total_lines": sum(chunk.get('line_count', 0) for chunk in chunk_analyses)
        }
    
    def _calculate_maintainability_score(self, chunk_analyses: List[Dict]) -> float:
        """Calculate maintainability score (0-10)"""
        if not chunk_analyses:
            return 5.0
        
        # Factors affecting maintainability
        avg_complexity = sum(chunk.get('complexity_score', 0) for chunk in chunk_analyses) / len(chunk_analyses)
        error_handling_ratio = sum(
            1 for chunk in chunk_analyses 
            if 'error_handling' in chunk.get('detected_patterns', [])
        ) / len(chunk_analyses)
        
        # Calculate score (higher is better)
        complexity_penalty = max(0, avg_complexity - 3) * 0.5
        error_handling_bonus = error_handling_ratio * 2
        
        score = 8.0 - complexity_penalty + error_handling_bonus
        return max(0.0, min(10.0, score))
    
    # Helper methods for fallback analysis
    def _extract_operations(self, content: str) -> List[str]:
        """Extract main operations from content"""
        operations = []
        operation_keywords = [
            'READ', 'WRITE', 'COMPUTE', 'MOVE', 'PERFORM', 'CALL',
            'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'OPEN', 'CLOSE'
        ]
        
        for keyword in operation_keywords:
            if re.search(rf'\b{keyword}\b', content, re.IGNORECASE):
                operations.append(keyword)
        
        return list(set(operations))
    
    def _extract_decision_points(self, content: str) -> List[str]:
        """Extract decision points from content"""
        decision_points = []
        
        # Find IF statements
        if_pattern = re.compile(r'IF\s+([^.]+?)(?:THEN|$)', re.IGNORECASE | re.MULTILINE)
        for match in if_pattern.finditer(content):
            decision_points.append(f"IF {match.group(1).strip()}")
        
        # Find WHEN statements
        when_pattern = re.compile(r'WHEN\s+([^.]+)', re.IGNORECASE)
        for match in when_pattern.finditer(content):
            decision_points.append(f"WHEN {match.group(1).strip()}")
        
        return decision_points[:5]  # Limit to first 5
    
    def _extract_transformations(self, content: str) -> List[str]:
        """Extract data transformations from content"""
        transformations = []
        
        # MOVE statements
        move_pattern = re.compile(r'MOVE\s+([^.]+?)\s+TO\s+([^.]+)', re.IGNORECASE)
        for match in move_pattern.finditer(content):
            transformations.append(f"MOVE {match.group(1).strip()} TO {match.group(2).strip()}")
        
        # COMPUTE statements
        compute_pattern = re.compile(r'COMPUTE\s+([^.]+)', re.IGNORECASE)
        for match in compute_pattern.finditer(content):
            transformations.append(f"COMPUTE {match.group(1).strip()}")
        
        return transformations[:5]  # Limit to first 5
    
    def _extract_error_handling(self, content: str) -> List[str]:
        """Extract error handling mechanisms from content"""
        error_handling = []
        
        error_patterns = [
            r'ON\s+ERROR',
            r'INVALID\s+KEY',
            r'AT\s+END',
            r'NOT\s+ON\s+EXCEPTION',
            r'ON\s+EXCEPTION'
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                error_handling.append(pattern.replace(r'\s+', ' '))
        
        return error_handling
    
    async def analyze_business_logic(self, program_name: str) -> Dict[str, Any]:
        """Specialized analysis focused on business logic"""
        try:
            #await self._ensure_llm_engine()  # ADD this line
            
            chunks = await self._get_program_chunks(program_name)
            
            if not chunks:
                return self._add_processing_info({"error": f"Program {program_name} not found"})
            
            # Focus on business logic extraction
            business_logic = await self._extract_comprehensive_business_logic(chunks)
            
            result = {
                "program_name": program_name,
                "business_logic": business_logic,
                "analysis_type": "business_logic_focused"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Business logic analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _extract_comprehensive_business_logic(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Extract comprehensive business logic from all chunks"""
        #await self._ensure_llm_engine()  # ADD this line
        
        all_content = '\n'.join([chunk[4] for chunk in chunks])
        
        prompt = f"""
        Analyze this complete COBOL program and extract the business logic:
        
        {all_content[:2000]}...
        
        Focus on:
        1. What business processes this program implements
        2. Key business rules and validations
        3. Data processing workflows
        4. Business calculations and transformations
        5. Decision points that affect business outcomes
        
        Provide a comprehensive business logic summary.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return {
            "summary": result.outputs[0].text.strip(),
            "extraction_method": "llm_analysis",
            "confidence": "high"
        }

    # ADD new methods for enhanced functionality
    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """Analyze full lifecycle of a component"""
        try:
            #
            # await self._ensure_llm_engine()
            
            # Get all chunks related to this component
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if component_type == "field":
                # Search for field usage across all programs
                cursor.execute("""
                    SELECT pc.program_name, pc.chunk_id, pc.content, pc.metadata
                    FROM program_chunks pc
                    WHERE pc.content LIKE ? OR pc.metadata LIKE ?
                """, (f"%{component_name}%", f"%{component_name}%"))
            else:
                # Search for program/file references
                cursor.execute("""
                    SELECT pc.program_name, pc.chunk_id, pc.content, pc.metadata
                    FROM program_chunks pc
                    WHERE pc.program_name = ?
                """, (component_name,))
            
            related_chunks = cursor.fetchall()
            conn.close()
            
            if not related_chunks:
                return self._add_processing_info({
                    "component_name": component_name,
                    "component_type": component_type,
                    "lifecycle_analysis": "No usage found",
                    "usage_count": 0
                })
            
            # Analyze lifecycle patterns
            lifecycle_analysis = await self._analyze_lifecycle_patterns(related_chunks, component_name)
            
            result = {
                "component_name": component_name,
                "component_type": component_type,
                "usage_count": len(related_chunks),
                "lifecycle_analysis": lifecycle_analysis,
                "programs_involved": list(set(chunk[0] for chunk in related_chunks))
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Lifecycle analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _analyze_lifecycle_patterns(self, chunks: List[tuple], component_name: str) -> Dict[str, Any]:
        """Analyze lifecycle patterns for a component"""
        #await self._ensure_llm_engine()
        
        # Combine relevant content
        all_content = '\n'.join([chunk[2] for chunk in chunks])
        
        prompt = f"""
        Analyze the lifecycle of component '{component_name}' based on this code:
        
        {all_content[:1500]}...
        
        Determine:
        1. Where this component is created/initialized
        2. How it's used throughout the programs
        3. Where it's modified or updated
        4. Any cleanup or finalization
        5. Dependencies and relationships
        
        Return as JSON:
        {{
            "creation_points": ["location1", "location2"],
            "usage_patterns": ["pattern1", "pattern2"],
            "modification_points": ["mod1", "mod2"],
            "dependencies": ["dep1", "dep2"],
            "lifecycle_summary": "summary text"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=600)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        return {
            "creation_points": [],
            "usage_patterns": [],
            "modification_points": [],
            "dependencies": [],
            "lifecycle_summary": "Analysis could not be completed"
        }
    
    async def find_dependencies(self, component_name: str) -> List[str]:
        """Find all dependencies for a component"""
        try:
            #await self._ensure_llm_engine()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search for component references
            cursor.execute("""
                SELECT DISTINCT pc.program_name, pc.metadata
                FROM program_chunks pc
                WHERE pc.content LIKE ? OR pc.metadata LIKE ?
            """, (f"%{component_name}%", f"%{component_name}%"))
            
            results = cursor.fetchall()
            conn.close()
            
            dependencies = set()
            
            for program_name, metadata_str in results:
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                        
                        # Extract various types of dependencies
                        if 'field_names' in metadata:
                            dependencies.update(metadata['field_names'])
                        if 'files' in metadata:
                            dependencies.update(metadata['files'])
                        if 'called_paragraphs' in metadata:
                            dependencies.update(metadata['called_paragraphs'])
                        if 'tables' in metadata:
                            dependencies.update(metadata['tables'])
                            
                    except json.JSONDecodeError:
                        continue
                
                # Add the program itself as a dependency
                dependencies.add(program_name)
            
            # Remove the component itself from dependencies
            dependencies.discard(component_name)
            
            return list(dependencies)
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            return []