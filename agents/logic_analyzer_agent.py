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
        super().__init__(coordinator, "logic_analyzer", db_path, gpu_id)
        self._engine = None  # Cached engine reference (starts as None)
        
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        # NEW: Lazy loading tracking
        self._engine_loaded = False
        self._using_shared_engine = False
        
        # NEW: Prompt length limits
        self.MAX_PROMPT_LENGTH = 1024  # Token limit
        self.ESTIMATED_CHARS_PER_TOKEN = 4  # Conservative estimate
        self.MAX_PROMPT_CHARS = self.MAX_PROMPT_LENGTH * self.ESTIMATED_CHARS_PER_TOKEN
                
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

    def _validate_and_truncate_prompt(self, prompt: str, preserve_structure: bool = True) -> str:
        """
        Validate prompt length and truncate if necessary while preserving structure
        
        Args:
            prompt: The prompt text to validate
            preserve_structure: Whether to preserve JSON structure indicators
            
        Returns:
            Truncated prompt that fits within token limits
        """
        if len(prompt) <= self.MAX_PROMPT_CHARS:
            return prompt
        
        self.logger.warning(f"Prompt length {len(prompt)} chars exceeds limit {self.MAX_PROMPT_CHARS}, truncating...")
        
        if preserve_structure:
            # Try to preserve the structure by finding key sections
            lines = prompt.split('\n')
            
            # Keep instruction lines (usually at start)
            instruction_lines = []
            code_lines = []
            json_template_lines = []
            
            in_code_section = False
            in_json_section = False
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['analyze', 'determine', 'provide', 'return as json']):
                    instruction_lines.append(line)
                elif line.strip().startswith('{') or line.strip().startswith('[') or in_json_section:
                    json_template_lines.append(line)
                    in_json_section = True
                    if line.strip().endswith('}') or line.strip().endswith(']'):
                        in_json_section = False
                elif len(line.strip()) > 0 and not line.startswith(' '):
                    if not in_code_section:
                        instruction_lines.append(line)
                else:
                    code_lines.append(line)
                    in_code_section = True
            
            # Calculate space allocation
            instruction_text = '\n'.join(instruction_lines)
            json_template_text = '\n'.join(json_template_lines)
            
            # Reserve space for instructions and JSON template
            reserved_space = len(instruction_text) + len(json_template_text) + 200  # buffer
            available_for_code = self.MAX_PROMPT_CHARS - reserved_space
            
            if available_for_code > 500:  # Minimum useful code length
                # Truncate code section
                code_text = '\n'.join(code_lines)
                if len(code_text) > available_for_code:
                    code_text = code_text[:available_for_code-20] + "\n..."
                
                # Reconstruct prompt
                truncated_prompt = f"{instruction_text}\n{code_text}\n{json_template_text}"
            else:
                # If structure preservation doesn't work, do simple truncation
                truncated_prompt = prompt[:self.MAX_PROMPT_CHARS-20] + "\n..."
        else:
            # Simple truncation
            truncated_prompt = prompt[:self.MAX_PROMPT_CHARS-20] + "..."
        
        self.logger.info(f"Prompt truncated from {len(prompt)} to {len(truncated_prompt)} chars")
        return truncated_prompt

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
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'logic_analyzer'
            result['using_shared_engine'] = self._using_shared_engine
            result['engine_loaded_lazily'] = self._engine_loaded
        return result
        
    async def analyze_program(self, program_name: str) -> Dict[str, Any]:
        """Analyze program logic comprehensively"""
        try:
            async with self.get_engine_context() as engine:
                # Get program chunks from database
                chunks = await self._get_program_chunks(program_name)
                
                if not chunks:
                    return self._add_processing_info({"error": f"Program {program_name} not found"})
                
                # Analyze each chunk
                chunk_analyses = []
                total_complexity = 0
                
                for chunk in chunks:
                    chunk_analysis = await self._analyze_chunk_logic(chunk, engine)
                    chunk_analyses.append(chunk_analysis)
                    total_complexity += chunk_analysis.get('complexity_score', 0)
                
                # Extract business rules
                business_rules = await self._extract_business_rules(chunks, engine)
                
                # Identify logic patterns
                logic_patterns = await self._identify_logic_patterns(chunks, engine)
                
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
    
    async def _analyze_lifecycle_patterns(self, chunks: List[tuple], component_name: str, engine) -> Dict[str, Any]:
        """Analyze lifecycle patterns for a component"""
        # Combine relevant content with length limit
        all_content = '\n'.join([chunk[2] for chunk in chunks])
        
        # Create prompt with careful length management
        base_prompt = f"""Analyze the lifecycle of component '{component_name}' based on this code:

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
}}"""
        
        # Calculate available space for content
        available_space = self.MAX_PROMPT_CHARS - len(base_prompt) - 100  # buffer
        if len(all_content) > available_space:
            all_content = all_content[:available_space] + "..."
        
        full_prompt = base_prompt.replace("based on this code:", f"based on this code:\n\n{all_content}")
        
        # Validate final prompt length
        full_prompt = self._validate_and_truncate_prompt(full_prompt, preserve_structure=True)
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=600)
        request_id = str(uuid.uuid4())
        
        async for result in engine.generate(full_prompt, sampling_params, request_id=request_id):
            response_text = result.outputs[0].text.strip()
            break
        
        try:
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

    async def analyze_complexity_trends(self, program_name: str) -> Dict[str, Any]:
        """Analyze complexity trends across program chunks"""
        try:
            async with self.get_engine_context() as engine:
                chunks = await self._get_program_chunks(program_name)
                
                if not chunks:
                    return self._add_processing_info({"error": f"Program {program_name} not found"})
                
                complexity_data = []
                
                for chunk in chunks:
                    complexity_score = self._calculate_complexity_score(chunk[4])  # content
                    complexity_data.append({
                        "chunk_id": chunk[2],
                        "chunk_type": chunk[3],
                        "complexity_score": complexity_score,
                        "line_count": len(chunk[4].split('\n'))
                    })
                
                # Calculate trends
                trends = self._calculate_complexity_trends(complexity_data)
                
                result = {
                    "program_name": program_name,
                    "complexity_data": complexity_data,
                    "trends": trends,
                    "analysis_type": "complexity_trends"
                }
                
                return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Complexity trends analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    def _calculate_complexity_trends(self, complexity_data: List[Dict]) -> Dict[str, Any]:
        """Calculate complexity trends from data"""
        if not complexity_data:
            return {}
        
        scores = [item['complexity_score'] for item in complexity_data]
        
        # Basic statistics
        avg_complexity = sum(scores) / len(scores)
        max_complexity = max(scores)
        min_complexity = min(scores)
        
        # Find high complexity chunks
        high_complexity_threshold = avg_complexity + (max_complexity - avg_complexity) * 0.5
        high_complexity_chunks = [
            item for item in complexity_data 
            if item['complexity_score'] > high_complexity_threshold
        ]
        
        # Complexity by chunk type
        complexity_by_type = {}
        for item in complexity_data:
            chunk_type = item['chunk_type']
            if chunk_type not in complexity_by_type:
                complexity_by_type[chunk_type] = []
            complexity_by_type[chunk_type].append(item['complexity_score'])
        
        # Average complexity by type
        avg_complexity_by_type = {
            chunk_type: sum(scores) / len(scores)
            for chunk_type, scores in complexity_by_type.items()
        }
        
        return {
            "average_complexity": avg_complexity,
            "max_complexity": max_complexity,
            "min_complexity": min_complexity,
            "high_complexity_chunks": len(high_complexity_chunks),
            "high_complexity_threshold": high_complexity_threshold,
            "complexity_by_type": avg_complexity_by_type,
            "total_chunks_analyzed": len(complexity_data)
        }
    
    async def generate_optimization_report(self, program_name: str) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations"""
        try:
            async with self.get_engine_context() as engine:
                # Get program analysis
                analysis_result = await self.analyze_program(program_name)
                
                if "error" in analysis_result:
                    return analysis_result
                
                # Generate detailed optimization recommendations
                optimization_report = await self._generate_optimization_recommendations(
                    analysis_result, engine
                )
                
                result = {
                    "program_name": program_name,
                    "optimization_report": optimization_report,
                    "base_analysis": analysis_result,
                    "report_type": "optimization_recommendations"
                }
                
                return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Optimization report generation failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _generate_optimization_recommendations(self, analysis_result: Dict, engine) -> Dict[str, Any]:
        """Generate detailed optimization recommendations using LLM"""
        
        # Prepare analysis summary for LLM
        summary = {
            "complexity_score": analysis_result.get("complexity_score", 0),
            "total_chunks": analysis_result.get("total_chunks", 0),
            "high_complexity_chunks": len([
                chunk for chunk in analysis_result.get("chunk_analyses", [])
                if chunk.get("complexity_score", 0) > 7
            ]),
            "patterns_found": len(analysis_result.get("logic_patterns", [])),
            "business_rules": len(analysis_result.get("business_rules", []))
        }
        
        prompt = f"""Based on this COBOL program analysis, generate detailed optimization recommendations:

Program Analysis Summary:
- Average Complexity Score: {summary['complexity_score']:.2f}
- Total Chunks: {summary['total_chunks']}
- High Complexity Chunks: {summary['high_complexity_chunks']}
- Logic Patterns Found: {summary['patterns_found']}
- Business Rules Identified: {summary['business_rules']}

Provide optimization recommendations in these categories:
1. Performance optimizations
2. Maintainability improvements
3. Code structure enhancements
4. Error handling improvements
5. Testing recommendations

Return as JSON:
{{
    "performance": ["recommendation1", "recommendation2"],
    "maintainability": ["recommendation1", "recommendation2"],
    "structure": ["recommendation1", "recommendation2"],
    "error_handling": ["recommendation1", "recommendation2"],
    "testing": ["recommendation1", "recommendation2"],
    "priority_recommendations": ["high_priority1", "high_priority2"]
}}"""
        
        # Validate prompt length
        prompt = self._validate_and_truncate_prompt(prompt, preserve_structure=True)
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=800)
        request_id = str(uuid.uuid4())
        
        async for result in engine.generate(prompt, sampling_params, request_id=request_id):
            response_text = result.outputs[0].text.strip()
            break
        
        try:
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        # Fallback recommendations
        return {
            "performance": [
                "Review file I/O operations for efficiency",
                "Optimize loop constructs and nested conditions"
            ],
            "maintainability": [
                "Break down high-complexity modules into smaller functions",
                "Add comprehensive documentation for business rules"
            ],
            "structure": [
                "Standardize naming conventions",
                "Group related functionality into cohesive modules"
            ],
            "error_handling": [
                "Add comprehensive error handling for all file operations",
                "Implement consistent error reporting mechanisms"
            ],
            "testing": [
                "Create unit tests for business rule validation",
                "Develop integration tests for file processing workflows"
            ],
            "priority_recommendations": [
                "Address high-complexity chunks first",
                "Implement error handling for critical operations"
            ]
        }
    
    async def analyze_code_quality(self, program_name: str) -> Dict[str, Any]:
        """Analyze overall code quality metrics"""
        try:
            async with self.get_engine_context() as engine:
                chunks = await self._get_program_chunks(program_name)
                
                if not chunks:
                    return self._add_processing_info({"error": f"Program {program_name} not found"})
                
                quality_metrics = await self._calculate_quality_metrics(chunks, engine)
                
                result = {
                    "program_name": program_name,
                    "quality_metrics": quality_metrics,
                    "analysis_type": "code_quality"
                }
                
                return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _calculate_quality_metrics(self, chunks: List[tuple], engine) -> Dict[str, Any]:
        """Calculate comprehensive code quality metrics"""
        
        total_lines = 0
        comment_lines = 0
        empty_lines = 0
        code_lines = 0
        
        complexity_scores = []
        
        for chunk in chunks:
            content = chunk[4]
            lines = content.split('\n')
            
            chunk_total = len(lines)
            chunk_comments = 0
            chunk_empty = 0
            chunk_code = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    chunk_empty += 1
                elif stripped.startswith('*') or stripped.startswith('//'):
                    chunk_comments += 1
                else:
                    chunk_code += 1
            
            total_lines += chunk_total
            comment_lines += chunk_comments
            empty_lines += chunk_empty
            code_lines += chunk_code
            
            # Calculate complexity for this chunk
            complexity = self._calculate_complexity_score(content)
            complexity_scores.append(complexity)
        
        # Calculate ratios
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        code_ratio = code_lines / total_lines if total_lines > 0 else 0
        
        # Calculate average complexity
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        # Quality score calculation (0-10)
        quality_score = self._calculate_overall_quality_score(
            comment_ratio, avg_complexity, complexity_scores
        )
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "empty_lines": empty_lines,
            "comment_ratio": round(comment_ratio, 3),
            "code_ratio": round(code_ratio, 3),
            "average_complexity": round(avg_complexity, 2),
            "max_complexity": max(complexity_scores) if complexity_scores else 0,
            "min_complexity": min(complexity_scores) if complexity_scores else 0,
            "quality_score": round(quality_score, 2),
            "total_chunks": len(chunks)
        }
    
    def _calculate_overall_quality_score(self, comment_ratio: float, 
                                       avg_complexity: float, 
                                       complexity_scores: List[float]) -> float:
        """Calculate overall quality score (0-10, higher is better)"""
        
        # Start with base score
        score = 5.0
        
        # Comment ratio bonus (good documentation)
        if comment_ratio > 0.1:  # More than 10% comments
            score += min(2.0, comment_ratio * 5)
        
        # Complexity penalty
        if avg_complexity > 5:
            score -= min(3.0, (avg_complexity - 5) * 0.5)
        
        # Consistency bonus (low variance in complexity)
        if complexity_scores:
            complexity_variance = sum(
                (score - avg_complexity) ** 2 for score in complexity_scores
            ) / len(complexity_scores)
            
            if complexity_variance < 2.0:  # Low variance
                score += 1.0
        
        return max(0.0, min(10.0, score))
    
    async def export_analysis_report(self, program_name: str, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive analysis report in specified format"""
        try:
            async with self.get_engine_context() as engine:
                # Get comprehensive analysis
                program_analysis = await self.analyze_program(program_name)
                quality_analysis = await self.analyze_code_quality(program_name)
                optimization_report = await self.generate_optimization_report(program_name)
                
                # Combine all analyses
                comprehensive_report = {
                    "program_name": program_name,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "program_analysis": program_analysis,
                    "quality_analysis": quality_analysis,
                    "optimization_report": optimization_report,
                    "report_format": format,
                    "generated_by": "LogicAnalyzerAgent"
                }
                
                return self._add_processing_info(comprehensive_report)
                
        except Exception as e:
            self.logger.error(f"Report export failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    async def compare_programs(self, program1: str, program2: str) -> Dict[str, Any]:
        """Compare logic patterns between two programs"""
        try:
            async with self.get_engine_context() as engine:
                # Analyze both programs
                analysis1 = await self.analyze_program(program1)
                analysis2 = await self.analyze_program(program2)
                
                if "error" in analysis1 or "error" in analysis2:
                    return self._add_processing_info({
                        "error": "One or both programs could not be analyzed",
                        "program1_status": "error" if "error" in analysis1 else "success",
                        "program2_status": "error" if "error" in analysis2 else "success"
                    })
                
                # Compare metrics
                comparison = self._compare_program_metrics(analysis1, analysis2)
                
                result = {
                    "program1": program1,
                    "program2": program2,
                    "comparison": comparison,
                    "analysis_type": "program_comparison"
                }
                
                return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Program comparison failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    def _compare_program_metrics(self, analysis1: Dict, analysis2: Dict) -> Dict[str, Any]:
        """Compare metrics between two program analyses"""
        
        metrics1 = analysis1.get("metrics", {})
        metrics2 = analysis2.get("metrics", {})
        
        comparison = {
            "complexity_comparison": {
                "program1_avg": metrics1.get("average_complexity", 0),
                "program2_avg": metrics2.get("average_complexity", 0),
                "difference": metrics2.get("average_complexity", 0) - metrics1.get("average_complexity", 0)
            },
            "maintainability_comparison": {
                "program1_score": metrics1.get("maintainability_score", 0),
                "program2_score": metrics2.get("maintainability_score", 0),
                "difference": metrics2.get("maintainability_score", 0) - metrics1.get("maintainability_score", 0)
            },
            "size_comparison": {
                "program1_lines": metrics1.get("total_lines", 0),
                "program2_lines": metrics2.get("total_lines", 0),
                "size_ratio": metrics2.get("total_lines", 1) / max(metrics1.get("total_lines", 1), 1)
            },
            "pattern_comparison": {
                "program1_patterns": len(analysis1.get("logic_patterns", [])),
                "program2_patterns": len(analysis2.get("logic_patterns", [])),
                "common_pattern_types": self._find_common_pattern_types(
                    analysis1.get("logic_patterns", []),
                    analysis2.get("logic_patterns", [])
                )
            }
        }
        
        return comparison
    
    def _find_common_pattern_types(self, patterns1: List[Dict], patterns2: List[Dict]) -> List[str]:
        """Find common pattern types between two sets of patterns"""
        types1 = set(pattern.get("pattern_type", "") for pattern in patterns1)
        types2 = set(pattern.get("pattern_type", "") for pattern in patterns2)
        return list(types1.intersection(types2))

    async def batch_analyze_programs(self, program_names: List[str]) -> Dict[str, Any]:
        """Batch analyze multiple programs"""
        try:
            async with self.get_engine_context() as engine:
                results = {}
                
                for program_name in program_names:
                    try:
                        analysis = await self.analyze_program(program_name)
                        results[program_name] = analysis
                    except Exception as e:
                        results[program_name] = {"error": str(e)}
                
                # Calculate batch statistics
                batch_stats = self._calculate_batch_statistics(results)
                
                result = {
                    "batch_analysis": results,
                    "batch_statistics": batch_stats,
                    "programs_analyzed": len(program_names),
                    "analysis_type": "batch_analysis"
                }
                
                return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    def _calculate_batch_statistics(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate statistics across batch analysis results"""
        
        successful_analyses = [
            result for result in results.values() 
            if "error" not in result
        ]
        
        if not successful_analyses:
            return {"error": "No successful analyses"}
        
        # Aggregate complexity scores
        complexity_scores = [
            analysis.get("complexity_score", 0) 
            for analysis in successful_analyses
        ]
        
        # Aggregate maintainability scores
        maintainability_scores = [
            analysis.get("metrics", {}).get("maintainability_score", 0)
            for analysis in successful_analyses
        ]
        
        # Count patterns across all programs
        total_patterns = sum(
            len(analysis.get("logic_patterns", []))
            for analysis in successful_analyses
        )
        
        return {
            "avg_complexity": sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            "avg_maintainability": sum(maintainability_scores) / len(maintainability_scores) if maintainability_scores else 0,
            "total_programs_analyzed": len(successful_analyses),
            "failed_analyses": len(results) - len(successful_analyses),
            "total_patterns_found": total_patterns,
            "complexity_distribution": {
                "low": len([s for s in complexity_scores if s < 3]),
                "medium": len([s for s in complexity_scores if 3 <= s < 7]),
                "high": len([s for s in complexity_scores if s >= 7])
            }
        }
        
    async def stream_logic_analysis(self, program_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream logic analysis results as they're processed"""
        try:
            async with self.get_engine_context() as engine:
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
                    chunk_analysis = await self._analyze_chunk_logic(chunk, engine)
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
                business_rules = await self._extract_business_rules(chunks, engine)
                logic_patterns = await self._identify_logic_patterns(chunks, engine)
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
    
    async def _analyze_chunk_logic(self, chunk: tuple, engine) -> Dict[str, Any]:
        """Analyze logic in a single chunk"""
        chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = chunk
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        # Use LLM for detailed logic analysis
        logic_analysis = await self._llm_analyze_logic(content, chunk_type, engine)
        
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
    
    async def _llm_analyze_logic(self, content: str, chunk_type: str, engine) -> Dict[str, Any]:
        """Use LLM to analyze logic in code chunk"""
        
        # Create base prompt
        base_prompt = f"""Analyze the logic in this {chunk_type} code chunk:

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
}}"""
        
        # Calculate available space for content
        available_space = self.MAX_PROMPT_CHARS - len(base_prompt) - 100  # buffer
        truncated_content = content[:available_space] + "..." if len(content) > available_space else content
        
        full_prompt = base_prompt.replace("code chunk:", f"code chunk:\n\n{truncated_content}")
        
        # Final validation
        full_prompt = self._validate_and_truncate_prompt(full_prompt, preserve_structure=True)
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=800)
        request_id = str(uuid.uuid4())
        
        async for result in engine.generate(full_prompt, sampling_params, request_id=request_id):
            response_text = result.outputs[0].text.strip()
            break
        
        try:
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
    
    async def _extract_business_rules(self, chunks: List[tuple], engine) -> List[BusinessRule]:
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
                    content,
                    engine
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
                    content,
                    engine
                )
                if rule:
                    business_rules.append(rule)
                    rule_id_counter += 1
        
        return business_rules
    
    async def _analyze_business_rule(self, rule_id: str, rule_type: str, 
                                   rule_code: str, context: str, engine) -> Optional[BusinessRule]:
        """Analyze a specific business rule using LLM"""
        
        # Create base prompt
        base_prompt = f"""Analyze this business rule from COBOL code:

Rule Type: {rule_type}
Rule Code: {rule_code}

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
}}"""
        
        # Calculate available space for context
        available_space = self.MAX_PROMPT_CHARS - len(base_prompt) - 100
        truncated_context = context[:available_space] + "..." if len(context) > available_space else context
        
        full_prompt = base_prompt.replace("Rule Code: {rule_code}", f"Rule Code: {rule_code}\n\nContext (surrounding code):\n{truncated_context}")
        
        # Final validation
        full_prompt = self._validate_and_truncate_prompt(full_prompt, preserve_structure=True)
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        request_id = str(uuid.uuid4())
        
        async for result in engine.generate(full_prompt, sampling_params, request_id=request_id):
            response_text = result.outputs[0].text.strip()
            break
        
        try:
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
    
    async def _identify_logic_patterns(self, chunks: List[tuple], engine) -> List[LogicPattern]:
        """Identify common logic patterns across chunks"""
        patterns = []
        pattern_id = 1
        
        for chunk in chunks:
            content = chunk[4]
            chunk_type = chunk[3]
            
            # Identify specific patterns using LLM
            identified_patterns = await self._llm_identify_patterns(content, chunk_type, engine)
            
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
    
    async def _llm_identify_patterns(self, content: str, chunk_type: str, engine) -> List[Dict[str, Any]]:
        """Use LLM to identify logic patterns"""
        
        # Create base prompt
        base_prompt = f"""Identify common logic patterns in this {chunk_type} code:

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
]"""
        
        # Calculate available space for content
        available_space = self.MAX_PROMPT_CHARS - len(base_prompt) - 100
        truncated_content = content[:available_space] + "..." if len(content) > available_space else content
        
        full_prompt = base_prompt.replace("code:", f"code:\n\n{truncated_content}")
        
        # Final validation
        full_prompt = self._validate_and_truncate_prompt(full_prompt, preserve_structure=True)
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)
        request_id = str(uuid.uuid4())
        
        async for result in engine.generate(full_prompt, sampling_params, request_id=request_id):
            response_text = result.outputs[0].text.strip()
            break
        
        try:
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
            async with self.get_engine_context() as engine:
                chunks = await self._get_program_chunks(program_name)
                
                if not chunks:
                    return self._add_processing_info({"error": f"Program {program_name} not found"})
                
                # Focus on business logic extraction
                business_logic = await self._extract_comprehensive_business_logic(chunks, engine)
                
                result = {
                    "program_name": program_name,
                    "business_logic": business_logic,
                    "analysis_type": "business_logic_focused"
                }
                
                return self._add_processing_info(result)
                
        except Exception as e:
            self.logger.error(f"Business logic analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _extract_comprehensive_business_logic(self, chunks: List[tuple], engine) -> Dict[str, Any]:
        """Extract comprehensive business logic from all chunks"""
        
        # Combine content with length management
        all_content = '\n'.join([chunk[4] for chunk in chunks])
        
        # Create base prompt
        base_prompt = """Analyze this complete COBOL program and extract the business logic:

Focus on:
1. What business processes this program implements
2. Key business rules and validations
3. Data processing workflows
4. Business calculations and transformations
5. Decision points that affect business outcomes

Provide a comprehensive business logic summary."""
        
        # Calculate available space for content
        available_space = self.MAX_PROMPT_CHARS - len(base_prompt) - 100
        truncated_content = all_content[:available_space] + "..." if len(all_content) > available_space else all_content
        
        full_prompt = base_prompt.replace("program and extract", f"program and extract the business logic:\n\n{truncated_content}\n\nFocus on")
        
        # Final validation
        full_prompt = self._validate_and_truncate_prompt(full_prompt, preserve_structure=False)
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)
        request_id = str(uuid.uuid4())
        
        async for result in engine.generate(full_prompt, sampling_params, request_id=request_id):
            response_text = result.outputs[0].text.strip()
            break
        
        return {
            "summary": response_text,
            "extraction_method": "llm_analysis",
            "confidence": "high"
        }

    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """Analyze full lifecycle of a component"""
        try:
            async with self.get_engine_context() as engine:
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
                lifecycle_analysis = await self._analyze_lifecycle_patterns(related_chunks, component_name, engine)
                
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