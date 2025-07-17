# agents/logic_analyzer_agent.py
"""
API-BASED Agent 5: Logic Analyzer
Analyzes program logic with streaming support for complex COBOL/JCL operations
Now uses HTTP API calls instead of direct GPU model loading
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

from agents.base_agent_api import BaseOpulenceAgent

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
    """API-BASED: Agent for analyzing program logic and business rules"""
    
    def __init__(self, coordinator, llm_engine=None, 
                 db_path: str = "opulence_data.db", gpu_id: int = 0):
        
        # ✅ FIXED: Proper super().__init__() call first
        super().__init__(coordinator, "logic_analyzer", db_path, gpu_id)
        
        # Store coordinator reference for API calls
        self.coordinator = coordinator
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(__name__)
        
        # API-specific settings
        self.api_params = {
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9
        }
        
        # Prompt length limits for API efficiency
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
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'logic_analyzer'
            result['api_based'] = True
            result['coordinator_type'] = getattr(self.coordinator, 'stats', {}).get('coordinator_type', 'api_based')
        return result
        
    async def analyze_program(self, program_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze program logic comprehensively"""
        try:
            # Get program chunks from database
            chunks = await self._get_program_chunks(program_name)
            
            if not chunks:
                return self._add_processing_info({"error": f"Program {program_name} not found"})
            
            # Analyze each chunk with API
            chunk_analyses = []
            total_complexity = 0
            
            for chunk in chunks:
                chunk_analysis = await self._analyze_chunk_logic_api(chunk)
                chunk_analyses.append(chunk_analysis)
                total_complexity += chunk_analysis.get('complexity_score', 0)
            
            # Extract business rules with API
            business_rules = await self._extract_business_rules_api(chunks)
            
            # Identify logic patterns with API
            logic_patterns = await self._identify_logic_patterns_api(chunks)
            
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
    

    def _generate_fallback_lifecycle_summary(self, component_name: str, component_type: str, 
                                       programs_using: set, total_references: int) -> str:
        """Generate fallback lifecycle summary"""
        summary = f"## Lifecycle Analysis: {component_name}\n\n"
        
        summary += f"**Component Type**: {component_type.title()}\n"
        summary += f"**Usage Scope**: Found in {len(programs_using)} programs with {total_references} total references\n\n"
        
        summary += "### Component Overview\n\n"
        summary += f"The {component_type} '{component_name}' is actively used across multiple programs in the system, "
        summary += f"indicating its importance to business operations.\n\n"
        
        summary += "### Usage Analysis\n\n"
        summary += f"**Programs Using This Component**:\n"
        for program in sorted(programs_using):
            summary += f"- {program}\n"
        
        summary += f"\n**Usage Characteristics**:\n"
        summary += f"- Total References: {total_references}\n"
        summary += f"- Program Distribution: {len(programs_using)} programs\n"
        summary += f"- Average References per Program: {total_references / len(programs_using):.1f}\n\n"
        
        summary += "### Recommendations\n\n"
        if len(programs_using) > 5:
            summary += "- High usage indicates critical component - ensure robust testing for any changes\n"
            summary += "- Consider impact analysis before modifications\n"
            summary += "- Implement comprehensive monitoring and alerts\n"
        else:
            summary += "- Moderate usage allows for easier change management\n"
            summary += "- Standard testing procedures should be sufficient\n"
            summary += "- Regular reviews recommended to ensure continued relevance\n"
        
        return summary

    async def _analyze_lifecycle_patterns_api(self, chunks: List[tuple], component_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze lifecycle patterns for a component"""
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
        
        try:
            response_text = await self._call_api_for_analysis(full_prompt, max_tokens=600)
            
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Failed to parse lifecycle analysis: {e}")
        
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
        """✅ API-BASED: Generate comprehensive optimization recommendations"""
        try:
            # Get program analysis
            analysis_result = await self.analyze_program(program_name)
            
            if "error" in analysis_result:
                return analysis_result
            
            # Generate detailed optimization recommendations with API
            optimization_report = await self._generate_optimization_recommendations_api(analysis_result)
            
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
    
    async def _generate_optimization_recommendations_api(self, analysis_result: Dict) -> Dict[str, Any]:
        """FIXED: Generate readable optimization recommendations"""
        
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
        
        prompt = f"""
        Generate comprehensive optimization recommendations for this COBOL program analysis:

        **Program Analysis Summary:**
        - Average Complexity Score: {summary['complexity_score']:.2f} out of 10
        - Total Code Sections: {summary['total_chunks']}
        - High Complexity Sections: {summary['high_complexity_chunks']}
        - Logic Patterns Identified: {summary['patterns_found']}
        - Business Rules Found: {summary['business_rules']}

        **Provide detailed optimization recommendations in these areas:**

        **Performance Optimization:**
        - How to improve program execution speed
        - Database and file I/O optimizations
        - Memory usage improvements

        **Maintainability Enhancement:**
        - Code organization improvements
        - Documentation and clarity enhancements
        - Modular design recommendations

        **Code Structure Optimization:**
        - Ways to reduce complexity
        - Better error handling implementation
        - Improved data flow design

        **Business Logic Refinement:**
        - Simplification opportunities
        - Rule consolidation possibilities
        - Process efficiency improvements

        **Testing and Quality Assurance:**
        - Testing strategy recommendations
        - Quality monitoring approaches
        - Change management best practices

        **Priority Actions:**
        - Most critical improvements to implement first
        - Quick wins for immediate impact

        Write as a comprehensive optimization strategy document.
        Use clear headings and actionable recommendations.
        """
        
        try:
            response_text = await self._call_api_for_readable_analysis(prompt, max_tokens=1000)
            
            return {
                "optimization_strategy": response_text,
                "priority_level": "high" if summary['complexity_score'] > 7 else "medium",
                "recommendation_type": "comprehensive_optimization",
                "analysis_summary": summary
            }
        except Exception as e:
            self.logger.warning(f"Failed to generate optimization recommendations: {e}")
            return self._generate_fallback_optimization_recommendations(summary)
        
    def _generate_fallback_optimization_recommendations(self, summary: Dict) -> Dict[str, Any]:
        """Generate fallback optimization recommendations"""
        recommendations = "## Optimization Recommendations\n\n"
        
        complexity = summary['complexity_score']
        high_complexity = summary['high_complexity_chunks']
        
        recommendations += "### Priority Actions\n\n"
        
        if complexity > 7:
            recommendations += "**High Priority - Complexity Reduction:**\n"
            recommendations += f"- Address {high_complexity} high-complexity code sections immediately\n"
            recommendations += "- Break down large procedural blocks into smaller, manageable functions\n"
            recommendations += "- Implement comprehensive error handling throughout the program\n\n"
        elif complexity > 4:
            recommendations += "**Medium Priority - Maintenance Enhancement:**\n"
            recommendations += "- Review and optimize moderate complexity sections\n"
            recommendations += "- Improve code documentation and inline comments\n"
            recommendations += "- Standardize naming conventions across the program\n\n"
        else:
            recommendations += "**Low Priority - Continuous Improvement:**\n"
            recommendations += "- Maintain current code quality standards\n"
            recommendations += "- Regular code reviews and documentation updates\n"
            recommendations += "- Monitor for performance optimization opportunities\n\n"
        
        recommendations += "### Performance Optimization\n\n"
        recommendations += "- Review file I/O operations for efficiency improvements\n"
        recommendations += "- Optimize database access patterns and query performance\n"
        recommendations += "- Consider memory usage optimization for large data processing\n\n"
        
        recommendations += "### Maintainability Improvements\n\n"
        recommendations += "- Enhance error handling and logging capabilities\n"
        recommendations += "- Improve code modularity and reusability\n"
        recommendations += "- Establish comprehensive testing procedures\n"
        
        return {
            "optimization_strategy": recommendations,
            "priority_level": "high" if complexity > 7 else "medium",
            "recommendation_type": "standard_optimization",
            "analysis_summary": summary
        }


    async def analyze_code_quality(self, program_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Analyze overall code quality metrics"""
        try:
            chunks = await self._get_program_chunks(program_name)
            
            if not chunks:
                return self._add_processing_info({"error": f"Program {program_name} not found"})
            
            quality_metrics = await self._calculate_quality_metrics_api(chunks)
            
            result = {
                "program_name": program_name,
                "quality_metrics": quality_metrics,
                "analysis_type": "code_quality"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _calculate_quality_metrics_api(self, chunks: List[tuple]) -> Dict[str, Any]:
        """✅ API-BASED: Calculate comprehensive code quality metrics"""
        
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
        """✅ API-BASED: Export comprehensive analysis report in specified format"""
        try:
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
                "generated_by": "LogicAnalyzerAgent_API"
            }
            
            return self._add_processing_info(comprehensive_report)
            
        except Exception as e:
            self.logger.error(f"Report export failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    async def compare_programs(self, program1: str, program2: str) -> Dict[str, Any]:
        """✅ API-BASED: Compare logic patterns between two programs"""
        try:
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
        """✅ API-BASED: Batch analyze multiple programs"""
        try:
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
        """✅ API-BASED: Stream logic analysis results as they're processed"""
        try:
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
                # Analyze chunk with API
                chunk_analysis = await self._analyze_chunk_logic_api(chunk)
                chunk_analyses.append(chunk_analysis)
                
                # Yield progress update
                progress = ((i + 1) / len(chunks)) * 100
                yield self._add_processing_info({
                    "status": "processing",
                    "progress": progress,
                    "current_chunk": chunk[2],  # chunk_id
                    "chunk_analysis": chunk_analysis
                })
            
            # Final analysis with API
            business_rules = await self._extract_business_rules_api(chunks)
            logic_patterns = await self._identify_logic_patterns_api(chunks)
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
    
    async def _analyze_chunk_logic_api(self, chunk: tuple) -> Dict[str, Any]:
        """✅ API-BASED: Analyze logic in a single chunk"""
        chunk_id, program_name, chunk_id_str, chunk_type, content, metadata_str = chunk
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        # Use API for detailed logic analysis
        logic_analysis = await self._llm_analyze_logic_api(content, chunk_type)
        
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
    
    async def _llm_analyze_logic_api(self, content: str, chunk_type: str) -> Dict[str, Any]:
        """✅ API-BASED: Use API to analyze logic in code chunk"""
        
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
        
        try:
            response_text = await self._call_api_for_analysis(full_prompt, max_tokens=800)
            self.logger.info(f"LLM Response received for Logic Analyzer {response_text}")
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Failed to parse API logic analysis: {e}")
        
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
    
    async def _extract_business_rules_api(self, chunks: List[tuple]) -> List[BusinessRule]:
        """✅ API-BASED: Extract business rules from program chunks"""
        business_rules = []
        rule_id_counter = 1
        
        for chunk in chunks:
            content = chunk[4]  # content field
            
            # Look for validation rules
            for match in self.business_rule_patterns['validation_rules'].finditer(content):
                rule = await self._analyze_business_rule_api(
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
                rule = await self._analyze_business_rule_api(
                    f"RULE_{rule_id_counter:03d}",
                    "calculation",
                    match.group(0),
                    content
                )
                if rule:
                    business_rules.append(rule)
                    rule_id_counter += 1
        
        return business_rules
    
    async def _analyze_business_rule_api(self, rule_id: str, rule_type: str, 
                                   rule_code: str, context: str) -> Optional[BusinessRule]:
        """✅ API-BASED: Analyze a specific business rule using API"""
        
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
        
        try:
            response_text = await self._call_api_for_analysis(full_prompt, max_tokens=400)
            
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
        except Exception as e:
            self.logger.warning(f"Failed to parse business rule analysis: {e}")
        
        return None
    
    async def _identify_logic_patterns_api(self, chunks: List[tuple]) -> List[LogicPattern]:
        """✅ API-BASED: Identify common logic patterns across chunks"""
        patterns = []
        pattern_id = 1
        
        for chunk in chunks:
            content = chunk[4]
            chunk_type = chunk[3]
            
            # Identify specific patterns using API
            identified_patterns = await self._llm_identify_patterns_api(content, chunk_type)
            
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
    
    async def _llm_identify_patterns_api(self, content: str, chunk_type: str) -> List[Dict[str, Any]]:
        """✅ API-BASED: Use API to identify logic patterns"""
        
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
        
        try:
            response_text = await self._call_api_for_analysis(full_prompt, max_tokens=600)
            
            if '[' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                return json.loads(response_text[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Failed to parse pattern identification: {e}")
        
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
    
    def _normalize_component_name(self, component_name) -> str:
        """FIXED: Normalize component name to handle tuples and other formats"""
        try:
            # If it's already a string, return as-is
            if isinstance(component_name, str):
                return component_name.strip()
            
            # If it's a tuple or list, take the first element
            if isinstance(component_name, (tuple, list)) and len(component_name) > 0:
                first_element = component_name[0]
                if isinstance(first_element, str):
                    self.logger.info(f"🔧 Normalized tuple/list component name: {component_name} -> {first_element}")
                    return first_element.strip()
            
            # If it's something else, convert to string
            normalized = str(component_name).strip()
            if normalized != str(component_name):
                self.logger.info(f"🔧 Converted component name to string: {component_name} -> {normalized}")
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"❌ Failed to normalize component name {component_name}: {e}")
            return str(component_name) if component_name else "UNKNOWN"


    async def analyze_complete_program_flow(self, program_name: str) -> Dict[str, Any]:
        """✅ ENHANCED: Complete program flow analysis using new relationship tables"""
        program_name = self._normalize_component_name(program_name)
        try:
            # Get program relationships
            program_relationships = await self._get_program_relationships(program_name)
            
            # Get copybook dependencies
            copybook_relationships = await self._get_copybook_relationships(program_name)
            
            # Get file access patterns
            file_access_patterns = await self._get_file_access_relationships(program_name)
            
            # Get field cross-references
            field_cross_references = await self._get_field_cross_references(program_name)
            
            # Analyze impact relationships
            impact_analysis = await self._get_impact_analysis(program_name)
            
            # Generate comprehensive flow analysis with API
            flow_analysis = await self._generate_complete_flow_analysis_api(
                program_name, program_relationships, copybook_relationships,
                file_access_patterns, field_cross_references, impact_analysis
            )
            
            result = {
                "program_name": program_name,
                "program_relationships": program_relationships,
                "copybook_relationships": copybook_relationships,
                "file_access_patterns": file_access_patterns,
                "field_cross_references": field_cross_references,
                "impact_analysis": impact_analysis,
                "complete_flow_analysis": flow_analysis,
                "analysis_type": "complete_program_flow",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Complete program flow analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
        
    async def _generate_complete_flow_analysis_api(self, program_name: str, 
                                             program_relationships: Dict,
                                             copybook_relationships: List,
                                             file_access_patterns: List,
                                             field_cross_references: List,
                                             impact_analysis: Dict) -> str:
        """Generate comprehensive flow analysis using API"""
        
        flow_summary = {
            "program_name": program_name,
            "outbound_calls": len(program_relationships.get("outbound_calls", [])),
            "inbound_calls": len(program_relationships.get("inbound_calls", [])),
            "copybooks_used": len(copybook_relationships),
            "files_accessed": len(file_access_patterns),
            "fields_defined": len(field_cross_references),
            "impact_count": len(impact_analysis.get("program_impacts", [])),
            "dependency_count": len(impact_analysis.get("program_dependencies", []))
        }
        
        prompt = f"""
        Generate a comprehensive program flow analysis for: {program_name}
        
        Flow Summary:
        - Calls {flow_summary['outbound_calls']} other programs
        - Called by {flow_summary['inbound_calls']} programs
        - Uses {flow_summary['copybooks_used']} copybooks
        - Accesses {flow_summary['files_accessed']} files
        - Defines {flow_summary['fields_defined']} fields
        - Impacts {flow_summary['impact_count']} components
        - Has {flow_summary['dependency_count']} dependencies
        
        Called Programs: {[call['called_program'] for call in program_relationships.get('outbound_calls', [])]}
        Files Accessed: {[file['file_name'] for file in file_access_patterns]}
        
        Provide a detailed analysis covering:
        
        **Program Flow Overview:**
        - Main business process this program implements
        - Position in the overall system workflow
        - Critical business functions performed
        
        **Call Flow Analysis:**
        - Program invocation patterns and sequences
        - Conditional vs unconditional calls
        - Parameter passing mechanisms
        
        **Data Flow Analysis:**
        - Input data sources and formats
        - Data transformations and processing logic
        - Output data destinations and formats
        - Field-level data lineage
        
        **Integration Points:**
        - External system interfaces
        - File system interactions
        - Database operations
        
        **Impact Assessment:**
        - Components affected by changes to this program
        - Dependencies that could impact this program
        - Risk assessment for modifications
        
        **Performance Characteristics:**
        - Processing complexity indicators
        - Resource utilization patterns
        - Optimization opportunities
        
        Write as a comprehensive technical analysis suitable for both development and business teams.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1200)
        except Exception as e:
            self.logger.error(f"Flow analysis generation failed: {e}")
            return self._generate_fallback_flow_analysis(program_name, flow_summary)


    async def _get_program_relationships(self, program_name: str) -> Dict[str, Any]:
        """FIXED: Get program relationships with column existence check"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Check what columns exist in program_relationships table
            cursor.execute("PRAGMA table_info(program_relationships)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            self.logger.debug(f"🔍 Available columns in program_relationships: {column_names}")
            
            # Build query based on available columns
            base_columns = ["calling_program", "called_program", "call_type", "call_location", "line_number", "call_statement"]
            available_base_columns = [col for col in base_columns if col in column_names]
            
            # Add optional columns if they exist
            optional_columns = []
            if "parameters" in column_names:
                optional_columns.append("parameters")
            if "conditional_call" in column_names:
                optional_columns.append("conditional_call")
            
            all_columns = available_base_columns + optional_columns
            columns_str = ", ".join(all_columns)
            
            # FIXED: Dynamic query based on available columns
            cursor.execute(f"""
                SELECT {columns_str}
                FROM program_relationships
                WHERE calling_program = ?
                ORDER BY line_number
            """, (str(program_name),))
            
            outbound_calls = cursor.fetchall()
            
            cursor.execute(f"""
                SELECT {columns_str}
                FROM program_relationships
                WHERE called_program = ?
                ORDER BY calling_program, line_number
            """, (str(program_name),))
            
            inbound_calls = cursor.fetchall()
            conn.close()
            
            # FIXED: Build result with available data
            def build_call_info(row):
                call_info = {}
                for i, col in enumerate(all_columns):
                    if i < len(row):
                        if col == "parameters" and row[i]:
                            try:
                                call_info[col] = json.loads(row[i])
                            except:
                                call_info[col] = []
                        elif col == "conditional_call":
                            call_info[col] = bool(row[i]) if row[i] is not None else False
                        else:
                            call_info[col] = row[i]
                    else:
                        # Default values for missing columns
                        if col == "parameters":
                            call_info[col] = []
                        elif col == "conditional_call":
                            call_info[col] = False
                        else:
                            call_info[col] = None
                return call_info
            
            result = {
                "outbound_calls": [build_call_info(row) for row in outbound_calls],
                "inbound_calls": [build_call_info(row) for row in inbound_calls]
            }
            
            self.logger.info(f"📞 Found {len(outbound_calls)} outbound and {len(inbound_calls)} inbound calls for {program_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get program relationships: {e}")
            return {"outbound_calls": [], "inbound_calls": []}
            
    async def _get_copybook_relationships(self, program_name: str) -> List[Dict[str, Any]]:
        """FIXED: Get copybook relationships with column existence check"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check available columns
            cursor.execute("PRAGMA table_info(copybook_relationships)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            base_columns = ["program_name", "copybook_name", "copy_location", "line_number", "copy_statement"]
            available_base_columns = [col for col in base_columns if col in column_names]
            
            optional_columns = []
            if "replacing_clause" in column_names:
                optional_columns.append("replacing_clause")
            if "usage_context" in column_names:
                optional_columns.append("usage_context")
            
            all_columns = available_base_columns + optional_columns
            columns_str = ", ".join(all_columns)
            
            cursor.execute(f"""
                SELECT {columns_str}
                FROM copybook_relationships
                WHERE program_name = ?
                ORDER BY line_number
            """, (str(program_name),))
            
            relationships = cursor.fetchall()
            conn.close()
            
            # Build result with available data
            result = []
            for row in relationships:
                rel_info = {}
                for i, col in enumerate(all_columns):
                    if i < len(row):
                        rel_info[col] = row[i]
                    else:
                        rel_info[col] = None
                result.append(rel_info)
            
            self.logger.info(f"📋 Found {len(relationships)} copybook relationships for {program_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get copybook relationships: {e}")
            return []
            
    async def _get_file_access_relationships(self, program_name: str) -> List[Dict[str, Any]]:
        """FIXED: Get file access relationships with column existence check"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check available columns
            cursor.execute("PRAGMA table_info(file_access_relationships)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            # Use different column names based on what's available
            if "logical_file_name" in column_names:
                file_name_col = "logical_file_name"
            elif "file_name" in column_names:
                file_name_col = "file_name"
            else:
                self.logger.warning("⚠️ No recognizable file name column found")
                return []
            
            base_columns = ["program_name", file_name_col, "access_type", "access_mode", "line_number"]
            available_base_columns = [col for col in base_columns if col in column_names]
            
            optional_columns = []
            if "physical_file_name" in column_names:
                optional_columns.append("physical_file_name")
            elif "physical_file" in column_names:
                optional_columns.append("physical_file")
            
            if "record_format" in column_names:
                optional_columns.append("record_format")
            if "access_statement" in column_names:
                optional_columns.append("access_statement")
            elif "access_location" in column_names:
                optional_columns.append("access_location")
            
            all_columns = available_base_columns + optional_columns
            columns_str = ", ".join(all_columns)
            
            cursor.execute(f"""
                SELECT {columns_str}
                FROM file_access_relationships
                WHERE program_name = ?
                ORDER BY line_number
            """, (str(program_name),))
            
            relationships = cursor.fetchall()
            conn.close()
            
            # Build result with standardized column names
            result = []
            for row in relationships:
                rel_info = {
                    "program_name": row[0] if len(row) > 0 else None,
                    "file_name": row[1] if len(row) > 1 else None,
                    "physical_file": None,
                    "access_type": None,
                    "access_mode": None,
                    "record_format": None,
                    "access_location": None,
                    "line_number": None
                }
                
                # Map columns based on position and what's available
                for i, col in enumerate(all_columns):
                    if i < len(row) and row[i] is not None:
                        if col in ["physical_file_name", "physical_file"]:
                            rel_info["physical_file"] = row[i]
                        elif col in ["access_statement", "access_location"]:
                            rel_info["access_location"] = row[i]
                        elif col in rel_info:
                            rel_info[col] = row[i]
                
                result.append(rel_info)
            
            self.logger.info(f"📁 Found {len(relationships)} file access relationships for {program_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get file access relationships: {e}")
            return []
    
    async def _get_field_cross_references(self, program_name: str) -> List[Dict[str, Any]]:
        """FIXED: Get field cross-references with table existence check"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if field_cross_reference table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='field_cross_reference'
            """)
            
            if not cursor.fetchone():
                self.logger.warning("⚠️ field_cross_reference table does not exist")
                conn.close()
                return []
            
            cursor.execute("""
                SELECT field_name, qualified_name, source_type, source_name,
                    definition_location, data_type, picture_clause, usage_clause,
                    level_number, parent_field, occurs_info, business_domain
                FROM field_cross_reference
                WHERE source_name = ? OR source_name IN (
                    SELECT copybook_name FROM copybook_relationships WHERE program_name = ?
                )
                ORDER BY level_number, field_name
            """, (str(program_name), str(program_name)))
            
            relationships = cursor.fetchall()
            conn.close()
            
            result = []
            for row in relationships:
                if len(row) >= 12:
                    rel_info = {
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
                    }
                    result.append(rel_info)
            
            self.logger.info(f"🔤 Found {len(result)} field cross-references for {program_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get field cross-references: {e}")
            return []

    async def _get_impact_analysis(self, program_name: str) -> Dict[str, Any]:
        """Get impact analysis for the program"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get what this program impacts
            cursor.execute("""
                SELECT source_artifact, source_type, dependent_artifact, dependent_type,
                    relationship_type, impact_level, change_propagation
                FROM impact_analysis
                WHERE source_artifact = ?
                ORDER BY impact_level DESC
            """, (program_name,))
            
            impacts = cursor.fetchall()
            
            # Get what impacts this program
            cursor.execute("""
                SELECT source_artifact, source_type, dependent_artifact, dependent_type,
                    relationship_type, impact_level, change_propagation
                FROM impact_analysis
                WHERE dependent_artifact = ?
                ORDER BY impact_level DESC
            """, (program_name,))
            
            dependencies = cursor.fetchall()
            conn.close()
            
            return {
                "program_impacts": [
                    {
                        "source_artifact": row[0],
                        "source_type": row[1],
                        "dependent_artifact": row[2],
                        "dependent_type": row[3],
                        "relationship_type": row[4],
                        "impact_level": row[5],
                        "change_propagation": row[6]
                    } for row in impacts
                ],
                "program_dependencies": [
                    {
                        "source_artifact": row[0],
                        "source_type": row[1],
                        "dependent_artifact": row[2],
                        "dependent_type": row[3],
                        "relationship_type": row[4],
                        "impact_level": row[5],
                        "change_propagation": row[6]
                    } for row in dependencies
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get impact analysis: {e}")
            return {"program_impacts": [], "program_dependencies": []}

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
        """✅ API-BASED: Specialized analysis focused on business logic"""
        try:
            chunks = await self._get_program_chunks(program_name)
            
            if not chunks:
                return self._add_processing_info({"error": f"Program {program_name} not found"})
            
            # Focus on business logic extraction with API
            business_logic = await self._extract_comprehensive_business_logic_api(chunks)
            
            result = {
                "program_name": program_name,
                "business_logic": business_logic,
                "analysis_type": "business_logic_focused"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Business logic analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _call_api_for_readable_analysis(self, prompt: str, max_tokens: int = None, 
                                         context: str = "logic analysis") -> str:
        """Enhanced API call that ensures readable responses for logic analysis"""
        try:
            # Enhance prompt to ensure readable output
            enhanced_prompt = f"""
            {prompt}
            
            IMPORTANT: Provide your response as clear, readable business prose. 
            Do not use JSON format unless specifically requested.
            Write in professional analysis style with proper sentences and paragraphs.
            Focus on business impact and practical insights.
            """
            
            params = self.api_params.copy()
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            result = await self.coordinator.call_model_api(
                prompt=enhanced_prompt,
                params=params,
                preferred_gpu_id=self.gpu_id
            )
            
            # Extract and clean the response
            if isinstance(result, dict):
                response_text = result.get('text', result.get('response', ''))
            else:
                response_text = str(result)
            
            # Clean up the response
            cleaned_response = self._clean_logic_response(response_text, context)
            
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            return f"Logic analysis failed for {context}. Please check system status."

    def _clean_logic_response(self, response_text: str, context: str) -> str:
        """Clean API response to ensure readable logic analysis format"""
        if not response_text or not response_text.strip():
            return f"No {context} information available."
        
        cleaned = response_text.strip()
        
        # Handle JSON responses - extract readable content
        if cleaned.startswith('{') and cleaned.endswith('}'):
            try:
                import json
                json_data = json.loads(cleaned)
                readable_parts = []
                
                # Extract meaningful text from common JSON fields
                text_fields = ['summary', 'analysis', 'description', 'business_logic', 
                            'main_operations', 'decision_points', 'optimizations']
                
                for field in text_fields:
                    if field in json_data:
                        value = json_data[field]
                        if isinstance(value, str) and len(value) > 10:
                            readable_parts.append(value)
                        elif isinstance(value, list):
                            if field == 'main_operations':
                                readable_parts.append(f"Main operations include: {', '.join(str(v) for v in value[:5])}")
                            elif field == 'decision_points':
                                readable_parts.append(f"Key decision points: {', '.join(str(v) for v in value[:3])}")
                            elif field == 'optimizations':
                                readable_parts.append(f"Optimization opportunities: {', '.join(str(v) for v in value[:3])}")
                
                if readable_parts:
                    cleaned = '. '.join(readable_parts) + '.'
                else:
                    cleaned = f"Logic analysis completed for {context} with structured findings available."
                    
            except json.JSONDecodeError:
                pass
        
        # Remove code blocks if present
        if '```' in cleaned:
            lines = cleaned.split('\n')
            clean_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    clean_lines.append(line)
            cleaned = '\n'.join(clean_lines)
        
        # Ensure minimum meaningful content
        if len(cleaned.strip()) < 20:
            cleaned = f"Logic analysis completed for {context}. Component shows standard programming patterns and business logic implementation."
        
        return cleaned.strip()


    async def _extract_comprehensive_business_logic_api(self, chunks: List[tuple]) -> Dict[str, Any]:
        """FIXED: Extract comprehensive business logic with readable output"""
        
        # Combine content with length management
        all_content = '\n'.join([chunk[4] for chunk in chunks])
        
        # Create base prompt for readable business analysis
        base_prompt = f"""
        Analyze this complete COBOL program and provide a comprehensive business logic summary:

        Program Content (truncated for analysis):
        {all_content[:1500]}...

        Provide a detailed business analysis covering:

        1. **Primary Business Purpose**: What business process does this program support?
        
        2. **Key Business Rules**: What are the main business rules and validations implemented?
        
        3. **Data Processing Workflow**: How does data flow through the business process?
        
        4. **Business Calculations**: What calculations or transformations support business operations?
        
        5. **Decision Logic**: What business decisions are automated by this program?
        
        6. **Business Impact**: How critical is this program to business operations?

        Write as a comprehensive business analysis report in clear prose.
        Do not use JSON format. Use proper headings and detailed explanations.
        """
        
        try:
            response_text = await self._call_api_for_readable_analysis(base_prompt, max_tokens=1200)
            
            return {
                "business_summary": response_text,
                "extraction_method": "enhanced_api_analysis",
                "confidence": "high",
                "analysis_type": "comprehensive_business_logic"
            }
        except Exception as e:
            self.logger.error(f"Failed to extract business logic: {e}")
            return {
                "business_summary": self._generate_fallback_business_summary(chunks),
                "extraction_method": "fallback_analysis",
                "confidence": "medium",
                "error": str(e)
            }
        
    def _generate_fallback_business_summary(self, chunks: List[tuple]) -> str:
        """Generate fallback business summary when API fails"""
        summary = "## Business Logic Analysis\n\n"
        
        # Analyze chunk types
        chunk_types = {}
        total_lines = 0
        
        for chunk in chunks:
            chunk_type = chunk[3]  # chunk_type field
            content = chunk[4]     # content field
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_lines += len(content.split('\n'))
        
        summary += f"**Program Structure**: This program contains {len(chunks)} code sections "
        summary += f"with {total_lines:,} total lines of code.\n\n"
        
        summary += "**Business Components Identified**:\n"
        for chunk_type, count in chunk_types.items():
            if chunk_type in ['data_division', 'working_storage']:
                summary += f"- Data Structures: {count} sections defining business data elements\n"
            elif chunk_type == 'procedure_division':
                summary += f"- Business Logic: {count} sections implementing business processes\n"
            elif chunk_type == 'file_section':
                summary += f"- File Operations: {count} sections handling business data files\n"
            else:
                summary += f"- {chunk_type.title()}: {count} sections\n"
        
        summary += "\n**Business Process Analysis**: "
        summary += "This program implements standard mainframe business processing patterns "
        summary += "including data validation, transformation, and output generation. "
        summary += "The program follows established COBOL business logic conventions "
        summary += "and appears to be part of a larger business system.\n"
        
        return summary

    async def analyze_full_lifecycle(self, component_name: str, component_type: str) -> Dict[str, Any]:
        """FIXED: Analyze complete lifecycle with readable documentation"""
        try:
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
                    "lifecycle_analysis": "No usage found for this component",
                    "usage_count": 0,
                    "recommendation": "Verify component name and ensure data has been processed"
                })
            
            # Generate comprehensive lifecycle analysis
            lifecycle_summary = await self._generate_lifecycle_summary_api(
                component_name, component_type, related_chunks
            )
            
            result = {
                "component_name": component_name,
                "component_type": component_type,
                "usage_count": len(related_chunks),
                "lifecycle_summary": lifecycle_summary,
                "programs_involved": list(set(chunk[0] for chunk in related_chunks)),
                "analysis_completeness": "comprehensive" if len(related_chunks) > 5 else "basic"
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Lifecycle analysis failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
        
    async def _generate_lifecycle_summary_api(self, component_name: str, component_type: str, 
                                        related_chunks: List[tuple]) -> str:
        """Generate comprehensive lifecycle summary with readable output"""
        
        # Analyze the chunks to understand usage patterns
        programs_using = set(chunk[0] for chunk in related_chunks)
        chunk_types = [chunk[1] for chunk in related_chunks]
        
        prompt = f"""
        Generate a comprehensive lifecycle analysis for {component_type} "{component_name}":

        **Usage Analysis:**
        - Found in {len(programs_using)} programs
        - Total references: {len(related_chunks)}
        - Programs: {', '.join(list(programs_using)[:5])}
        - Chunk types: {', '.join(set(chunk_types))}

        **Provide a detailed lifecycle analysis covering:**

        **Component Overview:**
        - What this component represents in the business context
        - Its role in the overall system architecture

        **Lifecycle Stages:**
        - Where and how the component is created or initialized
        - How it's used throughout different business processes
        - Where and when it gets modified or updated
        - End-of-life or archival considerations

        **Usage Patterns:**
        - Common ways this component is accessed
        - Frequency and timing of usage
        - Integration points with other components

        **Dependencies and Relationships:**
        - Other components that depend on this one
        - Components this one depends on
        - Critical relationship mapping

        **Business Impact:**
        - Importance to business operations
        - Risk assessment if component fails
        - Recommendations for maintenance and monitoring

        Write as a comprehensive technical documentation suitable for both development and business teams.
        Use clear section headings and detailed explanations.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1200)
        except Exception as e:
            self.logger.error(f"Lifecycle summary generation failed: {e}")
            return self._generate_fallback_lifecycle_summary(component_name, component_type, programs_using, len(related_chunks))


    # ==================== Cleanup and Context Management ====================
    
    def cleanup(self):
        """Cleanup method for API-based agent"""
        self.logger.info("🧹 Cleaning up API-based Logic Analyzer agent...")
        
        # Clear any cached data
        # No GPU resources to free up in API mode
        
        self.logger.info("✅ API-based Logic Analyzer cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __repr__(self):
        return (f"LogicAnalyzerAgent("
                f"api_based=True, "
                f"coordinator={type(self.coordinator).__name__}, "
                f"gpu_id={self.gpu_id})")

# ==================== Backwards Compatibility Functions ====================

async def quick_program_analysis_api(program_name: str, coordinator) -> Dict[str, Any]:
    """Quick program analysis using API-based agent"""
    agent = LogicAnalyzerAgent(coordinator)
    try:
        return await agent.analyze_program(program_name)
    finally:
        agent.cleanup()

async def quick_business_logic_analysis_api(program_name: str, coordinator) -> Dict[str, Any]:
    """Quick business logic analysis using API-based agent"""
    agent = LogicAnalyzerAgent(coordinator)
    try:
        return await agent.analyze_business_logic(program_name)
    finally:
        agent.cleanup()

async def quick_code_quality_analysis_api(program_name: str, coordinator) -> Dict[str, Any]:
    """Quick code quality analysis using API-based agent"""
    agent = LogicAnalyzerAgent(coordinator)
    try:
        return await agent.analyze_code_quality(program_name)
    finally:
        agent.cleanup()

async def quick_optimization_report_api(program_name: str, coordinator) -> Dict[str, Any]:
    """Quick optimization report using API-based agent"""
    agent = LogicAnalyzerAgent(coordinator)
    try:
        return await agent.generate_optimization_report(program_name)
    finally:
        agent.cleanup()

# ==================== Example Usage ====================

async def example_api_logic_usage():
    """Example of how to use the API-based logic analyzer"""
    
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
        # Create API-based logic analyzer
        logic_agent = LogicAnalyzerAgent(coordinator)
        
        # Analyze program logic (API calls instead of GPU loading)
        program_analysis = await logic_agent.analyze_program("CUSTOMER-PROCESSOR")
        print(f"Program analysis status: {program_analysis.get('status', 'completed')}")
        
        # Generate optimization report (API calls)
        optimization_report = await logic_agent.generate_optimization_report("CUSTOMER-PROCESSOR")
        print(f"Optimization report generated: {len(optimization_report.get('optimization_report', {}))}")
        
        # Analyze code quality (API calls)
        quality_analysis = await logic_agent.analyze_code_quality("CUSTOMER-PROCESSOR")
        print(f"Quality score: {quality_analysis.get('quality_metrics', {}).get('quality_score', 'N/A')}")
        
        # Compare programs (API calls)
        comparison = await logic_agent.compare_programs("PROGRAM-A", "PROGRAM-B")
        print(f"Program comparison completed: {comparison.get('analysis_type')}")
        
        # Batch analyze programs (API calls)
        batch_analysis = await logic_agent.batch_analyze_programs(["PROG1", "PROG2", "PROG3"])
        print(f"Batch analysis completed: {batch_analysis.get('programs_analyzed', 0)} programs")
        
        # Stream analysis results (API calls)
        print("Streaming analysis results:")
        async for result in logic_agent.stream_logic_analysis("CUSTOMER-PROCESSOR"):
            if result.get("status") == "processing":
                print(f"Progress: {result.get('progress', 0):.1f}%")
            elif result.get("status") == "completed":
                print(f"Analysis completed with {len(result.get('business_rules', []))} business rules found")
        
        # Business logic focus (API calls)
        business_logic = await logic_agent.analyze_business_logic("CUSTOMER-PROCESSOR")
        print(f"Business logic extracted: {len(business_logic.get('business_logic', {}).get('summary', ''))}")
        
        # Export comprehensive report (API calls)
        full_report = await logic_agent.export_analysis_report("CUSTOMER-PROCESSOR", format="json")
        print(f"Full report generated: {full_report.get('report_format')}")
        
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_api_logic_usage())