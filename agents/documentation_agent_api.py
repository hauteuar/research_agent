# agents/documentation_agent.py
"""
API-BASED Agent 6: Documentation Generator
Generates comprehensive technical documentation for mainframe systems
Now uses HTTP API calls instead of direct GPU model loading
"""

import asyncio
import uuid
import sqlite3
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime as dt
import markdown
from jinja2 import Template

from agents.base_agent_api import BaseOpulenceAgent

@dataclass
class DocumentSection:
    """Represents a documentation section"""
    title: str
    content: str
    section_type: str
    order: int
    subsections: List['DocumentSection'] = None

class DocumentationAgent(BaseOpulenceAgent):
    """API-BASED: Agent for generating technical documentation"""
    
    def __init__(self, coordinator, llm_engine=None, db_path: str = "opulence_data.db", gpu_id: int = 0):
        super().__init__(coordinator, "documentation", db_path, gpu_id)
        
        # Store coordinator reference for API calls
        self.coordinator = coordinator
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(__name__)
        
        # API-specific settings
        self.api_params = {
            "max_tokens": 800,
            "temperature": 0.2,
            "top_p": 0.9
        }
        
        # Documentation templates
        self.templates = self._load_templates()
        
        # Documentation sections
        self.section_types = [
            "overview",
            "architecture", 
            "data_flow",
            "business_logic",
            "technical_specs",
            "dependencies",
            "maintenance",
            "troubleshooting"
        ]

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
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result['gpu_used'] = self.gpu_id
            result['agent_type'] = 'documentation'
            result['api_based'] = True
            result['coordinator_type'] = getattr(self.coordinator, 'stats', {}).get('coordinator_type', 'api_based')
        return result
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load documentation templates"""
        templates = {}
        
        # Program documentation template
        templates['program'] = Template("""
# {{ program_name }} - Technical Documentation

## Overview
{{ overview }}

## Architecture
{{ architecture }}

## Business Logic
{{ business_logic }}

## Data Flow
{{ data_flow }}

## Technical Specifications
{{ technical_specs }}

## Dependencies
{{ dependencies }}

## Maintenance Notes
{{ maintenance }}

## Troubleshooting
{{ troubleshooting }}

---
*Generated on {{ generation_date }} by Opulence Documentation Agent (API-Based)*
        """.strip())
        
        # System documentation template
        templates['system'] = Template("""
# {{ system_name }} - System Documentation

## Executive Summary
{{ executive_summary }}

## System Architecture
{{ system_architecture }}

## Component Overview
{{ component_overview }}

## Data Architecture
{{ data_architecture }}

## Integration Points
{{ integration_points }}

## Operational Procedures
{{ operational_procedures }}

## Security Considerations
{{ security }}

## Performance Characteristics
{{ performance }}

## Maintenance and Support
{{ maintenance_support }}

---
*Generated on {{ generation_date }} by Opulence Documentation Agent (API-Based)*
        """.strip())
        
        # Field lineage template
        templates['lineage'] = Template("""
# {{ field_name }} - Field Lineage Documentation

## Field Overview
{{ field_overview }}

## Lifecycle Analysis
{{ lifecycle }}

## Usage Patterns
{{ usage_patterns }}

## Dependencies
{{ dependencies }}

## Quality Metrics
{{ quality_metrics }}

## Recommendations
{{ recommendations }}

---
*Generated on {{ generation_date }} by Opulence Documentation Agent (API-Based)*
        """.strip())
        
        return templates
    
    async def generate_program_documentation(self, program_name: str, 
                                           format_type: str = "markdown") -> Dict[str, Any]:
        """✅ API-BASED: Generate comprehensive documentation for a program"""
        try:
            # Gather program information
            program_info = await self._gather_program_info(program_name)
            
            if not program_info:
                return self._add_processing_info({"error": f"Program {program_name} not found"})
            
            # Generate each documentation section using API
            sections = {}
            
            sections['overview'] = await self._generate_overview_api(program_info)
            sections['architecture'] = await self._generate_architecture_api(program_info)
            sections['business_logic'] = await self._generate_business_logic_doc_api(program_info)
            sections['data_flow'] = await self._generate_data_flow_doc_api(program_info)
            sections['technical_specs'] = await self._generate_technical_specs(program_info)
            sections['dependencies'] = await self._generate_dependencies_doc(program_info)
            sections['maintenance'] = await self._generate_maintenance_doc_api(program_info)
            sections['troubleshooting'] = await self._generate_troubleshooting_doc_api(program_info)
            
            # Render documentation
            documentation = self.templates['program'].render(
                program_name=program_name,
                generation_date=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                **sections
            )
            
            # Convert to requested format
            if format_type == "html":
                documentation = markdown.markdown(documentation)
            elif format_type == "pdf":
                # For PDF generation, you'd integrate with a library like weasyprint
                documentation = {"content": documentation, "format": "markdown_for_pdf"}
            
            result = {
                "status": "success",
                "program_name": program_name,
                "documentation": documentation,
                "format": format_type,
                "sections_generated": len(sections),
                "generation_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed for {program_name}: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def generate_system_documentation(self, system_components: List[str], 
                                          system_name: str = "Mainframe System") -> Dict[str, Any]:
        """✅ API-BASED: Generate system-level documentation"""
        try:
            # Gather information for all components
            component_info = {}
            for component in system_components:
                component_info[component] = await self._gather_program_info(component)
            
            # Generate system-level sections using API
            sections = {}
            sections['executive_summary'] = await self._generate_executive_summary_api(component_info, system_name)
            sections['system_architecture'] = await self._generate_system_architecture_api(component_info)
            sections['component_overview'] = await self._generate_component_overview(component_info)
            sections['data_architecture'] = await self._generate_data_architecture_api(component_info)
            sections['integration_points'] = await self._generate_integration_points_api(component_info)
            sections['operational_procedures'] = await self._generate_operational_procedures_api(component_info)
            sections['security'] = await self._generate_security_doc_api(component_info)
            sections['performance'] = await self._generate_performance_doc_api(component_info)
            sections['maintenance_support'] = await self._generate_maintenance_support_doc_api(component_info)
            
            # Render documentation
            documentation = self.templates['system'].render(
                system_name=system_name,
                generation_date=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                **sections
            )
            
            result = {
                "status": "success",
                "system_name": system_name,
                "documentation": documentation,
                "components_analyzed": len(system_components),
                "generation_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"System documentation generation failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def generate_field_lineage_documentation(self, field_name: str) -> Dict[str, Any]:
        """✅ API-BASED: Generate documentation for field lineage"""
        try:
            # Gather field lineage information
            lineage_info = await self._gather_field_lineage_info(field_name)
            
            if not lineage_info:
                return self._add_processing_info({"error": f"Field {field_name} not found"})
            
            # Generate lineage documentation sections using API
            sections = {}
            sections['field_overview'] = await self._generate_field_overview(lineage_info)
            sections['lifecycle'] = await self._generate_field_lifecycle_doc_api(lineage_info)
            sections['usage_patterns'] = await self._generate_field_usage_patterns_api(lineage_info)
            sections['dependencies'] = await self._generate_field_dependencies(lineage_info)
            sections['quality_metrics'] = await self._generate_field_quality_metrics_api(lineage_info)
            sections['recommendations'] = await self._generate_field_recommendations_api(lineage_info)
            
            # Render documentation
            documentation = self.templates['lineage'].render(
                field_name=field_name,
                generation_date=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                **sections
            )
            
            result = {
                "status": "success",
                "field_name": field_name,
                "documentation": documentation,
                "generation_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Field lineage documentation failed for {field_name}: {str(e)}")
            return self._add_processing_info({"error": str(e)})
    
    async def _generate_field_overview(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field overview with readable output"""
        field_name = lineage_info['field_name']
        programs_count = len(lineage_info['programs_using'])
        operations_count = len(lineage_info['operations'])
        
        prompt = f"""
        Create a professional field overview for: {field_name}
        
        This field is used in {programs_count} programs and has {operations_count} different operations.
        Programs using this field: {', '.join(lineage_info['programs_using'][:5])}
        
        Write a clear overview that explains:
        - What this field represents in business terms
        - How it's used across the system
        - Its importance to business operations
        - Key characteristics and usage patterns
        
        Write as professional documentation prose, not JSON or bullet points.
        """
        
        return await self._call_api_for_readable_analysis(prompt, max_tokens=400, context="field overview")


    async def _generate_field_lifecycle_doc_api(self, lineage_info: Dict[str, Any]) -> str:
        """Generate readable field lifecycle documentation"""
        field_name = lineage_info['field_name'] 
        
        prompt = f"""
        Document the lifecycle of field {field_name} based on its usage patterns.
        
        Usage found in: {len(lineage_info['programs_using'])} programs
        Operations: {', '.join(lineage_info['operations'])}
        
        Write a comprehensive lifecycle description covering:
        - How the field is created and initialized
        - How it flows through different business processes
        - Where and how it gets updated or modified
        - Its role in the overall data architecture
        
        Write as flowing prose suitable for technical documentation.
        """
        
        return await self._call_api_for_readable_analysis(prompt, max_tokens=500, context="field lifecycle")

    async def _generate_field_usage_patterns_api(self, lineage_info: Dict[str, Any]) -> str:
        """Generate readable field usage patterns documentation"""
        field_name = lineage_info['field_name']
        
        prompt = f"""
        Analyze and document the usage patterns for field {field_name}.
        
        Found in {len(lineage_info['programs_using'])} programs with operations: {', '.join(lineage_info['operations'])}
        
        Describe:
        - Common ways this field is used across programs
        - Patterns in how it's accessed and modified
        - Business scenarios where it's most critical
        - Any notable usage characteristics or trends
        
        Write in clear, professional documentation style.
        """
        
        return await self._call_api_for_readable_analysis(prompt, max_tokens=400, context="usage patterns")

    async def _generate_field_dependencies(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field dependencies documentation"""
        field_name = lineage_info['field_name']
        programs = lineage_info['programs_using']
        
        deps_doc = f"## Field Dependencies\n\n"
        deps_doc += f"**Field:** {field_name}\n\n"
        
        deps_doc += "**Dependent Programs:**\n"
        for program in programs[:10]:
            deps_doc += f"- {program}\n"
        
        if len(programs) > 10:
            deps_doc += f"- ... and {len(programs) - 10} more programs\n"
        
        deps_doc += f"\n**Dependency Analysis:**\n"
        deps_doc += f"This field has {len(programs)} direct dependencies across the system. "
        deps_doc += "Changes to this field structure or data type could impact multiple programs "
        deps_doc += "and require coordinated testing and deployment.\n"
        
        return deps_doc

    async def _generate_field_quality_metrics_api(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field quality metrics documentation"""
        field_name = lineage_info['field_name']
        
        prompt = f"""
        Generate quality metrics analysis for field {field_name}.
        
        Usage: {len(lineage_info['programs_using'])} programs, {len(lineage_info['operations'])} operations
        
        Analyze and document:
        - Data quality considerations for this field
        - Consistency across different programs
        - Potential quality risks or concerns
        - Recommendations for quality monitoring
        
        Write as professional quality assessment documentation.
        """
        
        return await self._call_api_for_readable_analysis(prompt, max_tokens=400, context="quality metrics")

    async def _generate_field_recommendations_api(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field recommendations documentation"""
        field_name = lineage_info['field_name']
        
        prompt = f"""
        Generate recommendations for field {field_name} based on its usage analysis.
        
        Current usage: {len(lineage_info['programs_using'])} programs
        Operations: {', '.join(lineage_info['operations'][:5])}
        
        Provide recommendations for:
        - Best practices for using this field
        - Potential improvements or optimizations
        - Risk mitigation strategies
        - Maintenance considerations
        
        Write as actionable business recommendations.
        """
        
        return await self._call_api_for_readable_analysis(prompt, max_tokens=400, context="recommendations")

    async def _call_api_for_readable_analysis(self, prompt: str, max_tokens: int = None, 
                                         context: str = "documentation") -> str:
        """Make API call and ensure readable response for documentation"""
        try:
            # Add instruction for readable output
            enhanced_prompt = f"""
            {prompt}
            
            IMPORTANT: Respond with clear, readable prose. Do not use JSON format. 
            Write in professional documentation style with proper sentences and paragraphs.
            Focus on business value and practical information.
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
            
            # Clean up the response - remove JSON artifacts
            cleaned_response = self._clean_api_response(response_text, context)
            
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            return f"Documentation generation failed for {context}. Please check system status."

    def _clean_api_response(self, response_text: str, context: str) -> str:
        """Clean API response to ensure readable documentation format"""
        if not response_text or not response_text.strip():
            return f"No {context} information available."
        
        # Remove common JSON artifacts
        cleaned = response_text.strip()
        
        # Remove JSON wrapper if present
        if cleaned.startswith('{') and cleaned.endswith('}'):
            try:
                import json
                json_data = json.loads(cleaned)
                # Extract meaningful text from JSON
                if isinstance(json_data, dict):
                    text_parts = []
                    for key, value in json_data.items():
                        if isinstance(value, str) and len(value) > 10:
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.extend([str(item) for item in value if isinstance(item, str)])
                    
                    if text_parts:
                        cleaned = '. '.join(text_parts)
                    else:
                        cleaned = f"Analysis completed for {context}"
            except:
                pass
        
        # Remove markdown code blocks if present
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
        
        # Ensure minimum content
        if len(cleaned.strip()) < 20:
            cleaned = f"Documentation analysis completed for {context}. Please refer to the technical details for more information."
        
        return cleaned.strip()

    async def _gather_program_info(self, program_name: str) -> Dict[str, Any]:
        """Gather comprehensive information about a program"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get program chunks
        cursor.execute("""
            SELECT chunk_type, content, metadata
            FROM program_chunks 
            WHERE program_name = ?
            ORDER BY chunk_id
        """, (program_name,))
        
        chunks = cursor.fetchall()
        
        # Get field lineage information
        cursor.execute("""
            SELECT field_name, operation, paragraph, source_file
            FROM field_lineage 
            WHERE program_name = ?
        """, (program_name,))
        
        field_usage = cursor.fetchall()
        
        conn.close()
        
        if not chunks:
            return None
        
        # Organize information
        program_info = {
            "program_name": program_name,
            "chunks": chunks,
            "field_usage": field_usage,
            "total_chunks": len(chunks),
            "chunk_types": list(set(chunk[0] for chunk in chunks)),
            "all_content": '\n'.join([chunk[1] for chunk in chunks])
        }
        
        return program_info
    
    async def _gather_field_lineage_info(self, field_name: str) -> Dict[str, Any]:
        """Gather field lineage information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT program_name, paragraph, operation, source_file, last_used
            FROM field_lineage 
            WHERE field_name = ?
            ORDER BY last_used DESC
        """, (field_name,))
        
        lineage_records = cursor.fetchall()
        conn.close()
        
        if not lineage_records:
            return None
        
        return {
            "field_name": field_name,
            "lineage_records": lineage_records,
            "programs_using": list(set(record[0] for record in lineage_records)),
            "operations": list(set(record[2] for record in lineage_records))
        }
    
    async def _generate_overview_api(self, program_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate program overview section"""
        prompt = f"""
        Generate a comprehensive overview for this COBOL program:
        
        Program Name: {program_info['program_name']}
        Total Chunks: {program_info['total_chunks']}
        Chunk Types: {', '.join(program_info['chunk_types'])}
        
        Content Preview:
        {program_info['all_content'][:1500]}...
        
        Create an overview that includes:
        1. Purpose of the program
        2. Main functionality
        3. Business context
        4. Key characteristics
        
        Write in clear, professional documentation style.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Failed to generate overview: {e}")
            return f"Overview generation failed for {program_info['program_name']}"
    
    async def _generate_architecture_api(self, program_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate architecture section"""
        prompt = f"""
        Analyze the architecture of this COBOL program and document its structure:
        
        Program: {program_info['program_name']}
        Chunk Types: {', '.join(program_info['chunk_types'])}
        
        Code Structure:
        {program_info['all_content'][:1500]}...
        
        Document:
        1. Program structure and organization
        2. Major sections and their purposes
        3. Control flow and logic organization
        4. Design patterns used
        
        Format as technical architecture documentation.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=700)
        except Exception as e:
            self.logger.error(f"Failed to generate architecture: {e}")
            return f"Architecture documentation for {program_info['program_name']} - {len(program_info['chunk_types'])} sections identified."
    
    async def _generate_business_logic_doc_api(self, program_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate business logic documentation"""
        prompt = f"""
        Document the business logic implemented in this COBOL program:
        
        Program: {program_info['program_name']}
        
        Code Analysis:
        {program_info['all_content'][:2000]}...
        
        Document:
        1. Business processes implemented
        2. Business rules and validations
        3. Decision points and their business meaning
        4. Data processing workflows
        5. Business calculations
        
        Focus on business value and operational impact.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=800)
        except Exception as e:
            self.logger.error(f"Failed to generate business logic doc: {e}")
            return f"Business logic analysis for {program_info['program_name']} - processes {len(program_info['field_usage'])} data fields."
    
    async def _generate_data_flow_doc_api(self, program_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate data flow documentation"""
        # Extract file operations and data movements
        file_operations = []
        data_movements = []
        
        for chunk in program_info['chunks']:
            content = chunk[1]
            
            # Find file operations
            file_ops = re.findall(r'\b(READ|WRITE|REWRITE|DELETE|OPEN|CLOSE)\s+([A-Z][A-Z0-9-]*)', 
                                content, re.IGNORECASE)
            file_operations.extend(file_ops)
            
            # Find data movements
            moves = re.findall(r'MOVE\s+([^.]+?)\s+TO\s+([^.]+)', content, re.IGNORECASE)
            data_movements.extend(moves)
        
        prompt = f"""
        Document the data flow for this COBOL program:
        
        Program: {program_info['program_name']}
        
        File Operations Found: {file_operations[:10]}
        Data Movements Found: {data_movements[:10]}
        Field Usage: {[field[0] for field in program_info['field_usage'][:10]]}
        
        Document:
        1. Input data sources
        2. Output data destinations
        3. Data transformation steps
        4. Data validation points
        5. Data flow sequence
        
        Create a clear data flow narrative.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=700)
        except Exception as e:
            self.logger.error(f"Failed to generate data flow doc: {e}")
            return self._generate_fallback_data_flow_doc(program_info, file_operations, data_movements)
    
    def _generate_fallback_data_flow_doc(self, program_info: Dict[str, Any], 
                                       file_operations: List, data_movements: List) -> str:
        """Generate fallback data flow documentation"""
        doc = f"## Data Flow Analysis\n\n"
        doc += f"**Program:** {program_info['program_name']}\n\n"
        
        if file_operations:
            doc += "**File Operations:**\n"
            for op, file_name in file_operations[:5]:
                doc += f"- {op} {file_name}\n"
            doc += "\n"
        
        if data_movements:
            doc += "**Data Movements:**\n"
            for source, target in data_movements[:5]:
                doc += f"- {source.strip()} → {target.strip()}\n"
            doc += "\n"
        
        doc += f"**Fields Processed:** {len(program_info['field_usage'])} fields\n"
        
        return doc
    
    async def _generate_technical_specs(self, program_info: Dict[str, Any]) -> str:
        """Generate technical specifications"""
        # Analyze technical characteristics
        total_lines = sum(len(chunk[1].split('\n')) for chunk in program_info['chunks'])
        
        # Count different types of statements
        all_content = program_info['all_content']
        sql_count = len(re.findall(r'EXEC\s+SQL', all_content, re.IGNORECASE))
        perform_count = len(re.findall(r'\bPERFORM\b', all_content, re.IGNORECASE))
        if_count = len(re.findall(r'\bIF\b', all_content, re.IGNORECASE))
        
        specs = f"""
## Technical Specifications

**Program Metrics:**
- Total Lines of Code: {total_lines:,}
- Program Chunks: {program_info['total_chunks']}
- Chunk Types: {', '.join(program_info['chunk_types'])}

**Code Characteristics:**
- SQL Statements: {sql_count}
- PERFORM Statements: {perform_count}
- Conditional Statements: {if_count}
- Fields Referenced: {len(program_info['field_usage'])}

**Program Components:**
"""
        
        # Add chunk details
        chunk_summary = {}
        for chunk in program_info['chunks']:
            chunk_type = chunk[0]
            if chunk_type not in chunk_summary:
                chunk_summary[chunk_type] = 0
            chunk_summary[chunk_type] += 1
        
        for chunk_type, count in chunk_summary.items():
            specs += f"- {chunk_type.title()} Sections: {count}\n"
        
        return specs
    
    async def _generate_dependencies_doc(self, program_info: Dict[str, Any]) -> str:
        """Generate dependencies documentation"""
        # Extract dependencies from code
        copybooks = re.findall(r'COPY\s+([A-Z][A-Z0-9-]*)', program_info['all_content'], re.IGNORECASE)
        called_programs = re.findall(r'CALL\s+["\']([^"\']+)["\']', program_info['all_content'], re.IGNORECASE)
        files_used = list(set(field[3] for field in program_info['field_usage'] if field[3]))
        
        dependencies_doc = "## Dependencies\n\n"
        
        if copybooks:
            dependencies_doc += "**Copybooks:**\n"
            for copybook in set(copybooks):
                dependencies_doc += f"- {copybook}\n"
            dependencies_doc += "\n"
        
        if called_programs:
            dependencies_doc += "**Called Programs:**\n"
            for program in set(called_programs):
                dependencies_doc += f"- {program}\n"
            dependencies_doc += "\n"
        
        if files_used:
            dependencies_doc += "**Files/Tables Used:**\n"
            for file_name in set(files_used):
                dependencies_doc += f"- {file_name}\n"
            dependencies_doc += "\n"
        
        if not (copybooks or called_programs or files_used):
            dependencies_doc += "No external dependencies identified.\n"
        
        return dependencies_doc
    
    async def _generate_maintenance_doc_api(self, program_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate maintenance documentation"""
        prompt = f"""
        Generate maintenance documentation for this COBOL program:
        
        Program: {program_info['program_name']}
        Complexity: {program_info['total_chunks']} chunks
        
        Based on the program structure, provide:
        1. Common maintenance tasks
        2. Areas requiring careful attention
        3. Testing recommendations
        4. Change management considerations
        5. Performance monitoring points
        
        Write practical maintenance guidance.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Failed to generate maintenance doc: {e}")
            return f"Maintenance procedures for {program_info['program_name']} - {program_info['total_chunks']} components require regular monitoring."
    
    async def _generate_troubleshooting_doc_api(self, program_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate troubleshooting documentation"""
        # Identify potential error points
        error_handling = re.findall(r'\b(INVALID\s+KEY|AT\s+END|ON\s+ERROR)', 
                                  program_info['all_content'], re.IGNORECASE)
        
        prompt = f"""
        Generate troubleshooting documentation for this COBOL program:
        
        Program: {program_info['program_name']}
        Error Handling Found: {error_handling[:5]}
        
        Provide:
        1. Common error scenarios
        2. Diagnostic steps
        3. Log file locations and what to look for
        4. Recovery procedures
        5. Escalation guidelines
        
        Focus on practical troubleshooting steps.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Failed to generate troubleshooting doc: {e}")
            return self._generate_fallback_troubleshooting_doc(program_info, error_handling)
    
    def _generate_fallback_troubleshooting_doc(self, program_info: Dict[str, Any], error_handling: List) -> str:
        """Generate fallback troubleshooting documentation"""
        doc = f"## Troubleshooting Guide\n\n"
        doc += f"**Program:** {program_info['program_name']}\n\n"
        
        if error_handling:
            doc += "**Error Handling Mechanisms:**\n"
            for error in set(error_handling[:5]):
                doc += f"- {error[0]}\n"
            doc += "\n"
        
        doc += "**Common Issues:**\n"
        doc += "- Check log files for error messages\n"
        doc += "- Verify input data format and availability\n"
        doc += "- Monitor system resources during execution\n"
        doc += "- Review recent changes to dependencies\n"
        
        return doc
    
    async def generate_comprehensive_flow_documentation(self, component_name: str, 
                                                    component_type: str, 
                                                    analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 ENHANCED: Generate comprehensive flow documentation combining all analysis results"""
        try:
            # Extract data from analysis results
            lineage_data = analysis_results.get("lineage_analysis", {}).get("data", {})
            logic_data = analysis_results.get("logic_analysis", {}).get("data", {})
            
            # Generate different types of flow documentation based on component type
            if component_type == "file":
                documentation = await self._generate_file_flow_documentation_api(
                    component_name, lineage_data, logic_data
                )
            elif component_type in ["program", "cobol"]:
                documentation = await self._generate_program_flow_documentation_api(
                    component_name, lineage_data, logic_data
                )
            elif component_type == "field":
                documentation = await self._generate_field_flow_documentation_api(
                    component_name, lineage_data, logic_data
                )
            else:
                documentation = await self._generate_generic_flow_documentation_api(
                    component_name, component_type, lineage_data, logic_data
                )
            
            result = {
                "status": "success",
                "component_name": component_name,
                "component_type": component_type,
                "documentation": documentation,
                "format": "markdown",
                "documentation_type": "comprehensive_flow",
                "generation_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Comprehensive flow documentation failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    async def _generate_file_flow_documentation_api(self, file_name: str, 
                                                lineage_data: Dict, logic_data: Dict) -> str:
        """Generate comprehensive file flow documentation using API"""
        
        # Extract key information from analysis data
        file_info = {
            "file_name": file_name,
            "programs_using": len(lineage_data.get("programs_using", [])),
            "operations": len(lineage_data.get("operations", [])),
            "access_patterns": lineage_data.get("file_access_data", {}).get("access_patterns", {}),
            "field_definitions": len(lineage_data.get("field_definitions", [])),
            "field_categories": lineage_data.get("field_usage_analysis", {}).get("field_categories", {})
        }
        
        prompt = f"""
        Generate comprehensive file flow documentation for: {file_name}
        
        File Analysis Summary:
        - Used by {file_info['programs_using']} programs
        - {file_info['operations']} different operations
        - {file_info['field_definitions']} field definitions
        - Access patterns: {file_info['access_patterns']}
        
        Field Categories Found:
        - Input Fields: {len(file_info['field_categories'].get('input_fields', []))}
        - Derived Fields: {len(file_info['field_categories'].get('derived_fields', []))}
        - Updated Fields: {len(file_info['field_categories'].get('updated_fields', []))}
        - Static Fields: {len(file_info['field_categories'].get('static_fields', []))}
        - Unused Fields: {len(file_info['field_categories'].get('unused_fields', []))}
        
        Create comprehensive documentation covering:
        
        # {file_name} - File Flow Documentation
        
        ## Executive Summary
        - Business purpose and critical importance
        - Role in data processing ecosystem
        - Key stakeholders and usage patterns
        
        ## File Overview
        - Data characteristics and structure
        - Business domain and context
        - Criticality and compliance requirements
        
        ## Data Flow Lifecycle
        
        ### Input Phase
        - Data sources and origins
        - Data collection processes
        - Initial validation and quality checks
        
        ### Processing Phase
        - Programs that create the file
        - Data transformation and enrichment
        - Business rules and validations applied
        
        ### Consumption Phase
        - Programs that read the file
        - Downstream processing workflows
        - Data distribution and usage patterns
        
        ### Maintenance Phase
        - Update and modification processes
        - Data quality monitoring
        - Archival and retention policies
        
        ## Field-Level Analysis
        
        ### Input Fields
        - Fields populated from external sources
        - Data validation requirements
        - Source system mappings
        
        ### Derived Fields
        - Calculated and computed fields
        - Business logic and formulas
        - Dependencies and calculations
        
        ### Updated Fields
        - Fields modified during processing
        - Update patterns and frequencies
        - Change tracking mechanisms
        
        ### Static Fields
        - Reference and lookup data
        - Configuration parameters
        - Rarely changing information
        
        ### Unused Fields
        - Legacy fields no longer used
        - Potential cleanup opportunities
        - Impact assessment for removal
        
        ## Program Integration
        
        ### Producer Programs
        - Programs that create or populate the file
        - Data generation processes
        - Quality assurance measures
        
        ### Consumer Programs
        - Programs that read and process the file
        - Usage patterns and dependencies
        - Performance considerations
        
        ### Maintenance Programs
        - Backup and recovery processes
        - Data quality monitoring
        - Housekeeping and cleanup
        
        ## Data Quality and Governance
        
        ### Quality Metrics
        - Data completeness indicators
        - Accuracy and consistency measures
        - Validation rules and checks
        
        ### Compliance Considerations
        - Regulatory requirements
        - Data privacy and security
        - Audit trail requirements
        
        ### Change Management
        - Impact assessment procedures
        - Testing and validation requirements
        - Deployment coordination
        
        ## Performance and Optimization
        
        ### Current Performance
        - File size and growth patterns
        - Access frequency and patterns
        - Resource utilization characteristics
        
        ### Optimization Opportunities
        - Structure improvements
        - Processing efficiency enhancements
        - Storage optimization options
        
        ## Risk Assessment
        
        ### Dependencies and Vulnerabilities
        - Critical dependencies identification
        - Single points of failure
        - Risk mitigation strategies
        
        ### Business Impact
        - Consequences of file unavailability
        - Recovery time objectives
        - Business continuity planning
        
        ## Maintenance and Support
        
        ### Operational Procedures
        - Daily monitoring requirements
        - Troubleshooting guidelines
        - Escalation procedures
        
        ### Documentation Maintenance
        - Review and update schedules
        - Change notification processes
        - Knowledge transfer requirements
        
        Write as professional, comprehensive documentation suitable for technical teams, 
        business stakeholders, and governance committees.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1500)
        except Exception as e:
            self.logger.error(f"File flow documentation generation failed: {e}")
            return self._generate_fallback_file_documentation(file_name, file_info)

    def _generate_fallback_file_documentation(self, file_name: str, file_info: Dict) -> str:
        """Generate fallback file documentation when API fails"""
        doc = f"# {file_name} - File Flow Documentation\n\n"
        
        doc += "## Executive Summary\n\n"
        doc += f"File {file_name} is a critical data component in the mainframe system, "
        doc += f"serving {file_info['programs_using']} programs with {file_info['field_definitions']} data fields. "
        doc += "This file plays a vital role in the data processing ecosystem.\n\n"
        
        doc += "## File Overview\n\n"
        doc += f"**File Name:** {file_name}\n"
        doc += f"**Programs Using:** {file_info['programs_using']}\n"
        doc += f"**Operations Supported:** {file_info['operations']}\n"
        doc += f"**Field Count:** {file_info['field_definitions']}\n\n"
        
        doc += "## Data Flow Summary\n\n"
        access_patterns = file_info.get('access_patterns', {})
        if access_patterns:
            doc += "### Access Patterns\n"
            for pattern, programs in access_patterns.items():
                if programs:
                    doc += f"- **{pattern.title()}:** {len(programs)} programs\n"
        
        doc += "\n## Field Analysis\n\n"
        field_categories = file_info.get('field_categories', {})
        for category, fields in field_categories.items():
            if fields:
                doc += f"- **{category.replace('_', ' ').title()}:** {len(fields)} fields\n"
        
        doc += "\n## Recommendations\n\n"
        doc += "- Regular monitoring of file access patterns\n"
        doc += "- Periodic review of field usage and cleanup opportunities\n"
        doc += "- Documentation of business rules and validation logic\n"
        doc += "- Implementation of data quality monitoring\n"
        
        return doc

    async def _generate_program_flow_documentation_api(self, program_name: str, 
                                                    lineage_data: Dict, logic_data: Dict) -> str:
        """Generate comprehensive program flow documentation using API"""
        
        # Extract key information from analysis data
        program_info = {
            "program_name": program_name,
            "complexity_score": logic_data.get("complexity_score", 0),
            "business_rules": len(logic_data.get("business_rules", [])),
            "logic_patterns": len(logic_data.get("logic_patterns", [])),
            "program_relationships": logic_data.get("program_relationships", {}),
            "file_access": logic_data.get("file_access_patterns", []),
            "transformations": logic_data.get("data_transformations", {})
        }
        
        outbound_calls = len(program_info.get("program_relationships", {}).get("outbound_calls", []))
        inbound_calls = len(program_info.get("program_relationships", {}).get("inbound_calls", []))
        
        prompt = f"""
        Generate comprehensive program flow documentation for: {program_name}
        
        Program Analysis Summary:
        - Complexity Score: {program_info['complexity_score']}/10
        - Business Rules: {program_info['business_rules']}
        - Logic Patterns: {program_info['logic_patterns']}
        - Outbound Calls: {outbound_calls}
        - Inbound Calls: {inbound_calls}
        - File Operations: {len(program_info['file_access'])}
        
        Create comprehensive documentation covering:
        
        # {program_name} - Program Flow Documentation
        
        ## Executive Summary
        - Business purpose and objectives
        - Critical role in system operations
        - Key stakeholders and dependencies
        
        ## Program Overview
        - Functional description and scope
        - Business domain and context
        - Performance characteristics
        
        ## Program Flow Analysis
        
        ### Input Processing
        - Data sources and input validation
        - Parameter processing and validation
        - Initial setup and initialization
        
        ### Core Processing Logic
        - Main business algorithms and calculations
        - Decision points and business rules
        - Data transformation procedures
        
        ### Output Generation
        - Output file creation and formatting
        - Result validation and quality checks
        - Success and error reporting
        
        ### Error Handling
        - Exception processing procedures
        - Error recovery mechanisms
        - Logging and audit trail generation
        
        ## Program Relationships
        
        ### Called Programs
        - Downstream program invocations
        - Parameter passing mechanisms
        - Return value processing
        
        ### Calling Programs
        - Upstream program dependencies
        - Invocation contexts and triggers
        - Integration patterns
        
        ### Shared Resources
        - Common copybooks and data structures
        - Shared files and databases
        - System resources and utilities
        
        ## Data Flow Analysis
        
        ### Input Data Flow
        - File reading and data extraction
        - Data validation and cleansing
        - Format conversion and normalization
        
        ### Internal Data Processing
        - Working storage utilization
        - Data structure manipulation
        - Temporary file management
        
        ### Output Data Flow
        - Result file generation
        - Data formatting and validation
        - Distribution and delivery
        
        ## Business Logic Documentation
        
        ### Business Rules Implementation
        - Validation rules and constraints
        - Calculation formulas and algorithms
        - Decision logic and branching
        
        ### Data Transformations
        - Field-level transformations
        - Format conversions
        - Data enrichment processes
        
        ### Quality Assurance
        - Data quality checks
        - Business rule validation
        - Consistency verification
        
        ## Performance and Optimization
        
        ### Current Performance Profile
        - Processing time characteristics
        - Resource utilization patterns
        - Throughput capabilities
        
        ### Optimization Opportunities
        - Algorithm improvements
        - Resource utilization optimization
        - Processing efficiency enhancements
        
        ## Integration Architecture
        
        ### System Integration Points
        - External system interfaces
        - Database connections
        - File system interactions
        
        ### Data Synchronization
        - Timing and scheduling requirements
        - Data consistency mechanisms
        - Error recovery procedures
        
        ## Risk Assessment and Impact Analysis
        
        ### Critical Dependencies
        - Essential input data sources
        - Required system resources
        - Dependent downstream processes
        
        ### Failure Impact Analysis
        - Business impact of program failure
        - Recovery time requirements
        - Mitigation strategies
        
        ### Change Impact Assessment
        - Components affected by modifications
        - Testing requirements
        - Deployment considerations
        
        ## Maintenance and Support
        
        ### Operational Procedures
        - Monitoring and alerting
        - Troubleshooting guidelines
        - Performance tuning recommendations
        
        ### Development Guidelines
        - Coding standards and conventions
        - Testing and validation procedures
        - Documentation maintenance
        
        ### Knowledge Management
        - Key personnel and expertise
        - Training requirements
        - Knowledge transfer procedures
        
        Write as comprehensive technical documentation suitable for development teams, 
        operations staff, and business analysts.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1500)
        except Exception as e:
            self.logger.error(f"Program flow documentation generation failed: {e}")
            return self._generate_fallback_program_documentation(program_name, program_info)

    def _generate_fallback_program_documentation(self, program_name: str, program_info: Dict) -> str:
        """Generate fallback program documentation when API fails"""
        doc = f"# {program_name} - Program Flow Documentation\n\n"
        
        doc += "## Executive Summary\n\n"
        doc += f"Program {program_name} is a mainframe component with complexity score "
        doc += f"{program_info['complexity_score']}/10, implementing {program_info['business_rules']} "
        doc += f"business rules and {program_info['logic_patterns']} logic patterns.\n\n"
        
        doc += "## Program Overview\n\n"
        doc += f"**Program Name:** {program_name}\n"
        doc += f"**Complexity Score:** {program_info['complexity_score']}/10\n"
        doc += f"**Business Rules:** {program_info['business_rules']}\n"
        doc += f"**Logic Patterns:** {program_info['logic_patterns']}\n\n"
        
        doc += "## Integration Analysis\n\n"
        relationships = program_info.get('program_relationships', {})
        outbound = len(relationships.get('outbound_calls', []))
        inbound = len(relationships.get('inbound_calls', []))
        
        doc += f"**Program Calls:** {outbound} outbound calls\n"
        doc += f"**Called By:** {inbound} inbound calls\n"
        doc += f"**File Operations:** {len(program_info['file_access'])}\n\n"
        
        doc += "## Complexity Assessment\n\n"
        if program_info['complexity_score'] > 7:
            doc += "**High Complexity:** This program requires careful change management and extensive testing.\n"
        elif program_info['complexity_score'] > 4:
            doc += "**Medium Complexity:** Standard development and testing practices apply.\n"
        else:
            doc += "**Low Complexity:** Straightforward maintenance and modification procedures.\n"
        
        doc += "\n## Recommendations\n\n"
        doc += "- Regular code review and refactoring opportunities\n"
        doc += "- Comprehensive testing for any modifications\n"
        doc += "- Documentation of business rules and logic\n"
        doc += "- Performance monitoring and optimization\n"
        
        return doc

    async def _generate_field_flow_documentation_api(self, field_name: str, 
                                                    lineage_data: Dict, logic_data: Dict) -> str:
        """Generate comprehensive field flow documentation using API"""
        
        # Extract key information from analysis data
        field_info = {
            "field_name": field_name,
            "programs_using": len(lineage_data.get("programs_using", [])),
            "operations": lineage_data.get("operations", []),
            "lifecycle": lineage_data.get("lifecycle", {}),
            "transformations": len(lineage_data.get("transformations", [])),
            "impact_analysis": lineage_data.get("impact_analysis", {})
        }
        
        prompt = f"""
        Generate comprehensive field flow documentation for: {field_name}
        
        Field Analysis Summary:
        - Used in {field_info['programs_using']} programs
        - Operations: {field_info['operations']}
        - Transformations: {field_info['transformations']}
        - Impact Level: {field_info['impact_analysis'].get('risk_level', 'Unknown')}
        
        Create comprehensive documentation covering:
        
        # {field_name} - Field Flow Documentation
        
        ## Executive Summary
        - Business meaning and purpose
        - Critical importance to operations
        - Key usage patterns and dependencies
        
        ## Field Overview
        - Data type and characteristics
        - Business domain and context
        - Validation rules and constraints
        
        ## Field Lifecycle Analysis
        
        ### Creation and Initialization
        - Where the field is first populated
        - Source data and derivation rules
        - Initial validation and processing
        
        ### Usage and Processing
        - Programs that read the field
        - Business logic and calculations
        - Validation and verification processes
        
        ### Transformation and Updates
        - Modification patterns and rules
        - Data transformation logic
        - Update frequency and triggers
        
        ### Output and Distribution
        - Final usage and consumption
        - Output formatting and presentation
        - Distribution to downstream systems
        
        ## Cross-Program Usage
        
        ### Program Dependencies
        - Programs that depend on this field
        - Usage contexts and patterns
        - Critical dependencies identification
        
        ### Data Flow Patterns
        - Field movement across programs
        - Transformation chains
        - Data quality checkpoints
        
        ### Integration Points
        - Inter-program data sharing
        - Synchronization requirements
        - Consistency mechanisms
        
        ## Business Rules and Logic
        
        ### Validation Rules
        - Data quality constraints
        - Business rule validation
        - Error handling procedures
        
        ### Calculation Logic
        - Derivation formulas
        - Transformation algorithms
        - Business calculation rules
        
        ### Usage Constraints
        - Access permissions and security
        - Usage guidelines and standards
        - Compliance requirements
        
        ## Impact Analysis
        
        ### Change Impact Assessment
        - Programs affected by field changes
        - Downstream impact propagation
        - Testing and validation requirements
        
        ### Risk Assessment
        - Critical dependencies and vulnerabilities
        - Business impact of field issues
        - Mitigation strategies
        
        ### Optimization Opportunities
        - Usage pattern improvements
        - Performance enhancements
        - Architecture simplifications
        
        ## Data Quality and Governance
        
        ### Quality Metrics
        - Data completeness and accuracy
        - Consistency across programs
        - Validation effectiveness
        
        ### Governance Framework
        - Data stewardship responsibilities
        - Change management procedures
        - Documentation maintenance
        
        ### Compliance Considerations
        - Regulatory requirements
        - Privacy and security constraints
        - Audit trail requirements
        
        ## Maintenance and Support
        
        ### Monitoring Requirements
        - Quality monitoring procedures
        - Usage pattern tracking
        - Performance indicators
        
        ### Support Procedures
        - Issue identification and resolution
        - Escalation procedures
        - Knowledge management
        
        ### Documentation Maintenance
        - Review and update schedules
        - Change notification processes
        - Stakeholder communication
        
        Write as comprehensive field documentation suitable for data stewards, 
        developers, and business analysts.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1500)
        except Exception as e:
            self.logger.error(f"Field flow documentation generation failed: {e}")
            return self._generate_fallback_field_documentation(field_name, field_info)

    def _generate_fallback_field_documentation(self, field_name: str, field_info: Dict) -> str:
        """Generate fallback field documentation when API fails"""
        doc = f"# {field_name} - Field Flow Documentation\n\n"
        
        doc += "## Executive Summary\n\n"
        doc += f"Field {field_name} is used across {field_info['programs_using']} programs "
        doc += f"with {field_info['transformations']} transformations and "
        doc += f"{len(field_info['operations'])} different operations.\n\n"
        
        doc += "## Field Overview\n\n"
        doc += f"**Field Name:** {field_name}\n"
        doc += f"**Programs Using:** {field_info['programs_using']}\n"
        doc += f"**Operations:** {', '.join(field_info['operations'])}\n"
        doc += f"**Transformations:** {field_info['transformations']}\n\n"
        
        doc += "## Usage Analysis\n\n"
        impact = field_info.get('impact_analysis', {})
        risk_level = impact.get('risk_level', 'Unknown')
        
        doc += f"**Impact Level:** {risk_level}\n"
        doc += f"**Programs Affected:** {len(impact.get('affected_programs', []))}\n"
        doc += f"**Critical Operations:** {len(impact.get('critical_operations', []))}\n\n"
        
        doc += "## Lifecycle Summary\n\n"
        lifecycle = field_info.get('lifecycle', {})
        for stage, operations in lifecycle.items():
            if operations:
                doc += f"- **{stage.title()}:** {len(operations)} operations\n"
        
        doc += "\n## Recommendations\n\n"
        doc += "- Monitor field usage patterns for optimization opportunities\n"
        doc += "- Validate business rules and transformation logic\n"
        doc += "- Ensure comprehensive testing for any field modifications\n"
        doc += "- Document business meaning and validation requirements\n"
        
        return doc

    async def generate_impact_assessment_documentation(self, component_name: str, 
                                                    analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 NEW: Generate impact assessment documentation for change management"""
        try:
            # Extract impact data from analysis results
            impact_data = self._extract_impact_data(analysis_results)
            
            # Generate comprehensive impact assessment
            impact_documentation = await self._generate_impact_assessment_api(component_name, impact_data)
            
            result = {
                "status": "success",
                "component_name": component_name,
                "documentation": impact_documentation,
                "format": "markdown",
                "documentation_type": "impact_assessment",
                "impact_summary": impact_data,
                "generation_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Impact assessment documentation failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    def _extract_impact_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract impact data from analysis results"""
        impact_data = {
            "component_name": analysis_results.get("component_name"),
            "component_type": analysis_results.get("component_type"),
            "total_analyses": len(analysis_results.get("analyses", {})),
            "dependencies": [],
            "impacts": [],
            "complexity_indicators": {},
            "risk_factors": []
        }
        
        # Extract lineage impact data
        lineage_analysis = analysis_results.get("analyses", {}).get("lineage_analysis", {})
        if lineage_analysis.get("status") == "success":
            lineage_data = lineage_analysis.get("data", {})
            
            # Extract impact analysis
            if "impact_analysis" in lineage_data:
                impact_analysis = lineage_data["impact_analysis"]
                impact_data["dependencies"] = impact_analysis.get("affected_programs", [])
                impact_data["impacts"] = impact_analysis.get("affected_tables", [])
                impact_data["risk_factors"].append(f"Risk Level: {impact_analysis.get('risk_level', 'Unknown')}")
        
        # Extract logic complexity data
        logic_analysis = analysis_results.get("analyses", {}).get("logic_analysis", {})
        if logic_analysis.get("status") == "success":
            logic_data = logic_analysis.get("data", {})
            
            complexity_score = logic_data.get("complexity_score", 0)
            impact_data["complexity_indicators"]["complexity_score"] = complexity_score
            
            if complexity_score > 7:
                impact_data["risk_factors"].append("High complexity requires extensive testing")
            
            business_rules = len(logic_data.get("business_rules", []))
            impact_data["complexity_indicators"]["business_rules"] = business_rules
        
        return impact_data

    async def _generate_impact_assessment_api(self, component_name: str, impact_data: Dict) -> str:
        """Generate comprehensive impact assessment using API"""
        
        prompt = f"""
        Generate comprehensive change impact assessment documentation for: {component_name}
        
        Impact Analysis Data:
        - Component Type: {impact_data.get('component_type')}
        - Dependencies: {len(impact_data.get('dependencies', []))} components
        - Downstream Impacts: {len(impact_data.get('impacts', []))} components
        - Complexity Score: {impact_data.get('complexity_indicators', {}).get('complexity_score', 'N/A')}
        - Risk Factors: {impact_data.get('risk_factors', [])}
        
        Create comprehensive impact assessment covering:
        
        # Change Impact Assessment: {component_name}
        
        ## Executive Summary
        - Change scope and business impact
        - Risk assessment and mitigation requirements
        - Resource and timeline implications
        
        ## Component Analysis
        - Current role and importance
        - Integration complexity
        - Business criticality assessment
        
        ## Impact Scope Analysis
        
        ### Direct Impacts
        - Components directly affected by changes
        - Immediate functional impacts
        - System behavior modifications
        
        ### Indirect Impacts
        - Downstream effect propagation
        - Secondary system impacts
        - Performance and resource implications
        
        ### Data Flow Impacts
        - Data integrity considerations
        - Validation and quality impacts
        - Synchronization requirements
        
        ## Risk Assessment
        
        ### Technical Risks
        - System stability risks
        - Performance degradation risks
        - Integration failure risks
        
        ### Business Risks
        - Operational disruption risks
        - Data quality risks
        - Compliance and regulatory risks
        
        ### Mitigation Strategies
        - Risk reduction approaches
        - Contingency planning
        - Rollback procedures
        
        ## Testing Requirements
        
        ### Unit Testing
        - Component-level testing scope
        - Validation criteria
        - Test case requirements
        
        ### Integration Testing
        - Inter-component testing
        - End-to-end workflow validation
        - Performance testing requirements
        
        ### User Acceptance Testing
        - Business scenario validation
        - User interface testing
        - Operational readiness verification
        
        ## Implementation Planning
        
        ### Change Sequence
        - Recommended implementation order
        - Dependency management
        - Coordination requirements
        
        ### Resource Requirements
        - Development effort estimation
        - Testing resource needs
        - Deployment resource requirements
        
        ### Timeline Considerations
        - Critical path analysis
        - Milestone identification
        - Contingency time allocation
        
        ## Deployment Strategy
        
        ### Environment Progression
        - Development to production pathway
        - Environment-specific considerations
        - Validation checkpoints
        
        ### Rollback Planning
        - Rollback trigger criteria
        - Rollback procedures
        - Recovery time objectives
        
        ### Monitoring and Validation
        - Post-deployment monitoring
        - Success criteria validation
        - Issue identification procedures
        
        ## Communication Plan
        
        ### Stakeholder Notification
        - Internal team communication
        - Business stakeholder updates
        - User community notification
        
        ### Documentation Updates
        - Technical documentation revisions
        - User guide modifications
        - Process documentation updates
        
        ### Training Requirements
        - Staff training needs
        - Knowledge transfer sessions
        - Support procedure updates
        
        ## Success Criteria
        
        ### Technical Success Metrics
        - Functional validation criteria
        - Performance benchmarks
        - Quality assurance metrics
        
        ### Business Success Metrics
        - Operational improvement indicators
        - User satisfaction measures
        - Business value realization
        
        Write as comprehensive change management documentation suitable for project managers, 
        technical teams, and business stakeholders.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1500)
        except Exception as e:
            self.logger.error(f"Impact assessment generation failed: {e}")
            return self._generate_fallback_impact_assessment(component_name, impact_data)

    def _generate_fallback_impact_assessment(self, component_name: str, impact_data: Dict) -> str:
        """Generate fallback impact assessment when API fails"""
        doc = f"# Change Impact Assessment: {component_name}\n\n"
        
        doc += "## Executive Summary\n\n"
        doc += f"Changes to {component_name} ({impact_data.get('component_type', 'unknown')}) "
        doc += f"will affect {len(impact_data.get('dependencies', []))} direct dependencies "
        doc += f"and {len(impact_data.get('impacts', []))} downstream components.\n\n"
        
        doc += "## Impact Scope\n\n"
        doc += f"**Direct Dependencies:** {len(impact_data.get('dependencies', []))}\n"
        doc += f"**Downstream Impacts:** {len(impact_data.get('impacts', []))}\n"
        
        complexity = impact_data.get('complexity_indicators', {}).get('complexity_score', 0)
        if complexity > 7:
            doc += f"**Risk Level:** HIGH (Complexity: {complexity}/10)\n\n"
            doc += "### High Risk Considerations\n"
            doc += "- Extensive testing required due to high complexity\n"
            doc += "- Coordinated deployment across multiple components\n"
            doc += "- Comprehensive rollback planning essential\n"
        elif complexity > 4:
            doc += f"**Risk Level:** MEDIUM (Complexity: {complexity}/10)\n\n"
            doc += "### Medium Risk Considerations\n"
            doc += "- Standard testing procedures apply\n"
            doc += "- Coordinate with dependent components\n"
            doc += "- Plan for staged deployment\n"
        else:
            doc += f"**Risk Level:** LOW (Complexity: {complexity}/10)\n\n"
            doc += "### Low Risk Considerations\n"
            doc += "- Minimal coordination required\n"
            doc += "- Standard change management procedures\n"
            doc += "- Low impact on system operations\n"
        
        doc += "\n## Recommendations\n\n"
        doc += "1. **Testing Strategy:**\n"
        doc += "   - Comprehensive unit testing\n"
        doc += "   - Integration testing with dependent components\n"
        doc += "   - End-to-end workflow validation\n\n"
        
        doc += "2. **Deployment Approach:**\n"
        doc += "   - Staged deployment to minimize risk\n"
        doc += "   - Monitoring and validation at each stage\n"
        doc += "   - Rollback procedures ready for immediate use\n\n"
        
        doc += "3. **Communication Plan:**\n"
        doc += "   - Notify all stakeholders of planned changes\n"
        doc += "   - Update documentation and procedures\n"
        doc += "   - Provide training as needed\n"
        
        return doc

    async def generate_operational_runbook(self, component_name: str, 
                                        component_type: str,
                                        analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 NEW: Generate operational runbook for component maintenance"""
        try:
            # Extract operational data from analysis results
            operational_data = self._extract_operational_data(analysis_results)
            
            # Generate runbook documentation
            runbook_documentation = await self._generate_runbook_api(component_name, component_type, operational_data)
            
            result = {
                "status": "success",
                "component_name": component_name,
                "component_type": component_type,
                "documentation": runbook_documentation,
                "format": "markdown",
                "documentation_type": "operational_runbook",
                "operational_summary": operational_data,
                "generation_timestamp": dt.now().isoformat()
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            self.logger.error(f"Operational runbook generation failed: {str(e)}")
            return self._add_processing_info({"error": str(e)})

    def _extract_operational_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract operational data from analysis results"""
        operational_data = {
            "component_name": analysis_results.get("component_name"),
            "component_type": analysis_results.get("component_type"),
            "complexity_score": 0,
            "dependencies": [],
            "file_operations": [],
            "error_conditions": [],
            "performance_indicators": {},
            "monitoring_points": []
        }
        
        # Extract from logic analysis
        logic_analysis = analysis_results.get("analyses", {}).get("logic_analysis", {})
        if logic_analysis.get("status") == "success":
            logic_data = logic_analysis.get("data", {})
            operational_data["complexity_score"] = logic_data.get("complexity_score", 0)
            operational_data["error_conditions"] = logic_data.get("error_handling", [])
            
            # Extract file operations
            if "file_access_patterns" in logic_data:
                operational_data["file_operations"] = logic_data["file_access_patterns"]
        
        # Extract from lineage analysis
        lineage_analysis = analysis_results.get("analyses", {}).get("lineage_analysis", {})
        if lineage_analysis.get("status") == "success":
            lineage_data = lineage_analysis.get("data", {})
            operational_data["dependencies"] = lineage_data.get("programs_using", [])
            
            # Extract performance indicators
            if "usage_analysis" in lineage_data:
                usage_stats = lineage_data["usage_analysis"].get("statistics", {})
                operational_data["performance_indicators"] = {
                    "total_references": usage_stats.get("total_references", 0),
                    "programs_count": len(usage_stats.get("programs_using", [])),
                    "complexity_score": lineage_data.get("usage_analysis", {}).get("complexity_score", 0)
                }
        
        return operational_data

    async def _generate_runbook_api(self, component_name: str, component_type: str, operational_data: Dict) -> str:
        """Generate operational runbook using API"""
        
        prompt = f"""
        Generate comprehensive operational runbook for: {component_name}
        
        Component Information:
        - Type: {component_type}
        - Complexity Score: {operational_data.get('complexity_score', 0)}/10
        - Dependencies: {len(operational_data.get('dependencies', []))} components
        - File Operations: {len(operational_data.get('file_operations', []))}
        - Performance Indicators: {operational_data.get('performance_indicators', {})}
        
        Create comprehensive operational runbook covering:
        
        # Operational Runbook: {component_name}
        
        ## Component Overview
        - Purpose and business function
        - Operational characteristics
        - Critical dependencies and requirements
        
        ## Daily Operations
        
        ### Monitoring Procedures
        - Key performance indicators to monitor
        - Normal operating parameters
        - Alert thresholds and escalation criteria
        
        ### Health Checks
        - Routine health verification procedures
        - Status validation checkpoints
        - Performance baseline comparisons
        
        ### Maintenance Tasks
        - Regular maintenance procedures
        - Housekeeping and cleanup tasks
        - Preventive maintenance schedule
        
        ## Troubleshooting Guide
        
        ### Common Issues
        - Frequently encountered problems
        - Root cause analysis procedures
        - Quick resolution steps
        
        ### Error Conditions
        - Known error scenarios and symptoms
        - Diagnostic procedures
        - Recovery and resolution steps
        
        ### Performance Issues
        - Performance degradation indicators
        - Capacity and resource constraints
        - Optimization and tuning procedures
        
        ## Emergency Procedures
        
        ### System Failures
        - Failure detection and assessment
        - Emergency response procedures
        - Escalation and notification protocols
        
        ### Data Recovery
        - Backup and recovery procedures
        - Data integrity verification
        - Business continuity measures
        
        ### Security Incidents
        - Security breach detection
        - Incident response procedures
        - Recovery and hardening steps
        
        ## Maintenance Procedures
        
        ### Scheduled Maintenance
        - Planned maintenance windows
        - Pre-maintenance preparation
        - Post-maintenance validation
        
        ### Configuration Changes
        - Change approval procedures
        - Implementation guidelines
        - Rollback procedures
        
        ### Updates and Patches
        - Update assessment and planning
        - Testing and validation requirements
        - Deployment procedures
        
        ## Performance Management
        
        ### Capacity Planning
        - Resource utilization monitoring
        - Growth trend analysis
        - Capacity expansion planning
        
        ### Performance Tuning
        - Performance optimization techniques
        - Resource allocation adjustments
        - Configuration optimization
        
        ### Benchmarking
        - Performance baseline establishment
        - Benchmark testing procedures
        - Performance comparison analysis
        
        ## Integration Management
        
        ### Dependency Management
        - Critical dependency monitoring
        - Integration point validation
        - Coordination procedures
        
        ### Data Flow Management
        - Data flow monitoring procedures
        - Quality assurance checkpoints
        - Synchronization verification
        
        ### Interface Management
        - External interface monitoring
        - Communication protocol validation
        - Error handling and recovery
        
        ## Documentation and Knowledge Management
        
        ### Procedure Documentation
        - Standard operating procedures
        - Emergency response playbooks
        - Configuration and setup guides
        
        ### Knowledge Base
        - Common issues and solutions
        - Best practices and guidelines
        - Lessons learned documentation
        
        ### Training and Certification
        - Operator training requirements
        - Certification procedures
        - Knowledge transfer protocols
        
        ## Compliance and Governance
        
        ### Audit Requirements
        - Compliance monitoring procedures
        - Audit trail maintenance
        - Reporting requirements
        
        ### Security Compliance
        - Security control verification
        - Access management procedures
        - Security monitoring protocols
        
        ### Change Management
        - Change control procedures
        - Impact assessment requirements
        - Approval and documentation
        
        ## Contact Information
        
        ### Support Contacts
        - Primary support team contacts
        - Escalation contact hierarchy
        - Emergency contact procedures
        
        ### Vendor Support
        - Vendor contact information
        - Support contract details
        - Escalation procedures
        
        ### Subject Matter Experts
        - Technical expertise contacts
        - Business domain experts
        - Specialized support resources
        
        Write as practical operational documentation suitable for operations teams, 
        system administrators, and support staff.
        """
        
        try:
            return await self._call_api_for_readable_analysis(prompt, max_tokens=1500)
        except Exception as e:
            self.logger.error(f"Runbook generation failed: {e}")
            return self._generate_fallback_runbook(component_name, component_type, operational_data)

    def _generate_fallback_runbook(self, component_name: str, component_type: str, operational_data: Dict) -> str:
        """Generate fallback runbook when API fails"""
        doc = f"# Operational Runbook: {component_name}\n\n"
        
        doc += "## Component Overview\n\n"
        doc += f"**Component:** {component_name}\n"
        doc += f"**Type:** {component_type}\n"
        doc += f"**Complexity:** {operational_data.get('complexity_score', 0)}/10\n"
        doc += f"**Dependencies:** {len(operational_data.get('dependencies', []))}\n\n"
        
        doc += "## Daily Operations\n\n"
        doc += "### Monitoring Checklist\n"
        doc += "- [ ] Verify component is operational\n"
        doc += "- [ ] Check resource utilization\n"
        doc += "- [ ] Validate dependencies are available\n"
        doc += "- [ ] Review error logs for issues\n"
        doc += "- [ ] Confirm data flow integrity\n\n"
        
        doc += "### Health Check Procedures\n"
        complexity = operational_data.get('complexity_score', 0)
        if complexity > 7:
            doc += "**High Complexity Component - Enhanced Monitoring Required**\n"
            doc += "- Monitor every 15 minutes during business hours\n"
            doc += "- Detailed performance metrics tracking\n"
            doc += "- Proactive issue identification\n"
        elif complexity > 4:
            doc += "**Medium Complexity Component - Standard Monitoring**\n"
            doc += "- Monitor every 30 minutes during business hours\n"
            doc += "- Standard performance metrics\n"
            doc += "- Regular health checks\n"
        else:
            doc += "**Low Complexity Component - Basic Monitoring**\n"
            doc += "- Monitor every hour during business hours\n"
            doc += "- Basic availability checks\n"
            doc += "- Minimal performance tracking\n"
        
        doc += "\n## Troubleshooting\n\n"
        doc += "### Common Issues\n"
        if operational_data.get('file_operations'):
            doc += "- **File Access Issues:** Check file permissions and availability\n"
            doc += "- **Data Processing Errors:** Validate input data format and content\n"
        
        doc += "- **Performance Degradation:** Check system resources and dependencies\n"
        doc += "- **Integration Failures:** Verify dependent component status\n\n"
        
        doc += "### Emergency Contacts\n"
        doc += "- **Primary Support:** [Contact Information]\n"
        doc += "- **Technical Lead:** [Contact Information]\n"
        doc += "- **Business Owner:** [Contact Information]\n\n"
        
        doc += "## Maintenance Schedule\n\n"
        doc += "- **Daily:** Health checks and log reviews\n"
        doc += "- **Weekly:** Performance analysis and capacity review\n"
        doc += "- **Monthly:** Full system validation and optimization review\n"
        doc += "- **Quarterly:** Comprehensive audit and documentation update\n"
        
        return doc
        


    # System documentation methods
    async def _generate_executive_summary_api(self, component_info: Dict[str, Any], system_name: str) -> str:
        """✅ API-BASED: Generate executive summary for system documentation"""
        total_components = len(component_info)
        total_chunks = sum(info['total_chunks'] for info in component_info.values() if info)
        
        prompt = f"""
        Generate an executive summary for this mainframe system:
        
        System Name: {system_name}
        Total Components: {total_components}
        Total Code Chunks: {total_chunks}
        
        Components: {list(component_info.keys())}
        
        Create an executive summary that includes:
        1. System purpose and business value
        2. Key capabilities
        3. System scale and complexity
        4. Strategic importance
        
        Write for executive audience.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=500)
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
            return f"Executive Summary: {system_name} consists of {total_components} components with {total_chunks} total code sections."
    
    async def _generate_system_architecture_api(self, component_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate system architecture documentation"""
        component_types = {}
        for name, info in component_info.items():
            if info:
                for chunk_type in info['chunk_types']:
                    if chunk_type not in component_types:
                        component_types[chunk_type] = []
                    component_types[chunk_type].append(name)
        
        prompt = f"""
        Generate system architecture documentation based on these components:
        
        Component Types: {component_types}
        Total Components: {len(component_info)}
        
        Document:
        1. Overall system architecture
        2. Component relationships
        3. Data flow between components
        4. Integration patterns
        
        Focus on architectural patterns and design principles.
        """
        
        try:
            api_result = await self._call_api_for_analysis(prompt, max_tokens=600)
            return api_result
        except Exception as e:
            self.logger.error(f"Failed to generate system architecture: {e}")
            return self._generate_fallback_system_architecture(component_types)
    
    def _generate_fallback_system_architecture(self, component_types: Dict) -> str:
        """Generate fallback system architecture documentation"""
        arch_doc = "## System Architecture\n\n"
        arch_doc += "**Component Distribution:**\n"
        
        for comp_type, components in component_types.items():
            arch_doc += f"- {comp_type.title()}: {len(components)} components\n"
        
        arch_doc += "\n**System Integration:**\n"
        arch_doc += "The system consists of interconnected mainframe components "
        arch_doc += "that process business data through a series of batch and online operations.\n"
        
        return arch_doc
    
    async def _generate_component_overview(self, component_info: Dict[str, Any]) -> str:
        """Generate component overview documentation"""
        overview = "## Component Overview\n\n"
        
        for name, info in component_info.items():
            if info:
                overview += f"### {name}\n"
                overview += f"- Chunks: {info['total_chunks']}\n"
                overview += f"- Types: {', '.join(info['chunk_types'])}\n"
                overview += f"- Fields Used: {len(info['field_usage'])}\n\n"
        
        return overview
    
    async def _generate_data_architecture_api(self, component_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate data architecture documentation"""
        all_fields = set()
        all_files = set()
        
        for info in component_info.values():
            if info:
                all_fields.update(field[0] for field in info['field_usage'])
                all_files.update(field[3] for field in info['field_usage'] if field[3])
        
        prompt = f"""
        Generate data architecture documentation for a system with:
        
        Total Fields: {len(all_fields)}
        Data Sources: {len(all_files)}
        Key Files: {list(all_files)[:10]}
        
        Document:
        1. Data architecture overview
        2. Data sources and destinations
        3. Data transformation patterns
        4. Data quality considerations
        
        Focus on data management and flow.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Failed to generate data architecture: {e}")
            return self._generate_fallback_data_architecture(all_fields, all_files)
    
    def _generate_fallback_data_architecture(self, all_fields: set, all_files: set) -> str:
        """Generate fallback data architecture documentation"""
        data_arch = "## Data Architecture\n\n"
        data_arch += f"**Data Elements:** {len(all_fields)} unique fields\n"
        data_arch += f"**Data Sources:** {len(all_files)} files/tables\n\n"
        
        if all_files:
            data_arch += "**Key Data Sources:**\n"
            for file_name in sorted(all_files)[:10]:
                data_arch += f"- {file_name}\n"
        
        return data_arch
    
    async def _generate_integration_points_api(self, component_info: Dict[str, Any]) -> str:
        """✅ API-BASED: Generate integration points documentation"""
        # Analyze integration patterns
        all_files = set()
        called_programs = set()
        copybooks = set()
        
        for info in component_info.values():
            if info:
                # Extract file usage
                all_files.update(field[3] for field in info['field_usage'] if field[3])
                
                # Extract called programs and copybooks from content
                content = info.get('all_content', '')
                called_programs.update(re.findall(r'CALL\s+["\']([^"\']+)["\']', content, re.IGNORECASE))
                copybooks.update(re.findall(r'COPY\s+([A-Z][A-Z0-9-]*)', content, re.IGNORECASE))
        
        prompt = f"""
        Generate integration points documentation for a system with {len(component_info)} components.
        
        Components: {list(component_info.keys())}
        
        Integration Analysis:
        - Shared Files/Tables: {len(all_files)} ({list(all_files)[:5]})
        - Called Programs: {len(called_programs)} ({list(called_programs)[:5]})
        - Shared Copybooks: {len(copybooks)} ({list(copybooks)[:5]})
        
        Document:
        1. External system interfaces
        2. Internal component integration
        3. Data exchange mechanisms
        4. Shared resources and dependencies
        5. Communication patterns
        
        Focus on integration architecture and data flow.
        """
        
        try:
            return await self._call_api_for_analysis(prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Failed to generate integration points: {e}")
            return self._generate_fallback_integration_points(all_files, called_programs, copybooks)
    
    def _generate_fallback_integration_points(self, all_files: set, called_programs: set, copybooks: set) -> str:
        """Generate fallback integration points documentation"""
        doc = "## Integration Points\n\n"
        
        if all_files:
            doc += "**Shared Data Sources:**\n"
            for file_name in sorted(all_files)[:10]:
                doc += f"- {file_name}\n"
            doc += "\n"
        
        if called_programs:
            doc += "**Program Integration:**\n"
            for program in sorted(called_programs)[:10]:
                doc += f"- {program}\n"
            doc += "\n"
        
        if copybooks:
            doc += "**Shared Copybooks:**\n"
            for copybook in sorted(copybooks)[:10]:
                doc += f"- {copybook}\n"
            doc += "\n"
        
        if not (all_files or called_programs or copybooks):
            doc += "Integration analysis based on component dependencies and data flow patterns.\n"
        
        return doc