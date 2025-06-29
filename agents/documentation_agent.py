# agents/documentation_agent.py
"""
Agent 6: Documentation Generator
Generates comprehensive technical documentation for mainframe systems
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

import torch
from vllm import AsyncLLMEngine, SamplingParams

@dataclass
class DocumentSection:
    """Represents a documentation section"""
    title: str
    content: str
    section_type: str
    order: int
    subsections: List['DocumentSection'] = None

class DocumentationAgent:
    """Agent for generating technical documentation"""
    
    def __init__(self, llm_engine: AsyncLLMEngine, db_path: str, gpu_id: int):
        self.llm_engine = llm_engine
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(__name__)
        
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
*Generated on {{ generation_date }} by Opulence Documentation Agent*
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
*Generated on {{ generation_date }} by Opulence Documentation Agent*
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
*Generated on {{ generation_date }} by Opulence Documentation Agent*
        """.strip())
        
        return templates
    
    async def generate_program_documentation(self, program_name: str, 
                                           format_type: str = "markdown") -> Dict[str, Any]:
        """Generate comprehensive documentation for a program"""
        try:
            # Gather program information
            program_info = await self._gather_program_info(program_name)
            
            if not program_info:
                return {"error": f"Program {program_name} not found"}
            
            # Generate each documentation section
            sections = {}
            
            sections['overview'] = await self._generate_overview(program_info)
            sections['architecture'] = await self._generate_architecture(program_info)
            sections['business_logic'] = await self._generate_business_logic_doc(program_info)
            sections['data_flow'] = await self._generate_data_flow_doc(program_info)
            sections['technical_specs'] = await self._generate_technical_specs(program_info)
            sections['dependencies'] = await self._generate_dependencies_doc(program_info)
            sections['maintenance'] = await self._generate_maintenance_doc(program_info)
            sections['troubleshooting'] = await self._generate_troubleshooting_doc(program_info)
            
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
            
            return {
                "status": "success",
                "program_name": program_name,
                "documentation": documentation,
                "format": format_type,
                "sections_generated": len(sections),
                "generation_timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed for {program_name}: {str(e)}")
            return {"error": str(e)}
    
    async def generate_system_documentation(self, system_components: List[str], 
                                          system_name: str = "Mainframe System") -> Dict[str, Any]:
        """Generate system-level documentation"""
        try:
            # Gather information for all components
            component_info = {}
            for component in system_components:
                component_info[component] = await self._gather_program_info(component)
            
            # Generate system-level sections
            sections = {}
            sections['executive_summary'] = await self._generate_executive_summary(component_info, system_name)
            sections['system_architecture'] = await self._generate_system_architecture(component_info)
            sections['component_overview'] = await self._generate_component_overview(component_info)
            sections['data_architecture'] = await self._generate_data_architecture(component_info)
            sections['integration_points'] = await self._generate_integration_points(component_info)
            sections['operational_procedures'] = await self._generate_operational_procedures(component_info)
            sections['security'] = await self._generate_security_doc(component_info)
            sections['performance'] = await self._generate_performance_doc(component_info)
            sections['maintenance_support'] = await self._generate_maintenance_support_doc(component_info)
            
            # Render documentation
            documentation = self.templates['system'].render(
                system_name=system_name,
                generation_date=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                **sections
            )
            
            return {
                "status": "success",
                "system_name": system_name,
                "documentation": documentation,
                "components_analyzed": len(system_components),
                "generation_timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System documentation generation failed: {str(e)}")
            return {"error": str(e)}
    
    async def generate_field_lineage_documentation(self, field_name: str) -> Dict[str, Any]:
        """Generate documentation for field lineage"""
        try:
            # Gather field lineage information
            lineage_info = await self._gather_field_lineage_info(field_name)
            
            if not lineage_info:
                return {"error": f"Field {field_name} not found"}
            
            # Generate lineage documentation sections
            sections = {}
            sections['field_overview'] = await self._generate_field_overview(lineage_info)
            sections['lifecycle'] = await self._generate_field_lifecycle_doc(lineage_info)
            sections['usage_patterns'] = await self._generate_field_usage_patterns(lineage_info)
            sections['dependencies'] = await self._generate_field_dependencies(lineage_info)
            sections['quality_metrics'] = await self._generate_field_quality_metrics(lineage_info)
            sections['recommendations'] = await self._generate_field_recommendations(lineage_info)
            
            # Render documentation
            documentation = self.templates['lineage'].render(
                field_name=field_name,
                generation_date=dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                **sections
            )
            
            return {
                "status": "success",
                "field_name": field_name,
                "documentation": documentation,
                "generation_timestamp": dt.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Field lineage documentation failed for {field_name}: {str(e)}")
            return {"error": str(e)}
    
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
    
    async def _generate_overview(self, program_info: Dict[str, Any]) -> str:
        """Generate program overview section"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return result.outputs[0].text.strip()
    
    async def _generate_architecture(self, program_info: Dict[str, Any]) -> str:
        """Generate architecture section"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=700)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return result.outputs[0].text.strip()
    
    async def _generate_business_logic_doc(self, program_info: Dict[str, Any]) -> str:
        """Generate business logic documentation"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=800)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return result.outputs[0].text.strip()
    
    async def _generate_data_flow_doc(self, program_info: Dict[str, Any]) -> str:
        """Generate data flow documentation"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=700)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return result.outputs[0].text.strip()
    
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
    
    async def _generate_maintenance_doc(self, program_info: Dict[str, Any]) -> str:
        """Generate maintenance documentation"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return result.outputs[0].text.strip()
    
    async def _generate_troubleshooting_doc(self, program_info: Dict[str, Any]) -> str:
        """Generate troubleshooting documentation"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=600)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
        
        return result.outputs[0].text.strip()
    
    # System documentation methods
    async def _generate_executive_summary(self, component_info: Dict[str, Any], system_name: str) -> str:
        """Generate executive summary for system documentation"""
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
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=500)
        request_id = str(uuid.uuid4())
        result = await self.llm_engine.generate(prompt, sampling_params, request_id = str(uuid.uuid4()))
        
        return result.outputs[0].text.strip()
    
    async def _generate_system_architecture(self, component_info: Dict[str, Any]) -> str:
        """Generate system architecture documentation"""
        component_types = {}
        for name, info in component_info.items():
            if info:
                for chunk_type in info['chunk_types']:
                    if chunk_type not in component_types:
                        component_types[chunk_type] = []
                    component_types[chunk_type].append(name)
        
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
    
    async def _generate_data_architecture(self, component_info: Dict[str, Any]) -> str:
        """Generate data architecture documentation"""
        all_fields = set()
        all_files = set()
        
        for info in component_info.values():
            if info:
                all_fields.update(field[0] for field in info['field_usage'])
                all_files.update(field[3] for field in info['field_usage'] if field[3])
        
        data_arch = "## Data Architecture\n\n"
        data_arch += f"**Data Elements:** {len(all_fields)} unique fields\n"
        data_arch += f"**Data Sources:** {len(all_files)} files/tables\n\n"
        
        if all_files:
            data_arch += "**Key Data Sources:**\n"
            for file_name in sorted(all_files)[:10]:
                data_arch += f"- {file_name}\n"
        
        return data_arch
    
    async def _generate_integration_points(self, component_info: Dict[str, Any]) -> str:
        """Generate integration points documentation"""
        return "## Integration Points\n\nIntegration analysis based on component dependencies and data flow patterns.\n"
    
    async def _generate_operational_procedures(self, component_info: Dict[str, Any]) -> str:
        """Generate operational procedures documentation"""
        return "## Operational Procedures\n\nStandard operational procedures for system maintenance and monitoring.\n"
    
    async def _generate_security_doc(self, component_info: Dict[str, Any]) -> str:
        """Generate security documentation"""
        return "## Security Considerations\n\nSecurity protocols and access control measures for the system.\n"
    
    async def _generate_performance_doc(self, component_info: Dict[str, Any]) -> str:
        """Generate performance documentation"""
        return "## Performance Characteristics\n\nSystem performance metrics and optimization considerations.\n"
    
    async def _generate_maintenance_support_doc(self, component_info: Dict[str, Any]) -> str:
        """Generate maintenance and support documentation"""
        return "## Maintenance and Support\n\nOngoing maintenance procedures and support contact information.\n"
    
    # Field lineage documentation methods
    async def _generate_field_overview(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field overview for lineage documentation"""
        field_name = lineage_info['field_name']
        programs_count = len(lineage_info['programs_using'])
        operations_count = len(lineage_info['operations'])
        
        overview = f"""
## Field Overview

**Field Name:** {field_name}
**Programs Using:** {programs_count}
**Operation Types:** {operations_count}
**Operations:** {', '.join(lineage_info['operations'])}

This field is actively used across multiple programs in the system, indicating its importance in business operations.
        """.strip()
        
        return overview
    
    async def _generate_field_lifecycle_doc(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field lifecycle documentation"""
        records = lineage_info['lineage_records']
        
        lifecycle_doc = "## Field Lifecycle\n\n"
        lifecycle_doc += "**Usage Timeline:**\n"
        
        for record in records[:10]:  # Show recent usage
            program, paragraph, operation, source_file, last_used = record
            lifecycle_doc += f"- {operation} in {program}.{paragraph} ({last_used})\n"
        
        return lifecycle_doc
    
    async def _generate_field_usage_patterns(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field usage patterns documentation"""
        operations = {}
        for record in lineage_info['lineage_records']:
            op = record[2]
            if op not in operations:
                operations[op] = 0
            operations[op] += 1
        
        patterns_doc = "## Usage Patterns\n\n"
        for operation, count in operations.items():
            patterns_doc += f"- {operation}: {count} occurrences\n"
        
        return patterns_doc
    
    async def _generate_field_dependencies(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field dependencies documentation"""
        programs = lineage_info['programs_using']
        
        deps_doc = "## Dependencies\n\n"
        deps_doc += "**Programs Dependent on This Field:**\n"
        
        for program in programs:
            deps_doc += f"- {program}\n"
        
        return deps_doc
    
    async def _generate_field_quality_metrics(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field quality metrics documentation"""
        return "## Quality Metrics\n\nField quality assessment and data validation results.\n"
    
    async def _generate_field_recommendations(self, lineage_info: Dict[str, Any]) -> str:
        """Generate field recommendations documentation"""
        return "## Recommendations\n\nRecommendations for field usage optimization and data quality improvement.\n"
    
    async def generate_api_documentation(self, programs: List[str]) -> Dict[str, Any]:
        """Generate API-style documentation for program interfaces"""
        try:
            api_docs = {}
            
            for program_name in programs:
                program_info = await self._gather_program_info(program_name)
                if program_info:
                    api_docs[program_name] = await self._generate_program_api_doc(program_info)
            
            return {
                "status": "success",
                "api_documentation": api_docs,
                "programs_documented": len(api_docs)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _generate_program_api_doc(self, program_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API-style documentation for a single program"""
        # Extract inputs and outputs
        inputs = []
        outputs = []
        
        for field_record in program_info['field_usage']:
            field_name, operation, paragraph, source_file = field_record
            
            if operation.upper() in ['READ', 'INPUT']:
                inputs.append({"field": field_name, "source": source_file})
            elif operation.upper() in ['WRITE', 'OUTPUT', 'UPDATE']:
                outputs.append({"field": field_name, "destination": source_file})
        
        return {
            "program_name": program_info['program_name'],
            "description": f"Mainframe program {program_info['program_name']}",
            "inputs": inputs[:10],  # Limit for readability
            "outputs": outputs[:10],
            "complexity": program_info['total_chunks'],
            "components": program_info['chunk_types']
        }