# agents/code_parser_agent.py - FIXED VERSION
"""
Agent 1: Batch Code Parser & Chunker
Handles COBOL, JCL, CICS, and Copybook parsing with intelligent chunking
"""

import re
import asyncio
import sqlite3
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
from datetime import datetime
import logging

import torch
from vllm import AsyncLLMEngine, SamplingParams

@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""
    program_name: str
    chunk_id: str
    chunk_type: str  # paragraph, perform, job_step, proc, sql_block
    content: str
    metadata: Dict[str, Any]
    line_start: int
    line_end: int

class CodeParserAgent:
    def __init__(self, llm_engine: AsyncLLMEngine = None, db_path: str = None, 
                 gpu_id: int = None, coordinator=None):
        self.llm_engine = llm_engine
        self.db_path = db_path or "opulence_data.db"
        self.gpu_id = gpu_id
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
        
        # Add lock for thread safety
        self._engine_lock = asyncio.Lock()
        self._engine_created = False
        self._using_coordinator_llm = False
        
        # INITIALIZE COBOL PATTERNS - THIS WAS MISSING!
        self.cobol_patterns = {
            'program_id': re.compile(r'PROGRAM-ID\s+(\S+)', re.IGNORECASE),
            'paragraph': re.compile(r'^([A-Z0-9][A-Z0-9-]*)\s*\.\s*$', re.MULTILINE),
            'perform': re.compile(r'PERFORM\s+([A-Z0-9][A-Z0-9-]*)', re.IGNORECASE),
            'sql_block': re.compile(r'EXEC\s+SQL(.*?)END-EXEC', re.DOTALL | re.IGNORECASE),
            'file_control': re.compile(r'SELECT\s+(\S+)\s+ASSIGN\s+TO\s+(\S+)', re.IGNORECASE),
            'working_storage': re.compile(r'WORKING-STORAGE\s+SECTION', re.IGNORECASE),
            'procedure_division': re.compile(r'PROCEDURE\s+DIVISION', re.IGNORECASE)
        }
        
        # INITIALIZE JCL PATTERNS - THIS WAS MISSING!
        self.jcl_patterns = {
            'job_card': re.compile(r'^//(\S+)\s+JOB\s+', re.MULTILINE),
            'job_step': re.compile(r'^//(\S+)\s+EXEC\s+', re.MULTILINE),
            'dd_statement': re.compile(r'^//(\S+)\s+DD\s+', re.MULTILINE),
            'proc_call': re.compile(r'EXEC\s+(\S+)', re.IGNORECASE),
            'dataset': re.compile(r'DSN=([^,\s]+)', re.IGNORECASE)
        }

    async def _ensure_llm_engine(self):
        """Ensure LLM engine is available with thread safety"""
        async with self._engine_lock:  # Prevent race conditions
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
                                self.logger.info(f"CodeParser SHARING coordinator's LLM on GPU {gpu_id}")
                                return
                        except Exception as e:
                            self.logger.warning(f"Error checking GPU {gpu_id} for sharing: {e}")
                            continue
                    
                    # If no engine can be shared, get a new GPU
                    best_gpu = await self.coordinator.get_available_gpu_for_agent("code_parser")
                    if best_gpu is not None:
                        engine = await self.coordinator.get_or_create_llm_engine(best_gpu)
                        self.llm_engine = engine
                        self.gpu_id = best_gpu
                        self._using_coordinator_llm = True
                        self.logger.info(f"CodeParser using coordinator's NEW LLM on GPU {best_gpu}")
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
                                self.logger.info(f"CodeParser SHARING global coordinator's LLM on GPU {gpu_id}")
                                return
                        except Exception as e:
                            continue
                    
                    # If no sharing possible, get new GPU
                    best_gpu = await global_coordinator.get_available_gpu_for_agent("code_parser")
                    if best_gpu is not None:
                        engine = await global_coordinator.get_or_create_llm_engine(best_gpu)
                        self.llm_engine = engine
                        self.gpu_id = best_gpu
                        self._using_coordinator_llm = True
                        self.logger.info(f"CodeParser using global coordinator's NEW LLM on GPU {best_gpu}")
                        return
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get LLM from global coordinator: {e}")
            
            # Last resort: create own engine (should rarely happen)
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
            
            self.logger.warning(f"CodeParser creating FALLBACK LLM on GPU {best_gpu} with {best_memory:.1f}GB")
            
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
                
                self.logger.info(f"✅ CodeParser fallback LLM created on GPU {best_gpu}")
                
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
            result['agent_type'] = 'code_parser'
            result['using_coordinator_llm'] = self._using_coordinator_llm
        return result
    
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single code file"""
        try:
            await self._ensure_llm_engine()
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_type = self._detect_file_type(content, file_path.suffix)
            
            if file_type == 'cobol':
                chunks = await self._parse_cobol(content, file_path.name)
            elif file_type == 'jcl':
                chunks = await self._parse_jcl(content, file_path.name)
            elif file_type == 'copybook':
                chunks = await self._parse_copybook(content, file_path.name)
            else:
                chunks = await self._parse_generic(content, file_path.name)
            
            # Store chunks in database
            await self._store_chunks(chunks)
            
            # Generate metadata
            metadata = await self._generate_metadata(chunks, file_type)
            
            result = {
                "status": "success",
                "file_name": file_path.name,
                "file_type": file_type,
                "chunks_created": len(chunks),
                "metadata": metadata
            }
            
            return self._add_processing_info(result)
            
        except Exception as e:
            return self._add_processing_info({
                "status": "error",
                "file_name": file_path.name,
                "error": str(e)
            })
    
    def _detect_file_type(self, content: str, suffix: str) -> str:
        """Detect the type of mainframe file"""
        content_upper = content.upper()
        
        if 'IDENTIFICATION DIVISION' in content_upper or 'PROGRAM-ID' in content_upper:
            return 'cobol'
        elif content.strip().startswith('//') and 'JOB' in content_upper:
            return 'jcl'
        elif 'COPY' in content_upper and len(content.split('\n')) < 100:
            return 'copybook'
        elif suffix.lower() in ['.cbl', '.cob']:
            return 'cobol'
        elif suffix.lower() == '.jcl':
            return 'jcl'
        elif suffix.lower() in ['.cpy', '.copy']:
            return 'copybook'
        else:
            return 'unknown'

    # FIX: Correct _parse_cobol method
    async def _parse_cobol(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse COBOL program into logical chunks - FIXED VERSION"""
        await self._ensure_llm_engine()
        
        chunks = []
        lines = content.split('\n')
        
        # Extract program name
        program_match = self.cobol_patterns['program_id'].search(content)
        program_name = program_match.group(1) if program_match else filename
        
        # Find major sections
        sections = self._find_cobol_sections(content)
        
        # Parse each section
        for section_name, section_content, start_line, end_line in sections:
            if section_name == 'PROCEDURE DIVISION':
                # Further chunk by paragraphs
                paragraph_chunks = await self._parse_cobol_paragraphs(
                    section_content, program_name, start_line
                )
                chunks.extend(paragraph_chunks)
            else:
                # Create chunk for entire section
                chunk = CodeChunk(
                    program_name=program_name,
                    chunk_id=f"{program_name}_{section_name.replace(' ', '_')}",
                    chunk_type="section",
                    content=section_content,
                    metadata={
                        "section": section_name,
                        "field_names": self._extract_field_names(section_content),
                        "files": self._extract_file_references(section_content),
                        "operations": self._extract_operations(section_content)
                    },
                    line_start=start_line,
                    line_end=end_line
                )
                chunks.append(chunk)
        
        # Extract SQL blocks separately
        sql_chunks = await self._extract_sql_blocks(content, program_name)
        chunks.extend(sql_chunks)
        
        return chunks
    
    def _find_cobol_sections(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Find major COBOL sections"""
        sections = []
        lines = content.split('\n')
        
        section_markers = [
            'IDENTIFICATION DIVISION',
            'ENVIRONMENT DIVISION',
            'DATA DIVISION',
            'WORKING-STORAGE SECTION',
            'FILE SECTION',
            'PROCEDURE DIVISION'
        ]
        
        current_section = None
        section_start = 0
        section_content = []
        
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            
            # Check if this line starts a new section
            for marker in section_markers:
                if line_upper.startswith(marker):
                    # Save previous section
                    if current_section:
                        sections.append((
                            current_section,
                            '\n'.join(section_content),
                            section_start,
                            i - 1
                        ))
                    
                    # Start new section
                    current_section = marker
                    section_start = i
                    section_content = [line]
                    break
            else:
                if current_section:
                    section_content.append(line)
        
        # Add final section
        if current_section:
            sections.append((
                current_section,
                '\n'.join(section_content),
                section_start,
                len(lines) - 1
            ))
        
        return sections

    async def _extract_sql_blocks(self, content: str, program_name: str) -> List[CodeChunk]:
        """Extract SQL blocks from COBOL code"""
        await self._ensure_llm_engine()
        
        chunks = []
        sql_matches = self.cobol_patterns['sql_block'].finditer(content)
        
        for i, match in enumerate(sql_matches):
            sql_content = match.group(0)
            sql_inner = match.group(1).strip()
            
            # Analyze SQL with LLM
            metadata = await self._analyze_sql_with_llm(sql_inner)
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_SQL_{i+1}",
                chunk_type="sql_block",
                content=sql_content,
                metadata=metadata,
                line_start=content[:match.start()].count('\n'),
                line_end=content[:match.end()].count('\n')
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _analyze_sql_with_llm(self, sql_content: str) -> Dict[str, Any]:
        """Analyze SQL block with LLM"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this embedded SQL block:
        
        {sql_content}
        
        Extract:
        1. SQL operation type (SELECT, INSERT, UPDATE, DELETE)
        2. Tables accessed
        3. Key fields/columns
        4. Join conditions if any
        5. Purpose of the operation
        
        Return as JSON:
        {{
            "operation_type": "SELECT",
            "tables": ["table1", "table2"],
            "fields": ["field1", "field2"],
            "joins": ["table1.id = table2.id"],
            "purpose": "description"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=300)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        return {
            "operation_type": self._extract_sql_type(sql_content),
            "tables": self._extract_table_names(sql_content),
            "fields": [],
            "joins": [],
            "purpose": "SQL operation"
        }
    
    async def _parse_jcl(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse JCL into job steps"""
        await self._ensure_llm_engine()
        
        chunks = []
        lines = content.split('\n')
        
        # Extract job name
        job_match = self.jcl_patterns['job_card'].search(content)
        job_name = job_match.group(1) if job_match else filename
        
        current_step = None
        step_start = 0
        step_content = []
        
        for i, line in enumerate(lines):
            line = line.rstrip()
            
            # Check if this is a job step
            step_match = self.jcl_patterns['job_step'].match(line)
            if step_match:
                # Save previous step
                if current_step:
                    step_text = '\n'.join(step_content)
                    metadata = await self._analyze_jcl_step_with_llm(step_text)
                    
                    chunk = CodeChunk(
                        program_name=job_name,
                        chunk_id=f"{job_name}_{current_step}",
                        chunk_type="job_step",
                        content=step_text,
                        metadata=metadata,
                        line_start=step_start,
                        line_end=i - 1
                    )
                    chunks.append(chunk)
                
                # Start new step
                current_step = step_match.group(1)
                step_start = i
                step_content = [line]
            else:
                if current_step:
                    step_content.append(line)
                elif line.strip() and not line.startswith('//*'):
                    # Job card or other non-step content
                    if not chunks:  # First chunk for job header
                        chunk = CodeChunk(
                            program_name=job_name,
                            chunk_id=f"{job_name}_HEADER",
                            chunk_type="job_header",
                            content=line,
                            metadata={"job_name": job_name},
                            line_start=i,
                            line_end=i
                        )
                        chunks.append(chunk)
        
        # Add final step
        if current_step:
            step_text = '\n'.join(step_content)
            metadata = await self._analyze_jcl_step_with_llm(step_text)
            
            chunk = CodeChunk(
                program_name=job_name,
                chunk_id=f"{job_name}_{current_step}",
                chunk_type="job_step",
                content=step_text,
                metadata=metadata,
                line_start=step_start,
                line_end=len(lines) - 1
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _analyze_jcl_step_with_llm(self, content: str) -> Dict[str, Any]:
        """Analyze JCL step with LLM"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this JCL job step:
        
        {content}
        
        Extract:
        1. Program being executed
        2. Input datasets (DD statements)
        3. Output datasets
        4. Parameters passed
        5. Step purpose
        
        Return as JSON:
        {{
            "program": "program_name",
            "input_datasets": ["dsn1", "dsn2"],
            "output_datasets": ["dsn3", "dsn4"],
            "parameters": ["parm1=value1"],
            "purpose": "description"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=400)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        return {
            "program": self._extract_exec_program(content),
            "input_datasets": self._extract_input_datasets(content),
            "output_datasets": self._extract_output_datasets(content),
            "parameters": self._extract_parameters(content),
            "purpose": "Job step execution"
        }
    
    async def _parse_copybook(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse copybook into field definitions"""
        await self._ensure_llm_engine()
        
        chunks = []
        
        # Analyze copybook structure with LLM
        metadata = await self._analyze_copybook_with_llm(content)
        
        chunk = CodeChunk(
            program_name=filename,
            chunk_id=f"{filename}_COPYBOOK",
            chunk_type="copybook",
            content=content,
            metadata=metadata,
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )
        chunks.append(chunk)
        
        return chunks
    
    async def _analyze_copybook_with_llm(self, content: str) -> Dict[str, Any]:
        """Analyze copybook with LLM"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this COBOL copybook:
        
        {content}
        
        Extract:
        1. All field definitions with levels and types
        2. Record structure
        3. Field usage (COMP, COMP-3, etc.)
        4. Key fields
        5. Purpose of the copybook
        
        Return as JSON:
        {{
            "fields": [
                {{"name": "field1", "level": "01", "type": "PIC X(10)", "usage": "DISPLAY"}},
                {{"name": "field2", "level": "05", "type": "PIC 9(8)", "usage": "COMP"}}
            ],
            "record_structure": "description",
            "key_fields": ["field1", "field2"],
            "purpose": "description"
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=800)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        return {
            "fields": self._extract_copybook_fields(content),
            "record_structure": "COBOL copybook",
            "key_fields": [],
            "purpose": "Data structure definition"
        }
    
    async def _parse_generic(self, content: str, filename: str) -> List[CodeChunk]:
        """Parse unknown file type generically"""
        chunk = CodeChunk(
            program_name=filename,
            chunk_id=f"{filename}_GENERIC",
            chunk_type="generic",
            content=content,
            metadata={"file_type": "unknown"},
            line_start=0,
            line_end=len(content.split('\n')) - 1
        )
        return [chunk]
    
    async def _parse_cobol_paragraphs(self, content: str, program_name: str, start_offset: int) -> List[CodeChunk]:
        """Parse COBOL procedure division into paragraphs"""
        await self._ensure_llm_engine()
        
        chunks = []
        lines = content.split('\n')
        
        current_paragraph = None
        paragraph_start = 0
        paragraph_content = []
        
        for i, line in enumerate(lines):
            # Check if this is a paragraph header
            if self.cobol_patterns['paragraph'].match(line.strip()):
                # Save previous paragraph
                if current_paragraph:
                    chunk_content = '\n'.join(paragraph_content)
                    metadata = await self._analyze_paragraph_with_llm(chunk_content)
                    
                    chunk = CodeChunk(
                        program_name=program_name,
                        chunk_id=f"{program_name}_{current_paragraph}",
                        chunk_type="paragraph",
                        content=chunk_content,
                        metadata=metadata,
                        line_start=start_offset + paragraph_start,
                        line_end=start_offset + i - 1
                    )
                    chunks.append(chunk)
                
                # Start new paragraph
                current_paragraph = line.strip().rstrip('.')
                paragraph_start = i
                paragraph_content = [line]
            else:
                if current_paragraph:
                    paragraph_content.append(line)
        
        # Add final paragraph
        if current_paragraph:
            chunk_content = '\n'.join(paragraph_content)
            metadata = await self._analyze_paragraph_with_llm(chunk_content)
            
            chunk = CodeChunk(
                program_name=program_name,
                chunk_id=f"{program_name}_{current_paragraph}",
                chunk_type="paragraph",
                content=chunk_content,
                metadata=metadata,
                line_start=start_offset + paragraph_start,
                line_end=start_offset + len(lines) - 1
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _analyze_paragraph_with_llm(self, content: str) -> Dict[str, Any]:
        """Use LLM to analyze paragraph content"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this COBOL paragraph and extract key information:
        
        {content}
        
        Extract:
        1. Field names referenced
        2. File operations (READ, WRITE, REWRITE, DELETE)
        3. Database operations (SQL statements)
        4. Called paragraphs (PERFORM statements)
        5. Main purpose/operation
        
        Return as JSON format:
        {{
            "field_names": ["field1", "field2"],
            "file_operations": ["READ FILE1", "WRITE FILE2"],
            "sql_operations": ["SELECT", "UPDATE"],
            "called_paragraphs": ["PARA1", "PARA2"],
            "main_purpose": "description"
        }}
        """
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=500,
            stop=["```"]
        )
        
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            # Extract JSON from response
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to regex extraction
        return {
            "field_names": self._extract_field_names(content),
            "file_operations": self._extract_file_operations(content),
            "sql_operations": self._extract_sql_operations(content),
            "called_paragraphs": self._extract_perform_statements(content),
            "main_purpose": "Code analysis"
        }

    async def analyze_jcl(self, jcl_name: str) -> Dict[str, Any]:
        """Analyze JCL job flow"""
        await self._ensure_llm_engine()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_type, content, metadata FROM program_chunks 
            WHERE program_name = ? AND chunk_type IN ('job_step', 'job_header')
            ORDER BY chunk_id
        """, (jcl_name,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            return {"error": f"JCL {jcl_name} not found"}
        
        # Analyze job flow with LLM
        job_content = '\n'.join([chunk[1] for chunk in chunks])
        
        prompt = f"""
        Analyze this complete JCL job flow:
        
        {job_content}
        
        Provide:
        1. Job execution sequence
        2. Data flow between steps
        3. Critical dependencies
        4. Error handling steps
        5. Overall job purpose
        
        Format as detailed analysis.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return {
            "jcl_name": jcl_name,
            "total_steps": len([c for c in chunks if c[0] == 'job_step']),
            "analysis": result.outputs[0].text,
            "step_details": [json.loads(chunk[2]) for chunk in chunks if chunk[2]]
        }

    # Helper methods for extraction
    def _extract_field_names(self, content: str) -> List[str]:
        """Extract field names from COBOL code"""
        fields = []
        # Pattern for COBOL field definitions
        field_pattern = re.compile(r'\b\d+\s+([A-Z][A-Z0-9-]*)\s+PIC', re.IGNORECASE)
        fields.extend([match.group(1) for match in field_pattern.finditer(content)])
        
        # Pattern for field references
        ref_pattern = re.compile(r'\bMOVE\s+\S+\s+TO\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        fields.extend([match.group(1) for match in ref_pattern.finditer(content)])
        
        return list(set(fields))
    
    def _extract_file_references(self, content: str) -> List[str]:
        """Extract file references from code"""
        files = []
        file_pattern = re.compile(r'SELECT\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        files.extend([match.group(1) for match in file_pattern.finditer(content)])
        return list(set(files))
    
    def _extract_operations(self, content: str) -> List[str]:
        """Extract main operations from code"""
        operations = []
        op_patterns = [
            r'\b(READ|WRITE|REWRITE|DELETE|OPEN|CLOSE)\b',
            r'\b(PERFORM|CALL|INVOKE)\b',
            r'\b(ADD|SUBTRACT|MULTIPLY|DIVIDE|COMPUTE)\b'
        ]
        
        for pattern in op_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            operations.extend([match.group(1).upper() for match in matches])
        
        return list(set(operations))
    
    def _extract_file_operations(self, content: str) -> List[str]:
        """Extract file operations"""
        ops = []
        pattern = re.compile(r'\b(READ|WRITE|REWRITE|DELETE|OPEN|CLOSE)\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        for match in pattern.finditer(content):
            ops.append(f"{match.group(1).upper()} {match.group(2)}")
        return ops
    
    def _extract_sql_operations(self, content: str) -> List[str]:
        """Extract SQL operations"""
        ops = []
        pattern = re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', re.IGNORECASE)
        ops.extend([match.group(1).upper() for match in pattern.finditer(content)])
        return list(set(ops))
    
    def _extract_perform_statements(self, content: str) -> List[str]:
        """Extract PERFORM statements"""
        performs = []
        pattern = re.compile(r'PERFORM\s+([A-Z][A-Z0-9-]*)', re.IGNORECASE)
        performs.extend([match.group(1) for match in pattern.finditer(content)])
        return list(set(performs))
    
    def _extract_sql_type(self, sql_content: str) -> str:
        """Extract SQL operation type"""
        sql_upper = sql_content.upper()
        if 'SELECT' in sql_upper:
            return 'SELECT'
        elif 'INSERT' in sql_upper:
            return 'INSERT'
        elif 'UPDATE' in sql_upper:
            return 'UPDATE'
        elif 'DELETE' in sql_upper:
            return 'DELETE'
        else:
            return 'UNKNOWN'
    
    def _extract_table_names(self, sql_content: str) -> List[str]:
        """Extract table names from SQL"""
        tables = []
        # Simple pattern for table names after FROM/UPDATE/INTO
        pattern = re.compile(r'\b(?:FROM|UPDATE|INTO)\s+([A-Z][A-Z0-9_]*)', re.IGNORECASE)
        tables.extend([match.group(1) for match in pattern.finditer(sql_content)])
        return list(set(tables))
    
    def _extract_exec_program(self, jcl_content: str) -> str:
        """Extract program name from JCL EXEC statement"""
        pattern = re.compile(r'EXEC\s+(?:PGM=)?([A-Z][A-Z0-9]*)', re.IGNORECASE)
        match = pattern.search(jcl_content)
        return match.group(1) if match else "UNKNOWN"
    
    def _extract_input_datasets(self, jcl_content: str) -> List[str]:
        """Extract input datasets from JCL"""
        datasets = []
        # Look for DD statements with DSN parameter
        lines = jcl_content.split('\n')
        for line in lines:
            if '//DD' in line.upper() and 'DSN=' in line.upper():
                dsn_match = re.search(r'DSN=([^,\s]+)', line, re.IGNORECASE)
                if dsn_match:
                    datasets.append(dsn_match.group(1))
        return datasets
    
    def _extract_output_datasets(self, jcl_content: str) -> List[str]:
        """Extract output datasets from JCL"""
        datasets = []
        # Look for DD statements with DISP=(NEW, or DISP=(,CATLG
        lines = jcl_content.split('\n')
        for line in lines:
            if ('DISP=(NEW' in line.upper() or 'DISP=(,CATLG' in line.upper()) and 'DSN=' in line.upper():
                dsn_match = re.search(r'DSN=([^,\s]+)', line, re.IGNORECASE)
                if dsn_match:
                    datasets.append(dsn_match.group(1))
        return datasets
    
    def _extract_parameters(self, jcl_content: str) -> List[str]:
        """Extract parameters from JCL"""
        params = []
        pattern = re.compile(r'PARM=([^,\s]+)', re.IGNORECASE)
        params.extend([match.group(1) for match in pattern.finditer(jcl_content)])
        return params
    
    def _extract_copybook_fields(self, content: str) -> List[Dict[str, str]]:
        """Extract field definitions from copybook"""
        fields = []
        pattern = re.compile(r'^(\s*)(\d+)\s+([A-Z][A-Z0-9-]*)\s+(PIC\s+[^\s.]+)', re.MULTILINE | re.IGNORECASE)
        
        for match in pattern.finditer(content):
            fields.append({
                "level": match.group(2),
                "name": match.group(3),
                "type": match.group(4),
                "usage": "DISPLAY"  # Default
            })
        
        return fields
    
    async def _store_chunks(self, chunks: List[CodeChunk]):
        """Store parsed chunks in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk in chunks:
            cursor.execute("""
                INSERT OR REPLACE INTO program_chunks 
                (program_name, chunk_id, chunk_type, content, metadata, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk.program_name,
                chunk.chunk_id,
                chunk.chunk_type,
                chunk.content,
                json.dumps(chunk.metadata),
                hashlib.md5(chunk.content.encode()).hexdigest()
            ))
        
        conn.commit()
        conn.close()
    
    async def _generate_metadata(self, chunks: List[CodeChunk], file_type: str) -> Dict[str, Any]:
        """Generate overall metadata for the file"""
        all_fields = set()
        all_files = set()
        all_operations = set()
        
        for chunk in chunks:
            if 'field_names' in chunk.metadata:
                all_fields.update(chunk.metadata['field_names'])
            if 'files' in chunk.metadata:
                all_files.update(chunk.metadata['files'])
            if 'operations' in chunk.metadata:
                all_operations.update(chunk.metadata['operations'])
        
        return {
            "total_chunks": len(chunks),
            "file_type": file_type,
            "all_fields": list(all_fields),
            "all_files": list(all_files),
            "all_operations": list(all_operations),
            "processed_timestamp": datetime.now().isoformat()
        }

    async def analyze_job_flow(self, jcl_name: str) -> Dict[str, Any]:
        """Analyze detailed job flow"""
        return await self.analyze_jcl(jcl_name)
    
    async def analyze_cobol(self, cobol_name: str) -> Dict[str, Any]:
        """Analyze COBOL program structure and operations"""
        await self._ensure_llm_engine()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_type, content, metadata FROM program_chunks 
            WHERE program_name = ? AND chunk_type IN ('section', 'paragraph', 'sql_block')
            ORDER BY chunk_id
        """, (cobol_name,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            return {"error": f"COBOL program {cobol_name} not found"}
        
        # Analyze program structure with LLM
        program_content = '\n'.join([chunk[1] for chunk in chunks])
        
        prompt = f"""
        Analyze this complete COBOL program:
        
        {program_content}
        
        Provide:
        1. Program structure and organization
        2. Main business logic flow
        3. Data file operations
        4. SQL operations if any
        5. Key fields and data structures
        6. Overall program purpose
        
        Format as detailed analysis.
        """
        
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1000)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        return {
            "cobol_name": cobol_name,
            "total_chunks": len(chunks),
            "sections": len([c for c in chunks if c[0] == 'section']),
            "paragraphs": len([c for c in chunks if c[0] == 'paragraph']),
            "sql_blocks": len([c for c in chunks if c[0] == 'sql_block']),
            "analysis": result.outputs[0].text,
            "chunk_details": [json.loads(chunk[2]) for chunk in chunks if chunk[2]]
        }

    async def _analyze_code_with_llm(self, content: str) -> Dict[str, Any]:
        """Use LLM to analyze generic code content"""
        await self._ensure_llm_engine()
        
        prompt = f"""
        Analyze this code snippet:
        
        {content}
        
        Extract:
        1. Key functions or methods
        2. Main data structures used
        3. Overall purpose of the code
        4. Any potential issues or improvements
        
        Return as JSON:
        {{
            "functions": ["func1", "func2"],
            "data_structures": ["list", "dict"],
            "purpose": "description",
            "issues": ["issue1", "issue2"]
        }}
        """
        
        sampling_params = SamplingParams(temperature=0.1, max_tokens=500)
        result = await self.llm_engine.generate(prompt, sampling_params)
        
        try:
            response_text = result.outputs[0].text.strip()
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                return json.loads(response_text[json_start:json_end])
        except:
            pass
        
        return {
            "functions": [],
            "data_structures": [],
            "purpose": "Generic code analysis",
            "issues": []
        }